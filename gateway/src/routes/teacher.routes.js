const express = require("express");
const multer = require("multer");

const db = require("../db/client");
const ml = require("../services/mlWorkerClient");
const auditChain = require("../services/auditChain");
const evaluationPipeline = require("../services/evaluationPipeline");
const { sha256Hex, sha256Buffer, newId } = require("../services/hashing");
const { requireRole } = require("../middleware/roleAuth");

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

router.use(requireRole("teacher"));

// POST /teacher/batches — create + certify a batch.
// rubric.criteria[i]: { criterion_id, criterion_text, max_marks, weight }.
// criterion_id/max_marks are RCAJ-X requirements (the old Zone1/Zone2 pipeline
// only needed criterion_text/weight) -- required here, not defaulted, so a
// rubric author can't silently certify one without them.
router.post("/batches", async (req, res) => {
  const { rubric, certifiedBy } = req.body;
  if (!rubric || !Array.isArray(rubric.criteria) || rubric.criteria.length === 0) {
    return res.status(422).json({ error: "bad_rubric", detail: "rubric.criteria must be a non-empty array" });
  }
  if (rubric.criteria.length > 10) {
    return res.status(422).json({ error: "too_many_criteria", detail: "max 10 criteria" });
  }
  const badCriterion = rubric.criteria.find(
    (c) => !c.criterion_id || !c.criterion_text || typeof c.max_marks !== "number" || typeof c.weight !== "number"
  );
  if (badCriterion) {
    return res.status(422).json({
      error: "bad_criterion",
      detail: "each criterion needs criterion_id (string), criterion_text (string), max_marks (number), weight (number)",
    });
  }

  let modelHash;
  try {
    modelHash = await ml.rcajxModelHash();
  } catch (e) {
    return res.status(502).json({ error: "ml_worker_unreachable", detail: e.message });
  }

  const batchId = newId("batch");
  const rubricJson = JSON.stringify(rubric);
  const rubricHash = sha256Hex(rubricJson);
  const certifiedAt = new Date().toISOString();

  db.prepare(
    `INSERT INTO exam_batches (batch_id, rubric_json, rubric_hash, zone1_model_hash, zone2_model_hash, certified_at, certified_by)
     VALUES (?, ?, ?, ?, ?, ?, ?)`
  ).run(batchId, rubricJson, rubricHash, modelHash.zone1_model_hash, modelHash.zone2_model_hash, certifiedAt, certifiedBy || req.actorId);

  auditChain.append("certify", batchId, rubricHash);

  res.status(201).json({
    batch_id: batchId,
    rubric_hash: rubricHash,
    zone1_model_hash: modelHash.zone1_model_hash,
    zone2_model_hash: modelHash.zone2_model_hash,
    certified_at: certifiedAt,
  });
});

// GET /teacher/batches/:batch_id/submissions — list for review.
router.get("/batches/:batch_id/submissions", (req, res) => {
  const rows = db
    .prepare(
      `SELECT s.*, e.evaluation_id, e.final_score, e.proof_status, e.witness_hash,
              e.per_criterion_scores_json, e.explanations_json
       FROM submissions s LEFT JOIN evaluations e ON e.submission_id = s.submission_id
       WHERE s.batch_id = ? ORDER BY s.submitted_at ASC`
    )
    .all(req.params.batch_id);
  res.json({ submissions: rows });
});

// POST /teacher/submissions/:batch_id/upload-answer-sheet — OCR ingestion (plan Part C).
router.post("/submissions/:batch_id/upload-answer-sheet", upload.single("file"), async (req, res) => {
  const { batch_id } = req.params;
  const { student_id } = req.body;
  if (!req.file) return res.status(422).json({ error: "missing_file", detail: "file is required" });
  if (!student_id) return res.status(422).json({ error: "missing_student_id" });

  const answerSheetHash = sha256Buffer(req.file.buffer);

  let ocrResult;
  try {
    ocrResult = await ml.ocrTranscribe(req.file.buffer, req.file.originalname, req.file.mimetype);
  } catch (e) {
    // e.body is the ML-Worker's structured {error, detail} (see mlWorkerClient.js) when
    // the failure came from a clean HTTP error response; unwrap it for a legible message
    // instead of surfacing the raw "ML-Worker error 503: {...}" wrapper. Distinguish
    // "not configured" (no GEMINI_API_KEY set) from "configured but the live call failed"
    // (bad key, quota, network, unreadable image) so the Teacher view can say something
    // more useful than a generic failure.
    const inner = e.body?.detail || e.body;
    const errorType = inner?.error || "ocr_failure";
    const detail = inner?.detail || e.message;
    const status = errorType === "gemini_not_configured" ? 503 : 502;
    return res.status(status).json({ error: errorType, detail });
  }

  auditChain.append("ocr_transcribe", `${batch_id}:${student_id}`, answerSheetHash);

  try {
    const result = await evaluationPipeline.submitAndEvaluate({
      batchId: batch_id,
      studentId: student_id,
      answerText: ocrResult.answer_text,
      inputSource: "teacher_ocr",
      answerSheetHash,
    });
    res.status(201).json({ ...result, ocr_raw_response_hash: ocrResult.ocr_raw_response_hash, transcribed_text: ocrResult.answer_text });
  } catch (e) {
    if (e.code === "BATCH_NOT_FOUND") return res.status(404).json({ error: "batch_not_found" });
    res.status(502).json({ error: "evaluation_failure", detail: e.message });
  }
});

// POST /teacher/evaluations/:evaluation_id/override
router.post("/evaluations/:evaluation_id/override", (req, res) => {
  const { evaluation_id } = req.params;
  const { overriddenScore, reason } = req.body;
  const evaluation = db.prepare("SELECT * FROM evaluations WHERE evaluation_id = ?").get(evaluation_id);
  if (!evaluation) return res.status(404).json({ error: "evaluation_not_found" });
  if (typeof overriddenScore !== "number") {
    return res.status(422).json({ error: "bad_overridden_score" });
  }

  const overrideId = newId("override");
  const overriddenAt = new Date().toISOString();
  db.prepare(
    `INSERT INTO overrides (override_id, evaluation_id, teacher_id, original_score, overridden_score, reason, overridden_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)`
  ).run(overrideId, evaluation_id, req.actorId, evaluation.final_score, overriddenScore, reason || null, overriddenAt);

  auditChain.append("override", overrideId, sha256Hex(String(overriddenScore)));

  res.status(201).json({ override_id: overrideId, evaluation_id, original_score: evaluation.final_score, overridden_score: overriddenScore, overridden_at: overriddenAt });
});

module.exports = router;
