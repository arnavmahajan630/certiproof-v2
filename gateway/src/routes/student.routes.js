const express = require("express");
const multer = require("multer");
const fs = require("fs");
const pdfParse = require("pdf-parse");

const db = require("../db/client");
const evaluationPipeline = require("../services/evaluationPipeline");
const verifyService = require("../services/verifyService");
const scorecardService = require("../services/scorecardService");
const { requireRole } = require("../middleware/roleAuth");

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

// POST /student/submissions — typed answer submission. Synchronous: the response only
// returns once the full pipeline (Zone1 -> witness commit -> Zone2/prove -> persist ->
// scorecard) has completed.
router.post("/submissions", requireRole("student"), async (req, res) => {
  const { batch_id, student_id, answer_text } = req.body;
  if (!batch_id || !student_id || !answer_text) {
    return res.status(422).json({ error: "missing_fields", detail: "batch_id, student_id, answer_text are required" });
  }
  try {
    const result = await evaluationPipeline.submitAndEvaluate({
      batchId: batch_id,
      studentId: student_id,
      answerText: answer_text,
      inputSource: "student_typed",
    });
    res.status(201).json(result);
  } catch (e) {
    if (e.code === "BATCH_NOT_FOUND") return res.status(404).json({ error: "batch_not_found" });
    res.status(502).json({ error: "evaluation_failure", detail: e.message });
  }
});

// GET /student/submissions/:submission_id — status + score once evaluated.
router.get("/submissions/:submission_id", (req, res) => {
  const submission = db.prepare("SELECT * FROM submissions WHERE submission_id = ?").get(req.params.submission_id);
  if (!submission) return res.status(404).json({ error: "submission_not_found" });
  const evaluation = db.prepare("SELECT * FROM evaluations WHERE submission_id = ?").get(req.params.submission_id);
  res.json({ submission, evaluation: evaluation || null });
});

// GET /student/evaluations?student_id=... — student-facing lookup. Answer text and
// cryptographic internals are omitted; the student only needs score + timestamps.
router.get("/evaluations", requireRole("student"), (req, res) => {
  const studentId = String(req.query.student_id || "").trim();
  if (!studentId) {
    return res.status(422).json({ error: "missing_student_id", detail: "student_id is required" });
  }

  const evaluations = db
    .prepare(
      `SELECT
         s.submission_id,
         s.batch_id,
         s.student_id,
         s.input_source,
         s.submitted_at,
         e.evaluation_id,
         e.final_score,
         e.proof_status,
         e.evaluated_at,
         e.scorecard_generated_at,
         e.per_criterion_scores_json,
         e.explanations_json
       FROM submissions s
       LEFT JOIN evaluations e ON e.submission_id = s.submission_id
       WHERE s.student_id = ?
       ORDER BY s.submitted_at DESC`
    )
    .all(studentId);

  res.json({ student_id: studentId, evaluations });
});

// GET /student/evaluations/:evaluation_id/scorecard — download the PDF.
router.get("/evaluations/:evaluation_id/scorecard", (req, res) => {
  const evaluation = db.prepare("SELECT * FROM evaluations WHERE evaluation_id = ?").get(req.params.evaluation_id);
  if (!evaluation || !evaluation.scorecard_path) {
    return res.status(404).json({ error: "scorecard_not_found" });
  }
  res.download(evaluation.scorecard_path, `${req.params.evaluation_id}_scorecard.pdf`);
});

// POST /student/verify-scorecard — multipart PDF upload, integrity + full verify chain.
router.post("/verify-scorecard", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(422).json({ error: "missing_file" });

  let text;
  try {
    const parsed = await pdfParse(req.file.buffer);
    text = parsed.text;
  } catch (e) {
    return res.status(422).json({ error: "unreadable_pdf", detail: e.message });
  }

  const parsedLine = scorecardService.parseVerifyLine(text);
  if (!parsedLine) {
    return res.status(422).json({
      overall_valid: false,
      checks: [{ name: "scorecard_parse", passed: false, detail: "could not find embedded VERIFY-ID in this PDF" }],
      diff: null,
    });
  }

  const result = await verifyService.verifyScorecard(parsedLine.evaluationId, parsedLine.scorecardToken, req.file.buffer);
  res.json(result);
});

module.exports = router;
