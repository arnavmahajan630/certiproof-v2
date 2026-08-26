const express = require("express");
const fs = require("fs");

const db = require("../db/client");
const ml = require("../services/mlWorkerClient");
const auditChain = require("../services/auditChain");
const evaluationPipeline = require("../services/evaluationPipeline");
const verifyService = require("../services/verifyService");
const { sha256Hex, newId } = require("../services/hashing");

const router = express.Router();

// Auditor "Tests" panel — self-contained tamper vectors run against a freshly created
// batch/submission/evaluation, each applied then auto-restored inside one request so the
// live system is never left in a tampered state. Mirrors scripts/tamper_suite.sh's vectors,
// exposed over HTTP so the Auditor UI can run them and render the real /verify response.

const VECTORS = [
  {
    id: "input_tamper",
    category: "Input tampering",
    label: "Submitted answer edited after the fact",
    description: "The student's answer_text is changed in the DB after submission — the input_hash receipt issued at submit time no longer matches a re-hash of the stored text.",
  },
  {
    id: "witness_substitution",
    category: "Model / grading tampering",
    label: "Displayed scores fabricated after proving (witness-substitution)",
    description: "The flagship vector. The stored per-criterion scores (per_criterion_scores_json, what the Student/Teacher views actually render) are swapped for fabricated ones, as if someone edited the display record after the fact. The EZKL proof itself was generated against the real scores and stays structurally valid — proof_validity independently re-derives the true scores from the proof's own public output and catches the mismatch.",
  },
  {
    id: "stage0_substitution",
    category: "Model / grading tampering",
    label: "Stage 0 inputs (embeddings) fabricated after the fact",
    description: "The stored Stage 0 record (R/A embeddings, negation flags — what was embedded BEFORE proving, and what witness_hash was committed over) is swapped for a fabricated one. witness_hash, re-derived from the current record, no longer matches what was committed pre-proving.",
  },
  {
    id: "score_edit",
    category: "Model / grading tampering",
    label: "Final score edited directly in the DB",
    description: "The recorded final_score is overwritten. The proof's own embedded public output no longer matches the stored score.",
  },
  {
    id: "zone1_drift",
    category: "Model / grading tampering",
    label: "Zone 1 model swapped after certification",
    description: "The batch's certified zone1_model_hash no longer matches what the live ML-Worker reports for its reranker identity.",
  },
  {
    id: "zone2_drift",
    category: "Model / grading tampering",
    label: "Zone 2 model/circuit swapped after certification",
    description: "The batch's certified zone2_model_hash no longer matches the live ML-Worker's ONNX + circuit + verifying key identity.",
  },
  {
    id: "proof_corruption",
    category: "Proof & artifact tampering",
    label: "Proof file corrupted",
    description: "A single byte inside the EZKL proof artifact on disk is flipped. The verifier should reject it outright.",
  },
  {
    id: "scorecard_tamper",
    category: "Proof & artifact tampering",
    label: "Scorecard PDF edited after issuance",
    description: "The issued scorecard PDF's bytes are altered before being re-uploaded to Verify My Scorecard — its SHA-256 no longer matches the scorecard_hash recorded at generation time.",
  },
  {
    id: "rubric_edit",
    category: "Record & chain tampering",
    label: "Certified rubric edited in place",
    description: "A criterion weight in the batch's stored rubric_json is changed after certification — its self-consistency hash (rubric_hash) no longer matches.",
  },
  {
    id: "wrong_batch",
    category: "Record & chain tampering",
    label: "Evaluation re-associated with a different batch",
    description: "The submission is re-pointed at a different, also validly certified batch with different rubric weights. The proof's poseidon_commitment, recomputed against the new batch's weights, no longer matches.",
  },
  {
    id: "audit_chain_tamper",
    category: "Record & chain tampering",
    label: "Historical audit-chain entry edited",
    description: "A row already written to the append-only audit_chain is edited directly. Hash-link verification finds the first broken link.",
  },
];

router.get("/vectors", (req, res) => res.json({ vectors: VECTORS }));

// POST /dev/tamper/setup — certify a fresh throwaway batch + submit a genuine answer.
// Returns ids plus the baseline (untampered) /verify result so the UI can show "before".
router.post("/setup", async (req, res) => {
  try {
    const modelHash = await ml.rcajxModelHash();

    const batchId = newId("batch");
    const rubric = {
      criteria: [
        { criterion_id: "c1", criterion_text: "Mentions sunlight as the energy source", max_marks: 2, weight: 1.0 },
        { criterion_id: "c2", criterion_text: "Mentions water and carbon dioxide as inputs", max_marks: 2, weight: 1.0 },
        { criterion_id: "c3", criterion_text: "Mentions glucose and oxygen as outputs", max_marks: 2, weight: 1.0 },
      ],
    };
    const rubricJson = JSON.stringify(rubric);
    const rubricHash = sha256Hex(rubricJson);
    const certifiedAt = new Date().toISOString();

    db.prepare(
      `INSERT INTO exam_batches (batch_id, rubric_json, rubric_hash, zone1_model_hash, zone2_model_hash, certified_at, certified_by)
       VALUES (?, ?, ?, ?, ?, ?, ?)`
    ).run(batchId, rubricJson, rubricHash, modelHash.zone1_model_hash, modelHash.zone2_model_hash, certifiedAt, "tamper_test_panel");
    auditChain.append("certify", batchId, rubricHash);

    const answerText =
      "Photosynthesis is the process by which plants use sunlight as their energy source to convert water and carbon dioxide into glucose, releasing oxygen as a byproduct.";
    const result = await evaluationPipeline.submitAndEvaluate({
      batchId,
      studentId: `tamper_test_${Date.now()}`,
      answerText,
      inputSource: "student_typed",
    });

    const baseline = await verifyService.runVerification(result.evaluation_id);

    res.status(201).json({
      batch_id: batchId,
      submission_id: result.submission_id,
      evaluation_id: result.evaluation_id,
      baseline,
    });
  } catch (e) {
    res.status(502).json({ error: "setup_failed", detail: e.message });
  }
});

// Applies a tamper, runs a check, then always undoes the tamper — even if the check throws.
async function withRestore(apply, verify) {
  const restore = await apply();
  try {
    return await verify();
  } finally {
    await restore();
  }
}

// POST /dev/tamper/run  { evaluation_id, vector }
router.post("/run", async (req, res) => {
  const { evaluation_id, vector } = req.body;
  if (!evaluation_id || !vector) {
    return res.status(422).json({ error: "missing_fields", detail: "evaluation_id and vector are required" });
  }
  const meta = VECTORS.find((v) => v.id === vector);
  if (!meta) return res.status(422).json({ error: "unknown_vector", detail: vector });

  const bundle = verifyService.getEvaluationBundle(evaluation_id);
  if (!bundle) return res.status(404).json({ error: "evaluation_not_found" });
  const { evaluation, submission, batch } = bundle;

  try {
    let outcome;

    switch (vector) {
      case "input_tamper": {
        const original = submission.answer_text;
        const tampered = "a completely different answer, swapped in after submission";
        outcome = await withRestore(
          async () => {
            db.prepare("UPDATE submissions SET answer_text = ? WHERE submission_id = ?").run(tampered, submission.submission_id);
            return async () => db.prepare("UPDATE submissions SET answer_text = ? WHERE submission_id = ?").run(original, submission.submission_id);
          },
          async () => {
            const recomputed = sha256Hex(tampered);
            const hashMatches = recomputed === submission.input_hash;
            return {
              overall_valid: hashMatches,
              checks: [
                {
                  name: "input_hash",
                  passed: hashMatches,
                  detail: hashMatches ? null : "stored input_hash no longer matches sha256(current answer_text) — the answer was altered after submission",
                },
              ],
              diff: hashMatches ? null : { input_hash: { expected: submission.input_hash, found: recomputed } },
            };
          }
        );
        break;
      }

      case "witness_substitution": {
        const original = evaluation.per_criterion_scores_json;
        const rubric = JSON.parse(batch.rubric_json);
        const n = rubric.criteria.length;
        const fabricated = JSON.stringify(Array.from({ length: n }, () => 99.9));
        outcome = await withRestore(
          async () => {
            db.prepare("UPDATE evaluations SET per_criterion_scores_json = ? WHERE evaluation_id = ?").run(fabricated, evaluation_id);
            return async () => db.prepare("UPDATE evaluations SET per_criterion_scores_json = ? WHERE evaluation_id = ?").run(original, evaluation_id);
          },
          async () => verifyService.runVerification(evaluation_id)
        );
        break;
      }

      case "stage0_substitution": {
        const original = evaluation.criterion_scores_json;
        const stage0 = JSON.parse(original);
        const fabricated = JSON.stringify({ ...stage0, negation_flags: stage0.negation_flags.map(() => 1) });
        outcome = await withRestore(
          async () => {
            db.prepare("UPDATE evaluations SET criterion_scores_json = ? WHERE evaluation_id = ?").run(fabricated, evaluation_id);
            return async () => db.prepare("UPDATE evaluations SET criterion_scores_json = ? WHERE evaluation_id = ?").run(original, evaluation_id);
          },
          async () => verifyService.runVerification(evaluation_id)
        );
        break;
      }

      case "score_edit": {
        const original = evaluation.final_score;
        outcome = await withRestore(
          async () => {
            db.prepare("UPDATE evaluations SET final_score = ? WHERE evaluation_id = ?").run(99.9, evaluation_id);
            return async () => db.prepare("UPDATE evaluations SET final_score = ? WHERE evaluation_id = ?").run(original, evaluation_id);
          },
          async () => verifyService.runVerification(evaluation_id)
        );
        break;
      }

      case "zone1_drift": {
        const original = batch.zone1_model_hash;
        outcome = await withRestore(
          async () => {
            db.prepare("UPDATE exam_batches SET zone1_model_hash = ? WHERE batch_id = ?").run("deadbeef_tampered_zone1_hash", batch.batch_id);
            return async () => db.prepare("UPDATE exam_batches SET zone1_model_hash = ? WHERE batch_id = ?").run(original, batch.batch_id);
          },
          async () => verifyService.runVerification(evaluation_id)
        );
        break;
      }

      case "zone2_drift": {
        const original = batch.zone2_model_hash;
        outcome = await withRestore(
          async () => {
            db.prepare("UPDATE exam_batches SET zone2_model_hash = ? WHERE batch_id = ?").run("deadbeef_tampered_zone2_hash", batch.batch_id);
            return async () => db.prepare("UPDATE exam_batches SET zone2_model_hash = ? WHERE batch_id = ?").run(original, batch.batch_id);
          },
          async () => verifyService.runVerification(evaluation_id)
        );
        break;
      }

      case "proof_corruption": {
        if (!evaluation.proof_path || !fs.existsSync(evaluation.proof_path)) {
          return res.status(409).json({ error: "proof_file_unavailable", detail: "proof_path not resolvable from this process — re-run /dev/tamper/setup under the currently running ml-worker" });
        }
        const originalBytes = fs.readFileSync(evaluation.proof_path, "utf8");
        outcome = await withRestore(
          async () => {
            const parsed = JSON.parse(originalBytes);
            parsed.proof[0] = (parsed.proof[0] + 1) % 256;
            fs.writeFileSync(evaluation.proof_path, JSON.stringify(parsed));
            return async () => fs.writeFileSync(evaluation.proof_path, originalBytes);
          },
          async () => verifyService.runVerification(evaluation_id)
        );
        break;
      }

      case "scorecard_tamper": {
        if (!evaluation.scorecard_path || !fs.existsSync(evaluation.scorecard_path)) {
          return res.status(409).json({ error: "scorecard_unavailable", detail: "no scorecard on file for this evaluation" });
        }
        const pdfBuffer = fs.readFileSync(evaluation.scorecard_path);
        const corrupted = Buffer.from(pdfBuffer);
        corrupted[corrupted.length - 20] = corrupted[corrupted.length - 20] ^ 0xff;
        const result = await verifyService.verifyScorecard(evaluation_id, evaluation.scorecard_token, corrupted);
        outcome = result; // nothing was persisted — no restore needed
        break;
      }

      case "rubric_edit": {
        const original = batch.rubric_json;
        const rubric = JSON.parse(batch.rubric_json);
        rubric.criteria[0].weight = Number(rubric.criteria[0].weight) + 4.0;
        const tampered = JSON.stringify(rubric);
        outcome = await withRestore(
          async () => {
            db.prepare("UPDATE exam_batches SET rubric_json = ? WHERE batch_id = ?").run(tampered, batch.batch_id);
            return async () => db.prepare("UPDATE exam_batches SET rubric_json = ? WHERE batch_id = ?").run(original, batch.batch_id);
          },
          async () => verifyService.runVerification(evaluation_id)
        );
        break;
      }

      case "wrong_batch": {
        const modelHash = await ml.rcajxModelHash();
        const decoyBatchId = newId("batch");
        const decoyRubric = { criteria: JSON.parse(batch.rubric_json).criteria.map((c) => ({ ...c, weight: Number(c.weight) + 2.0 })) };
        const decoyRubricJson = JSON.stringify(decoyRubric);
        const decoyRubricHash = sha256Hex(decoyRubricJson);
        db.prepare(
          `INSERT INTO exam_batches (batch_id, rubric_json, rubric_hash, zone1_model_hash, zone2_model_hash, certified_at, certified_by)
           VALUES (?, ?, ?, ?, ?, ?, ?)`
        ).run(decoyBatchId, decoyRubricJson, decoyRubricHash, modelHash.zone1_model_hash, modelHash.zone2_model_hash, new Date().toISOString(), "tamper_test_panel");

        const original = submission.batch_id;
        outcome = await withRestore(
          async () => {
            db.prepare("UPDATE submissions SET batch_id = ? WHERE submission_id = ?").run(decoyBatchId, submission.submission_id);
            return async () => db.prepare("UPDATE submissions SET batch_id = ? WHERE submission_id = ?").run(original, submission.submission_id);
          },
          async () => verifyService.runVerification(evaluation_id)
        );
        break;
      }

      case "audit_chain_tamper": {
        const firstEntry = db.prepare("SELECT * FROM audit_chain ORDER BY entry_id ASC LIMIT 1").get();
        if (!firstEntry) return res.status(409).json({ error: "audit_chain_empty" });
        const original = firstEntry.payload_hash;
        outcome = await withRestore(
          async () => {
            db.prepare("UPDATE audit_chain SET payload_hash = ? WHERE entry_id = ?").run("tampered_payload_hash", firstEntry.entry_id);
            return async () => db.prepare("UPDATE audit_chain SET payload_hash = ? WHERE entry_id = ?").run(original, firstEntry.entry_id);
          },
          async () => {
            const integrity = auditChain.verifyChainIntegrity();
            return {
              overall_valid: integrity.intact,
              checks: [
                {
                  name: "audit_chain_integrity",
                  passed: integrity.intact,
                  detail: integrity.intact ? null : `chain broken at entry_id=${integrity.brokenAtEntryId}`,
                },
              ],
              diff: null,
            };
          }
        );
        break;
      }

      default:
        return res.status(422).json({ error: "unknown_vector", detail: vector });
    }

    res.json({ vector, label: meta.label, category: meta.category, evaluation_id, ...outcome, caught: outcome.overall_valid === false });
  } catch (e) {
    res.status(502).json({ error: "tamper_run_failed", detail: e.message });
  }
});

module.exports = router;
