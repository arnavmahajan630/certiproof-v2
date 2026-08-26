const db = require("../db/client");
const ml = require("./mlWorkerClient");
const auditChain = require("./auditChain");
const { sha256Hex, witnessHashFromRawJson, newId } = require("./hashing");
const scorecardService = require("./scorecardService");

// RCAJ-X evaluation flow (docs/api_contract.md "Evaluation pipeline" section).
// Order is load-bearing — do not reorder, in particular do not call /ezkl/prove
// before witness_hash is committed. Stage 0 (embed) -> witness commit -> Stage
// 1-3 proving (authoritative scores) + non-proving score/explain (display) run
// off the SAME Stage 0 values -> persist -> scorecard.
async function submitAndEvaluate({ batchId, studentId, answerText, inputSource, answerSheetHash }) {
  const batch = db.prepare("SELECT * FROM exam_batches WHERE batch_id = ?").get(batchId);
  if (!batch) {
    const err = new Error(`batch ${batchId} not found`);
    err.code = "BATCH_NOT_FOUND";
    throw err;
  }
  const rubric = JSON.parse(batch.rubric_json);
  const criteria = rubric.criteria.map((c) => ({
    criterion_id: c.criterion_id,
    criterion_text: c.criterion_text,
    max_marks: Number(c.max_marks),
  }));
  // criterion_weights for RCAJ_X's final_score aggregation, padded to MAX_CRITERIA=10
  // with zeros (matches the ml-worker's static circuit shape — see
  // app/rcajx/padded_model.py). Order must match `criteria` above.
  const criterionWeights = rubric.criteria.map((c) => Number(c.weight));
  while (criterionWeights.length < 10) criterionWeights.push(0);

  // Step 1: input_hash receipt, immediately.
  const submissionId = newId("sub");
  const inputHash = sha256Hex(answerText);
  const submittedAt = new Date().toISOString();
  db.prepare(
    `INSERT INTO submissions (submission_id, batch_id, student_id, answer_text, input_hash, input_source, answer_sheet_hash, submitted_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(submissionId, batchId, studentId, answerText, inputHash, inputSource, answerSheetHash || null, submittedAt);
  auditChain.append("submit", submissionId, inputHash);

  // Step 2: Stage 0 (embed) — R/A embeddings, chunks, negation_flags, max_marks.
  const { raw: stage0Raw, parsed: stage0 } = await ml.rcajxEmbed(answerText, criteria, {
    questionId: batch.batch_id, // rubric-locking cache keys off this repo's "question_id" concept; batch_id is the closest stable per-rubric identity CertiProof has
    rubricMode: "dev",
  });

  // Step 3: witness_hash committed BEFORE Stage 1-3 / proving — exact raw JSON string bytes.
  const witnessHash = witnessHashFromRawJson(stage0Raw);
  const evaluationId = newId("eval");
  auditChain.append("witness_commit", evaluationId, witnessHash);

  // Step 4a: prove — the SAME stage0 values just witness-committed. per_criterion_scores
  // and final_score here are the proof's own public output, authoritative.
  const proveResult = await ml.ezklProve(stage0, criterionWeights, evaluationId);
  const perCriterionScores = proveResult.per_criterion_scores.slice(0, criteria.length);

  // Step 4b: fast non-proving score + explanations, off the SAME stage0 values —
  // deterministic given identical inputs, so this reproduces the proof's own
  // per_criterion_scores (not re-trusted independently; per_criterion_scores for
  // display/scorecard/DB always comes from Step 4a above) while additionally
  // surfacing evidence chunks / reason text / confidence, which aren't part of the
  // proof's public output.
  const scoreResult = await ml.rcajxScore(stage0, criteria, criterionWeights.slice(0, criteria.length));

  // Step 5: persist.
  const evaluatedAt = new Date().toISOString();
  db.prepare(
    `INSERT INTO evaluations
       (evaluation_id, submission_id, criterion_scores_json, witness_hash, per_criterion_scores_json, explanations_json, proof_public_inputs_json, poseidon_commitment, final_score, proof_path, proof_status, evaluated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(
    evaluationId,
    submissionId,
    stage0Raw,
    witnessHash,
    JSON.stringify(perCriterionScores),
    JSON.stringify(scoreResult.explanations),
    JSON.stringify(proveResult.proof_public_inputs),
    proveResult.poseidon_commitment,
    proveResult.final_score,
    proveResult.proof_path,
    "proved",
    evaluatedAt
  );
  auditChain.append("proof_generated", evaluationId, sha256Hex(proveResult.proof_path));

  // Step 6: scorecard.
  const scorecard = await scorecardService.generateScorecard(evaluationId);

  const evaluation = db.prepare("SELECT * FROM evaluations WHERE evaluation_id = ?").get(evaluationId);

  return {
    submission_id: submissionId,
    input_hash: inputHash,
    evaluation_id: evaluationId,
    evaluation,
    scorecard,
  };
}

module.exports = { submitAndEvaluate };
