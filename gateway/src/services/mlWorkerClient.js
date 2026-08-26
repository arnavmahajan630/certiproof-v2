const ML_WORKER_URL = process.env.ML_WORKER_URL || "http://127.0.0.1:8001";

class MlWorkerError extends Error {
  constructor(status, body) {
    super(`ML-Worker error ${status}: ${JSON.stringify(body)}`);
    this.status = status;
    this.body = body;
  }
}

async function postRaw(path, body) {
  const res = await fetch(`${ML_WORKER_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) {
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = { error: "unknown", detail: text };
    }
    throw new MlWorkerError(res.status, parsed);
  }
  return text; // raw response body string
}

async function getJson(path) {
  const res = await fetch(`${ML_WORKER_URL}${path}`);
  const text = await res.text();
  if (!res.ok) {
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = { error: "unknown", detail: text };
    }
    throw new MlWorkerError(res.status, parsed);
  }
  return JSON.parse(text);
}

// Stage 0. Returns { raw, parsed } — `raw` is the exact response body string, which
// is what witness_hash must be computed over (numeric precision contract), committed
// BEFORE Stage 1-3 / proving runs. `criteria`: [{criterion_id, criterion_text, max_marks}].
async function rcajxEmbed(answerText, criteria, { questionId, rubricMode } = {}) {
  const raw = await postRaw("/rcajx/embed", {
    answer_text: answerText,
    criteria,
    question_id: questionId ?? null,
    rubric_mode: rubricMode || "dev",
  });
  return { raw, parsed: JSON.parse(raw) };
}

// Stages 1-3, fast non-proving preview + explanations. `stage0`: the parsed
// /rcajx/embed response (R, A, chunks, negation_flags, max_marks) -- pass the SAME
// values that were witness-hash-committed and proved, never re-derived.
async function rcajxScore(stage0, criteria, criterionWeights) {
  const raw = await postRaw("/rcajx/score", {
    R: stage0.R,
    A: stage0.A,
    chunks: stage0.chunks,
    negation_flags: stage0.negation_flags,
    max_marks: stage0.max_marks,
    criteria,
    criterion_weights: criterionWeights,
  });
  return JSON.parse(raw);
}

// Stages 1-3, proof-bound. Same stage0 values as rcajxScore/witness commit --
// per_criterion_scores and final_score in the result are the proof's own public
// output (rescaled_outputs), never independently recomputed.
async function ezklProve(stage0, criterionWeights, tag) {
  const raw = await postRaw("/ezkl/prove", {
    R: stage0.R,
    A: stage0.A,
    negation_flags: stage0.negation_flags,
    max_marks: stage0.max_marks,
    criterion_weights: criterionWeights,
    tag,
  });
  return JSON.parse(raw);
}

async function ezklVerify(proofPath) {
  const raw = await postRaw("/ezkl/verify", { proof_path: proofPath });
  return JSON.parse(raw);
}

// Independently recompute the Poseidon commitment of (R, A, negation_flags, max_marks,
// criterion_weights) from arbitrary (typically DB-stored) values, for the
// poseidon_commitment check.
async function ezklRecomputeCommitment(stage0, criterionWeights, tag) {
  const raw = await postRaw("/ezkl/recompute-commitment", {
    R: stage0.R,
    A: stage0.A,
    negation_flags: stage0.negation_flags,
    max_marks: stage0.max_marks,
    criterion_weights: criterionWeights,
    tag,
  });
  return JSON.parse(raw);
}

// Returns { zone1_model_hash, zone2_model_hash, rcajx_model_hash? } -- same column
// names as before (Phase 4: repointed meaning, not renamed). rcajx_model_hash is only
// present once the EZKL circuit has been built (see ml-worker/README.md "Circuit setup").
async function rcajxModelHash() {
  return getJson("/rcajx/model-hash");
}

async function ocrTranscribe(fileBuffer, filename, mimeType) {
  const form = new FormData();
  form.append("file", new Blob([fileBuffer], { type: mimeType }), filename);
  const res = await fetch(`${ML_WORKER_URL}/ocr/transcribe`, { method: "POST", body: form });
  const text = await res.text();
  if (!res.ok) {
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = { error: "unknown", detail: text };
    }
    throw new MlWorkerError(res.status, parsed);
  }
  return JSON.parse(text);
}

module.exports = {
  MlWorkerError,
  rcajxEmbed,
  rcajxScore,
  ezklProve,
  ezklVerify,
  ezklRecomputeCommitment,
  rcajxModelHash,
  ocrTranscribe,
};
