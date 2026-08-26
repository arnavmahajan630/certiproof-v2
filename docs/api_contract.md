# CertiProof API Contract

CertiProof's ML pipeline runs on **RCAJ-X**: a fine-tuned BGE-small bi-encoder (Stage 0) feeding a multi-head cross-attention + bounded scoring head (Stages 1-3), replacing the old Zone 1 cross-encoder + Zone 2 MLP split. Today only the final aggregation arithmetic was proven; now the evidence-weighing step itself (attention over answer chunks) is proven too.

## ML-Worker (FastAPI, stateless, no DB access)

### `POST /rcajx/embed`
Stage 0. Request: `{ "answer_text": str, "criteria": [{"criterion_id": str, "criterion_text": str, "max_marks": float}, ...] }` (1-10 criteria), optionally `question_id`/`rubric_mode` ("dev"|"locked") to use the rubric-locking cache.
Response: `{ "R": [[float x384], ...], "A": [[float x384], ...], "chunks": [str, ...], "negation_flags": [float, ...], "max_marks": [float, ...] }`. `R`/`A` are unpadded (real criteria/chunk counts, not yet padded to `MAX_CRITERIA=10`/`MAX_CHUNKS=24` — that padding only happens inside the circuit-facing `/ezkl/*` calls). **This response's raw bytes are what the Gateway commits as `witness_hash`, before any Stage 1-3 computation runs.**

### `POST /rcajx/score`
Stages 1-3, fast non-proving preview + explanations. Request: the `/rcajx/embed` response fields (`R`, `A`, `chunks`, `negation_flags`, `max_marks`) plus `criteria` and `criterion_weights`.
Response: `{ "per_criterion_scores": [float, ...], "final_score": float|null, "spread": [[float, ...], ...], "explanations": [{"criterion_id", "criterion_text", "score", "max_marks", "evidence_chunks", "evidence_weights", "confidence": "high_confidence"|"review_recommended", "negation_flag": bool, "reason_text": str}, ...] }`. Deterministic given the same inputs, so this reproduces the proof's own `per_criterion_scores` — but `per_criterion_scores` for storage/display always comes from `/ezkl/prove`'s response, never from here (this call exists for the `explanations`, which aren't part of the ONNX graph's output).

### `POST /ezkl/prove`
Request: `{ "R": [[float x384], ...], "A": [[float x384], ...], "negation_flags": [float, ...], "max_marks": [float, ...], "criterion_weights": [float x10], "tag": str }`. Internally pads `R`/`A`/`negation_flags`/`max_marks` to `(MAX_CRITERIA=10, MAX_CHUNKS=24)` with attention masking (see `app/rcajx/padded_model.py`) before proving — the circuit only ever sees fixed shapes.
Response:
```json
{
  "proof_path": "path/to/proof.json",
  "witness_path": "path/to/witness.json",
  "poseidon_commitment": "245d2bc4...",
  "proof_public_inputs": { "rescaled_outputs": [[...10 scores], [74.13], [...480 attn], [...20 spread]] },
  "per_criterion_scores": [1.8, 1.6, ...],
  "final_score": 74.13
}
```
`R`/`A`/`negation_flags`/`max_marks`/`criterion_weights` are all private (`hashed`) EZKL inputs — the circuit's public outputs are `per_criterion_scores`, `final_score`, `attn_weights`, and `spread` (in that order; `rescaled_outputs[0]` = scores, `[1]` = final_score, `[2]` = attention weights, `[3]` = spread), plus the Poseidon commitment of the private inputs. **The Gateway must store `per_criterion_scores`, `final_score`, `proof_public_inputs`, and `poseidon_commitment` verbatim from this response — never recompute independently** (numeric precision contract).

### `POST /ezkl/recompute-commitment`
Request: same shape as `/ezkl/prove` minus `tag`→same `tag` field. Response: `{ "poseidon_commitment": str }`.
Independently recomputes the commitment for arbitrary (typically DB-stored) values via a fresh witness — the Gateway's `/verify` calls this with the stored Stage 0 values and the batch's *current* `criterion_weights`, then compares against the proof's own embedded commitment. Catches both proving-step fabrication and wrong-batch re-association under one check.

### `POST /ezkl/verify`
Request: `{ "proof_path": str }`.
Response: `{ "verified": bool, "pretty_public_inputs": {...} | null, "embedded_poseidon_commitment": str | null, "error": str | null }`.
Structural check only — confirms the proof is cryptographically valid against the verifying key. Does **not** compare against any DB row; that's the Gateway's job. `embedded_poseidon_commitment` is `instances[0][0]` from the proof file, directly string-comparable against `/ezkl/recompute-commitment`'s output.

### `GET /rcajx/model-hash`
Response: `{ "zone1_model_hash": str, "zone2_model_hash": str, "rcajx_model_hash"?: str }`. Column names preserved from the old architecture (repointed meaning): `zone1_model_hash` = hash of the fine-tuned BGE-small encoder's weight bytes, `zone2_model_hash` = hash of the RCAJ_X checkpoint (attention + scoring head) weights+config. `rcajx_model_hash` (the compiled circuit's hash) is only present once the EZKL circuit has been built — see `ml-worker/README.md` "Circuit setup".

### `POST /ocr/transcribe`
Request: multipart file upload (image).
Response: `{ "answer_text": str, "ocr_raw_response_hash": str }`. Calls Gemini's vision model. No DB access; the Gateway persists everything. Unaffected by the RCAJ-X migration — feeds the same `answer_text` into `/rcajx/embed` either way.

All ML-Worker routes return `{"error": "...", "detail": "..."}` with a 4xx/5xx status and a distinguishable error type on failure, so the Gateway's `/verify` can report which artifact diverged.

---

## Gateway (Express + SQLite, owns state)

### Teacher
- `POST /teacher/batches` — create + certify an `exam_batches` row. `rubric.criteria[i]`: `{criterion_id, criterion_text, max_marks, weight}` — all four required (RCAJ-X needs `criterion_id`/`max_marks`; the old Zone1/Zone2 pipeline only needed `criterion_text`/`weight`).
- `POST /teacher/submissions/:batch_id/upload-answer-sheet` — `{student_id, file}` → OCR → creates a `submissions` row with `input_source='teacher_ocr'`, then evaluates (see below).
- `POST /teacher/evaluations/:evaluation_id/override` — new `overrides` row, original evaluation untouched.
- `GET /teacher/batches/:batch_id/submissions` — list for review. Each row is the `submissions` columns plus `evaluation_id`, `final_score`, `proof_status`, `witness_hash`, `per_criterion_scores_json` (stringified array), `explanations_json` (stringified array — evidence/reasoning per criterion).

### Student
- `POST /student/submissions` — `{batch_id, student_id, answer_text}` → `input_source='student_typed'` → same evaluation pipeline as OCR path.
- `GET /student/submissions/:submission_id` — status + score once evaluated.
- `GET /student/evaluations?student_id=...` — student-facing lookup. Returns `{ student_id, evaluations[] }` with score, submitted/evaluated/scorecard timestamps, batch, `per_criterion_scores_json`, `explanations_json`. Omits answer text and cryptographic internals.
- `GET /student/evaluations/:evaluation_id/scorecard` — downloads the generated scorecard PDF (now includes a 1-2 sentence evidence summary per criterion).
- `POST /student/verify-scorecard` — multipart PDF upload → integrity + full `/verify` chain, response shaped for a student audience.

### Auditor
- `GET /verify?evaluation_id=...` — runs the check chain: `proof_validity` (structural + per_criterion_scores/final_score binding), `poseidon_commitment`, `witness_hash`, `model_rubric_commitment`, `audit_chain_integrity`. Also returns `confidence_summary: {total_criteria, flagged: [...]}` — which criteria the model itself flagged `review_recommended`, straight from `explanations_json`.
- `GET /audit-chain` — raw chain for inspection/demo.

### Evaluation pipeline (shared by both submission paths, Gateway-orchestrated — order is load-bearing)
1. Compute `input_hash` immediately on `answer_text`, return as receipt.
2. Call ML-Worker `/rcajx/embed` (Stage 0) → `{R, A, chunks, negation_flags, max_marks}`.
3. **Immediately**, before anything else: `witness_hash = sha256(exact JSON string returned by /rcajx/embed)`, write to `audit_chain`.
4. Call ML-Worker `/ezkl/prove` with the same Stage 0 values + `criterion_weights` → `per_criterion_scores` (authoritative), `final_score`, `proof_public_inputs`, `poseidon_commitment`, `proof_path`.
5. Call ML-Worker `/rcajx/score` with the SAME Stage 0 values → `explanations` (evidence chunks, reason text, confidence — deterministic given identical inputs, so this reproduces step 4's scores without being independently trusted for them).
6. Persist `evaluations` row: `criterion_scores_json` (Stage 0 raw JSON, from step 2 — column name preserved, meaning repointed), `witness_hash`, `per_criterion_scores_json` (from step 4), `explanations_json` (from step 5), `proof_public_inputs_json` + `poseidon_commitment` (from step 4), `final_score` (from step 4), `proof_path`.
7. Once persisted, generate the scorecard PDF (`scorecardService.js`) → `scorecard_hash`, `scorecard_token`.

This document is the single source of truth both services build against — if either side needs to deviate, update this file first.
