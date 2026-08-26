# Auditor "Tests" Panel — Summary

A section on the **Auditor** view (`frontend/src/views/Auditor.jsx`), below Audit Chain, that lets you run all eleven tamper vectors live, in the browser, and see the real cryptographic `/verify` response for each — no terminal, no pre-baked data.

## What it does

1. **Set Up Fresh Test Case** — certifies a throwaway rubric (3 photosynthesis criteria), submits a genuine answer through the real pipeline (Stage 0 embed → witness commit → Stages 1-3 attention scoring → EZKL proof → scorecard), and shows the baseline result: everything valid.
2. Pick any vector (or **Run All Vectors**) — each one:
   - applies the tamper (a DB edit, a corrupted proof byte, a swapped PDF, etc.)
   - calls the actual `/verify` chain (or the scorecard-verify path)
   - shows the real check list and diff table, exactly like the main Verify panel
   - **automatically restores** the original state before returning, so the test case stays reusable and the rest of the demo is unaffected
3. Everything is live — this isn't mocked data, it's the same verification code path a real auditor hits.

## The eleven vectors, grouped

**Input tampering**
- **Answer edited after the fact** — `answer_text` changed post-submission; re-hashing it no longer matches the `input_hash` receipt issued at submit time.

**Model / grading tampering**
- **Displayed scores fabricated after proving (witness-substitution, the flagship vector)** — the displayed per-criterion scores (`per_criterion_scores_json`) swapped for fabricated ones, as if someone edited the display record after the fact. The EZKL proof itself was generated against the *real* scores and stays structurally valid — `proof_validity` independently re-derives the true scores from the proof's own public output and catches the mismatch.
- **Stage 0 inputs fabricated after the fact** — the stored Stage 0 record (embeddings, negation flags — what `witness_hash` was committed over, before proving) swapped for a fabricated one; re-hashing the current record no longer matches the independently-committed `witness_hash`.
- **Final score edited directly in the DB** — the proof's own embedded public output no longer matches the stored score.
- **Encoder model swapped** — certified `zone1_model_hash` no longer matches what the live ML-Worker reports for the fine-tuned encoder.
- **RCAJ_X model/circuit swapped** — same, for the checkpoint + circuit + verifying key identity.

**Proof & artifact tampering**
- **Proof file corrupted** — one byte flipped inside the EZKL proof on disk; the verifier rejects it outright.
- **Scorecard PDF edited after issuance** — re-uploaded PDF's SHA-256 no longer matches the hash recorded at generation time.

**Record & chain tampering**
- **Certified rubric edited in place** — a criterion weight changed after certification; `rubric_hash` self-consistency check fails.
- **Evaluation re-associated with a different batch** — submission re-pointed at another (also validly certified) batch with different weights; `poseidon_commitment`, recomputed against the new weights, no longer matches.
- **Historical audit-chain entry edited** — a row in the append-only chain is altered directly; hash-link verification finds the first broken link.

## How it's built

- **Backend:** `gateway/src/routes/devTamper.routes.js`, mounted at `/dev/tamper` in `gateway/src/app.js`.
  - `POST /dev/tamper/setup` — fresh batch + submission + evaluation, returns baseline `/verify` result.
  - `GET /dev/tamper/vectors` — vector metadata (id, category, label, description) for the UI to render.
  - `POST /dev/tamper/run` — `{ evaluation_id, vector }` → applies, verifies, restores, returns the check list.
- **Frontend:** `TestsPanel` component inside `Auditor.jsx`, calling three `api.tamper*` methods in `frontend/src/api/client.js`.
- Mirrors `scripts/tamper_suite.sh`'s vectors — same underlying tamper technique for each — just exposed over HTTP so it can be driven from the UI live, instead of the terminal.

## Status (RCAJ-X migration)

Vector logic and rubric shapes (`criterion_id`/`max_marks` now required) updated for the new Stage 0 → witness_hash → prove → score/explain pipeline and the new `per_criterion_scores_json`/`explanations_json` columns. **Not yet re-run end-to-end** — that requires a built EZKL circuit (`app/models/rcajx/rcajx_circuit/`), which is pending the cross-machine training + circuit-setup step (see `TRAINING_AND_EZKL_PLAN.md` at the repo root). Re-run `scripts/tamper_suite.sh` and this panel once the circuit is in place, especially the witness-substitution vector — that's the one the whole design exists for.
