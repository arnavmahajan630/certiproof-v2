# CertiProof Demo Script — Eight Acts

*Acts 1-6 are the base six-act demo; Act 3.5 is the OCR→scorecard track; Act 4.5 is the new explainability beat, tying the RCAJ-X migration's headline claim directly to the proof.*

## Act 1 — Certify
Teacher view. Define a rubric (3-5 criteria, each with max marks + weight) for a batch. Certify → Gateway hashes rubric + the fine-tuned encoder's identity + the RCAJ_X checkpoint's identity into `exam_batches`, locked before any submission exists. Say aloud: "This is the point of no return — anything that changes after this is exactly what the rest of the demo catches."

## Act 2 — Submit & Receipt
Student view. Type an answer, submit. Gateway returns `input_hash` immediately, before grading starts — "you get a receipt for exactly what you submitted, before anything else happens to it."

## Act 3 — Evaluate & Prove
Show the proving-progress UI (honest staged indication, not a generic spinner): Stage 0 embedding → witness commit → Stages 1-3 attention scoring → EZKL proving → done. Final score appears. Point out: "the witness_hash was written to the audit chain the instant the embeddings were computed — before proving even started, before the model even looked at which parts of the answer matter."

## Act 3.5 — OCR → Scorecard
Switch to Teacher view. Upload a photographed answer sheet for a different student. Gemini transcribes it live (or falls back to a pre-cached transcription if the API is unreachable — see note below). Same pipeline runs. Once evaluated, download the scorecard PDF — show the QR code (scan with a phone, opens the Auditor verify page pre-filled) and the per-criterion breakdown, now with a one-sentence evidence summary under each score. Switch to Student view, upload that exact PDF to "Verify My Scorecard" → passes.

## Act 4 — Break-It (centerpiece)
Pick one of: edit the stored per-criterion scores directly in the DB, edit the final score, corrupt the proof file, or edit the just-downloaded scorecard PDF. Re-run `/verify` (or re-upload the edited scorecard). Show the *specific* check that fails — "the proof itself is still structurally valid, but the scores it actually contains don't match what's displayed" or "scorecard hash doesn't match — this file was modified after issuance." This is the whole claim of the project in one moment.

## Act 4.5 — Explainability (new — the architecture's actual headline)
Still in the Auditor or Student view from Act 4's untampered record: point at a specific criterion's score and read its reason text aloud — "here's a student's answer, here's their score — and here's exactly which part of their answer the certified model weighted most heavily for each criterion." Then land the real point: "this isn't a separate explanation bolted on after the fact — it's the same attention computation the ZK proof is over. Previously we only proved the aggregation arithmetic; now we prove the evidence-weighing itself." If any criterion in this record was flagged `review_recommended`, show it in the Auditor's Confidence Shortlist — "the model tells you what it's unsure about, and that flag is part of what's proven too."

## Act 5 — Teacher Override
Teacher reviews the (untampered) evaluation, overrides the score with a reason. Show that this created a *new* `overrides` row — the original evaluation and its proof are untouched, still independently verifiable.

## Act 6 — Close
Auditor view: pull up the full audit chain for the batch, show it's hash-linked end to end. One line to land: "every one of these checks runs from public data only — no one has to trust our word for it, including the part where we explain the score."

---

**Network dependency note:** everything except Act 3.5's live Gemini call runs fully offline. Have a pre-cached transcription result ready (a known image → known text pair, cached locally) as fallback if the venue's network drops mid-demo — swap to it without breaking flow, and say so plainly rather than pretending it was live.

**Pre-cached artifacts:** EZKL proof generation is pre-cached for the two seeded demo records (`scripts/seed_demo_data.sh`) — proving is not run live on stage; verification (fast) is what runs live. Act 3's live proving is optional/bonus if time allows and hardware cooperates (the RCAJ-X circuit's proving cost is meaningfully higher than the old plain-MLP Zone 2 circuit — multi-head softmax instances vs. zero before); the pre-cached records are the reliable fallback.
