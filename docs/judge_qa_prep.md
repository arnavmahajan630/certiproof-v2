# Judge Q&A Prep

## The three answers worth having word-for-word

**"Couldn't someone just prove whatever score they want?"**
No — the ZK proof alone would let a compromised proving step prove *consistency* with a fabricated input, but it can't prove that input's provenance. That's why there's a second, independent mechanism: the moment the Gateway receives the real Stage 0 embeddings (before Stages 1-3 or EZKL ever run), it hashes them and commits `witness_hash` to the audit chain. If someone later substitutes different scores at display time, or a different Stage 0 record before proving, the resulting proof (or the stored display record) diverges from what was independently committed — `proof_validity` re-derives the true scores from the proof's own public output, and `witness_hash` re-derives the hash of the Stage 0 record. The ZK proof binds the *math*, the witness_hash binds the *provenance*. Both have to pass.

**"How do you know the OCR transcription is accurate or genuine?"**
We don't cryptographically verify it, and we say so directly rather than overclaiming. The teacher-upload → Gemini-transcribe path is a convenience feature, not a proof input — the transcribed text is trusted at exactly the level a human-typed answer already is. The ZK/witness_hash machinery starts *after* `answer_text` exists, regardless of whether a student typed it or a teacher's photo was OCR'd. Proving OCR provenance would need a separate mechanism (e.g. a signed attestation from the OCR provider) that's out of scope for this build and disclosed as such.

**"What's actually proven now, that wasn't before?" (the RCAJ-X migration, in one breath)**
Previously, only the final aggregation arithmetic was proven — the semantic matching itself (a 278M-parameter cross-encoder) was off-circuit, only provenance-committed via witness_hash, never proven. The architecture was specifically redesigned to keep the evidence-weighing step cheap enough to prove: a small multi-head cross-attention block over bounded embeddings, not a full semantic model. Now the attention computation that decides *which part of the answer supports which criterion* is proven directly, on top of the same witness_hash provenance guarantee the aggregation always had. The explanation shown to students/auditors is generated from that exact same attention computation — not a separate, unverified explanation bolted on afterward.

## Supporting context

**Scale (India):** Indian examination systems process crores of students annually — 1.4 crore+ in Class 12 boards, 20 lakh+ in NEET, 11.6 lakh+ in CUET. This is the argument for why tamper-evident, privacy-preserving, *explainable* grading verification matters at scale, not something the demo itself simulates.

**Regulatory anchor:** DPDP Act 2023 and its 2025 rules govern consent-based processing and data minimization for exactly the data this system touches (student answers, scores, and now the evidence chunks backing each score). The architecture's core privacy claim — verification and explanation without exposing raw answer text or model internals beyond what's needed — aligns with data-minimization by design, not as an afterthought.

**Feasibility grounding:** the RCAJ-X circuit (multi-head attention + sigmoid-bounded scoring) compiles and proves under EZKL 23.0.5 — confirmed by actually compiling it, not assumed. It is a meaningfully bigger circuit than the old plain-MLP Zone 2 (multi-head softmax instances vs. zero before), and its one-time trusted setup needs a machine with substantially more RAM than the demo laptop — a real, disclosed engineering constraint, not glossed over. Proving/verifying at demo time is what actually needs to be fast on the demo hardware, and that's the split this build makes.

**On novelty:** the underlying ZK primitives (Halo2/Plonk via EZKL) aren't novel — say this plainly if asked. The contribution here is the application: proving the evidence-weighing computation itself (not just downstream arithmetic), combined with an independently-timed provenance commitment (`witness_hash`), applied to a real, disclosed, staged pipeline for exam grading verification — and generating the human-facing explanation from the exact computation being proven, not a separate step.

**Why bounded attention, not the model's full semantic judgment, is what's proven:** current ZK tooling can't put a full free-form language model inside a provable circuit at reasonable cost. RCAJ-X's bi-encoder (Stage 0, off-circuit) does the heavy semantic embedding; the proven part (Stages 1-3) is a small, deliberately bounded attention + scoring computation over those embeddings — cheap enough to prove, expressive enough to ground an explanation in. This is a complete, honest claim on its own, not a compromise dressed up.

## The ten tamper vectors, in one line each

1. RCAJ_X model/circuit swap after certification → `zone2_model_hash` mismatch.
2. Encoder model/config swap after certification → `zone1_model_hash` mismatch.
3. Answer altered after submission → `input_hash` mismatch.
4. Recorded final score edited in DB → derived from the proof's own public output no longer matches the stored value (`proof_validity`).
5. Stored per-criterion scores edited after proving → derived from the proof's own public output no longer matches the stored `per_criterion_scores_json` (`proof_validity`) — the proof itself is untouched and structurally valid, only the display record was tampered.
6. **Stage 0 inputs (embeddings) fabricated after the fact** → `witness_hash` (committed independently, before Stages 1-3 ran) no longer matches a re-hash of the stored record. *This is the one to lead with if asked "what's the actual innovation here."*
7. Proof artifact corrupted/forged → EZKL verifier rejects outright.
8. Historical audit-chain row edited → hash-link verification finds the first broken link.
9. Evaluation re-associated with a different (also validly certified) batch → the `poseidon_commitment` check catches it: recomputing `Poseidon(stored Stage 0 values ++ the batch's *current* criterion_weights)` no longer matches the proof's embedded commitment, since it was originally computed against a different batch's weights.
10. Scorecard PDF edited after issuance → `sha256` of the re-uploaded PDF no longer matches `scorecard_hash` recorded at generation time — independent of, and complementary to, checks 1-9.
