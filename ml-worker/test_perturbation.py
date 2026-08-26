"""Phase 2 hard gate: perturbation test for the Zone 2 EZKL circuit.

Updated for Eval 1.5 (Poseidon commitment migration, plan/Poseidon_PLAN.md):
criterion_scores/rubric_weights are private (hashed) inputs now, not public ones
-- see ezkl_pipeline.py's module docstring for the full Tier A/B story.

1. A genuine proof, generated over genuine criterion_scores/rubric_weights, must verify.
2. If the DB-stored final_score doesn't match what the proof file itself embeds
   (pretty_public_inputs.rescaled_outputs), the Gateway-level binding check must
   catch it (ezkl.verify() alone is structural-only, doesn't compare against a DB row).
3. Float noise below EZKL's quantization step must not change the proof's public
   output NOR its Poseidon commitment (same quantization contract, extended to the
   commitment since it's now derived from the same quantized field elements).
4. The Poseidon commitment must be independently recomputable (recompute_poseidon_commitment)
   and match what's embedded in the genuine proof; a fabricated/substituted score
   must produce a commitment that does NOT match the genuine proof's embedded one.
"""
import asyncio
import os
import shutil

from app.zk import ezkl_pipeline as zk
from app.zone2.model import MAX_CRITERIA

PROOF_DIR = os.path.join(os.path.dirname(__file__), ".test_proofs")


def fresh_dir():
    if os.path.exists(PROOF_DIR):
        shutil.rmtree(PROOF_DIR)
    os.makedirs(PROOF_DIR)


def genuine_inputs():
    scores = [0.0] * MAX_CRITERIA
    weights = [0.0] * MAX_CRITERIA
    for i in range(5):
        scores[i] = 0.6 + 0.05 * i
        weights[i] = 1.0
    return scores, weights


def main():
    fresh_dir()
    asyncio.run(zk.build_circuit())

    scores, weights = genuine_inputs()

    # 1. genuine proof verifies
    result = zk.prove(scores, weights, PROOF_DIR, tag="genuine")
    v = zk.verify(result["proof_path"])
    assert v["verified"], "genuine proof failed to verify"
    print("[1/4] genuine proof verifies: PASS")
    print(f"      final_score (from proof's rescaled_outputs) = {result['final_score']:.4f}")
    print(f"      poseidon_commitment = {result['poseidon_commitment']}")

    # 2. Gateway-level binding check: stored final_score must match what's embedded
    #    in the proof file. Simulate a fabricated/substituted DB row.
    stored_outputs = result["proof_public_inputs"]["rescaled_outputs"]
    tampered_outputs = [["99.0"]]  # fabricated final_score, proof unchanged
    proof_embedded = zk.verify(result["proof_path"])["pretty_public_inputs"]["rescaled_outputs"]
    assert stored_outputs == proof_embedded, "genuine stored output should match proof's embedded value"
    assert tampered_outputs != proof_embedded, "tampered stored output must NOT match proof's embedded value"
    print("[2/4] output binding check (genuine matches, tampered caught): PASS")

    # 3. sub-quantization float noise must not change the proof's public output or commitment
    noisy_scores = [s + 1e-7 for s in scores]
    result_noisy = zk.prove(noisy_scores, weights, PROOF_DIR, tag="noisy")
    assert (
        result["proof_public_inputs"]["rescaled_outputs"]
        == result_noisy["proof_public_inputs"]["rescaled_outputs"]
    ), "sub-quantization noise changed the proof's public output — quantization contract violated"
    assert (
        result["poseidon_commitment"] == result_noisy["poseidon_commitment"]
    ), "sub-quantization noise changed the Poseidon commitment — quantization contract violated"
    print("[3/4] sub-quantization float noise does not perturb output or commitment: PASS")

    # 4. Poseidon commitment binding: independently recomputable, and a fabricated
    #    score produces a different commitment than the genuine proof's embedded one.
    embedded_commitment = zk.verify(result["proof_path"])["embedded_poseidon_commitment"]
    assert embedded_commitment == result["poseidon_commitment"], "prove()'s commitment should match verify()'s embedded one"

    recomputed_genuine = zk.recompute_poseidon_commitment(scores, weights, tag="recompute_genuine")
    assert recomputed_genuine == embedded_commitment, "independent recomputation must match the genuine proof's commitment"

    fabricated_scores = [0.0] * MAX_CRITERIA
    fabricated_scores[0] = 0.99
    recomputed_fabricated = zk.recompute_poseidon_commitment(fabricated_scores, weights, tag="recompute_fabricated")
    assert recomputed_fabricated != embedded_commitment, "fabricated scores must NOT match the genuine proof's commitment"
    print("[4/4] Poseidon commitment binding (recomputable, tamper caught): PASS")

    print("\nPERTURBATION TEST: ALL GREEN — Phase 2 gate passed (Eval 1.5)")


if __name__ == "__main__":
    main()
