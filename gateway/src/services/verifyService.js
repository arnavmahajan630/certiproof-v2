const db = require("../db/client");
const ml = require("./mlWorkerClient");
const auditChain = require("./auditChain");
const { sha256Hex } = require("./hashing");

function getEvaluationBundle(evaluationId) {
  const evaluation = db.prepare("SELECT * FROM evaluations WHERE evaluation_id = ?").get(evaluationId);
  if (!evaluation) return null;
  const submission = db.prepare("SELECT * FROM submissions WHERE submission_id = ?").get(evaluation.submission_id);
  const batch = db.prepare("SELECT * FROM exam_batches WHERE batch_id = ?").get(submission.batch_id);
  return { evaluation, submission, batch };
}

function currentCriterionWeights(batch) {
  const rubric = JSON.parse(batch.rubric_json);
  const weights = rubric.criteria.map((c) => Number(c.weight));
  while (weights.length < 10) weights.push(0);
  return weights;
}

// The check chain: proof_validity, poseidon_commitment, witness_hash,
// model_rubric_commitment, audit_chain_integrity. Returns { overall_valid, checks, diff }.
async function runVerification(evaluationId) {
  const bundle = getEvaluationBundle(evaluationId);
  const checks = [];
  let diff = null;

  if (!bundle) {
    return {
      overall_valid: false,
      checks: [{ name: "lookup", passed: false, detail: "evaluation_id not found" }],
      diff: null,
    };
  }
  const { evaluation, submission, batch } = bundle;
  const rubric = JSON.parse(batch.rubric_json);
  const nCriteria = rubric.criteria.length;

  // Check 1: proof validity — several sub-steps, all against a single fresh
  // /ezkl/verify call (also captured for check 2's poseidon_commitment below, so
  // the ML-Worker's /ezkl/verify only gets called once per /verify request).
  // (a) structural: does the proof file itself verify against the vk?
  // (b) binding: does the DB's stored proof_public_inputs_json match what's
  //     actually embedded in that proof file (pretty_public_inputs)?
  // (c) final_score: is the denormalized final_score column still consistent
  //     with the proof's own rescaled_outputs[1] (the final_score output)?
  // (d) per_criterion_scores: is the denormalized per_criterion_scores_json
  //     column still consistent with the proof's own rescaled_outputs[0]
  //     (per_criterion_scores output, padded to MAX_CRITERIA=10 -- compare only
  //     the first nCriteria entries, the rest are padding)?
  let check1Passed = false;
  let check1Detail = null;
  let ezklVerifyResult = null;
  try {
    ezklVerifyResult = await ml.ezklVerify(evaluation.proof_path);
    if (!ezklVerifyResult.verified) {
      check1Detail = ezklVerifyResult.error || "proof failed structural verification";
    } else {
      const storedOutputs = JSON.stringify(JSON.parse(evaluation.proof_public_inputs_json || "null"));
      const embeddedOutputs = JSON.stringify({
        rescaled_outputs: ezklVerifyResult.pretty_public_inputs.rescaled_outputs,
      });
      if (storedOutputs !== embeddedOutputs) {
        check1Detail = "stored proof_public_inputs_json does not match values embedded in the proof file";
        diff = diff || {};
        diff.proof_public_inputs = { expected: embeddedOutputs, found: storedOutputs };
      } else {
        const parsedInputs = JSON.parse(evaluation.proof_public_inputs_json);
        const rescaled = parsedInputs.rescaled_outputs;
        const derivedFinalScore = parseFloat(rescaled[1][0]);
        const derivedScores = rescaled[0].slice(0, nCriteria).map(Number);
        const storedScores = JSON.parse(evaluation.per_criterion_scores_json || "[]").map(Number);

        if (derivedFinalScore !== evaluation.final_score) {
          check1Detail = "stored final_score does not match the value derived from proof_public_inputs_json";
          diff = diff || {};
          diff.final_score = { expected: String(derivedFinalScore), found: String(evaluation.final_score) };
        } else if (JSON.stringify(derivedScores) !== JSON.stringify(storedScores)) {
          check1Detail = "stored per_criterion_scores_json does not match the scores derived from proof_public_inputs_json";
          diff = diff || {};
          diff.per_criterion_scores = { expected: JSON.stringify(derivedScores), found: JSON.stringify(storedScores) };
        } else {
          check1Passed = true;
        }
      }
    }
  } catch (e) {
    check1Detail = e.message;
  }
  checks.push({ name: "proof_validity", passed: check1Passed, detail: check1Passed ? null : check1Detail });

  // Check 2: poseidon_commitment. R/A/negation_flags/max_marks and criterion_weights
  // are private EZKL inputs; only their Poseidon commitment is public. This single
  // check covers:
  //   - proving-step fabrication (tamper vector 6): stored Stage 0 values or
  //     criterion_weights no longer match what was actually proven.
  //   - wrong-batch re-association (tamper vector 9): this evaluation's proof was
  //     generated against a different batch's criterion_weights than the one it's
  //     currently linked to.
  let poseidonCheckPassed = false;
  let poseidonCheckDetail = null;
  if (ezklVerifyResult?.verified) {
    const embeddedCommitment = ezklVerifyResult?.embedded_poseidon_commitment;
    if (!embeddedCommitment) {
      poseidonCheckDetail = "proof file has no embedded Poseidon commitment (unexpected)";
    } else if (embeddedCommitment !== evaluation.poseidon_commitment) {
      poseidonCheckDetail = "stored poseidon_commitment does not match the commitment embedded in the proof file";
      diff = diff || {};
      diff.poseidon_commitment = { expected: embeddedCommitment, found: evaluation.poseidon_commitment };
    } else {
      try {
        const stage0 = JSON.parse(evaluation.criterion_scores_json);
        const currentWeights = currentCriterionWeights(batch);

        const recomputed = await ml.ezklRecomputeCommitment(stage0, currentWeights, `verify_${evaluationId}`);
        if (recomputed.poseidon_commitment !== embeddedCommitment) {
          poseidonCheckDetail =
            "the proven commitment doesn't match what's expected from the currently-stored Stage 0 values and this batch's rubric weights — possible score tampering or wrong-batch re-association";
          diff = diff || {};
          diff.poseidon_commitment = { expected: embeddedCommitment, found: recomputed.poseidon_commitment };
        } else {
          poseidonCheckPassed = true;
        }
      } catch (e) {
        poseidonCheckDetail = `could not recompute commitment: ${e.message}`;
      }
    }
  } else {
    poseidonCheckDetail = "skipped — proof did not structurally verify (see proof_validity)";
  }
  checks.push({ name: "poseidon_commitment", passed: poseidonCheckPassed, detail: poseidonCheckPassed ? null : poseidonCheckDetail });

  // Check 3: witness_hash — re-hash the exact stored Stage 0 (criterion_scores_json) string bytes.
  const recomputedWitnessHash = sha256Hex(evaluation.criterion_scores_json);
  const check2Passed = recomputedWitnessHash === evaluation.witness_hash;
  if (!check2Passed) {
    diff = diff || {};
    diff.witness_hash = { expected: evaluation.witness_hash, found: recomputedWitnessHash };
  }
  checks.push({
    name: "witness_hash",
    passed: check2Passed,
    detail: check2Passed ? null : "hash(stored Stage 0 values) does not match the witness_hash committed before proving",
  });

  // Check 4: model/rubric commitment — compare batch's certified hashes against
  // what's currently reported live by the ML-Worker (drift-from-certified detection),
  // plus rubric_hash self-consistency (catches rubric_json edited in place after certification).
  let check3Passed = false;
  let check3Detail = null;
  try {
    const z = await ml.rcajxModelHash();
    const mismatches = [];
    if (z.zone1_model_hash !== batch.zone1_model_hash) mismatches.push("zone1_model_hash");
    if (z.zone2_model_hash !== batch.zone2_model_hash) mismatches.push("zone2_model_hash");
    const recomputedRubricHash = sha256Hex(batch.rubric_json);
    if (recomputedRubricHash !== batch.rubric_hash) {
      mismatches.push("rubric_hash");
      diff = diff || {};
      diff.rubric_hash = { expected: batch.rubric_hash, found: recomputedRubricHash };
    }
    if (mismatches.length > 0) {
      check3Detail = `drifted from certification: ${mismatches.join(", ")}`;
      diff = diff || {};
      if (mismatches.includes("zone1_model_hash")) {
        diff.zone1_model_hash = { expected: batch.zone1_model_hash, found: z.zone1_model_hash };
      }
      if (mismatches.includes("zone2_model_hash")) {
        diff.zone2_model_hash = { expected: batch.zone2_model_hash, found: z.zone2_model_hash };
      }
    } else {
      check3Passed = true;
    }
  } catch (e) {
    check3Detail = `could not reach ML-Worker to confirm model commitment: ${e.message}`;
  }
  checks.push({ name: "model_rubric_commitment", passed: check3Passed, detail: check3Detail });

  // Check 5: audit chain integrity, up to this entry.
  const integrity = auditChain.verifyChainIntegrity();
  checks.push({
    name: "audit_chain_integrity",
    passed: integrity.intact,
    detail: integrity.intact ? null : `chain broken at entry_id=${integrity.brokenAtEntryId}`,
  });

  const overall_valid = checks.every((c) => c.passed);
  return { overall_valid, checks, diff: overall_valid ? null : diff };
}

// Scorecard-specific pre-checks, run before the check chain.
async function verifyScorecard(evaluationId, scorecardToken, uploadedPdfBuffer) {
  const evaluation = db.prepare("SELECT * FROM evaluations WHERE evaluation_id = ?").get(evaluationId);
  if (!evaluation) {
    return {
      overall_valid: false,
      checks: [{ name: "lookup", passed: false, detail: "evaluation_id not found" }],
      diff: null,
    };
  }

  const uploadedHash = sha256Hex(uploadedPdfBuffer);
  const scorecardHashPassed = uploadedHash === evaluation.scorecard_hash;
  const checks = [
    {
      name: "scorecard_hash",
      passed: scorecardHashPassed,
      detail: scorecardHashPassed ? null : "this scorecard has been modified since issuance",
    },
  ];
  if (!scorecardHashPassed) {
    return {
      overall_valid: false,
      checks,
      diff: { scorecard_hash: { expected: evaluation.scorecard_hash, found: uploadedHash } },
    };
  }

  const tokenPassed = scorecardToken === evaluation.scorecard_token;
  checks.push({
    name: "scorecard_token",
    passed: tokenPassed,
    detail: tokenPassed ? null : "scorecard token does not match this evaluation record",
  });
  if (!tokenPassed) {
    return { overall_valid: false, checks, diff: null };
  }

  const chainResult = await runVerification(evaluationId);
  return {
    overall_valid: chainResult.overall_valid,
    checks: [...checks, ...chainResult.checks],
    diff: chainResult.diff,
  };
}

module.exports = { runVerification, verifyScorecard, getEvaluationBundle };
