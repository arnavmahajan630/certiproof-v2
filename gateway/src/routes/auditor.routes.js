const express = require("express");
const db = require("../db/client");
const verifyService = require("../services/verifyService");
const auditChain = require("../services/auditChain");

const router = express.Router();

// GET /verify?evaluation_id=...
// Alongside the cryptographic check chain, includes a confidence shortlist —
// which criteria the model itself flagged review_recommended (Auditor
// Integration Point, see certiproof-integration-plan/03) — so an auditor sees
// not just "the hashes match" but "here's what's worth a second look".
router.get("/verify", async (req, res) => {
  const { evaluation_id } = req.query;
  if (!evaluation_id) {
    return res.status(422).json({ error: "missing_evaluation_id" });
  }
  const result = await verifyService.runVerification(evaluation_id);

  const bundle = verifyService.getEvaluationBundle(evaluation_id);
  if (bundle?.evaluation?.explanations_json) {
    try {
      const explanations = JSON.parse(bundle.evaluation.explanations_json);
      result.confidence_summary = {
        total_criteria: explanations.length,
        flagged: explanations.filter((e) => e.confidence === "review_recommended"),
      };
    } catch {
      // explanations_json malformed/absent -- omit confidence_summary, don't fail the whole verify response
    }
  }

  res.json(result);
});

// GET /audit-chain
router.get("/audit-chain", (req, res) => {
  const entries = db.prepare("SELECT * FROM audit_chain ORDER BY entry_id ASC").all();
  const integrity = auditChain.verifyChainIntegrity();
  res.json({ entries, integrity });
});

module.exports = router;
