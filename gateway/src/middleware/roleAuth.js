// Stub, header-based role access — explicitly not real auth (doc 00 Section 6 scope guardrails).
function requireRole(role) {
  return (req, res, next) => {
    const suppliedRole = req.header("X-Role");
    if (suppliedRole !== role) {
      return res.status(403).json({ error: "forbidden", detail: `expected X-Role: ${role}` });
    }
    req.actorId = req.header("X-Actor-Id") || "unknown";
    next();
  };
}

module.exports = { requireRole };
