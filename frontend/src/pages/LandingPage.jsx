const ROLES = [
  {
    id: "student",
    icon: "🧑‍🎓",
    title: "Student",
    description: "Enter your student ID to see your score, evaluation time, and scorecard",
  },
  {
    id: "teacher",
    icon: "👩‍🏫",
    title: "Teacher",
    description: "Certify a rubric, grade submissions, review evaluations",
  },
  {
    id: "auditor",
    icon: "🔍",
    title: "Auditor",
    description: "Independently verify any evaluation's proof and audit trail",
  },
];

export default function LandingPage({ onSelectRole }) {
  return (
    <div className="landing">
      <h1>CertiProof</h1>
      <p className="subtitle">
        Proof that an AI-graded exam score came from the certified model, on the exact answer, using the
        exact rubric — checkable by anyone, from public data alone, without exposing the answer or the
        model itself.
      </p>

      <div className="role-grid">
        {ROLES.map((role) => (
          <div key={role.id} className="role-card" onClick={() => onSelectRole(role.id)}>
            <div className="role-icon">{role.icon}</div>
            <h3>{role.title}</h3>
            <p>{role.description}</p>
          </div>
        ))}
      </div>

      <p className="landing-footnote">
        A real zero-knowledge proof (EZKL/Halo2) binds the aggregation math; an independent, timestamped
        commitment binds the score's provenance — the one gap a proof alone can't close. Tamper with the
        model, score, proof, answer, or audit record and one of the two catches it. Nothing here is
        simulated. Per-criterion scores never appear in the public proof — only a cryptographic commitment
        to them does.
      </p>
    </div>
  );
}
