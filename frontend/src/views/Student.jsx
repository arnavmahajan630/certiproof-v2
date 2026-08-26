import { useState } from "react";
import { api } from "../api/client.js";

function formatWhen(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

function scoreBand(score) {
  if (score >= 70) return "";
  if (score >= 40) return "mid";
  return "low";
}

function parseJson(json, fallback) {
  if (!json) return fallback;
  try {
    return typeof json === "string" ? JSON.parse(json) : json;
  } catch {
    return fallback;
  }
}

export default function Student() {
  const [studentId, setStudentId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lookedUpId, setLookedUpId] = useState("");
  const [evaluations, setEvaluations] = useState(null);
  const [selectedId, setSelectedId] = useState(null);

  const lookup = async (e) => {
    e.preventDefault();
    const id = studentId.trim();
    if (!id) return;

    setLoading(true);
    setError(null);
    setEvaluations(null);
    setSelectedId(null);
    setLookedUpId(id);

    try {
      const data = await api.lookupStudentEvaluations(id);
      const rows = data.evaluations || [];
      setEvaluations(rows);
      const firstWithEval = rows.find((row) => row.evaluation_id) || rows[0];
      setSelectedId(firstWithEval?.submission_id || null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setEvaluations(null);
    setSelectedId(null);
    setError(null);
    setLookedUpId("");
  };

  const selected = evaluations?.find((row) => row.submission_id === selectedId) || null;
  const scores = parseJson(selected?.per_criterion_scores_json, []);
  const explanations = parseJson(selected?.explanations_json, []);

  return (
    <div className="student-view">
      <section className="panel">
        <h2>Your results</h2>
        <p className="muted">
          Enter your student ID to see your evaluation score, submission time, and when it was graded.
        </p>

        <form className="row" onSubmit={lookup}>
          <label style={{ flex: 1, minWidth: 220 }}>
            Student ID
            <input
              type="text"
              value={studentId}
              onChange={(e) => setStudentId(e.target.value)}
              placeholder="e.g. AD_123_L1"
              disabled={loading}
              autoComplete="username"
            />
          </label>
          <button className="primary" type="submit" disabled={loading || !studentId.trim()}>
            {loading ? "Looking up..." : "Look up"}
          </button>
        </form>

        {error && <div className="status fail">✗ {error}</div>}
      </section>

      {evaluations && (
        <section className="panel">
          {evaluations.length === 0 ? (
            <div className="empty-state">No evaluations found for {lookedUpId}.</div>
          ) : (
            <>
              {evaluations.length > 1 && (
                <table className="eval-table">
                  <thead>
                    <tr>
                      <th>Score</th>
                      <th>Submitted</th>
                      <th>Evaluated</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evaluations.map((row) => (
                      <tr
                        key={row.submission_id}
                        className={row.submission_id === selectedId ? "selected" : ""}
                        onClick={() => setSelectedId(row.submission_id)}
                      >
                        <td>
                          {row.evaluation_id ? (
                            <span className={`score-pill ${scoreBand(row.final_score || 0)}`}>
                              {Math.round(row.final_score || 0)}/100
                            </span>
                          ) : (
                            <span className="muted">Pending</span>
                          )}
                        </td>
                        <td className="muted">{formatWhen(row.submitted_at)}</td>
                        <td className="muted">{formatWhen(row.evaluated_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}

              {selected && (
                <div className="student-detail">
                  {selected.evaluation_id ? (
                    <div className="complete-banner">
                      <div className="label">Evaluation result</div>
                      <div className="score-hero" style={{ justifyContent: "center" }}>
                        <span className="value">{Math.round(selected.final_score || 0)}</span>
                        <span className="of">/ 100</span>
                      </div>
                    </div>
                  ) : (
                    <div className="status pending">Submitted — evaluation not ready yet</div>
                  )}

                  <dl>
                    <dt>Student ID</dt>
                    <dd>{selected.student_id}</dd>
                    <dt>Submitted</dt>
                    <dd>{formatWhen(selected.submitted_at)}</dd>
                    <dt>Evaluated</dt>
                    <dd>{formatWhen(selected.evaluated_at)}</dd>
                    {selected.scorecard_generated_at && (
                      <>
                        <dt>Scorecard issued</dt>
                        <dd>{formatWhen(selected.scorecard_generated_at)}</dd>
                      </>
                    )}
                    <dt>Status</dt>
                    <dd>{selected.proof_status || "pending"}</dd>
                  </dl>

                  {explanations.length > 0 && (
                    <>
                      <h3 style={{ fontSize: "0.85rem", margin: "16px 0 8px" }}>Per-criterion scores & reasoning</h3>
                      <div className="criteria-list">
                        {explanations.map((exp, i) => (
                          <div className="criteria-row explained" key={exp.criterion_id || i} style={{ flexDirection: "column", alignItems: "stretch", gap: 4 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                              <span className="name">{exp.criterion_text}</span>
                              <span>
                                <strong>{Number(scores[i] ?? exp.score).toFixed(1)}</strong> / {exp.max_marks}
                                {" "}
                                <span className={`badge ${exp.confidence === "review_recommended" ? "warn" : "ok"}`}>
                                  {exp.confidence === "review_recommended" ? "Flagged for review" : "High confidence"}
                                </span>
                              </span>
                            </div>
                            <p className="muted" style={{ margin: 0, fontSize: "0.85rem" }}>{exp.reason_text}</p>
                          </div>
                        ))}
                      </div>
                    </>
                  )}

                  {selected.evaluation_id && (
                    <div style={{ marginTop: 16 }}>
                      <a href={api.scorecardUrl(selected.evaluation_id)} target="_blank" rel="noreferrer">
                        <button type="button" className="primary">
                          Download scorecard PDF
                        </button>
                      </a>
                    </div>
                  )}
                </div>
              )}

              <button onClick={reset} style={{ marginTop: 16 }}>
                Look up another ID
              </button>
            </>
          )}
        </section>
      )}
    </div>
  );
}
