import { useEffect, useState } from "react";
import { api } from "../api/client.js";

export default function Auditor({ initialEvaluationId }) {
  const [evaluationId, setEvaluationId] = useState(initialEvaluationId || "");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [chain, setChain] = useState(null);
  const [chainLoading, setChainLoading] = useState(false);

  const runVerify = async (id) => {
    const targetId = (id ?? evaluationId).trim();
    if (!targetId) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await api.verify({ evaluation_id: targetId });
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (initialEvaluationId) {
      setEvaluationId(initialEvaluationId);
      runVerify(initialEvaluationId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialEvaluationId]);

  const loadChain = async () => {
    setChainLoading(true);
    try {
      setChain(await api.auditChain());
    } catch (err) {
      setError(err.message);
    } finally {
      setChainLoading(false);
    }
  };

  const exportReport = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `verification-report-${evaluationId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const CHECK_LABEL = {
    proof_validity: "ZK Proof",
    poseidon_commitment: "Poseidon Commitment",
    witness_hash: "Witness Hash",
    model_rubric_commitment: "Model / Rubric Commitment",
    audit_chain_integrity: "Audit Chain Integrity",
    scorecard_hash: "Scorecard Hash",
    scorecard_token: "Scorecard Token",
    input_hash: "Answer Input Hash",
  };

  const entries = chain?.entries || (Array.isArray(chain) ? chain : null);

  return (
    <div>
      <h2 style={{ marginBottom: 4 }}>Auditor</h2>
      <p className="muted" style={{ marginBottom: 20 }}>
        Independently verify an evaluation's cryptographic proof and audit trail using only public data —
        no trust in the institution's word required.
      </p>

      <div className="panel">
        <h2>Verify an Evaluation</h2>
        <div className="row">
          <label>
            Evaluation ID
            <input
              value={evaluationId}
              onChange={(e) => setEvaluationId(e.target.value)}
              placeholder="paste evaluation ID"
              style={{ minWidth: 280 }}
            />
          </label>
          <button className="primary" onClick={() => runVerify()} disabled={loading || !evaluationId.trim()}>
            {loading ? "Verifying..." : "Verify"}
          </button>
        </div>

        {loading && (
          <div className="status pending">
            <span className="spinner" /> Running independent cryptographic verification...
          </div>
        )}
        {error && !result && <div className="status fail">✗ {error}</div>}

        {result && (
          <div className="result-box">
            <div className={`status ${result.overall_valid ? "ok" : "fail"}`}>
              {result.overall_valid ? "✓ Evaluation Verified" : "✗ Evaluation Verification Failed"}
            </div>

            <ul className="check-list">
              {(result.checks || []).map((c) => (
                <li className={c.passed ? "ok" : "fail"} key={c.name}>
                  {c.passed ? "✓" : "✗"} {CHECK_LABEL[c.name] || c.name}
                  {c.detail && <span className="check-note">({c.detail})</span>}
                </li>
              ))}
            </ul>

            {result.confidence_summary && (
              <div style={{ marginTop: 14 }}>
                <h3 style={{ fontSize: "0.85rem", margin: "0 0 8px" }}>
                  Confidence Shortlist ({result.confidence_summary.flagged.length} of {result.confidence_summary.total_criteria} criteria flagged for review)
                </h3>
                {result.confidence_summary.flagged.length === 0 ? (
                  <p className="muted" style={{ fontSize: "0.82rem" }}>The model reported high confidence on every criterion.</p>
                ) : (
                  <div className="criteria-list">
                    {result.confidence_summary.flagged.map((exp) => (
                      <div className="criteria-row" key={exp.criterion_id} style={{ flexDirection: "column", alignItems: "stretch", gap: 4 }}>
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <span className="name">{exp.criterion_text}</span>
                          <span className="badge warn">{exp.score} / {exp.max_marks} — Review</span>
                        </div>
                        <p className="muted" style={{ margin: 0, fontSize: "0.78rem" }}>{exp.reason_text}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {result.diff && (
              <table className="diff-table">
                <thead>
                  <tr>
                    <th>Field</th>
                    <th>Expected (committed)</th>
                    <th>Found (current)</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.diff).map(([field, v]) => (
                    <tr key={field}>
                      <td>{field}</td>
                      <td className="mono small expected">{String(v.expected)}</td>
                      <td className="mono small found">{String(v.found)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            <button onClick={exportReport} style={{ marginTop: 14 }}>
              Export Verification Report
            </button>
          </div>
        )}
      </div>

      <div className="panel">
        <h2>Audit Chain</h2>
        <p className="muted">Full hash-linked chain, viewable independently of any single evaluation.</p>
        <button onClick={loadChain} disabled={chainLoading}>
          {chainLoading ? "Loading..." : "Load Chain"}
        </button>

        {entries && (
          entries.length === 0 ? (
            <div className="empty-state">No entries yet.</div>
          ) : (
            <table className="eval-table" style={{ marginTop: 14 }}>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Event</th>
                  <th>Ref</th>
                  <th>Entry Hash</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((e) => (
                  <tr key={e.entry_id}>
                    <td className="muted">{e.entry_id}</td>
                    <td>{e.event_type}</td>
                    <td className="mono small">{e.ref_id}</td>
                    <td className="mono small">{(e.entry_hash || "").slice(0, 20)}...</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )
        )}
      </div>

      <TestsPanel CHECK_LABEL={CHECK_LABEL} />
    </div>
  );
}

function CheckList({ checks, CHECK_LABEL }) {
  return (
    <ul className="check-list">
      {(checks || []).map((c) => (
        <li className={c.passed ? "ok" : "fail"} key={c.name}>
          {c.passed ? "✓" : "✗"} {CHECK_LABEL[c.name] || c.name}
          {c.detail && <span className="check-note">({c.detail})</span>}
        </li>
      ))}
    </ul>
  );
}

function DiffTable({ diff }) {
  if (!diff) return null;
  return (
    <table className="diff-table">
      <thead>
        <tr>
          <th>Field</th>
          <th>Expected (committed)</th>
          <th>Found (current)</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(diff).map(([field, v]) => (
          <tr key={field}>
            <td>{field}</td>
            <td className="mono small expected">{String(v.expected)}</td>
            <td className="mono small found">{String(v.found)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function TestsPanel({ CHECK_LABEL }) {
  const [vectors, setVectors] = useState([]);
  const [vectorsError, setVectorsError] = useState(null);
  const [testCase, setTestCase] = useState(null);
  const [settingUp, setSettingUp] = useState(false);
  const [setupError, setSetupError] = useState(null);
  const [results, setResults] = useState({}); // { [vectorId]: { status: 'running'|'done'|'error', data | error } }
  const [runningAll, setRunningAll] = useState(false);

  useEffect(() => {
    api
      .tamperVectors()
      .then((data) => setVectors(data.vectors || []))
      .catch((err) => setVectorsError(err.message));
  }, []);

  const setup = async () => {
    setSettingUp(true);
    setSetupError(null);
    setTestCase(null);
    setResults({});
    try {
      const data = await api.tamperSetup();
      setTestCase(data);
    } catch (err) {
      setSetupError(err.message);
    } finally {
      setSettingUp(false);
    }
  };

  const runVector = async (id) => {
    if (!testCase) return;
    setResults((prev) => ({ ...prev, [id]: { status: "running" } }));
    try {
      const data = await api.tamperRun(testCase.evaluation_id, id);
      setResults((prev) => ({ ...prev, [id]: { status: "done", data } }));
      return data;
    } catch (err) {
      setResults((prev) => ({ ...prev, [id]: { status: "error", error: err.message } }));
    }
  };

  const runAll = async () => {
    if (!testCase) return;
    setRunningAll(true);
    for (const v of vectors) {
      // eslint-disable-next-line no-await-in-loop
      await runVector(v.id);
    }
    setRunningAll(false);
  };

  const grouped = vectors.reduce((acc, v) => {
    (acc[v.category] = acc[v.category] || []).push(v);
    return acc;
  }, {});

  const totalRun = Object.keys(results).length;
  const totalCaught = Object.values(results).filter((r) => r.status === "done" && r.data?.caught).length;

  return (
    <div className="panel">
      <h2>Tests — Tamper Vectors</h2>
      <p className="muted">
        Sets up a fresh, throwaway batch and evaluation, then runs each tamper vector against it — applying it,
        calling the real <code>/verify</code> chain, and automatically restoring the untampered state afterward.
        Nothing here is simulated; every result below is the live cryptographic check response.
      </p>

      {!testCase && (
        <button className="primary" onClick={setup} disabled={settingUp}>
          {settingUp ? "Setting up..." : "Set Up Fresh Test Case"}
        </button>
      )}
      {setupError && <div className="status fail">✗ {setupError}</div>}

      {testCase && (
        <>
          <div className="result-box">
            <dl>
              <dt>Batch ID</dt>
              <dd className="mono small">{testCase.batch_id}</dd>
              <dt>Evaluation ID</dt>
              <dd className="mono small">{testCase.evaluation_id}</dd>
              <dt>Baseline (untampered)</dt>
              <dd className={testCase.baseline?.overall_valid ? "status ok" : "status fail"} style={{ display: "inline-flex" }}>
                {testCase.baseline?.overall_valid ? "✓ Valid" : "✗ Invalid"}
              </dd>
            </dl>
          </div>

          <div className="row" style={{ margin: "12px 0" }}>
            <button onClick={setup} disabled={settingUp}>
              {settingUp ? "Resetting..." : "New Test Case"}
            </button>
            <button className="primary" onClick={runAll} disabled={runningAll || vectors.length === 0}>
              {runningAll ? "Running all..." : "Run All Vectors"}
            </button>
            {totalRun > 0 && (
              <span className="muted" style={{ alignSelf: "center" }}>
                {totalCaught}/{totalRun} run so far caught as expected
              </span>
            )}
          </div>

          {vectorsError && <div className="status fail">✗ {vectorsError}</div>}

          {Object.entries(grouped).map(([category, vs]) => (
            <div key={category} style={{ marginTop: 20 }}>
              <h3 style={{ fontSize: "0.85rem", margin: "0 0 8px" }}>{category}</h3>
              {vs.map((v) => {
                const r = results[v.id];
                return (
                  <div key={v.id} className="tamper-row" style={{ borderTop: "1px solid var(--border)", padding: "10px 0" }}>
                    <div className="row" style={{ alignItems: "flex-start" }}>
                      <div style={{ flex: 1, minWidth: 240 }}>
                        <strong>{v.label}</strong>
                        <p className="muted" style={{ fontSize: "0.8rem", margin: "4px 0 0" }}>
                          {v.description}
                        </p>
                      </div>
                      <button onClick={() => runVector(v.id)} disabled={r?.status === "running" || runningAll}>
                        {r?.status === "running" ? "Running..." : "Run"}
                      </button>
                    </div>

                    {r?.status === "error" && <div className="status fail">✗ {r.error}</div>}

                    {r?.status === "done" && (
                      <div className="result-box" style={{ marginTop: 8 }}>
                        <div className={`status ${r.data.caught ? "ok" : "fail"}`}>
                          {r.data.caught ? "✓ Caught — /verify correctly flagged this as invalid" : "✗ Not caught — /verify still reports valid"}
                        </div>
                        <CheckList checks={r.data.checks} CHECK_LABEL={CHECK_LABEL} />
                        <DiffTable diff={r.data.diff} />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </>
      )}
    </div>
  );
}
