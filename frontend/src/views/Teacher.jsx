import { useEffect, useRef, useState } from "react";
import { api } from "../api/client.js";

let nextCriterionSeq = 1;
const emptyCriterion = () => ({
  criterion_id: `c${nextCriterionSeq++}`,
  criterion_text: "",
  max_marks: 2,
  weight: 1,
  reference_answer: "",
});

const OCR_STEPS = [
  "Answer sheet uploaded",
  "Gemini transcribing image",
  "Stage 0: embedding criteria + answer",
  "Committing witness hash",
  "Stages 1-3: attention scoring + EZKL proving",
];

function scoreBand(score) {
  if (score >= 70) return "";
  if (score >= 40) return "mid";
  return "low";
}

export default function Teacher() {
  const [teacherId, setTeacherId] = useState("ms_kamble");

  return (
    <div>
      <h2 style={{ marginBottom: 4 }}>Teacher</h2>
      <p className="muted" style={{ marginBottom: 20 }}>
        Certify a rubric, grade submissions — typed or scanned — and review evaluations.
      </p>

      <div className="row" style={{ marginBottom: 0 }}>
        <label style={{ minWidth: 220 }}>
          Certified / acting as (teacher id)
          <input type="text" value={teacherId} onChange={(e) => setTeacherId(e.target.value)} />
        </label>
      </div>

      <CertifyPanel teacherId={teacherId} />
      <UploadSheetPanel teacherId={teacherId} />
      <BatchReviewPanel teacherId={teacherId} />
    </div>
  );
}

function CertifyPanel({ teacherId }) {
  const [criteria, setCriteria] = useState([emptyCriterion()]);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const updateCriterion = (i, field, value) => {
    setCriteria((c) => c.map((row, idx) => (idx === i ? { ...row, [field]: value } : row)));
  };

  const certify = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const cleanCriteria = criteria
        .filter((c) => c.criterion_text.trim())
        .map((c) => ({
          criterion_id: c.criterion_id,
          criterion_text: c.criterion_text,
          max_marks: Number(c.max_marks),
          weight: Number(c.weight),
        }));
      const res = await api.certifyBatch({ rubric: { criteria: cleanCriteria } }, teacherId);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel">
      <h2>Certify a Batch</h2>
      <p className="muted">
        The rubric and both zones' model identity are hashed and locked here — before any submission can exist.
      </p>

      <form onSubmit={certify}>
        {criteria.map((c, i) => (
          <div key={i} style={{ borderTop: i > 0 ? "1px solid var(--border)" : "none", paddingTop: i > 0 ? 10 : 0, marginTop: i > 0 ? 10 : 0 }}>
            <div className="row">
              <label style={{ flex: 2, minWidth: 220 }}>
                Criterion {i + 1}
                <input
                  type="text"
                  value={c.criterion_text}
                  onChange={(e) => updateCriterion(i, "criterion_text", e.target.value)}
                  placeholder="e.g. Mentions oxygen as a byproduct"
                />
              </label>
              <label style={{ width: 90 }}>
                Max marks
                <input
                  type="number"
                  step="1"
                  min="1"
                  value={c.max_marks}
                  onChange={(e) => updateCriterion(i, "max_marks", parseFloat(e.target.value))}
                />
              </label>
              <label style={{ width: 90 }}>
                Weight
                <input
                  type="number"
                  step="0.1"
                  value={c.weight}
                  onChange={(e) => updateCriterion(i, "weight", parseFloat(e.target.value))}
                />
              </label>
              <button type="button" onClick={() => setCriteria((cs) => cs.filter((_, idx) => idx !== i))}>
                Remove
              </button>
            </div>
          </div>
        ))}
        <div className="row">
          <button type="button" onClick={() => setCriteria((c) => [...c, emptyCriterion()])}>
            + Add criterion
          </button>
          <button className="primary" type="submit" disabled={busy}>
            {busy ? "Certifying..." : "Certify Batch"}
          </button>
        </div>
      </form>

      {error && <div className="status fail">✗ {error}</div>}
      {result && (
        <div className="result-box">
          <div className="status ok">✓ Batch certified</div>
          <dl>
            <dt>Batch ID</dt>
            <dd className="mono small">{result.batch_id}</dd>
            <dt>Rubric Hash</dt>
            <dd className="mono small">{result.rubric_hash}</dd>
            <dt>Zone 1 Model Hash</dt>
            <dd className="mono small">{result.zone1_model_hash}</dd>
            <dt>Zone 2 Model Hash</dt>
            <dd className="mono small">{result.zone2_model_hash}</dd>
          </dl>
        </div>
      )}
    </section>
  );
}

function UploadSheetPanel({ teacherId }) {
  const [batchId, setBatchId] = useState("");
  const [studentId, setStudentId] = useState("");
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [stepIndex, setStepIndex] = useState(-1);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const timerRef = useRef(null);

  useEffect(() => () => clearInterval(timerRef.current), []);

  const acceptFile = (f) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Only image files are supported.");
      return;
    }
    setFile(f);
    setResult(null);
    setError(null);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (processing) return;
    acceptFile(e.dataTransfer.files?.[0] || null);
  };

  const runUpload = async () => {
    if (!file || !batchId || !studentId) return;
    setProcessing(true);
    setError(null);
    setResult(null);
    setStepIndex(0);

    timerRef.current = setInterval(() => {
      setStepIndex((i) => (i < OCR_STEPS.length - 1 ? i + 1 : i));
    }, 900);

    try {
      const fd = new FormData();
      fd.append("student_id", studentId);
      fd.append("file", file);
      const data = await api.uploadAnswerSheet(batchId, fd, teacherId);
      clearInterval(timerRef.current);
      setStepIndex(OCR_STEPS.length);
      setResult(data);
    } catch (err) {
      clearInterval(timerRef.current);
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  const reset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setStepIndex(-1);
  };

  return (
    <section className="panel">
      <h2>Upload Answer Sheet</h2>
      <p className="muted">
        Extracts text from a photographed answer sheet via the Gemini API, then runs it through the same
        real scoring and proving pipeline as a typed submission.
      </p>
      <span className="badge">OCR provenance not cryptographically verified — trusted like a typed answer</span>

      <div className="row">
        <label style={{ flex: 1, minWidth: 200 }}>
          Batch ID
          <input type="text" value={batchId} onChange={(e) => setBatchId(e.target.value)} disabled={processing} />
        </label>
        <label style={{ flex: 1, minWidth: 200 }}>
          Student ID
          <input type="text" value={studentId} onChange={(e) => setStudentId(e.target.value)} disabled={processing} />
        </label>
      </div>

      {!result && !file && (
        <div
          className={`dropzone${dragActive ? " active" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={onDrop}
        >
          <p className="dropzone-hint">Drag &amp; drop the answer sheet photo</p>
          <p className="dropzone-or">or</p>
          <label className="file-picker-btn" htmlFor="sheet-input">
            Choose Image
          </label>
          <input
            id="sheet-input"
            className="file-input-hidden"
            type="file"
            accept="image/*"
            onChange={(e) => acceptFile(e.target.files?.[0] || null)}
            disabled={processing}
          />
          <p className="dropzone-note">JPG or PNG</p>
        </div>
      )}

      {!result && file && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div className="file-chip">
            ✓ <span className="name">{file.name}</span>
          </div>
          {!processing && (
            <button onClick={() => setFile(null)} style={{ alignSelf: "flex-start", marginTop: 4 }}>
              Choose a different file
            </button>
          )}
          <button
            className="primary"
            onClick={runUpload}
            disabled={processing || !batchId || !studentId}
            style={{ alignSelf: "flex-start", marginTop: 8 }}
          >
            {processing ? "Evaluating..." : "Upload & Grade"}
          </button>
        </div>
      )}

      {(processing || (stepIndex >= 0 && !result && !error)) && (
        <div className="step-list">
          {OCR_STEPS.map((label, i) => {
            const isDone = i < stepIndex || (i === stepIndex && !processing);
            const isActive = processing && i === stepIndex;
            return (
              <div className={`step-row ${isDone ? "done" : isActive ? "active" : ""}`} key={label}>
                <span className="marker">{isDone ? "✓" : isActive ? "●" : "○"}</span>
                {label}
              </div>
            );
          })}
        </div>
      )}

      {error && <div className="status fail">✗ {error}</div>}

      {result && (
        <div className="result-box">
          <div className="complete-banner">
            <div className="label">✓ Evaluation Complete</div>
            <div className="score-hero" style={{ justifyContent: "center" }}>
              <span className="value">{result.evaluation.final_score?.toFixed?.(1) ?? result.evaluation.final_score}</span>
              <span className="of">/ 100</span>
            </div>
            <div className="eval-id-hero">
              <div className="label">Evaluation ID</div>
              <div className="mono small muted">{result.evaluation.evaluation_id}</div>
            </div>
          </div>

          <details className="tech-details" open>
            <summary>Transcribed text</summary>
            <p className="muted" style={{ marginTop: 8 }}>{result.transcribed_text}</p>
          </details>

          <button onClick={reset} style={{ marginTop: 14 }}>
            Upload Another Sheet
          </button>
        </div>
      )}
    </section>
  );
}

function BatchReviewPanel({ teacherId }) {
  const [batchId, setBatchId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [submissions, setSubmissions] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [proofStatus, setProofStatus] = useState({});

  const load = async (e) => {
    e?.preventDefault();
    if (!batchId) return;
    setLoading(true);
    setError(null);
    setSubmissions(null);
    setSelectedId(null);
    setProofStatus({});
    try {
      const res = await api.listBatchSubmissions(batchId);
      setSubmissions(res.submissions || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const evaluated = (submissions || []).filter((s) => s.evaluation_id);

  const selectRow = async (evaluationId) => {
    setSelectedId(evaluationId);
    if (proofStatus[evaluationId]) return;
    setProofStatus((prev) => ({ ...prev, [evaluationId]: "checking" }));
    try {
      const res = await api.verify({ evaluation_id: evaluationId });
      setProofStatus((prev) => ({ ...prev, [evaluationId]: res }));
    } catch (err) {
      setProofStatus((prev) => ({ ...prev, [evaluationId]: { overall_valid: false, error: err.message } }));
    }
  };

  const total = evaluated.length;
  const avgScore = total > 0 ? Math.round(evaluated.reduce((s, e) => s + (e.final_score || 0), 0) / total) : 0;
  const students = new Set(evaluated.map((e) => e.student_id)).size;

  const selected = evaluated.find((e) => e.evaluation_id === selectedId);
  const selectedProof = selectedId ? proofStatus[selectedId] : null;
  const selectedScores = selected?.per_criterion_scores_json ? JSON.parse(selected.per_criterion_scores_json) : null;
  const selectedExplanations = selected?.explanations_json ? JSON.parse(selected.explanations_json) : null;

  return (
    <section className="panel">
      <h2>Review Evaluations</h2>
      <p className="muted">Every score, criterion breakdown, and proof status for a certified batch — real per-criterion data, never a sample or placeholder.</p>
      <form className="row" onSubmit={load}>
        <label style={{ flex: 1, minWidth: 260 }}>
          Batch ID
          <input type="text" value={batchId} onChange={(e) => setBatchId(e.target.value)} placeholder="paste batch id" />
        </label>
        <button className="primary" type="submit" disabled={loading || !batchId}>
          {loading ? "Loading..." : "Load Batch"}
        </button>
      </form>

      {error && <div className="status fail">✗ {error}</div>}

      {submissions && (
        <>
          <div className="stat-grid">
            <div className="stat-card">
              <div className="stat-label">Total Evaluations</div>
              <div className="stat-value">{total}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Average Score</div>
              <div className="stat-value">{avgScore}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Students</div>
              <div className="stat-value">{students}</div>
            </div>
          </div>

          <div className="split-layout">
            <div>
              {total === 0 ? (
                <div className="empty-state">No evaluated submissions in this batch yet.</div>
              ) : (
                <table className="eval-table">
                  <thead>
                    <tr>
                      <th>Student</th>
                      <th>Source</th>
                      <th>Score</th>
                      <th>Submitted</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evaluated.map((s) => (
                      <tr
                        key={s.evaluation_id}
                        className={s.evaluation_id === selectedId ? "selected" : ""}
                        onClick={() => selectRow(s.evaluation_id)}
                      >
                        <td className="mono small">{s.student_id}</td>
                        <td className="muted">{s.input_source === "teacher_ocr" ? "OCR" : "typed"}</td>
                        <td>
                          <span className={`score-pill ${scoreBand(s.final_score || 0)}`}>
                            {Math.round(s.final_score || 0)}/100
                          </span>
                        </td>
                        <td className="muted">{new Date(s.submitted_at).toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>

            <div className="panel" style={{ margin: 0 }}>
              <h2>Evaluation Detail</h2>
              {!selected && <p className="muted">Select an evaluation from the table to view details.</p>}
              {selected && (
                <div>
                  <div className="score-hero">
                    <span className="value">{Math.round(selected.final_score || 0)}</span>
                    <span className="of">/ 100</span>
                  </div>

                  {selectedProof === "checking" && (
                    <div className="status pending">
                      <span className="spinner" /> Checking ZK proof...
                    </div>
                  )}
                  {selectedProof && selectedProof !== "checking" && (
                    <div className={`status ${selectedProof.overall_valid ? "ok" : "fail"}`}>
                      {selectedProof.overall_valid ? "✓ Evaluation Valid" : "✗ Evaluation Invalid"}
                    </div>
                  )}

                  <dl>
                    <dt>Evaluation ID</dt>
                    <dd className="mono small">{selected.evaluation_id}</dd>
                    <dt>Student</dt>
                    <dd>{selected.student_id}</dd>
                    <dt>Witness Hash</dt>
                    <dd className="mono small">{selected.witness_hash?.slice(0, 24)}...</dd>
                  </dl>

                  {selectedExplanations && (
                    <>
                      <h3 style={{ fontSize: "0.85rem", margin: "16px 0 8px" }}>Per-Criterion Scores &amp; Evidence</h3>
                      <div className="criteria-list">
                        {selectedExplanations.map((exp, i) => (
                          <div className="criteria-row" key={exp.criterion_id || i} style={{ flexDirection: "column", alignItems: "stretch", gap: 4 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                              <span className="name">{exp.criterion_text}</span>
                              <span>
                                <strong>{Number(selectedScores?.[i] ?? exp.score).toFixed(1)}</strong> / {exp.max_marks}{" "}
                                <span className={`badge ${exp.confidence === "review_recommended" ? "warn" : "ok"}`}>
                                  {exp.confidence === "review_recommended" ? "Review" : "Confident"}
                                </span>
                                {exp.negation_flag && <span className="badge warn">Negation flagged</span>}
                              </span>
                            </div>
                            <p className="muted" style={{ margin: 0, fontSize: "0.78rem" }}>{exp.reason_text}</p>
                          </div>
                        ))}
                      </div>
                      <p className="muted" style={{ fontSize: "0.76rem" }}>
                        Visible here via your authenticated batch review — the public cryptographic proof
                        never exposes these scores/evidence, only a commitment to them.
                      </p>
                    </>
                  )}

                  <OverrideForm evaluationId={selected.evaluation_id} teacherId={teacherId} currentScore={selected.final_score} />
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </section>
  );
}

function OverrideForm({ evaluationId, teacherId, currentScore }) {
  const [score, setScore] = useState(currentScore ?? "");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setScore(currentScore ?? "");
    setReason("");
    setResult(null);
    setError(null);
  }, [evaluationId, currentScore]);

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await api.overrideEvaluation(evaluationId, { overriddenScore: parseFloat(score), reason }, teacherId);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <details className="tech-details" style={{ marginTop: 16 }}>
      <summary>Override this score</summary>
      <p className="muted" style={{ margin: "8px 0" }}>
        Writes a new, separate record — the original evaluation and its proof are never modified.
      </p>
      <form onSubmit={submit}>
        <div className="row">
          <label style={{ width: 120 }}>
            New score
            <input type="number" step="0.1" value={score} onChange={(e) => setScore(e.target.value)} />
          </label>
          <label style={{ flex: 1, minWidth: 200 }}>
            Reason
            <input type="text" value={reason} onChange={(e) => setReason(e.target.value)} placeholder="why this score changed" />
          </label>
          <button className="primary" type="submit" disabled={busy}>
            {busy ? "Saving..." : "Save Override"}
          </button>
        </div>
      </form>
      {error && <div className="status fail">✗ {error}</div>}
      {result && (
        <div className="status ok">
          ✓ Override recorded: {result.original_score?.toFixed?.(1)} → {result.overridden_score?.toFixed?.(1)}
        </div>
      )}
    </details>
  );
}
