const BASE = "/api";

function apiErrorMessage(body, status) {
  if (body && typeof body === "object" && !(body instanceof Blob)) {
    const nested = body.detail;
    if (typeof nested === "string" && nested.trim()) return nested;
    if (nested && typeof nested === "object") {
      return nested.detail || nested.error || JSON.stringify(nested);
    }
    if (typeof body.error === "string") {
      return body.detail ? `${body.error}: ${body.detail}` : body.error;
    }
  }
  return `request failed: ${status}`;
}

async function req(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      ...(options.body && !(options.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...options.headers,
    },
  });
  const contentType = res.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await res.json() : await res.blob();
  if (!res.ok) {
    const err = new Error(apiErrorMessage(body, res.status));
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
}

function withRole(role, actorId, headers = {}) {
  return { ...headers, "X-Role": role, ...(actorId ? { "X-Actor-Id": actorId } : {}) };
}

export const api = {
  health: () => req("/health"),

  // Teacher
  certifyBatch: (payload, actorId) =>
    req("/teacher/batches", {
      method: "POST",
      headers: withRole("teacher", actorId),
      body: JSON.stringify({ ...payload, certifiedBy: actorId }),
    }),
  listBatchSubmissions: (batchId) =>
    req(`/teacher/batches/${batchId}/submissions`, { headers: withRole("teacher") }),
  uploadAnswerSheet: (batchId, formData, actorId) =>
    req(`/teacher/submissions/${batchId}/upload-answer-sheet`, {
      method: "POST",
      headers: withRole("teacher", actorId),
      body: formData,
    }),
  overrideEvaluation: (evaluationId, payload, actorId) =>
    req(`/teacher/evaluations/${evaluationId}/override`, {
      method: "POST",
      headers: withRole("teacher", actorId),
      body: JSON.stringify(payload),
    }),

  // Student
  submitAnswer: (payload) =>
    req("/student/submissions", {
      method: "POST",
      headers: withRole("student"),
      body: JSON.stringify(payload),
    }),
  getSubmission: (submissionId) =>
    req(`/student/submissions/${submissionId}`, { headers: withRole("student") }),
  lookupStudentEvaluations: (studentId) =>
    req(`/student/evaluations?${new URLSearchParams({ student_id: studentId })}`, {
      headers: withRole("student"),
    }),
  scorecardUrl: (evaluationId) => `${BASE}/student/evaluations/${evaluationId}/scorecard`,
  verifyScorecard: (formData) =>
    req("/student/verify-scorecard", {
      method: "POST",
      headers: withRole("student"),
      body: formData,
    }),

  // Auditor
  verify: (params) => req(`/verify?${new URLSearchParams(params)}`, { headers: withRole("auditor") }),
  auditChain: () => req("/audit-chain", { headers: withRole("auditor") }),

  // Auditor — Tests panel (tamper vectors)
  tamperVectors: () => req("/dev/tamper/vectors"),
  tamperSetup: () => req("/dev/tamper/setup", { method: "POST" }),
  tamperRun: (evaluationId, vector) =>
    req("/dev/tamper/run", {
      method: "POST",
      body: JSON.stringify({ evaluation_id: evaluationId, vector }),
    }),
};
