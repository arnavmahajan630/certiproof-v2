CREATE TABLE IF NOT EXISTS exam_batches (
  batch_id TEXT PRIMARY KEY,
  rubric_json TEXT NOT NULL,          -- full rubric spec: criterion_id + criterion_text + max_marks + weight, per criterion
  rubric_hash TEXT NOT NULL,
  zone1_model_hash TEXT NOT NULL,     -- RCAJ-X migration: hash of the fine-tuned BGE-small encoder's weight bytes (was bge-reranker-base identity)
  zone2_model_hash TEXT NOT NULL,     -- RCAJ-X migration: hash of the RCAJ_X checkpoint (attention + scoring head) weights+config (was the Zone2 MLP + circuit)
  certified_at TEXT NOT NULL,
  certified_by TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS submissions (
  submission_id TEXT PRIMARY KEY,
  batch_id TEXT NOT NULL REFERENCES exam_batches(batch_id),
  student_id TEXT NOT NULL,
  answer_text TEXT NOT NULL,
  input_hash TEXT NOT NULL,           -- hash of answer_text, returned as receipt at submission
  input_source TEXT NOT NULL DEFAULT 'student_typed',  -- 'student_typed' | 'teacher_ocr'
  answer_sheet_hash TEXT,             -- sha256 of uploaded image bytes, teacher_ocr only; audit-trail only, not proof-bound
  submitted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
  evaluation_id TEXT PRIMARY KEY,
  submission_id TEXT NOT NULL REFERENCES submissions(submission_id),
  -- RCAJ-X migration: this column now holds Stage 0's (/rcajx/embed) raw output --
  -- R/A embeddings + negation_flags + max_marks -- exact JSON string as received
  -- (see hashing.js). It previously held Zone 1's {criterion_scores, rubric_weights}.
  -- Renamed in spirit, kept in name: still "the JSON blob witness_hash is computed
  -- over, committed before the scoring/proving step runs".
  criterion_scores_json TEXT NOT NULL,
  witness_hash TEXT NOT NULL,              -- sha256(criterion_scores_json) i.e. sha256(Stage 0 response), committed BEFORE Stage 1-3/proving
  per_criterion_scores_json TEXT,          -- NEW: authoritative per-criterion scores, taken verbatim from the proof's own public output (rescaled_outputs[0]) -- never independently recomputed
  explanations_json TEXT,                  -- NEW: evidence chunks + reason text + confidence per criterion, from /rcajx/score using the SAME Stage 0 inputs the proof was generated from
  proof_public_inputs_json TEXT,           -- EZKL's own public outputs, verbatim from /ezkl/prove (rescaled_outputs: [per_criterion_scores, final_score, attn_weights, spread])
  poseidon_commitment TEXT,                -- EZKL's own commitment of the (private) Stage 1-3 input (R, A, negation_flags, max_marks, criterion_weights), verbatim from /ezkl/prove
  final_score REAL,                        -- from /ezkl/prove's proof_public_inputs, never recomputed
  proof_path TEXT,
  proof_status TEXT,
  scorecard_hash TEXT,
  scorecard_token TEXT,
  scorecard_path TEXT,
  scorecard_generated_at TEXT,
  evaluated_at TEXT
);

CREATE TABLE IF NOT EXISTS overrides (
  override_id TEXT PRIMARY KEY,
  evaluation_id TEXT NOT NULL REFERENCES evaluations(evaluation_id),
  teacher_id TEXT NOT NULL,
  original_score REAL NOT NULL,
  overridden_score REAL NOT NULL,
  reason TEXT,
  overridden_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_chain (
  entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type TEXT NOT NULL,           -- 'certify' | 'submit' | 'ocr_transcribe' | 'witness_commit' | 'proof_generated' | 'scorecard_generated' | 'override' | 'verify'
  ref_id TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  prev_entry_hash TEXT,
  entry_hash TEXT NOT NULL,
  created_at TEXT NOT NULL
);
