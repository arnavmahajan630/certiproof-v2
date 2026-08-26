# CertiProof

**CertiProof proves that a specific AI-generated exam score was produced by a specific certified model, on a specific candidate's exact answer, using a specific rubric — and lets anyone check that claim from public data alone, without seeing the raw answer text or the model's internals.**

Its ML pipeline runs on **RCAJ-X**: a fine-tuned BGE-small bi-encoder (Stage 0, off-circuit) feeding a multi-head cross-attention + bounded scoring head (Stages 1-3, proven via a zero-knowledge SNARK), backed by a second, independent commitment that closes a gap the ZK proof alone can't: a proof can only attest to consistency with whatever input it's given, not to that input's *provenance*. CertiProof binds both.

## What's actually proven — and what isn't

| Claim | Mechanism | Covers |
|---|---|---|
| "This final score is exactly what you get from running the certified model's evidence-weighing and aggregation on this exact answer and rubric" | EZKL zero-knowledge proof — `per_criterion_scores`/`final_score` public, the Stage 0 embeddings + rubric weights private with a Poseidon `poseidon_commitment` exposed as a public output instead | The math *and* the evidence-weighing. Previously only the aggregation arithmetic was proven; now the attention computation that decides which part of the answer supports which criterion is proven too. Tamper with any of it after the fact and the recomputed commitment/output no longer matches. |
| "Those Stage 0 embeddings are the genuine, unaltered output of the real embedding step on the real answer" | `witness_hash` — committed to an append-only, hash-linked audit chain the instant Stage 0 finishes, *before* proving even starts | The provenance. A proof alone can't distinguish a genuine Stage 0 record from a compromised one that's fabricated-but-internally-consistent; this closes that gap. |

Both checks run at verification time; both have to pass. This two-part split — not one bigger proof — is the actual design decision here, and it's deliberate: current ZK tooling can't put a full free-form language model inside a provable circuit at reasonable cost, so the heavy semantic embedding happens off-circuit (Stage 0, reproducible not proven) and the bounded attention + scoring computation that actually decides scores is what's proven. See `docs/judge_qa_prep.md` for the fuller reasoning.

**Explainability, generated from the same computation the proof is over:** each per-criterion score comes with evidence chunks, a confidence flag (`high_confidence` / `review_recommended`), and a one-paragraph reason — generated from the identical attention weights the EZKL proof is over, not a separate post-hoc explanation. Surfaced in the Student view, the Teacher's batch review, the Auditor's confidence shortlist, and the scorecard PDF.

**Explicitly not covered, by design, not oversight:** whether an OCR'd answer sheet was genuinely transcribed by the claimed model (that path is a disclosed convenience feature, trusted at the same level a typed answer already is — see the Q&A in `docs/judge_qa_prep.md`).

## Architecture

```
                    ┌──────────────────────┐
  Browser ──REST──▶ │  Gateway (Express)    │ ── system of record ──▶ SQLite
(React/Vite,        │  batches · submissions│      (batches, submissions,
 nginx in Docker)   │  evaluations · audit  │       evaluations, overrides,
                     └──────────┬───────────┘       audit_chain — see
                                │ HTTP                gateway/src/db/schema.sql)
                                ▼
                     ┌────────────────────────────┐
                     │  ML-Worker (FastAPI)         │
                     │  Stage 0: fine-tuned BGE-small│  embeddings, off-circuit
                     │  Stages 1-3: RCAJ-X + EZKL    │  attention + scoring, on-circuit (proven)
                     │  Gemini OCR (optional)        │  answer-sheet transcription
                     └────────────────────────────┘
```

Node owns state and the audit trail's timing guarantees; Python owns computation and is stateless with respect to what happened in what order. Full request/response shapes: `docs/api_contract.md`.

**End-to-end flow:** Teacher certifies a rubric (hashes rubric + the encoder's identity + the RCAJ_X checkpoint's identity, locked before any submission exists) → Student submits an answer (or a Teacher uploads a scanned sheet, transcribed via Gemini) and gets an `input_hash` receipt immediately → Stage 0 embeds the answer against the rubric → Gateway commits `witness_hash` *before* calling Stages 1-3 → RCAJ-X scores + EZKL proves it → explanations are generated from the same computation → a scorecard PDF (QR-coded, with per-criterion evidence) is generated → anyone — the student, an auditor, a re-uploaded scorecard — can independently `/verify` the whole chain.

**Note on the EZKL circuit:** the RCAJ-X circuit's one-time trusted setup needs substantially more RAM than typical demo hardware (confirmed by actually compiling it — see `ml-worker/README.md` "Circuit setup" and `TRAINING_AND_EZKL_PLAN.md` at the repo root for the full cross-machine workflow). `/rcajx/embed` and `/rcajx/score` (non-proving) work without a built circuit; `/ezkl/prove` and `/ezkl/verify` 503 until `app/models/rcajx/rcajx_circuit/` is in place.

## Quick start (Docker — recommended, cross-platform)

```bash
cp .env.example .env      # fill in GEMINI_API_KEY if you want live OCR transcription
docker compose up -d
```

| Service | URL |
|---|---|
| Frontend (Teacher / Student / Auditor views) | http://localhost:5173 |
| Gateway API | http://localhost:4000 |
| ML-Worker API | http://localhost:8001 |

First `up` takes a few minutes — the ml-worker image build installs torch/EZKL/etc, then first *container startup* downloads `BAAI/bge-reranker-base` (~1.1GB) and builds the EZKL circuit (compile → calibrate → SRS → trusted setup). Both are cached (`hf_cache`, `ezkl_cache` volumes); every restart after that is fast (~15s). `depends_on: condition: service_healthy` means `docker compose up` (without `-d`) visibly blocks in the right startup order — ml-worker, then gateway, then frontend.

```bash
docker compose logs -f ml-worker   # watch first-run model download / circuit build
docker compose ps                  # health status of all three
docker compose down                # stop (add -v to also wipe cache volumes)
```

<details>
<summary><strong>Cross-platform & permissions notes</strong> — read if something's not building/writable</summary>

**Multi-arch:** images build from `python:3.12-slim` and `node:20-slim` (official multi-arch bases: linux/amd64 and linux/arm64, i.e. Apple Silicon under Docker Desktop). `torch`/`onnx`/`onnxruntime`/`ezkl` all publish prebuilt wheels for both; `better-sqlite3` ships prebuilt binaries for both. Each image also includes a source-build fallback (`build-essential`, plus `python3` in the gateway image for node-gyp) in case a platform wheel is ever unavailable — untested on arm64 specifically (this machine is x86_64), so a source-build fallback there would be slower but should still complete.

**Non-root, uid 1000:** both containers run as a non-root user at uid/gid 1000 (`node:20-slim`'s built-in `node` user for the gateway; a created `appuser` for ml-worker) — not root. `ml-worker/app/models/{zone2_circuit,proofs}` and `gateway/data` are *bind-mounted* from the host (not named volumes), specifically so `scripts/tamper_suite.sh` and `seed_demo_data.sh` can edit the proof files and SQLite DB directly by host path, exactly as they do against local dev. A root-owned container process would leave those files unwritable from the host afterward. 1000 is the common first-user id on Linux and macOS — if your host user's `id -u` is different, rebuild with:
```bash
docker compose build --build-arg APP_UID=$(id -u) --build-arg APP_GID=$(id -g) ml-worker
```
(the gateway image reuses `node:20-slim`'s fixed uid 1000 and isn't build-arg configurable — on a mismatched host, run `sudo chown -R $(id -u):$(id -g) gateway/data` once instead.)

</details>

## Local dev (no Docker)

Three independent services, no shared root `package.json` / workspace.

<details>
<summary><strong>1. ML-Worker</strong> (FastAPI, Python)</summary>

```bash
cd ml-worker
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
cp .env.example .env   # fill in GEMINI_API_KEY if you want live OCR transcription
uvicorn app.main:app --port 8001
```

Startup loads the fine-tuned encoder + RCAJ_X checkpoint (fast, both committed to the repo) and checks for a built EZKL circuit — **not built automatically** (see "Note on the EZKL circuit" above and `ml-worker/README.md`).

```bash
curl http://127.0.0.1:8001/health   # {"status":"ok","circuit_ready":false}  <- true once the circuit's in place
```
</details>

<details>
<summary><strong>2. Gateway</strong> (Express, Node)</summary>

```bash
cd gateway
npm install
cp .env.example .env   # ML_WORKER_URL defaults to http://127.0.0.1:8001
npm start
```

Owns `data/certiproof.db` (SQLite) — batches, submissions, evaluations, overrides, audit chain. The ML-Worker must already be running.
</details>

<details>
<summary><strong>3. Frontend</strong> (React + Vite)</summary>

```bash
cd frontend
npm install
npm run dev
```

Opens on `http://localhost:5173`, proxies `/api/*` to the Gateway (see `vite.config.js`).
</details>

Start order matters here (Docker handles it automatically via healthchecks): **ML-Worker → Gateway → Frontend.** The Gateway fails evaluation *calls*, not startup, if the ML-Worker isn't reachable yet.

## Project structure

```
certiproof/
├── gateway/          Express + SQLite — system of record, audit chain, scorecard PDF/QR
├── ml-worker/         FastAPI — RCAJ-X (Stage 0 encoder + Stages 1-3 attention/scoring + EZKL circuit), Gemini OCR, training/ pipeline
├── frontend/          React + Vite — Teacher / Student / Auditor views
├── docs/
│   ├── api_contract.md      full request/response contract between all three services
│   ├── demo_script.md       the eight-act demo walkthrough
│   └── judge_qa_prep.md     the three answers worth memorizing + all ten tamper vectors
└── scripts/            demo data, test fixtures, and the tamper suite — see below
```

See `TRAINING_AND_EZKL_PLAN.md` at the repo root for the full training-data + EZKL circuit-setup workflow (synthetic + real data, cross-machine circuit build).

## Demo data, test fixtures & the tamper suite

```bash
cd scripts
./seed_demo_data.sh                 # one clean + one deliberately-tampered evaluation
./tamper_suite.sh                   # all 10 tamper vectors, scripted, against the live system
python3 run_fixture_tests.py        # 15 (paper × answer-quality) fixtures through the real pipeline
```

Works against either Docker or local dev — these just talk to whatever's listening on `:4000`/`:8001`.

- **`seed_demo_data.sh`** builds the two pre-cached records the demo's Break-It act uses, so that act doesn't depend on live proving on stage. The tampered one shows exactly `proof_validity: false` (per-criterion scores don't match what's embedded in the proof) — the project's core claim, in one API response.
- **`tamper_suite.sh`** exercises every row in the table below against the running system and asserts `/verify` catches each one, then restores state.
- **`fixtures/papers.json`** + **`answers.json`** define 3 rubrics × 5 answer-quality scenarios (excellent / partial / off-topic / minimal / verbose-padding) — 15 cases, used by `run_fixture_tests.py` to check both correctness (`/verify` passes) and scoring quality (does `excellent` actually outscore `off_topic`?). `generate_test_sheets.py` renders each as a simulated scanned answer sheet for testing the OCR-upload flow without real handwriting; drop real handwritten photos into `fixtures/sheets/` to test with the genuine article instead — the upload flow doesn't care which.

Both scripts require a built EZKL circuit (`ml-worker/app/models/rcajx/rcajx_circuit/`) — see `TRAINING_AND_EZKL_PLAN.md`.

### The ten tamper vectors `tamper_suite.sh` checks

| # | Vector | Caught by |
|---|---|---|
| 1 | RCAJ_X model/circuit swapped after certification | `zone2_model_hash` mismatch |
| 2 | Encoder model/config swapped after certification | `zone1_model_hash` mismatch |
| 3 | Answer altered after submission | `input_hash` mismatch |
| 4 | Recorded final score edited in the DB | derived from the proof's own public output no longer matches (`proof_validity`) |
| 5 | Stored per-criterion scores edited after proving | derived from the proof's own public output no longer matches `per_criterion_scores_json` (`proof_validity`) — proof itself untouched |
| 6 | **Stage 0 inputs (embeddings) fabricated after the fact** | the independently-committed `witness_hash` no longer matches a re-hash of the stored record — *the one to lead with; this is what the two-part design is for* |
| 7 | Proof artifact corrupted/forged | EZKL verifier rejects outright |
| 8 | Historical audit-chain row edited | hash-link verification finds the first broken link |
| 9 | Evaluation re-associated with a different (also validly certified) batch | `poseidon_commitment` check — recomputed against the batch's *current* criterion weights, no longer matches |
| 10 | Scorecard PDF edited after issuance | re-uploaded file's SHA-256 no longer matches `scorecard_hash` recorded at generation time |

Full one-line-each version with more context: `docs/judge_qa_prep.md`.

## Further reading

- **`docs/api_contract.md`** — the exact contract every route is built against; update this first if either service needs to deviate.
- **`docs/demo_script.md`** — the eight-act walkthrough (certify → submit → prove → OCR/scorecard → break-it → explainability → override → close).
- **`docs/judge_qa_prep.md`** — the three Q&A answers worth having word-for-word, India-context/regulatory grounding, and the tamper-vector table above with full reasoning.
- **`TRAINING_AND_EZKL_PLAN.md`** (repo root) — the training-data + EZKL circuit-setup workflow: synthetic + real (ASAP-SAS/Mohler) data, GPU training locally, CPU circuit build on a bigger-RAM machine, bringing artifacts back.
