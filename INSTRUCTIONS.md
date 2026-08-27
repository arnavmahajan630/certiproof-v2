# Instructions for agents — after cloning this repo

Read this first. It tells you what's already in the repo vs. what you need to build/fetch,
and which doc to go to for each task. Don't re-derive this by exploring the tree first —
this file is the map.

## What this repo is

CertiProof: a Gateway (Express/Node) + ML-Worker (FastAPI/Python, RCAJ-X model + EZKL
zero-knowledge proving) + frontend (React/Vite), see root `README.md` for the full
architecture and what's actually proven.

## Fastest path to a running system: Docker

```bash
cp .env.example .env      # fill in GEMINI_API_KEY if you want live OCR transcription
docker compose up -d
```
Frontend on `:5173`, Gateway on `:4000`, ML-Worker on `:8001`. First `up` takes a few
minutes (installs deps, downloads the base encoder, builds the EZKL circuit on first
container start). See root `README.md`'s "Quick start" for details, health checks, and
cross-platform notes (uid/gid, arm64).

If you just need the app running (not training, not rebuilding the circuit), stop here —
everything below is about the ML pipeline specifically.

## What's committed vs. what you need to build

Check `ml-worker/.gitignore`'s "ML-Worker runtime/generated artifacts" block — it documents
this exactly, but the short version:

| Path | Committed? | If missing |
|---|---|---|
| `ml-worker/data/raw/rubrics.json`, `data/train/*.json`, `data/test/*.json` | Yes | — already there after clone |
| `ml-worker/app/models/rcajx/rcaj_x_best.pt`, `rcajx.onnx` | Yes (small, deterministic) | — already there after clone |
| `ml-worker/app/models/rcajx/bge_small_finetuned/` | **No** (~127MB, over GitHub's plain-git limit) | Falls back automatically to the public base encoder (`BAAI/bge-small-en-v1.5` via Hugging Face) — pipeline still runs. Regenerate with `python training/finetune_encoder.py` for full quality. |
| `ml-worker/app/models/rcajx/rcajx_circuit/` | **No** (multi-GB, machine-specific) | `/rcajx/embed` and `/rcajx/score` work without it. `/ezkl/prove`/`/ezkl/verify` 503 until it exists. Building it needs a machine with 32GB+ RAM — see `TRAINING_AND_EZKL_PLAN.md`. Don't attempt this on a laptop. |

So: **a plain `git clone` + Docker up is enough for the full app to run and score**, just
without proof-generation until someone builds the circuit on a bigger machine.

## Task → doc map

- **Just run the app** → root `README.md`, "Quick start" / "Local dev (no Docker)".
- **Train or retrain the model, on a cloud/bigger GPU machine, and/or build the EZKL
  circuit** → `TRAINING_AND_EZKL_PLAN.md` (repo root). This is the canonical, in-order
  procedure — start there for anything involving `training/`, real datasets, or
  `app/zk/rcajx_ezkl_pipeline.py`.
- **Train specifically on a freshly-cloned Mac** → `docs/MAC_TRAINING.md`. Companion to the
  plan above — covers macOS/Apple Silicon (MPS) setup specifics only; it defers to the root
  plan for the EZKL circuit step (don't build the circuit on a Mac laptop).
- **Ingest a new external dataset** (ASAP-SAS-style, Mohler-style, or similar) →
  `ml-worker/training/ingest_external_dataset.py`'s docstring, then
  `TRAINING_AND_EZKL_PLAN.md` Part 2 for what to run after ingesting.
- **Understand what's actually proven / not proven, and why** → root `README.md`'s "What's
  actually proven" table, and `docs/judge_qa_prep.md`.
- **API request/response shapes** → `docs/api_contract.md`.
- **Demo script / walkthrough** → `docs/demo_script.md`.
- **Tamper-test vectors and their current pass/fail status** → `TAMPER_TESTS_SUMMARY.md`.
- **ml-worker internals** (module layout, circuit lifecycle, retrain script) →
  `ml-worker/README.md`.

## Known gotchas worth knowing before you start

- `ml-worker/scripts/retrain_and_deploy.sh` lives **inside** `ml-worker/`, not at the repo
  root — run it from `ml-worker/` (`cd ml-worker && ./scripts/retrain_and_deploy.sh`).
- `ml-worker/requirements.txt` pins the **CPU** torch wheel index (correct for the
  API-serving/Docker image). For GPU training use `requirements-training-gpu.txt` instead
  (Linux/Windows with CUDA) — on macOS just use the regular `requirements.txt` +
  `requirements-training.txt`, MPS needs no special wheel (see `docs/MAC_TRAINING.md`).
- Always run the full training pipeline start to finish on a new machine — encoder
  fine-tune, preprocess, train, benchmark, export — rather than trying to reuse or mix in
  partial artifacts (`bge_small_finetuned/`, `*_embedded.pt`, `rcaj_x_best.pt`, `rcajx.onnx`)
  from a previous machine or a previous killed run. It's fast enough at this data scale that
  there's no reason to reason about which artifacts are consistent with which — see
  `TRAINING_AND_EZKL_PLAN.md` Part 2.
- If you rebuild the EZKL circuit after the model changed, `rcajx_model_hash` (baked in at
  `setup()` time) must match the current checkpoint — rebuild the circuit whenever the model
  changes, not just when the underlying data does.
