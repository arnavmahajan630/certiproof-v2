# ml-worker (RCAJ-X)

FastAPI service implementing CertiProof's scoring pipeline on the RCAJ-X
architecture (bi-encoder embeddings + multi-head cross-attention + bounded
scoring head), replacing the old Zone1 cross-encoder + Zone2 MLP.

- `app/rcajx/` — runtime inference code (model, preprocessing, explain, guardrails, rubric cache)
- `app/zk/rcajx_ezkl_pipeline.py` — EZKL circuit lifecycle (see "Circuit setup" below)
- `training/` — retraining pipeline (not imported by the API)
- `data/` — rubrics + train/test examples (schema `training/ingest_external_dataset.py` targets)
- `app/models/rcajx/` — checkpoints + circuit artifacts (gitignored, rebuilt locally)

## Local setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-training.txt
.venv/bin/python3 -m spacy download en_core_web_sm
```

This works the same on Linux and macOS (Apple Silicon included) — `torch`,
`ezkl`, `onnxruntime`, and `sentence-transformers` all publish macOS wheels, and
every path in this codebase is resolved relative to the repo (via `Path(__file__)`),
never hardcoded — so cloning onto a different machine and re-running the setup
above is the entire porting story.

Run tests: `.venv/bin/python3 -m pytest app/rcajx/tests/`

`app/models/rcajx/rcaj_x_best.pt` and `rcajx.onnx` are committed (small,
deterministic). `app/models/rcajx/bge_small_finetuned/` (~130MB fine-tuned
encoder) and `rcajx_circuit/` (EZKL artifacts) are **not** — on a fresh clone,
either copy them in from wherever they were last built, or regenerate:
`.venv/bin/python3 training/finetune_encoder.py` for the encoder (needs
`data/train` + `data/test`, both committed), and see "Circuit setup" below for
the circuit.

## Circuit setup (run once, ideally on a machine with plenty of RAM)

`app/zk/rcajx_ezkl_pipeline.py`'s `build_circuit()` (settings → calibrate →
compile → SRS → trusted setup) is **not** run automatically at API startup,
unlike the old Zone2 pipeline. At this model's full size (10 criteria × 24
answer chunks), the circuit compiles to roughly logrows 20–25 depending on
EZKL's scale auto-search — during development, running `setup()` at logrows 25
OOM-killed a 16GB machine. `setup()` is a **one-time** step; only `prove()` /
`verify()` need to be fast on the actual demo machine.

Workflow:

1. On a machine with more RAM (a cloud VM, a friend's machine with 32GB+):
   ```bash
   ./scripts/retrain_and_deploy.sh --with-circuit
   ```
   or, if you just need the circuit rebuilt (model unchanged):
   ```bash
   .venv/bin/python3 -m app.zk.rcajx_ezkl_pipeline
   ```
2. Copy `app/models/rcajx/rcajx_circuit/` (settings.json, network.compiled,
   kzg.srs, vk.key, pk.key, rcajx_model_hash.txt) onto the demo machine, same
   path.
3. On the demo machine: `./scripts/retrain_and_deploy.sh` (no `--with-circuit`)
   or just `docker compose build ml-worker && docker compose up -d ml-worker`.
4. `curl http://localhost:8001/health` → `circuit_ready: true` confirms it picked
   up the copied artifacts.

If `rcaj_x_best.pt` / `bge_small_finetuned` change (retraining on new data) but
you don't rebuild the circuit, `rcajx_model_hash` (baked into the circuit at
setup time) goes stale relative to the new model weights — rebuild the circuit
whenever the model changes, not just when the data does.

## GPU training (optional, local)

`training/train.py` auto-detects CUDA (falls back to MPS, then CPU) — see
`requirements-training-gpu.txt` for a GPU-specific install (this repo's regular
`requirements.txt` pins the CPU-only torch wheel index, for the API-serving
image). The saved checkpoint always lands back on CPU regardless of training
device, so a GPU-trained `rcaj_x_best.pt` loads identically in the CPU-only
serving path. Force CPU even with a GPU present via
`RCAJX_TRAIN_DEVICE=cpu python3 training/train.py`.

Full cross-machine workflow (local GPU training → circuit build on a bigger-RAM
machine → bringing it back): `TRAINING_AND_EZKL_PLAN.md` at the repo root.

## Retraining on new data (e.g. ASAP-SAS / Mohler)

```bash
# 1. Ingest external data into data/{train,test}/ (see script docstring for format)
.venv/bin/python3 training/ingest_external_dataset.py \
    --csv /path/to/dataset.csv \
    --id-col <question-id-column> --text-col <answer-text-column> --score-col <score-column> \
    --max-score <dataset's max score> --split train --source-name asap_sas

# 2. Full retrain + deploy (add --with-circuit if this machine has the RAM for it,
#    or run just this on the beefier machine and copy the circuit dir over after)
./scripts/retrain_and_deploy.sh [--with-circuit] [--finetune-encoder]
```

`--finetune-encoder` re-runs the BGE-small hard-negative fine-tuning pass before
training RCAJ_X — worth it once you have enough real negation/confidently-wrong
examples in the new data (see `training/finetune_encoder.py`'s docstring),
skippable otherwise.

The existing synthetic data in `data/train/` + `data/test/` stays in place —
`ingest_external_dataset.py` adds alongside it, it doesn't replace it.
