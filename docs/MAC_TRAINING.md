# Training RCAJ-X on a freshly-cloned Mac

For an agent (or a person) that just cloned this repo on macOS and needs to run or resume
training. This is the Mac-specific companion to `TRAINING_AND_EZKL_PLAN.md` at the repo
root, which covers the cloud/bigger-machine + EZKL circuit workflow — read that one too if
you're also building the EZKL circuit, not just training.

## Why this doc exists

Everything in this codebase resolves paths via `Path(__file__)`, never hardcoded, and
`training/train.py`'s `_device()` already auto-detects Apple Silicon (CUDA → MPS → CPU) — so
a Mac clone is not a special case in the code. It's a special case in *setup order*: no CUDA
wheel step, and Apple Silicon uses MPS instead.

## 1. Setup

```bash
cd certiproof-v2/ml-worker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-training.txt
python3 -m spacy download en_core_web_sm
```

No separate GPU-torch install step here — the regular `requirements.txt` torch wheel runs
fine on macOS (CPU), and `train.py` will pick up MPS automatically on Apple Silicon without
any extra package. Do **not** use `requirements-training-gpu.txt` — that file assumes a CUDA
install step that doesn't apply on Mac.

## 2. Ground rule: run the full pipeline, don't try to reuse partial artifacts

`app/models/rcajx/rcaj_x_best.pt` and `rcajx.onnx` are committed to git (small,
deterministic) so they'll be present right after clone. `bge_small_finetuned/` (~127MB) and
`rcajx_circuit/` (multi-GB) are **not** committed (see `ml-worker/.gitignore`) — a fresh
clone won't have them, and that's expected, not an error.

Don't try to copy in or reuse a `bge_small_finetuned/` from another machine and skip the
fine-tune step — just run the full pipeline below every time (encoder fine-tune →
preprocess → train → benchmark → export). At this data scale it's minutes, and it avoids
ever having a checkpoint that's inconsistent with the encoder that produced its embeddings.

- Without `rcajx_circuit/` (you won't have this on a Mac — see below), `/rcajx/embed` and
  `/rcajx/score` still work; only `/ezkl/prove`/`/ezkl/verify` 503. Building that circuit
  needs a machine with real RAM headroom (32GB+) — that's the cloud-machine workflow in the
  root plan doc, not this one. Don't attempt the EZKL circuit build on a laptop-class Mac.

## 3. Run training

```bash
python3 training/finetune_encoder.py     # always run this -- don't skip it by reusing another machine's checkpoint
python3 -m app.rcajx.preprocessing       # builds data/train_embedded.pt, data/test_embedded.pt
python3 training/train.py                # prints "Training device: mps" on Apple Silicon
python3 training/benchmark.py             # results/benchmark_report.md
python3 -m app.rcajx.export_onnx          # app/models/rcajx/rcajx.onnx + parity test
```

Or, equivalently, the turnkey script (run from `ml-worker/`):
```bash
./scripts/retrain_and_deploy.sh --finetune-encoder
# (always pass --finetune-encoder here; omit --with-circuit -- don't attempt the circuit build on this machine)
```

**MPS-specific things to know:**
- If you hit an op MPS doesn't support (rare for this model — it's small custom
  attention/scoring layers, not an exotic architecture), force CPU instead:
  `RCAJX_TRAIN_DEVICE=cpu python3 training/train.py`. The saved checkpoint always lands back
  on CPU regardless of training device, so this doesn't affect anything downstream.
- Training at this data scale (hundreds of examples, a small custom model, not a large
  transformer finetune) is expected to take minutes on MPS, not hours — if it's running much
  longer than that, something's likely stuck (e.g. silently fell back to CPU with a large
  grid) rather than genuinely compute-bound.

## 4. After training

Read `results/benchmark_report.md`, specifically the per-`variant_type` MAE for any real
datasets (`asap_sas`/`mohler`) you've ingested — if it's much worse than synthetic variants,
suspect the ingestion mapping (`training/ingest_external_dataset.py`'s `--max-score`) before
suspecting the model.

To actually serve on this Mac (embedding/scoring only, no proving, since there's no local
circuit): `uvicorn app.main:app --port 8001 &` then `curl http://127.0.0.1:8001/health` —
expect `circuit_ready: false` unless you've copied in a `rcajx_circuit/` built elsewhere.

To get a circuit, hand `rcajx.onnx` + `rcaj_x_best.pt` off to the cloud-machine workflow in
`TRAINING_AND_EZKL_PLAN.md`.
