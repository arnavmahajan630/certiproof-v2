#!/usr/bin/env bash
# Turnkey retrain-and-deploy pipeline. Run this from ml-worker/ (or anywhere --
# it cd's into its own location first).
#
# On the machine doing the EZKL circuit setup (needs plenty of RAM -- see
# README.md "Circuit setup"):
#   ./scripts/retrain_and_deploy.sh --with-circuit
#
# On the actual demo machine, once rcajx_circuit/ has been copied over from
# wherever --with-circuit ran:
#   ./scripts/retrain_and_deploy.sh
#
# Both modes retrain the model on whatever's currently in data/ (run
# training/ingest_external_dataset.py first to fold in new data) and bring the
# ml-worker container up. Only --with-circuit additionally rebuilds the EZKL
# circuit (settings/calibrate/compile/setup) -- the heavy, memory-hungry step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

WITH_CIRCUIT=false
SKIP_FINETUNE=true
for arg in "$@"; do
  case "$arg" in
    --with-circuit) WITH_CIRCUIT=true ;;
    --finetune-encoder) SKIP_FINETUNE=false ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

PYTHON="${HERE}/.venv/bin/python3"
if [ ! -x "$PYTHON" ]; then
  echo "No .venv found at ${HERE}/.venv -- create one first:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-training.txt" >&2
  echo "  .venv/bin/python3 -m spacy download en_core_web_sm" >&2
  exit 1
fi

echo "=== [1/6] Preprocessing (data/ -> train_embedded.pt / test_embedded.pt) ==="
"$PYTHON" -m app.rcajx.preprocessing

if [ "$SKIP_FINETUNE" = false ]; then
  echo "=== [1b/6] Fine-tuning encoder on new data (--finetune-encoder passed) ==="
  "$PYTHON" training/finetune_encoder.py
  echo "    Re-running preprocessing so it picks up the newly fine-tuned encoder..."
  "$PYTHON" -m app.rcajx.preprocessing
else
  echo "=== [1b/6] Skipping encoder fine-tune (pass --finetune-encoder to include it) ==="
fi

echo "=== [2/6] Training RCAJ_X (ablation grid -> app/models/rcajx/rcaj_x_best.pt) ==="
"$PYTHON" training/train.py

echo "=== [3/6] Benchmarking (results/benchmark_report.md -- eyeball MAE before continuing) ==="
"$PYTHON" training/benchmark.py
echo "    ^ review results/benchmark_report.md now. Ctrl+C within 10s to abort before deploying."
sleep 10

echo "=== [4/6] ONNX export + PyTorch/ONNXRuntime parity check ==="
"$PYTHON" -m app.rcajx.export_onnx

if [ "$WITH_CIRCUIT" = true ]; then
  echo "=== [5/6] EZKL circuit rebuild (gen-settings -> calibrate -> compile -> srs -> setup) ==="
  echo "    This is the heavy step -- see README.md 'Circuit setup' if it's slow/crashes."
  "$PYTHON" -m app.zk.rcajx_ezkl_pipeline
else
  echo "=== [5/6] Skipping EZKL circuit rebuild (pass --with-circuit to include it) ==="
  echo "    If the model changed, app/models/rcajx/rcajx_circuit/ from a previous"
  echo "    --with-circuit run is now stale (rcajx_model_hash won't match) -- copy a"
  echo "    freshly-built one over before relying on /ezkl/prove."
fi

echo "=== [6/6] Bringing the service up ==="
cd "$(dirname "$HERE")"  # certiproof/ (docker-compose.yml lives here)
docker compose build ml-worker
docker compose up -d ml-worker

echo "=== Done. Check: curl http://localhost:8001/health ==="
