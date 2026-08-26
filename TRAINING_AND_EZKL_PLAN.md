# Training + EZKL Circuit Plan — Cross-Machine Workflow

The rest of CertiProof (Gateway, ML-Worker API, frontend, tamper vectors, docs) is migrated to RCAJ-X and demo-ready. What's left is the one thing that can't be done blind: **training on real data + building the EZKL circuit**, because the circuit's one-time trusted setup needs more RAM than a typical laptop has (confirmed by actually compiling it during development — see `ml-worker/README.md` "Circuit setup"). This doc is the exact, in-order procedure to do that across two machines and bring the result back cleanly.

**Machines:**
- **Machine A (yours, GPU)** — training: data ingestion, preprocessing, encoder fine-tuning, RCAJ_X training, benchmarking, ONNX export.
- **Machine B (college Ubuntu, 32GB RAM)** — the EZKL circuit build only: `gen-settings` → `calibrate-settings` → `compile-circuit` → `get-srs` → `setup`. CPU-only, no GPU needed here — EZKL's trusted setup is RAM-bound, not compute-bound in a way a GPU helps with.

**Why split it this way:** training benefits from your GPU and needs nothing exotic; the circuit build needs RAM your laptop doesn't have but no GPU at all. Machine B never needs your training data or a GPU — only the ONNX file Machine A produces.

---

## Part 0 — Before you leave for college (do this today, on Machine A)

Get the mechanical pipeline proven end-to-end on synthetic data only, so tomorrow's session is 100% about real data, not debugging plumbing.

```bash
cd certiproof/ml-worker
python3 -m venv .venv
source .venv/bin/activate

# GPU-enabled torch first (check your CUDA version: nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # adjust cu121 to match
pip install -r requirements-training-gpu.txt
python3 -m spacy download en_core_web_sm
```

```bash
python3 -m app.rcajx.preprocessing        # builds data/train_embedded.pt, data/test_embedded.pt
python3 training/train.py                 # prints "Training device: cuda" if the GPU install worked
python3 training/benchmark.py             # results/benchmark_report.md
python3 -m app.rcajx.export_onnx          # app/models/rcajx/rcajx.onnx + parity test
```

If all four run clean, the pipeline is proven. Everything past this point (real data in, circuit out) is running the same steps again, once, tomorrow — not new code paths.

**Also today:** get both datasets downloaded and skim their actual column headers (release-to-release these vary) — do this now, not at the venue, in case of network trouble:
- **ASAP-SAS** ("The Hewlett Foundation: Short Answer Scoring", Kaggle) — ships as **TSV**, not CSV. Typical columns: `Id`, `EssaySet`, `Score1`, `Score2`, `EssayText`.
- **Mohler dataset** (Mohler et al. 2011 short-answer grading corpus) — typically CSV with columns like `id`, `question`, `desired_answer`, `student_answer`, `score_avg` (0-5 scale) — column names vary by mirror, check the header row you actually have.

---

## Part 1 — Tomorrow, Machine A: ingest real data + retrain (GPU)

### 1.1 Ingest

`training/ingest_external_dataset.py` converts an external CSV/TSV into this repo's internal schema and appends alongside the existing synthetic set (never replaces it). It maps each dataset's **holistic** score onto one synthetic criterion per question — these datasets don't have multi-criterion rubrics, and there's no generic way to auto-split a single score into per-criterion ones. Read the script's docstring before running; verify your actual downloaded file's column names first (`head -3 your_file.tsv`).

```bash
# ASAP-SAS (TSV) -- adjust column names to what your file's header actually says
python3 training/ingest_external_dataset.py \
  --csv /path/to/asap_sas_train.tsv --delimiter $'\t' \
  --id-col EssaySet --text-col EssayText --score-col Score1 \
  --max-score 3 --split train --source-name asap_sas

# Mohler (CSV)
python3 training/ingest_external_dataset.py \
  --csv /path/to/mohler_dataset.csv \
  --id-col question --text-col student_answer --score-col score_avg \
  --max-score 5 --split train --source-name mohler
```

Split some of each into `--split test` too (a held-out slice, e.g. the last 15-20% of rows per dataset) so `benchmark.py` actually measures generalization on real data, not just train-set fit. Simplest approach: run the ingest command twice per dataset against two different pre-split files (train slice → `--split train`, held-out slice → `--split test`), or manually move a handful of the generated `data/train/asap_sas_*.json` / `data/train/mohler_*.json` files into `data/test/` afterward.

Sanity-check before training:
```bash
python3 -c "
import json
d = json.load(open('data/raw/rubrics.json'))
print(len(d), 'question ids total')
print([q for q in d if q['subject'] in ('asap_sas','mohler')][:3])
"
ls data/train | wc -l   # should have grown past the original 130
```

### 1.2 Re-preprocess, optionally fine-tune the encoder, retrain

```bash
python3 -m app.rcajx.preprocessing
python3 training/finetune_encoder.py     # optional -- worth it if the new data has enough hard-negative-style examples; skip if short on time
python3 -m app.rcajx.preprocessing       # re-run if you fine-tuned -- picks up the new encoder
python3 training/train.py                # GPU, ~minutes not hours at this data scale
python3 training/benchmark.py
```

**Read `results/benchmark_report.md` before continuing.** Specifically check the per-`variant_type` breakdown for `asap_sas`/`mohler` rows — if MAE there is wildly worse than the synthetic variants, something in the ingestion mapping is likely off (e.g. `--max-score` mismatched to the dataset's actual scale) — fix and re-run 1.1-1.2 before moving on, not after the circuit's built.

### 1.3 Export ONNX (still on Machine A)

```bash
python3 -m app.rcajx.export_onnx
```

Confirms `app/models/rcajx/rcajx.onnx` matches the new `rcaj_x_best.pt` and passes the PyTorch/ONNXRuntime parity check. **This is the only file Machine B actually needs.**

### 1.4 Package what goes to Machine B

```bash
mkdir -p /tmp/rcajx_handoff
cp app/models/rcajx/rcajx.onnx /tmp/rcajx_handoff/
cp app/models/rcajx/rcaj_x_best.pt /tmp/rcajx_handoff/   # for the model-hash step later, not needed for the circuit build itself
tar czf rcajx_handoff.tar.gz -C /tmp rcajx_handoff
```
Carry `rcajx_handoff.tar.gz` on a USB drive or upload it somewhere you can pull from on Machine B — don't rely on the venue's network for a multi-hundred-MB transfer under time pressure.

---

## Part 2 — At college, Machine B: circuit build (CPU, 32GB RAM)

### 2.1 Clone + minimal setup

You don't need training data, spacy, or a GPU here — only `ezkl`, `onnx`, `onnxruntime`, `torch` (CPU is fine and correct here) to run the circuit pipeline module.

```bash
git clone <your-repo-url> certiproof   # or: git pull, if it's already cloned here
cd certiproof/ml-worker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # CPU torch wheel index -- correct on this machine
```

### 2.2 Drop in the ONNX + checkpoint from Machine A

```bash
mkdir -p app/models/rcajx
cp /path/to/rcajx_handoff/rcajx.onnx app/models/rcajx/
cp /path/to/rcajx_handoff/rcaj_x_best.pt app/models/rcajx/
```

### 2.3 Run the circuit build

```bash
python3 -m app.zk.rcajx_ezkl_pipeline
```

This runs `gen_settings` → `calibrate_settings` → `compile_circuit` → `get_srs` → `setup`, in that order, and prints the resulting `logrows` before the heavy `setup()` step — **watch this number**. During development this circuit landed anywhere from logrows 20 to 25 depending on EZKL's scale auto-search (nondeterministic between runs); at logrows 25 the SRS alone is ~4.3GB and `setup()` needs enough RAM to OOM-kill a 16GB machine. 32GB gives real headroom over that, but isn't infinite.

**Watch memory while it runs** (separate terminal): `watch -n2 free -h`. If it's climbing toward the ceiling as `setup()` starts:
- Let it run a bit longer before panicking — the peak is usually brief.
- If it's clearly heading for OOM, **Ctrl+C**, don't let the OS kill it (same risk of taking the whole machine down that happened during development). Then see "If logrows lands too high" below.

On success you'll have:
```
app/models/rcajx/rcajx_circuit/
├── settings.json
├── network.compiled
├── kzg.srs
├── vk.key
├── pk.key
└── rcajx_model_hash.txt
```

### 2.4 Verify it locally before leaving Machine B

Cheap, fast — do this before packing up:
```bash
python3 -c "
import asyncio, torch
from app.zk import rcajx_ezkl_pipeline as zk
from app.rcajx.padded_model import pad_inputs, MAX_CRITERIA, MAX_CHUNKS, D_MODEL

print('circuit_is_ready:', zk.circuit_is_ready())
print('rcajx_model_hash:', zk.read_model_hash())

g = torch.Generator().manual_seed(0)
R = torch.randn(3, D_MODEL, generator=g); A = torch.randn(5, D_MODEL, generator=g)
neg = torch.zeros(3); mm = torch.tensor([2.,2.,2.]); w = torch.tensor([.4,.3,.3])
padded = pad_inputs(R, A, neg, mm, criterion_weights=w)
result = zk.prove(padded, 'app/models/rcajx/proofs', tag='handoff_smoke_test')
print('prove OK, final_score=', result['final_score'])
verify_result = zk.verify(result['proof_path'])
print('verify OK:', verify_result['verified'])
"
```
If both print `True`/succeed without error, the circuit is genuinely usable — not just "files exist."

### If logrows lands too high / setup OOMs even at 32GB

In order of preference:
1. **Retry** — `gen_settings`/`calibrate_settings`'s scale search is nondeterministic; a second run sometimes lands meaningfully lower (seen 20 vs 25 for the identical model during development). Delete `app/models/rcajx/rcajx_circuit/` and re-run 2.3.
2. **Shrink the circuit** — edit `MAX_CHUNKS` in `ml-worker/app/rcajx/padded_model.py` down from 24 (e.g. to 12 or 8), re-export ONNX **on Machine A** (padding constants must match between export and the model that produced the checkpoint — the checkpoint itself doesn't need retraining, only re-export), send the new ONNX back to Machine B, retry. Smaller `MAX_CHUNKS` was the most effective lever found during development (see `app/rcajx/padded_model.py`'s docstring).
3. **Find more RAM** — a cloud VM (even a few hours of a 64GB instance) if a college lab machine still isn't enough. `setup()` doesn't need a GPU, just RAM.

### 2.5 Package the result to bring back

```bash
tar czf rcajx_circuit_built.tar.gz -C certiproof/ml-worker/app/models/rcajx rcajx_circuit
```
Same rule as before: physical transfer (USB) preferred over relying on venue network for a large file under time pressure.

---

## Part 3 — Back on your machine: bring it together and go live

```bash
cd certiproof/ml-worker
mkdir -p app/models/rcajx
tar xzf /path/to/rcajx_circuit_built.tar.gz -C app/models/rcajx
```

Confirm the model that's about to serve matches the circuit that was just built (they must — the circuit was compiled from this exact `rcaj_x_best.pt`'s ONNX export):
```bash
ls app/models/rcajx/   # rcaj_x_best.pt, bge_small_finetuned/, rcajx.onnx, rcajx_circuit/ should all be present
```

Bring the service up:
```bash
./scripts/retrain_and_deploy.sh    # no --with-circuit -- the circuit's already built and in place
# or, without Docker:
uvicorn app.main:app --port 8001 &
curl http://127.0.0.1:8001/health   # circuit_ready: true confirms it
```

Then, from the repo root:
```bash
cd scripts
./seed_demo_data.sh          # pre-caches the two Break-It demo records
./tamper_suite.sh            # all 10 vectors, live -- confirm every one is still caught under the real circuit
python3 run_fixture_tests.py # 15 fixtures through the real pipeline
```

**This is the point where `TAMPER_TESTS_SUMMARY.md`'s "not yet re-run end-to-end" caveat gets resolved** — re-run the Auditor's Tests panel in the browser too (`Set Up Fresh Test Case` → `Run All Vectors`) and confirm all eleven show `caught: true`, especially `witness_substitution` and `stage0_substitution` — those are the ones the whole two-part design exists for.

If everything above passes: demo-ready. If you're short on time at this point, the pre-cached `seed_demo_data.sh` records are the fallback — Act 4 (Break-It) doesn't need live proving on stage either way (see `docs/demo_script.md`).
