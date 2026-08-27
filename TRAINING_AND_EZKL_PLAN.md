# Training + EZKL Circuit Plan — Cloud/Bigger-Machine Workflow

The rest of CertiProof (Gateway, ML-Worker API, frontend, tamper vectors, docs) is migrated to RCAJ-X and demo-ready. What's left is **training on real data + building the EZKL circuit**. Both are now planned to run on a bigger machine than a personal laptop — not because of RAM alone anymore, but because local-GPU training on a laptop is also thermally/acoustically unworkable in practice (the last local run was killed mid-training because the fan noise was unusable). This doc is the exact, in-order procedure for that, structured so it works whether training and the circuit build happen **on the same machine** or **on two separate machines**.

**Default recommendation: one machine, both steps.** Any cloud GPU instance with 32GB+ system RAM (most mid/large GPU instances — e.g. an A10/A100/L4 class instance on RunPod, Lambda, Vast.ai, AWS/GCP/Azure — clear this easily; it's system RAM that matters for the circuit step, not GPU VRAM) can do training and the EZKL circuit build back-to-back without ever touching your laptop. Only fall back to the two-machine split below if the GPU instance you can get/afford is RAM-constrained (e.g. a cheap 16GB-RAM GPU box).

---

## Ground rule: always run the full pipeline, start to finish, on the new machine

Don't try to carry over or reuse partial artifacts (`bge_small_finetuned/`, `*_embedded.pt`,
`rcaj_x_best.pt`, `rcajx.onnx`) from a previous machine or a previous killed run, and don't
reason about which of them is newer/older than another. It's not worth the bookkeeping: the
whole training pipeline (encoder fine-tune → preprocess → train → benchmark → export) is
minutes on any real GPU at this data scale, and running it end-to-end from committed source
data every time means there's never a mixed-generation set of artifacts to worry about.

Real data is already committed and ingested into `ml-worker/data/` (242 question/rubric
entries in `data/raw/rubrics.json` including `asap-set-*` (ASAP-SAS) and `mohler-*` sets
alongside the synthetic ones, 662 files in `data/train/`, 1238 in `data/test/`) — that's all
Part 1/2 below need. Neither `bge_small_finetuned/` nor `rcajx_circuit/` are committed to git
(see `ml-worker/.gitignore`); Part 2 regenerates the former from scratch, Part 3 the latter.

---

## Part 1 — Provision the machine + get the repo on it

```bash
git clone <your-repo-url> certiproof
cd certiproof/ml-worker
python3 -m venv .venv
source .venv/bin/activate
```

**GPU-enabled torch first** — check the instance's CUDA version (`nvidia-smi`), then:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # adjust cu121 to match
pip install -r requirements-training-gpu.txt
python3 -m spacy download en_core_web_sm
```

Everything needed to build the model from scratch is already committed
(`data/raw/rubrics.json`, `data/train/*.json`, `data/test/*.json`) — go straight to Part 2.

---

## Part 2 — Train (GPU), full pipeline every time

Run all of this in order, every time, on the new machine — don't skip the encoder fine-tune
step even if a `bge_small_finetuned/` happens to exist from somewhere else.

### 2.1 Fine-tune the encoder

```bash
python3 training/finetune_encoder.py
```
Expected to be quick (single-digit minutes on any real GPU) — the dataset is ~2k examples, not the bottleneck. This is also almost certainly what made the fan spin up hard on the laptop; on a cloud instance this is a non-issue.

### 2.2 Preprocess + train + benchmark + export

```bash
python3 -m app.rcajx.preprocessing        # builds/rebuilds data/train_embedded.pt, data/test_embedded.pt
python3 training/train.py                 # prints "Training device: cuda" if the GPU install worked
python3 training/benchmark.py             # results/benchmark_report.md
python3 -m app.rcajx.export_onnx          # app/models/rcajx/rcajx.onnx + parity test
```

**Read `results/benchmark_report.md` before continuing.** Specifically check the per-`variant_type` breakdown for `asap_sas`/`mohler` rows — if MAE there is wildly worse than the synthetic variants, something in the ingestion mapping (`training/ingest_external_dataset.py`'s `--max-score`, most likely) is off. Fix and re-ingest before moving on, not after the circuit's built — the circuit build (Part 3) takes real wall-clock time and shouldn't be spent on a model you're about to retrain anyway.

To fold in *more* external data before training (not required — the data above is already ingested):
```bash
python3 training/ingest_external_dataset.py \
  --csv /path/to/file.csv --id-col <col> --text-col <col> --score-col <col> \
  --max-score <n> --split train --source-name <name>
```
See the script's docstring — it appends alongside existing data, never replaces it.

---

## Part 3 — EZKL circuit build

**If this is the same machine as Part 2 and it has 32GB+ RAM:** just continue here directly, no packaging/transfer needed.

**If splitting to a second, RAM-heavier machine:** package the ONNX + checkpoint first —
```bash
mkdir -p /tmp/rcajx_handoff
cp app/models/rcajx/rcajx.onnx app/models/rcajx/rcaj_x_best.pt /tmp/rcajx_handoff/
tar czf rcajx_handoff.tar.gz -C /tmp rcajx_handoff
```
— transfer it (`scp`/cloud storage), then on the second machine:
```bash
git clone <your-repo-url> certiproof && cd certiproof/ml-worker
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # CPU torch wheel is correct here, no GPU needed for the circuit build
mkdir -p app/models/rcajx
cp /path/to/rcajx_handoff/{rcajx.onnx,rcaj_x_best.pt} app/models/rcajx/
```

### 3.1 Run the build

```bash
python3 -m app.zk.rcajx_ezkl_pipeline
```

This runs `gen_settings` → `calibrate_settings` → `compile_circuit` → `get_srs` → `setup`, in that order, and prints the resulting `logrows` before the heavy `setup()` step — **watch this number**. This model (10 criteria × 24 answer chunks) has landed anywhere from logrows 20 to 25 across runs (EZKL's scale auto-search is nondeterministic); at logrows 25 the SRS alone is ~4.3GB and `setup()` needs real headroom over that to not OOM. 32GB gives comfortable margin; 16GB does not.

**Watch memory while it runs** (separate terminal): `watch -n2 free -h`. If it's climbing toward the ceiling as `setup()` starts:
- Let it run a bit longer before panicking — the peak is usually brief.
- If it's clearly heading for OOM, **Ctrl+C** rather than letting the OS OOM-killer take it (a full OOM has taken down the whole machine before). See "If logrows lands too high" below.

On success:
```
app/models/rcajx/rcajx_circuit/
├── settings.json
├── network.compiled
├── kzg.srs
├── vk.key
├── pk.key
└── rcajx_model_hash.txt
```

### 3.2 Verify it locally before tearing the machine down

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
1. **Retry** — the scale search is nondeterministic; a second run sometimes lands meaningfully lower (seen 20 vs 25 for the identical model). Delete `app/models/rcajx/rcajx_circuit/` and re-run 3.1.
2. **Shrink the circuit** — edit `MAX_CHUNKS` in `ml-worker/app/rcajx/padded_model.py` down from 24 (e.g. to 12 or 8), re-export ONNX (padding constants must match between export and the checkpoint that produced it — the checkpoint itself doesn't need retraining, only re-export via `python3 -m app.rcajx.export_onnx`), retry the build.
3. **More RAM** — bump the cloud instance up a tier. `setup()` doesn't need a GPU, just RAM, so this is usually the cheapest lever if 1-2 don't land it.

### 3.3 Bring the result back (only if this was a second machine)

```bash
tar czf rcajx_circuit_built.tar.gz -C certiproof/ml-worker/app/models/rcajx rcajx_circuit
```
Transfer it back to wherever you'll run/demo from.

---

## Part 4 — Bring it together and go live

If Parts 2 and 3 ran on the same machine, everything's already in place — skip to verifying below. If split, unpack the circuit next to the model it was built from:

```bash
cd certiproof/ml-worker
mkdir -p app/models/rcajx
tar xzf /path/to/rcajx_circuit_built.tar.gz -C app/models/rcajx
ls app/models/rcajx/   # rcaj_x_best.pt, bge_small_finetuned/, rcajx.onnx, rcajx_circuit/ should all be present
```

The circuit must have been built from *this exact* `rcaj_x_best.pt`'s ONNX export — if the model changes again without a circuit rebuild, `rcajx_model_hash` goes stale relative to the new weights.

Bring the service up (note: the script lives at `ml-worker/scripts/retrain_and_deploy.sh`, run from inside `ml-worker/`):
```bash
cd ml-worker
./scripts/retrain_and_deploy.sh    # no --with-circuit -- the circuit's already built and in place
# or, without Docker:
uvicorn app.main:app --port 8001 &
curl http://127.0.0.1:8001/health   # circuit_ready: true confirms it
```

Then, from the repo root:
```bash
cd scripts
./seed_demo_data.sh          # pre-caches the two Break-It demo records
./tamper_suite.sh            # all vectors, live -- confirm every one is still caught under the real circuit
python3 run_fixture_tests.py # fixtures through the real pipeline
```

**This is the point where `TAMPER_TESTS_SUMMARY.md`'s "not yet re-run end-to-end" caveat gets resolved** — re-run the Auditor's Tests panel in the browser too (`Set Up Fresh Test Case` → `Run All Vectors`) and confirm every vector shows `caught: true`, especially `witness_substitution` and `stage0_substitution` — those are the ones the whole two-part design exists for.

If everything above passes: demo-ready. If you're short on time at this point, the pre-cached `seed_demo_data.sh` records are the fallback — Act 4 (Break-It) doesn't need live proving on stage either way (see `docs/demo_script.md`).
