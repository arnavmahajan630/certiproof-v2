"""EZKL circuit lifecycle for RCAJ-X (Stages 1-3: cross-attention + scoring +
aggregation) — the replacement for zk/ezkl_pipeline.py (Zone 2's plain MLP).

IMPORTANT — unlike the old Zone 2 pipeline, build_circuit() here is NOT called
automatically at API startup (see app/main.py). Confirmed empirically (see
certiproof-integration-plan follow-up notes) that ezkl.setup() on this circuit's
full (MAX_CRITERIA=10, MAX_CHUNKS=24) shape needs a trusted-setup pass sized
around logrows=20-25 depending on EZKL's scale auto-search — this OOM-crashed a
16GB laptop once during development. setup() is a ONE-TIME step (SRS + proving/
verifying keys); only prove()/verify() need to run fast on the actual demo
machine. Run `python -m app.zk.rcajx_ezkl_pipeline` as a standalone script,
ideally on a machine with more RAM (see ml-worker/README.md "Circuit setup"),
then copy the resulting rcajx_circuit/ directory onto the demo machine.

Calibration/witness input format (confirmed empirically against this EZKL
version, 23.0.5 — not documented anywhere obvious, so recorded here): for a
multi-input ONNX graph with NO batch dimension, both calibrate_settings and
gen_witness expect `{"input_data": [tensor0_flat, tensor1_flat, ..., tensorM_flat]}`
— ONE run, each of the M graph inputs flattened to a 1D list, NOT wrapped in an
extra per-example batching list (that format, which works for Zone 2's single-
input model, fails to deserialize here with a cryptic
"failed to deserialize FileSourceInner" error).

Poseidon commitment mechanics carried over unchanged from zk/ezkl_pipeline.py
(see that module's docstring for the hex/byte-order details) — only the input
shape changed (6 named tensors here vs. one flat (1,20) vector there).
"""
import asyncio
import hashlib
import json
import os

import ezkl
import numpy as np
import torch

from app.rcajx.model import RCAJ_X
from app.rcajx.padded_model import D_MODEL, MAX_CHUNKS, MAX_CRITERIA, PaddedRCAJX

HERE = os.path.dirname(os.path.dirname(__file__))  # ml-worker/app
MODELS_DIR = os.path.join(HERE, "models", "rcajx")
CIRCUIT_DIR = os.path.join(MODELS_DIR, "rcajx_circuit")
os.makedirs(CIRCUIT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(MODELS_DIR, "rcaj_x_best.pt")
ONNX_PATH = os.path.join(MODELS_DIR, "rcajx.onnx")
SETTINGS_PATH = os.path.join(CIRCUIT_DIR, "settings.json")
COMPILED_PATH = os.path.join(CIRCUIT_DIR, "network.compiled")
SRS_PATH = os.path.join(CIRCUIT_DIR, "kzg.srs")
VK_PATH = os.path.join(CIRCUIT_DIR, "vk.key")
PK_PATH = os.path.join(CIRCUIT_DIR, "pk.key")
CAL_DATA_PATH = os.path.join(CIRCUIT_DIR, "cal_data.json")
MODEL_HASH_PATH = os.path.join(CIRCUIT_DIR, "rcajx_model_hash.txt")

INPUT_NAMES = [
    "R_padded", "A_padded", "negation_flags_padded", "max_marks_padded",
    "chunk_mask", "criterion_weights_padded",
]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_model() -> RCAJ_X:
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model_config = {k: v for k, v in checkpoint["config"].items() if k in ["n_heads", "d_k", "d_v", "hidden"]}
    model = RCAJ_X(**model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def export_onnx():
    """Delegates to app.rcajx.export_onnx (also runs its own parity test)."""
    from app.rcajx import export_onnx as export_mod

    export_mod.export()
    export_mod.parity_test()


def _flatten_padded_example(R_p, A_p, neg_p, mm_p, mask, weights_p) -> list:
    tensors = [R_p, A_p, neg_p, mm_p, mask, weights_p]
    return [t.numpy().astype(np.float32).flatten().tolist() for t in tensors]


def _calibration_example():
    """One representative padded example spanning the real value range, used for
    both gen_settings' calibration and (as a template) for real gen_witness calls.
    Deliberately NOT random noise — chunk_mask must be a valid 0/1 mask (not
    arbitrary floats) and max_marks must be realistic (0 breaks the scoring
    head's sigmoid*max_marks bound in a way that isn't representative), so this
    pulls a real preprocessed example if one is available and falls back to a
    structurally-valid synthetic one otherwise."""
    from app.rcajx.padded_model import build_chunk_mask, pad_inputs

    data_path = os.path.join(os.path.dirname(HERE), "data", "train_embedded.pt")
    if os.path.exists(data_path):
        examples = torch.load(data_path, weights_only=False)
        if examples:
            ex = examples[0]
            weights = torch.ones(ex["R"].shape[0]) / ex["R"].shape[0]
            padded = pad_inputs(ex["R"], ex["A"], ex["negation_flags"], ex["max_marks"], criterion_weights=weights)
            return (
                padded["R_padded"], padded["A_padded"], padded["negation_flags_padded"],
                padded["max_marks_padded"], padded["chunk_mask"], padded["criterion_weights_padded"],
            )

    g = torch.Generator().manual_seed(0)
    R = torch.randn(MAX_CRITERIA, D_MODEL, generator=g)
    A = torch.randn(MAX_CHUNKS, D_MODEL, generator=g)
    neg = torch.randint(0, 2, (MAX_CRITERIA,), generator=g).float()
    mm = torch.rand(MAX_CRITERIA, generator=g) * 3 + 1
    w = torch.rand(MAX_CRITERIA, generator=g)
    mask = build_chunk_mask(MAX_CHUNKS, MAX_CHUNKS)
    return R, A, neg, mm, mask, w


def circuit_is_ready() -> bool:
    return all(
        os.path.exists(p)
        for p in [SETTINGS_PATH, COMPILED_PATH, SRS_PATH, VK_PATH, PK_PATH, MODEL_HASH_PATH]
    )


async def build_circuit(force: bool = False):
    """Full one-time setup: settings -> calibrate -> compile -> srs -> trusted
    setup. See module docstring — run this standalone
    (`python -m app.zk.rcajx_ezkl_pipeline`), NOT from API startup, ideally on a
    machine with plenty of RAM. Then copy CIRCUIT_DIR onto the demo machine."""
    if circuit_is_ready() and not force:
        return read_model_hash()

    if not os.path.exists(ONNX_PATH):
        export_onnx()

    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "hashed"
    run_args.output_visibility = "public"
    run_args.param_visibility = "fixed"

    ok = ezkl.gen_settings(ONNX_PATH, SETTINGS_PATH, py_run_args=run_args)
    assert ok, "gen_settings failed"

    flat = _flatten_padded_example(*_calibration_example())
    json.dump({"input_data": flat}, open(CAL_DATA_PATH, "w"))
    ezkl.calibrate_settings(CAL_DATA_PATH, ONNX_PATH, SETTINGS_PATH, target="resources")

    ok = ezkl.compile_circuit(ONNX_PATH, COMPILED_PATH, SETTINGS_PATH)
    assert ok, "compile_circuit failed"

    with open(SETTINGS_PATH) as f:
        logrows = json.load(f)["run_args"]["logrows"]
    print(f"[rcajx_ezkl_pipeline] circuit compiled at logrows={logrows} "
          f"(Zone2's old circuit was logrows=15 for scale) — expect setup() to be "
          f"significantly heavier; see module docstring if this OOMs.")

    await ezkl.get_srs(settings_path=SETTINGS_PATH, srs_path=SRS_PATH)

    ok = ezkl.setup(COMPILED_PATH, VK_PATH, PK_PATH, srs_path=SRS_PATH)
    assert ok, "setup failed"

    model_hash = hashlib.sha256(
        (
            _sha256_file(ONNX_PATH)
            + _sha256_file(SETTINGS_PATH)
            + _sha256_file(COMPILED_PATH)
            + _sha256_file(VK_PATH)
        ).encode()
    ).hexdigest()
    with open(MODEL_HASH_PATH, "w") as f:
        f.write(model_hash)

    return model_hash


def read_model_hash() -> str:
    with open(MODEL_HASH_PATH) as f:
        return f.read().strip()


def _hex_to_poseidon_le(hex_str: str) -> str:
    raw = hex_str[2:] if hex_str.startswith("0x") else hex_str
    return bytes.fromhex(raw)[::-1].hex()


def _commitment_from_witness(witness_path: str) -> str:
    with open(witness_path) as f:
        w = json.load(f)
    pretty = w["pretty_elements"]
    # All 6 inputs are private ("hashed") -> pretty["inputs"] has one list per
    # input tensor; concatenate them in declared order for one whole-input
    # commitment (same "hash everything, single global visibility" constraint
    # as Zone 2 — see zk/ezkl_pipeline.py's docstring for why per-input mixed
    # visibility isn't available in this EZKL version).
    field_elements = []
    for tensor_hexes in pretty["inputs"]:
        field_elements.extend(_hex_to_poseidon_le(h) for h in tensor_hexes)
    commitment = ezkl.poseidon_hash(field_elements)
    return commitment[0]


def _witness_input_json(R_p, A_p, neg_p, mm_p, mask, weights_p) -> dict:
    return {"input_data": _flatten_padded_example(R_p, A_p, neg_p, mm_p, mask, weights_p)}


def prove(padded_inputs: dict, proof_dir: str, tag: str) -> dict:
    """padded_inputs: dict with keys R_padded, A_padded, negation_flags_padded,
    max_marks_padded, chunk_mask, criterion_weights_padded (torch tensors, see
    app.rcajx.padded_model.pad_inputs). Returns dict with proof_path,
    witness_path, poseidon_commitment, proof_public_inputs, final_score —
    same contract as zk/ezkl_pipeline.py's prove()."""
    os.makedirs(proof_dir, exist_ok=True)
    payload = _witness_input_json(
        padded_inputs["R_padded"], padded_inputs["A_padded"],
        padded_inputs["negation_flags_padded"], padded_inputs["max_marks_padded"],
        padded_inputs["chunk_mask"], padded_inputs["criterion_weights_padded"],
    )

    input_path = os.path.join(proof_dir, f"{tag}_input.json")
    json.dump(payload, open(input_path, "w"))

    witness_path = os.path.join(proof_dir, f"{tag}_witness.json")
    ezkl.gen_witness(input_path, COMPILED_PATH, witness_path)
    poseidon_commitment = _commitment_from_witness(witness_path)

    proof_path = os.path.join(proof_dir, f"{tag}_proof.json")
    proof = ezkl.prove(witness_path, COMPILED_PATH, PK_PATH, proof_path, srs_path=SRS_PATH)
    assert proof is not None, "prove failed"

    with open(proof_path) as f:
        proof_json = json.load(f)

    pretty = proof_json["pretty_public_inputs"]
    rescaled_outputs = pretty["rescaled_outputs"]
    # outputs = [per_criterion_scores(10), final_score(1), attn_weights(n_heads*10*24), spread(10*n_heads)]
    # (order fixed by PaddedRCAJX.forward's return tuple / OUTPUT_NAMES in export_onnx.py — verified empirically, not just assumed)
    per_criterion_scores = [float(v) for v in rescaled_outputs[0]]
    final_score = float(rescaled_outputs[1][0])

    return {
        "proof_path": proof_path,
        "witness_path": witness_path,
        "poseidon_commitment": poseidon_commitment,
        "proof_public_inputs": {"rescaled_outputs": rescaled_outputs},
        "per_criterion_scores": per_criterion_scores,
        "final_score": final_score,
    }


def recompute_poseidon_commitment(padded_inputs: dict, tag: str) -> str:
    import tempfile

    payload = _witness_input_json(
        padded_inputs["R_padded"], padded_inputs["A_padded"],
        padded_inputs["negation_flags_padded"], padded_inputs["max_marks_padded"],
        padded_inputs["chunk_mask"], padded_inputs["criterion_weights_padded"],
    )
    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, f"{tag}_input.json")
        json.dump(payload, open(input_path, "w"))
        witness_path = os.path.join(tmp, f"{tag}_witness.json")
        ezkl.gen_witness(input_path, COMPILED_PATH, witness_path)
        return _commitment_from_witness(witness_path)


def verify(proof_path: str) -> dict:
    try:
        ok = ezkl.verify(proof_path, SETTINGS_PATH, VK_PATH, srs_path=SRS_PATH)
    except Exception as e:
        return {"verified": False, "error": str(e)}

    with open(proof_path) as f:
        proof_json = json.load(f)

    instances = proof_json.get("instances") or [[]]
    embedded_commitment = instances[0][0] if instances and instances[0] else None

    return {
        "verified": bool(ok),
        "pretty_public_inputs": proof_json.get("pretty_public_inputs"),
        "embedded_poseidon_commitment": embedded_commitment,
    }


if __name__ == "__main__":
    h = asyncio.run(build_circuit())
    print("rcajx_model_hash:", h)
