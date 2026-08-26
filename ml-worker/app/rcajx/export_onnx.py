"""Export the trained RCAJ_X model (wrapped for fixed shapes) to ONNX and run a
PyTorch-vs-ONNXRuntime parity test. Net-new — RCAj-x-mini-v1 has no ONNX export;
CertiProof's old zone2/export_onnx.py is the closest precedent (static shapes,
parity test, edge cases), followed here.
"""
import os

import numpy as np
import onnxruntime as ort
import torch

from app.rcajx.model import RCAJ_X
from app.rcajx.padded_model import D_MODEL, MAX_CHUNKS, MAX_CRITERIA, PaddedRCAJX, build_chunk_mask, pad_rows, pad_vec

HERE = os.path.dirname(__file__)  # ml-worker/app/rcajx
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "app", "models", "rcajx")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "rcaj_x_best.pt")
ONNX_PATH = os.path.join(MODELS_DIR, "rcajx.onnx")

TOLERANCE = 1e-4

INPUT_NAMES = [
    "R_padded", "A_padded", "negation_flags_padded", "max_marks_padded",
    "chunk_mask", "criterion_weights_padded",
]
OUTPUT_NAMES = ["per_criterion_scores", "final_score", "attn_weights", "spread"]


def load_model() -> RCAJ_X:
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model_config = {k: v for k, v in checkpoint["config"].items() if k in ["n_heads", "d_k", "d_v", "hidden"]}
    model = RCAJ_X(**model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _dummy_inputs(n_criteria: int, n_chunks: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n_criteria, D_MODEL, generator=g)
    A = torch.randn(n_chunks, D_MODEL, generator=g)
    neg = torch.randint(0, 2, (n_criteria,), generator=g).float()
    max_marks = (torch.rand(n_criteria, generator=g) * 3 + 1)
    weights = torch.rand(n_criteria, generator=g)
    return (
        pad_rows(R, MAX_CRITERIA),
        pad_rows(A, MAX_CHUNKS),
        pad_vec(neg, MAX_CRITERIA),
        pad_vec(max_marks, MAX_CRITERIA),
        build_chunk_mask(n_chunks),
        pad_vec(weights, MAX_CRITERIA),
    )


def export():
    model = load_model()
    wrapper = PaddedRCAJX(model)
    wrapper.eval()

    dummy = _dummy_inputs(MAX_CRITERIA, MAX_CHUNKS)
    torch.onnx.export(
        wrapper,
        dummy,
        ONNX_PATH,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=17,
        dynamo=False,
        dynamic_axes=None,  # static shapes throughout — non-negotiable for EZKL
    )
    print(f"exported ONNX -> {ONNX_PATH}")


def parity_test():
    model = load_model()
    wrapper = PaddedRCAJX(model)
    wrapper.eval()
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]

    cases = [
        _dummy_inputs(3, 5, seed=1),
        _dummy_inputs(1, 1, seed=2),
        _dummy_inputs(10, 24, seed=3),  # no padding at all
        _dummy_inputs(10, 1, seed=4),   # max criteria, min chunks
        _dummy_inputs(1, 24, seed=5),   # min criteria, max chunks
    ]

    max_diff = 0.0
    with torch.no_grad():
        for i, case in enumerate(cases):
            torch_out = wrapper(*case)
            onnx_in = {name: t.numpy().astype(np.float32) for name, t in zip(input_names, case)}
            onnx_out = sess.run(None, onnx_in)
            for t_out, o_out in zip(torch_out, onnx_out):
                diff = float(np.max(np.abs(t_out.numpy() - o_out)))
                max_diff = max(max_diff, diff)
                assert diff < TOLERANCE, f"parity mismatch at case {i}: diff={diff}"

    print(f"parity test PASSED across {len(cases)} cases (incl. no-padding, min/max shapes). max_diff={max_diff:.2e}")


if __name__ == "__main__":
    export()
    parity_test()
