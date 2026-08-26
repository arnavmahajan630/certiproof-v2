"""Eval 1.5 Poseidon migration — Step 1 smoke test (Tier A).

Toy 2-input model: one private (hashed/private -> Poseidon commitment public),
one public. Confirms whether EZKL 23.0.5's Python binding can differentiate
visibility per named ONNX input, or whether input_visibility is unavoidably
uniform across all inputs (which would force Tier B).
"""
import json
import os

import ezkl
import torch
import torch.nn as nn

WORKDIR = os.path.join(os.path.dirname(__file__), ".poseidon_smoke")
os.makedirs(WORKDIR, exist_ok=True)


class TwoInput(nn.Module):
    def forward(self, priv, pub):
        x = torch.cat([priv, pub], dim=-1)
        return x.sum(dim=-1, keepdim=True)


def path(name):
    return os.path.join(WORKDIR, name)


def try_visibility_value(value, label):
    print(f"\n=== Attempt: input_visibility = {value!r} ({label}) ===")
    onnx_path = path("network.onnx")
    settings_path = path(f"settings_{label}.json")

    run_args = ezkl.PyRunArgs()
    try:
        run_args.input_visibility = value
    except Exception as e:
        print(f"  FAILED to set input_visibility: {e}")
        return None
    run_args.output_visibility = "public"
    run_args.param_visibility = "fixed"

    try:
        ok = ezkl.gen_settings(onnx_path, settings_path, py_run_args=run_args)
    except Exception as e:
        print(f"  gen_settings raised: {e}")
        return None
    if not ok:
        print("  gen_settings returned False")
        return None

    settings = json.load(open(settings_path))
    print(f"  gen_settings OK. run_args.input_visibility (as stored) = {settings['run_args']['input_visibility']}")
    print(f"  model_input_scales = {settings['model_input_scales']}")
    print(f"  input_types = {settings['input_types']}")
    return settings


def main():
    torch.manual_seed(0)
    model = TwoInput()
    model.eval()

    dummy_priv = torch.rand(1, 2)
    dummy_pub = torch.rand(1, 2)

    onnx_path = path("network.onnx")
    torch.onnx.export(
        model,
        (dummy_priv, dummy_pub),
        onnx_path,
        input_names=["priv_input", "pub_input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    print(f"[export] 2-input ONNX written -> {onnx_path}")

    # Attempt 1: single string, see what gen_settings does with 2 inputs.
    s1 = try_visibility_value("hashed/private", "uniform_hashed_private")

    # Attempt 2: does the Rust binding accept a list (undocumented but worth trying,
    # since PyRunArgs' setter might coerce more than help() documents)?
    s2 = try_visibility_value(["hashed/private", "public"], "list_mixed")

    # Attempt 3: comma-separated string (another plausible undocumented convention).
    s3 = try_visibility_value("hashed/private,public", "csv_mixed")

    print("\n=== Summary ===")
    print(f"uniform hashed/private on both inputs: {'OK' if s1 else 'FAILED'}")
    print(f"list [hashed/private, public]:         {'OK' if s2 else 'FAILED'}")
    print(f"csv 'hashed/private,public':            {'OK' if s3 else 'FAILED'}")

    if s2 or s3:
        winning = s2 or s3
        ins = winning["model_input_scales"]
        print(f"\nPossible per-input differentiation signal: model_input_scales has {len(ins)} entries for 2 inputs.")
    else:
        print("\nNo per-input mixed-visibility mechanism found via PyRunArgs.input_visibility.")
        print("Tier A (mixed visibility) is NOT supported by this binding -> fall back to Tier B.")


if __name__ == "__main__":
    main()
