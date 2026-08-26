"""EZKL toolchain smoke test — trivial Linear(4,1) full round trip.
Gate: must pass before any real model work begins (per build plan Hour 0).
"""
import asyncio
import json
import os

import ezkl
import torch
import torch.nn as nn

WORKDIR = os.path.join(os.path.dirname(__file__), ".smoke")
os.makedirs(WORKDIR, exist_ok=True)


class Trivial(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(x)


def path(name):
    return os.path.join(WORKDIR, name)


async def main():
    torch.manual_seed(0)
    model = Trivial()
    model.eval()

    dummy_input = torch.rand(1, 4)

    onnx_path = path("network.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    print("[1/8] ONNX export ok")

    data_path = path("input.json")
    data = dict(input_data=[dummy_input.detach().numpy().reshape(-1).tolist()])
    json.dump(data, open(data_path, "w"))
    print("[2/8] input.json written")

    settings_path = path("settings.json")
    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "public"
    run_args.output_visibility = "public"
    run_args.param_visibility = "fixed"

    ok = ezkl.gen_settings(onnx_path, settings_path, py_run_args=run_args)
    assert ok, "gen_settings failed"
    print("[3/8] settings generated")

    cal_path = path("cal_data.json")
    json.dump(data, open(cal_path, "w"))
    ezkl.calibrate_settings(cal_path, onnx_path, settings_path, target="resources")
    print("[4/8] settings calibrated")

    compiled_path = path("network.compiled")
    ok = ezkl.compile_circuit(onnx_path, compiled_path, settings_path)
    assert ok, "compile_circuit failed"
    print("[5/8] circuit compiled")

    srs_path = path("kzg.srs")
    await ezkl.get_srs(settings_path=settings_path, srs_path=srs_path)
    print("[6/8] SRS fetched")

    vk_path = path("vk.key")
    pk_path = path("pk.key")
    ok = ezkl.setup(compiled_path, vk_path, pk_path, srs_path=srs_path)
    assert ok, "setup failed"
    print("[7/8] trusted setup ok")

    witness_path = path("witness.json")
    ezkl.gen_witness(data_path, compiled_path, witness_path)
    print("      witness generated")

    proof_path = path("proof.json")
    proof = ezkl.prove(
        witness_path,
        compiled_path,
        pk_path,
        proof_path,
        srs_path=srs_path,
    )
    assert proof is not None, "prove failed"
    print("[8/8] proof generated")

    verified = ezkl.verify(proof_path, settings_path, vk_path, srs_path=srs_path)
    assert verified, "verify failed on genuine proof"
    print("VERIFY (genuine): PASS")

    # Perturbation sanity: tamper the public input in the proof file and confirm verify fails.
    with open(proof_path) as f:
        proof_json = json.load(f)
    tampered_path = path("proof_tampered.json")
    tampered = json.loads(json.dumps(proof_json))
    if tampered.get("instances"):
        inst = tampered["instances"][0][0]
        tampered["instances"][0][0] = inst[:-1] + ("0" if inst[-1] != "0" else "1")
    json.dump(tampered, open(tampered_path, "w"))

    try:
        tampered_ok = ezkl.verify(tampered_path, settings_path, vk_path, srs_path=srs_path)
    except Exception:
        tampered_ok = False
    print(f"VERIFY (tampered): {'FAIL as expected' if not tampered_ok else 'UNEXPECTEDLY PASSED'}")
    assert not tampered_ok, "tampered proof must not verify"

    print("\nSMOKE TEST: ALL GREEN")


if __name__ == "__main__":
    asyncio.run(main())
