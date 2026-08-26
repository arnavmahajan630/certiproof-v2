"""zone1_model_hash / zone2_model_hash equivalents for RCAJ-X.

The DB schema (gateway/src/db/schema.sql) keeps its existing zone1_model_hash /
zone2_model_hash columns unchanged (Phase 4 of the integration plan) — only their
meaning is repointed:
  - zone1_model_hash -> hash of the fine-tuned BGE-small encoder weight bytes
    (checkpoints/bge_small_finetuned), replacing the old bge-reranker-base hash.
  - zone2_model_hash -> hash of the RCAJ_X attention+scoring-head weight bytes
    (rcaj_x_best.pt) + its config, replacing the old ScoreAggregator MLP hash.

Weight-file-bytes hashing pattern ported from app/zone1/encoder.py's
_weight_file_hash()/model_hash() — real bytes, not just a path/name string, so
swapping in a different checkpoint under the same filename is still a
certification-breaking change.
"""
import glob
import hashlib
import json
import os

import torch

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "rcajx")
ENCODER_DIR = os.path.join(MODELS_DIR, "bge_small_finetuned")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "rcaj_x_best.pt")
BASE_ENCODER_NAME = "BAAI/bge-small-en-v1.5"


def _hash_files(paths: list[str]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


def encoder_model_hash() -> str:
    """zone1_model_hash: hash of the encoder's weight files. Prefers the
    fine-tuned checkpoint (bge_small_finetuned/, gitignored — see
    preprocessing.py's fallback comment); if that hasn't been produced yet on
    this machine, falls back to hashing the public base model's weight bytes
    from the Hugging Face cache (same one app.rcajx.preprocessing just loaded
    into), matching that module's own fallback so this never crashes on a
    fresh clone. The two cases are distinguishable in the hash payload, so a
    later switch from base to fine-tuned is correctly treated as a
    certification-breaking change."""
    if os.path.isdir(ENCODER_DIR):
        candidates = sorted(glob.glob(os.path.join(ENCODER_DIR, "*.safetensors"))) or sorted(
            glob.glob(os.path.join(ENCODER_DIR, "pytorch_model*.bin"))
        )
        if not candidates:
            raise RuntimeError(f"{ENCODER_DIR} exists but has no weight file")
        payload = json.dumps(
            {"encoder_source": "bge_small_finetuned", "weight_file_hash": _hash_files(candidates)},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    from huggingface_hub import snapshot_download

    snapshot_dir = snapshot_download(BASE_ENCODER_NAME, local_files_only=True)
    candidates = sorted(glob.glob(os.path.join(snapshot_dir, "*.safetensors"))) or sorted(
        glob.glob(os.path.join(snapshot_dir, "pytorch_model*.bin"))
    )
    if not candidates:
        raise RuntimeError(f"no weight file found in HF cache for {BASE_ENCODER_NAME} at {snapshot_dir}")
    payload = json.dumps(
        {"encoder_source": BASE_ENCODER_NAME, "weight_file_hash": _hash_files(candidates)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def scoring_model_hash() -> str:
    """zone2_model_hash: hash of the RCAJ_X checkpoint's state_dict bytes + config
    (n_heads/d_k/d_v/hidden — the architecture-defining fields, not training
    hyperparams like lr/weight_decay which don't affect inference)."""
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    arch_config = {k: v for k, v in checkpoint["config"].items() if k in ["n_heads", "d_k", "d_v", "hidden"]}
    state_dict_bytes = hashlib.sha256()
    for key in sorted(checkpoint["state_dict"].keys()):
        state_dict_bytes.update(key.encode())
        state_dict_bytes.update(checkpoint["state_dict"][key].numpy().tobytes())
    payload = json.dumps(
        {"config": arch_config, "state_dict_hash": state_dict_bytes.hexdigest()},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
