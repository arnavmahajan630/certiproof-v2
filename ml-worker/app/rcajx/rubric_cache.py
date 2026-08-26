import hashlib
import json
import os
from pathlib import Path

import torch

from .preprocessing import embed_criteria

ML_WORKER_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = str(ML_WORKER_ROOT / "data" / "rubric_cache")


def _rubric_hash(rubric: dict) -> str:
    canonical = json.dumps(
        {
            "question_id": rubric["question_id"],
            "criteria": [
                {"criterion_id": c["criterion_id"], "text": c["text"], "max_marks": c["max_marks"]}
                for c in rubric["criteria"]
            ],
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _cache_path(question_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{question_id}.json")


def get_rubric_R(rubric: dict, mode: str = "dev") -> torch.Tensor:
    """
    mode='dev': always recomputes R from the rubric's criteria text (current/default behavior).
    mode='locked': caches R on disk per rubric hash. A second call for the same question_id
    with different criteria text raises ValueError rather than silently re-embedding —
    a locked rubric's identity is its hash, and that must not drift underneath a cached score.
    """
    criteria_texts = [c["text"] for c in rubric["criteria"]]

    if mode == "dev":
        return embed_criteria(criteria_texts)

    if mode != "locked":
        raise ValueError(f"Unknown rubric mode: {mode!r}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    h = _rubric_hash(rubric)
    path = _cache_path(rubric["question_id"])

    if os.path.exists(path):
        with open(path, "r") as f:
            cached = json.load(f)
        if cached["rubric_hash"] != h:
            raise ValueError(
                f"Rubric hash mismatch for question_id={rubric['question_id']!r}: "
                f"cached rubric text differs from the text submitted now. "
                f"A locked rubric ID cannot be silently re-embedded with different content."
            )
        return torch.tensor(cached["R"], dtype=torch.float32)

    R = embed_criteria(criteria_texts)
    with open(path, "w") as f:
        json.dump({"question_id": rubric["question_id"], "rubric_hash": h, "R": R.tolist()}, f)
    return R
