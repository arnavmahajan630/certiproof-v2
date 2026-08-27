"""Ingests the pre-filtered real-data JSONL (CBSE + ASAP-SAS + Mohler, combined
and quality-checked upstream — no empty answers, no score/max_score anomalies,
no duplicate answer texts) into this repo's data/{train,test}/*.json schema.

Unlike ingest_external_dataset.py (raw CSV/TSV, one column set per dataset),
this source is JSONL with a uniform schema across all three sources and
already carries per-question text + a holistic score, so it's ingested
directly rather than through the CSV path. Each row maps its holistic score
onto a single synthetic criterion per question_id, same convention as
ingest_external_dataset.py (see that file's docstring for why: none of these
three source datasets ship multi-criterion rubrics).

Split: rows are grouped by question_id (sorted by id for determinism) and the
last ~20% of each group goes to data/test/, the rest to data/train/ — held-out
per question, not just per row, so benchmark.py measures generalization within
each question rather than memorization. Groups of size 1 go entirely to train.

Usage:
    python training/ingest_condensed_jsonl.py --jsonl /path/to/asap-mohler-condensed.jsonl
"""
import argparse
import json
from pathlib import Path

ML_WORKER_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ML_WORKER_ROOT / "data"
RUBRICS_PATH = DATA_DIR / "raw" / "rubrics.json"


def _load_rubrics() -> list[dict]:
    if RUBRICS_PATH.exists():
        with open(RUBRICS_PATH) as f:
            return json.load(f)
    return []


def _save_rubrics(rubrics: list[dict]):
    RUBRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUBRICS_PATH, "w") as f:
        json.dump(rubrics, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True, help="path to the condensed JSONL")
    ap.add_argument("--test-frac", type=float, default=0.2, help="fraction of each question's rows held out to test")
    args = ap.parse_args()

    with open(args.jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    by_question: dict[str, list[dict]] = {}
    for r in rows:
        by_question.setdefault(r["question_id"], []).append(r)

    rubrics = _load_rubrics()
    rubric_map = {r["question_id"]: r for r in rubrics}

    (DATA_DIR / "train").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "test").mkdir(parents=True, exist_ok=True)

    written = {"train": 0, "test": 0}
    for question_id, group in by_question.items():
        group.sort(key=lambda r: r["id"])

        if len(group) >= 3:
            n_test = max(1, round(len(group) * args.test_frac))
        else:
            n_test = 0
        n_test = min(n_test, len(group) - 1)
        split_index = len(group) - n_test

        if question_id not in rubric_map:
            first = group[0]
            max_score = first["human_scores"]["max_score"]
            rubric = {
                "question_id": question_id,
                "question_text": first["question"],
                "subject": first["subject"],
                "criteria": [
                    {
                        "criterion_id": f"{question_id}_c1",
                        "text": " ".join(first.get("criteria") or ["Overall answer quality (holistic score from source dataset)."]),
                        "max_marks": max_score,
                    }
                ],
            }
            rubric_map[question_id] = rubric
            rubrics.append(rubric)

        criterion_id = f"{question_id}_c1"
        for i, row in enumerate(group):
            split = "test" if i >= split_index else "train"
            answer_id = f"{question_id}_{split}_{i:03d}"
            record = {
                "answer_id": answer_id,
                "question_id": question_id,
                "derived_from_train_id": None,
                "variant_type": row.get("variant_type", "external_real_data"),
                "answer_text": row["answer_text"],
                "human_reviewed": True,
                "ai_generated": False,
                "human_scores": {criterion_id: row["human_scores"]["holistic"]},
                "data_source": row["source"],
            }
            with open(DATA_DIR / split / f"{answer_id}.json", "w") as out:
                json.dump(record, out, indent=2)
            written[split] += 1

    _save_rubrics(rubrics)
    print(f"Wrote {written['train']} train / {written['test']} test examples across {len(by_question)} question ids")
    print(f"Rubrics file now has {len(rubrics)} question ids total -> {RUBRICS_PATH}")


if __name__ == "__main__":
    main()
