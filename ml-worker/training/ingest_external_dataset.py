"""Converts an external labeled dataset (e.g. ASAP-SAS, Mohler) into this repo's
internal data/{train,test}/*.json schema, so it can sit alongside the existing
synthetic set and be picked up by preprocessing.py / train.py unchanged.

This is net-new — RCAj-x-mini-v1's old ingest_data.py targeted a since-removed
`datainj/` format and is not a working reference for this. ASAP-SAS and Mohler
are both HOLISTIC single-score datasets (one score per answer), not multi-
criterion rubrics like this system's data/raw/rubrics.json expects. There is no
generic way to auto-split a holistic score into per-criterion scores, so this
script's default (--single-criterion) maps the whole external score onto ONE
synthetic criterion per question — defensible and immediately usable, but you
may want to hand-author a real multi-criterion rubric for these questions later
and re-derive per-criterion labels some other way. That's a data-modeling call,
not something this script can make for you.

Expected input: a CSV with (at minimum) columns for a per-question/prompt id, the
answer text, and a numeric score. Exact column names vary by dataset release --
pass them via --id-col/--text-col/--score-col rather than hardcoding, since we
don't know which exact CSV export you'll have tomorrow.

Usage:
    python training/ingest_external_dataset.py \
        --csv /path/to/asap_sas_train.csv \
        --id-col EssaySet --text-col EssayText --score-col Score1 \
        --max-score 3 --split train --source-name asap_sas

Writes one data/{split}/{source_name}_{question_id}_{n}.json per row, in the
existing schema (see app/rcajx/preprocessing.py's preprocess_dataset / data/raw
/rubrics.json's criteria shape) and appends one synthetic single-criterion
rubric entry per distinct question id to data/raw/rubrics.json (skipping ids
that already exist there — never overwrites an existing rubric).
"""
import argparse
import csv
import json
import os
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
    ap.add_argument("--csv", required=True, help="path to the external dataset CSV/TSV")
    ap.add_argument("--delimiter", default=",", help="field delimiter (use $'\\t' for ASAP-SAS's native TSV export)")
    ap.add_argument("--id-col", required=True, help="column holding the question/prompt id")
    ap.add_argument("--text-col", required=True, help="column holding the answer text")
    ap.add_argument("--score-col", required=True, help="column holding the numeric score")
    ap.add_argument("--max-score", type=float, required=True, help="max possible score for --score-col (this dataset's scale)")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--source-name", required=True, help="short tag prefixed to generated ids, e.g. 'asap_sas' or 'mohler'")
    ap.add_argument("--single-criterion", action="store_true", default=True,
                     help="map the holistic score onto one synthetic criterion per question (default, and currently the only supported mode)")
    args = ap.parse_args()

    rubrics = _load_rubrics()
    rubric_map = {r["question_id"]: r for r in rubrics}

    split_dir = DATA_DIR / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    written = 0
    with open(args.csv, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=args.delimiter)
        for row in reader:
            raw_qid = row[args.id_col]
            question_id = f"{args.source_name}_{raw_qid}"
            answer_text = row[args.text_col].strip()
            if not answer_text:
                continue
            try:
                score = float(row[args.score_col])
            except (ValueError, KeyError):
                continue

            criterion_id = f"{question_id}_c1"
            if question_id not in rubric_map:
                rubric = {
                    "question_id": question_id,
                    "question_text": f"[{args.source_name}] prompt {raw_qid} (holistic score, single synthetic criterion)",
                    "subject": args.source_name,
                    "criteria": [
                        {"criterion_id": criterion_id, "text": "Overall answer quality (holistic score from source dataset)", "max_marks": args.max_score}
                    ],
                }
                rubric_map[question_id] = rubric
                rubrics.append(rubric)

            counts[question_id] = counts.get(question_id, 0) + 1
            n = counts[question_id]
            answer_id = f"{question_id}_{args.split}_{n:03d}"

            record = {
                "answer_id": answer_id,
                "question_id": question_id,
                "derived_from_train_id": None,
                "variant_type": "external_real_data",
                "answer_text": answer_text,
                "human_reviewed": True,
                "ai_generated": False,
                "human_scores": {criterion_id: score},
                "data_source": args.source_name,
            }
            with open(split_dir / f"{answer_id}.json", "w") as out:
                json.dump(record, out, indent=2)
            written += 1

    _save_rubrics(rubrics)
    print(f"Wrote {written} {args.split} examples across {len(counts)} question ids from {args.csv}")
    print(f"Rubrics file now has {len(rubrics)} question ids total -> {RUBRICS_PATH}")
    print("Next: re-run preprocessing (app/rcajx/preprocessing.py) + training/train.py "
          "to fold this into the model, or use training/retrain_and_deploy.sh for the full chain.")


if __name__ == "__main__":
    main()
