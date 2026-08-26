"""
Fine-tunes the frozen BGE-small encoder on triplets mined from data/train + data/test.

NOT run as part of the hardening implementation pass — run manually:
    python src/finetune_encoder.py
Expected wall-clock time on an RTX 3050: single-digit minutes for ~500-1000 triplets
(per rcaj-x-hardening-plan/03 Part B).

After it finishes, the manual next steps are:
  1. Point src/preprocessing.py's `encoder` at checkpoints/bge_small_finetuned instead of
     "BAAI/bge-small-en-v1.5".
  2. Re-run: python src/preprocessing.py && python src/train.py && python src/benchmark.py
  3. Compare negation_flipped / confidently_wrong / paraphrase MAE before vs. after in
     results/benchmark_report.md — the plan expects the gain to concentrate there.
"""
import json
import os
import random
from pathlib import Path

import spacy
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

ML_WORKER_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ML_WORKER_ROOT / "data"
CHECKPOINT_DIR = ML_WORKER_ROOT / "app" / "models" / "rcajx"

_nlp = spacy.load("en_core_web_sm")


def _first_sentence(text: str) -> str:
    doc = _nlp(text)
    for sent in doc.sents:
        if sent.text.strip():
            return sent.text.strip()
    return text


def _load_examples(split: str) -> list[dict]:
    rows = []
    split_dir = DATA_DIR / split
    for fname in sorted(os.listdir(split_dir)):
        if fname.endswith(".json"):
            with open(split_dir / fname) as f:
                rows.append(json.load(f))
    return rows


def build_triplets(train_rows: list[dict], test_rows: list[dict], rubrics: list[dict]) -> list[InputExample]:
    """
    Anchor = criterion text.
    Positive = first sentence of the question's full-credit train anchor (an approximation,
    same coarse style as preprocessing.py's existing negation_flags heuristic — not treated
    as exact ground truth).
    Hard negatives (priority order, per rcaj-x-hardening-plan/03 Part B):
      1. negation_flipped test variants for the same question_id.
      2. confidently_wrong test variants for the same question_id.
    Easy negative: one sentence from an answer to a different question_id, to stabilize
    early training.
    """
    train_by_qid: dict[str, list[dict]] = {}
    for r in train_rows:
        train_by_qid.setdefault(r["question_id"], []).append(r)

    test_by_qid_variant: dict[tuple[str, str], list[dict]] = {}
    for r in test_rows:
        test_by_qid_variant.setdefault((r["question_id"], r["variant_type"]), []).append(r)

    rubric_map = {r["question_id"]: r for r in rubrics}
    all_qids = list(train_by_qid.keys())
    triplets = []

    for qid, anchors in train_by_qid.items():
        criteria = rubric_map[qid]["criteria"]
        full_correct = next((a for a in anchors if a.get("style") == "x_type_full_correct"), anchors[0])
        positive_sentence = _first_sentence(full_correct["answer_text"])

        neg_flip = test_by_qid_variant.get((qid, "negation_flipped"), [])
        conf_wrong = test_by_qid_variant.get((qid, "confidently_wrong"), [])
        hard_negatives = [_first_sentence(r["answer_text"]) for r in neg_flip]
        hard_negatives += [_first_sentence(r["answer_text"]) for r in conf_wrong]
        if not hard_negatives:
            continue

        for c in criteria:
            for neg in hard_negatives:
                triplets.append(InputExample(texts=[c["text"], positive_sentence, neg]))

            other_qid = random.choice([q for q in all_qids if q != qid])
            other_anchor = random.choice(train_by_qid[other_qid])
            easy_neg = _first_sentence(other_anchor["answer_text"])
            triplets.append(InputExample(texts=[c["text"], positive_sentence, easy_neg]))

    return triplets


def main():
    with open(DATA_DIR / "raw" / "rubrics.json") as f:
        rubrics = json.load(f)
    train_rows = _load_examples("train")
    test_rows = _load_examples("test")

    triplets = build_triplets(train_rows, test_rows, rubrics)
    print(f"Built {len(triplets)} triplets.")

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    train_dataloader = DataLoader(triplets, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    output_path = CHECKPOINT_DIR / "bge_small_finetuned"
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=int(0.1 * len(train_dataloader)),
        optimizer_params={"lr": 3e-5},
        output_path=str(output_path),
        show_progress_bar=True,
    )
    print(f"Fine-tuned encoder saved to {output_path}")
    print("Next: re-run preprocessing (app/rcajx/preprocessing.py) + train.py, then")
    print("benchmark.py, and compare negation_flipped / confidently_wrong / paraphrase")
    print("MAE before vs. after.")


if __name__ == "__main__":
    main()
