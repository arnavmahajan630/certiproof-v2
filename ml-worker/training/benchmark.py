import torch
import json
import pandas as pd
import os
import sys
from pathlib import Path

ML_WORKER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_WORKER_ROOT))

from app.rcajx.model import RCAJ_X
from report import generate_benchmark_report
from app.rcajx.explain import generate_explanation, check_explanation_score_consistency

DATA_DIR = ML_WORKER_ROOT / "data"
RESULTS_DIR = ML_WORKER_ROOT / "results"
CHECKPOINT_DIR = ML_WORKER_ROOT / "app" / "models" / "rcajx"

def run_benchmark():
    print("Running benchmark...")
    checkpoint = torch.load(CHECKPOINT_DIR / "rcaj_x_best.pt", weights_only=False)
    model_config = {k: v for k, v in checkpoint["config"].items() if k in ["n_heads", "d_k", "d_v", "hidden"]}
    model = RCAJ_X(**model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test_examples = torch.load(DATA_DIR / "test_embedded.pt", weights_only=False)
    train_examples = torch.load(DATA_DIR / "train_embedded.pt", weights_only=False)

    manifest = {}
    manifest_path = DATA_DIR / "dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    # To calculate paired delta, we need predictions for the source train examples
    train_preds = {}
    with torch.no_grad():
        for ex in train_examples:
            out = model(ex["R"], ex["A"], ex["negation_flags"], ex["max_marks"])
            pred = out["per_criterion_scores"].tolist()
            train_preds[ex["answer_id"]] = sum(pred) / len(pred) if pred else 0

    results = []
    consistency_rows = []
    has_placeholder = False

    with torch.no_grad():
        for ex in test_examples:
            if ex.get("placeholder"):
                has_placeholder = True

            out = model(ex["R"], ex["A"], ex["negation_flags"], ex["max_marks"])
            pred_scores = out["per_criterion_scores"].tolist()

            assert all(0 <= p <= m for p, m in zip(pred_scores, ex["max_marks"].tolist())), \
                f"Out-of-bounds prediction for {ex['answer_id']}"

            # calculate MAE and within_1_mark per row
            human = list(ex["human_scores"].values())
            errors = [abs(h - p) for h, p in zip(human, pred_scores)]
            mae = sum(errors) / len(pred_scores) if pred_scores else 0
            within_1_mark = all(e <= 1.0 for e in errors)

            # calculate paired delta
            source_id = ex["derived_from_train_id"]
            test_pred_mean = sum(pred_scores) / len(pred_scores) if pred_scores else 0
            source_pred_mean = train_preds.get(source_id, 0)
            paired_delta = test_pred_mean - source_pred_mean

            # calculate met/not-met
            # threshold: >= 50% of max_marks. Since our max_marks vary, we'd need them.
            # We can get max_marks from the rubric, but for simplicity we assume max=2 here, or we fetch from rubric.
            # Let's fetch from rubric:
            with open(DATA_DIR / "raw" / "rubrics.json", "r") as f:
                rubrics = json.load(f)
            rubric_map = {r["question_id"]: r for r in rubrics}
            criteria = rubric_map[ex["question_id"]]["criteria"]
            max_marks_list = [c["max_marks"] for c in criteria]

            y_true_met = [h >= (m * 0.5) for h, m in zip(human, max_marks_list)]
            y_pred_met = [p >= (m * 0.5) for p, m in zip(pred_scores, max_marks_list)]

            results.append({
                "answer_id": ex["answer_id"],
                "question_id": ex["question_id"],
                "variant_type": ex["variant_type"],
                "derived_from_train_id": source_id,
                "human_scores": ex["human_scores"],
                "pred_scores": pred_scores,
                "mean_spread": out["spread"].mean().item(),
                "mae": mae,
                "within_1_mark": within_1_mark,
                "paired_delta": paired_delta,
                "y_true_met": y_true_met,
                "y_pred_met": y_pred_met,
                "placeholder": ex.get("placeholder", False)
            })

            explanations = generate_explanation(
                model, ex["R"], ex["A"], ex["chunks"], criteria, ex["negation_flags"]
            )
            for exp in explanations:
                for flag in check_explanation_score_consistency(exp):
                    consistency_rows.append({
                        "answer_id": ex["answer_id"],
                        "question_id": ex["question_id"],
                        "criterion_id": exp["criterion_id"],
                        "variant_type": ex["variant_type"],
                        "flag": flag,
                        "score": exp["score"],
                        "max_marks": exp["max_marks"],
                    })

    df = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)
    print(f"Benchmark results saved to {RESULTS_DIR / 'benchmark_results.csv'}")

    consistency_df = pd.DataFrame(consistency_rows, columns=[
        "answer_id", "question_id", "criterion_id", "variant_type", "flag", "score", "max_marks"
    ])
    consistency_df.to_csv(RESULTS_DIR / "consistency_flags.csv", index=False)
    print(f"Consistency flags saved to {RESULTS_DIR / 'consistency_flags.csv'} ({len(consistency_df)} flagged)")

    generate_benchmark_report(df, has_placeholder)

if __name__ == "__main__":
    run_benchmark()
