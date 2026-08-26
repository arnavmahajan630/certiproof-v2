import pandas as pd
from sklearn.metrics import precision_score, recall_score
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

def generate_benchmark_report(df, has_placeholder):
    print("Generating benchmark report...")
    
    report_lines = []
    if has_placeholder:
        report_lines.append("⚠️ PLACEHOLDER DATA — NOT A REAL RESULT\n")
    
    report_lines.append("# RCAJ-X Benchmark Report\n")
    
    # 1. Overall accuracy
    overall_mae = df["mae"].mean()
    overall_within_1 = df["within_1_mark"].mean() * 100
    
    # Flatten y_true and y_pred
    all_y_true = [item for sublist in df["y_true_met"] for item in sublist]
    all_y_pred = [item for sublist in df["y_pred_met"] for item in sublist]
    
    overall_precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    overall_recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    
    report_lines.append("## Overall Metrics")
    report_lines.append(f"- **Mean Absolute Error (MAE):** {overall_mae:.3f}")
    report_lines.append(f"- **% Within 1 Mark:** {overall_within_1:.1f}%")
    report_lines.append(f"- **Precision (Criterion Met):** {overall_precision:.3f}")
    report_lines.append(f"- **Recall (Criterion Met):** {overall_recall:.3f}\n")
    
    # 2. Per-variant_type breakdown
    report_lines.append("## Accuracy by Variant Type")
    summary = df.groupby("variant_type").agg(
        mean_mae=("mae", "mean"),
        pct_within_1=("within_1_mark", "mean"),
        mean_spread=("mean_spread", "mean"),
        n=("answer_id", "count"),
    ).reset_index()
    
    summary["pct_within_1"] = summary["pct_within_1"] * 100
    
    # Add Precision/Recall per category
    precisions = []
    recalls = []
    for vt in summary["variant_type"]:
        sub_df = df[df["variant_type"] == vt]
        sub_y_true = [item for sublist in sub_df["y_true_met"] for item in sublist]
        sub_y_pred = [item for sublist in sub_df["y_pred_met"] for item in sublist]
        precisions.append(precision_score(sub_y_true, sub_y_pred, zero_division=0))
        recalls.append(recall_score(sub_y_true, sub_y_pred, zero_division=0))
        
    summary["precision"] = precisions
    summary["recall"] = recalls
    
    report_lines.append(summary.to_markdown(index=False, floatfmt=".3f"))
    report_lines.append("\n")
    
    # 3. Paired-delta table
    report_lines.append("## Paired Delta Analysis")
    expected_directions = {
        "paraphrase": "~0",
        "scattered_evidence": "~0",
        "diffuse_padded": "~0 to slight negative",
        "partial_credit_shift": "Negative",
        "negation_flipped": "Strong negative",
        "confidently_wrong": "Strong negative (Low spread)",
        "typo_injected": "~0",
        "genuinely_ambiguous": "Variable (High spread)"
    }
    
    delta_summary = df.groupby("variant_type").agg(
        mean_delta=("paired_delta", "mean"),
        mean_spread=("mean_spread", "mean")
    ).reset_index()
    
    report_lines.append("| Variant Type | Mean Score Delta vs Source | Mean Spread | Expected Direction | Status |")
    report_lines.append("|---|---|---|---|---|")
    
    for _, row in delta_summary.iterrows():
        vt = row["variant_type"]
        delta = row["mean_delta"]
        spread = row["mean_spread"]
        expected = expected_directions.get(vt, "Unknown")
        
        # Simple heuristic to flag mismatches
        status = "Match"
        if "Negative" in expected and delta > -0.2:
            status = "Mismatch (Expected Negative)"
        elif "~0" in expected and abs(delta) > 0.5:
            status = "Mismatch (Expected ~0)"
        elif "High spread" in expected and spread < 0.2:
            status = "Mismatch (Expected High Spread)"
            
        report_lines.append(f"| {vt} | {delta:.3f} | {spread:.3f} | {expected} | {status} |")
        
    report_lines.append("\n")
    
    # 5. Weakest categories
    report_lines.append("## Weakest Categories")
    weakest = summary.sort_values(by="mean_mae", ascending=False).head(2)
    for _, row in weakest.iterrows():
        report_lines.append(f"- **{row['variant_type']}** (MAE: {row['mean_mae']:.3f})")
    
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_report.md"
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Benchmark report saved to {out_path}")
