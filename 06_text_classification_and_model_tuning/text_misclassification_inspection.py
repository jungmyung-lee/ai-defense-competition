"""
text_misclassification_inspection.py

Author: Jungmyung Lee

Description:
    Misclassification analysis tool for text classifiers.

    This script loads:
        - test labels
        - model predictions
        - predicted probability scores (optional)

    It extracts ONLY misclassified samples and generates:
        1) detailed CSV of wrong predictions
        2) sorted "hard cases" list by confidence gap
        3) textual analysis summary report

    Intended to highlight model understanding & error analysis
    rather than just reporting accuracy.

Outputs:
    ./outputs/text_error_analysis/<dataset_name>/
        - misclassified_samples.csv
        - hard_cases_topN.csv
        - error_analysis_report.txt

Python Dependencies:
    - pandas
    - NumPy
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime


# =========================================================
# Utilities
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_prediction_file(csv_path,
                         text_col,
                         true_col="y_true",
                         pred_col="y_pred",
                         prob_col=None):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] Prediction file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    for col in [text_col, true_col, pred_col]:
        if col not in df.columns:
            raise ValueError(f"[ERROR] Missing column: {col}")

    if prob_col is not None and prob_col not in df.columns:
        print(f"[WARN] Probability column '{prob_col}' not found — continuing without scores")
        prob_col = None

    return df, prob_col


# =========================================================
# Analysis Functions
# =========================================================
def extract_misclassified(df, text_col, true_col, pred_col, prob_col=None):
    wrong = df[df[true_col] != df[pred_col]].copy()

    if prob_col is None:
        wrong["confidence_gap"] = np.nan
        return wrong

    # if probability is provided → compute "confidence margin"
    # (1 - probability of predicted class)
    wrong["confidence_gap"] = 1.0 - wrong[prob_col]

    return wrong


def extract_hard_cases(wrong_df, top_k=30):
    if wrong_df["confidence_gap"].isna().all():
        return wrong_df.head(top_k)

    return wrong_df.sort_values(by="confidence_gap", ascending=True).head(top_k)


# =========================================================
# Report Writer
# =========================================================
def save_error_report(output_dir, base, wrong_df):

    path = os.path.join(output_dir, f"{base}_error_analysis_report.txt")

    total_wrong = len(wrong_df)

    with open(path, "w", encoding="utf-8") as f:

        f.write("TEXT CLASSIFICATION — MISCLASSIFICATION ANALYSIS\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write(f"Total misclassified samples : {total_wrong}\n\n")

        if total_wrong == 0:
            f.write("No misclassified samples found.\n")
            return

        # Per-class confusion trend hint
        f.write("Misclassification Trends (true → predicted)\n")
        f.write("--------------------------------------------------\n")

        crosstab = pd.crosstab(wrong_df["y_true"], wrong_df["y_pred"])

        f.write(str(crosstab))
        f.write("\n\n")

        f.write("Interpretation Guide:\n")
        f.write(" - Rows with dominant columns may indicate systematic bias\n")
        f.write(" - High concentration in single confusion pair implies\n")
        f.write("   semantic similarity or label ambiguity\n")

    print(f"[+] Saved error analysis report → {path}")


# =========================================================
# Save Outputs
# =========================================================
def save_outputs(output_dir, base, wrong_df, hard_cases_df):

    wrong_path = os.path.join(output_dir, f"{base}_misclassified_samples.csv")
    wrong_df.to_csv(wrong_path, index=False)
    print(f"[+] Saved misclassified samples → {wrong_path}")

    hard_path = os.path.join(output_dir, f"{base}_hard_cases_topN.csv")
    hard_cases_df.to_csv(hard_path, index=False)
    print(f"[+] Saved hard cases list → {hard_path}")


# =========================================================
# Pipeline Runner
# =========================================================
def run_pipeline(pred_csv,
                 text_col,
                 true_col="y_true",
                 pred_col="y_pred",
                 prob_col=None,
                 top_k=30):

    base = os.path.splitext(os.path.basename(pred_csv))[0]
    output_dir = os.path.join("outputs", "text_error_analysis", base)
    ensure_dir(output_dir)

    print("\n[ TEXT MISCLASSIFICATION INSPECTION ]")
    print(f"Input   : {pred_csv}")
    print(f"Output  : {output_dir}\n")

    df, prob_col = load_prediction_file(
        pred_csv,
        text_col,
        true_col=true_col,
        pred_col=pred_col,
        prob_col=prob_col
    )

    wrong_df = extract_misclassified(
        df,
        text_col,
        true_col,
        pred_col,
        prob_col=prob_col
    )

    hard_cases_df = extract_hard_cases(wrong_df, top_k=top_k)

    save_outputs(output_dir, base, wrong_df, hard_cases_df)
    save_error_report(output_dir, base, wrong_df)

    print("\n[ DONE ]\n")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect misclassified text samples for qualitative error analysis"
    )

    parser.add_argument("--pred_csv", type=str, required=True,
                        help="Path to prediction CSV file")

    parser.add_argument("--text_col", type=str, default="text",
                        help="Column containing original text")

    parser.add_argument("--true_col", type=str, default="y_true",
                        help="Ground truth label column")

    parser.add_argument("--pred_col", type=str, default="y_pred",
                        help="Predicted label column")

    parser.add_argument("--prob_col", type=str, default=None,
                        help="Optional probability/confidence column")

    parser.add_argument("--top_k", type=int, default=30,
                        help="Number of hard cases to export")

    args = parser.parse_args()

    run_pipeline(
        args.pred_csv,
        args.text_col,
        true_col=args.true_col,
        pred_col=args.pred_col,
        prob_col=args.prob_col,
        top_k=args.top_k
    )
