"""
evaluation_metrics_report.py

Author: Jungmyung Lee

Description:
    Generates a unified evaluation report from prediction results.
    Input must contain prediction columns:

        y_true  = ground-truth labels
        y_pred  = model predictions

    Computes metrics including:
    - Accuracy
    - Precision / Recall / F1 (macro & weighted)
    - Per-class support
    - Confusion matrix

    Saves:
    1) Metrics summary report (TXT)
    2) Metrics table (CSV)

Outputs:
    ./outputs/evaluation_report/<result_name>/

Python Dependencies:
    - NumPy
    - pandas
    - scikit-learn

Run Example:
    python evaluation_metrics_report.py --csv model_predictions.csv --label label
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# Directory Handling
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# Load predictions CSV
# =========================================================
def load_predictions(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    df = pd.read_csv(path)

    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError("[ERROR] CSV must contain columns: y_true, y_pred")

    return df


# =========================================================
# Compute aggregate metrics
# =========================================================
def compute_global_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# =========================================================
# Per-class statistics
# =========================================================
def compute_per_class_stats(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    stats = {}

    for cls in labels:
        cls_true = (y_true == cls).astype(int)
        cls_pred = (y_pred == cls).astype(int)

        prec = precision_score(cls_true, cls_pred, zero_division=0)
        rec  = recall_score(cls_true, cls_pred, zero_division=0)
        f1   = f1_score(cls_true, cls_pred, zero_division=0)

        support = int(np.sum(cls_true))

        stats[str(cls)] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": support
        }

    return stats


# =========================================================
# Save CSV metrics table
# =========================================================
def save_metrics_csv(output_dir, base_name, global_metrics, per_class_stats):
    rows = []

    # global summary as one row
    gm = {"metric_scope": "global"}
    gm.update(global_metrics)
    rows.append(gm)

    # per-class rows
    for cls, s in per_class_stats.items():
        row = {"metric_scope": f"class_{cls}"}
        row.update(s)
        rows.append(row)

    df = pd.DataFrame(rows)

    path = os.path.join(output_dir, f"{base_name}_metrics_table.csv")
    df.to_csv(path, index=False)

    print(f"[+] Saved metrics CSV → {path}")


# =========================================================
# Save human-readable TXT report
# =========================================================
def save_text_report(output_dir, base_name, global_metrics, per_class_stats, cm):
    path = os.path.join(output_dir, f"{base_name}_evaluation_report.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("MODEL EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write("[Global Metrics]\n")
        for k, v in global_metrics.items():
            f.write(f" - {k}: {v}\n")

        f.write("\n[Per-Class Metrics]\n")
        for cls, s in per_class_stats.items():
            f.write(f"\nClass {cls}\n")
            for k, v in s.items():
                f.write(f" - {k}: {v}\n")

        f.write("\n[Confusion Matrix]\n")
        for row in cm.tolist():
            f.write(f" {row}\n")

        f.write("\nNotes:\n")
        f.write(" - Macro metrics treat all classes equally\n")
        f.write(" - Weighted metrics reflect class frequency\n")

    print(f"[+] Saved evaluation report → {path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(csv_path):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs", "evaluation_report", base)

    ensure_dir(output_dir)

    print("\n[ EVALUATION REPORT GENERATOR ]")
    print(f"Input  : {csv_path}")
    print(f"Output : {output_dir}\n")

    df = load_predictions(csv_path)

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    global_metrics = compute_global_metrics(y_true, y_pred)
    per_class_stats = compute_per_class_stats(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    save_metrics_csv(output_dir, base, global_metrics, per_class_stats)
    save_text_report(output_dir, base, global_metrics, per_class_stats, cm)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate evaluation metrics report from prediction CSV"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV with columns: y_true, y_pred"
    )

    args = parser.parse_args()

    run_pipeline(args.csv)
