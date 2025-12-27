"""
zscore_outlier_detection.py

Author: Jungmyung Lee

Description:
    Performs Z-score based outlier detection on numeric columns.

    Steps:
    1) Load CSV dataset
    2) Select numeric columns only
    3) Compute Z-score for each value
    4) Mark rows as outliers if any column exceeds threshold
    5) Save:
        - outlier rows CSV
        - cleaned dataset (outliers removed)
        - analysis summary report

    Outlier rule:
        |z| >= threshold  (default = 3.0)

Outputs:
    ./outputs/outlier_detection/<dataset_name>/

Python Dependencies:
    - pandas
    - NumPy

Run Example:
    python zscore_outlier_detection.py --csv sample.csv --threshold 3.0
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime


# =========================================================
# Utils
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# Load dataset
# =========================================================
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    df = pd.read_csv(path)

    if len(df) == 0:
        raise ValueError("[ERROR] CSV file is empty")

    return df


# =========================================================
# Compute Z-score matrix
# =========================================================
def compute_zscores(df):
    num_df = df.select_dtypes(include=[np.number])

    if num_df.shape[1] == 0:
        raise ValueError("[ERROR] No numeric columns found in dataset")

    means = num_df.mean()
    stds = num_df.std(ddof=0).replace(0, np.nan)

    z = (num_df - means) / stds

    return z, num_df.columns.tolist()


# =========================================================
# Outlier detection rule
# =========================================================
def detect_outliers(zmat, threshold):
    abs_z = np.abs(zmat)

    # Row-wise mask
    row_mask = (abs_z >= threshold).any(axis=1)

    # Column-wise outlier counts
    col_counts = (abs_z >= threshold).sum(axis=0)

    return row_mask, col_counts


# =========================================================
# Save results
# =========================================================
def save_outlier_table(output_dir, base, outliers):
    path = os.path.join(output_dir, f"{base}_outliers.csv")
    outliers.to_csv(path, index=False)
    print(f"[+] Saved outlier rows → {path}")


def save_cleaned_table(output_dir, base, cleaned):
    path = os.path.join(output_dir, f"{base}_cleaned_no_outliers.csv")
    cleaned.to_csv(path, index=False)
    print(f"[+] Saved cleaned dataset → {path}")


def save_summary_report(output_dir, base, df, outliers, cols, threshold, col_counts=None):

    path = os.path.join(output_dir, f"{base}_outlier_report.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("Z-SCORE OUTLIER DETECTION REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write(f"Rows total     : {len(df)}\n")
        f.write(f"Outliers found : {len(outliers)}\n")
        f.write(f"Remaining rows : {len(df) - len(outliers)}\n\n")

        f.write(f"Threshold rule : |z| >= {threshold}\n")
        f.write("Evaluated columns:\n")

        for c in cols:
            f.write(f" - {c}\n")

        f.write("\nClass balance notice:\n")
        f.write("Removing outliers may affect distribution — review before training.\n")

        if col_counts is not None:
            f.write("\nPer-column outlier counts:\n")
            for c, cnt in zip(cols, col_counts):
                f.write(f" - {c}: {int(cnt)}\n")


    print(f"[+] Saved analysis report → {path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(csv_path, threshold):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs", "outlier_detection", base)

    ensure_dir(output_dir)

    print("\n[ Z-SCORE OUTLIER DETECTION ]")
    print(f"Input     : {csv_path}")
    print(f"Output    : {output_dir}")
    print(f"Threshold : {threshold}\n")

    df = load_dataset(csv_path)

    zmat, cols = compute_zscores(df)

    row_mask, col_counts = detect_outliers(zmat, threshold)

    outliers = df[row_mask]
    cleaned = df[~row_mask]

    save_outlier_table(output_dir, base, outliers)
    save_cleaned_table(output_dir, base, cleaned)
    save_summary_report(output_dir, base, df, outliers, cols, threshold, col_counts)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Z-score based numeric outlier detection pipeline"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV dataset"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Outlier Z-score threshold (default=3.0)"
    )

    args = parser.parse_args()

    run_pipeline(args.csv, args.threshold)
