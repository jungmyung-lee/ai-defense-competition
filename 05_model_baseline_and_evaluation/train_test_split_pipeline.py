"""
train_test_split_pipeline.py

Author: Jungmyung Lee

Description:
    Splits a dataset into Train / Test subsets with:
    1) Optional stratified sampling by label column
    2) Deterministic random seed for reproducibility
    3) Class distribution statistics before & after split

    Saves resulting CSV files and a split summary report.

Outputs:
    ./outputs/train_test_split/<dataset_name>/

Python Dependencies:
    - NumPy
    - pandas
    - scikit-learn

Run Example:
    python train_test_split_pipeline.py --csv data.csv --label label --test 0.2
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split


# =========================================================
# Directory handling
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# Load CSV dataset
# =========================================================
def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if len(df) == 0:
        raise ValueError("[ERROR] CSV file is empty")

    return df


# =========================================================
# Compute class distribution
# =========================================================
def compute_class_distribution(df, label_col):
    values, counts = np.unique(df[label_col], return_counts=True)

    total = np.sum(counts)
    ratio = counts / total

    dist = {}

    for v, c, r in zip(values, counts, ratio):
        dist[str(v)] = {
            "count": int(c),
            "ratio": float(round(r, 6))
        }

    return dist


# =========================================================
# Save split report
# =========================================================
def save_split_report(output_dir, base, label_col,
                      full_dist, train_dist, test_dist,
                      test_ratio, seed):

    path = os.path.join(output_dir, f"{base}_split_report.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("TRAIN / TEST SPLIT REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write(f"Label column : {label_col}\n")
        f.write(f"Test ratio   : {test_ratio}\n")
        f.write(f"Random seed  : {seed}\n\n")

        f.write("[Original Dataset Class Distribution]\n")
        for cls, s in full_dist.items():
            f.write(f" - {cls}: count={s['count']}  ratio={s['ratio']}\n")

        f.write("\n[Training Set Class Distribution]\n")
        for cls, s in train_dist.items():
            f.write(f" - {cls}: count={s['count']}  ratio={s['ratio']}\n")

        f.write("\n[Test Set Class Distribution]\n")
        for cls, s in test_dist.items():
            f.write(f" - {cls}: count={s['count']}  ratio={s['ratio']}\n")

        f.write("\nNotes:\n")
        f.write(" - Stratified split preserves class ratios where possible\n")
        f.write(" - Differences may occur on small class sample sizes\n")

    print(f"[+] Saved split report → {path}")


# =========================================================
# Save CSV outputs
# =========================================================
def save_split_files(output_dir, base, train_df, test_df):
    train_path = os.path.join(output_dir, f"{base}_train.csv")
    test_path  = os.path.join(output_dir, f"{base}_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[+] Saved train CSV → {train_path}")
    print(f"[+] Saved test  CSV → {test_path}")


# =========================================================
# Main pipeline
# =========================================================
def run_pipeline(csv_path, label_col, test_ratio, seed):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs", "train_test_split", base)

    ensure_dir(output_dir)

    print("\n[ TRAIN / TEST SPLIT PIPELINE ]")
    print(f"Input CSV : {csv_path}")
    print(f"Output    : {output_dir}\n")

    df = load_dataset(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"[ERROR] Label column not found: {label_col}")

    # ------------------------------------
    # global distribution
    # ------------------------------------
    full_dist = compute_class_distribution(df, label_col)

    # ------------------------------------
    # stratified split (recommended)
    # ------------------------------------
    train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=df[label_col]
    )

    # ------------------------------------
    # compute split distributions
    # ------------------------------------
    train_dist = compute_class_distribution(train_df, label_col)
    test_dist  = compute_class_distribution(test_df, label_col)

    # ------------------------------------
    # save results
    # ------------------------------------
    save_split_files(output_dir, base, train_df, test_df)

    save_split_report(
        output_dir,
        base,
        label_col,
        full_dist,
        train_dist,
        test_dist,
        test_ratio,
        seed
    )

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/Test split pipeline with stratified sampling"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to dataset CSV"
    )

    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Name of label column"
    )

    parser.add_argument(
        "--test",
        type=float,
        default=0.2,
        help="Test split ratio (default=0.2)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default=42)"
    )

    args = parser.parse_args()

    run_pipeline(args.csv, args.label, args.test, args.seed)
