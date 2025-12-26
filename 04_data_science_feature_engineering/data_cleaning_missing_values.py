"""
data_cleaning_missing_values.py

Author: Jungmyung Lee

Description:
    Loads a CSV dataset, detects missing values, and applies
    basic cleaning strategies depending on feature type.

    Missing value handling:
    - Numeric columns:
        mean / median / zero fill (selectable)
    - Categorical columns:
        most frequent category fill
        OR "UNKNOWN" token assignment

    Saves:
    - cleaned dataset CSV
    - missing value summary report

Outputs:
    ./outputs/data_cleaning/<dataset_name>/

Python Dependencies:
    - pandas
    - NumPy

Run Example:
    python data_cleaning_missing_values.py --csv sample_data.csv --num_strategy mean --cat_strategy mode
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime


# =========================================================
# Directory handling
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# Load CSV
# =========================================================
def load_csv_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    df = pd.read_csv(path)

    if len(df) == 0:
        raise ValueError("[ERROR] CSV file is empty")

    return df


# =========================================================
# Missing Value Summary
# =========================================================
def compute_missing_summary(df):
    miss = df.isna().sum()
    ratio = (df.isna().sum() / len(df)) * 100.0

    report = pd.DataFrame({
        "missing_count": miss,
        "missing_percent": ratio
    })

    return report.sort_values("missing_percent", ascending=False)


# =========================================================
# Numeric column imputation
# =========================================================
def fill_numeric(df, strategy="mean"):
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        if df[col].isna().sum() == 0:
            continue

        if strategy == "mean":
            value = df[col].mean()
        elif strategy == "median":
            value = df[col].median()
        elif strategy == "zero":
            value = 0
        else:
            raise ValueError(f"Unsupported numeric strategy: {strategy}")

        df[col] = df[col].fillna(value)

    return df


# =========================================================
# Categorical column imputation
# =========================================================
def fill_categorical(df, strategy="mode"):
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in cat_cols:
        if df[col].isna().sum() == 0:
            continue

        if strategy == "mode":
            value = df[col].mode().iloc[0]
        elif strategy == "unknown":
            value = "UNKNOWN"
        else:
            raise ValueError(f"Unsupported categorical strategy: {strategy}")

        df[col] = df[col].fillna(value)

    return df


# =========================================================
# Save outputs
# =========================================================
def save_cleaned_dataset(output_dir, base, df):
    out_path = os.path.join(output_dir, f"{base}_cleaned.csv")
    df.to_csv(out_path, index=False)
    print(f"[+] Saved cleaned dataset → {out_path}")


def save_missing_report(output_dir, base, before, after):
    path = os.path.join(output_dir, f"{base}_missing_value_report.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("DATA CLEANING — MISSING VALUE REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write("[ BEFORE CLEANING ]\n")
        f.write(before.to_string())
        f.write("\n\n")

        f.write("[ AFTER CLEANING ]\n")
        f.write(after.to_string())
        f.write("\n\n")

    print(f"[+] Saved missing value report → {path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(csv_path, num_strategy, cat_strategy):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs", "data_cleaning", base)

    ensure_dir(output_dir)

    print("\n[ DATA CLEANING — MISSING VALUE PIPELINE ]")
    print(f"Input  : {csv_path}")
    print(f"Output : {output_dir}")
    print(f"Numeric strategy     : {num_strategy}")
    print(f"Categorical strategy : {cat_strategy}\n")

    df = load_csv_dataset(csv_path)

    before_stats = compute_missing_summary(df)

    df = fill_numeric(df, num_strategy)
    df = fill_categorical(df, cat_strategy)

    after_stats = compute_missing_summary(df)

    save_cleaned_dataset(output_dir, base, df)
    save_missing_report(output_dir, base, before_stats, after_stats)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSV dataset loader & missing value cleaning pipeline"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV dataset"
    )

    parser.add_argument(
        "--num_strategy",
        type=str,
        default="mean",
        choices=["mean", "median", "zero"],
        help="Numeric fill strategy"
    )

    parser.add_argument(
        "--cat_strategy",
        type=str,
        default="mode",
        choices=["mode", "unknown"],
        help="Categorical fill strategy"
    )

    args = parser.parse_args()

    run_pipeline(
        args.csv,
        args.num_strategy,
        args.cat_strategy
    )
