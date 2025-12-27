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
# Drop rows/columns based on missing ratio threshold
# =========================================================
def drop_by_missing_ratio(df, threshold, axis="column"):
    """
    Drop columns or rows whose missing-value ratio exceeds the given threshold.
    axis: "column" or "row"
    threshold: float in [0, 1]
    """
    if threshold is None:
        return df, None

    if axis == "column":
        miss_ratio = df.isna().mean()
        to_drop = miss_ratio[miss_ratio > threshold].index.tolist()
        df_dropped = df.drop(columns=to_drop)
        return df_dropped, {"axis": "column", "dropped": to_drop, "threshold": threshold}

    elif axis == "row":
        miss_ratio = df.isna().mean(axis=1)
        to_drop_idx = df.index[miss_ratio > threshold].tolist()
        df_dropped = df.drop(index=to_drop_idx)
        return df_dropped, {"axis": "row", "dropped": to_drop_idx, "threshold": threshold}

    else:
        raise ValueError(f"Unsupported drop axis: {axis}")



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


def save_missing_report(output_dir, base, before, after, drop_info=None):
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

        if drop_info is not None:
            f.write("[ DROPPED OBJECTS BASED ON MISSING RATIO ]\n")
            f.write(f"Axis      : {drop_info['axis']}\n")
            f.write(f"Threshold : {drop_info['threshold']}\n")
            f.write("Dropped   :\n")
            for obj in drop_info["dropped"]:
                f.write(f" - {obj}\n")

    print(f"[+] Saved missing value report → {path}")



# =========================================================
# Pipeline
# =========================================================
def run_pipeline(csv_path, num_strategy, cat_strategy, drop_thresh=None, drop_axis="column"):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs", "data_cleaning", base)

    ensure_dir(output_dir)

    print("\n[ DATA CLEANING — MISSING VALUE PIPELINE ]")
    print(f"Input  : {csv_path}")
    print(f"Output : {output_dir}")
    print(f"Numeric strategy     : {num_strategy}")
    print(f"Categorical strategy : {cat_strategy}\n")

    df = load_csv_dataset(csv_path)

    # 1) Before cleaning summary
    before_stats = compute_missing_summary(df)

    # 2) Optional drop by missing-value ratio
    drop_info = None
    if drop_thresh is not None:
        df, drop_info = drop_by_missing_ratio(df, drop_thresh, axis=drop_axis)

    # 3) Imputation
    df = fill_numeric(df, num_strategy)
    df = fill_categorical(df, cat_strategy)

    # 4) After cleaning summary
    after_stats = compute_missing_summary(df)

    save_cleaned_dataset(output_dir, base, df)
    save_missing_report(output_dir, base, before_stats, after_stats, drop_info)


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

    parser.add_argument(
        "--drop_thresh",
        type=float,
        default=None,
        help="Optional missing-value ratio threshold (0~1) used to drop rows/columns"
    )

    parser.add_argument(
        "--drop_axis",
        type=str,
        default="column",
        choices=["column", "row"],
        help="Apply drop_thresh to 'column' or 'row' (default: column)"
    )


    args = parser.parse_args()

    run_pipeline(
    args.csv,
    args.num_strategy,
    args.cat_strategy,
    drop_thresh=args.drop_thresh,
    drop_axis=args.drop_axis
        
    )
