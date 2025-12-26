"""
scaling_minmax_standard.py

Author: Jungmyung Lee

Description:
    Compares two feature scaling strategies:

    1) Min-Max Scaling   (0~1 normalization)
    2) Standard Scaling  (Z-score normalization)

    Pipeline:
    - Load CSV dataset
    - Select numeric columns only
    - Apply both scaling transforms
    - Save:
        - minmax_scaled.csv
        - standard_scaled.csv
        - scaling statistics report

Outputs:
    ./outputs/feature_scaling/<dataset_name>/

Python Dependencies:
    - pandas
    - NumPy
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
# Min-Max scaling
# =========================================================
def apply_minmax(df):
    num_df = df.select_dtypes(include=[np.number]).copy()

    mins = num_df.min()
    maxs = num_df.max().replace(mins, mins + 1e-9)  # avoid div-zero

    scaled = (num_df - mins) / (maxs - mins)

    return scaled, mins, maxs


# =========================================================
# Standard scaling (Z-score)
# =========================================================
def apply_standard(df):
    num_df = df.select_dtypes(include=[np.number]).copy()

    means = num_df.mean()
    stds = num_df.std(ddof=0).replace(0, 1e-9)

    scaled = (num_df - means) / stds

    return scaled, means, stds


# =========================================================
# Save scaled datasets
# =========================================================
def save_scaled_table(output_dir, base, name, scaled_df, original_df):
    out = original_df.copy()
    for c in scaled_df.columns:
        out[c] = scaled_df[c]

    path = os.path.join(output_dir, f"{base}_{name}.csv")
    out.to_csv(path, index=False)

    print(f"[+] Saved {name} scaled dataset → {path}")


# =========================================================
# Save comparison report
# =========================================================
def save_scaling_report(output_dir, base, num_cols, mm_stats, std_stats):
    path = os.path.join(output_dir, f"{base}_scaling_report.txt")

    mm_mins, mm_maxs = mm_stats
    std_means, std_stds = std_stats

    with open(path, "w", encoding="utf-8") as f:
        f.write("FEATURE SCALING COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write(f"Scaled numeric columns ({len(num_cols)}):\n")
        for c in num_cols:
            f.write(f" - {c}\n")
        f.write("\n")

        f.write("[ Min-Max Scaling Stats ]\n")
        for c in num_cols:
            f.write(f"{c:20s}  min={mm_mins[c]:.4f}  max={mm_maxs[c]:.4f}\n")
        f.write("\n")

        f.write("[ Standard Scaling Stats ]\n")
        for c in num_cols:
            f.write(f"{c:20s}  mean={std_means[c]:.4f}  std={std_stds[c]:.4f}\n")
        f.write("\n")

        f.write("Interpretation Guide:\n")
        f.write(" - Min-Max preserves shape but rescales range to [0,1]\n")
        f.write(" - Standard Scaling centers distribution to mean 0, std 1\n")

    print(f"[+] Saved scaling report → {path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(csv_path):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs", "feature_scaling", base)

    ensure_dir(output_dir)

    print("\n[ FEATURE SCALING COMPARISON ]")
    print(f"Input  : {csv_path}")
    print(f"Output : {output_dir}\n")

    df = load_dataset(csv_path)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) == 0:
        raise ValueError("[ERROR] No numeric columns to scale")

    # ---- Min-Max ----
    minmax_scaled, mins, maxs = apply_minmax(df)
    save_scaled_table(output_dir, base, "minmax_scaled", minmax_scaled, df)

    # ---- Standard ----
    standard_scaled, means, stds = apply_standard(df)
    save_scaled_table(output_dir, base, "standard_scaled", standard_scaled, df)

    # ---- Report ----
    save_scaling_report(
        output_dir,
        base,
        num_cols,
        (mins, maxs),
        (means, stds)
    )

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Min-Max and Standard feature scaling"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV dataset"
    )

    args = parser.parse_args()

    run_pipeline(args.csv)
