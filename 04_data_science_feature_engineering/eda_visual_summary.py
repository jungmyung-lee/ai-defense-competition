"""
eda_visual_summary.py

Author: Jungmyung Lee

Description:
    Generates an automated visual EDA (Exploratory Data Analysis)
    report for a CSV dataset.

    Performs:
    - Numeric feature distribution histograms
    - Boxplot outlier inspection
    - Categorical value count plots
    - Correlation matrix heatmap
    - Dataset statistics summary report

Outputs:
    ./outputs/eda_visual/<dataset_name>/

Python Dependencies:
    - pandas
    - NumPy
    - matplotlib
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# Save histogram per numeric column
# =========================================================
def save_numeric_histograms(df, output_dir, base, num_bins=30):

    num_cols = df.select_dtypes(include=[np.number]).columns

    paths = []

    for col in num_cols:
        values = df[col].dropna()

        plt.figure()
        plt.hist(values, bins=num_bins)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title(f"Histogram — {col}")

        path = os.path.join(output_dir, f"{base}_hist_{col}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        paths.append(path)

    print(f"[+] Saved {len(paths)} histograms")
    return list(num_cols)


# =========================================================
# Save boxplots for numeric columns
# =========================================================
def save_boxplots(df, output_dir, base):
    num_cols = df.select_dtypes(include=[np.number]).columns

    paths = []

    for col in num_cols:
            values = df[col].dropna()

            plt.figure()
            plt.boxplot(values, vert=True)
            plt.ylabel(col)
            plt.title(f"Boxplot — {col}")

            path = os.path.join(output_dir, f"{base}_box_{col}.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

            paths.append(path)

    print(f"[+] Saved {len(paths)} boxplots")
    return list(num_cols)


# =========================================================
# Save categorical count plots
# =========================================================
def save_categorical_counts(df, output_dir, base, max_classes=12):
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    paths = []

    for col in cat_cols:
        counts = df[col].value_counts().head(max_classes)

        plt.figure(figsize=(8, 4))
        plt.bar(counts.index.astype(str), counts.values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("count")
        plt.title(f"Category Frequency — {col}")

        path = os.path.join(output_dir, f"{base}_count_{col}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        paths.append(path)

    print(f"[+] Saved {len(paths)} categorical count plots")
    return list(cat_cols)


# =========================================================
# Correlation matrix heatmap (numeric only)
# =========================================================
def save_correlation_heatmap(df, output_dir, base):
    num_df = df.select_dtypes(include=[np.number])

    if num_df.shape[1] <= 1:
        print("[INFO] Skipped correlation heatmap (not enough numeric columns)")
        return None

    corr = num_df.corr()

    plt.figure(figsize=(6, 5))
    plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix")
    plt.colorbar()

    path = os.path.join(output_dir, f"{base}_correlation_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print("[+] Saved correlation heatmap")
    return path


# =========================================================
# Save textual statistics report
# =========================================================
def save_summary_report(output_dir, base, df, num_cols, cat_cols):
    path = os.path.join(output_dir, f"{base}_eda_report.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("EDA VISUAL SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {len(df.columns)}\n\n")

        f.write("Numeric Columns:\n")
        for c in num_cols:
            f.write(f" - {c}\n")
        f.write("\n")

        f.write("Categorical Columns:\n")
        for c in cat_cols:
            f.write(f" - {c}\n")
        f.write("\n")

        f.write("[ Descriptive Statistics — Numeric ]\n")
        f.write(df[num_cols].describe().to_string())
        f.write("\n\n")

        if cat_cols:
            f.write("[ Top Category Values ]\n")
            for c in cat_cols:
                head = df[c].value_counts().head(5)
                f.write(f"\n{c}:\n{head.to_string()}\n")

    print(f"[+] Saved EDA summary report → {path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(csv_path, num_bins=30, max_classes=12):

    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs", "eda_visual", base)

    ensure_dir(output_dir)

    print("\n[ EDA VISUAL SUMMARY ]")
    print(f"Input  : {csv_path}")
    print(f"Output : {output_dir}\n")

    df = load_dataset(csv_path)

    num_cols = save_numeric_histograms(df, output_dir, base, num_bins=num_bins)
    box_cols = save_boxplots(df, output_dir, base)

    cat_cols = save_categorical_counts(df, output_dir, base, max_classes=max_classes)

    corr_path = save_correlation_heatmap(df, output_dir, base)

    save_summary_report(
        output_dir,
        base,
        df,
        num_cols,
        cat_cols,
        hist_paths=None,   
        box_paths=None,
        cat_paths=None,
        corr_path=corr_path
    )


    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate automated EDA visual summary report"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV dataset"
    )

    parser.add_argument(
        "--num_bins",
        type=int,
        default=30,
        help="Number of bins for numeric histograms (default=30)"
    )

    parser.add_argument(
        "--max_classes",
        type=int,
        default=12,
        help="Maximum number of categories to visualize per categorical column (default=12)"
    )


    args = parser.parse_args()

    run_pipeline(
        args.csv,
        num_bins=args.num_bins,
        max_classes=args.max_classes
    )

