"""
object_counting_and_class_stats.py

Author: Jungmyung Lee

Description:
    Performs statistical analysis on detection result tables.

    Supports detection CSV formats:
    - YOLO image inference CSV
    - YOLO video detection CSV (frame column)
    - bbox_export standardized CSV

    Computes:
    - Total detection count
    - Class-wise counts
    - Frame-wise object distribution
    - Confidence statistics per class
      (mean / max / min / std)

    Saves:
    - Class statistics CSV
    - Frame statistics CSV
    - Human-readable analysis report

Outputs:
    ./outputs/detection_stats/<result_name>/

Python Dependencies:
    - pandas
    - NumPy

Run Example:
    python object_counting_and_class_stats.py --csv sample_yolov8_detections.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime


# =========================================================
# Directory Utils
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# Load Detection CSV
# =========================================================
def load_detection_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    df = pd.read_csv(path)

    required = {"class_name", "confidence", "x", "y", "w", "h"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"[ERROR] CSV missing required columns\n"
            f"Required: {required}\n"
            f"Found: {set(df.columns)}"
        )

    return df


# =========================================================
# Detect whether frame column exists (video logs)
# =========================================================
def detect_frame_mode(df):
    return "frame" in df.columns


# =========================================================
# Class-wise statistics
# =========================================================
def compute_class_stats(df):
    groups = df.groupby("class_name")

    rows = []

    for cls, g in groups:
        rows.append({
            "class_name": cls,
            "count": len(g),
            "mean_conf": float(np.mean(g["confidence"])),
            "max_conf": float(np.max(g["confidence"])),
            "min_conf": float(np.min(g["confidence"])),
            "std_conf": float(np.std(g["confidence"]))
        })

    return pd.DataFrame(rows).sort_values("count", ascending=False)


# =========================================================
# Frame-wise statistics (video only)
# =========================================================
def compute_frame_stats(df):
    
    if "frame" in df.columns:
        frame_col = "frame"
    elif "source_id" in df.columns:
        frame_col = "source_id"
    else:
        return None

    groups = df.groupby(frame_col)

    rows = []

    for frame_value, g in groups:
        rows.append({
            
            "frame": int(frame_value),
            "object_count": len(g),
            "unique_classes": len(g["class_name"].unique())
        })

    return pd.DataFrame(rows).sort_values("frame")



# =========================================================
# Save CSV results
# =========================================================
def save_stats_tables(output_dir, base, class_df, frame_df):
    class_path = os.path.join(output_dir, f"{base}_class_stats.csv")
    class_df.to_csv(class_path, index=False)

    print(f"[+] Saved class stats → {class_path}")

    if frame_df is not None:
        frame_path = os.path.join(output_dir, f"{base}_frame_stats.csv")
        frame_df.to_csv(frame_path, index=False)
        print(f"[+] Saved frame stats → {frame_path}")


# =========================================================
# Save Human-readable Report
# =========================================================
def save_summary_report(output_dir, base, df, class_df, frame_df):
    path = os.path.join(output_dir, f"{base}_detection_stats_report.txt")

    total_dets = len(df)
    num_classes = len(df["class_name"].unique())

    with open(path, "w", encoding="utf-8") as f:
        f.write("DETECTION STATISTICS REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write(f"Total detections : {total_dets}\n")
        f.write(f"Unique classes   : {num_classes}\n\n")

        f.write("Class-wise counts:\n")
        for _, r in class_df.iterrows():
            f.write(f" - {r['class_name']}: {int(r['count'])}\n")
        f.write("\n")

        if frame_df is not None:
            f.write("Frame-wise summary:\n")
            f.write(f" Min objects/frame : {frame_df['object_count'].min()}\n")
            f.write(f" Max objects/frame : {frame_df['object_count'].max()}\n")
            f.write(f" Mean objects/frame: {frame_df['object_count'].mean():.2f}\n")
            f.write("\n")

        f.write("Confidence statistics per class:\n\n")
        for _, r in class_df.iterrows():
            f.write(f"[{r['class_name']}]\n")
            f.write(f" count     : {int(r['count'])}\n")
            f.write(f" mean conf : {r['mean_conf']:.4f}\n")
            f.write(f" max conf  : {r['max_conf']:.4f}\n")
            f.write(f" min conf  : {r['min_conf']:.4f}\n")
            f.write(f" std conf  : {r['std_conf']:.4f}\n\n")

    print(f"[+] Saved summary report → {path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(csv_path):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs", "detection_stats", base)

    ensure_dir(output_dir)

    print("\n[ DETECTION STATISTICS ANALYSIS ]")
    print(f"Input  : {csv_path}")
    print(f"Output : {output_dir}\n")

    df = load_detection_csv(csv_path)

    class_df = compute_class_stats(df)
    frame_df = compute_frame_stats(df)

    save_stats_tables(output_dir, base, class_df, frame_df)
    save_summary_report(output_dir, base, df, class_df, frame_df)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute object counting & class statistics from detection CSV"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Detection CSV file (image or video inference)"
    )

    args = parser.parse_args()

    run_pipeline(args.csv)
