"""
export_detection_boxes_to_csv.py

Author: Jungmyung Lee

Description:
    Converts detection result files into a unified bounding-box dataset.

    Supports input formats:
    1) YOLO inference CSV (from image or video pipeline)
    2) JSON-style detection logs (future extension ready)

    Extracts & saves standardized fields:
    - source_id (image name or frame index)
    - class_id / class_name
    - confidence
    - bbox (x, y, w, h)

    Useful for:
    - Dataset curation
    - Tracking / post-processing pipelines
    - Statistical detection analysis

Outputs:
    ./outputs/detection_export/<result_name>/

Python Dependencies:
    - pandas
    - NumPy

Run Example:
    python export_detection_boxes_to_csv.py --csv sample_yolov8_detections.csv
"""

import os
import argparse
import pandas as pd
from datetime import datetime


# =========================================================
# Directory handling
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# Load detection CSV
# (compatible with previous YOLO scripts)
# =========================================================
def load_detection_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    df = pd.read_csv(path)

    required_cols = {"class_id", "class_name", "confidence", "x", "y", "w", "h"}

    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"[ERROR] CSV missing required columns:\nExpected: {required_cols}\nFound: {set(df.columns)}"
        )

    return df


# =========================================================
# Infer source identifier column
# (image results vs video results)
# =========================================================
def infer_source_column(df):
    if "frame" in df.columns:
        return "frame", "frame"
    elif "image" in df.columns:
        return "image", "image"
    else:
        # fallback — treat as single source file
        return None, "single_source"


# =========================================================
# Normalize export format
# =========================================================
def normalize_detection_table(df, source_key):
    rows = []

    for _, r in df.iterrows():
        rows.append({
            "source_id": r[source_key] if source_key else 0,
            "class_id": int(r["class_id"]),
            "class_name": str(r["class_name"]),
            "confidence": float(r["confidence"]),
            "x": int(r["x"]),
            "y": int(r["y"]),
            "w": int(r["w"]),
            "h": int(r["h"])
        })

    return pd.DataFrame(rows)


# =========================================================
# Save export dataset
# =========================================================
def save_export(output_dir, base, export_df):
    out_path = os.path.join(output_dir, f"{base}_bbox_export.csv")
    export_df.to_csv(out_path, index=False)

    print(f"[+] Saved detection export CSV → {out_path}")


# =========================================================
# Save summary report
# =========================================================
def save_summary_report(output_dir, base, export_df):
    path = os.path.join(output_dir, f"{base}_bbox_export_report.txt")

    num_boxes = len(export_df)
    num_sources = len(export_df["source_id"].unique())
    num_classes = len(export_df["class_id"].unique())

    with open(path, "w", encoding="utf-8") as f:
        f.write("DETECTION BOX EXPORT REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write(f"Total detections : {num_boxes}\n")
        f.write(f"Unique sources   : {num_sources}\n")
        f.write(f"Unique classes   : {num_classes}\n\n")

        f.write("Class distribution:\n")
        class_counts = export_df["class_name"].value_counts()

        for cls, cnt in class_counts.items():
            f.write(f" - {cls}: {cnt}\n")

    print(f"[+] Saved summary report → {path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(csv_path):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs", "detection_export", base)

    ensure_dir(output_dir)

    print("\n[ DETECTION BOX EXPORT PIPELINE ]")
    print(f"Input  : {csv_path}")
    print(f"Output : {output_dir}\n")

    df = load_detection_csv(csv_path)

    source_col, mode = infer_source_column(df)
    print(f"[INFO] Source column mode = {mode}")

    export_df = normalize_detection_table(df, source_col)

    save_export(output_dir, base, export_df)
    save_summary_report(output_dir, base, export_df)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export YOLO detection bounding boxes into standardized CSV format"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Input detection CSV (from image or video pipeline)"
    )

    args = parser.parse_args()

    run_pipeline(args.csv)
