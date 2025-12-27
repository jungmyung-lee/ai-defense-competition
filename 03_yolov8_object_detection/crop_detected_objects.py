"""
crop_detected_objects.py

Author: Jungmyung Lee

Description:
    Automatically crops detected object regions from an image
    using bounding box coordinates stored in a detection CSV.

    Supports detection result formats:
    - YOLO image inference CSV
    - YOLO video frame CSV
    - Standardized bbox_export CSV

    For each bounding box, this script saves:
    - Cropped ROI image patch
    - Crop metadata record (CSV + TXT)

Outputs:
    ./outputs/object_crops/<source_name>/

Python Dependencies:
    - OpenCV
    - pandas
    - NumPy

Run Example:
    python crop_detected_objects.py --image sample.jpg --csv sample_yolov8_detections.csv
"""

import os
import cv2
import argparse
import pandas as pd
from datetime import datetime


# =========================================================
# Directory Handling
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# Load Image
# =========================================================
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Image not found: {path}")

    img = cv2.imread(path)

    if img is None:
        raise ValueError("[ERROR] Failed to load image")

    return img


# =========================================================
# Load Detection CSV
# =========================================================
def load_detection_table(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"x", "y", "w", "h", "class_name", "confidence"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"[ERROR] Detection CSV missing required columns.\n"
            f"Required: {required}\n"
            f"Found: {set(df.columns)}"
        )

    return df


# =========================================================
# Perform Safe Crop
# =========================================================
def crop_roi(image, x, y, w, h):
    h_img, w_img = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)

    return image[y1:y2, x1:x2]


# =========================================================
# Save Crop Metadata CSV + TXT
# =========================================================
def save_crop_reports(output_dir, base, records):
    import pandas as pd

    df = pd.DataFrame(records)

    csv_path = os.path.join(output_dir, f"{base}_crop_metadata.csv")
    df.to_csv(csv_path, index=False)

    txt_path = os.path.join(output_dir, f"{base}_crop_summary.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("OBJECT CROP SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Total crops: {len(records)}\n\n")

        for r in records:
            f.write(f"[Crop {r['crop_id']}]\n")
            f.write(f" - file : {r['crop_path']}\n")
            f.write(f" - class: {r['class_name']}\n")
            f.write(f" - conf : {r['confidence']}\n")
            f.write(f" - bbox : (x={r['x']}, y={r['y']}, w={r['w']}, h={r['h']})\n\n")

    print(f"[+] Saved crop metadata CSV → {csv_path}")
    print(f"[+] Saved crop summary TXT → {txt_path}")


# =========================================================
# Main Crop Pipeline
# =========================================================
def run_pipeline(image_path, csv_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("outputs", "object_crops", base)

    ensure_dir(output_dir)

    print("\n[ OBJECT ROI CROP PIPELINE ]")
    print(f"Image : {image_path}")
    print(f"CSV   : {csv_path}")
    print(f"Output: {output_dir}\n")

    image = load_image(image_path)
    df = load_detection_table(csv_path)

    crop_records = []

    for i, row in df.iterrows():
        roi = crop_roi(
            image,
            int(row["x"]),
            int(row["y"]),
            int(row["w"]),
            int(row["h"])
        )

        crop_name = f"{base}_crop_{i+1}.png"
        crop_path = os.path.join(output_dir, crop_name)

        cv2.imwrite(crop_path, roi)

        crop_records.append({
            "crop_id": i + 1,
            "crop_path": crop_path,
            "class_name": str(row["class_name"]),
            "confidence": float(row["confidence"]),
            "x": int(row["x"]),
            "y": int(row["y"]),
            "w": int(row["w"]),
            "h": int(row["h"])
        })

    save_crop_reports(output_dir, base, crop_records)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatically crop detected object regions from an image or video"
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Path to original source image (for image mode)"
    )

    parser.add_argument(
        "--video",
        type=str,
        help="Path to original source video (for video mode)"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Detection CSV containing bounding box coordinates"
    )

    args = parser.parse_args()

    if not args.image and not args.video:
        parser.error("You must provide either --image or --video")

    if args.video:
        run_video_pipeline(args.video, args.csv)
    else:
        run_image_pipeline(args.image, args.csv)

