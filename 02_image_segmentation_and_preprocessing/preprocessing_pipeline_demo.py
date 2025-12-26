"""
preprocessing_pipeline_demo.py

Author: Jungmyung Lee

Description:
    Demonstration of a full image preprocessing pipeline:

    1) Load + Grayscale conversion
    2) Noise reduction (Gaussian blur)
    3) Contrast normalization (CLAHE)
    4) Otsu threshold binarization
    5) Morphology opening (noise cleanup)
    6) Contour-based ROI extraction

    Saves intermediate pipeline outputs and generates
    an object statistics report.

Outputs:
    ./outputs/preprocessing_pipeline/<image_name>/

Python Dependencies:
    - OpenCV
    - NumPy

Run Example:
    python preprocessing_pipeline_demo.py --image sample.jpg --min_area 80
"""

import os
import cv2
import argparse
import numpy as np
from datetime import datetime


# =========================================================
# Directory handling
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# Image loading
# =========================================================
def load_image_gray(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("[ERROR] Failed to load image")

    return img


# =========================================================
# Step 1 — Noise Reduction
# =========================================================
def apply_gaussian_denoise(gray):
    return cv2.GaussianBlur(gray, (5, 5), sigmaX=1.2)


# =========================================================
# Step 2 — Contrast Normalization (CLAHE)
# =========================================================
def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# =========================================================
# Step 3 — Otsu Threshold
# =========================================================
def binarize_otsu(gray):
    _, bin_img = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return bin_img


# =========================================================
# Step 4 — Morphology Opening
# =========================================================
def morphology_open(bin_img, k=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)


# =========================================================
# Step 5 — Contour Extraction
# =========================================================
def extract_contours(bin_img, min_area):
    contours, _ = cv2.findContours(
        bin_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []

    for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)

            objects.append({
                "contour": c,
                "area": float(area),
                "bbox": (int(x), int(y), int(w), int(h))
            })

    objects = sorted(objects, key=lambda o: o["area"], reverse=True)
    return objects


# =========================================================
# Visualization helpers
# =========================================================
def draw_object_boxes(gray, objects):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for idx, obj in enumerate(objects):
        x, y, w, h = obj["bbox"]
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{idx+1}",
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return vis


def crop_object_rois(gray, objects, output_dir, base):
    crops = []
    for i, obj in enumerate(objects):
        x, y, w, h = obj["bbox"]
        roi = gray[y:y+h, x:x+w]

        path = os.path.join(output_dir, f"{base}_roi_{i+1}.png")
        cv2.imwrite(path, roi)
        crops.append(path)

    return crops


# =========================================================
# Stats Report
# =========================================================
def save_object_report(output_dir, base, objects):
    path = os.path.join(output_dir, f"{base}_pipeline_object_stats.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("PREPROCESSING PIPELINE — OBJECT REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Detected objects: {len(objects)}\n\n")

        for idx, obj in enumerate(objects):
            x, y, w, h = obj["bbox"]
            f.write(f"[Object {idx+1}]\n")
            f.write(f" - area : {obj['area']}\n")
            f.write(f" - bbox : (x={x}, y={y}, w={w}, h={h})\n\n")

    print(f"[+] Saved object report → {path}")


# =========================================================
# Pipeline Runner
# =========================================================
def run_pipeline(image_path, min_area):
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("outputs", "preprocessing_pipeline", base)

    ensure_dir(output_dir)

    print("\n[ PREPROCESSING PIPELINE DEMO ]")
    print(f"Input  : {image_path}")
    print(f"Output : {output_dir}")
    print(f"Min area : {min_area}\n")

    gray = load_image_gray(image_path)

    # Save original
    cv2.imwrite(os.path.join(output_dir, f"{base}_step0_gray.png"), gray)

    # Step 1 — denoise
    denoise = apply_gaussian_denoise(gray)
    cv2.imwrite(os.path.join(output_dir, f"{base}_step1_denoise.png"), denoise)

    # Step 2 — CLAHE contrast enhance
    enhanced = apply_clahe(denoise)
    cv2.imwrite(os.path.join(output_dir, f"{base}_step2_clahe.png"), enhanced)

    # Step 3 — binarization
    binary = binarize_otsu(enhanced)
    cv2.imwrite(os.path.join(output_dir, f"{base}_step3_binary.png"), binary)

    # Step 4 — morphology cleanup
    opened = morphology_open(binary, k=3)
    cv2.imwrite(os.path.join(output_dir, f"{base}_step4_opening.png"), opened)

    # Step 5 — object extraction
    objects = extract_contours(opened, min_area=min_area)

    # Visualization
    bbox_vis = draw_object_boxes(gray, objects)
    cv2.imwrite(os.path.join(output_dir, f"{base}_step5_bbox_preview.png"), bbox_vis)

    # Save ROIs
    crop_object_rois(gray, objects, output_dir, base)

    # Save report
    save_object_report(output_dir, base, objects)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full preprocessing pipeline demonstration"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--min_area",
        type=float,
        default=60.0,
        help="Minimum contour area filter (default=60)"
    )

    args = parser.parse_args()

    run_pipeline(args.image, args.min_area)
