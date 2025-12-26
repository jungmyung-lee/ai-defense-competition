"""
object_contour_detection.py

Author: Jungmyung Lee

Description:
    Extracts object contours from an input image and generates:
    1) Contour bounding boxes
    2) Contour masks
    3) Sorted contour list by area
    4) Object-level statistics report

    Supports grayscale or binary images.
    Includes optional area filtering to ignore small/noise contours.

Outputs:
    ./outputs/contours/<image_name>/

Python Dependencies:
    - OpenCV
    - NumPy

Run Example:
    python object_contour_detection.py --image sample.png --min_area 100
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
# Load as grayscale
# =========================================================
def load_gray(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("[ERROR] Failed to load image. Possibly corrupted.")

    return img


# =========================================================
# Ensure binary image (auto-Otsu if needed)
# =========================================================
def ensure_binary(gray):
    # if already binary (0 / 255 only), return as is
    u = np.unique(gray)
    if np.array_equal(u, [0]) or np.array_equal(u, [255]) or np.array_equal(u, [0, 255]):
        return gray

    # else → Otsu binarization
    _, bin_img = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return bin_img


# =========================================================
# Extract contours
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

    # sort by area (desc)
    objects = sorted(objects, key=lambda o: o["area"], reverse=True)

    return objects


# =========================================================
# Draw contour bounding boxes
# =========================================================
def draw_bbox_visual(gray, objects):
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


# =========================================================
# Create contour-only mask
# =========================================================
def contour_mask(gray, objects):
    mask = np.zeros_like(gray)

    for obj in objects:
        cv2.drawContours(mask, [obj["contour"]], -1, 255, thickness=-1)

    return mask


# =========================================================
# Save contour stats report
# =========================================================
def save_contour_report(output_dir, base, objects):
    path = os.path.join(output_dir, f"{base}_contour_stats.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("OBJECT CONTOUR EXTRACTION REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write(f"Detected objects: {len(objects)}\n\n")

        for idx, obj in enumerate(objects):
            x, y, w, h = obj["bbox"]

            f.write(f"[Object {idx+1}]\n")
            f.write(f" - area : {obj['area']}\n")
            f.write(f" - bbox : (x={x}, y={y}, w={w}, h={h})\n")
            f.write("\n")

    print(f"[+] Saved contour report → {path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(image_path, min_area):
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("outputs", "contours", base)

    ensure_dir(output_dir)

    print("\n[ CONTOUR DETECTION PIPELINE ]")
    print(f"Input  : {image_path}")
    print(f"Output : {output_dir}")
    print(f"Min area filter : {min_area}\n")

    gray = load_gray(image_path)

    # Save reference
    ref_path = os.path.join(output_dir, f"{base}_gray.png")
    cv2.imwrite(ref_path, gray)

    # Ensure binary
    bin_img = ensure_binary(gray)

    bin_path = os.path.join(output_dir, f"{base}_binary.png")
    cv2.imwrite(bin_path, bin_img)

    # Extract objects
    objects = extract_contours(bin_img, min_area)

    # Save mask
    mask = contour_mask(gray, objects)
    mask_path = os.path.join(output_dir, f"{base}_contour_mask.png")
    cv2.imwrite(mask_path, mask)

    # Save bbox visualization
    vis = draw_bbox_visual(gray, objects)
    vis_path = os.path.join(output_dir, f"{base}_bbox_preview.png")
    cv2.imwrite(vis_path, vis)

    # Save stats report
    save_contour_report(output_dir, base, objects)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Object contour extraction & statistics pipeline"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to grayscale or binary image"
    )

    parser.add_argument(
        "--min_area",
        type=float,
        default=50.0,
        help="Minimum contour area filter (default=50)"
    )

    args = parser.parse_args()

    run_pipeline(args.image, args.min_area)
