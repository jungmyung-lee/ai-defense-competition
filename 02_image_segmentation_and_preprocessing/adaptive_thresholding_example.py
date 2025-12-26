"""
adaptive_thresholding_example.py

Author: Jungmyung Lee

Description:
    Applies multiple thresholding methods to an input image:
    1) Global threshold (fixed value)
    2) Otsu automatic threshold selection
    3) Adaptive Mean thresholding
    4) Adaptive Gaussian thresholding

    Saves binarized outputs and pixel-ratio statistics for comparison.

Outputs:
    ./outputs/thresholding/<image_name>/

Python Dependencies:
    - OpenCV
    - NumPy

Run Example:
    python adaptive_thresholding_example.py --image sample.jpg --th 120
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
# Load image
# =========================================================
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    img = cv2.imread(path)

    if img is None:
        raise ValueError("[ERROR] Failed to load image. Possibly corrupted.")

    return img


# =========================================================
# Convert → grayscale
# =========================================================
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# =========================================================
# Compute binary pixel statistics
# =========================================================
def compute_binary_stats(binary_img):
    flat = binary_img.flatten()

    white = int(np.sum(flat == 255))
    black = int(np.sum(flat == 0))
    total = len(flat)

    return {
        "white_pixels": white,
        "black_pixels": black,
        "white_ratio": round(white / total, 6),
        "black_ratio": round(black / total, 6)
    }


# =========================================================
# Save stats text report
# =========================================================
def save_stats_report(output_dir, base, stats_dict):
    path = os.path.join(output_dir, f"{base}_threshold_stats.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("THRESHOLDING COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        for name, stats in stats_dict.items():
            f.write(f"[{name}]\n")
            for k, v in stats.items():
                f.write(f" - {k}: {v}\n")
            f.write("\n")

    print(f"[+] Saved threshold statistics → {path}")


# =========================================================
# Thresholding methods
# =========================================================
def apply_threshold_methods(gray, global_th):
    results = {}
    stats = {}

    # -------------------------------
    # Global threshold
    # -------------------------------
    _, global_bin = cv2.threshold(
        gray, global_th, 255, cv2.THRESH_BINARY
    )

    results["global_threshold"] = global_bin
    stats["global_threshold"] = compute_binary_stats(global_bin)

    # -------------------------------
    # Otsu automatic threshold
    # -------------------------------
    otsu_th, otsu_bin = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    results["otsu_threshold"] = otsu_bin
    stats["otsu_threshold"] = compute_binary_stats(otsu_bin)
    stats["otsu_threshold"]["selected_threshold"] = float(otsu_th)

    # -------------------------------
    # Adaptive Mean
    # -------------------------------
    mean_bin = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    results["adaptive_mean"] = mean_bin
    stats["adaptive_mean"] = compute_binary_stats(mean_bin)

    # -------------------------------
    # Adaptive Gaussian
    # -------------------------------
    gauss_bin = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    results["adaptive_gaussian"] = gauss_bin
    stats["adaptive_gaussian"] = compute_binary_stats(gauss_bin)

    return results, stats


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(image_path, global_th):
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("outputs", "thresholding", base)

    ensure_dir(output_dir)

    print("\n[ THRESHOLDING PIPELINE ]")
    print(f"Input  : {image_path}")
    print(f"Output : {output_dir}\n")

    img = load_image(image_path)
    gray = to_gray(img)

    # Save grayscale reference
    gray_path = os.path.join(output_dir, f"{base}_gray.png")
    cv2.imwrite(gray_path, gray)

    # Apply methods
    results, stats = apply_threshold_methods(gray, global_th)

    # Save images
    for name, bin_img in results.items():
        path = os.path.join(output_dir, f"{base}_{name}.png")
        cv2.imwrite(path, bin_img)

    # Save report
    save_stats_report(output_dir, base, stats)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thresholding comparison: Global / Otsu / Adaptive"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--th",
        type=int,
        default=120,
        help="Global threshold value (default=120)"
    )

    args = parser.parse_args()

    run_pipeline(args.image, args.th)
