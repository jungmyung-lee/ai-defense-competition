"""
edge_detection_sobel_canny.py

Author: Jungmyung Lee

Description:
    Applies and compares two edge detection methods:
    1) Sobel gradient magnitude (x + y combined)
    2) Canny edge detection

    Saves edge maps and computes edge-density statistics
    for quantitative comparison.

Outputs:
    ./outputs/edge_detection/<image_name>/

Python Dependencies:
    - OpenCV
    - NumPy

Run Example:
    python edge_detection_sobel_canny.py --image sample.jpg --low 80 --high 150
"""

import os
import cv2
import argparse
import numpy as np
from datetime import datetime


# =========================================================
# Directory Handling
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# Image Loading
# =========================================================
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    img = cv2.imread(path)

    if img is None:
        raise ValueError("[ERROR] Failed to load image. Possibly corrupted.")

    return img


# =========================================================
# Convert → Grayscale
# =========================================================
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# =========================================================
# Sobel Gradient Magnitude
# =========================================================
def sobel_edge(img_gray):
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    mag = np.sqrt((sobel_x ** 2) + (sobel_y ** 2))

    mag_norm = np.uint8(255 * (mag / np.max(mag) + 1e-6))

    _, sobel_binary = cv2.threshold(mag_norm, 50, 255, cv2.THRESH_BINARY)

    return mag_norm, sobel_binary


# =========================================================
# Canny Edge Detection
# =========================================================
def canny_edge(img_gray, low_t, high_t):
    canny = cv2.Canny(img_gray, threshold1=low_t, threshold2=high_t)
    return canny


# =========================================================
# Edge Statistics
# =========================================================
def compute_edge_stats(edge_map):
    flat = edge_map.flatten()

    edge_pixels = int(np.sum(flat > 0))
    total_pixels = len(flat)

    edge_ratio = edge_pixels / total_pixels

    return {
        "total_pixels": total_pixels,
        "edge_pixels": edge_pixels,
        "edge_ratio": round(edge_ratio, 6)
    }


# =========================================================
# Save Edge Statistics Report
# =========================================================
def save_stats_report(output_dir, base_name, sobel_stats, canny_stats):
    path = os.path.join(output_dir, f"{base_name}_edge_comparison_stats.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("EDGE DETECTION COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write("[Sobel Edge Map]\n")
        for k, v in sobel_stats.items():
            f.write(f" - {k}: {v}\n")

        f.write("\n[Canny Edge Map]\n")
        for k, v in canny_stats.items():
            f.write(f" - {k}: {v}\n")

        f.write("\nInterpretation:\n")
        f.write(" - Higher edge_ratio generally indicates stronger edge response\n")
        f.write(" - Sobel detects gradient magnitude (texture + outlines)\n")
        f.write(" - Canny detects clean object boundaries with hysteresis\n")

    print(f"[+] Saved stats report → {path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(image_path, low_t, high_t):
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("outputs", "edge_detection", base)

    ensure_dir(output_dir)

    print("\n[ EDGE DETECTION PROCESSING ]")
    print(f"Input  : {image_path}")
    print(f"Output : {output_dir}\n")

    img = load_image(image_path)
    gray = to_gray(img)

    # Save grayscale reference
    gray_path = os.path.join(output_dir, f"{base}_gray.png")
    cv2.imwrite(gray_path, gray)

    # -------------------------------
    # Sobel Gradient Magnitude
    # -------------------------------
    sobel_mag, sobel_bin = sobel_edge(gray)

    sobel_mag_path = os.path.join(output_dir, f"{base}_sobel_magnitude.png")
    sobel_bin_path = os.path.join(output_dir, f"{base}_sobel_binary.png")

    cv2.imwrite(sobel_mag_path, sobel_mag)
    cv2.imwrite(sobel_bin_path, sobel_bin)

    sobel_stats = compute_edge_stats(sobel_bin)

    # -------------------------------
    # Canny Edge Detection
    # -------------------------------
    canny = canny_edge(gray, low_t, high_t)

    canny_path = os.path.join(
        output_dir,
        f"{base}_canny_{low_t}_{high_t}.png"
    )

    cv2.imwrite(canny_path, canny)

    canny_stats = compute_edge_stats(canny)

    # -------------------------------
    # Save comparison report
    # -------------------------------
    save_stats_report(output_dir, base, sobel_stats, canny_stats)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Edge detection comparison: Sobel vs Canny"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--low",
        type=int,
        default=80,
        help="Canny low threshold (default=80)"
    )

    parser.add_argument(
        "--high",
        type=int,
        default=150,
        help="Canny high threshold (default=150)"
    )

    args = parser.parse_args()

    run_pipeline(args.image, args.low, args.high)
