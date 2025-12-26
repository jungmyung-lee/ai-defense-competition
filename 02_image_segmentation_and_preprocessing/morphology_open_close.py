"""
morphology_open_close.py

Author: Jungmyung Lee

Description:
    Applies and compares morphological operations on a binary or grayscale image:
    1) Erosion
    2) Dilation
    3) Opening  (erosion → dilation)
    4) Closing  (dilation → erosion)

    Supports configurable kernel size and shape,
    and generates pixel statistics before / after operations.

Outputs:
    ./outputs/morphology/<image_name>/

Python Dependencies:
    - OpenCV
    - NumPy

Run Example:
    python morphology_open_close.py --image sample.png --k 3 --shape rect
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
# Load image (supports binary or grayscale)
# =========================================================
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("[ERROR] Failed to load image. Possibly corrupted.")

    return img


# =========================================================
# Pixel statistics (binary-friendly)
# =========================================================
def compute_pixel_stats(img):
    flat = img.flatten()

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
# Build structuring element
# =========================================================
def build_kernel(size, shape):
    if shape == "rect":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    elif shape == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    elif shape == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))

    else:
        raise ValueError(f"[ERROR] Unknown kernel shape: {shape}")


# =========================================================
# Apply morphology operations
# =========================================================
def apply_morphology(img, kernel):
    results = {}
    stats = {}

    # Erosion
    erosion = cv2.erode(img, kernel, iterations=1)
    results["erosion"] = erosion
    stats["erosion"] = compute_pixel_stats(erosion)

    # Dilation
    dilation = cv2.dilate(img, kernel, iterations=1)
    results["dilation"] = dilation
    stats["dilation"] = compute_pixel_stats(dilation)

    # Opening = erosion → dilation
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    results["opening"] = opening
    stats["opening"] = compute_pixel_stats(opening)

    # Closing = dilation → erosion
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    results["closing"] = closing
    stats["closing"] = compute_pixel_stats(closing)

    return results, stats


# =========================================================
# Save statistics report
# =========================================================
def save_stats_report(output_dir, base, before_stats, after_stats):
    path = os.path.join(output_dir, f"{base}_morphology_stats.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("MORPHOLOGY OPERATION STATISTICS REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write("[Original Image]\n")
        for k, v in before_stats.items():
            f.write(f" - {k}: {v}\n")

        for name, stats in after_stats.items():
            f.write(f"\n[{name}]\n")
            for k, v in stats.items():
                f.write(f" - {k}: {v}\n")

    print(f"[+] Saved morphology statistics → {path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(image_path, k_size, shape):
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("outputs", "morphology", base)

    ensure_dir(output_dir)

    print("\n[ MORPHOLOGY PIPELINE ]")
    print(f"Input  : {image_path}")
    print(f"Output : {output_dir}")
    print(f"Kernel : size={k_size}, shape={shape}\n")

    img = load_image(image_path)

    # Save original
    orig_path = os.path.join(output_dir, f"{base}_original.png")
    cv2.imwrite(orig_path, img)

    before_stats = compute_pixel_stats(img)

    kernel = build_kernel(k_size, shape)

    # Apply ops
    results, after_stats = apply_morphology(img, kernel)

    # Save images
    for name, out in results.items():
        path = os.path.join(output_dir, f"{base}_{name}.png")
        cv2.imwrite(path, out)

    # Save report
    save_stats_report(output_dir, base, before_stats, after_stats)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Morphology operations: erosion / dilation / opening / closing"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to grayscale or binary image"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Kernel size (default=3)"
    )

    parser.add_argument(
        "--shape",
        type=str,
        default="rect",
        choices=["rect", "ellipse", "cross"],
        help="Kernel shape (rect | ellipse | cross)"
    )

    args = parser.parse_args()

    run_pipeline(args.image, args.k, args.shape)
