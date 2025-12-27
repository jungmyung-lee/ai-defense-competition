"""
denoise_gaussian_median.py

Author: Jungmyung Lee

Description:
    Compares multiple noise reduction filters:
    1) Gaussian Blur
    2) Median Blur
    3) Bilateral Filter (optional)

    Computes:
    - Global pixel variance reduction
    - Edge-preservation score (Sobel gradient energy)

    Saves filtered images and analysis report.

Outputs:
    ./outputs/denoise/<image_name>/

Python Dependencies:
    - OpenCV
    - NumPy

Run Example:
    python denoise_gaussian_median.py --image sample.jpg --bilateral 1
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
# Load grayscale image
# =========================================================
def load_gray(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("[ERROR] Failed to load image. Possibly corrupted.")

    return img


# =========================================================
# Noise / edge statistics
# =========================================================
def compute_variance(img):
    return float(np.var(img))


def sobel_energy(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))


# =========================================================
# Apply filters
# =========================================================
def apply_filters(img, include_bilateral):
    results = {}
    stats = {}

    # Original baseline
    results["original"] = img
    stats["original"] = {
        "variance": compute_variance(img),
        "edge_energy": sobel_energy(img)
    }

    # Gaussian Blur
    gauss = cv2.GaussianBlur(img, (5, 5), sigmaX=1.2)
    results["gaussian"] = gauss
    stats["gaussian"] = {
        "variance": compute_variance(gauss),
        "edge_energy": sobel_energy(gauss)
    }

    # Median Blur
    median = cv2.medianBlur(img, 5)
    results["median"] = median
    stats["median"] = {
        "variance": compute_variance(median),
        "edge_energy": sobel_energy(median)
    }

    # Bilateral Filter (optional)
    if include_bilateral:
        bilateral = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
        results["bilateral"] = bilateral
        stats["bilateral"] = {
            "variance": compute_variance(bilateral),
            "edge_energy": sobel_energy(bilateral)
        }

    return results, stats


# =========================================================
# Save analysis report
# =========================================================
def save_report(output_dir, base, stats):
    path = os.path.join(output_dir, f"{base}_denoise_analysis.txt")

    # baseline: original 이미지의 variance / edge_energy
    orig_var = stats["original"]["variance"]
    orig_edge = stats["original"]["edge_energy"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("DENOISING FILTER COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        for name, s in stats.items():
            f.write(f"[{name}]\n")
            f.write(f" - variance     : {s['variance']}\n")
            f.write(f" - edge_energy  : {s['edge_energy']}\n")

            # original은 기준값이니까 reduction은 계산하지 않음
            if name != "original":
                var_reduction = orig_var - s["variance"]
                # 0으로 나누기 방지
                if orig_var != 0:
                    var_reduction_ratio = var_reduction / orig_var
                else:
                    var_reduction_ratio = 0.0

                f.write(f" - variance_reduction       : {var_reduction}\n")
                f.write(f" - variance_reduction_ratio : {var_reduction_ratio}\n")

            f.write("\n")

        f.write("Interpretation Guide:\n")
        f.write(" - Lower variance  = stronger noise suppression\n")
        f.write(" - Higher edge_energy = better edge preservation\n")
        f.write(" - variance_reduction / ratio are computed w.r.t. the original image.\n")

    print(f"[+] Saved denoise analysis → {path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(image_path, include_bilateral):
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("outputs", "denoise", base)

    ensure_dir(output_dir)

    print("\n[ DENOISING FILTER PIPELINE ]")
    print(f"Input  : {image_path}")
    print(f"Output : {output_dir}\n")

    img = load_gray(image_path)

    # Save original
    orig_path = os.path.join(output_dir, f"{base}_original.png")
    cv2.imwrite(orig_path, img)

    # Apply filters
    results, stats = apply_filters(img, include_bilateral)

    # Save filtered outputs
    for name, out in results.items():
        path = os.path.join(output_dir, f"{base}_{name}.png")
        cv2.imwrite(path, out)

    # Save report
    save_report(output_dir, base, stats)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Noise reduction filter comparison pipeline"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to grayscale image"
    )

    parser.add_argument(
        "--bilateral",
        type=int,
        default=0,
        help="Include bilateral filter? (1 = yes)"
    )

    args = parser.parse_args()

    run_pipeline(args.image, args.bilateral == 1)
