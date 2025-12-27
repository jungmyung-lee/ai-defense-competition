"""
brightness_contrast_adjust.py

Author: Jungmyung Lee

Description:
    Adjusts image brightness and contrast using:
    1) Manual α–β adjustment (y = αx + β)
    2) Automatic contrast stretching based on min–max normalization

    Saves processed images and per-channel numeric statistics.

Outputs:
    ./outputs/brightness_contrast/<image_name>/

Python Dependencies:
    - OpenCV
    - NumPy

Run Example:
    python brightness_contrast_adjust.py --image sample.jpg --alpha 1.2 --beta 15
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
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] File not found: {image_path}")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("[ERROR] Failed to load image. Possibly corrupted.")

    return img


# =========================================================
# α–β brightness / contrast adjustment
# y = αx + β
# =========================================================
def apply_alpha_beta(img, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# =========================================================
# Automatic Contrast Stretching
# scales pixel range → [0, 255]
# =========================================================
def auto_contrast_stretch(img):
    img_float = img.astype(np.float32)

    min_val = np.min(img_float)
    max_val = np.max(img_float)

    if max_val - min_val < 1e-6:
        return img.copy()

    stretched = (img_float - min_val) / (max_val - min_val)
    stretched = (stretched * 255.0).clip(0, 255).astype(np.uint8)

    return stretched


# =========================================================
# Channel statistics for report
# =========================================================
def compute_stats(img):
    stats = {}
    channels = cv2.split(img)
    names = ["B", "G", "R"]

    for name, ch in zip(names, channels):
        flat = ch.flatten()

        stats[name] = {
            "min": int(np.min(flat)),
            "max": int(np.max(flat)),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "p25": int(np.percentile(flat, 25)),
            "p50": int(np.percentile(flat, 50)),
            "p75": int(np.percentile(flat, 75)),
            "num_pixels": int(len(flat))
        }

    return stats


# =========================================================
# Save stats to text file
# =========================================================
def save_stats_report(output_dir, base_name, tag, stats):
    path = os.path.join(output_dir, f"{base_name}_{tag}_stats.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("IMAGE BRIGHTNESS / CONTRAST REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        for ch, s in stats.items():
            f.write(f"[{ch} Channel]\n")
            for k, v in s.items():
                f.write(f" - {k}: {v}\n")
            f.write("\n")

    print(f"[+] Saved report → {path}")


# =========================================================
# Processing Pipeline
# =========================================================
def run_pipeline(image_path, alpha, beta):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("outputs", "brightness_contrast", base_name)

    ensure_dir(output_dir)

    print("\n[ BRIGHTNESS / CONTRAST PROCESSING ]")
    print(f"Input  : {image_path}")
    print(f"Output : {output_dir}\n")

    img = load_image(image_path)

    # Save original
    orig_path = os.path.join(output_dir, f"{base_name}_original.png")
    cv2.imwrite(orig_path, img)

    # Manual α–β adjustment
    manual = apply_alpha_beta(img, alpha=alpha, beta=beta)

    manual_path = os.path.join(
        output_dir,
        f"{base_name}_alpha{alpha}_beta{beta}.png"
    )
    cv2.imwrite(manual_path, manual)

    manual_stats = compute_stats(manual)
    
    manual_tag = f"alpha{alpha}_beta{beta}"
    save_stats_report(output_dir, base_name, manual_tag, manual_stats)

    # Automatic Contrast Stretching
    auto = auto_contrast_stretch(img)

    auto_path = os.path.join(output_dir, f"{base_name}_auto_stretch.png")
    cv2.imwrite(auto_path, auto)

    auto_stats = compute_stats(auto)
    save_stats_report(output_dir, base_name, "auto_stretch", auto_stats)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adjust brightness/contrast (αβ + automatic contrast stretching)"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.2,
        help="Contrast scale factor (default=1.2)"
    )

    parser.add_argument(
        "--beta",
        type=int,
        default=15,
        help="Brightness offset (default=15)"
    )

    args = parser.parse_args()

    run_pipeline(args.image, args.alpha, args.beta)
