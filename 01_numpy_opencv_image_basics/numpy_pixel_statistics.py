"""
numpy_pixel_statistics.py

Author: Jungmyung Lee

Description:
    Computes matrix-level pixel statistics for an input image, including:
    1) Global statistics (whole image)
    2) Per-channel statistics (B / G / R)
    3) Masked-region statistics (user-defined ROI mask)

    Supports mean, std, min, max, and percentile values.

Outputs:
    ./outputs/pixel_statistics/<image_name>/

Python Dependencies:
    - OpenCV
    - NumPy

Run Example:
    python numpy_pixel_statistics.py --image sample.jpg --mask 1
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
# Compute statistics for a given matrix
# =========================================================
def compute_stats(arr):
    flat = arr.flatten()

    return {
        "min": int(np.min(flat)),
        "max": int(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p25": int(np.percentile(flat, 25)),
        "p50": int(np.percentile(flat, 50)),
        "p75": int(np.percentile(flat, 75)),
        "num_pixels": int(len(flat))
    }


# =========================================================
# Compute stats for mask region only
# =========================================================
def compute_mask_stats(img, mask):
    masked_pixels = img[mask > 0]

    if len(masked_pixels) == 0:
        return None

    return compute_stats(masked_pixels)


# =========================================================
# Example mask generator (center square ROI)
# =========================================================
def generate_center_mask(img, ratio=0.5):
    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    size = int(min(h, w) * ratio)

    x1 = (w - size) // 2
    y1 = (h - size) // 2

    mask[y1:y1+size, x1:x1+size] = 255

    return mask

# =========================================================
# Example rect mask generator (user-defined ROI: x, y, w, h)
# =========================================================
def generate_rect_mask(img, x, y, w, h):
    h_img, w_img = img.shape[:2]

    mask = np.zeros((h_img, w_img), dtype=np.uint8)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(x + w, w_img)
    y2 = min(y + h, h_img)

    if x1 >= x2 or y1 >= y2:
        
        return mask

    mask[y1:y2, x1:x2] = 255
    return mask



# =========================================================
# Save statistics report
# =========================================================
def save_stats_report(output_dir, base, global_stats, channel_stats, mask_stats):
    path = os.path.join(output_dir, f"{base}_pixel_stats.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("NUMPY IMAGE PIXEL STATISTICS REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        # -------------------
        # Global
        # -------------------
        f.write("[Global Image Statistics]\n")
        for k, v in global_stats.items():
            f.write(f" - {k}: {v}\n")

        # -------------------
        # Channels
        # -------------------
        f.write("\n[Per-Channel Statistics]\n")
        for name, stats in channel_stats.items():
            f.write(f"\n[{name} Channel]\n")
            for k, v in stats.items():
                f.write(f" - {k}: {v}\n")

        # -------------------
        # Mask ROI
        # -------------------
        if mask_stats is not None:
            f.write("\n[Masked Region Statistics]\n")
            for k, v in mask_stats.items():
                f.write(f" - {k}: {v}\n")
        else:
            f.write("\n[Masked Region Statistics]\n")
            f.write(" - No valid masked pixels\n")

    print(f"[+] Saved stats report → {path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(image_path, mask_mode, roi_x=0, roi_y=0, roi_w=0, roi_h=0):
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("outputs", "pixel_statistics", base)

    ensure_dir(output_dir)

    print("\n[ NUMPY PIXEL STATISTICS PROCESSING ]")
    print(f"Input  : {image_path}")
    print(f"Output : {output_dir}\n")

    img = load_image(image_path)

    # Save reference
    orig_path = os.path.join(output_dir, f"{base}_original.png")
    cv2.imwrite(orig_path, img)

    # -------------------------------
    # Global image stats
    # -------------------------------
    global_stats = compute_stats(img)

    # -------------------------------
    # Channel-wise stats
    # -------------------------------
    names = ["B", "G", "R"]
    channels = cv2.split(img)

    channel_stats = {}

    for name, ch in zip(names, channels):
        channel_stats[name] = compute_stats(ch)

    # -------------------------------
    # Mask ROI stats (optional)
    # -------------------------------
    mask_stats = None

    if mask_mode == 1:
        # Center ROI mask
        mask = generate_center_mask(img, ratio=0.4)

        mask_path = os.path.join(output_dir, f"{base}_mask_center.png")
        cv2.imwrite(mask_path, mask)

        mask_stats = compute_mask_stats(img, mask)

    elif mask_mode == 2:
        # User-defined rect ROI
        mask = generate_rect_mask(img, roi_x, roi_y, roi_w, roi_h)

        mask_path = os.path.join(output_dir, f"{base}_mask_user_rect.png")
        cv2.imwrite(mask_path, mask)

        mask_stats = compute_mask_stats(img, mask)


    # -------------------------------
    # Save report
    # -------------------------------
    save_stats_report(output_dir, base, global_stats, channel_stats, mask_stats)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute NumPy-based pixel statistics (global, channel, mask ROI)"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

        parser.add_argument(
        "--mask",
        type=int,
        default=0,
        help="Mask mode: 0 = no mask, 1 = center ROI mask, 2 = user-defined rectangle"
    )

    parser.add_argument(
        "--roi_x",
        type=int,
        default=0,
        help="Top-left x for user-defined ROI (used when --mask 2)"
    )

    parser.add_argument(
        "--roi_y",
        type=int,
        default=0,
        help="Top-left y for user-defined ROI (used when --mask 2)"
    )

    parser.add_argument(
        "--roi_w",
        type=int,
        default=0,
        help="Width for user-defined ROI (used when --mask 2)"
    )

    parser.add_argument(
        "--roi_h",
        type=int,
        default=0,
        help="Height for user-defined ROI (used when --mask 2)"
    )


        args = parser.parse_args()

        run_pipeline(
            args.image,
            mask_mode=args.mask,
            roi_x=args.roi_x,
            roi_y=args.roi_y,
            roi_w=args.roi_w,
            roi_h=args.roi_h,
        )
    
