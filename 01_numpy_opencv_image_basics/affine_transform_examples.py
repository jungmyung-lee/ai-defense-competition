"""
affine_transform_examples.py

Author: Jungmyung Lee

Description:
    Applies common affine transformations to an input image, including:
    1) Fixed-angle rotation (center pivot)
    2) Safe-crop rotation (auto-fit, no clipping)
    3) Center crop and region crop
    4) Uniform + aspect-preserving resize

    Saves transformed images along with basic dimension statistics.

Outputs:
    ./outputs/affine_transform/<image_name>/

Python Dependencies:
    - OpenCV
    - NumPy

Run Example:
    python affine_transform_examples.py --image sample.jpg --angle 25 --crop 256 --resize 512
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
# Rotation (center pivot, possible clipping)
# =========================================================
def rotate_image_fixed(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR)

    return rotated


# =========================================================
# Rotation — auto expand canvas to avoid cropping
# =========================================================
def rotate_image_safe(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(img, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR)

    return rotated


# =========================================================
# Center Crop (square)
# =========================================================
def center_crop(img, crop_size):
    h, w = img.shape[:2]

    if crop_size > min(h, w):
        crop_size = min(h, w)

    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2

    return img[start_y:start_y+crop_size, start_x:start_x+crop_size]


# =========================================================
# Region Crop (x,y,w,h)
# =========================================================
def region_crop(img, x, y, cw, ch):
    h, w = img.shape[:2]

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    x2 = min(x + cw, w)
    y2 = min(y + ch, h)

    return img[y:y2, x:x2]


# =========================================================
# Resize — keep aspect ratio, pad to square
# =========================================================
def resize_preserve_aspect(img, target):
    h, w = img.shape[:2]
    scale = target / max(h, w)

    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    pad_top = (target - resized.shape[0]) // 2
    pad_bottom = target - resized.shape[0] - pad_top
    pad_left = (target - resized.shape[1]) // 2
    pad_right = target - resized.shape[1] - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return padded


# =========================================================
# Save Dimension Report
# =========================================================
def save_dim_report(output_dir, base, tag, img):
    path = os.path.join(output_dir, f"{base}_{tag}_dims.txt")
    h, w = img.shape[:2]

    with open(path, "w", encoding="utf-8") as f:
        f.write("IMAGE TRANSFORMATION DIMENSION REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Width  : {w}\n")
        f.write(f"Height : {h}\n")
        f.write(f"Shape  : {img.shape}\n")

    print(f"[+] Saved dim report → {path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(image_path, angle, crop_size, resize_size):
    base = os.path.splitext(os.path.basename(image_path))[0]
    output = os.path.join("outputs", "affine_transform", base)

    ensure_dir(output)

    print("\n[ AFFINE TRANSFORM PROCESSING ]")
    print(f"Input  : {image_path}")
    print(f"Output : {output}\n")

    img = load_image(image_path)

    # Save original
    orig_path = os.path.join(output, f"{base}_original.png")
    cv2.imwrite(orig_path, img)

    # -------------------------------
    # Rotation (fixed canvas)
    # -------------------------------
    rot_fixed = rotate_image_fixed(img, angle)
    path_fixed = os.path.join(output, f"{base}_rotate_fixed_{angle}.png")
    cv2.imwrite(path_fixed, rot_fixed)
    save_dim_report(output, base, "rotate_fixed", rot_fixed)

    # -------------------------------
    # Rotation (safe, no clipping)
    # -------------------------------
    rot_safe = rotate_image_safe(img, angle)
    path_safe = os.path.join(output, f"{base}_rotate_safe_{angle}.png")
    cv2.imwrite(path_safe, rot_safe)
    save_dim_report(output, base, "rotate_safe", rot_safe)

    # -------------------------------
    # Center Crop
    # -------------------------------
    cropped_center = center_crop(img, crop_size)
    crop_path = os.path.join(output, f"{base}_center_crop_{crop_size}.png")
    cv2.imwrite(crop_path, cropped_center)
    save_dim_report(output, base, "center_crop", cropped_center)

    # -------------------------------
    # Example region crop
    # -------------------------------
    region = region_crop(img, x=20, y=20, cw=crop_size, ch=crop_size)
    reg_path = os.path.join(output, f"{base}_region_crop.png")
    cv2.imwrite(reg_path, region)
    save_dim_report(output, base, "region_crop", region)

    # -------------------------------
    # Resize (aspect preserved + padding)
    # -------------------------------
    resized = resize_preserve_aspect(img, resize_size)
    res_path = os.path.join(output, f"{base}_resize_{resize_size}.png")
    cv2.imwrite(res_path, resized)
    save_dim_report(output, base, "resize_square", resized)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Affine transform examples: rotation / crop / resize"
    )

    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")

    parser.add_argument("--angle", type=float, default=30,
                        help="Rotation angle (default=30)")

    parser.add_argument("--crop", type=int, default=256,
                        help="Crop size (default=256)")

    parser.add_argument("--resize", type=int, default=512,
                        help="Output square size (default=512)")

    args = parser.parse_args()

    run_pipeline(args.image, args.angle, args.crop, args.resize)
