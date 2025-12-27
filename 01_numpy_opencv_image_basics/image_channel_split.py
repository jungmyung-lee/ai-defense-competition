"""
image_channel_split.py

Author: Jungmyung Lee

Description:
    Loads an input image, converts BGR → RGB, splits color channels,
    computes per-channel statistics & histograms, and saves analysis
    results as images + text reports.

Outputs:
    ./outputs/image_channel_analysis/<image_name>/

Python Dependencies:
    - OpenCV
    - NumPy

Run Example:
    python image_channel_split.py --image sample.jpg
"""

import os
import cv2
import argparse
import numpy as np
from datetime import datetime


# =========================================================
# Directory Utilities
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# Load Image
# =========================================================
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image not found: {image_path}")

    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
            raise ValueError("[ERROR] Failed to load image. Possibly corrupted.")

    return img_bgr


# =========================================================
# BGR → RGB
# =========================================================
def convert_bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# =========================================================
# Split channels
# =========================================================
def split_channels(img_rgb):
    r, g, b = cv2.split(img_rgb)

    return {
        "R": r,
        "G": g,
        "B": b
    }


# =========================================================
# Compute statistics
# =========================================================
def compute_channel_stats(channel):
    flat = channel.flatten()

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
# Compute histogram using NumPy only
# =========================================================
def compute_histogram(channel):
    flat = channel.flatten()

    # 256-bin grayscale histogram over integer intensity range 0–255
    # Using range (0, 256) ensures bin alignment with discrete pixel values
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))

    # Store normalized histogram ratio distribution
    hist_ratio = hist / np.sum(hist)

    return hist, hist_ratio




# =========================================================
# Save channel visualization images
# =========================================================
def save_channel_images(img_rgb, channels, output_dir, base_name):
    zeros = np.zeros_like(list(channels.values())[0])

    red_img   = np.dstack([channels["R"], zeros, zeros])
    green_img = np.dstack([zeros, channels["G"], zeros])
    blue_img  = np.dstack([zeros, zeros, channels["B"]])

    cv2.imwrite(os.path.join(output_dir, f"{base_name}_R.png"),
                cv2.cvtColor(red_img, cv2.COLOR_RGB2BGR))

    cv2.imwrite(os.path.join(output_dir, f"{base_name}_G.png"),
                cv2.cvtColor(green_img, cv2.COLOR_RGB2BGR))

    cv2.imwrite(os.path.join(output_dir, f"{base_name}_B.png"),
                cv2.cvtColor(blue_img, cv2.COLOR_RGB2BGR))

    print("[+] Saved channel visualization images")


# =========================================================
# Save numeric results as text + CSV
# =========================================================
def save_reports(stats_dict, hist_dict, output_dir, base_name):

    # ---------- Text Report ----------
    txt_path = os.path.join(output_dir, f"{base_name}_channel_stats.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("IMAGE CHANNEL ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        for ch in ["R", "G", "B"]:
            stats = stats_dict[ch]

            f.write(f"[{ch} Channel]\n")
            for k, v in stats.items():
                f.write(f" - {k}: {v}\n")
            f.write("\n")

    print(f"[+] Saved stats report → {txt_path}")

    # ---------- CSV Histogram Export ----------
    csv_path = os.path.join(output_dir, f"{base_name}_histograms.csv")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("intensity,R_count,G_count,B_count,R_ratio,G_ratio,B_ratio\n")

        for i in range(256):
            r_c, g_c, b_c = hist_dict["R"][0][i], hist_dict["G"][0][i], hist_dict["B"][0][i]
            r_r, g_r, b_r = hist_dict["R"][1][i], hist_dict["G"][1][i], hist_dict["B"][1][i]

            f.write(f"{i},{r_c},{g_c},{b_c},{r_r:.6f},{g_r:.6f},{b_r:.6f}\n")

    print(f"[+] Saved histogram CSV → {csv_path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("outputs", "image_channel_analysis", base_name)

    ensure_dir(output_dir)

    print("\n[ RUNNING IMAGE CHANNEL ANALYSIS ]")
    print(f"Input  : {image_path}")
    print(f"Output : {output_dir}\n")

    img_bgr = load_image(image_path)
    img_rgb = convert_bgr_to_rgb(img_bgr)

    # save converted image
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_rgb.png"),
                cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    channels = split_channels(img_rgb)

    # stats + histogram
    stats_dict = {}
    hist_dict = {}

    for name, ch in channels.items():
        stats_dict[name] = compute_channel_stats(ch)
        hist_dict[name] = compute_histogram(ch)

    save_channel_images(img_rgb, channels, output_dir, base_name)
    save_reports(stats_dict, hist_dict, output_dir, base_name)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load image → RGB convert → channel split → histogram & stats export"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    args = parser.parse_args()

    run_pipeline(args.image)
