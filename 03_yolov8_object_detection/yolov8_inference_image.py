"""
yolov8_inference_image.py

Author: Jungmyung Lee

Description:
    Performs YOLOv8 object detection inference on a single image and saves:

    1) Visualization with bounding boxes + labels
    2) JSON-style detection result table (CSV)

    Outputs include:
    - class name
    - confidence score
    - bounding box (x, y, w, h)

Outputs:
    ./outputs/yolov8_inference/<image_name>/

Python Dependencies:
    - ultralytics
    - OpenCV
    - NumPy
    - pandas

Run Example:
    python yolov8_inference_image.py --image sample.jpg --model yolov8n.pt
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO


# =========================================================
# Directory Handling
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# Load Image (BGR)
# =========================================================
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Image not found: {path}")

    img = cv2.imread(path)

    if img is None:
        raise ValueError("[ERROR] Failed to load image")

    return img


# =========================================================
# Run YOLOv8 Inference
# =========================================================
def run_yolo_inference(model_path, img_bgr):
    model = YOLO(model_path)

    # YOLO expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = model(img_rgb, verbose=False)[0]

    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        w = x2 - x1
        h = y2 - y1

        detections.append({
            "class_id": cls_id,
            "class_name": model.names[cls_id],
            "confidence": round(conf, 6),
            "x": int(x1),
            "y": int(y1),
            "w": int(w),
            "h": int(h)
        })

    return detections


# =========================================================
# Draw Detection Visualization
# =========================================================
def draw_detections(img_bgr, detections):
    vis = img_bgr.copy()

    for det in detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]

        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(
            vis,
            label,
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return vis


# =========================================================
# Save Results
# =========================================================
def save_results(output_dir, base, vis_img, detections):
    # Save visualization
    vis_path = os.path.join(output_dir, f"{base}_yolov8_result.png")
    cv2.imwrite(vis_path, vis_img)

    # Save detection table
    df = pd.DataFrame(detections)
    csv_path = os.path.join(output_dir, f"{base}_yolov8_detections.csv")
    df.to_csv(csv_path, index=False)

    print(f"[+] Saved visualization → {vis_path}")
    print(f"[+] Saved detection CSV → {csv_path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(image_path, model_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("outputs", "yolov8_inference", base)

    ensure_dir(output_dir)

    print("\n[ YOLOv8 INFERENCE PIPELINE ]")
    print(f"Image : {image_path}")
    print(f"Model : {model_path}")
    print(f"Output: {output_dir}\n")

    img = load_image(image_path)

    detections = run_yolo_inference(model_path, img)

    vis = draw_detections(img, detections)

    save_results(output_dir, base, vis, detections)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv8 object detection inference on a single image"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLOv8 model (default=yolov8n.pt)"
    )

    args = parser.parse_args()

    run_pipeline(args.image, args.model)
