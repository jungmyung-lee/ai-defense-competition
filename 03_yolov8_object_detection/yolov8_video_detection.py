"""
yolov8_video_detection.py

Author: Jungmyung Lee

Description:
    Performs YOLOv8 object detection on a video stream:

    1) Reads video or webcam frames
    2) Runs YOLOv8 inference per frame
    3) Draws bounding boxes + class labels + confidence
    4) Saves:
        - Output annotated video
        - Frame-wise detection log (CSV)

Outputs:
    ./outputs/yolov8_video/<video_name>/

Python Dependencies:
    - ultralytics
    - OpenCV
    - NumPy
    - pandas

Run Example:
    python yolov8_video_detection.py --video sample.mp4 --model yolov8n.pt

Webcam Example:
    python yolov8_video_detection.py --video 0
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO


# =========================================================
# Directory handling
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# Build output directory from video name
# =========================================================
def get_output_dir(video_path):
    if str(video_path).isdigit():
        base = "webcam"
    else:
        base = os.path.splitext(os.path.basename(video_path))[0]

    output_dir = os.path.join("outputs", "yolov8_video", base)
    ensure_dir(output_dir)

    return base, output_dir


# =========================================================
# YOLO inference per frame
# =========================================================
def detect_objects(model, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = model(rgb, verbose=False)[0]

    detections = []

    for box in result.boxes:
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
# Draw bounding boxes
# =========================================================
def draw_detections(frame, detections):
    vis = frame.copy()

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
# Save CSV log
# =========================================================
def append_frame_log(log_rows, frame_idx, detections):
    for det in detections:
        row = {"frame": frame_idx}
        row.update(det)
        log_rows.append(row)


def save_detection_log(output_dir, base, log_rows):
    df = pd.DataFrame(log_rows)

    csv_path = os.path.join(output_dir, f"{base}_video_detections.csv")
    df.to_csv(csv_path, index=False)

    print(f"[+] Saved detection CSV → {csv_path}")


# =========================================================
# Video processing pipeline
# =========================================================
def process_video(video_path, model_path, write_output=True):
    base, output_dir = get_output_dir(video_path)

    print("\n[ YOLOv8 VIDEO DETECTION ]")
    print(f"Video : {video_path}")
    print(f"Model : {model_path}")
    print(f"Output: {output_dir}\n")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(int(video_path) if str(video_path).isdigit() else video_path)

    if not cap.isOpened():
        raise RuntimeError("[ERROR] Failed to open video source")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if write_output:
        out_path = os.path.join(output_dir, f"{base}_yolov8_output.mp4")
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )
        print(f"[+] Output video writer initialized → {out_path}")

    log_rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(model, frame)
        append_frame_log(log_rows, frame_idx, detections)

        vis = draw_detections(frame, detections)

        if write_output:
            writer.write(vis)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    save_detection_log(output_dir, base, log_rows)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv8 object detection on video stream"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file or webcam index (e.g., 0)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLOv8 model (default=yolov8n.pt)"
    )

    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save output video (log only)"
    )

    args = parser.parse_args()

    process_video(
        args.video,
        args.model,
        write_output=not args.no_save
    )
