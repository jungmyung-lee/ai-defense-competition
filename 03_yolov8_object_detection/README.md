# 03_yolov8_object_detection

A collection of **YOLOv8-based detection**, **logging**, and **post-processing** utilities implemented using **ultralytics YOLO**, **OpenCV**, **NumPy**, and **pandas**.

Each script runs as an independent CLI tool and produces both processed artifacts and **structured tabular outputs**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Learning Objectives](#learning-objectives)
- [Applications](#applications)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Scripts](#scripts)
  - [1) yolov8_inference_image.py](#1-yolov8_inference_imagepy)
  - [2) yolov8_video_detection.py](#2-yolov8_video_detectionpy)
  - [3) export_detection_boxes_to_csv.py](#3-export_detection_boxes_to_csvpy)
  - [4) object_counting_and_class_stats.py](#4-object_counting_and_class_statspy)
  - [5) crop_detected_objects.py](#5-crop_detected_objectspy)
- [Output Directory Structure](#output-directory-structure)
- [Dataset Notes](#dataset-notes)

---

## Project Overview

This project implements an end-to-end **YOLOv8 detection and analysis pipeline**, including:

- **single-image inference** with visualization + CSV export
- **video / webcam detection** with frame-wise logging
- **standardized bounding-box export** for downstream processing
- **class-level & frame-level statistics**
- **automatic object cropping from detections**

The toolkit emphasizes:

- reproducible **detection logs**
- standardized **data schemas**
- quantitative **post-processing analysis**
- seamless workflow from  
  **detection → export → analysis → cropping**

---

## Learning Objectives

- build a reusable **detection-pipeline engineering workflow**
- normalize YOLOv8 outputs into **clean, tabular datasets**
- analyze detections using **NumPy / pandas statistics**
- support dataset curation and ROI extraction

---

## Applications

- dataset inspection before model training
- detection quality auditing
- object-count monitoring in videos
- dataset curation / patch extraction
- tracking & post-processing pipelines

---

## Project Structure

**03_yolov8_object_detection/**  
│  
├── README.md  
├── yolov8_inference_image.py  
├── yolov8_video_detection.py  
├── export_detection_boxes_to_csv.py  
├── object_counting_and_class_stats.py  
└── crop_detected_objects.py  

All outputs are written under:

./outputs/

---

## Installation

pip install ultralytics opencv-python numpy pandas

**Headless environments:**

pip install opencv-python-headless

---

## Scripts

### 1) yolov8_inference_image.py

**Single-image YOLOv8 inference + CSV export**

**Saves:**

- annotated detection visualization
- detection table (CSV):
  - class_id / class_name  
  - confidence  
  - x, y, w, h  

**Run:**

python yolov8_inference_image.py
--image sample.jpg
--model yolov8n.pt

**Output:**

outputs/yolov8_inference/<image_name>/


---

### 2) yolov8_video_detection.py

**Video / webcam YOLOv8 detection + frame log**

**Performs:**

- frame-wise inference
- bounding box + label overlay
- optional annotated output video
- CSV detection log

**Run (video file):**

python yolov8_video_detection.py
--video sample.mp4
--model yolov8n.pt

**Run (webcam):**

python yolov8_video_detection.py --video 0

**Log-only mode:**

python yolov8_video_detection.py --video sample.mp4 --no_save

**Output:**

outputs/yolov8_video/<video_name>/


---

### 3) export_detection_boxes_to_csv.py

**Normalize detection logs into standardized bounding-box dataset**

**Supports:**

- image inference CSV
- video detection CSV
- earlier pipeline outputs

**Exports unified schema:**

- source_id  
- class_id / class_name  
- confidence  
- x, y, w, h  

**Run:**

python export_detection_boxes_to_csv.py
--csv sample_yolov8_detections.csv

**Output:**

outputs/detection_export/<result_name>/

**Includes:**

- bbox_export.csv  
- export summary report  

---

### 4) object_counting_and_class_stats.py

**Detection statistics & class distribution analysis**

**Computes:**

- total detections
- unique classes
- class-wise counts
- class-wise confidence stats
- frame-wise object distribution (if available)

**Run:**

python object_counting_and_class_stats.py
--csv sample_yolov8_detections.csv

**Output:**

outputs/detection_stats/<result_name>/

**Includes:**

- class_stats.csv  
- frame_stats.csv (video only)  
- analysis report  

---

### 5) crop_detected_objects.py

**Automatic ROI cropping from detection bounding boxes**

**For each detection:**

- crops ROI image patch
- stores crop metadata (CSV + TXT)

**Supports:**

- YOLO image CSV
- YOLO video CSV
- bbox_export CSV

**Run:**

python crop_detected_objects.py
--image sample.jpg
--csv sample_yolov8_detections.csv

**Output:**

outputs/object_crops/<source_name>/

**Includes:**

- cropped patches  
- crop_metadata.csv  
- crop_summary.txt  

---

## Output Directory Structure

outputs/
├─ yolov8_inference/  
├─ yolov8_video/  
├─ detection_export/  
├─ detection_stats/  
└─ object_crops/  

---

## Dataset Notes

This toolkit is intended for:

- dataset curation
- detection analysis
- ROI extraction workflows
- research & experimentation
