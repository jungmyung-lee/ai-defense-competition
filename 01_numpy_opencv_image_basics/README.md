# 01_numpy_opencv_image_basics

A collection of practical and analysis-driven image-processing scripts implemented using **NumPy-based matrix operations** and **OpenCV utilities**.

This project was designed as an exploratory toolkit for:

- understanding how preprocessing operations affect pixel-level statistics
- building reproducible and experiment-oriented processing pipelines
- evaluating classical computer-vision operations using quantitative metrics
- inspecting datasets prior to ML / CV model development

Each script runs as an independent CLI tool and generates both processed image artifacts and **numeric analysis reports**.

---

## Project Overview

This project implements several foundational — yet interpretation-focused — image-processing operations, including:

- **Affine transformations** (rotation, cropping, aspect-preserving resize)
- **Brightness & contrast adjustment** (α–β scaling + automatic stretching)
- **Sobel vs Canny edge-detection comparison**
- **RGB channel decomposition with histogram export**
- **Global / per-channel / masked-region pixel statistics**

The toolkit emphasizes:

- pixel-level understanding  
- interpretable statistics  
- reproducible outputs  
- experiment-driven comparison  

---

## Learning Objectives

This project was developed to:

### Strengthen interpretation-focused image-processing skills

- design an interpretable image-processing toolkit using **NumPy + OpenCV**
- analyze how pixel-intensity distributions change across transformations
- compare classical CV operations using **numeric metrics instead of visuals alone**

### Develop reproducible experimental workflows

- generate consistent output structures across runs
- store quantitative reports alongside processed artifacts
- treat preprocessing as a **measurable and explainable** step in ML pipelines

### Reinforce core foundations

- image representation & channel structure
- matrix-based computation
- statistical interpretation of pixel distributions

---

## Applications

This toolkit can be used for:

- dataset inspection prior to model training
- verifying preprocessing consistency across experiments
- image-quality diagnostics & artifact detection
- quantitative comparison of transformation effects
- exploratory study of classical computer vision
- lightweight research prototyping environments

This project serves as an **experimental platform for numerically interpreting image transformations**.

---

## Project Structure

**01_numpy_opencv_image_basics/**
│  
├── README.md  
├── affine_transform_examples.py  
├── brightness_contrast_adjust.py  
├── edge_detection_sobel_canny.py  
├── image_channel_split.py  
└── numpy_pixel_statistics.py  

**All scripts output results under:**

./outputs/

---

## Installation

pip install opencv-python numpy

**Server / headless environments:**

pip install opencv-python-headless

---

## Scripts

### 1) affine_transform_examples.py

**Affine Transform — Rotation / Crop / Aspect-Preserving Resize**

Applies common affine operations and saves:

- processed images  
- dimension / shape reports  

**Includes:**

- fixed-canvas rotation (may clip)
- safe rotation with expanded canvas
- center crop (square)
- region crop (x, y, w, h)
- aspect-preserving resize + padding

**Run:**

python affine_transform_examples.py
--image sample.jpg --angle 25 --crop 256 --resize 512

**Output:**

outputs/affine_transform/<image_name>/

---

### 2) brightness_contrast_adjust.py

**Brightness & Contrast Adjustment (α–β + Auto Stretch)**

**Performs:**

- manual α–β contrast scaling
- automatic min–max contrast stretching
- per-channel numeric statistics export

**Run:**

python brightness_contrast_adjust.py
--image sample.jpg --alpha 1.2 --beta 15

**Output:**

outputs/brightness_contrast/<image_name>/

---

### 3) edge_detection_sobel_canny.py

**Edge Detection — Sobel vs Canny (Quantitative Comparison)**

**Computes:**

- Sobel gradient magnitude
- Sobel binary edge map
- Canny edge map
- edge-density statistics

**Run:**

python edge_detection_sobel_canny.py
--image sample.jpg --low 80 --high 150

**Output:**

outputs/edge_detection/<image_name>/

---

### 4) image_channel_split.py

**RGB Channel Split + Histogram + Statistics Export**

**Performs:**

- BGR → RGB conversion
- channel separation (R / G / B)
- per-channel numeric statistics
- NumPy-based 256-bin histogram
- CSV export

**Run:**

python image_channel_split.py --image sample.jpg

**Output:**

outputs/image_channel_analysis/<image_name>/

---

### 5) numpy_pixel_statistics.py

**Global / Channel / Masked-ROI Pixel Statistics**

**Computes statistics over:**

1. whole image  
2. individual channels  
3. optional ROI mask region  

**Run:**

python numpy_pixel_statistics.py
--image sample.jpg --mask 1

**Output:**

outputs/pixel_statistics/<image_name>/

---

## Output Directory Structure

outputs/  
├─ affine_transform/  
├─ brightness_contrast/  
├─ edge_detection/  
├─ image_channel_analysis/  
└─ pixel_statistics/  

Each run produces:

- processed artifacts  
- numeric summary reports  
- reproducible experiment logs  

---

## Dataset Notes

This toolkit is intended for:

- dataset inspection prior to model development
- preprocessing behavior analysis
- research-driven experimentation
- educational exploration of classical CV techniques

Supported data types include:

- natural images
- scanned documents
- experiment / research images
- ML training datasets
