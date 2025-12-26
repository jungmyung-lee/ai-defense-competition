# 02_image_segmentation_and_preprocessing

A collection of analysis-driven classical image-processing utilities implemented using **NumPy-based matrix operations** and **OpenCV pipelines**.

This toolkit focuses on:

- comparing multiple **thresholding algorithms**
- evaluating **noise-reduction filters** using quantitative metrics
- interpreting **morphological transformations** at the pixel level
- extracting **object-level contour statistics**
- demonstrating a **full preprocessing workflow**

Each script runs as an independent CLI tool and produces processed artifacts and **numeric analysis reports**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Learning Objectives](#learning-objectives)
- [Applications](#applications)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Scripts](#scripts)
  - [1) adaptive_thresholding_example.py](#1-adaptive_thresholding_examplepy)
  - [2) denoise_gaussian_median.py](#2-denoise_gaussian_medianpy)
  - [3) morphology_open_close.py](#3-morphology_open_closepy)
  - [4) object_contour_detection.py](#4-object_contour_detectionpy)
  - [5) preprocessing_pipeline_demo.py](#5-preprocessing_pipeline_demopy)
- [Output Directory Structure](#output-directory-structure)
- [Dataset Notes](#dataset-notes)

---

## Project Overview

This project implements several **interpretation-focused image-processing operations**, including:

- **Global / Otsu / Adaptive thresholding comparison**
- **Gaussian / Median / Bilateral noise-filter evaluation**
- **Morphological operations** (erosion, dilation, opening, closing)
- **Object contour extraction with bounding-box statistics**
- **End-to-end preprocessing pipeline demonstration**

The toolkit emphasizes:

- pixel-level interpretation  
- quantitative comparison  
- reproducible outputs  
- experiment-oriented workflows  

---

## Learning Objectives

This project was developed to:

### Strengthen interpretation-focused preprocessing skills

- compare **thresholding strategies** using numeric outcome metrics
- analyze **variance reduction vs edge preservation** in denoising filters
- interpret **morphology effects** on binary pixel distributions

### Develop reproducible experimentation workflows

- generate consistent structured outputs
- store quantitative reports with artifacts
- treat preprocessing as a **measurable and explainable** stage

### Reinforce core CV fundamentals

- binary image representation
- contour geometry interpretation
- metric-based preprocessing evaluation

---

## Applications

This toolkit can be used for:

- dataset quality inspection prior to model training
- preprocessing design and pipeline validation
- binary segmentation pre-analysis
- document / object extraction workflows
- research-driven experimentation and study

This project serves as an **analysis platform for understanding classical CV preprocessing behavior**.

---

## Project Structure

**02_image_thresholding_denoise_morphology/**
│  
├── adaptive_thresholding_example.py  
├── denoise_gaussian_median.py  
├── morphology_open_close.py  
├── object_contour_detection.py  
└── preprocessing_pipeline_demo.py  

**All scripts output results under:**

./outputs/

---

## Installation

pip install opencv-python numpy

**Server / headless environments:**

pip install opencv-python-headless

---

## Scripts

### 1) adaptive_thresholding_example.py

**Thresholding Comparison — Global / Otsu / Adaptive Mean / Adaptive Gaussian**

Applies multiple thresholding methods and exports:

- binarized images  
- pixel-ratio statistics  
- Otsu-selected threshold value  

**Run:**

python adaptive_thresholding_example.py  
--image sample.jpg --th 120

**Output:**

outputs/thresholding/<image_name>/

---

### 2) denoise_gaussian_median.py

**Noise Reduction — Gaussian / Median / (Optional) Bilateral Filter**

Computes:

- global pixel variance reduction
- Sobel-gradient edge-energy score

Exports filtered images and an interpretation-oriented analysis report.

**Run:**

python denoise_gaussian_median.py  
--image sample.jpg --bilateral 1

**Output:**

outputs/denoise/<image_name>/

---

### 3) morphology_open_close.py

**Morphology Operations — Erosion / Dilation / Opening / Closing**

Supports configurable:

- kernel size
- kernel shape (rect / ellipse / cross)

Generates pixel statistics:

- white / black pixel counts
- ratio comparison before / after transformation

**Run:**

python morphology_open_close.py  
--image sample.png --k 3 --shape rect

**Output:**

outputs/morphology/<image_name>/

---

### 4) object_contour_detection.py

**Object Contour Extraction + Bounding-Box Statistics**

Computes and exports:

- contour mask
- bounding-box visualization
- area-sorted contour list
- object-level statistics report

Supports automatic Otsu binarization if input is grayscale.

**Run:**

python object_contour_detection.py  
--image sample.png --min_area 100

**Output:**

outputs/contours/<image_name>/

---

### 5) preprocessing_pipeline_demo.py

**Full Image Preprocessing Pipeline Demonstration**

Pipeline stages:

1. grayscale load  
2. Gaussian denoise  
3. CLAHE contrast normalization  
4. Otsu thresholding  
5. morphology opening  
6. contour-based ROI extraction  

Exports intermediate step outputs and object statistics.

**Run:**

python preprocessing_pipeline_demo.py  
--image sample.jpg --min_area 80

**Output:**

outputs/preprocessing_pipeline/<image_name>/

---

## Output Directory Structure

outputs/  
├─ thresholding/  
├─ denoise/  
├─ morphology/  
├─ contours/  
└─ preprocessing_pipeline/  

Each run produces:

- processed artifacts  
- numeric comparison reports  
- reproducible experiment logs  

---

## Dataset Notes

This toolkit is intended for:

- preprocessing behavior analysis
- segmentation / binarization study
- research-driven dataset inspection
- classical CV experimentation

Supported input types:

- grayscale images
- binary images
- scanned documents
- experiment / research imagery
