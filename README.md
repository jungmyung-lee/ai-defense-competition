# AI Defense Competition — Experimental Learning & Technical Preparation

This repository, **ai-defense-competition**, is a consolidated collection of six experiment-driven technical toolkits I built to prepare for the **AI Defense Competition**, where participants must solve several analytical problems within a **90-minute time-limited environment** using large-scale CSV and image-based datasets.

The main goals of this repository are to:

- build foundational skills for **fast, structured experimentation**
- practice **error-reduction and accuracy-improvement under time pressure**
- analyze data and models using **numeric evidence rather than intuition**
- strengthen **implementation readiness** for real-world ML environments

Although the actual competition environment, datasets, and test questions are confidential, this repository captures the technical study process I went through to simulate realistic workflows and prepare for:

- preprocessing and feature engineering on large CSV files
- classical CV and YOLO-based detection pipelines
- baseline modeling, evaluation, and error analysis under constraints

---

## Table of Contents

- [Background & Motivation](#background--motivation)
- [Overall Repository Structure](#overall-repository-structure)
- [Toolkit Summaries](#toolkit-summaries)
- [01 — NumPy & OpenCV Image Basics](#01--numpy--opencv-image-basics)
- [02 — Image Segmentation & Preprocessing](#02--image-segmentation--preprocessing)
- [03 — YOLOv8 Object Detection Toolkit](#03--yolov8-object-detection-toolkit)
- [04 — Data Science & Feature Engineering](#04--data-science--feature-engineering)
- [05 — Model Baseline & Evaluation](#05--model-baseline--evaluation)
- [06 — Text Classification & Model Tuning](#06--text-classification--model-tuning)
- [How This Repository Was Used for Competition Practice](#️-how-this-repository-was-used-for-competition-practice)
- [Limitations](#️-limitations)
- [Reflections & Learning Outcomes](#-reflections--learning-outcomes)
- [Future Extensions](#-future-extensions)

---

## Background & Motivation

The AI Defense Competition requires participants to:

- process **large-scale structured signals (primarily CSV)**
- sometimes combine them with **image-based operational data**
- and produce accurate predictions or analyses under a strict  
  **90-minute time limit for three problems**

In that setting, performance depends heavily on:

- how quickly you can understand the task
- how efficiently you can build and debug pipelines
- how effectively you can **reduce error rate in a short time**

This repository was created to simulate that pressure by:

- practicing end-to-end workflows  
  (**raw data → preprocessing → modeling → evaluation**)
- emphasizing **reproducible experiment logs** over ad-hoc scripts
- treating every step as something that must be **explainable and defensible**

---

## Motivation for Defense-Specialized Model Adaptation

One recurring theme in the competition setting was:

> “How do we adapt generic models (e.g., standard YOLO) to **defense-specific objects and classes?”**

Example strategic questions I explored:

- how to retrain a general YOLO model so it can  
  **distinguish tanks, helicopters, and ground vehicles** not present in the original dataset
- how to design a dataset and label scheme to separate  
  **South Korean vs North Korean soldiers / allied vs enemy assets**
- how to structure CSV-based detection logs so that  
  **downstream decision logic stays simple and robust**

In this repository, those ideas appear mainly as:

- data-pipeline design practice (logging, CSV export, evaluation)
- image-processing experiments for classical CV interpretation
- YOLO logging & post-processing utilities for structured analysis

These explorations served two purposes:

**1) Preparation for real implementation**

- understanding detectors, preprocessing, and metrics at a **code level**

**2) Problem-solving strategy design**

- thinking through how I would **adapt & train a defense-specific model** if given appropriate data

> Due to confidentiality, the actual competition data and adaptation details cannot be disclosed,  
> but this repository documents the foundational blocks I practiced beforehand.

---

## Overall Repository Structure

Each folder is an independent, experiment-ready toolkit:

01_numpy_opencv_image_basics/  
02_image_segmentation_and_preprocessing/  
03_yolov8_object_detection/  
04_data_science_feature_engineering/  
05_model_baseline_and_evaluation/  
06_text_classification_and_model_tuning/  

Common design principles: 

- **CLI-first** — every script runs from terminal  
- structured outputs under `outputs/`  
- numeric reports + artifacts (images / CSVs / logs)  
- deterministic seeds where applicable  

---

## Toolkit Summaries

---

## 01 — NumPy & OpenCV Image Basics

Folder: `01_numpy_opencv_image_basics/`

Focuses on **pixel-level interpretation & quantitative analysis**:

- affine transformations (rotation, crop, aspect-preserving resize)
- brightness & contrast adjustment (α–β + auto stretch)
- Sobel vs Canny edge-detection comparison
- RGB channel histograms (256-bin)
- global / per-channel / masked-ROI statistics

Used to practice:

- reasoning in terms of **matrices & distributions**, not visuals only
- always saving **both artifacts and numeric reports**

---

## 02 — Image Segmentation & Preprocessing

Folder: `02_image_segmentation_and_preprocessing/`

Implements interpretation-focused classical CV preprocessing:

- Global / Otsu / Adaptive thresholding comparison
- Gaussian / Median / (optional) Bilateral denoising
- morphology ops (erosion, dilation, opening, closing)
- contour extraction + bounding-box statistics
- full preprocessing pipelines with step outputs

Designed to simulate:

- binary segmentation & document/object extraction
- evaluation using **pixel ratios & contour metrics**

---

## 03 — YOLOv8 Object Detection Toolkit

Folder: `03_yolov8_object_detection/`

Provides a practical YOLOv8 workflow:

- single-image inference + annotated output + CSV logs
- video / webcam detection with frame logs
- standardized bounding-box schema  
  *(source_id, class, confidence, x/y/w/h)*
- class-wise & frame-wise statistics
- automatic ROI cropping + metadata exports

Relevant for defense-style settings where:

- detections must be stored as **structured CSV**
- downstream systems may consume **numeric outputs only**

This establishes:

- how detections are logged
- how statistics are computed
- how crops/logs support future fine-tuning workflows

---

## 04 — Data Science & Feature Engineering

Folder: `04_data_science_feature_engineering/`

Toolkit for tabular data preparation:

- missing-value detection & cleaning
- numeric: mean / median / zero-fill
- categorical: mode / UNKNOWN token
- automated visual EDA (histograms, boxplots, heatmaps)
- Min-Max vs Standard scaling comparison
- Z-score outlier detection
- PCA dimensionality reduction

Used to practice:

- cleaning large CSV datasets similar to competition data
- understanding how preprocessing impacts model behavior

---

## 05 — Model Baseline & Evaluation

Folder: `05_model_baseline_and_evaluation/`

Implements a **baseline-first ML workflow**:

- Logistic Regression classifier baseline
- stratified K-Fold cross-validation (LogReg / RF)
- unified evaluation-report generator
- RandomForest w/ feature-importance export
- stratified train/test split reports

Key emphasis:

- macro & weighted metrics (not just accuracy)
- fast, inspectable artifacts for **90-minute workflows**

---

## 06 — Text Classification & Model Tuning

Folder: `06_text_classification_and_model_tuning/`

Lightweight interpretability-oriented NLP toolkit:

- deterministic text preprocessing
- BoW vs TF-IDF comparison
- Logistic Regression baseline
- Naive Bayes vs LogReg comparison
- small-scale hyperparameter search
- misclassification & hard-case inspection

Useful for:

- small datasets
- macro-F1 focused evaluation contexts

---

## How This Repository Was Used for Competition Practice

Constraints were similar to:

- receiving **large CSV inputs** with minimal inspection time
- needing to quickly choose & train a model
- reducing error **as efficiently as possible**
- sometimes reasoning about **defense-specific model adaptation**

These toolkits helped me practice:

- rapidly setting up preprocessing & feature engineering
- logging every step for fast debugging
- lowering error using **structured reasoning rather than guesswork**

The real competition input was primarily **large CSV data**,  
and these scripts made that environment feel familiar under time pressure.

---

## Limitations

This repository does **not** include:

- real competition datasets
- real test questions
- classified or restricted data

Defense-specific adaptation is discussed at:

- strategy level
- pipeline design level
- logging & evaluation design level

Some parts were explored as **pre-implementation plans**:

- dataset & annotation structuring
- detection-log schema design
- evaluation metric planning

The focus was:

> clarity, reproducibility, and implementation readiness  
> rather than leaderboard optimization.

---

## Reflections & Learning Outcomes

Through this repository I:

- became comfortable working under **strict time limits**
- developed a habit of building **experiment-ready scripts**
- learned to prioritize:
  - interpretable metrics
  - reproducible logs
  - structured experimentation

These principles carried into later projects.

---

## Influence on Later Socially Impactful Projects

Skills practiced here influenced:

- **LifeStep internship projects**
  - EMG-based real-time joint-angle regression
  - AI Basketball Shooting Coach (pose-based + explainable ML)

- inclusive biomechanics & sports-technology work
- education & community-oriented AI initiatives

Shared mindset:

> “Start with a clean, explainable pipeline —  
> then iterate using metrics, not guesses.”

---

## Future Extensions

Possible future work includes:

- stress-testing pipelines with noisy or incomplete data
- simulating compute-constrained environments
- experimenting with synthetic defense-style datasets
- extending logging into ultra-fast review dashboards
