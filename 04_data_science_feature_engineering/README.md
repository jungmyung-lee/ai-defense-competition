# 04_data_analysis_preprocessing

A collection of **data preprocessing, EDA, scaling, PCA, and outlier-analysis** utilities implemented using **pandas**, **NumPy**, **matplotlib**, and **scikit-learn**.

Each script runs as an independent CLI tool and produces both processed datasets and **analysis-oriented reports**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Learning Objectives](#learning-objectives)
- [Applications](#applications)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Scripts](#scripts)
  - [1) data_cleaning_missing_values.py](#1-data_cleaning_missing_valuespy)
  - [2) eda_visual_summary.py](#2-eda_visual_summarypy)
  - [3) scaling_minmax_standard.py](#3-scaling_minmax_standardpy)
  - [4) zscore_outlier_detection.py](#4-zscore_outlier_detectionpy)
  - [5) pca_feature_reduction.py](#5-pca_feature_reductionpy)
- [Output Directory Structure](#output-directory-structure)
- [Dataset Notes](#dataset-notes)

---

## Project Overview

This project implements a modular, experiment-driven **data preparation & exploratory analysis toolkit**, including:

- automated **missing-value detection and cleaning**
- visual **EDA summary generation**
- feature-scaling comparison (Min-Max vs Standard)
- **Z-score outlier detection & filtering**
- PCA-based **dimensionality reduction**

The toolkit emphasizes:

- reproducible data cleaning workflows  
- explainable preprocessing decisions  
- statistically interpretable outputs  
- compatibility with ML model pipelines  

---

## Learning Objectives

- design structured preprocessing & EDA workflows
- analyze dataset quality before modeling
- compare scaling methods using quantitative reports
- evaluate outliers using statistical thresholds
- reduce feature dimensionality while retaining variance

---

## Applications

- dataset validation before ML training
- preprocessing pipeline prototyping
- experiment-ready feature engineering
- research & educational data exploration
- quantitative interpretation of preprocessing effects

---

## Project Structure

**04_data_analysis_preprocessing/**  
│  
├── README.md  
├── data_cleaning_missing_values.py  
├── eda_visual_summary.py  
├── pca_feature_reduction.py  
├── scaling_minmax_standard.py  
└── zscore_outlier_detection.py  

All outputs are stored under:

./outputs/

---

## Installation

pip install pandas numpy matplotlib scikit-learn

---

## Scripts

### 1) data_cleaning_missing_values.py

**Missing Value Detection & Cleaning Pipeline**

Handles missing values by feature type:

**- numeric columns:**
  - mean / median / zero-fill (selectable)
- categorical columns:
  - most-frequent fill
  - or `UNKNOWN` token assignment

**Saves:**

- cleaned dataset CSV
- missing-value summary report

**Run:**

python data_cleaning_missing_values.py
--csv sample.csv
--num_strategy mean
--cat_strategy mode

**Output:**

outputs/data_cleaning/<dataset_name>/

**Includes:**

- cleaned.csv  
- missing_value_report.txt  

---

### 2) eda_visual_summary.py

**Automated Visual EDA Summary Generator**

**Generates:**

- numeric histograms
- numeric boxplots
- categorical frequency plots
- correlation matrix heatmap
- textual dataset statistics report

**Run:**

python eda_visual_summary.py --csv sample.csv

**Output:**

outputs/eda_visual/<dataset_name>/

**Includes:**

- histogram images  
- boxplot images  
- category frequency plots  
- correlation heatmap  
- eda_report.txt  

---

### 3) scaling_minmax_standard.py

**Feature Scaling Comparison**

**Applies:**

1) Min-Max Scaling (0–1 normalization)  
2) Standard Scaling (Z-score normalization)

**Saves:**

- minmax_scaled.csv
- standard_scaled.csv
- scaling statistics report

**Run:**

python scaling_minmax_standard.py --csv sample.csv

**Output:**

outputs/feature_scaling/<dataset_name>/

**Includes:**

- minmax_scaled.csv  
- standard_scaled.csv  
- scaling_report.txt  

---

### 4) zscore_outlier_detection.py

**Z-Score Based Outlier Detection**

**Steps:**

1) compute column-wise Z-scores  
2) mark rows exceeding threshold  
3) split into:
   - outlier rows
   - cleaned dataset (outliers removed)

**Default rule:**

|z| >= 3.0

**Run:**

python zscore_outlier_detection.py
--csv sample.csv
--threshold 3.0

**Output:**

outputs/outlier_detection/<dataset_name>/

**Includes:**

- outliers.csv  
- cleaned_no_outliers.csv  
- outlier_report.txt  

---

### 5) pca_feature_reduction.py

**Principal Component Analysis (PCA) Feature Reduction**

**Pipeline:**

- select numeric features
- standardize (z-score)
- apply PCA using either:
  - fixed component count, or
  - target explained variance

**Saves:**

- PCA-reduced dataset
- loading matrix (feature → PC mapping)
- explained-variance report

**Run (fixed components):**

python pca_feature_reduction.py
--csv sample.csv
--n_components 3

**Run (variance target):**

python pca_feature_reduction.py
--csv sample.csv
--var 0.9

**Output:**

outputs/pca_reduction/<dataset_name>/

**Includes:**

- pca_reduced.csv  
- pca_loadings.csv  
- pca_variance_report.txt  

---

## Output Directory Structure

outputs/
├─ data_cleaning/
├─ eda_visual/
├─ pca_reduction/
├─ feature_scaling/
└─ outlier_detection/

---

## Dataset Notes

This toolkit is intended for:

- preprocessing diagnostics
- quantitative EDA workflows
- ML dataset preparation
- research & coursework experimentation

Supported dataset types:

- tabular numeric datasets
- mixed numeric + categorical datasets
- experimental / research CSV tables
