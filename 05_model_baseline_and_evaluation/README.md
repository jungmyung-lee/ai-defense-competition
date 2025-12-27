# 05_model_baseline_and_evaluation

A collection of **baseline ML classification, evaluation, and dataset splitting pipelines** implemented using **NumPy, pandas, and scikit-learn**.

This toolkit provides reproducible experiment utilities for:

- baseline model training
- cross-validation stability analysis
- evaluation report generation
- RandomForest interpretation
- stratified train/test dataset splitting

Each script runs as an independent CLI tool and produces both processed outputs and **analysis-oriented reports**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Learning Objectives](#learning-objectives)
- [Applications](#applications)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Scripts](#scripts)
  - [1) baseline_logreg_classifier.py](#1-baseline_logreg_classifierpy)
  - [2) cross_validation_scores.py](#2-cross_validation_scorespy)
  - [3) evaluation_metrics_report.py](#3-evaluation_metrics_reportpy)
  - [4) randomforest_classification.py](#4-randomforest_classificationpy)
  - [5) train_test_split_pipeline.py](#5-train_test_split_pipelinepy)
- [Output Directory Structure](#output-directory-structure)
- [Dataset Notes](#dataset-notes)

---

## Project Overview

This project implements a modular, experiment-driven **baseline ML evaluation toolkit**, including:

- Logistic Regression baseline classifier
- stratified K-Fold cross-validation scoring
- unified evaluation metrics report generator
- RandomForest classification + feature importance export
- stratified Train / Test dataset splitting pipeline

**The toolkit emphasizes:**

- reproducible ML experiments  
- numerically interpretable reports  
- evaluation beyond accuracy only  
- dataset-aware model comparison  

---

## Learning Objectives

This toolkit was developed to:

### Build reproducible ML baselines

- generate deterministic baseline scores
- stabilize evaluation using **fixed random seeds**
- separate training / inference / evaluation outputs

### Strengthen quantitative model evaluation

- compute macro + weighted metrics
- inspect confusion matrix behavior
- analyze class imbalance effects

### Support experiment-ready workflows

- export prediction tables for reuse
- compare baseline vs advanced models
- document performance using text reports

---

## Applications

This toolkit is useful for:

- ML model benchmarking baselines
- dataset quality inspection via metrics
- academic / research experiment pipelines
- performance reporting in competitions
- pre-deployment model sanity-checking

This project serves as an **evaluation-first ML experimentation toolkit**.

---

## Project Structure

**05_model_baseline_and_evaluation**  
│  
├── README.md  
├── baseline_logreg_classifier.py  
├── cross_validation_scores.py  
├── evaluation_metrics_report.py  
├── randomforest_classification.py  
└── train_test_split_pipeline.py  

All outputs are stored under:

./outputs/

---

## Installation

pip install numpy pandas scikit-learn

---

# Scripts

---

## 1) baseline_logreg_classifier.py

**Logistic Regression Baseline Classifier**

**Trains a baseline classifier using:**

- standardized feature scaling
- deterministic random seed
- fixed feature pipeline

**Computes:**

- Accuracy
- Precision / Recall / F1 (macro + weighted)
- Confusion matrix
- Prediction CSV export

**Run:**

python baseline_logreg_classifier.py 
  --train dataset_train.csv 
  --test dataset_test.csv 
  --label label 
  --seed 42

**Output:**

outputs/baseline_logreg/<dataset_name>/
`<dataset_name>` is derived from the **training CSV file name** (without extension).  
For example, `dataset_train.csv` → `outputs/baseline_logreg/dataset_train/`.

**Includes:**

- `<name>_logreg_predictions.csv`
- `<name>_logreg_baseline_report.txt`

---

## 2) cross_validation_scores.py

- For `logreg`, features are standardized with `StandardScaler`; for `rf`, raw features are used.

**Stratified K-Fold Cross-Validation Score Runner**

**Supports selectable baseline models:**

- Logistic Regression (`--model logreg`)
- RandomForest (`--model rf`)

**Computes per-fold:**

- Accuracy
- Precision / Recall / F1 (macro + weighted)

**Then aggregates:**

- mean & std across folds

**Run (Logistic Regression example):**

python cross_validation_scores.py 
  --csv data.csv 
  --label label 
  --k 5 
  --model logreg

**Run (RandomForest example):**

python cross_validation_scores.py 
  --csv data.csv 
  --label label 
  --k 5 
  --model rf

**Output:**

outputs/cross_validation/<dataset_name>/

**Includes:**

- `<name>_cv_fold_scores.csv`
- `<name>_cv_summary_report.txt`

---

## 3) evaluation_metrics_report.py

**Unified Evaluation Report Generator**

Input CSV must contain two columns:

- `y_true`: ground-truth labels  
- `y_pred`: model predictions

**Computes:**

- global metrics
- per-class metrics
- support counts
- confusion matrix

**Run:**

python evaluation_metrics_report.py
--csv model_predictions.csv

**Output:**

outputs/evaluation_report/<result_name>/

**Includes:**

- `<name>_metrics_table.csv`
- `<name>_evaluation_report.txt`

---

## 4) randomforest_classification.py

**RandomForest Classification Pipeline**

**Trains a configurable RandomForest model:**

- adjustable tree count
- optional max depth
- deterministic random seed

**Computes:**

- Accuracy
- Precision / Recall / F1 (macro + weighted)
- Confusion matrix
- Feature importance ranking

**Run:**

python randomforest_classification.py 
  --train dataset_train.csv 
  --test dataset_test.csv 
  --label label 
  --trees 200 
  --depth None 
  --seed 42

- `--trees`: number of trees (`n_estimators`, default: 200)  
- `--depth`: maximum tree depth (`max_depth`, default: None)  
- `--seed`: random_state for reproducibility (default: 42)

**Output:**

outputs/randomforest/<dataset_name>/

`<dataset_name>` is derived from the **training CSV file name** (without extension).  
For example, `dataset_train.csv` → `outputs/randomforest/dataset_train/`.

**Includes:**

- `<name>_rf_predictions.csv`
- `<name>_rf_feature_importance.txt`
- `<name>_rf_report.txt`

---

## 5) train_test_split_pipeline.py

**Stratified Train / Test Split Tool**

**Features:**

- stratified splitting by label (always preserves class ratios where possible)
- fixed random seed
- class-distribution reports before and after the split
  
**Exports:**

- train CSV
- test CSV
- class distribution statistics

**Run:**

python train_test_split_pipeline.py
--csv data.csv
--label label
--test 0.2

**Output:**

outputs/train_test_split/<dataset_name>/

`<dataset_name>` is derived from the **original CSV file name** (without extension).

**Includes:**

- `<name>_train.csv`
- `<name>_test.csv`
- `<name>_split_report.txt`

---

## Output Directory Structure

outputs/  
├─ baseline_logreg/  
├─ cross_validation/  
├─ evaluation_report/  
├─ randomforest/  
└─ train_test_split/  

**Each run produces:**

- prediction tables  
- numeric evaluation reports  
- reproducible experiment logs  

---

## Dataset Notes

This toolkit is intended for:

- ML research experiments
- benchmark baselines
- metric-driven model comparison
- supervised classification workflows

Supported dataset types:

- tabular numeric datasets
- mixed feature datasets
- academic / research datasets
