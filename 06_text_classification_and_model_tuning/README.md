# 06_text_classification_and_model_tuning

A collection of experiment-driven NLP utilities for text preprocessing, feature engineering, baseline modeling, error analysis, and comparative model evaluation.

This toolkit focuses on **interpretability-oriented NLP workflows**, including:

- TF-IDF / BoW feature comparison
- baseline text classification pipelines
- Naive Bayes vs Logistic Regression experimental comparison
- hyperparameter search for TF-IDF + LogReg
- misclassification inspection and hard-case analysis
- deterministic preprocessing for reproducible experiments

Each script runs as an independent CLI tool and produces both processed outputs and **analysis-oriented reports**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Learning Objectives](#learning-objectives)
- [Applications](#applications)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Scripts](#scripts)
  - [1) text_cleaning_preprocessing.py](#1-text_cleaning_preprocessingpy)
  - [2) bow_tfidf_vectorization_comparison.py](#2-bow_tfidf_vectorization_comparisonpy)
  - [3) text_logreg_baseline_classifier.py](#3-text_logreg_baseline_classifierpy)
  - [4) naive_bayes_text_classification.py](#4-naive_bayes_text_classificationpy)
  - [5) hyperparameter_search_logreg_tfidf.py](#5-hyperparameter_search_logreg_tfidfpy)
  - [6) text_misclassification_inspection.py](#6-text_misclassification_inspectionpy)
- [Output Directory Structure](#output-directory-structure)
- [Dataset Notes](#dataset-notes)

---

## Project Overview

This project implements a lightweight but analysis-focused NLP experimentation toolkit designed for:

- small-sample competition settings
- research-driven exploratory modeling
- interpretable classification workflows

Core capabilities include:

- reproducible text preprocessing pipelines
- BoW / TF-IDF feature interpretation
- baseline text classifier evaluation
- comparative model experiments
- hyperparameter search with ranking reports
- qualitative and quantitative error analysis

The toolkit emphasizes:

- experiment reproducibility  
- transparent preprocessing  
- metric-driven model comparison  
- interpretation beyond raw accuracy  

---

## Learning Objectives

This project was developed to:

### Strengthen feature-level understanding of text models

- compare BoW and TF-IDF representations
- analyze sparsity and vocabulary statistics
- identify top contributing terms

### Support interpretable model experimentation

- evaluate Naive Bayes vs Logistic Regression
- quantify macro-F1 and class-wise recall behavior
- examine small-sample generalization effects

### Encourage analysis-driven workflows

- export ranking, prediction, and evaluation tables
- inspect misclassified samples
- generate “hard case” lists for model debugging

---

## Applications

This toolkit is useful for:

- NLP competition experimentation
- dataset-specific feature analysis
- baseline classification benchmarking
- academic research and reporting
- model debugging and qualitative review

This project serves as an **interpretation-centered NLP experimentation platform**.

---

## Project Structure

**06_text_classification_and_model_tuning**  
│  
├── README.md  
├── bow_tfidf_vectorization_comparison.py  
├── hyperparameter_search_logreg_tfidf.py  
├── naive_bayes_text_classification.py  
├── text_cleaning_preprocessing.py  
├── text_logreg_baseline_classifier.py  
└── text_misclassification_inspection.py  

All outputs are written to:

./outputs/

---

## Installation

pip install pandas numpy scikit-learn

---

# Scripts

---

## 1) text_cleaning_preprocessing.py

Text normalization and preprocessing pipeline for NLP datasets.

Performs:

- lowercasing
- whitespace normalization
- punctuation removal
- optional digit removal
- tokenization
- optional stopword removal

Exports both cleaned text and preprocessing logs.

Run:

python text_cleaning_preprocessing.py
--csv data.csv
--text_col text

Output:

outputs/text_preprocessing/<dataset_name>/

Includes:

- `<name>_cleaned_text.csv`
- `<name>_sample_before_after.txt`
- `<name>_preprocessing_log.txt`

---

## 2) bow_tfidf_vectorization_comparison.py

Compares Bag-of-Words and TF-IDF feature representations.

Computes:

- dimensionality and sparsity statistics
- vocabulary size
- top-k highest frequency / importance terms

Run:

python bow_tfidf_vectorization_comparison.py
--csv data.csv
--text_col cleaned_text

Output:

outputs/text_features/<dataset_name>/

Includes:

- `<name>_bow_features_stats.txt`
- `<name>_tfidf_features_stats.txt`
- `<name>_vocab_top_terms.csv`

---

## 3) text_logreg_baseline_classifier.py

Baseline text classifier using:

- TF-IDF representation
- Logistic Regression
- train-only vocabulary fitting

Computes:

- train / test accuracy
- macro-F1 score
- confusion matrix

Run:

python text_logreg_baseline_classifier.py
--csv data.csv
--text_col cleaned_text
--label_col label

Output:

outputs/text_classification_baseline/<dataset_name>/

Includes:

- `<name>_baseline_report.txt`
- `<name>_predictions.csv`

---

## 4) naive_bayes_text_classification.py

Comparative experiment:

- Multinomial Naive Bayes
- Logistic Regression baseline

Shared TF-IDF features across models.

Evaluates:

- accuracy
- macro-F1
- class-wise recall & F1

Run:

python naive_bayes_text_classification.py
--csv data.csv
--text_col cleaned_text
--label_col label

Output:

outputs/text_model_comparison/<dataset_name>/

Includes:

- `<name>_model_comparison_report.txt`
- `<name>_model_predictions.csv`

---

## 5) hyperparameter_search_logreg_tfidf.py

Small-scale hyperparameter search for:

- TF-IDF configuration
- Logistic Regression regularization

Designed for limited-resource competition settings.

Primary metric:

- macro-F1

Secondary:

- accuracy

Run:

python hyperparameter_search_logreg_tfidf.py
--csv data.csv
--text_col cleaned_text
--label_col label

Output:

outputs/text_hyperparameter_search/<dataset_name>/

Includes:

- `<name>_search_results_ranking.csv`
- `<name>_best_model_report.txt`
- `<name>_best_model_predictions.csv`

---

## 6) text_misclassification_inspection.py

Misclassification analysis and qualitative debugging tool.

Extracts:

- only wrong predictions
- optional probability-based confidence gap
- hard-case samples list

Outputs human-interpretable analysis reports.

Run:

python text_misclassification_inspection.py
--pred_csv predictions.csv
--text_col text

Output:

outputs/text_error_analysis/<dataset_name>/

Includes:

- `<name>_misclassified_samples.csv`
- `<name>_hard_cases_topN.csv`
- `<name>_error_analysis_report.txt`

---

## Output Directory Structure

outputs/
├─ text_preprocessing/        
├─ text_features/              
├─ text_classification_baseline/
├─ text_model_comparison/       
├─ text_hyperparameter_search/  
└─ text_error_analysis/         


Each run produces:

- processed datasets
- prediction and evaluation tables
- experiment-ready analysis reports

---

## Dataset Notes

This toolkit is designed for:

- small to medium NLP datasets
- academic research experiments
- applied ML competitions
- interpretability-focused workflows

Recommended input:

- pre-tokenized or cleaned text
- labeled classification datasets
