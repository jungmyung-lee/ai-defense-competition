"""
baseline_logreg_classifier.py

Author: Jungmyung Lee

Description:
    Trains a Logistic Regression classification baseline model using:
    1) Train / Test CSV datasets
    2) Standardized feature scaling
    3) Deterministic random seed for reproducibility

    Computes evaluation metrics including:
    - Accuracy
    - Precision / Recall / F1-score (macro & weighted)
    - Confusion matrix counts

    Saves prediction outputs and a baseline performance report.

Outputs:
    ./outputs/baseline_logreg/<dataset_name>/

Python Dependencies:
    - NumPy
    - pandas
    - scikit-learn

Run Example:
    python baseline_logreg_classifier.py --train dataset_train.csv --test dataset_test.csv --label label
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# Directory handling
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# Load dataset CSV
# =========================================================
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    df = pd.read_csv(path)

    if len(df) == 0:
        raise ValueError("[ERROR] CSV file is empty")

    return df


# =========================================================
# Split features / labels
# =========================================================
def split_features_labels(df, label_col):
    if label_col not in df.columns:
        raise ValueError(f"[ERROR] Label column not found: {label_col}")

    X = df.drop(columns=[label_col]).values
    y = df[label_col].values

    return X, y


# =========================================================
# Compute evaluation metrics
# =========================================================
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


# =========================================================
# Save prediction outputs
# =========================================================
def save_predictions(output_dir, base_name, test_df, y_true, y_pred):
    out = test_df.copy()
    out["y_true"] = y_true
    out["y_pred"] = y_pred

    path = os.path.join(output_dir, f"{base_name}_logreg_predictions.csv")
    out.to_csv(path, index=False)

    print(f"[+] Saved prediction file → {path}")


# =========================================================
# Save evaluation report
# =========================================================
def save_report(output_dir, base_name, metrics, seed):
    path = os.path.join(output_dir, f"{base_name}_logreg_baseline_report.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("LOGISTIC REGRESSION BASELINE REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Random seed : {seed}\n\n")

        f.write("[Performance Metrics]\n")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                f.write(f" - {k}: {v}\n")

        f.write("\n[Confusion Matrix]\n")
        for row in metrics["confusion_matrix"]:
            f.write(f" {row}\n")

        f.write("\nNotes:\n")
        f.write(" - Logistic Regression is used as a baseline classifier\n")
        f.write(" - Provides a reproducible performance reference\n")

    print(f"[+] Saved baseline report → {path}")


# =========================================================
# Baseline training pipeline
# =========================================================
def run_pipeline(train_csv, test_csv, label_col, seed):
    base_name = os.path.splitext(os.path.basename(train_csv))[0]
    output_dir = os.path.join("outputs", "baseline_logreg", base_name)

    ensure_dir(output_dir)

    print("\n[ LOGISTIC REGRESSION BASELINE PIPELINE ]")
    print(f"Train CSV : {train_csv}")
    print(f"Test  CSV : {test_csv}")
    print(f"Output    : {output_dir}\n")

    # -------------------------------
    # Load datasets
    # -------------------------------
    train_df = load_dataset(train_csv)
    test_df  = load_dataset(test_csv)

    # -------------------------------
    # Split features / labels
    # -------------------------------
    X_train, y_train = split_features_labels(train_df, label_col)
    X_test,  y_test  = split_features_labels(test_df, label_col)

    # -------------------------------
    # Feature Scaling
    # -------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # -------------------------------
    # Train Logistic Regression
    # -------------------------------
    model = LogisticRegression(
        max_iter=500,
        random_state=seed,
        n_jobs=None
    )

    model.fit(X_train_scaled, y_train)

    # -------------------------------
    # Inference
    # -------------------------------
    y_pred = model.predict(X_test_scaled)

    # -------------------------------
    # Metrics
    # -------------------------------
    metrics = compute_metrics(y_test, y_pred)

    # -------------------------------
    # Save outputs
    # -------------------------------
    save_predictions(output_dir, base_name, test_df, y_test, y_pred)
    save_report(output_dir, base_name, metrics, seed)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Logistic Regression baseline classifier pipeline"
    )

    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to training CSV"
    )

    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test CSV"
    )

    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Label column name"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default=42)"
    )

    args = parser.parse_args()

    run_pipeline(args.train, args.test, args.label, args.seed)
