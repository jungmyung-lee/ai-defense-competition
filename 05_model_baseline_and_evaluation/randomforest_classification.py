"""
randomforest_classification.py

Author: Jungmyung Lee

Description:
    Trains a RandomForest classification model using:
    1) Train / Test CSV datasets
    2) Deterministic random seed for reproducibility
    3) Classifier with configurable tree & depth parameters

    Computes evaluation metrics including:
    - Accuracy
    - Precision / Recall / F1-score (macro & weighted)
    - Confusion matrix
    - Feature importance ranking

    Saves prediction outputs and a RandomForest performance report.

Outputs:
    ./outputs/randomforest/<dataset_name>/

Python Dependencies:
    - NumPy
    - pandas
    - scikit-learn

Run Example:
    python randomforest_classification.py --train dataset_train.csv --test dataset_test.csv --label label
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
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
# Load dataset
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

    feature_names = [c for c in df.columns if c != label_col]

    X = df[feature_names].values
    y = df[label_col].values

    return X, y, feature_names


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
# Save predictions CSV
# =========================================================
def save_predictions(output_dir, base_name, test_df, y_true, y_pred):
    out = test_df.copy()
    out["y_true"] = y_true
    out["y_pred"] = y_pred

    path = os.path.join(output_dir, f"{base_name}_rf_predictions.csv")
    out.to_csv(path, index=False)

    print(f"[+] Saved prediction file → {path}")


# =========================================================
# Save feature importance ranking
# =========================================================
def save_feature_importance(output_dir, base_name, feature_names, importances):
    data = list(zip(feature_names, importances))

    data_sorted = sorted(data, key=lambda x: x[1], reverse=True)

    path = os.path.join(output_dir, f"{base_name}_rf_feature_importance.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("RANDOM FOREST FEATURE IMPORTANCE\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        for name, score in data_sorted:
            f.write(f"{name}: {round(float(score), 6)}\n")

    print(f"[+] Saved feature importance → {path}")


# =========================================================
# Save evaluation report
# =========================================================
def save_report(output_dir, base_name, metrics, seed, n_trees, max_depth):
    path = os.path.join(output_dir, f"{base_name}_rf_report.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("RANDOM FOREST CLASSIFICATION REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write(f"Random seed : {seed}\n")
        f.write(f"n_estimators: {n_trees}\n")
        f.write(f"max_depth   : {max_depth}\n\n")

        f.write("[Performance Metrics]\n")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                f.write(f" - {k}: {v}\n")

        f.write("\n[Confusion Matrix]\n")
        for row in metrics["confusion_matrix"]:
            f.write(f" {row}\n")

        f.write("\nNotes:\n")
        f.write(" - RandomForest models nonlinear decision boundaries\n")
        f.write(" - Feature importance helps interpret dataset structure\n")

    print(f"[+] Saved evaluation report → {path}")


# =========================================================
# Training pipeline
# =========================================================
def run_pipeline(train_csv, test_csv, label_col, seed, n_trees, max_depth):
    base_name = os.path.splitext(os.path.basename(train_csv))[0]
    output_dir = os.path.join("outputs", "randomforest", base_name)

    ensure_dir(output_dir)

    print("\n[ RANDOM FOREST CLASSIFICATION PIPELINE ]")
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
    X_train, y_train, feature_names = split_features_labels(train_df, label_col)
    X_test,  y_test,  _            = split_features_labels(test_df, label_col)

    # -------------------------------
    # Train RandomForest
    # -------------------------------
    model = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # -------------------------------
    # Inference
    # -------------------------------
    y_pred = model.predict(X_test)

    # -------------------------------
    # Metrics
    # -------------------------------
    metrics = compute_metrics(y_test, y_pred)

    # -------------------------------
    # Save outputs
    # -------------------------------
    save_predictions(output_dir, base_name, test_df, y_test, y_pred)

    save_feature_importance(
        output_dir,
        base_name,
        feature_names,
        model.feature_importances_
    )

    save_report(
        output_dir,
        base_name,
        metrics,
        seed,
        n_trees,
        max_depth
    )

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RandomForest classification pipeline"
    )

    parser.add_argument("--train", type=str, required=True,
                        help="Path to training CSV")

    parser.add_argument("--test", type=str, required=True,
                        help="Path to test CSV")

    parser.add_argument("--label", type=str, required=True,
                        help="Label column name")

    parser.add_argument("--trees", type=int, default=200,
                        help="Number of trees (default=200)")

    parser.add_argument("--depth", type=int, default=None,
                        help="Max tree depth (default=None)")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default=42)")

    args = parser.parse_args()

    run_pipeline(
        args.train,
        args.test,
        args.label,
        args.seed,
        args.trees,
        args.depth
    )
