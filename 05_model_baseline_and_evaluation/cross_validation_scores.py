"""
cross_validation_scores.py

Author: Jungmyung Lee

Description:
    Performs stratified K-Fold cross-validation on a dataset and computes:
    - Accuracy
    - Precision / Recall / F1-score (macro & weighted)

    Supports selectable baseline models:
    1) Logistic Regression
    2) RandomForest

    Saves fold-level results and aggregated statistics.

Outputs:
    ./outputs/cross_validation/<dataset_name>/

Python Dependencies:
    - NumPy
    - pandas
    - scikit-learn

Run Example:
    python cross_validation_scores.py --csv data.csv --label label --k 5 --model logreg
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
# Build model
# =========================================================
def build_model(model_type, seed):
    if model_type == "logreg":
        return LogisticRegression(
            max_iter=500,
            random_state=seed
        )

    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=seed,
            n_jobs=-1
        )

    else:
        raise ValueError(f"[ERROR] Unknown model type: {model_type}")


# =========================================================
# Compute metrics for one fold
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
    }


# =========================================================
# Aggregate fold metrics
# =========================================================
def aggregate_scores(fold_scores):
    keys = fold_scores[0].keys()

    summary = {}

    for k in keys:
        vals = [fs[k] for fs in fold_scores]
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals))
        }

    return summary


# =========================================================
# Save fold score table
# =========================================================
def save_fold_scores(output_dir, base, fold_scores):
    csv_path = os.path.join(output_dir, f"{base}_cv_fold_scores.csv")

    rows = []
    for i, s in enumerate(fold_scores):
        row = {"fold": i + 1}
        row.update(s)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    print(f"[+] Saved fold scores CSV → {csv_path}")


# =========================================================
# Save summary report
# =========================================================
def save_summary_report(output_dir, base, model_type, k, seed, summary):
    path = os.path.join(output_dir, f"{base}_cv_summary_report.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("CROSS-VALIDATION PERFORMANCE REPORT\n")
        f.write(f"Generated : {datetime.now()}\n\n")

        f.write(f"Model     : {model_type}\n")
        f.write(f"K-Fold    : {k}\n")
        f.write(f"Seed      : {seed}\n\n")

        f.write("[Aggregated Metrics]\n")
        for metric, vals in summary.items():
            f.write(f" - {metric}: mean={vals['mean']}  std={vals['std']}\n")

        f.write("\nNotes:\n")
        f.write(" - Stratified K-Fold preserves class ratios per fold\n")
        f.write(" - Use as baseline model stability reference\n")

    print(f"[+] Saved summary report → {path}")


# =========================================================
# Cross-validation pipeline
# =========================================================
def run_pipeline(csv_path, label_col, k, model_type, seed):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs", "cross_validation", base)

    ensure_dir(output_dir)

    print("\n[ CROSS-VALIDATION PIPELINE ]")
    print(f"Dataset : {csv_path}")
    print(f"Output  : {output_dir}")
    print(f"Model   : {model_type}")
    print(f"K-Fold  : {k}\n")

    df = load_dataset(csv_path)
    X, y, _ = split_features_labels(df, label_col)

    kfold = StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=seed
    )

    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f" - Fold {fold_idx + 1} / {k}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # scaling only for LR
        if model_type == "logreg":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)

        model = build_model(model_type, seed)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        metrics = compute_metrics(y_val, y_pred)
        fold_scores.append(metrics)

    # aggregate
    summary = aggregate_scores(fold_scores)

    # save results
    save_fold_scores(output_dir, base, fold_scores)
    save_summary_report(output_dir, base, model_type, k, seed, summary)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stratified K-Fold cross-validation scoring pipeline"
    )

    parser.add_argument("--csv", type=str, required=True,
                        help="Path to dataset CSV")

    parser.add_argument("--label", type=str, required=True,
                        help="Label column name")

    parser.add_argument("--k", type=int, default=5,
                        help="Number of folds (default=5)")

    parser.add_argument("--model", type=str, default="logreg",
                        choices=["logreg", "rf"],
                        help="Model type: logreg | rf")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default=42)")

    args = parser.parse_args()

    run_pipeline(
        args.csv,
        args.label,
        args.k,
        args.model,
        args.seed
    )
