"""
text_logreg_baseline_classifier.py

Author: Jungmyung Lee

Description:
    Baseline text classification pipeline using
    TF-IDF features + Logistic Regression.

    Pipeline:
    - Load CSV dataset
    - Select text + label columns
    - Train / Test Split
    - TF-IDF vectorization (train only fit)
    - Train Logistic Regression classifier
    - Evaluate using accuracy + macro-F1
    - Save predictions, confusion matrix, and report

Outputs:
    ./outputs/text_classification_baseline/<dataset_name>/

Python Dependencies:
    - pandas
    - NumPy
    - scikit-learn
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# Utilities
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_dataset(csv_path, text_col, label_col):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise ValueError(f"[ERROR] Column '{text_col}' not found")

    if label_col not in df.columns:
        raise ValueError(f"[ERROR] Column '{label_col}' not found")

    if len(df) == 0:
        raise ValueError("[ERROR] Dataset is empty")

    return df


# =========================================================
# Model Training
# =========================================================
def train_tfidf_logreg(X_train, y_train,
                       max_features=5000,
                       ngram_range=(1, 2),
                       C=1.0):

    # Vectorizer (fit only on training data)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        norm="l2"
    )

    X_train_vec = vectorizer.fit_transform(X_train)

    clf = LogisticRegression(
        C=C,
        max_iter=200,
        n_jobs=None
    )

    clf.fit(X_train_vec, y_train)

    return clf, vectorizer


def evaluate_model(clf, vectorizer, X_train, X_test, y_train, y_test):
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    train_pred = clf.predict(X_train_vec)
    test_pred = clf.predict(X_test_vec)

    results = {
        "train_acc": accuracy_score(y_train, train_pred),
        "test_acc": accuracy_score(y_test, test_pred),
        "test_f1_macro": f1_score(y_test, test_pred, average="macro"),
        "confusion": confusion_matrix(y_test, test_pred),
        "report": classification_report(y_test, test_pred, digits=4),
        "y_test": y_test,
        "y_pred": test_pred
    }

    return results


# =========================================================
# Save Outputs
# =========================================================
def save_results(output_dir, base, results, X_test, text_col_name="text"):
    # 1) Metrics report (그대로)
    path_report = os.path.join(output_dir, f"{base}_baseline_report.txt")
    with open(path_report, "w", encoding="utf-8") as f:
        f.write("BASELINE TEXT CLASSIFICATION REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write(f"Train Accuracy : {results['train_acc']:.4f}\n")
        f.write(f"Test Accuracy  : {results['test_acc']:.4f}\n")
        f.write(f"Macro-F1       : {results['test_f1_macro']:.4f}\n\n")

        f.write("Classification Report\n")
        f.write(results["report"])
        f.write("\n")

        f.write("Confusion Matrix\n")
        f.write(str(results["confusion"]))
        f.write("\n")

    print(f"[+] Saved evaluation report → {path_report}")

    # 2) 예측 테이블에 text + y_true + y_pred 저장
    pred_table = pd.DataFrame({
        text_col_name: X_test,
        "y_true": results["y_test"],
        "y_pred": results["y_pred"]
    })

    path_pred = os.path.join(output_dir, f"{base}_predictions.csv")
    pred_table.to_csv(path_pred, index=False)

    print(f"[+] Saved predictions → {path_pred}")



# =========================================================
# Pipeline Runner
# =========================================================
def run_pipeline(csv_path,
                 text_col,
                 label_col,
                 test_size=0.2,
                 random_state=42):

    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs",
                              "text_classification_baseline",
                              base)

    ensure_dir(output_dir)

    print("\n[ TEXT LOGISTIC REGRESSION BASELINE ]")
    print(f"Input   : {csv_path}")
    print(f"Column  : {text_col}")
    print(f"Label   : {label_col}")
    print(f"Output  : {output_dir}\n")

    df = load_dataset(csv_path, text_col, label_col)

    X = df[text_col].astype(str).tolist()
    y = df[label_col].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    clf, vectorizer = train_tfidf_logreg(X_train, y_train)

    results = evaluate_model(
    clf, vectorizer,
    X_train, X_test,
    y_train, y_test
    )

    
    save_results(output_dir, base, results, X_test, text_col_name="text")


    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TF-IDF + Logistic Regression baseline text classifier"
    )

    parser.add_argument("--csv", type=str, required=True,
                        help="Path to dataset CSV")

    parser.add_argument("--text_col", type=str, default="cleaned_text",
                        help="Column name containing processed text")

    parser.add_argument("--label_col", type=str, default="label",
                        help="Target label column")

    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test set ratio")

    args = parser.parse_args()

    run_pipeline(
        args.csv,
        args.text_col,
        args.label_col,
        test_size=args.test_size
    )
