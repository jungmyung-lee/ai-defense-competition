"""
hyperparameter_search_logreg_tfidf.py

Author: Jungmyung Lee

Description:
    Small-scale hyperparameter search for
    TF-IDF + Logistic Regression text classifier.

    This script performs a joint grid search over:
        - TF-IDF parameters
        - Logistic Regression regularization parameters

    Designed for competition environments where
    computational resources are limited.

    Evaluation metric:
        - Macro-F1 (primary)
        - Accuracy (secondary)

Outputs:
    ./outputs/text_hyperparameter_search/<dataset_name>/
        - search_results_ranking.csv
        - best_model_report.txt
        - best_model_predictions.csv

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
from sklearn.metrics import accuracy_score, f1_score, classification_report


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
# Search Space Definition
# =========================================================
TFIDF_GRID = [
    {"ngram_range": (1, 1), "min_df": 1, "max_df": 1.0},
    {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.95},
]

LOGREG_GRID = [
    {"C": 0.5, "penalty": "l2"},
    {"C": 1.0, "penalty": "l2"},
    {"C": 2.0, "penalty": "l2"},
]


# =========================================================
# Train & Evaluate Function
# =========================================================
def train_and_eval(texts_train, texts_test, y_train, y_test,
                   tfidf_cfg, logreg_cfg):

    vectorizer = TfidfVectorizer(
        ngram_range=tfidf_cfg["ngram_range"],
        min_df=tfidf_cfg["min_df"],
        max_df=tfidf_cfg["max_df"],
        sublinear_tf=True,
        norm="l2"
    )

    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)

    model = LogisticRegression(
        C=logreg_cfg["C"],
        penalty=logreg_cfg["penalty"],
        max_iter=200
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    result = {
        "acc": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "y_pred": y_pred,
        "vectorizer": vectorizer,
        "model": model
    }

    return result


# =========================================================
# Ranking Table Writer
# =========================================================
def save_ranking_table(output_dir, base, rows):

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["macro_f1", "acc"], ascending=False)

    path = os.path.join(output_dir, f"{base}_search_results_ranking.csv")
    df.to_csv(path, index=False)

    print(f"[+] Saved ranked results → {path}")

    return df


# =========================================================
# Best Model Report Writer
# =========================================================
def save_best_model_report(output_dir, base, best_row, y_test, y_pred):

    path = os.path.join(output_dir, f"{base}_best_model_report.txt")

    with open(path, "w", encoding="utf-8") as f:

        f.write("BEST MODEL — HYPERPARAMETER SEARCH RESULT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write("TF-IDF Parameters\n")
        f.write(f"  ngram_range : {best_row['tfidf_ngram']}\n")
        f.write(f"  min_df      : {best_row['tfidf_min_df']}\n")
        f.write(f"  max_df      : {best_row['tfidf_max_df']}\n\n")

        f.write("Logistic Regression Parameters\n")
        f.write(f"  C        : {best_row['logreg_C']}\n")
        f.write(f"  penalty  : {best_row['logreg_penalty']}\n\n")

        f.write("Performance\n")
        f.write(f"  Accuracy : {best_row['acc']:.4f}\n")
        f.write(f"  Macro-F1 : {best_row['macro_f1']:.4f}\n\n")

        f.write("Classification Report\n")
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write("\n")

    print(f"[+] Saved best-model report → {path}")


def save_best_predictions(output_dir, base, y_test, y_pred):

    df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })

    path = os.path.join(output_dir, f"{base}_best_model_predictions.csv")
    df.to_csv(path, index=False)

    print(f"[+] Saved best-model predictions → {path}")


# =========================================================
# Pipeline Runner
# =========================================================
def run_pipeline(csv_path,
                 text_col,
                 label_col,
                 test_size=0.25,
                 random_state=42):

    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs",
                              "text_hyperparameter_search",
                              base)

    ensure_dir(output_dir)

    print("\n[ HYPERPARAMETER SEARCH — TFIDF + LOGREG ]")
    print(f"Input   : {csv_path}")
    print(f"Text    : {text_col}")
    print(f"Label   : {label_col}")
    print(f"Output  : {output_dir}\n")

    df = load_dataset(csv_path, text_col, label_col)

    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    search_rows = []
    best_cfg = None
    best_macro_f1 = -1.0
    best_pred = None

    # -----------------------------------------------------
    # Joint grid search
    # -----------------------------------------------------
    for tcfg in TFIDF_GRID:
        for lcfg in LOGREG_GRID:

            res = train_and_eval(
                X_train, X_test,
                y_train, y_test,
                tcfg, lcfg
            )

            row = {
                "tfidf_ngram": str(tcfg["ngram_range"]),
                "tfidf_min_df": tcfg["min_df"],
                "tfidf_max_df": tcfg["max_df"],
                "logreg_C": lcfg["C"],
                "logreg_penalty": lcfg["penalty"],
                "acc": res["acc"],
                "macro_f1": res["macro_f1"]
            }

            search_rows.append(row)

            # Track best
            if res["macro_f1"] > best_macro_f1:
                best_macro_f1 = res["macro_f1"]
                best_cfg = row
                best_pred = res["y_pred"]

    # -----------------------------------------------------
    # Save ranking + best model report
    # -----------------------------------------------------
    ranking_df = save_ranking_table(output_dir, base, search_rows)

    best_row = ranking_df.iloc[0]
    save_best_model_report(output_dir, base, best_row, y_test, best_pred)
    save_best_predictions(output_dir, base, y_test, best_pred)

    print("\n[ DONE ]\n")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for TF-IDF + Logistic Regression"
    )

    parser.add_argument("--csv", type=str, required=True,
                        help="Path to dataset CSV")

    parser.add_argument("--text_col", type=str, default="cleaned_text",
                        help="Column containing processed text")

    parser.add_argument("--label_col", type=str, default="label",
                        help="Target label column")

    parser.add_argument("--test_size", type=float, default=0.25,
                        help="Test set ratio")

    args = parser.parse_args()

    run_pipeline(
        args.csv,
        args.text_col,
        args.label_col,
        test_size=args.test_size
    )
