"""
naive_bayes_text_classification.py

Author: Jungmyung Lee

Description:
    Comparative text classification experiment using:
    - Multinomial Naive Bayes
    - Logistic Regression (baseline)

    Designed for small-data competition settings where
    model robustness and generalization stability matter.

    Pipeline:
    - Load CSV dataset
    - Train/Test split
    - TF-IDF feature extraction (fit on train only)
    - Train both models on identical features
    - Evaluate:
        * accuracy
        * macro-F1
        * class-wise recall & F1
    - Save report + prediction tables

Outputs:
    ./outputs/text_model_comparison/<dataset_name>/

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

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
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
# Shared TF-IDF feature extractor
# =========================================================
def build_tfidf_features(X_train, X_test,
                         max_features=6000,
                         ngram_range=(1,2)):

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        norm="l2"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, vectorizer


# =========================================================
# Model Definitions
# =========================================================
def train_nb(X_train, y_train):
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    return model


def train_logreg(X_train, y_train):
    model = LogisticRegression(
        C=1.0,
        max_iter=200
    )
    model.fit(X_train, y_train)
    return model


# =========================================================
# Evaluation
# =========================================================
def evaluate_model(model, X_test, y_test, labels, name):

    y_pred = model.predict(X_test)

    results = {
        "name": name,
        "acc": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "class_recall": recall_score(y_test, y_pred, average=None, labels=labels),
        "class_f1": f1_score(y_test, y_pred, average=None, labels=labels),
        "report": classification_report(y_test, y_pred, digits=4),
        "y_pred": y_pred
    }

    return results


def save_prediction_table(output_dir, base, X_test, y_test, res_nb, res_lr,
                          text_col_name="text"):

    
    df = pd.DataFrame({
        text_col_name: X_test,
        "y_true": y_test,
        "y_pred": res_lr["y_pred"],      
        "pred_nb": res_nb["y_pred"],     
        "pred_logreg": res_lr["y_pred"]  
    })

    path = os.path.join(output_dir, f"{base}_model_predictions.csv")
    df.to_csv(path, index=False)

    print(f"[+] Saved prediction comparison → {path}")



# =========================================================
# Report Writer
# =========================================================
def save_comparison_report(output_dir, base, labels, res_nb, res_lr):

    path = os.path.join(output_dir, f"{base}_model_comparison_report.txt")

    with open(path, "w", encoding="utf-8") as f:

        f.write("TEXT CLASSIFICATION MODEL COMPARISON\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write("Dataset setting: small-sample experiment focus\n")
        f.write("Models compared:\n")
        f.write(" - Multinomial Naive Bayes\n")
        f.write(" - Logistic Regression (TF-IDF baseline)\n\n")

        # Summary metrics
        f.write("GLOBAL PERFORMANCE\n")
        f.write("--------------------------------------\n")
        f.write(f"[Naive Bayes]\n")
        f.write(f"Accuracy  : {res_nb['acc']:.4f}\n")
        f.write(f"Macro-F1  : {res_nb['macro_f1']:.4f}\n\n")

        f.write(f"[Logistic Regression]\n")
        f.write(f"Accuracy  : {res_lr['acc']:.4f}\n")
        f.write(f"Macro-F1  : {res_lr['macro_f1']:.4f}\n\n")

        # Class-wise metrics
        f.write("CLASS-WISE PERFORMANCE (Recall / F1)\n")
        f.write("--------------------------------------\n")

        for i, label in enumerate(labels):
            f.write(f"{label}\n")
            f.write(f"  NB   — recall={res_nb['class_recall'][i]:.4f}  "
                    f"f1={res_nb['class_f1'][i]:.4f}\n")
            f.write(f"  LOGR — recall={res_lr['class_recall'][i]:.4f}  "
                    f"f1={res_lr['class_f1'][i]:.4f}\n\n")

        # Full sklearn reports
        f.write("\n\n[Naive Bayes Classification Report]\n")
        f.write(res_nb["report"])
        f.write("\n")

        f.write("\n[Logistic Regression Classification Report]\n")
        f.write(res_lr["report"])
        f.write("\n")

    print(f"[+] Saved model comparison report → {path}")


# =========================================================
# Pipeline Runner
# =========================================================
def run_pipeline(csv_path,
                 text_col,
                 label_col,
                 test_size=0.25,
                 random_state=42):

    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs", "text_model_comparison", base)
    ensure_dir(output_dir)

    print("\n[ TEXT MODEL COMPARISON — NB vs LOGISTIC ]")
    print(f"Input   : {csv_path}")
    print(f"Text    : {text_col}")
    print(f"Label   : {label_col}")
    print(f"Output  : {output_dir}\n")

    df = load_dataset(csv_path, text_col, label_col)

    X = df[text_col].astype(str).tolist()
    y = df[label_col].astype(str).tolist()

    labels = sorted(list(set(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Shared TF-IDF representation
    X_train_vec, X_test_vec, _ = build_tfidf_features(X_train, X_test)

    # Train models
    nb_model = train_nb(X_train_vec, y_train)
    lr_model = train_logreg(X_train_vec, y_train)

    # Evaluate
    res_nb = evaluate_model(nb_model, X_test_vec, y_test, labels, "NaiveBayes")
    res_lr = evaluate_model(lr_model, X_test_vec, y_test, labels, "LogisticRegression")

    # Save report + predictions
    save_comparison_report(output_dir, base, labels, res_nb, res_lr)
    save_prediction_table(output_dir, base, X_test, y_test, res_nb, res_lr, text_col_name="text")


    print("\n[ DONE ]\n")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Multinomial Naive Bayes vs Logistic Regression for text classification"
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
