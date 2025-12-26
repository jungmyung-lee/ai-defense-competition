"""
bow_tfidf_vectorization_comparison.py

Author: Jungmyung Lee

Description:
    Compares Bag-of-Words (CountVectorizer) and
    TF-IDF feature representations for text classification datasets.

    Pipeline:
    - Load CSV dataset
    - Select target text column
    - Build BoW & TF-IDF vocabularies
    - Compute sparsity & dimensionality statistics
    - Extract top-k frequent terms
    - Save feature reports

Outputs:
    ./outputs/text_features/<dataset_name>/
        - bow_features_stats.txt
        - tfidf_features_stats.txt
        - vocab_top_terms.csv

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

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# =========================================================
# Utilities
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_dataset(csv_path, text_col):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise ValueError(f"[ERROR] Column '{text_col}' not found in dataset")

    if len(df) == 0:
        raise ValueError("[ERROR] Dataset is empty")

    return df


# =========================================================
# BoW Vectorization
# =========================================================
def build_bow_features(texts,
                       min_df=2,
                       ngram_range=(1, 1)):

    vectorizer = CountVectorizer(
        min_df=min_df,
        ngram_range=ngram_range
    )

    X = vectorizer.fit_transform(texts)

    return X, vectorizer


# =========================================================
# TF-IDF Vectorization
# =========================================================
def build_tfidf_features(texts,
                         min_df=2,
                         ngram_range=(1, 1)):

    vectorizer = TfidfVectorizer(
        min_df=min_df,
        ngram_range=ngram_range,
        norm="l2",
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(texts)

    return X, vectorizer


# =========================================================
# Statistics & Reporting
# =========================================================
def compute_matrix_stats(X):
    nnz = X.count_nonzero()
    total = X.shape[0] * X.shape[1]

    sparsity = 1.0 - (nnz / total)

    return {
        "rows": X.shape[0],
        "cols": X.shape[1],
        "nnz": nnz,
        "sparsity": sparsity
    }


def extract_top_terms(vectorizer, X, top_k=30):
    vocab = np.array(list(vectorizer.vocabulary_.keys()))
    freqs = np.asarray(X.sum(axis=0)).ravel()

    order = np.argsort(freqs)[::-1][:top_k]

    return list(zip(vocab[order], freqs[order]))


def save_feature_stats_report(path, name, stats, top_terms):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{name} FEATURE REPRESENTATION REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write("Matrix Statistics\n")
        f.write(f"Rows        : {stats['rows']}\n")
        f.write(f"Columns     : {stats['cols']}\n")
        f.write(f"Non-zero    : {stats['nnz']}\n")
        f.write(f"Sparsity    : {stats['sparsity']*100:.2f}%\n\n")

        f.write("Top Terms (frequency)\n")
        for term, freq in top_terms:
            f.write(f"{term:20s} {freq:.0f}\n")


def save_vocab_top_table(output_dir, base, bow_terms, tfidf_terms):
    df = pd.DataFrame({
        "bow_term": [t for t, _ in bow_terms],
        "bow_freq": [f for _, f in bow_terms],
        "tfidf_term": [t for t, _ in tfidf_terms],
        "tfidf_score": [f for _, f in tfidf_terms],
    })

    path = os.path.join(output_dir, f"{base}_vocab_top_terms.csv")
    df.to_csv(path, index=False)

    print(f"[+] Saved top-term comparison → {path}")


# =========================================================
# Pipeline Runner
# =========================================================
def run_pipeline(csv_path,
                 text_col,
                 min_df=2,
                 ngram=1,
                 top_k=30):

    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs", "text_features", base)
    ensure_dir(output_dir)

    print("\n[ TEXT FEATURE REPRESENTATION COMPARISON ]")
    print(f"Input   : {csv_path}")
    print(f"Column  : {text_col}")
    print(f"Output  : {output_dir}\n")

    df = load_dataset(csv_path, text_col)
    texts = df[text_col].astype(str).tolist()

    ngram_range = (1, ngram)

    # --------------------------
    # BoW
    # --------------------------
    bow_X, bow_vec = build_bow_features(
        texts,
        min_df=min_df,
        ngram_range=ngram_range
    )

    bow_stats = compute_matrix_stats(bow_X)
    bow_top = extract_top_terms(bow_vec, bow_X, top_k=top_k)

    bow_report = os.path.join(output_dir, f"{base}_bow_features_stats.txt")
    save_feature_stats_report(bow_report, "BAG-OF-WORDS", bow_stats, bow_top)

    print(f"[+] Saved BoW report → {bow_report}")

    # --------------------------
    # TF-IDF
    # --------------------------
    tfidf_X, tfidf_vec = build_tfidf_features(
        texts,
        min_df=min_df,
        ngram_range=ngram_range
    )

    tfidf_stats = compute_matrix_stats(tfidf_X)
    tfidf_top = extract_top_terms(tfidf_vec, tfidf_X, top_k=top_k)

    tfidf_report = os.path.join(output_dir, f"{base}_tfidf_features_stats.txt")
    save_feature_stats_report(tfidf_report, "TF-IDF", tfidf_stats, tfidf_top)

    print(f"[+] Saved TF-IDF report → {tfidf_report}")

    # --------------------------
    # Combined top-term table
    # --------------------------
    save_vocab_top_table(output_dir, base, bow_top, tfidf_top)

    print("\n[ DONE ]\n")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare BoW and TF-IDF vector representations"
    )

    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV dataset")

    parser.add_argument("--text_col", type=str, default="cleaned_text",
                        help="Column name containing processed text")

    parser.add_argument("--min_df", type=int, default=2,
                        help="Minimum document frequency threshold")

    parser.add_argument("--ngram", type=int, default=1,
                        help="Max n-gram size (1=unigram, 2=bigram, etc.)")

    parser.add_argument("--top_k", type=int, default=30,
                        help="Number of top terms to export")

    args = parser.parse_args()

    run_pipeline(
        args.csv,
        args.text_col,
        min_df=args.min_df,
        ngram=args.ngram,
        top_k=args.top_k
    )
