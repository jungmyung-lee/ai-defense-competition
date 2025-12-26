"""
text_cleaning_preprocessing.py

Author: Jungmyung Lee

Description:
    Text normalization & preprocessing pipeline for NLP-based
    classification tasks. This script loads a CSV dataset,
    performs multi-stage text cleaning, and outputs both
    processed text and analysis logs.

Processing steps:
    - Lowercasing
    - Whitespace normalization
    - Punctuation removal
    - Digit handling (keep / remove)
    - Stopword removal
    - Tokenization

Outputs:
    ./outputs/text_preprocessing/<dataset_name>/
        - cleaned_text.csv
        - preprocessing_log.txt
        - sample_before_after.txt

Python Dependencies:
    - pandas
    - re
    - string
"""

import os
import re
import argparse
import string
import pandas as pd
from datetime import datetime


# =========================================================
# Helpers
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_dataset(csv_path, text_col):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise ValueError(f"[ERROR] Column '{text_col}' not found in CSV")

    if len(df) == 0:
        raise ValueError("[ERROR] CSV file is empty")

    return df


# =========================================================
# Text processing operations
# =========================================================
STOPWORDS = {
    "a","an","the","and","or","of","in","on","for","to",
    "is","are","was","were","be","been","this","that"
}


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def remove_punctuation(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


def remove_digits(text):
    return re.sub(r"\d+", "", text)


def tokenize(text):
    return text.split(" ")


def remove_stopwords(tokens):
    return [t for t in tokens if t.lower() not in STOPWORDS]


# =========================================================
# Pipeline
# =========================================================
def preprocess_text_series(series,
                           keep_digits=False,
                           remove_stop=True):
    cleaned = []
    tokenized = []
    logs = []

    for idx, raw in enumerate(series):
        original = str(raw)

        # Step 1 — lowercase
        t = original.lower()

        # Step 2 — normalize whitespace
        t = normalize_whitespace(t)

        # Step 3 — punctuation removal
        t = remove_punctuation(t)

        # Step 4 — digit processing
        if not keep_digits:
            t = remove_digits(t)

        # Step 5 — tokenize
        tokens = tokenize(t)

        # Step 6 — stopword removal
        if remove_stop:
            tokens = remove_stopwords(tokens)

        # Reconstruct text
        cleaned_text = " ".join(tokens)

        cleaned.append(cleaned_text)
        tokenized.append(tokens)

        # Collect log
        logs.append({
            "index": idx,
            "original": original,
            "cleaned": cleaned_text,
            "token_count": len(tokens)
        })

    return cleaned, tokenized, logs


# =========================================================
# Save outputs
# =========================================================
def save_output_files(df,
                      cleaned,
                      tokens,
                      logs,
                      output_dir,
                      base):

    # 1) Save cleaned dataset
    out_df = df.copy()
    out_df["cleaned_text"] = cleaned
    out_df["tokens"] = [" ".join(t) for t in tokens]

    csv_path = os.path.join(output_dir, f"{base}_cleaned_text.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"[+] Saved cleaned dataset → {csv_path}")

    # 2) Save before/after samples
    sample_path = os.path.join(output_dir, f"{base}_sample_before_after.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("TEXT CLEANING — SAMPLE BEFORE / AFTER\n\n")
        for rec in logs[:30]:
            f.write(f"[INDEX {rec['index']}]\n")
            f.write(f"ORIGINAL: {rec['original']}\n")
            f.write(f"CLEANED : {rec['cleaned']}\n")
            f.write(f"TOKENS  : {rec['token_count']}\n")
            f.write("-" * 60 + "\n")
    print(f"[+] Saved sample comparison → {sample_path}")

    # 3) Save log summary
    log_path = os.path.join(output_dir, f"{base}_preprocessing_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("TEXT PREPROCESSING LOG\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Total rows processed: {len(logs)}\n\n")

        lengths = [rec["token_count"] for rec in logs]
        f.write(f"Min tokens: {min(lengths)}\n")
        f.write(f"Max tokens: {max(lengths)}\n")
        f.write(f"Avg tokens: {sum(lengths)/len(lengths):.2f}\n")

    print(f"[+] Saved preprocessing log → {log_path}")


# =========================================================
# Runner
# =========================================================
def run_pipeline(csv_path,
                 text_col,
                 keep_digits=False,
                 remove_stop=True):

    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs", "text_preprocessing", base)
    ensure_dir(output_dir)

    print("\n[ TEXT PREPROCESSING PIPELINE ]")
    print(f"Input  : {csv_path}")
    print(f"Column : {text_col}")
    print(f"Output : {output_dir}\n")

    df = load_dataset(csv_path, text_col)

    cleaned, tokens, logs = preprocess_text_series(
        df[text_col],
        keep_digits=keep_digits,
        remove_stop=remove_stop
    )

    save_output_files(df, cleaned, tokens, logs, output_dir, base)

    print("\n[ DONE ]\n")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text normalization & preprocessing pipeline"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV dataset"
    )

    parser.add_argument(
        "--text_col",
        type=str,
        default="text",
        help="Column name containing text data"
    )

    parser.add_argument(
        "--keep_digits",
        action="store_true",
        help="Do not remove numeric digits"
    )

    parser.add_argument(
        "--keep_stopwords",
        action="store_true",
        help="Do not remove stopwords"
    )

    args = parser.parse_args()

    run_pipeline(
        args.csv,
        args.text_col,
        keep_digits=args.keep_digits,
        remove_stop=not args.keep_stopwords
    )
