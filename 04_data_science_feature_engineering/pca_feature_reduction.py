"""
pca_feature_reduction.py

Author: Jungmyung Lee

Description:
    Performs Principal Component Analysis (PCA) on numeric features
    to reduce dimensionality while retaining maximum variance.

    Pipeline:
    1) Load CSV dataset
    2) Select numeric columns only
    3) Standardize features (z-score)
    4) Apply PCA with user-defined component count OR variance target
    5) Save:
        - PCA-transformed dataset
        - Component loading matrix
        - Explained variance report

Outputs:
    ./outputs/pca_reduction/<dataset_name>/

Python Dependencies:
    - pandas
    - NumPy
    - scikit-learn

Run Example:
    python pca_feature_reduction.py --csv sample.csv --n_components 3

Variance target example:
    python pca_feature_reduction.py --csv sample.csv --var 0.9
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =========================================================
# Utils
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
# Select numeric columns and standardize
# =========================================================
def prepare_numeric_matrix(df):
    num_df = df.select_dtypes(include=[np.number])

    if num_df.shape[1] == 0:
        raise ValueError("[ERROR] No numeric columns available for PCA")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num_df)

    return num_df.columns.tolist(), X_scaled, scaler


# =========================================================
# Fit PCA
# =========================================================
def run_pca(X, n_components=None, var_target=None):
    if var_target is not None:
        pca = PCA(n_components=var_target)
    elif n_components is not None:
        pca = PCA(n_components=n_components)
    else:
        raise ValueError("Either n_components or var must be specified")

    X_pca = pca.fit_transform(X)

    return pca, X_pca


# =========================================================
# Save PCA dataset
# =========================================================
def save_pca_dataset(output_dir, base, df_original, X_pca):
    pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]

    out_df = df_original.copy()
    for i, name in enumerate(pca_cols):
        out_df[name] = X_pca[:, i]

    path = os.path.join(output_dir, f"{base}_pca_reduced.csv")
    out_df.to_csv(path, index=False)

    print(f"[+] Saved PCA reduced dataset → {path}")


# =========================================================
# Save PCA loading matrix
# =========================================================
def save_loading_matrix(output_dir, base, feature_names, pca):
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    path = os.path.join(output_dir, f"{base}_pca_loadings.csv")
    loadings.to_csv(path)

    print(f"[+] Saved PCA loading matrix → {path}")


# =========================================================
# Save variance report
# =========================================================
def save_variance_report(output_dir, base, pca):
    path = os.path.join(output_dir, f"{base}_pca_variance_report.txt")

    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)

    with open(path, "w", encoding="utf-8") as f:
        f.write("PCA VARIANCE EXPLAINABILITY REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write(f"Principal components: {len(ev)}\n\n")

        for i, (v, c) in enumerate(zip(ev, cum)):
            f.write(
                f"PC{i+1:02d}  "
                f"var={v:.6f}   "
                f"cumulative={c:.6f}\n"
            )

        f.write("\nInterpretation Guide:\n")
        f.write(" - Higher cumulative variance = more information retained\n")
        f.write(" - Choose smallest dimension that satisfies target variance\n")

    print(f"[+] Saved variance report → {path}")


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(csv_path, n_components, var_target):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("outputs", "pca_reduction", base)

    ensure_dir(output_dir)

    print("\n[ PCA FEATURE DIMENSION REDUCTION ]")
    print(f"Input  : {csv_path}")
    print(f"Output : {output_dir}")

    if n_components:
        print(f"Mode   : n_components = {n_components}")
    if var_target:
        print(f"Mode   : variance target = {var_target}\n")

    df = load_dataset(csv_path)

    feature_names, X_scaled, _ = prepare_numeric_matrix(df)

    pca, X_pca = run_pca(
        X_scaled,
        n_components=n_components,
        var_target=var_target
    )

    save_pca_dataset(output_dir, base, df, X_pca)
    save_loading_matrix(output_dir, base, feature_names, pca)
    save_variance_report(output_dir, base, pca)

    print("\n[ DONE ]\n")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCA-based numeric feature dimension reduction"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV dataset"
    )

    parser.add_argument(
        "--n_components",
        type=int,
        default=None,
        help="Number of principal components"
    )

    parser.add_argument(
        "--var",
        type=float,
        default=None,
        help="Target explained variance ratio (e.g., 0.9)"
    )

    args = parser.parse_args()

    run_pipeline(
        args.csv,
        args.n_components,
        args.var
    )
