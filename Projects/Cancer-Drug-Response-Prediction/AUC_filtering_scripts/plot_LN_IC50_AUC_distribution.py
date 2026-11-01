"""
Script: PCA of Gene-Expression Distributions Across Datasets

Purpose
-------
Compare the overall gene-expression distributions of:

    - GDSC1
    - GDSC2
    - CTRPv1
    - CTRPv2
    - CCLE

The script:

    1. Loads raw gene-expression data for each dataset.
    2. Uses the same landmark-gene list for all datasets.
    3. Keeps cellosaurus_id + the shared landmark genes.
    4. Applies the same arcsinh transformation used in the
       gene-expression preprocessing pipeline.
    5. Combines the datasets.
    6. Applies PCA to the combined gene-expression data.
    7. Creates a PC1 vs PC2 plot coloured by dataset.
    8. Saves the PCA figure.

No LCO splitting is performed.
No dataset-specific StandardScaler is applied.

The purpose is to visualise overall dataset-level differences
in gene-expression space.

This script does not modify the original files.
"""

# ============================================================
# Import libraries
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


# ============================================================
# Configuration
# ============================================================

DEFAULT_CELL_ID = "cellosaurus_id"

DATASETS = (
    "GDSC1",
    "GDSC2",
    "CTRPv1",
    "CTRPv2",
    "CCLE",
)


# ============================================================
# Load CSV
# ============================================================

def load_csv(
    path: Path,
) -> pd.DataFrame:
    """Load a CSV file."""

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


# ============================================================
# Load gene-expression data
# ============================================================

def load_gene_expression(
    path: Path,
    cell_id: str,
) -> pd.DataFrame:
    """Load and clean gene-expression data."""

    data = load_csv(path)

    if cell_id not in data.columns:
        raise ValueError(
            f"{path} is missing required column: "
            f"{cell_id!r}."
        )

    data[cell_id] = (
        data[cell_id]
        .astype(str)
        .str.strip()
    )

    # Keep one row per cell line
    data = data.drop_duplicates(
        subset=[cell_id],
        keep="first",
    )

    return data


# ============================================================
# Load landmark genes
# ============================================================

def load_landmark_genes(
    path: Path,
) -> list[str]:
    """Load landmark-gene names."""

    data = load_csv(path)

    if data.empty:
        raise ValueError(
            f"Landmark-gene file is empty: {path}"
        )

    genes = (
        data.iloc[:, 0]
        .astype(str)
        .str.strip()
        .tolist()
    )

    return genes


# ============================================================
# Select landmark genes
# ============================================================

def select_shared_landmark_genes(
    datasets: dict[str, pd.DataFrame],
    landmark_genes: list[str],
) -> list[str]:
    """
    Select landmark genes that are present in every dataset.
    """

    shared_genes = set(landmark_genes)

    for data in datasets.values():

        available_genes = set(
            data.columns
        )

        shared_genes &= available_genes

    genes = [
        gene
        for gene in landmark_genes
        if gene in shared_genes
    ]

    if not genes:
        raise ValueError(
            "No landmark genes are shared across all datasets."
        )

    return genes


# ============================================================
# Apply arcsinh transformation
# ============================================================

def apply_arcsinh(
    data: pd.DataFrame,
    genes: list[str],
) -> pd.DataFrame:
    """Apply arcsinh transformation to gene-expression values."""

    output = data.copy()

    output[genes] = np.arcsinh(
        output[genes].astype(float)
    )

    return output


# ============================================================
# Prepare dataset
# ============================================================

def prepare_dataset(
    data: pd.DataFrame,
    dataset_name: str,
    cell_id: str,
    genes: list[str],
) -> pd.DataFrame:
    """Prepare one dataset for PCA."""

    missing_genes = [
        gene
        for gene in genes
        if gene not in data.columns
    ]

    if missing_genes:
        raise ValueError(
            f"{dataset_name} is missing "
            f"{len(missing_genes)} landmark genes."
        )

    output = data[
        [cell_id] + genes
    ].copy()

    output = apply_arcsinh(
        output,
        genes,
    )

    output["Dataset"] = dataset_name

    return output


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data-root",
        required=True,
        help=(
            "Root data directory containing the dataset "
            "subdirectories."
        ),
    )

    parser.add_argument(
        "--landmark-genes",
        required=True,
        help="Path to landmark_gene_reduced.csv.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path where the PCA figure will be saved.",
    )

    parser.add_argument(
        "--cell-id",
        default=DEFAULT_CELL_ID,
        help=(
            "Cell-line identifier. "
            f"Default: {DEFAULT_CELL_ID}."
        ),
    )

    args = parser.parse_args()

    data_root = Path(
        args.data_root
    )

    landmark_path = Path(
        args.landmark_genes
    )

    output_path = Path(
        args.output
    )

    # --------------------------------------------------------
    # Load landmark genes
    # --------------------------------------------------------

    landmark_genes = load_landmark_genes(
        landmark_path
    )

    print(
        f"Landmark genes in list: "
        f"{len(landmark_genes):,}"
    )

    # --------------------------------------------------------
    # Load all datasets
    # --------------------------------------------------------

    datasets = {}

    for dataset_name in DATASETS:

        path = (
            data_root
            / dataset_name
            / "gene_expression.csv"
        )

        print(
            f"\nLoading {dataset_name}:"
        )

        print(path)

        datasets[dataset_name] = (
            load_gene_expression(
                path,
                args.cell_id,
            )
        )

        print(
            f"Cell lines: "
            f"{len(datasets[dataset_name]):,}"
        )

    # --------------------------------------------------------
    # Find landmark genes shared by all datasets
    # --------------------------------------------------------

    genes = select_shared_landmark_genes(
        datasets,
        landmark_genes,
    )

    print(
        f"\nShared landmark genes used for PCA: "
        f"{len(genes):,}"
    )

    # --------------------------------------------------------
    # Prepare each dataset
    # --------------------------------------------------------

    prepared = []

    for dataset_name in DATASETS:

        print(
            f"\nProcessing {dataset_name}..."
        )

        dataset = prepare_dataset(
            data=datasets[dataset_name],
            dataset_name=dataset_name,
            cell_id=args.cell_id,
            genes=genes,
        )

        prepared.append(
            dataset
        )

        print(
            f"Processed cell lines: "
            f"{len(dataset):,}"
        )

    # --------------------------------------------------------
    # Combine datasets
    # --------------------------------------------------------

    combined = pd.concat(
        prepared,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Extract gene-expression matrix
    # --------------------------------------------------------

    X = combined[genes].copy()

    # --------------------------------------------------------
    # Check for missing / infinite values
    # --------------------------------------------------------

    if X.isna().any().any():
        raise ValueError(
            "Missing values detected in the "
            "gene-expression matrix."
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Infinite values detected in the "
            "gene-expression matrix."
        )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    print(
        "\nRunning PCA..."
    )

    pca = PCA(
        n_components=2,
        random_state=42,
    )

    principal_components = pca.fit_transform(
        X
    )

    combined["PC1"] = principal_components[:, 0]

    combined["PC2"] = principal_components[:, 1]

    explained_variance = (
        pca.explained_variance_ratio_
        * 100
    )

    # --------------------------------------------------------
    # Create PCA figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    for dataset_name in DATASETS:

        subset = combined[
            combined["Dataset"]
            == dataset_name
        ]

        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            alpha=0.6,
            s=25,
            label=dataset_name,
        )

    plt.xlabel(
        f"PC1 ({explained_variance[0]:.1f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({explained_variance[1]:.1f}% variance)"
    )

    plt.title(
        "PCA of Gene-Expression Profiles Across Datasets"
    )

    plt.legend(
        title="Dataset"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        "PCA COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        f"Datasets: {len(DATASETS)}"
    )

    print(
        f"Shared landmark genes: {len(genes):,}"
    )

    print(
        f"PC1 variance explained: "
        f"{explained_variance[0]:.2f}%"
    )

    print(
        f"PC2 variance explained: "
        f"{explained_variance[1]:.2f}%"
    )

    print(
        f"Total PC1 + PC2 variance: "
        f"{explained_variance.sum():.2f}%"
    )

    print(
        f"\nFigure saved to:\n"
        f"{output_path.resolve()}"
    )


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":
    main()