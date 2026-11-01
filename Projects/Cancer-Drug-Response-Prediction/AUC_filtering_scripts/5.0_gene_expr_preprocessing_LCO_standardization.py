"""
Script: Gene-Expression Preprocessing and LCO Standardisation

Purpose
-------
Preprocess gene-expression data and apply split-specific
StandardScaler following the LCO splits.

The script:

    1. Loads raw gene-expression data.
    2. Selects the landmark genes.
    3. Applies an arcsinh transformation.
    4. Loads the LCO train/validation/test splits.
    5. For each LCO split:
        - identifies the training cell lines,
        - fits StandardScaler using training cell lines ONLY,
        - transforms training gene expression,
        - transforms validation gene expression,
        - transforms test gene expression,
        - saves the fitted scaler.
    6. Saves the standardised gene-expression data for each split.

The raw gene-expression data is never modified.

StandardScaler is fitted separately for each LCO split so that
validation and test gene-expression data cannot influence the
scaling parameters.

This script is dataset-agnostic.
"""

# ============================================================
# Import libraries
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

DEFAULT_CELL_ID = "cellosaurus_id"

REQUIRED_ROLES = (
    "train",
    "validation",
    "test",
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
# Load gene expression
# ============================================================

def load_gene_expression(
    path: Path,
    cell_id: str,
) -> pd.DataFrame:
    """Load raw gene-expression data."""

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

    return (
        data.iloc[:, 0]
        .astype(str)
        .str.strip()
        .tolist()
    )


# ============================================================
# Select landmark genes
# ============================================================

def select_landmark_genes(
    gene_expression: pd.DataFrame,
    landmark_genes: list[str],
) -> list[str]:
    """Select landmark genes present in the data."""

    genes = [
        gene
        for gene in landmark_genes
        if gene in gene_expression.columns
    ]

    if not genes:
        raise ValueError(
            "No landmark genes were found in the "
            "gene-expression data."
        )

    return genes


# ============================================================
# Apply arcsinh transformation
# ============================================================

def apply_arcsinh(
    gene_expression: pd.DataFrame,
    genes: list[str],
) -> pd.DataFrame:
    """Apply arcsinh transformation to gene-expression values."""

    output = gene_expression.copy()

    output[genes] = np.arcsinh(
        output[genes].astype(float)
    )

    return output


# ============================================================
# Load LCO split
# ============================================================

def load_split(
    split_directory: Path,
    cell_id: str,
) -> dict[str, pd.DataFrame]:
    """Load train, validation, and test response files."""

    split_data = {}

    for role in REQUIRED_ROLES:

        path = (
            split_directory
            / f"{role}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {role}.csv in "
                f"{split_directory}"
            )

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

        split_data[role] = data

    return split_data


# ============================================================
# Get cell-line IDs
# ============================================================

def get_cell_line_ids(
    data: pd.DataFrame,
    cell_id: str,
) -> set[str]:
    """Return unique cell-line identifiers."""

    return set(
        data[cell_id]
        .astype(str)
        .str.strip()
    )


# ============================================================
# Get gene-expression feature columns
# ============================================================

def get_gene_expression_columns(
    data: pd.DataFrame,
    cell_id: str,
) -> list[str]:
    """Return gene-expression feature columns."""

    return [
        column
        for column in data.columns
        if column != cell_id
    ]


# ============================================================
# Standardise one LCO split
# ============================================================

def standardise_split(
    gene_expression: pd.DataFrame,
    split_data: dict[str, pd.DataFrame],
    cell_id: str,
) -> tuple[
    dict[str, pd.DataFrame],
    StandardScaler,
]:
    """
    Fit StandardScaler on training cell lines only
    and transform train, validation, and test.
    """

    genes = get_gene_expression_columns(
        gene_expression,
        cell_id,
    )

    if not genes:
        raise ValueError(
            "No gene-expression feature columns were found."
        )

    # --------------------------------------------------------
    # Get cell lines for each split
    # --------------------------------------------------------

    role_ids = {
        role: get_cell_line_ids(
            split_data[role],
            cell_id,
        )
        for role in REQUIRED_ROLES
    }

    # --------------------------------------------------------
    # Check that all split cell lines have gene expression
    # --------------------------------------------------------

    gene_expression_ids = set(
        gene_expression[cell_id]
    )

    for role in REQUIRED_ROLES:

        missing_ids = (
            role_ids[role]
            - gene_expression_ids
        )

        if missing_ids:

            examples = list(
                missing_ids
            )[:5]

            raise ValueError(
                f"{len(missing_ids)} {cell_id} values "
                f"from {role} are missing from the "
                f"gene-expression data. "
                f"Examples: {examples}"
            )

    # --------------------------------------------------------
    # Select training gene expression
    # --------------------------------------------------------

    training_expression = gene_expression[
        gene_expression[cell_id].isin(
            role_ids["train"]
        )
    ].copy()

    if training_expression.empty:
        raise ValueError(
            "No training gene-expression rows were found."
        )

    # --------------------------------------------------------
    # Fit scaler on training cell lines ONLY
    # --------------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(
        training_expression[genes]
    )

    # --------------------------------------------------------
    # Transform train, validation, and test
    # --------------------------------------------------------

    standardised = {}

    for role in REQUIRED_ROLES:

        role_expression = gene_expression[
            gene_expression[cell_id].isin(
                role_ids[role]
            )
        ].copy()

        role_expression[genes] = (
            scaler.transform(
                role_expression[genes]
            )
        )

        standardised[role] = role_expression

    return standardised, scaler


# ============================================================
# Save standardised data
# ============================================================

def save_standardised_split(
    standardised: dict[str, pd.DataFrame],
    scaler: StandardScaler,
    output_directory: Path,
) -> None:
    """Save standardised gene expression and scaler."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for role in REQUIRED_ROLES:

        output_path = (
            output_directory
            / f"{role}_gene_expression_standardised.csv"
        )

        standardised[role].to_csv(
            output_path,
            index=False,
        )

    scaler_path = (
        output_directory
        / "gene_expression_scaler.pkl"
    )

    joblib.dump(
        scaler,
        scaler_path,
    )


# ============================================================
# Process one LCO split
# ============================================================

def process_split(
    split_directory: Path,
    gene_expression: pd.DataFrame,
    output_directory: Path,
    cell_id: str,
) -> None:
    """Process one LCO split."""

    split_data = load_split(
        split_directory,
        cell_id,
    )

    standardised, scaler = standardise_split(
        gene_expression=gene_expression,
        split_data=split_data,
        cell_id=cell_id,
    )

    save_standardised_split(
        standardised=standardised,
        scaler=scaler,
        output_directory=output_directory,
    )

    print(
        f"\nCompleted {split_directory.name}"
    )

    print(
        f"  Training cell lines: "
        f"{standardised['train'][cell_id].nunique():,}"
    )

    print(
        f"  Validation cell lines: "
        f"{standardised['validation'][cell_id].nunique():,}"
    )

    print(
        f"  Test cell lines: "
        f"{standardised['test'][cell_id].nunique():,}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--gene-expression",
        required=True,
        help="Raw gene-expression file.",
    )

    parser.add_argument(
        "--landmark-genes",
        required=True,
        help="Landmark-gene list.",
    )

    parser.add_argument(
        "--split-dir",
        required=True,
        help=(
            "Directory containing split_1, split_2, ..., "
            "with train.csv, validation.csv, and test.csv."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Directory where the split-specific "
            "gene-expression files and scalers will be saved."
        ),
    )

    parser.add_argument(
        "--cell-id",
        default=DEFAULT_CELL_ID,
        help=(
            "Cell-line identifier. "
            f"Default: {DEFAULT_CELL_ID}."
        ),
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of LCO split sets. Default: 5.",
    )

    args = parser.parse_args()

    input_path = Path(
        args.gene_expression
    )

    landmark_path = Path(
        args.landmark_genes
    )

    split_directory = Path(
        args.split_dir
    )

    output_directory = Path(
        args.output
    )

    # --------------------------------------------------------
    # Load raw gene expression
    # --------------------------------------------------------

    gene_expression = load_gene_expression(
        input_path,
        args.cell_id,
    )

    # --------------------------------------------------------
    # Load landmark genes
    # --------------------------------------------------------

    landmark_genes = load_landmark_genes(
        landmark_path
    )

    # --------------------------------------------------------
    # Select landmark genes
    # --------------------------------------------------------

    genes = select_landmark_genes(
        gene_expression,
        landmark_genes,
    )

    # --------------------------------------------------------
    # Keep cell ID + landmark genes
    # --------------------------------------------------------

    processed = gene_expression[
        [args.cell_id] + genes
    ].copy()

    # --------------------------------------------------------
    # Apply arcsinh transformation
    # --------------------------------------------------------

    processed = apply_arcsinh(
        processed,
        genes,
    )

    print("\n==============================")
    print("GENE-EXPRESSION PREPROCESSING")
    print("==============================")

    print(
        f"Input rows: {len(gene_expression):,}"
    )

    print(
        f"Processed rows: {len(processed):,}"
    )

    print(
        f"Landmark genes used: {len(genes):,}"
    )

    # --------------------------------------------------------
    # Process each LCO split
    # --------------------------------------------------------

    print(
        f"\nProcessing {args.n_splits} LCO split sets..."
    )

    for split_index in range(
        1,
        args.n_splits + 1,
    ):

        current_split_directory = (
            split_directory
            / f"split_{split_index}"
        )

        current_output_directory = (
            output_directory
            / f"split_{split_index}"
        )

        if not current_split_directory.exists():
            raise FileNotFoundError(
                f"Missing split directory: "
                f"{current_split_directory}"
            )

        process_split(
            split_directory=current_split_directory,
            gene_expression=processed,
            output_directory=current_output_directory,
            cell_id=args.cell_id,
        )

    print(
        "\n=============================="
    )

    print(
        "GENE-EXPRESSION PROCESSING COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        f"Output:\n"
        f"{output_directory.resolve()}"
    )


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":
    main()