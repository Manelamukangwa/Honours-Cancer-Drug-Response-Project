"""
Script: Training Input Audit

Purpose
-------
Audit the exact training input that will be supplied to the
Random Forest model.

For each LCO split, the script:
    1. Loads the standardised training response data.
    2. Loads the single processed gene-expression file.
    3. Loads the single processed gene-expression file.
    4. Looks up gene-expression rows using the cell-line identifier.
    5. Loads drug fingerprints.
    6. Merges response, gene-expression, and fingerprint data.
    7. Removes identifiers and response columns from the feature matrix.
    8. Saves the first 30 rows of the final training input.

The gene-expression file is not split according to the LCO splits.
The LCO response data determines which gene-expression rows are
needed for each split.

Gene-expression data is assumed to have already been:
    - restricted to landmark genes
    - transformed using arcsinh

No gene-expression scaling is performed because Random Forest
does not require feature scaling.

The audit does not train a model.

This script is dataset-agnostic.
"""

# ============================================================
# Import libraries
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

DEFAULT_CELL_ID = "cellosaurus_id"
DEFAULT_DRUG_ID = "pubchem_id"
DEFAULT_RESPONSE_COLUMN = "LN_IC50_standardised"
DEFAULT_N_SPLITS = 5
AUDIT_ROWS = 30

EXCLUDED_COLUMNS = [
    "cellosaurus_id",
    "pubchem_id",
    "LN_IC50_standardised",
    "LN_IC50_curvecurator",
    "cell_line_name",
    "tissue",
]


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
# Load response split
# ============================================================

def load_response_split(
    split_dir: Path,
    split_index: int,
    split_name: str,
) -> pd.DataFrame:
    """Load one standardised response split."""

    path = (
        split_dir
        / f"split_{split_index}"
        / f"{split_name}_standardised.csv"
    )

    return load_csv(path)


# ============================================================
# Validate response data
# ============================================================

def validate_response_data(
    data: pd.DataFrame,
    cell_id: str,
    drug_id: str,
    response_column: str,
) -> None:
    """Validate required response columns."""

    required_columns = [
        cell_id,
        drug_id,
        response_column,
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Response data is missing required columns: "
            f"{missing}"
        )


# ============================================================
# Load processed gene expression
# ============================================================

def load_gene_expression(
    path: Path,
    cell_id: str,
) -> pd.DataFrame:
    """Load the processed gene-expression file."""

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
# Load fingerprints
# ============================================================

def load_fingerprints(
    path: Path,
    drug_id: str,
) -> pd.DataFrame:
    """Load drug fingerprint data."""

    data = load_csv(path)

    if drug_id not in data.columns:
        raise ValueError(
            f"{path} is missing required column: "
            f"{drug_id!r}."
        )

    data[drug_id] = (
        data[drug_id]
        .astype(str)
        .str.strip()
    )

    data = data.drop_duplicates(
        subset=[drug_id],
        keep="first",
    )

    return data


# ============================================================
# Match gene expression to response data
# ============================================================

def match_gene_expression(
    response_data: pd.DataFrame,
    gene_expression: pd.DataFrame,
    cell_id: str,
) -> pd.DataFrame:
    """Keep gene-expression rows required by the response data."""

    cell_ids = (
        response_data[cell_id]
        .astype(str)
        .str.strip()
        .unique()
    )

    matched = gene_expression[
        gene_expression[cell_id].isin(cell_ids)
    ].copy()

    return matched


# ============================================================
# Build complete training data
# ============================================================

def build_training_data(
    response_data: pd.DataFrame,
    gene_expression: pd.DataFrame,
    fingerprints: pd.DataFrame,
    cell_id: str,
    drug_id: str,
) -> pd.DataFrame:
    """Merge response, gene expression, and fingerprints."""

    data = response_data.merge(
        gene_expression,
        on=cell_id,
        how="inner",
    )

    data = data.merge(
        fingerprints,
        on=drug_id,
        how="inner",
    )

    return data


# ============================================================
# Build final feature matrix
# ============================================================

def build_feature_matrix(
    data: pd.DataFrame,
    response_column: str,
) -> pd.DataFrame:
    """Remove identifiers and response columns from training data."""

    excluded = [
        column
        for column in EXCLUDED_COLUMNS
        if column in data.columns
    ]

    features = data.drop(
        columns=excluded,
        errors="ignore",
    )

    if features.empty:
        raise ValueError(
            "No feature columns remain after removing "
            "identifiers and response columns."
        )

    return features


# ============================================================
# Audit one split
# ============================================================

def audit_split(
    split_dir: Path,
    gene_expression: pd.DataFrame,
    fingerprints: pd.DataFrame,
    split_index: int,
    cell_id: str,
    drug_id: str,
    response_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the final training input for one LCO split."""

    train = load_response_split(
        split_dir,
        split_index,
        "train",
    )

    training_data = train.copy()

    validate_response_data(
        training_data,
        cell_id,
        drug_id,
        response_column,
    )

    matched_gene_expression = match_gene_expression(
        training_data,
        gene_expression,
        cell_id,
    )

    training_data = build_training_data(
        training_data,
        matched_gene_expression,
        fingerprints,
        cell_id,
        drug_id,
    )

    features = build_feature_matrix(
        training_data,
        response_column,
    )

    return training_data, features


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--split-dir",
        required=True,
        help=(
            "Directory containing the standardised "
            "LCO split folders."
        ),
    )

    parser.add_argument(
        "--gene-expression",
        required=True,
        help=(
            "Processed gene-expression file produced by "
            "the gene-expression preprocessing script."
        ),
    )

    parser.add_argument(
        "--fingerprints",
        required=True,
        help="Drug fingerprint file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory for audit outputs.",
    )

    parser.add_argument(
        "--cell-id",
        default=DEFAULT_CELL_ID,
        help=(
            "Cell-line identifier used to match gene expression. "
            f"Default: {DEFAULT_CELL_ID}."
        ),
    )

    parser.add_argument(
        "--drug-id",
        default=DEFAULT_DRUG_ID,
        help=(
            "Drug identifier used to match fingerprints. "
            f"Default: {DEFAULT_DRUG_ID}."
        ),
    )

    parser.add_argument(
        "--response-column",
        default=DEFAULT_RESPONSE_COLUMN,
        help=(
            "Standardised response column. "
            f"Default: {DEFAULT_RESPONSE_COLUMN}."
        ),
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=(
            "Number of LCO splits. "
            f"Default: {DEFAULT_N_SPLITS}."
        ),
    )

    args = parser.parse_args()

    split_dir = Path(args.split_dir)

    gene_expression_path = Path(
        args.gene_expression
    )

    fingerprints_path = Path(
        args.fingerprints
    )

    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load shared processed data
    # --------------------------------------------------------

    gene_expression = load_gene_expression(
        gene_expression_path,
        args.cell_id,
    )

    fingerprints = load_fingerprints(
        fingerprints_path,
        args.drug_id,
    )

    print()
    print("==============================")
    print("TRAINING INPUT AUDIT")
    print("==============================")

    print(
        f"Processed gene-expression rows: "
        f"{len(gene_expression):,}"
    )

    print(
        f"Fingerprint rows: "
        f"{len(fingerprints):,}"
    )

    # --------------------------------------------------------
    # Audit each LCO split
    # --------------------------------------------------------

    for split_index in range(
        1,
        args.n_splits + 1,
    ):

        training_data, features = audit_split(
            split_dir=split_dir,
            gene_expression=gene_expression,
            fingerprints=fingerprints,
            split_index=split_index,
            cell_id=args.cell_id,
            drug_id=args.drug_id,
            response_column=args.response_column,
        )

        # ----------------------------------------------------
        # Save only the first 30 rows
        # ----------------------------------------------------

        audit_data = training_data.head(
            AUDIT_ROWS
        ).copy()

        audit_features = features.head(
            AUDIT_ROWS
        ).copy()

        split_output_dir = (
            output_dir
            / f"split_{split_index}"
        )

        split_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        audit_data.to_csv(
            split_output_dir
            / "training_input_audit.csv",
            index=False,
        )

        audit_features.to_csv(
            split_output_dir
            / "training_features_audit.csv",
            index=False,
        )

        print()
        print(f"Split {split_index}")
        print(
            f"Training rows: "
            f"{len(training_data):,}"
        )
        print(
            f"Training feature columns: "
            f"{features.shape[1]:,}"
        )
        print(
            f"Audit rows saved: "
            f"{len(audit_data):,}"
        )

    print()
    print("==============================")
    print("TRAINING INPUT AUDIT COMPLETE")
    print("==============================")

    print(
        f"Output:\n{output_dir.resolve()}"
    )


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":
    main()