"""
Script: Cross-Study Feature Preprocessing

Purpose
-------
Prepare gene-expression and drug-fingerprint features for
cross-study datasets supplied through the command line. 

    - CTRPv1
    - CTRPv2
    - GDSC1
    - CCLE

Gene expression:
    1. Load raw gene-expression data.
    2. Select the same landmark genes used for GDSC2.
    3. Apply arcsinh transformation.
    4. Keep cellosaurus_id.
    5. Save the processed gene-expression data.

Drug fingerprints:
    1. Load the raw 128-bit fingerprint matrix.
    2. PubChem IDs are stored as column headers.
    3. Transpose the matrix.
    4. Convert the PubChem ID index into a pubchem_id column.
    5. Name fingerprint features consistently as 0 ... 127.
    6. Save the processed fingerprint data.

Important
---------
Gene-expression data is NOT scaled.

Fingerprint feature names are explicitly set to:

    0, 1, 2, ..., 127

This ensures that the fingerprint columns match the GDSC2
fingerprint representation.

The raw files are never modified.
"""

# ============================================================
# Import libraries
# ============================================================
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================

DEFAULT_CELL_ID = "cellosaurus_id"
DEFAULT_DRUG_ID = "pubchem_id"

# ============================================================
# Load CSV
# ============================================================

def load_csv(path: Path,) -> pd.DataFrame:
    """Load a CSV file."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path,low_memory=False,)

# ============================================================
# Gene-expression preprocessing
# ============================================================

def load_gene_expression(path: Path,cell_id: str,) -> pd.DataFrame:
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

    # Keep one row per cell line.
    data = data.drop_duplicates(
        subset=[cell_id],
        keep="first",
    )

    return data

# ============================================================
# Load landmark genes
# ============================================================

def load_landmark_genes(path: Path,) -> list[str]:
    """Load landmark-gene names."""

    data = load_csv(path)
    if data.empty:
        raise ValueError(f"Landmark-gene file is empty: {path}")

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

def select_landmark_genes(gene_expression: pd.DataFrame,landmark_genes: list[str],) -> list[str]:
    """Select landmark genes present in the data."""

    genes = [
        gene
        for gene in landmark_genes
        if gene in gene_expression.columns
    ]

    if not genes:
        raise ValueError(
            "No landmark genes were found "
            "in the gene-expression data."
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
# Process gene expression
# ============================================================

def process_gene_expression(
    input_path: Path,
    landmark_path: Path,
    output_path: Path,
    cell_id: str,
) -> None:
    """Process one cross-study gene-expression file."""

    print("\n" + "=" * 70)
    print("GENE-EXPRESSION PREPROCESSING")
    print("=" * 70)

    print(
        f"Input:\n{input_path}"
    )

    gene_expression = load_gene_expression(
        input_path,
        cell_id,
    )

    landmark_genes = load_landmark_genes(
        landmark_path
    )

    genes = select_landmark_genes(
        gene_expression,
        landmark_genes,
    )

    missing_genes = [
        gene
        for gene in landmark_genes
        if gene not in gene_expression.columns
    ]

    print(
        f"Input rows: {len(gene_expression):,}"
    )

    print(
        f"Landmark genes requested: "
        f"{len(landmark_genes):,}"
    )

    print(
        f"Landmark genes found: "
        f"{len(genes):,}"
    )

    print(
        f"Landmark genes missing: "
        f"{len(missing_genes):,}"
    )

    if missing_genes:
        print(
            "\nWarning: Some landmark genes were not found."
        )

    # --------------------------------------------------------
    # Keep cell ID + landmark genes
    # --------------------------------------------------------

    processed = gene_expression[
        [cell_id] + genes
    ].copy()

    # --------------------------------------------------------
    # Apply arcsinh transformation
    # --------------------------------------------------------

    processed = apply_arcsinh(
        processed,
        genes,
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save processed gene expression
    # --------------------------------------------------------

    processed.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved:\n{output_path}"
    )

    print(
        f"Rows: {len(processed):,}"
    )

    print(
        f"Columns: {len(processed.columns):,}"
    )


# ============================================================
# Drug fingerprint preprocessing
# ============================================================

def process_fingerprints(
    input_path: Path,
    output_path: Path,
    drug_id: str,
) -> None:
    """
    Process one raw fingerprint matrix.

    Raw structure:

        128 rows
        PubChem IDs as column names

    Example:

        46883536  56649450  25227847
        1         1         1
        1         1         0
        0         0         0
        ...

    After transposition:

        pubchem_id  0  1  2  ...  127
        46883536    1  1  0  ...    1
        56649450    1  1  0  ...    0
    """

    print("\n" + "=" * 70)
    print("DRUG FINGERPRINT PREPROCESSING")
    print("=" * 70)

    print(
        f"Input:\n{input_path}"
    )

    fingerprints = load_csv(
        input_path
    )

    print(
        f"Raw shape: {fingerprints.shape}"
    )

    # --------------------------------------------------------
    # Verify raw fingerprint structure
    # --------------------------------------------------------

    if fingerprints.shape[0] != 128:
        raise ValueError(
            f"Expected 128 fingerprint rows, but found "
            f"{fingerprints.shape[0]} in {input_path}."
        )

    if fingerprints.shape[1] == 0:
        raise ValueError(
            f"No fingerprint columns found in {input_path}."
        )

    # --------------------------------------------------------
    # PubChem IDs are the raw column headers
    # --------------------------------------------------------

    pubchem_ids = (
        fingerprints.columns
        .astype(str)
        .str.strip()
    )

    if pubchem_ids.duplicated().any():
        duplicate_ids = (
            pubchem_ids[
                pubchem_ids.duplicated()
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate PubChem IDs found in raw "
            f"fingerprint file {input_path}: "
            f"{duplicate_ids[:10]}"
        )

    # --------------------------------------------------------
    # Convert fingerprint values to numeric
    # --------------------------------------------------------

    fingerprints = fingerprints.apply(
        pd.to_numeric,
        errors="raise",
    )

    # --------------------------------------------------------
    # Transpose the COMPLETE matrix
    #
    # Rows become drugs.
    # Columns become fingerprint bits.
    # --------------------------------------------------------

    transposed = fingerprints.T.copy()

    # --------------------------------------------------------
    # Set the PubChem ID as the identifier
    # --------------------------------------------------------

    transposed.index = pubchem_ids

    transposed.index.name = drug_id

    # --------------------------------------------------------
    # Explicitly assign fingerprint feature names
    #
    # This is the critical correction.
    #
    # We do NOT use the fingerprint values as column names.
    # --------------------------------------------------------

    transposed.columns = [
        str(i)
        for i in range(128)
    ]

    # --------------------------------------------------------
    # Convert index into a normal column
    # --------------------------------------------------------

    processed = transposed.reset_index()

    # --------------------------------------------------------
    # Ensure PubChem ID is the first column
    # --------------------------------------------------------

    processed = processed[
        [drug_id]
        + [
            str(i)
            for i in range(128)
        ]
    ]

    # --------------------------------------------------------
    # Validate final structure
    # --------------------------------------------------------

    expected_columns = [
        drug_id
    ] + [
        str(i)
        for i in range(128)
    ]

    if processed.columns.tolist() != expected_columns:
        raise ValueError(
            "Fingerprint columns are not in the expected "
            "order."
        )

    if processed[drug_id].duplicated().any():
        raise ValueError(
            "Duplicate PubChem IDs were created during "
            "fingerprint preprocessing."
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save processed fingerprints
    # --------------------------------------------------------

    processed.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nFingerprint preprocessing summary:"
    )

    print(
        f"Fingerprint bits: 128"
    )

    print(
        f"Drugs: {len(processed):,}"
    )

    print(
        f"Output shape: {processed.shape}"
    )

    print(
        f"Output:\n{output_path}"
    )


# ============================================================
# Process one dataset
# ============================================================

def process_dataset(
    dataset: str,
    project_root: Path,
    landmark_path: Path,
) -> None:
    """Process gene expression and fingerprints for one dataset."""

    # --------------------------------------------------------
    # Input paths
    # --------------------------------------------------------

    gene_expression_path = (
        project_root
        / "data"
        / "raw"
        / dataset
        / "gene_expression.csv"
    )

    fingerprint_path = (
        project_root
        / "data"
        / "raw"
        / dataset
        / "drug_fingerprints"
        / "pubchem_id_to_demorgan_128_map.csv"
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    gene_expression_output = (
        project_root
        / "data"
        / "gene_expr_transformed"
        / f"gene_expression_{dataset}_processed.csv"
    )

    fingerprint_output = (
        project_root
        / "data"
        / "cross_study_fingerprints"
        / f"{dataset}_fingerprints_128_transposed.csv"
    )

    # --------------------------------------------------------
    # Gene expression
    # --------------------------------------------------------

    process_gene_expression(
        input_path=gene_expression_path,
        landmark_path=landmark_path,
        output_path=gene_expression_output,
        cell_id=DEFAULT_CELL_ID,
    )

    # --------------------------------------------------------
    # Fingerprints
    # --------------------------------------------------------

    process_fingerprints(
        input_path=fingerprint_path,
        output_path=fingerprint_output,
        drug_id=DEFAULT_DRUG_ID,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--project-root",
        required=True,
        help="Project root directory.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Cross-study dataset names to process.",
    )
    
    args = parser.parse_args()

    project_root = Path(
        args.project_root
    )

    landmark_path = (
        project_root
        / "data"
        / "meta"
        / "gene_lists"
        / "landmark_genes_reduced.csv"
    )

    if not landmark_path.exists():
        raise FileNotFoundError(
            f"Landmark-gene file not found:\n"
            f"{landmark_path}"
        )

    # --------------------------------------------------------
    # Process datasets
    # --------------------------------------------------------

    for dataset in args.datasets:

        print("\n")
        print("#" * 70)
        print(f"PROCESSING {dataset}")
        print("#" * 70)

        process_dataset(
            dataset=dataset,
            project_root=project_root,
            landmark_path=landmark_path,
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CROSS-STUDY FEATURE PREPROCESSING COMPLETE")
    print("=" * 70)

    print("\nProcessed datasets:")

    for dataset in args.datasets:
        print(f"  - {dataset}")

    print("\nGene-expression output:")

    print(
        project_root
        / "data"
        / "gene_expr_transformed"
    )

    print("\nFingerprint output:")

    print(
        project_root
        / "data"
        / "cross_study_fingerprints"
    )
# ============================================================
# Run script
# ============================================================
if __name__ == "__main__":
    main()
