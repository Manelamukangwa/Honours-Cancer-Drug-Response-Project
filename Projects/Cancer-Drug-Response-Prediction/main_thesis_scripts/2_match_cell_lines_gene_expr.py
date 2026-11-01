"""
Script: Match Cell Lines with Gene-Expression Data

Purpose
-------
Keep only response observations for cell lines that have
matching gene-expression data.

This script is dataset-agnostic and is intended to be run
after response-data cleaning 

The response dataset is filtered so that only cell lines
present in the gene-expression dataset are retained.

No train/validation/test splitting is performed here.

Inputs
------
1. A cleaned response dataset containing: cell_line_name

2. A gene-expression dataset containing: cell_line_name

Output
------
A response dataset containing only observations for cell lines
that are also present in the gene-expression dataset.

The original input files are never modified.
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

DEFAULT_CELL_LINE_COLUMN = "cell_line_name"


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
# Check required column
# ============================================================

def check_column(
    data: pd.DataFrame,
    column: str,
    file_path: Path,
) -> None:
    """Check that the required column exists."""

    if column not in data.columns:
        raise ValueError(
            f"{file_path} is missing required column: "
            f"{column!r}."
        )


# ============================================================
# Clean cell-line identifiers
# ============================================================

def clean_cell_line_names(
    data: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Standardise cell-line identifiers for matching."""

    output = data.copy()

    output[column] = (
        output[column]
        .astype("string")
        .str.strip()
    )

    return output


# ============================================================
# Match response cell lines
# ============================================================

def match_cell_lines(
    response_data: pd.DataFrame,
    gene_expression: pd.DataFrame,
    cell_line_column: str,
) -> pd.DataFrame:
    """
    Keep response observations whose cell lines are present
    in the gene-expression dataset.
    """

    response_data = clean_cell_line_names(
        response_data,
        cell_line_column,
    )

    gene_expression = clean_cell_line_names(
        gene_expression,
        cell_line_column,
    )

    gene_expression_cell_lines = set(
        gene_expression[cell_line_column]
        .dropna()
        .unique()
    )

    matched = response_data[
        response_data[cell_line_column].isin(
            gene_expression_cell_lines
        )
    ].copy()

    return matched


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--response",
        required=True,
        help="Cleaned response dataset.",
    )

    parser.add_argument(
        "--gene-expression",
        required=True,
        help="Gene-expression dataset.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the matched response dataset.",
    )

    parser.add_argument(
        "--cell-line-column",
        default=DEFAULT_CELL_LINE_COLUMN,
        help=(
            "Column containing the cell-line identifier. "
            f"Default: {DEFAULT_CELL_LINE_COLUMN}."
        ),
    )

    args = parser.parse_args()

    response_path = Path(
        args.response
    )

    gene_expression_path = Path(
        args.gene_expression
    )

    output_path = Path(
        args.output
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    response_data = load_csv(
        response_path
    )

    gene_expression = load_csv(
        gene_expression_path
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    check_column(
        response_data,
        args.cell_line_column,
        response_path,
    )

    check_column(
        gene_expression,
        args.cell_line_column,
        gene_expression_path,
    )

    # --------------------------------------------------------
    # Count before matching
    # --------------------------------------------------------

    response_cell_lines_before = (
        response_data[args.cell_line_column]
        .astype("string")
        .str.strip()
        .nunique()
    )

    gene_expression_cell_lines = (
        gene_expression[args.cell_line_column]
        .astype("string")
        .str.strip()
        .nunique()
    )

    response_rows_before = len(
        response_data
    )

    # --------------------------------------------------------
    # Match cell lines
    # --------------------------------------------------------

    matched = match_cell_lines(
        response_data,
        gene_expression,
        args.cell_line_column,
    )

    # --------------------------------------------------------
    # Count after matching
    # --------------------------------------------------------

    response_rows_after = len(
        matched
    )

    response_cell_lines_after = (
        matched[args.cell_line_column]
        .nunique()
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save matched response data
    # --------------------------------------------------------

    matched.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        "CELL-LINE / GENE-EXPRESSION MATCHING COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        f"Response rows before matching: "
        f"{response_rows_before:,}"
    )

    print(
        f"Response rows after matching: "
        f"{response_rows_after:,}"
    )

    print(
        f"Response cell lines before matching: "
        f"{response_cell_lines_before:,}"
    )

    print(
        f"Gene-expression cell lines: "
        f"{gene_expression_cell_lines:,}"
    )

    print(
        f"Matched response cell lines: "
        f"{response_cell_lines_after:,}"
    )

    print(
        f"Output:\n{output_path.resolve()}"
    )


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":
    main()