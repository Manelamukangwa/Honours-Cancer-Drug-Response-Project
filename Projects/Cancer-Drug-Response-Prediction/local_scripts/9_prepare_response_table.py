"""
Script 01: Prepare a clean drug-response table for LCO splitting.

Purpose
-------
Takes the raw standardised GDSC2 drug-response data, keeps only the columns needed,
checks the important IDs, removes rows where the response is missing,
and saves a clean standardised table ready for the LCO splitter.

Raw standardised GDSC2 response data → Script 01 → Clean response table → LCO splitter

This script does NOT:
1. Merge in any cell-line or drug feature view.
2. Drop combination-drug rows.
3. Apply curve-fit quality filtering (R2, SignalQuality, etc.).
4. Rename columns to the splitter's schema.
5. Perform any splitting.

Kept columns:
    cellosaurus_id
    cell_line_name
    pubchem_id
    LN_IC50_standardised
    tissue
"""

# ============================================================
# Section A — Import Libraries
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ============================================================
# Section B — Folder and File Configuration
# ============================================================

# Root directory for the project data
Data_root = (
    "/Users/manellamukangwa/"
    "Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction/data"
)

# Standardised GDSC2 response dataset
Raw_response_file = f"{Data_root}/processed/GDSC2_standardised.csv"

# Cleaned response dataset
Clean_response_file = (
    f"{Data_root}/processed/GDSC2_response_standardised_clean.csv"
)

# Column identifiers
Cell_line_identifier = "cell_line_name"
Drug_identifier = "pubchem_id"
Response_column = "LN_IC50_standardised"


# ============================================================
# Section C — Define Columns Needed
# ============================================================

# Columns that will be retained from the raw response file
keep_columns = [
    "cellosaurus_id",
    Cell_line_identifier,
    Drug_identifier,
    Response_column,
    "tissue",
]

# Columns containing identifiers for cells and drugs
id_columns = [
    "cellosaurus_id",
    Cell_line_identifier,
    Drug_identifier,
]


# ============================================================
# Section D — Load and Select the Data
# ============================================================

def load_and_select(input_path: Path) -> pd.DataFrame:
    """
    Load the standardised GDSC2 response dataset and select only
    the columns needed.
    """

    response_data = pd.read_csv(
        input_path,
        usecols=keep_columns,
        dtype={
            "cellosaurus_id": str,
            Cell_line_identifier: str,
            Drug_identifier: str,
            "tissue": str,
        },
        low_memory=False,
    )

    return response_data


# ============================================================
# Section E — Check for Missing IDs
# ============================================================

def report_missing_ids(df: pd.DataFrame) -> None:
    """
    Check for missing cell and drug identifiers.

    Reports missing values and raises an error if any are found.
    """

    for column in id_columns:

        n_missing_values = (
            df[column].isna().sum()
            + (df[column].astype(str).str.strip() == "").sum()
        )

        print(
            f" {column}: {n_missing_values} missing values"
        )

    total_missing_values = sum(
        df[column].isna().sum()
        + (df[column].astype(str).str.strip() == "").sum()
        for column in id_columns
    )

    if total_missing_values > 0:
        raise ValueError(
            f"Found {total_missing_values} missing ID values "
            f"across {id_columns}. "
            "Resolve before processing; the splitter will reject "
            "these rows."
        )


# ============================================================
# Section F — Remove Rows with Missing Response Values
# ============================================================

def drop_missing_response(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with no response value and report how many were dropped.
    """

    n_before_dropped = len(df)

    cleaned_data = (
        df.loc[df[Response_column].notna()]
        .reset_index(drop=True)
    )

    n_after_dropped = len(cleaned_data)

    n_rows_removed = (
        n_before_dropped - n_after_dropped
    )

    percentage_missing_response = (
        100 * n_rows_removed / n_before_dropped
        if n_before_dropped
        else 0.0
    )

    print(f" Rows before: {n_before_dropped}")
    print(f" Rows after: {n_after_dropped}")

    print(
        f" Dropped (missing {Response_column}): "
        f"{n_rows_removed} "
        f"({percentage_missing_response:.1f}%)"
    )

    return cleaned_data


# ============================================================
# Section G — Summarise the Cleaned Data
# ============================================================

def summarize_cleaned_data(df: pd.DataFrame) -> None:
    """
    Print basic summary statistics for a sanity check.
    """

    # Number of unique cell lines
    print(
        f" Unique cell lines: "
        f"{df[Cell_line_identifier].nunique()}"
    )

    # Number of unique drugs
    print(
        f" Unique drugs ({Drug_identifier}): "
        f"{df[Drug_identifier].nunique()}"
    )

    # Identify combination-drug rows
    n_drug_combo = (
        ~df[Drug_identifier]
        .str.match(r"^\d+$")
    ).sum()

    print(
        " Rows with non-numeric "
        f"{Drug_identifier} "
        f"(combination drugs): {n_drug_combo}"
    )

    # Number of tissue types
    print(
        f" Tissue types: "
        f"{df['tissue'].nunique()}"
    )


# ============================================================
# Section H — Set Up How the Script Will Be Run
# ============================================================

def main() -> None:
    """
    Main function controlling the data-cleaning workflow.
    """

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # Input file
    parser.add_argument(
        "--input",
        default=Raw_response_file,
        help=(
            "Path to the standardised GDSC2 response dataset "
            f"(default: {Raw_response_file})"
        ),
    )

    # Output file
    parser.add_argument(
        "--output",
        default=Clean_response_file,
        help=(
            "Path to write the clean response table "
            f"(default: {Clean_response_file})"
        ),
    )

    args = parser.parse_args()


    # ========================================================
    # Section I — Locate Input and Output Files
    # ========================================================

    input_path = Path(args.input)
    output_path = Path(args.output)


    # ========================================================
    # Section J — Run the Data-Cleaning Steps
    # ========================================================

    print(
        f"Loading and selecting columns from "
        f"{input_path} ..."
    )

    df = load_and_select(input_path)


    # --------------------------------------------------------
    # Check ID columns
    # --------------------------------------------------------

    print(
        "\nChecking ID columns for missing values..."
    )

    report_missing_ids(df)


    # --------------------------------------------------------
    # Remove missing response values
    # --------------------------------------------------------

    print(
        f"\nDropping rows with missing "
        f"{Response_column} ..."
    )

    df = drop_missing_response(df)


    # --------------------------------------------------------
    # Summarise cleaned data
    # --------------------------------------------------------

    print(
        "\nSummary of cleaned response table:"
    )

    summarize_cleaned_data(df)


    # ========================================================
    # Section K — Save Cleaned Data
    # ========================================================

    # Make sure the output folder exists
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save cleaned data
    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nWrote clean response table to "
        f"{output_path.resolve()}"
    )


# ============================================================
# Section L — Start the Script
# ============================================================

if __name__ == "__main__":
    main()