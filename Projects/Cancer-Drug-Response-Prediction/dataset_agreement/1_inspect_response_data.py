
"""
Inspect a drug-response dataset.

This script provides a basic overview of a CSV or Excel file, including:
- Number of rows and columns
- Column names and data types
- First five rows
- Missing values per column
- Number of duplicate rows
- Potential cell-line and drug identifier columns
- Number of unique values and example values in potential identifier columns

The script is dataset-agnostic and accepts the input file path through
the command line.

Usage:
    python inspect_response_data.py path/to/response_file.csv
    python inspect_response_data.py path/to/response_file.xlsx

Dependencies:
    pandas
    openpyxl (for Excel files)
"""

import argparse
from pathlib import Path

import pandas as pd


def inspect_file(file_path):
    """Load and inspect a CSV or Excel response dataset."""

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the dataset
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            "Unsupported file format. Please provide a CSV, XLSX, or XLS file."
        )

    print("\n" + "=" * 60)
    print(f"FILE: {file_path.name}")
    print("=" * 60)

    # Dataset structure
    print(f"\nNumber of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn names and data types:")
    print(df.dtypes.to_string())

    # Preview
    print("\nFirst five rows:")
    print(df.head().to_string())

    # Missing values
    print("\nMissing values per column:")
    missing = df.isna().sum()
    if missing.any():
        print(missing[missing > 0].to_string())
    else:
        print("No missing values.")

    # Duplicate rows
    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

    # Potential identifier columns
    keywords = [
        "cell",
        "drug",
        "compound",
        "cosmic",
        "sanger",
        "sample",
        "name",
        "id",
    ]

    identifier_columns = [
        column
        for column in df.columns
        if any(keyword in str(column).lower() for keyword in keywords)
    ]

    print("\nPotential identifier columns:")

    if not identifier_columns:
        print("No potential identifier columns detected.")
        return

    for column in identifier_columns:
        unique_count = df[column].nunique(dropna=True)
        examples = df[column].dropna().astype(str).unique()[:10]

        print(f"\nColumn: {column}")
        print(f"Unique non-missing values: {unique_count}")
        print(f"Example values: {list(examples)}")


def main():
    """Parse command-line arguments and inspect the input file."""

    parser = argparse.ArgumentParser(
        description="Inspect a drug-response dataset."
    )

    parser.add_argument(
        "file",
        type=str,
        help="Path to a CSV or Excel response file.",
    )

    args = parser.parse_args()
    inspect_file(args.file)


if __name__ == "__main__":
    main()
