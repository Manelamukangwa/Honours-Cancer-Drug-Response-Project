"""
Check identifier consistency across pharmacogenomic response datasets.

Purpose
-------
Inspect cell-line and drug identifiers across CTRPv2, GDSC2, GDSC1,
CTRPv1, and CCLE to determine whether identifiers are missing and
how many unique identifiers are shared between datasets.

Identifiers
-----------
- cellosaurus_id: Cell-line identifier
- pubchem_id: Drug identifier

For each dataset, the script reports the number of rows, unique
identifiers, and missing identifiers. It then reports the number
of shared cell-line and drug identifiers for every pair of datasets.

Input
-----
Dataset paths supplied through the terminal in LABEL=FILE.csv format.

Output
------
Identifier statistics and pairwise identifier overlap printed
to the terminal.

Note
----
This script checks identifier overlap only. It does not match
cell-line–drug combinations or calculate response correlations.
"""

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


IDENTIFIERS = ["cellosaurus_id", "pubchem_id"]


def load_dataset(label, file_path):
    """Load a dataset and inspect its cell-line and drug identifiers."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(
        path,
        dtype={column: "string" for column in IDENTIFIERS},
        low_memory=False
    )

    missing_columns = set(IDENTIFIERS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{label} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Remove surrounding whitespace and treat empty strings as missing.
    for column in IDENTIFIERS:
        df[column] = df[column].str.strip().replace("", pd.NA)

    print("\n" + "=" * 60)
    print(f"Dataset: {label}")
    print(f"File: {path.name}")
    print(f"Rows: {len(df):,}")

    for column in IDENTIFIERS:
        values = df[column].dropna()

        print(f"\n{column}")
        print(f"  Unique identifiers: {values.nunique():,}")
        print(f"  Missing values: {df[column].isna().sum():,}")

    return df


def compare_identifiers(datasets):
    """Report shared cell-line and drug identifiers between datasets."""

    print("\n\n" + "=" * 60)
    print("PAIRWISE IDENTIFIER OVERLAP")
    print("=" * 60)

    for (name1, df1), (name2, df2) in combinations(
        datasets.items(), 2
    ):
        print(f"\n{name1} vs {name2}")

        for column in IDENTIFIERS:
            ids1 = set(df1[column].dropna())
            ids2 = set(df2[column].dropna())
            shared_ids = ids1.intersection(ids2)

            print(
                f"  {column}: {len(shared_ids):,} shared "
                f"identifiers "
                f"(dataset 1: {len(ids1):,}; "
                f"dataset 2: {len(ids2):,})"
            )


def main():
    """Parse dataset paths, inspect datasets, and compare identifiers."""

    parser = argparse.ArgumentParser(
        description=(
            "Check cellosaurus_id and pubchem_id consistency "
            "across pharmacogenomic datasets."
        )
    )

    parser.add_argument(
        "datasets",
        nargs="+",
        help='Dataset paths in LABEL=FILE.csv format, e.g. CTRPv2="/path/file.csv"'
    )

    args = parser.parse_args()
    datasets = {}

    for item in args.datasets:
        if "=" not in item:
            parser.error(
                f"Invalid dataset argument: {item}. "
                "Use LABEL=FILE.csv."
            )

        label, file_path = item.split("=", 1)

        if label in datasets:
            parser.error(f"Duplicate dataset label: {label}")

        datasets[label] = load_dataset(label, file_path)

    compare_identifiers(datasets)


if __name__ == "__main__":
    main()
