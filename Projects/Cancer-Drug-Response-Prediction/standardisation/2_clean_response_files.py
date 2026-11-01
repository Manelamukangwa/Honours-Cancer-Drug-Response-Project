"""
Script: Clean Response Data

Purpose
-------
Clean AUC-filtered response data.

This script:

1. Checks the required columns.
2. Keeps only the required columns.
3. Removes observations with missing response values.
4. Removes exact duplicate rows.

Repeated cell-line/drug observations with different response values
are retained.

This script is run after AUC filtering.

Identifiers
-----------
Cell line      : cell_line_name
Cellosaurus ID : cellosaurus_id
Drug           : pubchem_id
Response       : LN_IC50_curvecurator
"""

# ============================================================
# Import libraries
# ============================================================
from pathlib import Path
import argparse
import pandas as pd
# ============================================================
# Configuration
# ============================================================
cell_line_column = "cell_line_name"
cellosaurus_column = "cellosaurus_id"
drug_column = "pubchem_id"
response_column = "LN_IC50_curvecurator"

required_columns = [
    cell_line_column,
    cellosaurus_column,
    drug_column,
    response_column,
]

# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser(
        description="Clean AUC-filtered response data."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the AUC-filtered input CSV file."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the cleaned output CSV file."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(f"Loading: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = pd.read_csv(
        input_path,
        low_memory=False
    )

    print(f"Original observations: {len(data):,}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # --------------------------------------------------------
    # Keep required columns
    # --------------------------------------------------------

    data = data[required_columns].copy()

    # --------------------------------------------------------
    # Remove missing response values
    # --------------------------------------------------------

    missing_response = (
        data[response_column]
        .isna()
        .sum()
    )

    data = data.dropna(
        subset=[response_column]
    )

    print(f"Missing response observations removed: {missing_response:,}")

    # --------------------------------------------------------
    # Remove exact duplicate rows
    # --------------------------------------------------------

    duplicate_count = (
        data.duplicated()
        .sum()
    )

    data = data.drop_duplicates()

    print(f"Exact duplicate observations removed: {duplicate_count:,}")

    print(f"Final observations: {len(data):,}")

    # --------------------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()