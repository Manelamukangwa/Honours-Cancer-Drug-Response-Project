"""
Script: Check High-AUC Drug and Cell-Line Distribution

Purpose
-------
Check whether observations with AUC values between 0.85 and 1.00
are concentrated in particular drugs or cell lines.

The dataset is supplied through the terminal.
"""

from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Check drug and cell-line distribution of high-AUC observations."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file."
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = pd.read_csv(input_path, low_memory=False)

    required_columns = [
        "AUC_curvecurator",
        "drug_name",
        "cell_line_name"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data["AUC_curvecurator"] = pd.to_numeric(
        data["AUC_curvecurator"],
        errors="coerce"
    )

    high_auc = data[
        (data["AUC_curvecurator"] >= 0.85) &
        (data["AUC_curvecurator"] <= 1.00)
    ]

    remaining = data[
        data["AUC_curvecurator"] < 0.85
    ]

    print("\nHigh-AUC observations")
    print("---------------------")
    print(f"Total: {len(high_auc):,}")

    print("\nTop 10 drugs in high-AUC observations")
    print("--------------------------------------")
    print(high_auc["drug_name"].value_counts().head(10))

    print("\nTop 10 cell lines in high-AUC observations")
    print("-------------------------------------------")
    print(high_auc["cell_line_name"].value_counts().head(10))

    print("\nUnique drugs")
    print("------------")
    print(f"High AUC: {high_auc['drug_name'].nunique():,}")
    print(f"Remaining: {remaining['drug_name'].nunique():,}")

    print("\nUnique cell lines")
    print("-----------------")
    print(f"High AUC: {high_auc['cell_line_name'].nunique():,}")
    print(f"Remaining: {remaining['cell_line_name'].nunique():,}")


if __name__ == "__main__":
    main()