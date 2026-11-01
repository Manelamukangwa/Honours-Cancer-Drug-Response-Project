"""
Script: Check High-AUC Observations

Purpose
-------
Check whether observations with AUC values between 0.85 and 1.00
have valid LN_IC50_curvecurator values.

The dataset is supplied through the terminal.
"""

from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Check LN_IC50_curvecurator values for observations with high AUC."
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

    print(f"Loading: {input_path}")

    data = pd.read_csv(input_path, low_memory=False)

    required_columns = [
        "AUC_curvecurator",
        "LN_IC50_curvecurator"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data["AUC_curvecurator"] = pd.to_numeric(
        data["AUC_curvecurator"],
        errors="coerce"
    )

    data["LN_IC50_curvecurator"] = pd.to_numeric(
        data["LN_IC50_curvecurator"],
        errors="coerce"
    )

    high_auc = data[
        (data["AUC_curvecurator"] >= 0.85) &
        (data["AUC_curvecurator"] <= 1.00)
    ]

    remaining = data[
        data["AUC_curvecurator"] < 0.85
    ]

    valid_remaining = remaining["LN_IC50_curvecurator"].dropna()

    print()
    print("LN_IC50_curvecurator for remaining observations")
    print("-----------------------------------------------")
    print(f"Observations: {len(remaining):,}")
    print(f"Valid LN_IC50_curvecurator: {len(valid_remaining):,}")
    print(
        f"Missing LN_IC50_curvecurator: "
        f"{remaining['LN_IC50_curvecurator'].isna().sum():,}"
    )
    print(f"Minimum: {valid_remaining.min()}")
    print(f"Maximum: {valid_remaining.max()}")
    print(f"Median:  {valid_remaining.median()}")
    print(f"Mean:    {valid_remaining.mean()}")
    print(f"SD:      {valid_remaining.std()}")

    valid_ln_ic50 = high_auc["LN_IC50_curvecurator"].dropna()

    print()
    print("Results")
    print("-------")
    print(f"Total observations: {len(data):,}")
    print(f"AUC 0.85–1.00 observations: {len(high_auc):,}")
    print(f"Valid LN_IC50_curvecurator: {len(valid_ln_ic50):,}")
    print(
        f"Missing LN_IC50_curvecurator: "
        f"{high_auc['LN_IC50_curvecurator'].isna().sum():,}"
    )

    print(
        f"Valid LN_IC50_curvecurator: "
        f"{high_auc['LN_IC50_curvecurator'].notna().sum():,}"
    )

    print()
    print("LN_IC50_curvecurator for AUC 0.85–1.00")
    print("-----------------------------------------")
    print(f"Minimum: {valid_ln_ic50.min()}")
    print(f"Maximum: {valid_ln_ic50.max()}")
    print(f"Median:  {valid_ln_ic50.median()}")
    print(f"Mean:    {valid_ln_ic50.mean()}")
    print(f"SD:      {valid_ln_ic50.std()}")

    print(
        f"Missing LN_IC50_curvecurator: "
        f"{high_auc['LN_IC50_curvecurator'].isna().sum():,}"
    )

if __name__ == "__main__":
    main()