"""
Script: AUC Filtering

Purpose
-------
Remove individual drug-cell-line observations using an AUC
threshold (0.85 - 1.0).

Drug-cell-line observations with AUC values between the specified
minimum and maximum values (inclusive) are removed.

The filtering is applied to individual observations.
Other observations belonging to the same drug are retained.

This script is applied to the raw response files before the
cleaning script.
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
auc_column = "AUC_curvecurator"

# Only the AUC column is required at this stage.
required_columns = [
    auc_column,
]

# Default AUC thresholds
AUC_MIN = 0.85
AUC_MAX = 1.00


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser(
        description=(
            "Filter individual observations using an AUC threshold."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw input CSV file."
    )

    parser.add_argument(
        "--auc-min",
        type=float,
        default=AUC_MIN,
        help="Minimum AUC threshold (inclusive)."
    )

    parser.add_argument(
        "--auc-max",
        type=float,
        default=AUC_MAX,
        help="Maximum AUC threshold (inclusive)."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the AUC-filtered output CSV file."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(f"Loading: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    data = pd.read_csv(
        input_path,
        low_memory=False
    )

    print(
        f"Original observations: "
        f"{len(data):,}"
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Convert AUC to numeric
    # --------------------------------------------------------

    data[auc_column] = pd.to_numeric(
        data[auc_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove individual observations
    # --------------------------------------------------------

    remove_mask = (
        (data[auc_column] >= args.auc_min)
        &
        (data[auc_column] <= args.auc_max)
    )

    filtered_data = data.loc[
        ~remove_mask
    ].copy()

    print(
        f"AUC range removed: "
        f"{args.auc_min} - {args.auc_max}"
    )

    print(
        f"Observations removed: "
        f"{remove_mask.sum():,}"
    )

    print(
        f"Observations remaining: "
        f"{len(filtered_data):,}"
    )

    # --------------------------------------------------------
    # Save filtered dataset
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    filtered_data.to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}") 
    
if __name__ == "__main__":
    main()