"""
LCO Response Standardisation Script
-----------------------------------
Purpose
-------
Standardise the response variable for each LCO split without data leakage.

For each LCO split:

    1. Fit the StandardScaler using TRAINING data only.
    2. Apply the fitted scaler to the validation data.
    3. Apply the fitted scaler to the test data.
    4. Apply the same fitted scaler to the supplied cross-study dataset.

The scaler is therefore never fitted using validation, test, or cross-study evaluation data.

The script is dataset-independent. It is not hardcoded to GDSC2, CTRPv2, or any particular cross-study dataset.
provide e.g. --cross-study-file data/processed/CTRPv2_response_unseen_in_GDSC2.csv when running script

python standardise_lco.py \
    --split-dir data/processed/lco_splits_gdsc2 \
    --cross-study-file data/processed/CTRPv2_response_unseen_in_GDSC2.csv \
    --output-dir data/processed/lco_splits_gdsc2_standardised \
    --response-column LN_IC50_curvecurator \
    --standardised-column LN_IC50_standardised \
    --n-splits 5

Inputs
------
LCO split directory containing:

    cv_split_0_train.csv
    cv_split_0_validation.csv
    cv_split_0_test.csv
    ...
    cv_split_4_train.csv
    cv_split_4_validation.csv
    cv_split_4_test.csv

Cross-study evaluation response file supplied when running the script.

Outputs
-------
For each LCO split:

    train_standardised.csv
    validation_standardised.csv
    test_standardised.csv
    crossstudy_standardised.csv
    scaler.pkl

A summary file containing the training mean and standard deviation
used for each split is also produced.
"""

# ============================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# SECTION 2: CONFIGURATION
# ============================================================

DEFAULT_N_SPLITS = 5

DEFAULT_RESPONSE_COLUMN = "LN_IC50_curvecurator"

# ============================================================
# SECTION 3: LOAD LCO SPLIT FILE
# ============================================================
def load_split_file(
    split_directory: Path,
    split_index: int,
    role: str
) -> pd.DataFrame:
    """
    Load one LCO split file.
    """
    input_path = (split_directory/ f"cv_split_{split_index}_{role}.csv")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {role} file for split {split_index}: "
            f"{input_path}"
        )

    df = pd.read_csv(input_path,low_memory=False)
    return df

# ============================================================
# SECTION 4: LOAD CROSS-STUDY DATA
# ============================================================
def load_cross_study_file(
    input_path: Path
) -> pd.DataFrame:
    """
    Load the supplied cross-study evaluation dataset.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cross-study evaluation file not found: "
            f"{input_path}"
        )

    df = pd.read_csv(input_path,low_memory=False)
    return df

# ============================================================
# SECTION 5: VALIDATE RESPONSE COLUMN
# ============================================================
def validate_response_column(
    df: pd.DataFrame,
    response_column: str,
    dataset_label: str
) -> None:
    """
    Check that the response column exists and contains valid values.
    """

    if response_column not in df.columns:
        raise ValueError(
            f"{dataset_label} does not contain the response "
            f"column '{response_column}'."
        )

    response = pd.to_numeric(df[response_column],errors="coerce")

    if response.isna().any():
        raise ValueError(
            f"{dataset_label} contains missing or non-numeric "
            f"values in '{response_column}'."
        )

# ============================================================
# SECTION 6: FIT STANDARDISER ON TRAINING DATA ONLY
# ============================================================
def fit_scaler(
    training_data: pd.DataFrame,
    response_column: str
) -> StandardScaler:
    """
    Fit the scaler using training data only.
    """

    scaler = StandardScaler()

    scaler.fit(
        training_data[
            [response_column]
        ]
    )
    return scaler

# ============================================================
# SECTION 7: APPLY FITTED STANDARDISER
# ============================================================
def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    response_column: str,
    standardised_column: str
) -> pd.DataFrame:
    """
    Apply an already-fitted scaler.
    The scaler is NOT fitted in this function.
    """
    output = df.copy()
    output[standardised_column] = (
        scaler.transform(
            output[
                [response_column]
            ]
        ).ravel()
    )
    return output

# ============================================================
# SECTION 8: PROCESS ONE LCO SPLIT
# ============================================================
def process_split(
    split_index: int,
    split_directory: Path,
    cross_study_data: pd.DataFrame,
    output_directory: Path,
    response_column: str,
    standardised_column: str
) -> dict:
    """
    Process one LCO split.
    The scaler is fitted only on the training data and then applied to validation, test, and the supplied cross-study
    dataset.
    """
    print(f"\n================ SPLIT {split_index} "f"================")

    # --------------------------------------------------------
    # Load train / validation / test
    # --------------------------------------------------------
    train = load_split_file(
        split_directory,
        split_index,
        "train"
    )

    validation = load_split_file(
        split_directory,
        split_index,
        "validation"
    )

    test = load_split_file(
        split_directory,
        split_index,
        "test"
    )

    # --------------------------------------------------------
    # Validate response columns
    # --------------------------------------------------------
    validate_response_column(
        train,
        response_column,
        f"Split {split_index} training data"
    )

    validate_response_column(
        validation,
        response_column,
        f"Split {split_index} validation data"
    )

    validate_response_column(
        test,
        response_column,
        f"Split {split_index} test data"
    )

    validate_response_column(
        cross_study_data,
        response_column,
        "Cross-study evaluation data"
    )

    # --------------------------------------------------------
    # Fit scaler ONLY on training data
    # --------------------------------------------------------
    scaler = fit_scaler(
        train,
        response_column
    )

    training_mean = float(scaler.mean_[0])

    training_std = float(scaler.scale_[0])

    print(
        f"Training mean used for standardisation: "
        f"{training_mean:.6f}"
    )

    print(
        f"Training SD used for standardisation: "
        f"{training_std:.6f}"
    )
    # --------------------------------------------------------
    # Apply same scaler to train / validation / test
    # --------------------------------------------------------
    train_standardised = apply_scaler(
        train,
        scaler,
        response_column,
        standardised_column
    )

    validation_standardised = apply_scaler(
        validation,
        scaler,
        response_column,
        standardised_column
    )

    test_standardised = apply_scaler(
        test,
        scaler,
        response_column,
        standardised_column
    )

    # --------------------------------------------------------
    # Apply same training-fitted scaler to cross-study data
    # --------------------------------------------------------
    cross_study_standardised = apply_scaler(
        cross_study_data,
        scaler,
        response_column,
        standardised_column
    )

    # --------------------------------------------------------
    # Create output directory for this split
    # --------------------------------------------------------
    split_output_directory = (output_directory/ f"split_{split_index}")

    split_output_directory.mkdir(parents=True,exist_ok=True)

    # --------------------------------------------------------
    # Save standardised train
    # --------------------------------------------------------
    train_standardised.to_csv(
        split_output_directory
        / "train_standardised.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save standardised validation
    # --------------------------------------------------------
    validation_standardised.to_csv(
        split_output_directory
        / "validation_standardised.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save standardised test
    # --------------------------------------------------------
    test_standardised.to_csv(
        split_output_directory
        / "test_standardised.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save standardised cross-study data
    # --------------------------------------------------------
    cross_study_standardised.to_csv(
        split_output_directory
        / "crossstudy_standardised.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save fitted scaler
    # --------------------------------------------------------
    joblib.dump(
        scaler,
        split_output_directory
        / "scaler.pkl"
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------
    print(f"Training rows: {len(train_standardised)}")

    print(f"Validation rows: {len(validation_standardised)}")

    print(f"Test rows: {len(test_standardised)}")

    print(
        f"Cross-study rows: "
        f"{len(cross_study_standardised)}"
    )

    print(
        f"Saved split {split_index} outputs to: "
        f"{split_output_directory}"
    )

    # --------------------------------------------------------
    # Return information for summary file
    # --------------------------------------------------------
    return {
        "Split": split_index,
        "Training_rows": len(train_standardised),
        "Validation_rows": len(validation_standardised),
        "Test_rows": len(test_standardised),
        "Crossstudy_rows": len(cross_study_standardised),
        "Training_mean": training_mean,
        "Training_SD": training_std
    }


# ============================================================
# SECTION 9: MAIN
# ============================================================
def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    # --------------------------------------------------------
    # LCO split directory
    # --------------------------------------------------------
    parser.add_argument(
        "--split-dir",
        required=True,
        help=(
            "Directory containing the LCO split files."
        )
    )

    # --------------------------------------------------------
    # Cross-study dataset
    # --------------------------------------------------------
    parser.add_argument(
        "--cross-study-file",
        required=True,
        help=(
            "Path to the cross-study evaluation response file. "
            "The supplied dataset is transformed using each "
            "LCO split's training-fitted scaler."
        )
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory where standardised split files "
            "will be written."
        )
    )

    # --------------------------------------------------------
    # Response column
    # --------------------------------------------------------
    parser.add_argument(
        "--response-column",
        default=DEFAULT_RESPONSE_COLUMN,
        help=(
            "Response column to standardise."
        )
    )

    # --------------------------------------------------------
    # Standardised response column
    # --------------------------------------------------------
    parser.add_argument(
        "--standardised-column",
        default="LN_IC50_standardised",
        help=(
            "Name of the new standardised response column."
        )
    )

    # --------------------------------------------------------
    # Number of LCO splits
    # --------------------------------------------------------
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=(
            "Number of LCO split sets. "
            "Default: 5."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Convert paths
    # --------------------------------------------------------
    split_directory = Path(args.split_dir)

    cross_study_file = Path(args.cross_study_file)

    output_directory = Path(args.output_dir)

    output_directory.mkdir(parents=True,exist_ok=True)

    # --------------------------------------------------------
    # Load cross-study evaluation data
    # --------------------------------------------------------
    print(
        f"Loading cross-study evaluation data from: "
        f"{cross_study_file}"
    )

    cross_study_data = load_cross_study_file(cross_study_file)

    print(
        f"Cross-study rows: "
        f"{len(cross_study_data)}"
    )

    # --------------------------------------------------------
    # Process each LCO split
    # --------------------------------------------------------
    summary_results = []

    for split_index in range(args.n_splits):
        split_summary = process_split(
            split_index=split_index,
            split_directory=split_directory,
            cross_study_data=cross_study_data,
            output_directory=output_directory,
            response_column=args.response_column,
            standardised_column=args.standardised_column
        )

        summary_results.append(split_summary)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------
    summary = pd.DataFrame(summary_results)

    summary.to_csv(output_directory/ "standardisation_summary.csv",index=False)

    print("\n================ STANDARDISATION COMPLETE ================\n")

    print(f"Processed {args.n_splits} LCO split sets.")

    print(
        f"Results saved to:\n"
        f"{output_directory.resolve()}"
    )

if __name__ == "__main__":
    main()