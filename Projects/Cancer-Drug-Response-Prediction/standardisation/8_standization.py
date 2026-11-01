"""
Script: LCO Response Standardisation

Purpose
-------
Standardise the response variable for each LCO split without
data leakage.

For each LCO split:

    1. Fit StandardScaler using TRAINING data only from the
       within-study dataset.
    2. Apply the fitted scaler to the within-study validation data.
    3. Apply the fitted scaler to the within-study test data.
    4. Apply the same fitted scaler to the cross-study datasets
       created specifically for that LCO split.

The scaler is NEVER fitted using validation, test, or cross-study
data.

The script is fully dataset-agnostic.

The within-study dataset is defined by --split-dir.

Cross-study datasets must be organised by LCO split:

    cross-study-directory/
        split_1/
            cross-study_unseen_dataset1.csv 
             cross-study_unseen_dataset2.csv 

        split_2/
            cross-study_unseen_dataset1.csv
             cross-study_unseen_dataset1.csv
        ... 

        split_5/
            cross-study_unseen_dataset1.csv
            cross-study_unseen_dataset1.csv

For each LCO split, only the corresponding split-specific
cross-study datasets are standardised.

Output
------
For each split, the script saves:

    train_standardised.csv
    validation_standardised.csv
    test_standardised.csv
    <cross-study-name>_standardised.csv
    scaler.pkl

A summary file is also saved:

    standardisation_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler


DEFAULT_N_SPLITS = 5

DEFAULT_RESPONSE_COLUMN = (
    "LN_IC50_curvecurator"
)

DEFAULT_STANDARDISED_COLUMN = (
    "LN_IC50_standardised"
)


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


def validate_response_column(
    data: pd.DataFrame,
    response_column: str,
    dataset_label: str,
) -> None:
    """
    Check that the response column exists and
    contains valid numeric values.
    """

    if response_column not in data.columns:
        raise ValueError(
            f"{dataset_label} is missing required "
            f"column {response_column!r}."
        )

    response = pd.to_numeric(
        data[response_column],
        errors="coerce",
    )

    if response.isna().any():
        raise ValueError(
            f"{dataset_label} contains missing or "
            f"non-numeric values in "
            f"{response_column!r}."
        )


def fit_scaler(
    training_data: pd.DataFrame,
    response_column: str,
) -> StandardScaler:
    """
    Fit StandardScaler using training data only.
    """

    scaler = StandardScaler()

    scaler.fit(
        training_data[
            [response_column]
        ]
    )

    return scaler


def apply_scaler(
    data: pd.DataFrame,
    scaler: StandardScaler,
    response_column: str,
    standardised_column: str,
) -> pd.DataFrame:
    """
    Apply an already-fitted scaler to a dataset.
    """

    output = data.copy()

    output[
        standardised_column
    ] = (
        scaler.transform(
            output[
                [response_column]
            ]
        ).ravel()
    )

    return output


def get_dataset_name(
    file_path: Path,
) -> str:
    """
    Get a dataset name from its filename.

    Examples
    --------
    CTRPv1_unseen.csv
        -> CTRPv1

    CTRPv2_unseen.csv
        -> CTRPv2

    GDSC1_unseen.csv
        -> GDSC1

    CCLE_unseen.csv
        -> CCLE
    """

    dataset_name = file_path.stem

    if dataset_name.endswith(
        "_unseen"
    ):
        dataset_name = dataset_name[
            :-len("_unseen")
        ]

    if dataset_name.endswith(
        "_clean"
    ):
        dataset_name = dataset_name[
            :-len("_clean")
        ]

    return dataset_name


def process_split(
    split_index: int,
    split_directory: Path,
    cross_study_directory: Path,
    output_directory: Path,
    response_column: str,
    standardised_column: str,
) -> dict:
    """
    Process one LCO split.

    The scaler is fitted ONLY on the training
    data for this split.

    Only the cross-study datasets belonging to
    this LCO split are used.
    """

    print()
    print("==============================")
    print(
        f"LCO SPLIT {split_index}"
    )
    print("==============================")

    # --------------------------------------------------
    # INPUT FILES
    # --------------------------------------------------

    split_input_directory = (
        split_directory
        / f"split_{split_index}"
    )

    train_file = (
        split_input_directory
        / "train.csv"
    )

    validation_file = (
        split_input_directory
        / "validation.csv"
    )

    test_file = (
        split_input_directory
        / "test.csv"
    )

    # --------------------------------------------------
    # LOAD WITHIN-STUDY DATA
    # --------------------------------------------------

    train = load_csv(
        train_file
    )

    validation = load_csv(
        validation_file
    )

    test = load_csv(
        test_file
    )

    # --------------------------------------------------
    # LOAD SPLIT-SPECIFIC CROSS-STUDY DATA
    # --------------------------------------------------

    split_cross_study_directory = (
        cross_study_directory
        / f"split_{split_index}"
    )

    if not split_cross_study_directory.exists():
        raise FileNotFoundError(
            "Cross-study directory not found for "
            f"split {split_index}:\n"
            f"{split_cross_study_directory}"
        )

    cross_study_files = sorted(
        split_cross_study_directory.glob(
            "*_unseen.csv"
        )
    )

    if not cross_study_files:
        raise ValueError(
            "No unseen cross-study files found for "
            f"split {split_index}:\n"
            f"{split_cross_study_directory}"
        )

    cross_study_data = {}

    for cross_study_file in cross_study_files:

        dataset_name = get_dataset_name(
            cross_study_file
        )

        if dataset_name in cross_study_data:
            raise ValueError(
                f"Duplicate cross-study dataset "
                f"name for split {split_index}: "
                f"{dataset_name}"
            )

        cross_study_data[
            dataset_name
        ] = load_csv(
            cross_study_file
        )

        print(
            f"Loaded {dataset_name}: "
            f"{len(cross_study_data[dataset_name]):,} "
            f"observations"
        )

    # --------------------------------------------------
    # VALIDATE WITHIN-STUDY DATA
    # --------------------------------------------------

    validate_response_column(
        train,
        response_column,
        f"Split {split_index} training data",
    )

    validate_response_column(
        validation,
        response_column,
        f"Split {split_index} validation data",
    )

    validate_response_column(
        test,
        response_column,
        f"Split {split_index} test data",
    )

    # --------------------------------------------------
    # VALIDATE CROSS-STUDY DATA
    # --------------------------------------------------

    for (
        dataset_name,
        data,
    ) in cross_study_data.items():

        validate_response_column(
            data,
            response_column,
            f"{dataset_name} cross-study data",
        )

    # --------------------------------------------------
    # FIT SCALER
    #
    # IMPORTANT:
    # ONLY TRAINING DATA IS USED HERE.
    # --------------------------------------------------

    scaler = fit_scaler(
        training_data=train,
        response_column=response_column,
    )

    # --------------------------------------------------
    # APPLY SCALER TO WITHIN-STUDY DATA
    # --------------------------------------------------

    train_standardised = apply_scaler(
        data=train,
        scaler=scaler,
        response_column=response_column,
        standardised_column=standardised_column,
    )

    validation_standardised = apply_scaler(
        data=validation,
        scaler=scaler,
        response_column=response_column,
        standardised_column=standardised_column,
    )

    test_standardised = apply_scaler(
        data=test,
        scaler=scaler,
        response_column=response_column,
        standardised_column=standardised_column,
    )

    # --------------------------------------------------
    # APPLY SAME SPLIT-SPECIFIC SCALER TO CROSS-STUDY DATA
    # --------------------------------------------------

    cross_study_standardised = {}

    for (
        dataset_name,
        data,
    ) in cross_study_data.items():

        cross_study_standardised[
            dataset_name
        ] = apply_scaler(
            data=data,
            scaler=scaler,
            response_column=response_column,
            standardised_column=standardised_column,
        )

    # --------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------

    split_output_directory = (
        output_directory
        / f"split_{split_index}"
    )

    split_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # SAVE WITHIN-STUDY DATA
    # --------------------------------------------------

    train_standardised.to_csv(
        split_output_directory
        / "train_standardised.csv",
        index=False,
    )

    validation_standardised.to_csv(
        split_output_directory
        / "validation_standardised.csv",
        index=False,
    )

    test_standardised.to_csv(
        split_output_directory
        / "test_standardised.csv",
        index=False,
    )

    # --------------------------------------------------
    # SAVE CROSS-STUDY DATA
    # --------------------------------------------------

    for (
        dataset_name,
        data,
    ) in cross_study_standardised.items():

        output_file = (
            split_output_directory
            / f"{dataset_name}_standardised.csv"
        )

        data.to_csv(
            output_file,
            index=False,
        )

    # --------------------------------------------------
    # SAVE SPLIT-SPECIFIC SCALER
    # --------------------------------------------------

    joblib.dump(
        scaler,
        split_output_directory
        / "scaler.pkl",
    )

    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------

    print(
        f"Training rows: "
        f"{len(train_standardised):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_standardised):,}"
    )

    print(
        f"Test rows: "
        f"{len(test_standardised):,}"
    )

    for (
        dataset_name,
        data,
    ) in cross_study_standardised.items():

        print(
            f"{dataset_name} rows: "
            f"{len(data):,}"
        )

    print(
        "Scaler fitted using training data only."
    )

    print(
        f"Saved to:\n"
        f"{split_output_directory}"
    )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    summary = {
        "Split": split_index,
        "Training_rows": len(
            train_standardised
        ),
        "Validation_rows": len(
            validation_standardised
        ),
        "Test_rows": len(
            test_standardised
        ),
    }

    for (
        dataset_name,
        data,
    ) in cross_study_standardised.items():

        summary[
            f"{dataset_name}_rows"
        ] = len(data)

    return summary


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # --------------------------------------------------
    # WITHIN-STUDY LCO SPLITS
    # --------------------------------------------------

    parser.add_argument(
        "--split-dir",
        required=True,
        help=(
            "Directory containing the LCO split "
            "folders for the within-study dataset."
        ),
    )

    # --------------------------------------------------
    # CROSS-STUDY DATASETS
    # --------------------------------------------------

    parser.add_argument(
        "--cross-study-dir",
        required=True,
        help=(
            "Directory containing the split-specific "
            "unseen cross-study datasets."
        ),
    )

    # --------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory for standardised outputs."
        ),
    )

    # --------------------------------------------------
    # RESPONSE COLUMN
    # --------------------------------------------------

    parser.add_argument(
        "--response-column",
        default=DEFAULT_RESPONSE_COLUMN,
        help=(
            "Response column to standardise."
        ),
    )

    # --------------------------------------------------
    # STANDARDISED COLUMN
    # --------------------------------------------------

    parser.add_argument(
        "--standardised-column",
        default=DEFAULT_STANDARDISED_COLUMN,
        help=(
            "Name of the standardised response column."
        ),
    )

    # --------------------------------------------------
    # NUMBER OF SPLITS
    # --------------------------------------------------

    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=(
            "Number of LCO splits. "
            "Default: 5."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # PATHS
    # --------------------------------------------------

    split_directory = Path(
        args.split_dir
    )

    cross_study_directory = Path(args.cross_study_dir)

    output_directory = Path(
        args.output_dir
    )

    # --------------------------------------------------
    # CHECK CROSS-STUDY DIRECTORY
    # --------------------------------------------------

    if not cross_study_directory.exists():
        raise FileNotFoundError(
            "Cross-study directory not found:\n"
            f"{cross_study_directory}"
        )

    # --------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # PROCESS ALL LCO SPLITS
    # --------------------------------------------------

    summaries = []

    for split_index in range(
        1,
        args.n_splits + 1,
    ):

        summary = process_split(
            split_index=split_index,
            split_directory=split_directory,
            cross_study_directory=cross_study_directory,
            output_directory=output_directory,
            response_column=args.response_column,
            standardised_column=args.standardised_column,
        )

        summaries.append(
            summary
        )

    # --------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------

    pd.DataFrame(
        summaries
    ).to_csv(
        output_directory
        / "standardisation_summary.csv",
        index=False,
    )

    # --------------------------------------------------
    # COMPLETE
    # --------------------------------------------------

    print()
    print("==============================")
    print("RESPONSE STANDARDISATION COMPLETE")
    print("==============================")

    print(
        f"Outputs saved to:\n"
        f"{output_directory.resolve()}"
    )

if __name__ == "__main__":
    main()
