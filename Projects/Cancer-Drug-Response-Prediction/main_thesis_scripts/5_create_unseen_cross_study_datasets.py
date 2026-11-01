"""
Script: Create Unseen Cross-Study Datasets

Purpose
-------
Create cross-study datasets containing only cell lines that are
unseen by the within-study training set for each individual LCO split.

For each LCO split:
    1. Load the within-study training data.
    2. Identify the cell lines present in that training set.
    3. Remove those cell lines from each cross-study dataset.
    4. Keep cross-study observations whose cell_line_name does not
       occur in that split's within-study training set.

This process is repeated independently for all 5 LCO splits.

Important
---------
Filtering is based on cell_line_name.

A cell line is considered "unseen" if it does not occur in the
within-study TRAINING set for the current LCO split.

Therefore, a cell line that occurs only in the within-study validation
or test set is still retained in the cross-study dataset.

This is intentionally less conservative than defining unseen
cell lines using the entire within-study dataset.

The original input datasets are never modified.

Preprocessing
-------------
This version is designed for the current experiment:

- No AUC filtering.
- No response standardisation.
- Raw LN_IC50_curvecurator values are retained.
"""

# ============================================================
# Import Libraries
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

CELL_LINE_COLUMN = "cell_line_name"
TRAIN_FILENAME = "train.csv"


# ============================================================
# Helper functions
# ============================================================

def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file."""

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def get_training_cell_lines(
    training_data: pd.DataFrame,
) -> set[str]:
    """
    Get all cell lines present in the within-study training data.
    """

    if CELL_LINE_COLUMN not in training_data.columns:
        raise ValueError(
            f"Training dataset is missing required column: "
            f"{CELL_LINE_COLUMN!r}"
        )

    cell_lines = (
        training_data[CELL_LINE_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return set(cell_lines)


def create_unseen_dataset(
    cross_study_data: pd.DataFrame,
    training_cell_lines: set[str],
) -> tuple[pd.DataFrame, int]:
    """
    Remove cross-study observations whose cell lines occur
    in the within-study training set for the current LCO split.

    Returns
    -------
    unseen_data:
        Cross-study observations containing only cell lines
        unseen in the current within-study training set.

    removed_count:
        Number of observations removed.
    """

    if CELL_LINE_COLUMN not in cross_study_data.columns:
        raise ValueError(
            f"Cross-study dataset is missing required column: "
            f"{CELL_LINE_COLUMN!r}"
        )

    cell_lines = (
        cross_study_data[CELL_LINE_COLUMN]
        .astype(str)
        .str.strip()
    )

    seen_in_training = cell_lines.isin(
        training_cell_lines
    )

    unseen_data = cross_study_data.loc[
        ~seen_in_training
    ].copy()

    removed_count = int(
        seen_in_training.sum()
    )

    return unseen_data, removed_count


def get_dataset_name(path: Path) -> str:
    """
    Convert a dataset filename into its dataset name.

    Examples
    --------
    GDSC1.csv
        -> GDSC1

    CTRPv1.csv
        -> CTRPv1
    """

    return path.stem


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--split-dir",
        required=True,
        help=(
            "Directory containing the LCO split directories."
        ),
    )

    parser.add_argument(
        "--cross-study",
        nargs="+",
        required=True,
        help=(
            "Paths to one or more cross-study response datasets."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory where split-specific unseen "
            "cross-study datasets will be saved."
        ),
    )

    args = parser.parse_args()

    split_dir = Path(
        args.split_dir
    )

    cross_study_paths = [
        Path(path)
        for path in args.cross_study
    ]

    output_dir = Path(
        args.output_dir
    )

    # --------------------------------------------------------
    # Check split directory
    # --------------------------------------------------------

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: "
            f"{split_dir}"
        )

    # --------------------------------------------------------
    # Process each LCO split
    # --------------------------------------------------------

    split_directories = sorted(
        path
        for path in split_dir.glob("split_*")
        if path.is_dir()
    )

    if not split_directories:
        raise ValueError(
            f"No split directories found in: {split_dir}"
        )

    for current_split_dir in split_directories:

        split_name = current_split_dir.name

        training_path = (
            current_split_dir
            / TRAIN_FILENAME
        )

        if not training_path.exists():
            raise FileNotFoundError(
                f"Training file not found for "
                f"{split_name}:\n"
                f"{training_path}"
            )

        print("\n" + "=" * 70)

        print(
            f"PROCESSING {split_name.upper()}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Load within-study training data
        # ----------------------------------------------------

        training_data = load_csv(
            training_path
        )

        print(
            f"Within-study training observations: "
            f"{len(training_data):,}"
        )

        # ----------------------------------------------------
        # Get training cell lines
        # ----------------------------------------------------

        training_cell_lines = (
            get_training_cell_lines(
                training_data
            )
        )

        print(
            f"Within-study training cell lines: "
            f"{len(training_cell_lines):,}"
        )

        # ----------------------------------------------------
        # Create output directory for split
        # ----------------------------------------------------

        split_output_dir = (
            output_dir / split_name
        )

        split_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Process each cross-study dataset
        # ----------------------------------------------------

        for cross_study_path in cross_study_paths:

            print("\n" + "-" * 70)

            print(
                f"Processing cross-study dataset:\n"
                f"{cross_study_path}"
            )

            cross_study = load_csv(
                cross_study_path
            )

            original_rows = len(
                cross_study
            )

            original_cell_lines = (
                cross_study[CELL_LINE_COLUMN]
                .dropna()
                .astype(str)
                .str.strip()
                .nunique()
                if CELL_LINE_COLUMN in cross_study.columns
                else 0
            )

            # ------------------------------------------------
            # Filter using ONLY the current training set
            # ------------------------------------------------

            unseen_data, removed_count = (
                create_unseen_dataset(
                    cross_study,
                    training_cell_lines,
                )
            )

            unseen_cell_lines = (
                unseen_data[CELL_LINE_COLUMN]
                .dropna()
                .astype(str)
                .str.strip()
                .nunique()
            )

            # ------------------------------------------------
            # Output filename
            # ------------------------------------------------

            dataset_name = get_dataset_name(
                cross_study_path
            )

            output_path = (
                split_output_dir
                / f"{dataset_name}_unseen.csv"
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            unseen_data.to_csv(
                output_path,
                index=False,
            )

            # ------------------------------------------------
            # Print summary
            # ------------------------------------------------

            print(
                f"Original observations: "
                f"{original_rows:,}"
            )

            print(
                f"Original cell lines: "
                f"{original_cell_lines:,}"
            )

            print(
                f"Observations removed: "
                f"{removed_count:,}"
            )

            print(
                f"Unseen observations remaining: "
                f"{len(unseen_data):,}"
            )

            print(
                f"Unseen cell lines remaining: "
                f"{unseen_cell_lines:,}"
            )

            print(
                f"Saved:\n"
                f"{output_path}"
            )

    # --------------------------------------------------------
    # Completion message
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "UNSEEN CROSS-STUDY DATASETS CREATED"
    )

    print("=" * 70)

    print(
        "Definition:"
    )

    print(
        "Cross-study cell lines are considered unseen "
        "if they are absent from the within-study TRAINING "
        "set of the corresponding LCO split."
    )

    print(
        "\nAUC filtering: NOT APPLIED"
    )

    print(
        "Response standardisation: NOT APPLIED"
    )

    print(
        f"\nOutput directory:\n"
        f"{output_dir.resolve()}"
    )


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":
    main()