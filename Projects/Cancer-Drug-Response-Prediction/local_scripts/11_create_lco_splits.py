
"""
Script 03: Create 5 LCO split sets for a drug-response dataset.

Purpose
-------
Takes a clean GDSC2_response_standardised_clean.csv, verifies that it contains
the required columns and identifiers, and creates 5 LCO split sets
using GroupKFold.

Each split set contains:
    train / validation / test

The cell lines are kept separate between train, validation, and test.

It does NOT:
- implement LPO
- implement LDO
- implement LTO
- train any model
- compute any metric
- evaluate model performance
- re-clean the input response data
"""

# Import Libraries
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, train_test_split


# ============================================================
# Section A: Configuration
# ============================================================

Data_root = (
    "/Users/manellamukangwa/Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction/data"
)

cell_line_column = "cell_line_name"
drug_column = "pubchem_id"

# GDSC2 uses the standardised response scale
response_column = "LN_IC50_standardised"

tissue_column = "tissue"

required_columns = [
    cell_line_column,
    drug_column,
    response_column,
    tissue_column
]

id_columns = [
    cell_line_column,
    drug_column
]

# Default locations
Default_input = f"{Data_root}/processed/GDSC2_response_standardised_clean.csv"
Default_output_directory = f"{Data_root}/processed/lco_splits_gdsc2"


# Creates a fixed set of splitting modes: LCO
test_modes = frozenset({"LCO"})
required_roles = ("train", "validation", "test")

# Creates a dictionary that holds information from dataframes
Fold = dict[str, pd.DataFrame]


# ============================================================
# Section B: Settings needed to run the LCO split
# ============================================================

@dataclass(frozen=True)
class SplitConfig:
    """Settings needed to reproduce the LCO split roles."""

    n_cv_splits: int
    validation_ratio: float = 0.1
    random_state: int = 42

    def validate(self) -> None:
        """Basic sanity checks before any rows are partitioned."""

        if self.n_cv_splits < 2:
            raise ValueError("n_cv_splits must be at least 2")

        if not 0 < self.validation_ratio < 1:
            raise ValueError(
                "validation_ratio must be strictly between 0 and 1"
            )


# ============================================================
# Section C: Load clean response table and VERIFY it
# ============================================================

def load_clean_response_table(input_path: Path) -> pd.DataFrame:
    """Load clean raw GDSC2 response table and verify it is cleaned."""

    data = pd.read_csv(
        input_path,
        dtype={
            cell_line_column: str,
            drug_column: str,
            tissue_column: str
        },
        low_memory=False,
    )

    # Check all expected columns are present
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Input file is missing expected columns: {missing_columns}."
        )

    # Check no ID columns have missing/blank values
    for column in id_columns:

        n_missing = (
            data[column].isna().sum()
            + (
                data[column].astype(str).str.strip() == ""
            ).sum()
        )

        if n_missing > 0:
            raise ValueError(
                f"Column {column!r} contains "
                f"{n_missing} missing/blank values."
            )

        print(
            f"Column {column!r} has "
            f"{n_missing} missing/blank values."
        )

    # Check response column for missing values
    n_missing_response = data[response_column].isna().sum()

    if n_missing_response > 0:
        raise ValueError(
            f"Column {response_column!r} contains "
            f"{n_missing_response} missing values."
        )

    print(
        f"Column {response_column!r} has "
        f"{n_missing_response} missing values."
    )

    return data


# ============================================================
# Section D: Shuffle rows once, using the constant seed
# ============================================================

def shuffle_rows(
    data: pd.DataFrame,
    random_state: int
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)

    return data.iloc[
        rng.permutation(len(data))
    ].reset_index(drop=True)


# ============================================================
# Section E: Get LCO grouping values
# ============================================================

def group_values(data: pd.DataFrame) -> np.ndarray:
    return data[cell_line_column].astype(str).to_numpy()


# ============================================================
# Section F: Pick out specific rows by index
# ============================================================

def subset(
    data: pd.DataFrame,
    indices: np.ndarray
) -> pd.DataFrame:

    return data.iloc[
        np.asarray(indices, dtype=int)
    ].reset_index(drop=True).copy()


# ============================================================
# Section G: Split training/validation by cell line
# ============================================================

def grouped_train_validation_indices(
    candidate_indices: np.ndarray,
    all_groups: np.ndarray,
    *,
    validation_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:

    candidate_groups = np.unique(
        all_groups[candidate_indices]
    )

    if len(candidate_groups) < 2:
        raise ValueError(
            "At least two training groups are required "
            "to create validation data"
        )

    train_groups, validation_groups = train_test_split(
        candidate_groups,
        test_size=validation_ratio,
        shuffle=True,
        random_state=random_state,
    )

    candidate_mask = np.zeros(
        len(all_groups),
        dtype=bool
    )

    candidate_mask[candidate_indices] = True

    train_indices = np.flatnonzero(
        candidate_mask
        & np.isin(all_groups, train_groups)
    )

    validation_indices = np.flatnonzero(
        candidate_mask
        & np.isin(all_groups, validation_groups)
    )

    return train_indices, validation_indices


# ============================================================
# Section H: Build the LCO split sets
# ============================================================

def create_lco_splits(
    data: pd.DataFrame,
    config: SplitConfig
) -> list[Fold]:

    config.validate()

    # Shuffle all rows once, using the constant seed
    shuffled = shuffle_rows(
        data,
        config.random_state
    )

    # Group rows by cell line
    groups = group_values(shuffled)

    n_groups = len(np.unique(groups))

    if config.n_cv_splits > n_groups:
        raise ValueError(
            f"n_cv_splits={config.n_cv_splits} exceeds "
            f"the {n_groups} unique cell lines"
        )

    print(
        f"n_cv_splits={config.n_cv_splits}, "
        f"unique cell lines={n_groups}"
    )

    # GroupKFold divides all cell lines into the required groups
    splitter = GroupKFold(
        n_splits=config.n_cv_splits
    )

    folds: list[Fold] = []

    dummy_target = np.zeros(
        len(shuffled),
        dtype=float
    )

    # For each split, build train / validation / test
    for outer_train_indices, test_indices in splitter.split(
        dummy_target,
        groups=groups
    ):

        train_indices, validation_indices = (
            grouped_train_validation_indices(
                outer_train_indices,
                groups,
                validation_ratio=config.validation_ratio,
                random_state=config.random_state,
            )
        )

        fold = {
            "train": subset(
                shuffled,
                train_indices
            ),
            "validation": subset(
                shuffled,
                validation_indices
            ),
            "test": subset(
                shuffled,
                test_indices
            ),
        }

        if sum(
            len(fold[role])
            for role in required_roles
        ) != len(shuffled):

            raise AssertionError(
                "A split set does not partition all "
                "source rows exactly once"
            )

        folds.append(fold)

    return folds


# ============================================================
# Section I: Check that no cell line is shared between roles
# ============================================================

def key_set(data: pd.DataFrame) -> set[str]:

    return set(
        map(str, group_values(data))
    )


def validate_folds(
    folds: list[Fold]
) -> None:

    if not folds:
        raise ValueError(
            "No split sets were created"
        )

    for split_index, fold in enumerate(folds):

        for role in required_roles:

            if (
                role not in fold
                or fold[role].empty
            ):

                raise ValueError(
                    f"Split set {split_index} has an "
                    f"empty or missing {role!r} role"
                )

        role_keys = {
            role: key_set(fold[role])
            for role in required_roles
        }

        role_pairs = (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test")
        )

        for left, right in role_pairs:

            overlap = (
                role_keys[left]
                & role_keys[right]
            )

            if overlap:

                example = next(iter(overlap))

                raise ValueError(
                    f"Split set {split_index}: "
                    f"{left} and {right} share a cell line, "
                    f"for example {example!r}"
                )


# ============================================================
# Section J: Write split files and manifest
# ============================================================

def write_split_files(
    folds: list[Fold],
    *,
    output_directory: Path,
    dataset_name: str,
    config: SplitConfig,
) -> None:

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    for split_index, fold in enumerate(folds):

        for role in required_roles:

            fold[role].to_csv(
                output_directory
                / f"cv_split_{split_index}_{role}.csv",
                index=False
            )

    manifest = {
        "test_mode": "LCO",
        "n_cv_splits": len(folds),
        "validation_ratio": config.validation_ratio,
        "random_state": config.random_state,
        "splits": [
            {
                "split_index": split_index,
                "dataset_name": dataset_name,
                "role_sizes": {
                    role: len(fold[role])
                    for role in required_roles
                },
            }
            for split_index, fold
            in enumerate(folds)
        ],
    }

    (
        output_directory
        / "split_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True
        ) + "\n",
        encoding="utf-8",
    )


# ============================================================
# Section K: Set up how the script will be run via terminal
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        default=Default_input,
        help=(
            "Path to cleaned raw GDSC2 response table "
            f"(default: {Default_input})"
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=Default_output_directory,
        help=(
            "Directory to write the LCO split-set files "
            f"(default: {Default_output_directory})"
        ),
    )

    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name recorded in the manifest"
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of LCO split sets (default: 5)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Constant random seed, used once"
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    # Load clean raw GDSC2 response file and verify it
    print(
        f"Loading the cleaned {args.dataset_name} "
        f"response table from {input_path} ..."
    )

    data = load_clean_response_table(
        input_path
    )

    print(f"  Rows: {len(data)}")

    print(
        f"  Unique cell lines available for LCO grouping: "
        f"{data[cell_line_column].nunique()}"
    )

    # Build LCO split configuration
    config = SplitConfig(
        n_cv_splits=args.n_splits,
        validation_ratio=args.validation_ratio,
        random_state=args.seed,
    )

    # Create LCO split sets
    print(
        f"\nCreating {args.n_splits} LCO split sets "
        f"(seed={args.seed}) ..."
    )

    folds = create_lco_splits(
        data,
        config
    )

    print(
        f"  Created {len(folds)} LCO split sets "
        "(train/validation/test each)"
    )

    # Validate no cell-line leakage
    validate_folds(folds)

    print(
        "Validation passed: no cell line shared between "
        "train, validation, and test"
    )

    # Write split files
    write_split_files(
        folds,
        output_directory=out_dir,
        dataset_name=args.dataset_name,
        config=config,
    )

    print(
        f"\nWrote {len(folds)} LCO split sets to "
        f"{out_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
