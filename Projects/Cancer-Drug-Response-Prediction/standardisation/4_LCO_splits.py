"""
Script: Create LCO Splits
Purpose
-------
Create Leave-Cell-Line-Out (LCO) split sets from a
response dataset that has already been:

1. Cleaned
2. AUC-filtered
3. Matched with gene-expression data

By default, 5 split sets are created.

Each split set contains: train / validation / test

Cell lines are kept completely separate between train, validation, and test.
"""
# ============================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, train_test_split
# ============================================================
# SECTION 2: CONFIGURATION
# ============================================================
cell_line_column = "cell_line_name"

required_columns = [cell_line_column,]

required_roles = (
    "train",
    "validation",
    "test",
)

Fold = dict[str, pd.DataFrame]

# ============================================================
# SECTION 3: SPLIT SETTINGS
# ============================================================
@dataclass(frozen=True)
class SplitConfig:
    """Settings needed to reproduce the LCO splits."""

    n_cv_splits: int = 5
    validation_ratio: float = 0.10
    random_state: int = 42

    def validate(self) -> None:

        if self.n_cv_splits < 2:
            raise ValueError("n_cv_splits must be at least 2.")

        if not 0 < self.validation_ratio < 1:
            raise ValueError("validation_ratio must be strictly between 0 and 1.")

# ============================================================
# SECTION 4: LOAD RESPONSE DATA
# ============================================================
def load_response_data(input_path: Path) -> pd.DataFrame:
    """Load and validate the response dataset."""
    data = pd.read_csv(
        input_path,
        low_memory=False,
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
            f"Input file is missing required columns: "
            f"{missing_columns}"
        )
    # --------------------------------------------------------
    # Check cell-line values before converting to strings
    # --------------------------------------------------------
    missing_cell_lines = (
        data[cell_line_column].isna().sum()
    )
    if missing_cell_lines > 0:
        raise ValueError(
            f"Column {cell_line_column!r} contains "
            f"{missing_cell_lines} missing values."
        )
    # --------------------------------------------------------
    # Clean cell-line identifiers
    # --------------------------------------------------------
    data[cell_line_column] = (
        data[cell_line_column]
        .astype(str)
        .str.strip()
    )

    empty_cell_lines = (
        data[cell_line_column]
        .eq("")
        .sum()
    )

    if empty_cell_lines > 0:
        raise ValueError(
            f"Column {cell_line_column!r} contains "
            f"{empty_cell_lines} empty values."
        )

    return data
# ============================================================
# SECTION 5: SHUFFLE ROWS ONCE
# ============================================================
def shuffle_rows(data: pd.DataFrame, random_state: int,) -> pd.DataFrame:
    """Shuffle the response rows once."""

    rng = np.random.default_rng(random_state)

    return data.iloc[rng.permutation(len(data))].reset_index(drop=True)
# ============================================================
# SECTION 6: GET LCO GROUPING VALUES
# ============================================================
def group_values(data: pd.DataFrame,) -> np.ndarray:
    """Use cell_line_name as the LCO grouping identifier."""

    return (
        data[cell_line_column]
        .astype(str)
        .to_numpy()
    )
# ============================================================
# SECTION 7: SELECT ROWS BY INDEX
# ============================================================
def subset(
    data: pd.DataFrame,
    indices: np.ndarray,
) -> pd.DataFrame:
    """Return rows corresponding to the supplied indices."""

    return data.iloc[
        np.asarray(indices, dtype=int)
    ].reset_index(drop=True).copy()

# ============================================================
# SECTION 8: TRAIN / VALIDATION SPLIT
# ============================================================
def grouped_train_validation_indices(
    candidate_indices: np.ndarray,
    all_groups: np.ndarray,
    *,
    validation_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split candidate cell lines into train and validation groups.

    Cell lines, rather than individual observations, are split.
    """

    candidate_groups = np.unique(all_groups[candidate_indices])

    if len(candidate_groups) < 2:
        raise ValueError("At least two cell lines are required to create"
                         "training and validation data."
        )

    train_groups, validation_groups = train_test_split(
        candidate_groups,
        test_size=validation_ratio,
        shuffle=True,
        random_state=random_state,
    )

    candidate_mask = np.zeros(
        len(all_groups),
        dtype=bool,
    )

    candidate_mask[candidate_indices] = True

    train_indices = np.flatnonzero(
        candidate_mask
        &
        np.isin(
            all_groups,
            train_groups,
        )
    )

    validation_indices = np.flatnonzero(
        candidate_mask
        &
        np.isin(
            all_groups,
            validation_groups,
        )
    )

    return (
        train_indices,
        validation_indices,
    )

# ============================================================
# SECTION 9: CREATE LCO SPLITS
# ============================================================
def create_lco_splits(
    data: pd.DataFrame,
    config: SplitConfig,
) -> list[Fold]:
    """Create the LCO split sets."""

    config.validate()

    # --------------------------------------------------------
    # Shuffle rows once
    # --------------------------------------------------------

    shuffled = shuffle_rows(
        data,
        config.random_state,
    )

    # --------------------------------------------------------
    # Group observations by cell line
    # --------------------------------------------------------

    groups = group_values(shuffled)

    n_groups = len(np.unique(groups))

    if config.n_cv_splits > n_groups:
        raise ValueError(
            f"n_cv_splits={config.n_cv_splits} exceeds "
            f"the {n_groups} unique cell lines."
        )

    print(f"\nn_cv_splits={config.n_cv_splits}")

    print(f"Unique cell lines={n_groups}")

    # --------------------------------------------------------
    # Create LCO test groups
    # --------------------------------------------------------

    splitter = GroupKFold(n_splits=config.n_cv_splits)

    folds: list[Fold] = []

    dummy_target = np.zeros(
        len(shuffled),
        dtype=float,
    )

    for (
        outer_train_indices,
        test_indices,
    ) in splitter.split(
        dummy_target,
        groups=groups,
    ):

        # ----------------------------------------------------
        # Divide remaining cell lines into train/validation
        # ----------------------------------------------------

        (
            train_indices,
            validation_indices,
        ) = grouped_train_validation_indices(
            outer_train_indices,
            groups,
            validation_ratio=config.validation_ratio,
            random_state=config.random_state,
        )

        fold = {
            "train": subset(
                shuffled,
                train_indices,
            ),

            "validation": subset(
                shuffled,
                validation_indices,
            ),

            "test": subset(
                shuffled,
                test_indices,
            ),
        }

        # ----------------------------------------------------
        # Check that all rows occur exactly once
        # ----------------------------------------------------

        total_rows = sum(
            len(fold[role])
            for role in required_roles
        )

        if total_rows != len(shuffled):
            raise AssertionError(
                "A split set does not partition all "
                "source rows exactly once."
            )

        folds.append(fold)

    return folds
# ============================================================
# SECTION 10: VALIDATE NO CELL-LINE LEAKAGE
# ============================================================
def key_set(
    data: pd.DataFrame,
) -> set[str]:
    """Return the cell lines present in a dataset."""

    return set(
        map(
            str,
            group_values(data),
        )
    )

def validate_folds(
    folds: list[Fold],
) -> None:
    """Check that cell lines do not overlap between roles."""

    if not folds:
        raise ValueError("No split sets were created.")

    for split_index, fold in enumerate(folds):

        # ----------------------------------------------------
        # Check that all roles exist and contain data
        # ----------------------------------------------------
        for role in required_roles:

            if (
                role not in fold
                or fold[role].empty
            ):
                raise ValueError(
                    f"Split set {split_index} has an empty "
                    f"or missing {role!r} role."
                )

        # ----------------------------------------------------
        # Get cell lines for each role
        # ----------------------------------------------------
        role_keys = {
            role: key_set(
                fold[role]
            )
            for role in required_roles
        }

        # ----------------------------------------------------
        # Check for overlap
        # ----------------------------------------------------

        role_pairs = (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )

        for left, right in role_pairs:

            overlap = (
                role_keys[left]
                &
                role_keys[right]
            )

            if overlap:

                example = next(
                    iter(overlap)
                )

                raise ValueError(
                    f"Split set {split_index}: "
                    f"{left} and {right} share a cell line. "
                    f"For example: {example!r}"
                )
# ============================================================
# SECTION 11: WRITE SPLIT FILES
# ============================================================
def write_split_files(
    folds: list[Fold],
    *,
    output_directory: Path,
    dataset_name: str,
    config: SplitConfig,
) -> None:
    """Write split CSV files and the JSON manifest."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    # --------------------------------------------------------
    # Write train / validation / test files
    # --------------------------------------------------------

    for split_index, fold in enumerate(
        folds
    ):

        split_directory = (
            output_directory
            / f"split_{split_index + 1}"
        )

        split_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for role in required_roles:

            fold[role].to_csv(
                split_directory
                / f"{role}.csv",
                index=False,
            )
    # --------------------------------------------------------
    # Create JSON manifest
    # --------------------------------------------------------

    manifest = {
        "test_mode": "LCO",

        "dataset_name": dataset_name,

        "n_cv_splits": len(folds),

        "validation_ratio": config.validation_ratio,

        "random_state": config.random_state,

        "grouping_column": cell_line_column,

        "splits": [
            {
                "split_index": split_index + 1,

                "dataset_name": dataset_name,

                "role_sizes": {
                    role: len(fold[role])
                    for role in required_roles
                },

                "cell_line_counts": {
                    role: fold[role][
                        cell_line_column
                    ].nunique()
                    for role in required_roles
                },
            }

            for split_index, fold
            in enumerate(folds)
        ],
    }

    manifest_path = (
        output_directory
        / "split_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

# ============================================================
# SECTION 12: MAIN
# ============================================================
def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the gene-expression-matched response CSV.",
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where the LCO split files will be saved.",
    )

    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name recorded in the split manifest.",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of LCO split sets. Default: 5.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.10,
        help="Validation proportion. Default: 0.10.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    output_directory = Path(args.out_dir)

    # --------------------------------------------------------
    # Load response data
    # --------------------------------------------------------

    print(f"Loading response data from: \n {input_path}")

    data = load_response_data(input_path)

    print(f"\nObservations: {len(data):,}")

    print(f"Unique cell lines: {data[cell_line_column].nunique():,}")

    # --------------------------------------------------------
    # Build split configuration
    # --------------------------------------------------------

    config = SplitConfig(
        n_cv_splits=args.n_splits,
        validation_ratio=args.validation_ratio,
        random_state=args.seed,
    )

    # --------------------------------------------------------
    # Create LCO splits
    # --------------------------------------------------------

    print(f"\nCreating {config.n_cv_splits} LCO split sets...")

    folds = create_lco_splits(data,config,)

    print(f"Created {len(folds)} LCO split sets.")

    # --------------------------------------------------------
    # Validate splits
    # --------------------------------------------------------

    validate_folds(folds)

    print("Validation passed: no cell line is shared between train, validation, and test.")

    # --------------------------------------------------------
    # Write files
    # --------------------------------------------------------

    write_split_files(
        folds,
        output_directory=output_directory,
        dataset_name=args.dataset_name,
        config=config,
    )

    print(f"\nWrote {len(folds)} LCO split sets to:")

    print(output_directory.resolve())

    print(f"\nManifest:")

    print(
        (
            output_directory
            / "split_manifest.json"
        ).resolve()
    )

# ============================================================
# SECTION 13: RUN SCRIPT
# ============================================================
if __name__ == "__main__":
    main()