"""
LCO SPLITS SCRIPT: Create 5 LCO split sets for a drug-response dataset.

Purpose
-------
Takes a clean GDSC2 response dataset and creates exactly 5 LCO split sets.

Before splitting, the script restricts the response data to cell lines
that have matching gene-expression data.

Each split set contains:
    train / validation / test

The cell lines are kept completely separate between train, validation,
and test.

It does NOT:
- implement LPO, LDO, LTO
- train any model
- compute any metric
- evaluate model performance
- re-clean the original response data

The original cleaned response file is NOT modified.
"""

# =============================================================
# SECTION 1: IMPORT LIBRARIES
# =============================================================

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, train_test_split


# =============================================================
# SECTION 2: CONFIGURATION
# =============================================================

Data_root = (
    "/Users/manellamukangwa/Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction/data"
)

cell_line_column = "cell_line_name"
cellosaurus_column = "cellosaurus_id"
drug_column = "pubchem_id"

response_column = "LN_IC50_curvecurator"

tissue_column = "tissue"

required_columns = [
    cell_line_column,
    cellosaurus_column,
    drug_column,
    response_column,
    tissue_column,
]

id_columns = [
    cell_line_column,
    cellosaurus_column,
    drug_column,
]

# -------------------------------------------------------------
# Default locations
# -------------------------------------------------------------

Default_input = (
    f"{Data_root}/processed/GDSC2_response_cleaned.csv"
)

Default_gene_expression = (
    f"{Data_root}/raw/GDSC2/gene_expression.csv"
)

Default_output_directory = (
    f"{Data_root}/processed/lco_splits_gdsc2"
)

required_roles = (
    "train",
    "validation",
    "test",
)

Fold = dict[str, pd.DataFrame]


# =============================================================
# SECTION 3: SPLIT SETTINGS
# =============================================================

@dataclass(frozen=True)
class SplitConfig:
    """Settings needed to reproduce the LCO split roles."""

    n_cv_splits: int
    validation_ratio: float = 0.1
    random_state: int = 42

    def validate(self) -> None:

        if self.n_cv_splits < 2:
            raise ValueError(
                "n_cv_splits must be at least 2"
            )

        if not 0 < self.validation_ratio < 1:
            raise ValueError(
                "validation_ratio must be strictly between 0 and 1"
            )


# =============================================================
# SECTION 4: LOAD CLEAN RESPONSE TABLE
# =============================================================

def load_clean_response_table(
    input_path: Path
) -> pd.DataFrame:
    """
    Load the cleaned GDSC2 response table and verify it.
    """

    data = pd.read_csv(
        input_path,
        dtype={
            cell_line_column: str,
            cellosaurus_column: str,
            drug_column: str,
            tissue_column: str,
        },
        low_memory=False,
    )

    # ---------------------------------------------------------
    # Check required columns
    # ---------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Input file is missing expected columns: "
            f"{missing_columns}"
        )

    # ---------------------------------------------------------
    # Check ID columns
    # ---------------------------------------------------------

    for column in id_columns:

        n_missing = (
            data[column].isna().sum()
            +
            (
                data[column]
                .astype(str)
                .str.strip()
                .eq("")
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

    # ---------------------------------------------------------
    # Check response
    # ---------------------------------------------------------

    n_missing_response = (
        data[response_column].isna().sum()
    )

    if n_missing_response > 0:
        raise ValueError(
            f"Column {response_column!r} contains "
            f"{n_missing_response} missing values."
        )

    print(
        f"Column {response_column!r} has "
        f"{n_missing_response} missing values."
    )

    # Make identifiers clean strings
    data[cell_line_column] = (
        data[cell_line_column]
        .astype(str)
        .str.strip()
    )

    data[cellosaurus_column] = (
        data[cellosaurus_column]
        .astype(str)
        .str.strip()
    )

    data[drug_column] = (
        data[drug_column]
        .astype(str)
        .str.strip()
    )

    return data


# =============================================================
# SECTION 5: FILTER TO CELL LINES WITH GENE EXPRESSION
# =============================================================

def restrict_to_gene_expression_cells(
    data: pd.DataFrame,
    gene_expression_path: Path,
) -> pd.DataFrame:
    """
    Keep only response rows whose cellosaurus_id occurs in
    the GDSC2 gene-expression dataset.

    The original response dataset is not modified.
    """

    print(
        f"\nLoading GDSC2 gene-expression data from "
        f"{gene_expression_path} ..."
    )

    gene_expression = pd.read_csv(
        gene_expression_path,
        usecols=[cellosaurus_column],
        low_memory=False,
    )

    gene_expression[cellosaurus_column] = (
        gene_expression[cellosaurus_column]
        .astype(str)
        .str.strip()
    )

    gene_expression_cells = set(
        gene_expression[cellosaurus_column]
    )

    response_cells = set(
        data[cellosaurus_column]
    )

    missing_cells = sorted(
        response_cells - gene_expression_cells
    )

    filtered_data = data[
        data[cellosaurus_column].isin(
            gene_expression_cells
        )
    ].copy()

    print(
        f"  Response cell lines before filtering: "
        f"{len(response_cells)}"
    )

    print(
        f"  Gene-expression cell lines available: "
        f"{len(gene_expression_cells)}"
    )

    print(
        f"  Cell lines without gene expression: "
        f"{len(missing_cells)}"
    )

    if missing_cells:
        print(
            "  These cell lines will not be included in "
            "the LCO splits:"
        )

        for cell in missing_cells:
            print(f"    {cell}")

    print(
        f"  Response rows before filtering: "
        f"{len(data)}"
    )

    print(
        f"  Response rows after filtering: "
        f"{len(filtered_data)}"
    )

    print(
        f"  Cell lines remaining for LCO: "
        f"{filtered_data[cellosaurus_column].nunique()}"
    )

    if filtered_data.empty:
        raise ValueError(
            "No response rows remain after restricting "
            "to cell lines with gene expression."
        )

    return filtered_data


# =============================================================
# SECTION 6: SHUFFLE ROWS ONCE
# =============================================================

def shuffle_rows(
    data: pd.DataFrame,
    random_state: int
) -> pd.DataFrame:

    rng = np.random.default_rng(
        random_state
    )

    return data.iloc[
        rng.permutation(len(data))
    ].reset_index(drop=True)


# =============================================================
# SECTION 7: GET LCO GROUPING VALUES
# =============================================================

def group_values(
    data: pd.DataFrame
) -> np.ndarray:
    """
    Use Cellosaurus ID as the LCO grouping identifier.
    """

    return (
        data[cellosaurus_column]
        .astype(str)
        .to_numpy()
    )


# =============================================================
# SECTION 8: SELECT ROWS BY INDEX
# =============================================================

def subset(
    data: pd.DataFrame,
    indices: np.ndarray
) -> pd.DataFrame:

    return data.iloc[
        np.asarray(indices, dtype=int)
    ].reset_index(drop=True).copy()


# =============================================================
# SECTION 9: TRAIN / VALIDATION SPLIT
# =============================================================

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
            "to create validation data."
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
        &
        np.isin(
            all_groups,
            train_groups
        )
    )

    validation_indices = np.flatnonzero(
        candidate_mask
        &
        np.isin(
            all_groups,
            validation_groups
        )
    )

    return (
        train_indices,
        validation_indices
    )


# =============================================================
# SECTION 10: CREATE LCO SPLITS
# =============================================================

def create_lco_splits(
    data: pd.DataFrame,
    config: SplitConfig
) -> list[Fold]:

    config.validate()

    # Shuffle once
    shuffled = shuffle_rows(
        data,
        config.random_state
    )

    # Group by cellosaurus ID
    groups = group_values(
        shuffled
    )

    n_groups = len(
        np.unique(groups)
    )

    if config.n_cv_splits > n_groups:
        raise ValueError(
            f"n_cv_splits={config.n_cv_splits} exceeds "
            f"the {n_groups} unique cell lines."
        )

    print(
        f"\nn_cv_splits={config.n_cv_splits}, "
        f"unique cell lines={n_groups}"
    )

    splitter = GroupKFold(
        n_splits=config.n_cv_splits
    )

    folds: list[Fold] = []

    dummy_target = np.zeros(
        len(shuffled),
        dtype=float
    )

    for (
        outer_train_indices,
        test_indices
    ) in splitter.split(
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

        # Make sure all rows occur exactly once
        if sum(
            len(fold[role])
            for role in required_roles
        ) != len(shuffled):

            raise AssertionError(
                "A split set does not partition all "
                "source rows exactly once."
            )

        folds.append(fold)

    return folds


# =============================================================
# SECTION 11: VALIDATE NO CELL-LINE LEAKAGE
# =============================================================

def key_set(
    data: pd.DataFrame
) -> set[str]:

    return set(
        map(
            str,
            group_values(data)
        )
    )


def validate_folds(
    folds: list[Fold]
) -> None:

    if not folds:
        raise ValueError(
            "No split sets were created."
        )

    for split_index, fold in enumerate(folds):

        for role in required_roles:

            if (
                role not in fold
                or fold[role].empty
            ):
                raise ValueError(
                    f"Split set {split_index} has "
                    f"an empty or missing {role!r} role."
                )

        role_keys = {
            role: key_set(fold[role])
            for role in required_roles
        }

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
                    f"{left} and {right} share a cell line, "
                    f"for example {example!r}"
                )


# =============================================================
# SECTION 12: WRITE SPLIT FILES
# =============================================================

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
                /
                f"cv_split_{split_index}_{role}.csv",
                index=False
            )

    manifest = {
        "test_mode": "LCO",
        "n_cv_splits": len(folds),
        "validation_ratio": config.validation_ratio,
        "random_state": config.random_state,
        "grouping_column": cellosaurus_column,
        "splits": [
            {
                "split_index": split_index,
                "dataset_name": dataset_name,
                "role_sizes": {
                    role: len(fold[role])
                    for role in required_roles
                },
                "cell_line_counts": {
                    role: fold[role][
                        cellosaurus_column
                    ].nunique()
                    for role in required_roles
                },
            }
            for split_index, fold
            in enumerate(folds)
        ],
    }

    (
        output_directory
        /
        "split_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True
        )
        +
        "\n",
        encoding="utf-8",
    )


# =============================================================
# SECTION 13: RUN SCRIPT FROM TERMINAL
# =============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        default=Default_input,
        help=(
            "Path to cleaned GDSC2 response table "
            f"(default: {Default_input})"
        ),
    )

    parser.add_argument(
        "--gene-expression",
        default=Default_gene_expression,
        help=(
            "Path to GDSC2 gene-expression table "
            f"(default: {Default_gene_expression})"
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=Default_output_directory,
        help=(
            "Directory to write LCO split files "
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
        help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Validation proportion (default: 0.1)"
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    gene_expression_path = Path(
        args.gene_expression
    )

    out_dir = Path(
        args.out_dir
    )

    # ---------------------------------------------------------
    # Load response data
    # ---------------------------------------------------------

    print(
        f"Loading cleaned {args.dataset_name} "
        f"response table from:"
    )

    print(
        f"  {input_path}"
    )

    data = load_clean_response_table(
        input_path
    )

    print(
        f"\nOriginal response rows: "
        f"{len(data)}"
    )

    print(
        f"Original unique cell lines: "
        f"{data[cellosaurus_column].nunique()}"
    )

    # ---------------------------------------------------------
    # Restrict to cells with gene expression
    # ---------------------------------------------------------

    data = restrict_to_gene_expression_cells(
        data,
        gene_expression_path
    )

    # ---------------------------------------------------------
    # Build configuration
    # ---------------------------------------------------------

    config = SplitConfig(
        n_cv_splits=args.n_splits,
        validation_ratio=args.validation_ratio,
        random_state=args.seed,
    )

    # ---------------------------------------------------------
    # Create LCO splits
    # ---------------------------------------------------------

    print(
        f"\nCreating exactly {args.n_splits} "
        f"LCO split sets..."
    )

    folds = create_lco_splits(
        data,
        config
    )

    print(
        f"Created {len(folds)} LCO split sets."
    )

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    validate_folds(
        folds
    )

    print(
        "Validation passed: no cell line is shared "
        "between train, validation, and test."
    )

    # ---------------------------------------------------------
    # Write files
    # ---------------------------------------------------------

    write_split_files(
        folds,
        output_directory=out_dir,
        dataset_name=args.dataset_name,
        config=config,
    )

    print(
        f"\nWrote {len(folds)} LCO split sets to:"
    )

    print(
        f"  {out_dir.resolve()}"
    )


if __name__ == "__main__":
    main()