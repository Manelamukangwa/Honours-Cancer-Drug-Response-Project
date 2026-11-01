"""
Script: Drug Coverage and Shared-Drug Filtering

Purpose
-------
Restrict the LCO splits and the unseen cross-study datasets to the
set of drugs shared between datasets, so that the Random Forest is
trained, tuned and evaluated on the same drug set.

The script:

    1. Loads the within-study LCO splits (train/validation/test).
    2. Computes drug coverage for the within-study dataset and for
       each full cross-study dataset
       (number of cell lines and observations per drug).
    3. Selects the drugs shared between datasets
       (optionally requiring a minimum number of cell lines per drug
       in every dataset).
    4. Filters train/validation/test and each unseen cross-study
       file to the shared drugs.
    5. Writes the filtered data in the same layout as the LCO split
       directory, so the Random Forest script can read it directly.

Overlap modes
-------------
all:
    One drug set shared by the within-study dataset AND every
    cross-study dataset. The model is trained once and all
    cross-study datasets are evaluated on identical drugs.

pairwise:
    One drug set per cross-study dataset (within-study dataset vs
    that cross-study dataset). Output is written to one
    sub-directory per cross-study dataset.

Important
---------
Unseen cell lines are NOT redefined here. The *_unseen.csv files must
already exist (created by the unseen cross-study script), so a cell
line is still unseen if it is absent from the within-study TRAINING
set of that split.

Cell-line splits are not changed. Only rows are removed, based on drug.

The original input files are never modified.

This script is dataset-agnostic.
"""

# ============================================================
# Import libraries
# ============================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

DEFAULT_CELL_LINE_COLUMN = "cell_line_name"
DEFAULT_DRUG_COLUMN = "pubchem_id"

REQUIRED_ROLES = (
    "train",
    "validation",
    "test",
)

ALL_MODE_TAG = "shared_all"


# ============================================================
# Load CSV
# ============================================================

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


# ============================================================
# Check required columns
# ============================================================

def check_columns(
    data: pd.DataFrame,
    columns: list[str],
    file_path: Path,
) -> None:
    """Check that the required columns exist."""

    missing = [
        column
        for column in columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{file_path} is missing required columns: "
            f"{missing}"
        )


# ============================================================
# Standardise identifiers (used for matching only)
# ============================================================

def clean_ids(
    values: pd.Series,
) -> pd.Series:
    """
    Standardise identifiers for matching.

    Strips whitespace and removes a trailing '.0' so that
    46883536 and 46883536.0 are treated as the same drug.
    The original columns in the saved files are not changed.
    """

    return (
        values
        .astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )


# ============================================================
# Natural sort for split directories
# ============================================================

def split_sort_key(
    path: Path,
) -> int:
    """Sort split_1, split_2, ..., split_10 numerically."""

    suffix = path.name.split("_")[-1]

    return (
        int(suffix)
        if suffix.isdigit()
        else 10**9
    )


# ============================================================
# Load full within-study data
# ============================================================

def load_within_study(
    split_directory: Path,
    cell_line_column: str,
    drug_column: str,
) -> pd.DataFrame:
    """
    Rebuild the full within-study response data.

    Train + validation + test of one LCO split partition every
    row exactly once, so concatenating them recovers the complete
    within-study dataset.
    """

    first_split = sorted(
        (
            path
            for path in split_directory.glob("split_*")
            if path.is_dir()
        ),
        key=split_sort_key,
    )[0]

    parts = []

    for role in REQUIRED_ROLES:

        path = (
            first_split
            / f"{role}.csv"
        )

        data = load_csv(path)

        check_columns(
            data,
            [cell_line_column, drug_column],
            path,
        )

        parts.append(data)

    return pd.concat(
        parts,
        ignore_index=True,
    )


# ============================================================
# Compute drug coverage
# ============================================================

def compute_drug_coverage(
    data: pd.DataFrame,
    cell_line_column: str,
    drug_column: str,
) -> pd.DataFrame:
    """
    Compute drug coverage.

    Returns one row per drug with:
        n_cell_lines   : number of unique cell lines tested
        n_observations : number of response observations
    """

    keys = pd.DataFrame(
        {
            "drug": clean_ids(
                data[drug_column]
            ),
            "cell_line": (
                data[cell_line_column]
                .astype("string")
                .str.strip()
            ),
        }
    ).dropna()

    coverage = (
        keys
        .groupby("drug")
        .agg(
            n_cell_lines=("cell_line", "nunique"),
            n_observations=("cell_line", "size"),
        )
    )

    return coverage


# ============================================================
# Select drugs meeting the coverage threshold
# ============================================================

def eligible_drugs(
    coverage: pd.DataFrame,
    min_cell_lines: int,
) -> set[str]:
    """Drugs tested in at least min_cell_lines cell lines."""

    return set(
        coverage.index[
            coverage["n_cell_lines"]
            >= min_cell_lines
        ]
    )


# ============================================================
# Build the drug sets
# ============================================================

def build_drug_sets(
    within_name: str,
    within_coverage: pd.DataFrame,
    cross_coverage: dict[str, pd.DataFrame],
    mode: str,
    min_cell_lines: int,
) -> dict[str, dict]:
    """
    Build the shared drug set(s).

    Returns
    -------
    {tag: {"drugs": set[str], "cross_study": [dataset names]}}
    """

    within_drugs = eligible_drugs(
        within_coverage,
        min_cell_lines,
    )

    drug_sets: dict[str, dict] = {}

    if mode == "all":

        shared = set(within_drugs)

        for coverage in cross_coverage.values():
            shared &= eligible_drugs(
                coverage,
                min_cell_lines,
            )

        drug_sets[ALL_MODE_TAG] = {
            "drugs": shared,
            "cross_study": list(cross_coverage),
        }

    elif mode == "pairwise":

        for name, coverage in cross_coverage.items():

            drug_sets[name] = {
                "drugs": (
                    within_drugs
                    & eligible_drugs(
                        coverage,
                        min_cell_lines,
                    )
                ),
                "cross_study": [name],
            }

    else:
        raise ValueError(
            f"Unknown mode: {mode!r}"
        )

    for tag, info in drug_sets.items():

        if not info["drugs"]:
            raise ValueError(
                f"No shared drugs were found for {tag!r} "
                f"with min_cell_lines={min_cell_lines}. "
                f"Check that {within_name} and the cross-study "
                f"datasets use the same drug identifier."
            )

    return drug_sets


# ============================================================
# Build coverage summary table
# ============================================================

def build_coverage_summary(
    within_name: str,
    within_coverage: pd.DataFrame,
    cross_coverage: dict[str, pd.DataFrame],
    drug_sets: dict[str, dict],
) -> pd.DataFrame:
    """One row per drug: coverage in every dataset + shared flags."""

    frames = {
        within_name: within_coverage,
        **cross_coverage,
    }

    summary = pd.concat(
        frames,
        axis=1,
    )

    summary.columns = [
        f"{dataset}_{column}"
        for dataset, column in summary.columns
    ]

    summary = summary.fillna(0).astype(int)

    summary.index.name = "drug"

    for tag, info in drug_sets.items():

        summary[f"in_{tag}"] = (
            summary.index.isin(
                info["drugs"]
            )
        )

    return summary.reset_index()


# ============================================================
# Filter by drug
# ============================================================

def filter_by_drugs(
    data: pd.DataFrame,
    drug_column: str,
    drugs: set[str],
) -> pd.DataFrame:
    """Keep rows whose drug is in the shared set."""

    keep = clean_ids(
        data[drug_column]
    ).isin(drugs)

    return data.loc[keep].copy()


# ============================================================
# Filter and save one file
# ============================================================

def filter_and_save(
    input_path: Path,
    output_path: Path,
    cell_line_column: str,
    drug_column: str,
    drugs: set[str],
    label: str,
) -> dict:
    """Filter one response file to the shared drugs and save it."""

    data = load_csv(input_path)

    check_columns(
        data,
        [cell_line_column, drug_column],
        input_path,
    )

    filtered = filter_by_drugs(
        data,
        drug_column,
        drugs,
    )

    if filtered.empty:
        raise ValueError(
            f"{label}: no rows remain after restricting "
            f"to the shared drugs ({input_path})."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    filtered.to_csv(
        output_path,
        index=False,
    )

    return {
        "file": label,
        "rows_before": len(data),
        "rows_after": len(filtered),
        "drugs_before": int(
            clean_ids(data[drug_column]).nunique()
        ),
        "drugs_after": int(
            clean_ids(filtered[drug_column]).nunique()
        ),
        "cell_lines_after": int(
            filtered[cell_line_column]
            .astype("string")
            .str.strip()
            .nunique()
        ),
    }


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
            "Within-study LCO split directory containing "
            "split_1, split_2, ... with train/validation/test.csv "
            "and the *_unseen.csv files."
        ),
    )

    parser.add_argument(
        "--within-study-dataset",
        required=True,
        help="Within-study dataset name, e.g. CTRPv2.",
    )

    parser.add_argument(
        "--cross-study",
        nargs="+",
        required=True,
        help=(
            "Paths to the FULL cross-study response datasets "
            "(same files given to the unseen cross-study script). "
            "The dataset name is the file name without extension."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the filtered data will be saved.",
    )

    parser.add_argument(
        "--mode",
        choices=["all", "pairwise"],
        default="all",
        help=(
            "all: one drug set shared by every dataset. "
            "pairwise: one drug set per cross-study dataset. "
            "Default: all."
        ),
    )

    parser.add_argument(
        "--min-cell-lines",
        type=int,
        default=1,
        help=(
            "Minimum number of cell lines a drug must be tested "
            "in, in every dataset, to be kept. Default: 1."
        ),
    )

    parser.add_argument(
        "--cell-line-column",
        default=DEFAULT_CELL_LINE_COLUMN,
        help=f"Default: {DEFAULT_CELL_LINE_COLUMN}.",
    )

    parser.add_argument(
        "--drug-column",
        default=DEFAULT_DRUG_COLUMN,
        help=f"Default: {DEFAULT_DRUG_COLUMN}.",
    )

    args = parser.parse_args()

    split_dir = Path(
        args.split_dir
    )

    output_dir = Path(
        args.output_dir
    )

    within_name = args.within_study_dataset

    cross_study_paths = [
        Path(path)
        for path in args.cross_study
    ]

    cell_line_column = args.cell_line_column
    drug_column = args.drug_column

    if args.min_cell_lines < 1:
        raise ValueError(
            "--min-cell-lines must be at least 1."
        )

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}"
        )

    split_directories = sorted(
        (
            path
            for path in split_dir.glob("split_*")
            if path.is_dir()
        ),
        key=split_sort_key,
    )

    if not split_directories:
        raise ValueError(
            f"No split directories found in: {split_dir}"
        )

    # --------------------------------------------------------
    # Drug coverage: within-study
    # --------------------------------------------------------

    within_data = load_within_study(
        split_dir,
        cell_line_column,
        drug_column,
    )

    within_coverage = compute_drug_coverage(
        within_data,
        cell_line_column,
        drug_column,
    )

    # --------------------------------------------------------
    # Drug coverage: cross-study
    # --------------------------------------------------------

    cross_coverage: dict[str, pd.DataFrame] = {}

    for path in cross_study_paths:

        data = load_csv(path)

        check_columns(
            data,
            [cell_line_column, drug_column],
            path,
        )

        cross_coverage[path.stem] = (
            compute_drug_coverage(
                data,
                cell_line_column,
                drug_column,
            )
        )

    # --------------------------------------------------------
    # Select shared drugs
    # --------------------------------------------------------

    drug_sets = build_drug_sets(
        within_name,
        within_coverage,
        cross_coverage,
        args.mode,
        args.min_cell_lines,
    )

    # --------------------------------------------------------
    # Save coverage summary
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage_summary = build_coverage_summary(
        within_name,
        within_coverage,
        cross_coverage,
        drug_sets,
    )

    coverage_path = (
        output_dir
        / "drug_coverage_summary.csv"
    )

    coverage_summary.to_csv(
        coverage_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("DRUG COVERAGE")
    print("=" * 70)

    print(
        f"{within_name}: "
        f"{len(within_coverage):,} drugs"
    )

    for name, coverage in cross_coverage.items():
        print(
            f"{name}: "
            f"{len(coverage):,} drugs"
        )

    print(
        f"Mode: {args.mode}"
    )

    print(
        f"Minimum cell lines per drug: "
        f"{args.min_cell_lines}"
    )

    for tag, info in drug_sets.items():
        print(
            f"Shared drugs [{tag}]: "
            f"{len(info['drugs']):,}"
        )

    # --------------------------------------------------------
    # Filter splits and unseen datasets
    # --------------------------------------------------------

    split_rows = []

    for tag, info in drug_sets.items():

        drugs = info["drugs"]

        tag_root = (
            output_dir
            / tag
        )

        tag_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            {drug_column: sorted(drugs)}
        ).to_csv(
            tag_root / "shared_drugs.csv",
            index=False,
        )

        for current_split_dir in split_directories:

            split_name = current_split_dir.name

            split_output_dir = (
                tag_root
                / within_name
                / split_name
            )

            print("\n" + "-" * 70)
            print(
                f"[{tag}] {split_name}"
            )
            print("-" * 70)

            # ------------------------------------------------
            # Within-study train / validation / test
            # ------------------------------------------------

            for role in REQUIRED_ROLES:

                record = filter_and_save(
                    input_path=(
                        current_split_dir
                        / f"{role}.csv"
                    ),
                    output_path=(
                        split_output_dir
                        / f"{role}.csv"
                    ),
                    cell_line_column=cell_line_column,
                    drug_column=drug_column,
                    drugs=drugs,
                    label=f"{role}",
                )

                split_rows.append(
                    {
                        "tag": tag,
                        "split": split_name,
                        **record,
                    }
                )

                print(
                    f"{role:<11} rows "
                    f"{record['rows_before']:,} -> "
                    f"{record['rows_after']:,} | "
                    f"drugs {record['drugs_after']:,} | "
                    f"cell lines "
                    f"{record['cell_lines_after']:,}"
                )

            # ------------------------------------------------
            # Unseen cross-study datasets
            # ------------------------------------------------

            for name in info["cross_study"]:

                unseen_path = (
                    current_split_dir
                    / f"{name}_unseen.csv"
                )

                if not unseen_path.exists():
                    raise FileNotFoundError(
                        f"Unseen file not found: "
                        f"{unseen_path}\n"
                        f"Run the unseen cross-study script "
                        f"first."
                    )

                record = filter_and_save(
                    input_path=unseen_path,
                    output_path=(
                        split_output_dir
                        / f"{name}_unseen.csv"
                    ),
                    cell_line_column=cell_line_column,
                    drug_column=drug_column,
                    drugs=drugs,
                    label=f"{name}_unseen",
                )

                split_rows.append(
                    {
                        "tag": tag,
                        "split": split_name,
                        **record,
                    }
                )

                print(
                    f"{name + '_unseen':<11} rows "
                    f"{record['rows_before']:,} -> "
                    f"{record['rows_after']:,} | "
                    f"drugs {record['drugs_after']:,} | "
                    f"cell lines "
                    f"{record['cell_lines_after']:,}"
                )

    # --------------------------------------------------------
    # Save per-split summary and manifest
    # --------------------------------------------------------

    split_summary_path = (
        output_dir
        / "split_filtering_summary.csv"
    )

    pd.DataFrame(split_rows).to_csv(
        split_summary_path,
        index=False,
    )

    manifest = {
        "within_study_dataset": within_name,
        "cross_study_datasets": list(cross_coverage),
        "mode": args.mode,
        "min_cell_lines": args.min_cell_lines,
        "cell_line_column": cell_line_column,
        "drug_column": drug_column,
        "n_splits": len(split_directories),
        "shared_drug_sets": {
            tag: {
                "n_shared_drugs": len(info["drugs"]),
                "cross_study": info["cross_study"],
            }
            for tag, info in drug_sets.items()
        },
    }

    (
        output_dir
        / "drug_overlap_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DRUG-OVERLAP FILTERING COMPLETE")
    print("=" * 70)

    print(
        "Definition:\n"
        "Models are trained, tuned and evaluated only on drugs "
        "shared between datasets. Unseen cell lines are still "
        "defined by the within-study TRAINING set of each split."
    )

    print(
        f"\nCoverage summary:\n{coverage_path.resolve()}"
    )

    print(
        f"\nOutput directory:\n{output_dir.resolve()}"
    )


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":
    main()
