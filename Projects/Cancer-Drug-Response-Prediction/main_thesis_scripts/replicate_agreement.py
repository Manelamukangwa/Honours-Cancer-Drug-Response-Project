#!/usr/bin/env python3
"""
Assessment of Within-Study Replicate Variability and Cross-Study Agreement

Dataset-agnostic script for cleaned drug-response datasets.

For each dataset, it:
  1. Audits repeated cell-line–drug pairs.
  2. Summarises within-study replicate variability using the standard
     deviation, range, and number of distinct response values.
  3. Compares every pair of datasets using shared cell-line–drug pairs.
  4. Calculates Pearson and Spearman correlations and Bland–Altman summaries.

Input files and column names are supplied through the terminal.

Expected input format for each dataset:
  NAME=PATH,CELL_COLUMN,DRUG_COLUMN,RESPONSE_COLUMN

Example:
  python replicate_agreement.py \
    --dataset "CTRPv2=processed/CTRPv2_response_clean.csv,cell_line_name,pubchem_id,LN_IC50_curvecurator_normalised" \
    --dataset "GDSC2=processed/GDSC2_response_clean.csv,cell_line_name,pubchem_id,LN_IC50_curvecurator" \
    --output-dir results/replicate_agreement

Dependencies:
  pandas, numpy

Notes:
  - Rows with missing cell-line or drug identifiers are excluded from pair-level
    analyses. Rows with missing response values are excluded from numeric
    summaries and cross-study comparisons.
  - If the cleaned dataset has already had repeated cell-line–drug pairs removed,
    the within-study audit will report that no repeated pairs remain. It cannot
    recover replicate information removed during cleaning.
  - Cross-study matching uses exact cell-line and drug identifier matches.
"""

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_dataset_spec(spec):
    """Parse NAME=PATH,CELL_COLUMN,DRUG_COLUMN,RESPONSE_COLUMN."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            "Dataset must use NAME=PATH,CELL_COLUMN,DRUG_COLUMN,RESPONSE_COLUMN"
        )

    name, details = spec.split("=", 1)
    parts = details.rsplit(",", 3)

    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Dataset must use NAME=PATH,CELL_COLUMN,DRUG_COLUMN,RESPONSE_COLUMN. "
            "Quote the full argument if the path contains spaces."
        )

    path, cell_col, drug_col, response_col = [part.strip() for part in parts]
    if not all([name.strip(), path, cell_col, drug_col, response_col]):
        raise argparse.ArgumentTypeError("Dataset name, path, and columns cannot be empty.")

    return {
        "name": name.strip(),
        "path": Path(path),
        "cell_col": cell_col,
        "drug_col": drug_col,
        "response_col": response_col,
    }


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_dataset(spec):
    if not spec["path"].is_file():
        raise FileNotFoundError(f"Input file not found: {spec['path']}")

    df = pd.read_csv(spec["path"])
    required = [spec["cell_col"], spec["drug_col"], spec["response_col"]]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{spec['name']}: missing columns {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[required].copy()
    df.columns = ["cell_line", "drug", "response"]
    df["response"] = pd.to_numeric(df["response"], errors="coerce")
    df = df.dropna(subset=["cell_line", "drug"])
    df["cell_line"] = df["cell_line"].astype(str).str.strip()
    df["drug"] = df["drug"].astype(str).str.strip()
    df = df[(df["cell_line"] != "") & (df["drug"] != "")]
    df["dataset"] = spec["name"]
    return df


def within_study_audit(name, df, output_dir):
    pair_cols = ["cell_line", "drug"]
    counts = df.groupby(pair_cols, dropna=False).size().rename("row_count")
    repeated_keys = counts[counts > 1].reset_index()
    numeric = df.dropna(subset=["response"])

    if numeric.empty:
        summaries = pd.DataFrame(
            columns=pair_cols + [
                "row_count", "non_missing_response_count",
                "distinct_non_missing_responses", "mean_response",
                "sd_response", "min_response", "max_response",
                "response_range"
            ]
        )
    else:
        summaries = (
            numeric.groupby(pair_cols)["response"]
            .agg(
                non_missing_response_count="count",
                distinct_non_missing_responses="nunique",
                mean_response="mean",
                sd_response="std",
                min_response="min",
                max_response="max",
            )
            .reset_index()
        )
        summaries["response_range"] = summaries["max_response"] - summaries["min_response"]
        summaries = summaries.merge(
            counts.reset_index(), on=pair_cols, how="left"
        )
        summaries = summaries[
            summaries["row_count"] > 1
        ]

    if not repeated_keys.empty:
        repeated_rows = df.merge(repeated_keys, on=pair_cols, how="inner")
        repeated_rows = repeated_rows.sort_values(pair_cols)
    else:
        repeated_rows = pd.DataFrame(
            columns=["cell_line", "drug", "response", "dataset", "row_count"]
        )

    prefix = safe_name(name)
    repeated_keys.to_csv(output_dir / f"{prefix}_repeated_pairs.csv", index=False)
    summaries.to_csv(output_dir / f"{prefix}_within_study_replicate_summary.csv", index=False)
    repeated_rows.to_csv(output_dir / f"{prefix}_repeated_pair_rows.csv", index=False)

    nonmissing_pair_counts = numeric.groupby(pair_cols).size()
    report = {
        "dataset": name,
        "input_rows_after_identifier_filter": len(df),
        "rows_with_non_missing_response": int(df["response"].notna().sum()),
        "unique_cell_line_drug_pairs": int(counts.size),
        "pairs_with_multiple_rows": int((counts > 1).sum()),
        "pairs_with_multiple_non_missing_responses": int(
            (numeric.groupby(pair_cols)["response"].nunique() > 1).sum()
        ),
        "repeated_pairs_with_no_non_missing_response": int(
            sum(
                1 for _, group in df.groupby(pair_cols)
                if len(group) > 1 and group["response"].notna().sum() == 0
            )
        ),
    }

    if not summaries.empty:
        report["repeated_pairs_with_response_sd"] = int(summaries["sd_response"].notna().sum())
        report["median_within_pair_sd"] = summaries["sd_response"].median()
        report["median_within_pair_range"] = summaries["response_range"].median()
    else:
        report["repeated_pairs_with_response_sd"] = 0
        report["median_within_pair_sd"] = np.nan
        report["median_within_pair_range"] = np.nan

    return report


def compare_datasets(name_a, df_a, name_b, df_b, output_dir):
    pair_cols = ["cell_line", "drug"]
    a = df_a.dropna(subset=["response"]).groupby(pair_cols, as_index=False)["response"].mean()
    b = df_b.dropna(subset=["response"]).groupby(pair_cols, as_index=False)["response"].mean()
    a = a.rename(columns={"response": "response_a"})
    b = b.rename(columns={"response": "response_b"})

    shared = a.merge(b, on=pair_cols, how="inner")
    shared["difference_a_minus_b"] = shared["response_a"] - shared["response_b"]
    shared["mean_of_studies"] = (shared["response_a"] + shared["response_b"]) / 2

    n = len(shared)
    if n >= 2:
        pearson = shared["response_a"].corr(shared["response_b"], method="pearson")
        spearman = shared["response_a"].corr(shared["response_b"], method="spearman")
        mean_diff = shared["difference_a_minus_b"].mean()
        sd_diff = shared["difference_a_minus_b"].std(ddof=1)
        lower_loa = mean_diff - 1.96 * sd_diff if pd.notna(sd_diff) else np.nan
        upper_loa = mean_diff + 1.96 * sd_diff if pd.notna(sd_diff) else np.nan
        mae = shared["difference_a_minus_b"].abs().mean()
        rmse = np.sqrt((shared["difference_a_minus_b"] ** 2).mean())
    else:
        pearson = spearman = mean_diff = sd_diff = lower_loa = upper_loa = mae = rmse = np.nan

    pair_label = f"{safe_name(name_a)}_vs_{safe_name(name_b)}"
    shared.to_csv(output_dir / f"{pair_label}_shared_pairs.csv", index=False)

    summary = {
        "dataset_a": name_a,
        "dataset_b": name_b,
        "shared_cell_line_drug_pairs": n,
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "mean_difference_a_minus_b": mean_diff,
        "sd_of_differences": sd_diff,
        "lower_95_percent_limits_of_agreement": lower_loa,
        "upper_95_percent_limits_of_agreement": upper_loa,
        "mean_absolute_difference": mae,
        "root_mean_squared_difference": rmse,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Audit within-study replicate variability and cross-study agreement."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset_spec,
        metavar="NAME=PATH,CELL_COL,DRUG_COL,RESPONSE_COL",
        help="Dataset specification; repeat this argument for each dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/replicate_agreement",
        help="Directory for output CSV files (default: results/replicate_agreement).",
    )
    args = parser.parse_args()

    if len(args.dataset) < 1:
        parser.error("Provide at least one --dataset argument.")

    names = [item["name"] for item in args.dataset]
    if len(names) != len(set(names)):
        parser.error("Dataset names must be unique.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    within_reports = []

    for spec in args.dataset:
        df = load_dataset(spec)
        loaded[spec["name"]] = df
        report = within_study_audit(spec["name"], df, output_dir)
        within_reports.append(report)

    within_summary = pd.DataFrame(within_reports)
    within_summary.to_csv(output_dir / "within_study_audit_summary.csv", index=False)

    agreement_reports = []
    for name_a, name_b in itertools.combinations(loaded.keys(), 2):
        agreement_reports.append(
            compare_datasets(name_a, loaded[name_a], name_b, loaded[name_b], output_dir)
        )

    agreement_summary = pd.DataFrame(agreement_reports)
    agreement_summary.to_csv(output_dir / "cross_study_agreement_summary.csv", index=False)

    print("\nWithin-study replicate audit")
    print("=" * 72)
    print(within_summary.to_string(index=False))
    print("\nCross-study agreement benchmark")
    print("=" * 72)
    if agreement_summary.empty:
        print("Only one dataset was supplied; no cross-study comparison was performed.")
    else:
        print(agreement_summary.to_string(index=False))
    print(f"\nOutput files saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
