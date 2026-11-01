
"""
Compare raw and cell-line-held-out calibrated agreement for
LN_IC50_curvecurator.

CTRPv2 is the reference dataset. Calibration is fitted using matched
measurements from CTRPv2 training cell lines and the full cross-study
dataset. Evaluation uses the corresponding LCO split's *_unseen.csv file.

Calibration is a practical benchmark, not a definitive measurement ceiling.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CELL = "cellosaurus_id"
CELL_NAME = "cell_line_name"
DRUG = "pubchem_id"
RESPONSE = "LN_IC50_curvecurator"
KEYS = [CELL, DRUG]
N_SPLITS = 5


def load_dataset(path):
    """Load measurements and resolve repeated cell-line–drug combinations."""
    df = pd.read_csv(
        path,
        usecols=KEYS + [RESPONSE],
        dtype={CELL: "string", DRUG: "string"},
    )

    df[RESPONSE] = pd.to_numeric(df[RESPONSE], errors="coerce")
    df = df.dropna(subset=KEYS).drop_duplicates()

    def resolve(values):
        values = values.dropna().unique()
        return values[0] if len(values) == 1 else np.nan

    return (
        df.groupby(KEYS, as_index=False)[RESPONSE]
        .agg(resolve)
        .dropna(subset=[RESPONSE])
    )


def load_cell_identifiers(path):
    """Read cell-line identifiers for leakage checks."""
    df = pd.read_csv(
        path,
        usecols=[CELL, CELL_NAME],
        dtype={CELL: "string", CELL_NAME: "string"},
    )
    return df.dropna(subset=[CELL, CELL_NAME])


def calculate_metrics(reference, predictions):
    """Calculate agreement metrics against CTRPv2."""
    difference = predictions - reference

    return {
        "R2": r2_score(reference, predictions),
        "MAE": mean_absolute_error(reference, predictions),
        "RMSE": np.sqrt(mean_squared_error(reference, predictions)),
        "mean_difference": difference.mean(),
    }


def match_measurements(reference, cross_study):
    """Match the same cell-line–drug combinations across datasets."""
    return (
        reference.merge(
            cross_study,
            on=KEYS,
            how="inner",
            suffixes=("_reference", "_cross_study"),
        )
        .dropna(
            subset=[
                f"{RESPONSE}_reference",
                f"{RESPONSE}_cross_study",
            ]
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate raw and calibrated LN_IC50_curvecurator agreement "
            "using five LCO splits."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        metavar="LABEL=FILE.csv",
        help="Full dataset files, including CTRPv2=FILE.csv.",
    )
    parser.add_argument(
        "--lco-dir",
        required=True,
        help="Directory containing split_1 through split_5.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paths = {}

    for item in args.datasets:
        if "=" not in item:
            parser.error(f"Invalid input: {item}. Use LABEL=FILE.csv.")

        label, filename = item.split("=", 1)

        if not label or not filename or label in paths:
            parser.error(f"Invalid or repeated dataset input: {item}")

        if not Path(filename).is_file():
            parser.error(f"File not found: {filename}")

        paths[label] = filename

    if "CTRPv2" not in paths or len(paths) < 2:
        parser.error("Supply CTRPv2 and at least one cross-study dataset.")

    lco_dir = Path(args.lco_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_number in range(1, N_SPLITS + 1):
        if not (lco_dir / f"split_{split_number}" / "train.csv").is_file():
            parser.error(f"Missing train.csv for split_{split_number}.")

    reference_all = load_dataset(paths["CTRPv2"])
    summaries = []
    paired_results = []

    for split_number in range(1, N_SPLITS + 1):
        split_dir = lco_dir / f"split_{split_number}"
        train_file = split_dir / "train.csv"
        train_reference = load_dataset(train_file)

        train_ids = load_cell_identifiers(train_file)

        for label, filename in paths.items():
            if label == "CTRPv2":
                continue

            unseen_file = split_dir / f"{label}_unseen.csv"
            if not unseen_file.is_file():
                raise FileNotFoundError(
                    f"Missing cross-study unseen file: {unseen_file}"
                )

            cross_study_all = load_dataset(filename)
            cross_study_unseen = load_dataset(unseen_file)
            unseen_ids = load_cell_identifiers(unseen_file)

            # Ensure calibration training and evaluation use separate
            # cell lines, including when identifiers have aliases.
            overlapping_names = set(train_ids[CELL_NAME]) & set(
                unseen_ids[CELL_NAME]
            )
            overlapping_ids = set(train_ids[CELL]) & set(unseen_ids[CELL])

            if overlapping_names or overlapping_ids:
                raise ValueError(
                    f"Cell-line overlap between training and unseen data "
                    f"in split_{split_number} for {label}. "
                    f"Overlapping names: {len(overlapping_names)}; "
                    f"overlapping Cellosaurus IDs: {len(overlapping_ids)}."
                )

            # Fit calibration using CTRPv2 training measurements only.
            calibration_train = match_measurements(
                train_reference, cross_study_all
            )

            # Evaluate only on the designated cross-study unseen file.
            evaluation = match_measurements(
                reference_all, cross_study_unseen
            )

            if len(calibration_train) < 5:
                raise ValueError(
                    f"Too few calibration pairs for {label}, "
                    f"split_{split_number}: {len(calibration_train)}."
                )

            if len(evaluation) < 5:
                raise ValueError(
                    f"Too few evaluation pairs for {label}, "
                    f"split_{split_number}: {len(evaluation)}."
                )

            x_train = calibration_train[
                f"{RESPONSE}_cross_study"
            ].to_numpy(dtype=float)
            y_train = calibration_train[
                f"{RESPONSE}_reference"
            ].to_numpy(dtype=float)

            x_test = evaluation[
                f"{RESPONSE}_cross_study"
            ].to_numpy(dtype=float)
            y_test = evaluation[
                f"{RESPONSE}_reference"
            ].to_numpy(dtype=float)

            model = LinearRegression()
            model.fit(x_train.reshape(-1, 1), y_train)

            calibrated_predictions = model.predict(
                x_test.reshape(-1, 1)
            )

            raw = calculate_metrics(y_test, x_test)
            calibrated = calculate_metrics(y_test, calibrated_predictions)

            summaries.append({
                "split": split_number,
                "dataset": label,
                "response": RESPONSE,
                "n_calibration_pairs": len(calibration_train),
                "n_evaluation_pairs": len(evaluation),
                **{f"raw_{k}": v for k, v in raw.items()},
                **{
                    f"calibrated_{k}": v
                    for k, v in calibrated.items()
                },
            })

            paired_results.append(pd.DataFrame({
                "split": split_number,
                "dataset": label,
                CELL: evaluation[CELL].to_numpy(),
                DRUG: evaluation[DRUG].to_numpy(),
                "CTRPv2_LN_IC50_curvecurator": y_test,
                "cross_study_LN_IC50_curvecurator": x_test,
                "calibrated_prediction": calibrated_predictions,
            }))

            print(
                f"Split {split_number} | {label} | "
                f"Calibration pairs: {len(calibration_train):,} | "
                f"Evaluation pairs: {len(evaluation):,} | "
                f"Raw MAE: {raw['MAE']:.4f} | "
                f"Calibrated MAE: {calibrated['MAE']:.4f}"
            )

    split_summary = pd.DataFrame(summaries)

    metric_columns = [
        "raw_R2", "raw_MAE", "raw_RMSE", "raw_mean_difference",
        "calibrated_R2", "calibrated_MAE", "calibrated_RMSE",
        "calibrated_mean_difference",
    ]

    # Summarise the five split-level results, giving each split equal weight.
    aggregate_summary = (
        split_summary.groupby(["dataset", "response"])[metric_columns]
        .agg(["mean", "std"])
    )
    aggregate_summary.columns = [
        f"{metric}_{stat}" for metric, stat in aggregate_summary.columns
    ]
    aggregate_summary = aggregate_summary.reset_index()

    split_file = output_dir / "ln_ic50_calibration_by_split.csv"
    summary_file = output_dir / "ln_ic50_calibration_summary.csv"
    pairs_file = output_dir / "ln_ic50_calibration_held_out_pairs.csv"

    split_summary.to_csv(split_file, index=False)
    aggregate_summary.to_csv(summary_file, index=False)
    pd.concat(paired_results, ignore_index=True).to_csv(
        pairs_file, index=False
    )

    print("\nCalibration is a practical benchmark, not a true measurement ceiling.")
    print(f"Split-level results: {split_file}")
    print(f"Mean ± SD summary: {summary_file}")
    print(f"Evaluation pairs: {pairs_file}")


if __name__ == "__main__":
    main()
