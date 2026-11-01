
"""
Compare response measurements between CTRPv2 and cross-study datasets.

Matches:
    cellosaurus_id + pubchem_id

Responses analysed separately:
    AUC_curvecurator
    LN_IC50_curvecurator

Outputs:
    - Agreement summary CSV
    - Matched paired measurements CSV
    - Bland-Altman plots for each dataset and response

Difference = cross-study measurement - CTRPv2 measurement.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IDENTIFIERS = ["cellosaurus_id", "pubchem_id"]

RESPONSES = [
    "AUC_curvecurator",
    "LN_IC50_curvecurator",
]

DEFAULT_OUTPUT_DIR = (
    "/Users/manellamukangwa/Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction/local_scripts/"
    "dataset_agreement/results/Bland-Altman plots"
)


def load_dataset(label, file_path):
    """Load the required columns and resolve duplicate combinations."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    # Read identifiers as strings to preserve their exact representation.
    df = pd.read_csv(
        path,
        usecols=lambda col: col in IDENTIFIERS + RESPONSES,
        dtype={
            "cellosaurus_id": "string",
            "pubchem_id": "string",
        },
        low_memory=False,
    )

    required = IDENTIFIERS + RESPONSES
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")

    for col in IDENTIFIERS:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace("", pd.NA)

    for col in RESPONSES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(
            [np.inf, -np.inf], np.nan
        )

    df = df.dropna(subset=IDENTIFIERS).copy()

    records = []
    conflict_counts = {response: 0 for response in RESPONSES}

    for keys, group in df.groupby(IDENTIFIERS, sort=False):
        record = dict(zip(IDENTIFIERS, keys))

        for response in RESPONSES:
            values = group[response].dropna().unique()

            if len(values) > 1:
                # Do not arbitrarily select one conflicting value.
                conflict_counts[response] += 1
                record[response] = np.nan
            elif len(values) == 1:
                record[response] = values[0]
            else:
                record[response] = np.nan

        records.append(record)

    result = pd.DataFrame(
        records,
        columns=IDENTIFIERS + RESPONSES,
    )

    print(f"\n{label}")
    print(f"  Input rows: {len(df):,}")
    print(f"  Unique combinations: {len(result):,}")

    for response in RESPONSES:
        print(
            f"  Conflicting {response} combinations: "
            f"{conflict_counts[response]:,}"
        )

    return result


def calculate_agreement(valid, reference_col, study_col):
    """Calculate paired agreement statistics for one response."""

    reference = valid[reference_col].to_numpy(dtype=float)
    study = valid[study_col].to_numpy(dtype=float)

    differences = study - reference
    absolute_differences = np.abs(differences)

    n = len(differences)
    mean_difference = float(np.mean(differences))
    sd_difference = (
        float(np.std(differences, ddof=1)) if n >= 2 else np.nan
    )

    # Conventional 95% limits of agreement.
    # These are descriptive estimates; their interpretation depends
    # on the distribution and independence of paired differences.
    lower_loa = (
        mean_difference - 1.96 * sd_difference
        if n >= 2 else np.nan
    )
    upper_loa = (
        mean_difference + 1.96 * sd_difference
        if n >= 2 else np.nan
    )

    return {
        "Valid pairs": n,
        "Mean difference": mean_difference,
        "MAE": float(np.mean(absolute_differences)),
        "RMSE": float(np.sqrt(np.mean(differences ** 2))),
        "SD of differences": sd_difference,
        "Lower 95% LoA": lower_loa,
        "Upper 95% LoA": upper_loa,
    }


def make_bland_altman_plot(
    valid,
    reference_col,
    study_col,
    dataset_name,
    response,
    output_dir,
):
    """Create and save one Bland-Altman plot."""

    reference = valid[reference_col].to_numpy(dtype=float)
    study = valid[study_col].to_numpy(dtype=float)

    # X-axis: average of the two measurements.
    averages = (reference + study) / 2

    # Y-axis: paired difference.
    differences = study - reference

    mean_difference = float(np.mean(differences))
    sd_difference = (
        float(np.std(differences, ddof=1))
        if len(differences) >= 2 else np.nan
    )

    lower_loa = (
        mean_difference - 1.96 * sd_difference
        if len(differences) >= 2 else np.nan
    )
    upper_loa = (
        mean_difference + 1.96 * sd_difference
        if len(differences) >= 2 else np.nan
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        averages,
        differences,
        alpha=0.30,
        s=12,
        edgecolors="none",
    )

    ax.axhline(
        mean_difference,
        linestyle="-",
        linewidth=2,
        label=f"Mean difference = {mean_difference:.4f}",
    )

    if len(differences) >= 2:
        ax.axhline(
            upper_loa,
            linestyle="--",
            linewidth=1.5,
            label=f"Upper 95% LoA = {upper_loa:.4f}",
        )
        ax.axhline(
            lower_loa,
            linestyle="--",
            linewidth=1.5,
            label=f"Lower 95% LoA = {lower_loa:.4f}",
        )

    ax.axhline(0, color="grey", linewidth=1, alpha=0.8)

    ax.set_title(
        f"Bland-Altman Plot: {dataset_name} vs CTRPv2\n"
        f"{response} (n = {len(valid):,})"
    )
    ax.set_xlabel("Mean of CTRPv2 and cross-study measurements")
    ax.set_ylabel("Cross-study measurement − CTRPv2 measurement")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()

    safe_dataset = re.sub(r"[^A-Za-z0-9_-]+", "_", dataset_name)
    filename = f"Bland_Altman_{safe_dataset}_{response}.png"

    plot_path = output_dir / filename

    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plot saved: {plot_path}")


def compare_dataset(reference, study, study_name, output_dir):
    """Match combinations and assess each response independently."""

    merged = reference.merge(
        study,
        on=IDENTIFIERS,
        how="inner",
        suffixes=("_ref", "_study"),
        validate="one_to_one",
    )

    results = []
    paired_records = []

    print(f"\nComparing CTRPv2 with {study_name}")
    print(f"  Shared combinations: {len(merged):,}")

    for response in RESPONSES:
        reference_col = f"{response}_ref"
        study_col = f"{response}_study"

        valid = merged[
            [*IDENTIFIERS, reference_col, study_col]
        ].dropna(subset=[reference_col, study_col]).copy()

        # Exclude non-finite measurements if any remain.
        finite_mask = (
            np.isfinite(valid[reference_col].to_numpy(dtype=float))
            & np.isfinite(valid[study_col].to_numpy(dtype=float))
        )
        valid = valid.loc[finite_mask].copy()

        stats = calculate_agreement(
            valid, reference_col, study_col
        )

        results.append({
            "Dataset": study_name,
            "Response": response,
            "Shared combinations": len(merged),
            **stats,
        })

        print(f"  {response}: {len(valid):,} valid pairs")

        if not valid.empty:
            pairs = valid[IDENTIFIERS].copy()
            pairs["Dataset"] = study_name
            pairs["Response"] = response
            pairs["CTRPv2_value"] = valid[reference_col].to_numpy()
            pairs["Cross_study_value"] = valid[study_col].to_numpy()
            pairs["Mean_measurement"] = (
                pairs["CTRPv2_value"] + pairs["Cross_study_value"]
            ) / 2
            pairs["Difference"] = (
                pairs["Cross_study_value"] - pairs["CTRPv2_value"]
            )
            paired_records.append(pairs)

            make_bland_altman_plot(
                valid,
                reference_col,
                study_col,
                study_name,
                response,
                output_dir,
            )
        else:
            print("  No valid pairs; plot not generated.")

    return results, paired_records


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate cross-study agreement and generate "
            "Bland-Altman plots against CTRPv2."
        )
    )

    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset paths in LABEL=FILE.csv format; include CTRPv2.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for plots and result CSV files.",
    )

    args = parser.parse_args()

    dataset_paths = {}

    for item in args.datasets:
        if "=" not in item:
            parser.error(f"Invalid input: {item}. Use LABEL=FILE.csv.")

        label, file_path = item.split("=", 1)

        if label in dataset_paths:
            parser.error(f"Duplicate dataset label: {label}")

        dataset_paths[label] = file_path

    if "CTRPv2" not in dataset_paths:
        parser.error("A dataset labelled CTRPv2 is required.")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        label: load_dataset(label, path)
        for label, path in dataset_paths.items()
    }

    reference = datasets["CTRPv2"]
    all_results = []
    all_pairs = []

    for study_name, study in datasets.items():
        if study_name == "CTRPv2":
            continue

        results, paired_records = compare_dataset(
            reference,
            study,
            study_name,
            output_dir,
        )

        all_results.extend(results)
        all_pairs.extend(paired_records)

    summary = pd.DataFrame(all_results)

    summary_path = output_dir / "agreement_summary.csv"
    summary.to_csv(summary_path, index=False)

    if all_pairs:
        paired_data = pd.concat(all_pairs, ignore_index=True)
    else:
        paired_data = pd.DataFrame()

    paired_path = output_dir / "paired_measurements.csv"
    paired_data.to_csv(paired_path, index=False)

    print("\n" + "=" * 120)
    print("CROSS-STUDY AGREEMENT SUMMARY")
    print("=" * 120)

    if not summary.empty:
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 240,
            "display.float_format", "{:.5f}".format,
        ):
            print(summary.to_string(index=False))

    print("\nOutput files:")
    print(f"  Summary CSV: {summary_path}")
    print(f"  Paired measurements CSV: {paired_path}")
    print(f"  Bland-Altman plots: {output_dir}")

    print("\nInterpretation notes:")
    print("- Differences are cross-study values minus CTRPv2 values.")
    print("- Positive bias means higher values in the cross-study dataset.")
    print("- MAE and RMSE describe the size of paired differences.")
    print("- Limits of agreement are mean difference ± 1.96 SD.")
    print("- Limits of agreement assume the differences are suitably")
    print("  distributed; inspect the plots before interpreting them.")
    print("- These are descriptive agreement estimates, not an agreement")
    print("  ceiling or a predefined threshold of acceptable agreement.")


if __name__ == "__main__":
    main()
