
#!/usr/bin/env python3
"""
Plot smooth LNIC50_curvecurator response distributions for:
CTRPv1, CTRPv2, CCLE, GDSC1, and GDSC2.

Usage:
    python distribution_plot.py /path/to/data/raw

Expected files may be organised as:
    data/raw/CTRPv1/CTRPv1.csv
    data/raw/CTRPv2/CTRPv2.csv
    data/raw/CCLE/CCLE.csv
    data/raw/GDSC1/GDSC1.csv
    data/raw/GDSC2/GDSC2.csv

Alternatively, CSV files can be directly inside data/raw.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATASETS = ["CTRPv1", "CTRPv2", "CCLE", "GDSC1", "GDSC2"]

PALETTE = {
    "CTRPv1": "#4477AA",
    "CTRPv2": "#EE6677",
    "CCLE": "#228833",
    "GDSC1": "#CCBB44",
    "GDSC2": "#AA3377",
}

# Higher values produce smoother curves.
BANDWIDTH_ADJUST = 1.8


def find_dataset_file(data_root, dataset):
    """Find a dataset CSV in its folder or the supplied root folder."""

    candidates = [
        data_root / dataset / f"{dataset}.csv",
        data_root / f"{dataset}.csv",
    ]

    for path in candidates:
        if path.is_file():
            return path

    # Case-insensitive fallback for filename differences.
    matches = [
        path
        for path in data_root.rglob("*.csv")
        if path.stem.lower() == f"{dataset.lower()}"
    ]

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {dataset}.csv under {data_root}"
    )


def load_dataset(data_root, dataset):
    """Load valid curated LN_IC50 measurements."""

    csv_path = find_dataset_file(data_root, dataset)
    df = pd.read_csv(csv_path, low_memory=False)

    possible_columns = [
        "LN_IC50_curvecurator",
        "LNIC50_curvecurator",
    ]

    response_column = next(
        (col for col in possible_columns if col in df.columns),
        None,
    )

    if response_column is None:
        raise ValueError(
            f"{dataset}: curated response column not found.\n"
            f"Expected one of: {possible_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    values = pd.to_numeric(
        df[response_column], errors="coerce"
    ).to_numpy(dtype=float)

    # Remove missing and infinite measurements.
    values = values[np.isfinite(values)]

    if len(values) < 2:
        raise ValueError(
            f"{dataset}: fewer than two valid measurements."
        )

    print(
        f"{dataset}: {len(values):,} valid measurements "
        f"(column: {response_column})"
    )

    return values


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot smooth curated LN_IC50 distributions "
            "for five cancer drug-response datasets."
        )
    )

    parser.add_argument(
        "data_root",
        type=Path,
        help="Path to the parent raw-data directory.",
    )

    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()

    if not data_root.is_dir():
        parser.error(f"Data directory does not exist: {data_root}")

    sns.set_theme(
        style="white",
        context="talk",
        font="sans-serif",
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    for dataset in DATASETS:
        values = load_dataset(data_root, dataset)

        sns.kdeplot(
            x=values,
            color=PALETTE[dataset],
            label=f"{dataset} (n={len(values):,})",
            linewidth=2.8,
            fill=False,
            bw_adjust=BANDWIDTH_ADJUST,
            cut=0,
            common_norm=False,
            warn_singular=True,
            ax=ax,
        )

    ax.set_title(
        "Drug Response Measurement Distributions Across Datasets",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    ax.set_xlabel(
        r"$\ln(IC_{50})$ (curve-curated)",
        fontsize=14,
        labelpad=12,
    )

    ax.set_ylabel(
        "Probability density",
        fontsize=14,
        labelpad=12,
    )

    ax.legend(
        title="Dataset",
        fontsize=10,
        title_fontsize=11,
        frameon=False,
        loc="best",
    )

    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.grid(axis="x", visible=False)

    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")

    sns.despine(ax=ax)
    fig.tight_layout()

    output_path = (
        data_root / "LNIC50_distribution_all_datasets.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"\nPlot saved to:\n{output_path}")

    plt.show()


if __name__ == "__main__":
    main()
