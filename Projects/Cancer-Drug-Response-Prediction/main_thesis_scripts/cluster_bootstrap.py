
#!/usr/bin/env python
"""Observation-level bootstrap CIs and forest plots for R2, RMSE and Pearson."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ALIASES = {
    "actual": ["actual_response", "actual"],
    "pred": ["predicted_response", "predicted"],
    "split": ["split", "lco_split"],
}

REQUIRED = ["dataset", "cellosaurus_id", "pubchem_id"]
KEY = ["dataset", "split", "cellosaurus_id", "pubchem_id"]
METRICS = ["R2", "RMSE", "Pearson"]


def clean(series):
    """Normalise identifiers and labels."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .fillna("<MISSING>")
    )


def load(label, path):
    """Load and validate one model's predictions."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        sys.exit(f"[{label}] Could not read {path}: {exc}")

    rename = {}

    for standard, candidates in ALIASES.items():
        found = [col for col in candidates if col in df.columns]

        if len(found) != 1:
            sys.exit(
                f"[{label}] Expected exactly one of {candidates}; "
                f"found {found}. Available columns: {list(df.columns)}"
            )

        rename[found[0]] = standard

    missing = [col for col in REQUIRED if col not in df.columns]
    if missing:
        sys.exit(f"[{label}] Missing required columns: {missing}")

    df = df.rename(columns=rename).copy()
    df = df[KEY + ["actual", "pred"]]

    for col in KEY:
        df[col] = clean(df[col])

    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")

    invalid = (
        df[["actual", "pred"]].isna().any(axis=1)
        | ~np.isfinite(df[["actual", "pred"]]).all(axis=1)
    )

    if invalid.any():
        print(
            f"[{label}] Dropping {int(invalid.sum())} "
            "rows with invalid responses."
        )
        df = df.loc[~invalid].copy()

    if df.empty:
        sys.exit(f"[{label}] No valid observations remain.")

    duplicate_mask = df.duplicated(KEY, keep=False)
    duplicate_keys = df.loc[duplicate_mask, KEY].drop_duplicates().shape[0]

    if duplicate_keys:
        print(
            f"[{label}] WARNING: {duplicate_keys} repeated keys "
            f"({int(duplicate_mask.sum())} rows involved). "
            "Rows will be retained and matched by occurrence."
        )

    # Order repeated observations by measured response.
    df = df.sort_values(
        KEY + ["actual", "pred"], kind="mergesort"
    ).reset_index(drop=True)

    df["_occurrence"] = df.groupby(KEY, dropna=False).cumcount()

    return df


def calculate_metrics(actual, predicted):
    """Calculate R2, RMSE and Pearson correlation."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) == 0:
        return np.array([np.nan, np.nan, np.nan])

    errors = actual - predicted
    rmse = np.sqrt(np.mean(errors**2))

    total_variation = np.sum((actual - np.mean(actual))**2)

    r2 = (
        1 - np.sum(errors**2) / total_variation
        if total_variation > 0
        else np.nan
    )

    if (
        len(actual) < 2
        or np.std(actual) == 0
        or np.std(predicted) == 0
    ):
        pearson = np.nan
    else:
        pearson = np.corrcoef(actual, predicted)[0, 1]

    return np.array([r2, rmse, pearson])


def get_units(reference, pool_splits=False):
    """Yield dataset/split evaluation groups."""
    for dataset, group in reference.groupby("dataset", sort=True):

        if pool_splits:
            lines_per_split = (
                group.groupby("cellosaurus_id")["split"].nunique()
            )

            if lines_per_split.max() == 1:
                yield dataset, "ALL", group.index.to_numpy()
                continue

            print(
                f"[{dataset}] Cell lines occur in multiple splits; "
                "keeping splits separate."
            )

        for split, subset in group.groupby("split", sort=True):
            yield dataset, split, subset.index.to_numpy()


def make_forest_plot(results, output_path):
    """Create a forest plot of model estimates and 95% bootstrap CIs."""
    plot_data = results[
        results["comparison"].fillna("").astype(str).str.strip() == ""
    ].copy()

    if plot_data.empty:
        print("No model estimates available for plotting.")
        return

    metric_titles = {
        "R2": r"$R^2$",
        "RMSE": "RMSE",
        "Pearson": "Pearson correlation",
    }

    plot_data["group"] = (
        plot_data["dataset"].astype(str)
        + " / split "
        + plot_data["split"].astype(str)
    )

    groups = list(dict.fromkeys(
        plot_data.sort_values(["dataset", "split"])["group"].tolist()
    ))

    models = list(dict.fromkeys(plot_data["model"].astype(str).tolist()))

    # Three panels, one per metric.
    rows_per_group = max(len(models), 1)
    group_gap = 1
    total_height = len(groups) * (rows_per_group + group_gap)

    fig, axes = plt.subplots(
        ncols=3,
        figsize=(18, max(6, 0.32 * total_height + 2)),
        squeeze=False,
    )
    axes = axes[0]

    for ax, metric in zip(axes, METRICS):
        subset = plot_data[plot_data["metric"] == metric]
        y_positions = []
        tick_labels = []
        position = total_height

        for group in groups:
            group_data = subset[subset["group"] == group]

            for model in models:
                row = group_data[group_data["model"] == model]

                if row.empty:
                    position -= 1
                    continue

                row = row.iloc[0]
                estimate = float(row["estimate"])
                low = float(row["ci_low"])
                high = float(row["ci_high"])

                if not np.isfinite([estimate, low, high]).all():
                    position -= 1
                    continue

                ax.errorbar(
                    estimate,
                    position,
                    xerr=[[
                        max(0, estimate - low)
                    ], [
                        max(0, high - estimate)
                    ]],
                    fmt="o",
                    capsize=3,
                    markersize=4,
                )

                y_positions.append(position)
                tick_labels.append(f"{group} | {model}")
                position -= 1

            position -= group_gap

        ax.set_title(metric_titles[metric])
        ax.set_xlabel("Estimate (95% CI)")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.grid(axis="x", alpha=0.25)

        if metric in ("R2", "Pearson"):
            ax.axvline(0, linestyle="--", linewidth=0.8)

    axes[0].set_ylabel("Dataset / split | model")

    fig.suptitle(
        "Model Performance with 95% Bootstrap Confidence Intervals",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Forest plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Observation-level bootstrap confidence intervals "
            "and forest plots."
        )
    )

    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        metavar="LABEL=PATH",
        help="Model prediction CSV files.",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="bootstrap_results.csv")
    parser.add_argument(
        "--plot",
        default=None,
        help="Forest plot output path; defaults to a PNG beside the CSV.",
    )
    parser.add_argument(
        "--pool-splits",
        action="store_true",
        help="Pool splits only when cell lines are disjoint.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Check inputs and observation matching without bootstrapping.",
    )

    args = parser.parse_args()

    if args.n_boot < 1:
        sys.exit("--n-boot must be at least 1.")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_path = Path(args.plot) if args.plot else output_path.with_name(
        output_path.stem + "_forest.png"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    dfs = {}

    for item in args.files:
        if "=" not in item:
            sys.exit(f"Invalid file argument {item!r}; use LABEL=PATH.")

        label, path = item.split("=", 1)

        if label in dfs:
            sys.exit(f"Duplicate model label: {label}")

        dfs[label] = load(label, path)

        print(f"\n[{label}] {len(dfs[label]):,} valid rows")
        print(
            dfs[label].groupby(["dataset", "split"]).agg(
                rows=("pred", "size"),
                cell_lines=("cellosaurus_id", "nunique"),
            ).to_string()
        )

    if not dfs:
        sys.exit("Provide at least one model file.")

    # Match the same dataset/split/cell-line/drug observations.
    match_key = KEY + ["_occurrence"]
    common = None

    for label, df in dfs.items():
        model_keys = df[match_key].drop_duplicates()

        common = (
            model_keys if common is None
            else common.merge(model_keys, on=match_key, how="inner")
        )

    if common is None or common.empty:
        sys.exit(
            "No shared observations across models. Check dataset/split "
            "labels and the prediction files."
        )

    for label in list(dfs):
        original_n = len(dfs[label])

        dfs[label] = (
            dfs[label]
            .merge(
                common,
                on=match_key,
                how="inner",
                validate="one_to_one",
            )
            .sort_values(match_key)
            .reset_index(drop=True)
        )

        print(
            f"[{label}] matched {len(dfs[label]):,} of "
            f"{original_n:,} rows across models"
        )

    labels = list(dfs)
    reference = dfs[labels[0]]

    # Check that all models use the same measured responses.
    for label in labels[1:]:
        differences = np.abs(
            dfs[label]["actual"].to_numpy()
            - reference["actual"].to_numpy()
        )
        max_difference = differences.max()

        if max_difference > 1e-6:
            sys.exit(
                f"[{label}] Actual responses differ from {labels[0]} "
                f"(maximum difference {max_difference:.6g}). "
                "Resolve observation matching before bootstrapping."
            )

    if args.inspect:
        print("\nInspection complete. No bootstrap was run.")
        return

    rng = np.random.default_rng(args.seed)
    output_rows = []

    for dataset, split, indices in get_units(
        reference, args.pool_splits
    ):
        n = len(indices)

        if n < 2:
            print(f"WARNING [{dataset}/{split}]: only {n} observation.")

        y = reference.loc[indices, "actual"].to_numpy(float)

        predictions = {
            label: dfs[label].loc[indices, "pred"].to_numpy(float)
            for label in labels
        }

        estimates = {
            label: calculate_metrics(y, predictions[label])
            for label in labels
        }

        boot = {
            label: np.full((args.n_boot, 3), np.nan)
            for label in labels
        }

        # Resample observations; use identical indices for every model.
        for b in range(args.n_boot):
            sample_idx = rng.integers(0, n, size=n)
            y_sample = y[sample_idx]

            for label in labels:
                boot[label][b] = calculate_metrics(
                    y_sample, predictions[label][sample_idx]
                )

        base = {
            "dataset": dataset,
            "split": split,
            "n_observations": n,
            "n_cell_lines": reference.loc[
                indices, "cellosaurus_id"
            ].nunique(),
            "bootstrap_replicates": args.n_boot,
        }

        # Individual model estimates and percentile confidence intervals.
        for label in labels:
            ci_low, ci_high = np.nanpercentile(
                boot[label], [2.5, 97.5], axis=0
            )

            for j, metric in enumerate(METRICS):
                output_rows.append({
                    **base,
                    "model": label,
                    "comparison": "",
                    "metric": metric,
                    "estimate": estimates[label][j],
                    "ci_low": ci_low[j],
                    "ci_high": ci_high[j],
                })

        # Paired metric differences relative to the first model.
        for label in labels[1:]:
            differences = boot[label] - boot[labels[0]]

            ci_low, ci_high = np.nanpercentile(
                differences, [2.5, 97.5], axis=0
            )

            for j, metric in enumerate(METRICS):
                output_rows.append({
                    **base,
                    "model": label,
                    "comparison": f"minus {labels[0]}",
                    "metric": metric,
                    "estimate": (
                        estimates[label][j] - estimates[labels[0]][j]
                    ),
                    "ci_low": ci_low[j],
                    "ci_high": ci_high[j],
                })

    results = pd.DataFrame(output_rows)
    results.to_csv(output_path, index=False)

    print(f"\nBootstrap results saved to: {output_path}")
    print(f"Bootstrap replicates: {args.n_boot}")
    print(results.round(4).to_string(index=False))

    make_forest_plot(results, plot_path)


if __name__ == "__main__":
    main()
