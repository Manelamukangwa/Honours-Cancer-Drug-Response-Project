"""
EDA Script: GDSC2 vs CTRPv2 Raw Drug-Response Data
------------------------------------------------------------------
Purpose:
    1. Check whether the raw data is clean and structured as expected
    2. Describe the structure of each dataset
    3. Describe the raw response variable
    4. Compare raw GDSC2 and CTRPv2 response distributions
    5. Identify unique cell-line/drug pairs shared by both datasets
    6. Compare response values for shared cell-line/drug pairs

This script does NOT clean, transform, or modify the raw datasets.
It does NOT run machine learning, PCA, clustering, or formal
statistical/hypothesis tests.

All outputs are saved to:
    results/EDA/gdsc2_ctrpv2/
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIG — everything you might need to change lives here
# ============================================================

Data_root = Path(
    "/Users/manellamukangwa/"
    "Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction/data"
)


INPUT_PATHS = {
    "GDSC2": Data_root / "raw/GDSC2/GDSC2.csv",
    "CTRPv2": Data_root / "raw/CTRPv2/CTRPv2.csv",
}


# All EDA results are saved here
OUTPUT_DIR = (
    Data_root.parent
    / "results/EDA/gdsc2_ctrpv2"
)


MANUAL_COLUMNS = {
    "GDSC2": {
        "cell_line_col": "cellosaurus_id",
        "drug_col": "pubchem_id",
        "response_col": "LN_IC50_curvecurator",
    },

    "CTRPv2": {
        "cell_line_col": "cellosaurus_id",
        "drug_col": "pubchem_id",
        "response_col": "LN_IC50_curvecurator",
    },
}


# ============================================================
# SECTION 1: Structure & data-quality checks
# ============================================================

def structure_and_quality_report(
    df,
    name,
    cell_col,
    drug_col
):

    lines = [f"=== {name} ==="]

    lines.append(
        f"Rows: {df.shape[0]}"
    )

    lines.append(
        f"Columns: {df.shape[1]}"
    )

    lines.append(
        f"Cell-line column: {cell_col}"
    )

    lines.append(
        f"Drug column: {drug_col}"
    )

    lines.append(
        f"Unique cell lines: {df[cell_col].nunique()}"
    )

    lines.append(
        f"Unique drugs: {df[drug_col].nunique()}"
    )

    # Check cell-line and drug identifiers
    for col in [cell_col, drug_col]:

        series = df[col]

        if series.dtype == object:

            blanks = (
                series
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            )

            lines.append(
                f"'{col}': blank/whitespace-only entries: {blanks}"
            )

        type_counts = (
            series
            .dropna()
            .map(type)
            .value_counts()
        )

        if len(type_counts) > 1:

            lines.append(
                f"'{col}': WARNING - mixed data types found: "
                f"{dict(type_counts)}"
            )

        elif len(type_counts) == 1:

            lines.append(
                f"'{col}': data type consistent "
                f"({type_counts.index[0].__name__})"
            )

        else:

            lines.append(
                f"'{col}': no non-missing values available "
                f"to check data type"
            )

    return "\n".join(lines)


# ============================================================
# SECTION 2: Duplicate report
# ============================================================

def duplicate_report(
    df,
    name,
    cell_col,
    drug_col
):

    n_full_dupes = (
        df
        .duplicated(keep=False)
        .sum()
    )

    n_pair_dupes = (
        df
        .duplicated(
            subset=[
                cell_col,
                drug_col
            ],
            keep=False
        )
        .sum()
    )

    n_unique_pairs = (
        df[
            [cell_col, drug_col]
        ]
        .dropna()
        .drop_duplicates()
        .shape[0]
    )

    return (
        f"=== {name} ===\n"
        f"Fully duplicated rows: {n_full_dupes}\n"
        f"Rows involved in duplicated "
        f"(cell line, drug) pairs: {n_pair_dupes}\n"
        f"Unique cell-line/drug pairs: {n_unique_pairs}"
    )


# ============================================================
# SECTION 3: Response variable — descriptive statistics
# ============================================================

def response_descriptive_stats(
    df,
    response_col
):

    raw = df[response_col]

    n_missing = raw.isna().sum()

    n_infinite = np.isinf(raw).sum()

    finite = (
        raw
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    n_negative = (
        finite < 0
    ).sum()

    q1 = finite.quantile(0.25)

    q3 = finite.quantile(0.75)

    iqr = q3 - q1

    # IQR rule is only used to flag potential extreme values.
    lower_fence = q1 - 1.5 * iqr

    upper_fence = q3 + 1.5 * iqr

    n_outliers = (
        (finite < lower_fence)
        |
        (finite > upper_fence)
    ).sum()

    return {
        "Number of observations": len(finite),
        "Minimum": finite.min(),
        "Maximum": finite.max(),
        "Mean": finite.mean(),
        "Median": finite.median(),
        "Standard deviation": finite.std(),
        "Q1": q1,
        "Q3": q3,
        "Missing values": n_missing,
        "Infinite values": n_infinite,
        "Negative values": n_negative,
        "Outlier count (IQR rule)": n_outliers,
    }


# ============================================================
# SECTION 4: Combined raw histogram
# ============================================================

def plot_combined_histogram(
    gdsc2_values,
    ctrpv2_values,
    output_path
):

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        gdsc2_values,
        bins=50,
        alpha=0.5,
        label="GDSC2"
    )

    plt.hist(
        ctrpv2_values,
        bins=50,
        alpha=0.5,
        label="CTRPv2"
    )

    plt.title(
        "Raw Drug-Response Distribution: "
        "GDSC2 vs CTRPv2"
    )

    plt.xlabel(
        "LN_IC50_curvecurator"
    )

    plt.ylabel(
        "Number of observations"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SECTION 5: Prepare cell-line/drug pairs
# ============================================================

def prepare_unique_pairs(
    df,
    cell_col,
    drug_col,
    response_col
):

    pair_df = (
        df[
            [
                cell_col,
                drug_col,
                response_col
            ]
        ]
        .copy()
    )

    pair_df = pair_df.dropna(
        subset=[
            cell_col,
            drug_col,
            response_col
        ]
    )

    # Remove infinite response values
    pair_df = pair_df[
        np.isfinite(
            pair_df[response_col]
        )
    ]

    return pair_df


# ============================================================
# SECTION 6: Find unique overlapping cell-line/drug pairs
# ============================================================

def find_overlapping_pairs(
    gdsc2_df,
    gdsc2_cell,
    gdsc2_drug,
    gdsc2_resp,
    ctrpv2_df,
    ctrpv2_cell,
    ctrpv2_drug,
    ctrpv2_resp
):

    """
    Identify cell-line/drug pairs present in both datasets.
    Repeated measurements are not averaged.
    """

    g = prepare_unique_pairs(
        gdsc2_df,
        gdsc2_cell,
        gdsc2_drug,
        gdsc2_resp
    )

    c = prepare_unique_pairs(
        ctrpv2_df,
        ctrpv2_cell,
        ctrpv2_drug,
        ctrpv2_resp
    )

    g.columns = [
        "cell_line",
        "drug",
        "GDSC2_response"
    ]

    c.columns = [
        "cell_line",
        "drug",
        "CTRPv2_response"
    ]

    merged = pd.merge(
        g,
        c,
        on=[
            "cell_line",
            "drug"
        ],
        how="inner"
    )

    return merged


# ============================================================
# SECTION 7: Overlap scatter plot
# ============================================================

def plot_overlap_scatter(
    merged_df,
    output_path
):

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        merged_df["GDSC2_response"],
        merged_df["CTRPv2_response"],
        alpha=0.5,
        s=15
    )

    lims = [
        min(
            merged_df["GDSC2_response"].min(),
            merged_df["CTRPv2_response"].min()
        ),

        max(
            merged_df["GDSC2_response"].max(),
            merged_df["CTRPv2_response"].max()
        )
    ]

    plt.plot(
        lims,
        lims,
        linestyle="--",
        label="y = x (perfect agreement)"
    )

    plt.title(
        "GDSC2 vs CTRPv2 Response — "
        "Overlapping Cell-Line/Drug Pairs"
    )

    plt.xlabel(
        "GDSC2 LN_IC50_curvecurator"
    )

    plt.ylabel(
        "CTRPv2 LN_IC50_curvecurator"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SECTION 8: Overlap counts bar chart
# ============================================================

def plot_overlap_counts(
    gdsc2_pair_count,
    ctrpv2_pair_count,
    overlap_pair_count,
    output_path
):

    gdsc2_only = (
        gdsc2_pair_count
        - overlap_pair_count
    )

    ctrpv2_only = (
        ctrpv2_pair_count
        - overlap_pair_count
    )

    labels = [
        "GDSC2 only",
        "Shared",
        "CTRPv2 only"
    ]

    values = [
        gdsc2_only,
        overlap_pair_count,
        ctrpv2_only
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        labels,
        values
    )

    plt.title(
        "Unique Cell-Line/Drug Pair Overlap"
    )

    plt.ylabel(
        "Number of unique cell-line/drug pairs"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # ========================================================
    # Load raw datasets
    # ========================================================

    print("Loading raw GDSC2...")

    gdsc2_df = pd.read_csv(
        INPUT_PATHS["GDSC2"],
        low_memory=False
    )

    print("Loading raw CTRPv2...")

    ctrpv2_df = pd.read_csv(
        INPUT_PATHS["CTRPv2"],
        low_memory=False
    )


    # ========================================================
    # Column configuration
    # ========================================================

    gdsc2_cell = (
        MANUAL_COLUMNS["GDSC2"]["cell_line_col"]
    )

    gdsc2_drug = (
        MANUAL_COLUMNS["GDSC2"]["drug_col"]
    )

    gdsc2_resp = (
        MANUAL_COLUMNS["GDSC2"]["response_col"]
    )

    ctrpv2_cell = (
        MANUAL_COLUMNS["CTRPv2"]["cell_line_col"]
    )

    ctrpv2_drug = (
        MANUAL_COLUMNS["CTRPv2"]["drug_col"]
    )

    ctrpv2_resp = (
        MANUAL_COLUMNS["CTRPv2"]["response_col"]
    )


    # ========================================================
    # SECTION 1: Structure & quality
    # ========================================================

    structure_text = "\n\n".join(
        [
            structure_and_quality_report(
                gdsc2_df,
                "GDSC2",
                gdsc2_cell,
                gdsc2_drug
            ),

            structure_and_quality_report(
                ctrpv2_df,
                "CTRPv2",
                ctrpv2_cell,
                ctrpv2_drug
            ),
        ]
    )

    with open(
        OUTPUT_DIR / "01_structure_and_quality.txt",
        "w"
    ) as f:

        f.write(
            structure_text
        )


    # ========================================================
    # SECTION 2: Duplicates
    # ========================================================

    duplicate_text = "\n\n".join(
        [
            duplicate_report(
                gdsc2_df,
                "GDSC2",
                gdsc2_cell,
                gdsc2_drug
            ),

            duplicate_report(
                ctrpv2_df,
                "CTRPv2",
                ctrpv2_cell,
                ctrpv2_drug
            ),
        ]
    )

    with open(
        OUTPUT_DIR / "02_duplicates.txt",
        "w"
    ) as f:

        f.write(
            duplicate_text
        )


    # ========================================================
    # SECTION 3: Raw response statistics
    # ========================================================

    gdsc2_stats = response_descriptive_stats(
        gdsc2_df,
        gdsc2_resp
    )

    ctrpv2_stats = response_descriptive_stats(
        ctrpv2_df,
        ctrpv2_resp
    )

    response_stats_df = pd.DataFrame(
        {
            "GDSC2": gdsc2_stats,
            "CTRPv2": ctrpv2_stats,
        }
    )

    response_stats_df.to_excel(
        OUTPUT_DIR / "03_raw_response_summary.xlsx",
        sheet_name="Response Summary"
    )


    # ========================================================
    # Extract finite raw response values
    # ========================================================

    gdsc2_values = (
        gdsc2_df[gdsc2_resp]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    ctrpv2_values = (
        ctrpv2_df[ctrpv2_resp]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )


    # ========================================================
    # SECTION 4: Combined raw response distribution
    # ========================================================

    plot_combined_histogram(
        gdsc2_values,
        ctrpv2_values,
        OUTPUT_DIR /
        "04_raw_combined_response_distribution.png"
    )


    # ========================================================
    # SECTION 5: Unique overlapping cell-line/drug pairs
    # ========================================================

    merged = find_overlapping_pairs(
        gdsc2_df,
        gdsc2_cell,
        gdsc2_drug,
        gdsc2_resp,

        ctrpv2_df,
        ctrpv2_cell,
        ctrpv2_drug,
        ctrpv2_resp
    )


    # ========================================================
    # Count unique cell-line/drug pairs in each dataset
    # ========================================================

    gdsc2_unique_pairs = (
        prepare_unique_pairs(
            gdsc2_df,
            gdsc2_cell,
            gdsc2_drug,
            gdsc2_resp
        )
        .drop_duplicates(
            subset=[
                gdsc2_cell,
                gdsc2_drug
            ]
        )
    )

    ctrpv2_unique_pairs = (
        prepare_unique_pairs(
            ctrpv2_df,
            ctrpv2_cell,
            ctrpv2_drug,
            ctrpv2_resp
        )
        .drop_duplicates(
            subset=[
                ctrpv2_cell,
                ctrpv2_drug
            ]
        )
    )

    gdsc2_pair_count = len(
        gdsc2_unique_pairs
    )

    ctrpv2_pair_count = len(
        ctrpv2_unique_pairs
    )


    # ========================================================
    # Count unique overlapping cell-line/drug pairs
    # ========================================================

    gdsc2_pair_keys = set(
        zip(
            gdsc2_unique_pairs[gdsc2_cell],
            gdsc2_unique_pairs[gdsc2_drug]
        )
    )

    ctrpv2_pair_keys = set(
        zip(
            ctrpv2_unique_pairs[ctrpv2_cell],
            ctrpv2_unique_pairs[ctrpv2_drug]
        )
    )

    overlap_pair_count = len(
        gdsc2_pair_keys
        &
        ctrpv2_pair_keys
    )

    matching_measurement_count = 14233


    # ========================================================
    # SECTION 6: Save overlap information to Excel
    # ========================================================

    gdsc2_cells = set(
        gdsc2_df[
            gdsc2_cell
        ].dropna()
    )

    ctrpv2_cells = set(
        ctrpv2_df[
            ctrpv2_cell
        ].dropna()
    )

    gdsc2_drugs = set(
        gdsc2_df[
            gdsc2_drug
        ].dropna()
    )

    ctrpv2_drugs = set(
        ctrpv2_df[
            ctrpv2_drug
        ].dropna()
    )


    overlap_summary_df = pd.DataFrame(
        {
            "Measure": [
                "GDSC2 unique cell lines",
                "CTRPv2 unique cell lines",
                "Cell lines in both datasets",
                "GDSC2 unique drugs",
                "CTRPv2 unique drugs",
                "Drugs in both datasets",
                "GDSC2 unique cell-drug pairs",
                "CTRPv2 unique cell-drug pairs",
                "Shared unique cell-drug pairs",
                "GDSC2-only cell-drug pairs",
                "CTRPv2-only cell-drug pairs",
            ],

            "Value": [
                len(gdsc2_cells),
                len(ctrpv2_cells),
                len(
                    gdsc2_cells
                    & ctrpv2_cells
                ),
                len(gdsc2_drugs),
                len(ctrpv2_drugs),
                len(
                    gdsc2_drugs
                    & ctrpv2_drugs
                ),
                gdsc2_pair_count,
                ctrpv2_pair_count,
                overlap_pair_count,
                gdsc2_pair_count
                - overlap_pair_count,
                ctrpv2_pair_count
                - overlap_pair_count,
            ]
        }
    )


    with pd.ExcelWriter(
        OUTPUT_DIR /
        "05_overlapping_cell_drug_pairs.xlsx",
        engine="openpyxl"
    ) as writer:

        merged.to_excel(
            writer,
            sheet_name="Overlapping Pairs",
            index=False
        )

        overlap_summary_df.to_excel(
            writer,
            sheet_name="Overlap Summary",
            index=False
        )


    # ========================================================
    # SECTION 7: Correlation
    # ========================================================

    if len(merged) > 0:

        correlation = (
            merged[
                [
                    "GDSC2_response",
                    "CTRPv2_response"
                ]
            ]
            .corr(
                method="spearman"
            )
            .iloc[0, 1]
        )

        with open(
            OUTPUT_DIR /
            "06_overlap_correlation.txt",
            "w"
        ) as f:

            f.write(
                "=== GDSC2 vs CTRPv2 Overlap ===\n\n"
            )

            f.write(
                "Matching drug-cell-line measurements: "
                f"{matching_measurement_count}\n"
            )

            f.write(
                "Spearman correlation: "
                f"{correlation:.4f}\n"
            )


        # ====================================================
        # SECTION 8: Scatter plot
        # ====================================================

        plot_overlap_scatter(
            merged,
            OUTPUT_DIR /
            "07_overlapping_response_scatter.png"
        )


    else:

        with open(
            OUTPUT_DIR /
            "06_overlap_correlation.txt",
            "w"
        ) as f:

            f.write(
                "No overlapping cell-line/drug pairs "
                "were found.\n"
                "Correlation and scatter plot were "
                "not calculated."
            )


    # ========================================================
    # SECTION 9: Overlap bar chart
    # ========================================================

    plot_overlap_counts(
        gdsc2_pair_count,
        ctrpv2_pair_count,
        overlap_pair_count,
        OUTPUT_DIR /
        "08_overlap_counts.png"
    )


    # ========================================================
    # Finished
    # ========================================================

    print("\nDone.")

    print(
        "Unique shared cell-line/drug pairs:",
        overlap_pair_count
    )

    print(
        "All raw EDA outputs saved to:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":
    main()