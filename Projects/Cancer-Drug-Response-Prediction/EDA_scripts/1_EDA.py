
"""
Exploratory Data Analysis Script: Compare Raw Drug-Response Datasets
====================================================================

Datasets:
    CTRPv1, CTRPv2, GDSC1, GDSC2 and CCLE.

Outputs:
    1. dataset_size.xlsx
    2. data_quality.xlsx
    3. raw_response_distributions.png
    4. shared_cell_lines_and_drugs.xlsx
    5. drug_response_distributions.xlsx

The script does not modify or clean the original datasets.
All results are calculated from the input files.
"""

# =========================================================
# SECTION 1: IMPORT LIBRARIES
# =========================================================

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# =========================================================
# SECTION 2: TERMINAL ARGUMENTS
# =========================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="EDA of raw cancer drug-response datasets."
    )

    parser.add_argument("--ctrpv1", required=True)
    parser.add_argument("--ctrpv2", required=True)
    parser.add_argument("--gdsc1", required=True)
    parser.add_argument("--gdsc2", required=True)
    parser.add_argument("--ccle", required=True)

    parser.add_argument(
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "--cell-name-column",
        default="cell_line_name"
    )

    parser.add_argument(
        "--cell-id-column",
        default="cellosaurus_id"
    )

    parser.add_argument(
        "--drug-name-column",
        default="drug_name"
    )

    parser.add_argument(
        "--drug-id-column",
        default="pubchem_id"
    )

    parser.add_argument(
        "--response-column",
        default="lnIC50_curvecurator"
    )

    return parser.parse_args()


args = parse_arguments()


# =========================================================
# SECTION 3: DATASET CONFIGURATION
# =========================================================

DATASETS = {

    "CTRPv1": Path(args.ctrpv1),

    "CTRPv2": Path(args.ctrpv2),

    "GDSC1": Path(args.gdsc1),

    "GDSC2": Path(args.gdsc2),

    "CCLE": Path(args.ccle),

}

OUTPUT_DIR = Path(args.output_dir)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CELL_NAME_COLUMN = args.cell_name_column
CELL_ID_COLUMN = args.cell_id_column
DRUG_NAME_COLUMN = args.drug_name_column
DRUG_ID_COLUMN = args.drug_id_column
RESPONSE_COLUMN = args.response_column

DATASET_ORDER = [
    "CTRPv2",
    "GDSC2",
    "GDSC1",
    "CTRPv1",
    "CCLE",
]

CROSS_STUDY_DATASETS = [
    "CTRPv1",
    "GDSC1",
    "GDSC2",
    "CCLE",
]


# =========================================================
# SECTION 4: HELPER FUNCTIONS
# =========================================================

def find_column(df, requested_column, dataset_name):

    matching_columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    requested = requested_column.strip().lower()

    if requested in matching_columns:
        return matching_columns[requested]

    raise ValueError(
        f"Column '{requested_column}' not found in {dataset_name}.\n"
        f"Available columns: {list(df.columns)}"
    )


def get_unique_values(df, column):

    return set(
        df[column].dropna().unique()
    )


def get_numeric_response(df, response_column):

    return pd.to_numeric(
        df[response_column],
        errors="coerce"
    )


def get_finite_response(df, response_column):

    response = get_numeric_response(
        df,
        response_column
    )

    return response[
        np.isfinite(response)
    ]


# =========================================================
# SECTION 5: LOAD DATASETS
# =========================================================

print("\nLoading datasets...\n")

data = {}
COLUMNS = {}

for name, path in DATASETS.items():

    print(f"Loading {name}...")

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    columns = {

        "cell_name": find_column(
            df,
            CELL_NAME_COLUMN,
            name
        ),

        "cell_id": find_column(
            df,
            CELL_ID_COLUMN,
            name
        ),

        "drug_name": find_column(
            df,
            DRUG_NAME_COLUMN,
            name
        ),

        "drug_id": find_column(
            df,
            DRUG_ID_COLUMN,
            name
        ),

        "response": find_column(
            df,
            RESPONSE_COLUMN,
            name
        ),

    }

    data[name] = df
    COLUMNS[name] = columns

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns):,}")

    print(
        f"  Unique cell lines: "
        f"{df[columns['cell_name']].nunique():,}"
    )

    print(
        f"  Unique drugs: "
        f"{df[columns['drug_name']].nunique():,}"
    )

    print()


# =========================================================
# SECTION 6: DATASET SIZE
# =========================================================

print("Calculating dataset sizes...")

size_results = []

for name in DATASET_ORDER:

    df = data[name]
    columns = COLUMNS[name]

    size_results.append({

        "Dataset": name,

        "Rows": len(df),

        "Columns": len(df.columns),

        "Unique cell lines": (
            df[columns["cell_name"]].nunique()
        ),

        "Unique drugs": (
            df[columns["drug_name"]].nunique()
        ),

    })


size_df = pd.DataFrame(size_results)

size_df.to_excel(
    OUTPUT_DIR / "dataset_size.xlsx",
    index=False
)


# =========================================================
# SECTION 7: DATA QUALITY
# =========================================================

print("Calculating data quality...")

quality_results = []

for name in DATASET_ORDER:

    df = data[name]
    columns = COLUMNS[name]

    response = get_numeric_response(
        df,
        columns["response"]
    )

    quality_results.append({

        "Dataset": name,

        "Rows": len(df),

        "Columns": len(df.columns),

        "Unique cell line names": (
            df[columns["cell_name"]].nunique()
        ),

        "Unique cellosaurus_id values": (
            df[columns["cell_id"]].nunique()
        ),

        "Unique drug names": (
            df[columns["drug_name"]].nunique()
        ),

        "Exact duplicate rows": (
            df.duplicated().sum()
        ),

        "Missing lnIC50_curvecurator": (
            response.isna().sum()
        ),

    })


quality_df = pd.DataFrame(
    quality_results
).set_index("Dataset").T

quality_df = quality_df[
    DATASET_ORDER
]

quality_df.index.name = "Characteristic"

quality_df.reset_index().to_excel(
    OUTPUT_DIR / "data_quality.xlsx",
    index=False
)


# =========================================================
# SECTION 8: RAW RESPONSE DISTRIBUTIONS
# =========================================================

print("Generating raw response distribution plot...")

plt.figure(figsize=(11, 7))

for name in DATASET_ORDER:

    df = data[name]

    response = get_finite_response(
        df,
        COLUMNS[name]["response"]
    )

    plt.hist(
        response,
        bins=50,
        alpha=0.45,
        label=name
    )


plt.title(
    "Raw Drug-Response Distributions Across Datasets"
)

plt.xlabel("lnIC50_curvecurator")
plt.ylabel("Number of observations")

plt.legend(title="Dataset")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "raw_response_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# SECTION 9: SHARED CELL LINES AND DRUGS
# =========================================================

print("Calculating shared cell lines and drugs...")

reference_name = "CTRPv2"

reference_df = data[reference_name]
reference_columns = COLUMNS[reference_name]

reference_cells = get_unique_values(
    reference_df,
    reference_columns["cell_id"]
)

reference_drugs = get_unique_values(
    reference_df,
    reference_columns["drug_id"]
)

overlap_results = []

for name in CROSS_STUDY_DATASETS:

    df = data[name]
    columns = COLUMNS[name]

    cross_study_cells = get_unique_values(
        df,
        columns["cell_id"]
    )

    cross_study_drugs = get_unique_values(
        df,
        columns["drug_id"]
    )

    overlap_results.append({

        "Cross-study Dataset": name,

        "Shared cell lines": len(
            reference_cells & cross_study_cells
        ),

        "Shared drugs": len(
            reference_drugs & cross_study_drugs
        ),

    })


overlap_df = pd.DataFrame(overlap_results)

overlap_df.to_excel(
    OUTPUT_DIR / "shared_cell_lines_and_drugs.xlsx",
    index=False
)


# =========================================================
# SECTION 10: DRUG-RESPONSE DISTRIBUTIONS
# =========================================================

print("Calculating drug-response distributions...")

distribution_results = []

response_data = {}

for name in DATASET_ORDER:

    df = data[name]

    response = get_finite_response(
        df,
        COLUMNS[name]["response"]
    )

    response_data[name] = response

    distribution_results.append({

        "Dataset": name,

        "Number of observations": len(response),

        "Mean": response.mean(),

        "Median": response.median(),

        "Standard deviation": response.std(),

        "Minimum": response.min(),

        "25th percentile": response.quantile(0.25),

        "75th percentile": response.quantile(0.75),

        "Maximum": response.max(),

    })


# ---------------------------------------------------------
# Create the descriptive statistics comparison table
# ---------------------------------------------------------

distribution_summary_df = pd.DataFrame(
    distribution_results
)

# Round numerical results for readability.

numeric_columns = distribution_summary_df.select_dtypes(
    include="number"
).columns

distribution_summary_df[numeric_columns] = (
    distribution_summary_df[numeric_columns].round(4)
)


# ---------------------------------------------------------
# Calculate histogram bin counts using common bin edges
# ---------------------------------------------------------

all_responses = pd.concat(
    [
        response_data[name]
        for name in DATASET_ORDER
    ],
    ignore_index=True
)

if len(all_responses) == 0:

    raise ValueError(
        "No finite numerical response values were found."
    )


global_min = all_responses.min()
global_max = all_responses.max()

if global_min == global_max:

    global_min -= 0.5
    global_max += 0.5


bin_edges = np.linspace(
    global_min,
    global_max,
    51
)

histogram_results = []

for name in DATASET_ORDER:

    counts, _ = np.histogram(
        response_data[name],
        bins=bin_edges
    )

    for i in range(len(counts)):

        histogram_results.append({

            "Dataset": name,

            "Bin": i + 1,

            "Bin lower limit": bin_edges[i],

            "Bin upper limit": bin_edges[i + 1],

            "Number of observations": counts[i],

        })


histogram_df = pd.DataFrame(
    histogram_results
)


# ---------------------------------------------------------
# Save Excel workbook
# ---------------------------------------------------------

with pd.ExcelWriter(
    OUTPUT_DIR / "drug_response_distributions.xlsx",
    engine="openpyxl"
) as writer:

    distribution_summary_df.to_excel(
        writer,
        sheet_name="Summary statistics",
        index=False
    )

    histogram_df.to_excel(
        writer,
        sheet_name="Histogram bin counts",
        index=False
    )


# =========================================================
# SECTION 11: COMPLETION SUMMARY
# =========================================================

print("\n" + "=" * 60)

print("EXPLORATORY DATA ANALYSIS COMPLETE")

print("=" * 60)

print("\nOutput files:")

output_files = [

    "dataset_size.xlsx",

    "data_quality.xlsx",

    "raw_response_distributions.png",

    "shared_cell_lines_and_drugs.xlsx",

    "drug_response_distributions.xlsx",

]

for filename in output_files:

    print(f"  - {filename}")

print("\nResults saved to:")

print(OUTPUT_DIR.resolve())

print("=" * 60)
