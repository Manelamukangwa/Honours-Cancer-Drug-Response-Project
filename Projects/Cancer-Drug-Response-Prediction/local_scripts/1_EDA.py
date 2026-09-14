"""
Exploratpry Data Analysis Script: Compare Raw Drug-Response Datasets
---------------------------------------------------------------------------------------------------
Purpose: This script performs simple exploratory data analysis of two raw drug-response datasets.

    It answers five questions:
    --------------------------
    1. How large are the datasets?
    2. Are there missing/infinite responses or duplicated cell-line/drug measurements?
    3. What do the raw response distributions look like?
    4. How much cell-line, drug, and cell-line/drug overlap exists between the datasets?
    5. For shared cell-line/drug measurements, how similar are the response values?

    - This script does not clean, transform, or modify the data.
    - It does not run machine learning, PCA, clustering, or formal statistical tests.
    - To use this script with different datasets, change only the configuration section below.
"""

# =========================================================
# Section 1: IMPORT LIBRARIES
# =========================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# Section 2: CONFIGURATION: 
# Where the datasets are located
# Which columns contain the cell line, drug, and response
# Where to save EDA results 
# ========================================================
DATA_ROOT = Path(
    "/Users/manellamukangwa/"
    "Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction/data"
)

DATASETS = {
    "GDSC2": {
        "path": DATA_ROOT / "raw/GDSC2/GDSC2.csv",
        "cell": "cellosaurus_id",
        "drug": "pubchem_id",
        "response": "LN_IC50_curvecurator",
    },

    "CTRPv2": {
        "path": DATA_ROOT / "raw/CTRPv2/CTRPv2.csv",
        "cell": "cellosaurus_id",
        "drug": "pubchem_id",
        "response": "LN_IC50_curvecurator",
    },
}

OUTPUT_DIR = (DATA_ROOT.parent / "results/EDA_results/gdsc2_ctrpv2")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# Section 3: LOADS EACH DATASET
# Reports: number of rows, columns, unique cell lines & drug
# ============================================================
data = {}

for name, info in DATASETS.items():
    print(f"Loading {name}...")
    df = pd.read_csv(info["path"],low_memory=False)
    data[name] = df

    print(f"{name}:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Cell lines: {df[info['cell']].nunique()}")
    print(f"  Drugs: {df[info['drug']].nunique()}")

    print()

# -----------------------------------------------------------
# Save basic dataset information
# ------------------------------------------------------------
size_results = []

for name, df in data.items():
    info = DATASETS[name]
    size_results.append({
        "Dataset": name,
        "Rows": len(df),
        "Columns": len(df.columns),
        "Unique cell lines": df[info["cell"]].nunique(),
        "Unique drugs": df[info["drug"]].nunique()
    })

pd.DataFrame(size_results).to_excel(OUTPUT_DIR / "1_dataset_size.xlsx",index=False)

# ===============================================================================
# Section 4: DATA QUALITY MISSING / INFINITE RESPONSES AND DUPLICATES
# Checks where response data has: missing values, infinite values, duplicate rows 
# or duplicated cell-line/drug combinations
# ================================================================================
quality_results = []

for name, df in data.items():
    info = DATASETS[name]
    response = pd.to_numeric(df[info["response"]],errors="coerce")

    missing = response.isna().sum()
    infinite = np.isinf(response).sum()
    duplicated_rows = df.duplicated().sum()

    duplicated_pairs = df.duplicated(
        subset=[info["cell"],info["drug"]], keep=False).sum()

    quality_results.append({
        "Dataset": name,
        "Missing responses": missing,
        "Infinite responses": infinite, 
        "Fully duplicated rows": duplicated_rows,
        "Rows in duplicated cell-drug pairs": duplicated_pairs
    })
quality_df = pd.DataFrame(quality_results)

quality_df.to_excel(OUTPUT_DIR / "2_data_quality.xlsx",index=False)

# ==========================================================================
# Section 5: RAW RESPONSE DISTRIBUTIONS
# Creates a histogram for the response values in each dataset
# Where does response values lie, what do the distributions look like
# Do the datasets have different range or shape
# ============================================================================
plt.figure(figsize=(9, 6))

for name, df in data.items():
    response_col = DATASETS[name]["response"]
    response = pd.to_numeric(df[response_col],errors="coerce")
    response = response[np.isfinite(response)]

    plt.hist(response,bins=50,alpha=0.5,label=name)

plt.title("Raw Drug-Response Distributions")
plt.xlabel("LN_IC50_curvecurator")
plt.ylabel("Number of observations")
plt.legend()
plt.tight_layout()
    
# Saves raw response distribution
plt.savefig(
    OUTPUT_DIR /
    "3_raw_response_distributions.png",
    dpi=300,
    bbox_inches="tight"
    )
plt.close()

# ================================================================================================
# Section 6: DATASET OVERLAP (CELL-LINE, DRUG AND CELL-LINE/DRUG OVERLAP)
# Checks what the two datasets have in common: shared cell lines, drug and cell-line + drug pairs
# ===============================================================================================
names = list(data.keys())

dataset_a = names[0]
dataset_b = names[1]

a = data[dataset_a]
b = data[dataset_b]

a_info = DATASETS[dataset_a]
b_info = DATASETS[dataset_b]

# Unique cell lines
a_cells = set(a[a_info["cell"]].dropna())
b_cells = set(b[b_info["cell"]].dropna())

# Unique drugs
a_drugs = set(a[a_info["drug"]].dropna())
b_drugs = set(b[b_info["drug"]].dropna())

# Unique cell-drug pairs
a_pairs = set(zip(a[a_info["cell"]].dropna(),a[a_info["drug"]].dropna()))

b_pairs = set(zip(b[b_info["cell"]].dropna(),b[b_info["drug"]].dropna()))

overlap_results = pd.DataFrame({
    "Measure": [
        f"{dataset_a} cell lines",
        f"{dataset_b} cell lines",
        "Shared cell lines",

        f"{dataset_a} drugs",
        f"{dataset_b} drugs",
        "Shared drugs",

        f"{dataset_a} cell-drug pairs",
        f"{dataset_b} cell-drug pairs",
        "Shared cell-drug pairs"
    ],

    "Value": [
        len(a_cells),
        len(b_cells),
        len(a_cells & b_cells),

        len(a_drugs),
        len(b_drugs),
        len(a_drugs & b_drugs),

        len(a_pairs),
        len(b_pairs),
        len(a_pairs & b_pairs)
    ]
})

overlap_results.to_excel(OUTPUT_DIR /"4_dataset_overlap.xlsx",index=False)

# ========================================================================================
# Section 7: COMPARE SHARED RESPONSE VALUES FOR SHARED CELL-DRUG PAIRS
# For cell-drug pairs that exist in both datasets, it:
# Takes the response from dataset A and B and matches them using same cell line + drug
# Removes missing/infinite responses
# ========================================================================================

# Keep only the columns needed
a_pairs_df = a[
    [
        a_info["cell"],
        a_info["drug"],
        a_info["response"]
    ]
].copy()

b_pairs_df = b[
    [
        b_info["cell"],
        b_info["drug"],
        b_info["response"]
    ]
].copy()

# Rename columns so they can be merged
a_pairs_df.columns = [
    "cell_line",
    "drug",
    f"{dataset_a}_response"
]

b_pairs_df.columns = [
    "cell_line",
    "drug",
    f"{dataset_b}_response"
]

# ------------------------------------------------------------------
# remove invalid responses to correctly compare shared responses
# -------------------------------------------------------------------

# Remove missing
a_pairs_df = a_pairs_df.dropna()

b_pairs_df = b_pairs_df.dropna()

# Removes infinite responses
a_pairs_df = a_pairs_df[
    np.isfinite(
        pd.to_numeric(
            a_pairs_df[f"{dataset_a}_response"],
            errors="coerce"
        )
    )
]

b_pairs_df = b_pairs_df[
    np.isfinite(
        pd.to_numeric(
            b_pairs_df[f"{dataset_b}_response"],
            errors="coerce"
        )
    )
]

# Find cell-drug pairs present in both datasets
shared = pd.merge(
    a_pairs_df,
    b_pairs_df,
    on=["cell_line", "drug"],
    how="inner"
)

# Save shared measurements
shared.to_excel(
    OUTPUT_DIR /
    "05_shared_cell_drug_measurements.xlsx",
    index=False
)
# Calculate Spearman correlation
if len(shared) >= 2:
    correlation = shared[
        [
            f"{dataset_a}_response",
            f"{dataset_b}_response"
        ]
    ].corr(
        method="spearman"
    ).iloc[0, 1]
else:
    correlation = np.nan

with open(
    OUTPUT_DIR /
    "06_shared_response_correlation.txt",
    "w"
) as f:
    f.write(
        f"Datasets: {dataset_a} vs {dataset_b}\n"
    )
    f.write(
        f"Shared measurement rows: {len(shared)}\n"
    )
    f.write(
        f"Spearman correlation: {correlation:.4f}\n"
        if not np.isnan(correlation)
        else "Spearman correlation: Not available\n"
    )

# Scatter plot
if len(shared) > 0:
    plt.figure(figsize=(6, 6))
    plt.scatter(shared[f"{dataset_a}_response"],shared[f"{dataset_b}_response"],alpha=0.5,s=15)
    plt.xlabel(f"{dataset_a} response")
    plt.ylabel(f"{dataset_b} response")
    plt.title( f"{dataset_a} vs {dataset_b}: ""Shared Cell-Drug Responses")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "07_shared_response_scatter.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("EDA complete.")
print(f"Results saved to: {OUTPUT_DIR}")
