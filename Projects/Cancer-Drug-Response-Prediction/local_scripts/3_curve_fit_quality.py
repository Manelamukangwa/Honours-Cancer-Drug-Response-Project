"""
Curve-Fit Quality Check Script
------------------------------
Purpose
-------
Evaluate the curve-fit quality of extreme-response flagged drugs.

This script:
    1. Loads the extreme-response drugs identified
    2. Calculates the overall dataset R2.
    3. Calculates the average R2 for each flagged drug.

R2 is used as the curve-fit quality measure:
    - Higher R2 = better fit.
    - Lower R2 = poorer fit.

IMPORTANT
---------
This script is a QUALITY CHECK only.

It does NOT:
    - remove drugs
    - apply an R2 threshold
    - decide which drugs should be removed

The removal decision is made by Script 3.

Results are saved separately for both datasets in one Excel file.
"""

# ============================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================
import pandas as pd
from pathlib import Path


# ============================================================
# SECTION 2: CONFIGURATION
# ============================================================
DATA_ROOT = Path(
    "/Users/manellamukangwa/"
    "Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction"
)

# Folder containing the datasets
DATA_DIR = DATA_ROOT / "data"

# ------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------
DATASETS = {
    "GDSC2": {
        "path": DATA_DIR / "raw" / "GDSC2" / "GDSC2.csv",
        "drug_column": "pubchem_id",
    },

    "CTRPv2": {
        "path": DATA_DIR / "raw" / "CTRPv2"  / "CTRPv2.csv",
        "drug_column": "pubchem_id",
    },
}

# ------------------------------------------------------------
# Flagged-drug file produced by Script 1
# ------------------------------------------------------------
FLAGGED_DRUG_FILE = (
    DATA_ROOT
    / "results"
    / "extreme_response"
    / "flagged_drugs.xlsx"
)

# ------------------------------------------------------------
# Output folder
# ------------------------------------------------------------
RESULTS_DIR = (
    DATA_ROOT
    / "results"
    / "curve_fit_quality"
)

RESULTS_DIR.mkdir(parents=True,exist_ok=True)

# ============================================================
# SECTION 3: LOAD FLAGGED DRUGS
# ============================================================
flagged = pd.read_excel(FLAGGED_DRUG_FILE)

flagged["pubchem_id"] = pd.to_numeric(
    flagged["pubchem_id"],
    errors="coerce"
).astype("Int64")

# ============================================================
# SECTION 4: RUN CURVE-FIT QUALITY CHECK
# ============================================================
all_results = []

for name, info in DATASETS.items():
    print(f"\nChecking curve-fit quality: {name}")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    df = pd.read_csv(info["path"],low_memory=False)

    df[info["drug_column"]] = pd.to_numeric(
        df[info["drug_column"]],
        errors="coerce"
        ).astype("Int64")
    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------
    required_columns = [
        info["drug_column"],
        "R2"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{name} is missing columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Keep flagged drugs for this dataset only
    # --------------------------------------------------------
    dataset_flagged = flagged[
        flagged["Dataset"] == name
    ]

    flagged_drugs = (
        dataset_flagged[info["drug_column"]]
        .dropna()
        .unique()
    )

    # --------------------------------------------------------
    # Calculate overall dataset R2
    # --------------------------------------------------------
    dataset_r2 = df["R2"].mean()

    # --------------------------------------------------------
    # Keep measurements belonging to flagged drugs
    # --------------------------------------------------------
    flagged_rows = df[
        df[info["drug_column"]].isin(flagged_drugs)
    ].copy()

    # --------------------------------------------------------
    # Calculate average R2 for each flagged drug
    # --------------------------------------------------------
    drug_r2 = (
        flagged_rows
        .groupby(info["drug_column"])["R2"]
        .mean()
        .reset_index()
    )

    # Rename columns so Script 3 can use them
    drug_r2 = drug_r2.rename(
         columns={
             "R2": "Drug_R2"
        }
    )

    # Add dataset information
    drug_r2["Dataset"] = name

    # Add overall dataset R2
    drug_r2["Dataset_R2"] = dataset_r2

    # Reorder columns
    drug_r2 = drug_r2[
        [
            "Dataset",
            "pubchem_id",
            "Drug_R2",
            "Dataset_R2"
        ]
    ]

    all_results.append(drug_r2)

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------
    print(
        f"Extreme-response flagged drugs: "
        f"{len(flagged_drugs)}"
    )

    print(
        f"Flagged drugs with R2 results: "
        f"{len(drug_r2)}"
    )

    print(
        f"Overall dataset R2: "
        f"{dataset_r2:.4f}"
    )

# ============================================================
# SECTION 5: COMBINE RESULTS
# ============================================================
final_results = pd.concat(
    all_results,
    ignore_index=True
)

# ============================================================
# SECTION 6: SAVE RESULTS
# ============================================================
output_path = (
    RESULTS_DIR
    / "flagged_drug_fit_quality_by_drug.xlsx"
)

final_results.to_excel(output_path,index=False)

print("\n================ ANALYSIS COMPLETE ================\n")

print(f"Results saved to:\n{output_path}")