"""
Script 3: Remove Extreme-Response Drugs with Poor Curve Fits
-------------------------------------------------------------
Purpose
-------
Remove extreme-response drugs when:

    Drug_R2 < Dataset_R2

The original raw datasets are not modified.
Cleaned datasets are saved separately.
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

DATA_DIR = DATA_ROOT / "data"

# Curve-fit results produced by Script 2
CURVE_FIT_FILE = (
    DATA_ROOT
    / "results"
    / "curve_fit_quality"
    / "flagged_drug_fit_quality_by_drug.xlsx"
)

# Folder for cleaned datasets
RESULTS_DIR = (
    DATA_ROOT
    / "results"
    / "cleaned_results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SECTION 3: DATASETS
# ============================================================
DATASETS = {
    "GDSC2": {
        "data": DATA_DIR / "raw" / "GDSC2" / "GDSC2.csv",
        "output": RESULTS_DIR / "GDSC2_cleaned.csv"
    },

    "CTRPv2": {
        "data": DATA_DIR / "raw" / "CTRPv2" / "CTRPv2.csv",
        "output": RESULTS_DIR / "CTRPv2_cleaned.csv"
    }
}


# ============================================================
# SECTION 4: LOAD CURVE-FIT RESULTS
# ============================================================
fit_results = pd.read_excel(
    CURVE_FIT_FILE
)


# ============================================================
# SECTION 5: PROCESS EACH DATASET
# ============================================================
for name, info in DATASETS.items():

    print(f"\nProcessing {name}")

    # --------------------------------------------------------
    # Load raw dataset
    # --------------------------------------------------------
    df = pd.read_csv(
        info["data"],
        low_memory=False
    )

    # --------------------------------------------------------
    # Get results for this dataset
    # --------------------------------------------------------
    dataset_fit = fit_results[
        fit_results["Dataset"] == name
    ]

    # --------------------------------------------------------
    # Identify extreme-response drugs with poorer R2
    # --------------------------------------------------------
    drugs_to_remove = dataset_fit.loc[
        dataset_fit["Drug_R2"] < dataset_fit["Dataset_R2"],
        "pubchem_id"
    ].dropna().unique()

    # --------------------------------------------------------
    # Remove those drugs
    # --------------------------------------------------------
    cleaned_df = df[
        ~df["pubchem_id"].isin(drugs_to_remove)
    ].copy()

    # --------------------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------------------
    cleaned_df.to_csv(
        info["output"],
        index=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------
    print(f"Drugs removed: {len(drugs_to_remove)}")
    print(f"Rows removed: {len(df) - len(cleaned_df)}")
    print(f"Rows remaining: {len(cleaned_df)}")
    print(f"Saved to: {info['output']}")


# ============================================================
# SECTION 6: COMPLETE
# ============================================================
print("\n================ SCRIPT 3 COMPLETE ================\n")