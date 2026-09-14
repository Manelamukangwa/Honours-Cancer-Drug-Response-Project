"""
Extreme-Response / Dead-Drug Analysis Script
---------------------------------------------
Purpose
-------
Identify drugs with unusually extreme response values.

- Uses the 1st and 99th percentiles of the response distribution.
- Flags drugs whose average response is outside these limits.
- Does not remove or modify any data.
- Saves the flagged drugs from both datasets in one Excel file.

To use this script with different datasets, change only the configuration section.
"""

# ===========================================================
# SECTION 1: IMPORT LIBRARIES
# ===========================================================
import pandas as pd
from pathlib import Path

# ===========================================================
# SECTION 2: CONFIGURATION
# ===========================================================
DATA_ROOT = Path(
    "/Users/manellamukangwa/"
    "Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction"
)

# Folder containing the  datasets 
DATA_DIR = DATA_ROOT / "data" 

# Folder for extreme-response results
RESULTS_DIR = DATA_ROOT / "results" / "extreme_response"

DATASETS = {
    "GDSC2": {
        "path": DATA_DIR / "raw" / "GDSC2" / "GDSC2.csv",
        "response_column": "LN_IC50_curvecurator",
        "drug_column": "pubchem_id",
    },

    "CTRPv2": {
        "path": DATA_DIR / "raw" / "CTRPv2" / "CTRPv2.csv",
        "response_column": "LN_IC50_curvecurator",
        "drug_column": "pubchem_id",
    },
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================
# SECTION 3: RUN ANALYSIS
# ===========================================================
all_flagged_drugs = []

for name, info in DATASETS.items():

    print(f"\nAnalysing extreme responses: {name}")

    df = pd.read_csv(info["path"], low_memory=False)

    df[info["response_column"]] = pd.to_numeric(
        df[info["response_column"]],
        errors="coerce"
    )

    df = df.dropna(subset=[info["drug_column"], info["response_column"]])

    lower_limit = df[info["response_column"]].quantile(0.01)
    upper_limit = df[info["response_column"]].quantile(0.99)

    print(f"Lower response limit: {lower_limit:.4f}")
    print(f"Upper response limit: {upper_limit:.4f}")

    drug_response = (
        df.groupby(info["drug_column"])[info["response_column"]]
        .mean()
        .reset_index()
    )

    flagged_drugs = drug_response[
        (drug_response[info["response_column"]] < lower_limit)
        | (drug_response[info["response_column"]] > upper_limit)
    ].copy()

    flagged_drugs["Dataset"] = name

    all_flagged_drugs.append(flagged_drugs)

    print(f"Flagged drugs: {len(flagged_drugs)}")

# ===========================================================
# SECTION 4: SAVE RESULTS
# ===========================================================
all_flagged_drugs = pd.concat(all_flagged_drugs, ignore_index=True)

output_path = RESULTS_DIR / "flagged_drugs.xlsx"

all_flagged_drugs.to_excel(output_path, index=False)

print(f"\nResults saved to:\n{output_path}")
print("\n================ ANALYSIS COMPLETE ================\n")