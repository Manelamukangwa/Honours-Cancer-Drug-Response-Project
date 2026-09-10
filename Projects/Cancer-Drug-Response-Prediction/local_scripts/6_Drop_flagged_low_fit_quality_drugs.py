"""
Drop flagged low-fit-quality drugs from GDSC2 and CTRPv2_normalised,
and save cleaned datasets.

Purpose
-------
Removes the specific drugs previously confirmed to have poor curve-fit
quality (low R²) and near-100% extreme-response rates 
these drugs showed unusually high extreme-response rates and substantially lower curve-fit R², 
suggesting that poor or uninformative curve fitting may contribute to their extreme responses.

Inputs
------
GDSC2.csv, CTRPv2_normalised.csv (must contain: pubchem_id)

Outputs
-------
GDSC2_cleaned.csv, CTRPv2_normalised_cleaned.csv
"""

import pandas as pd
import os

Data_root = (
    "/Users/manellamukangwa/Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction"
)
Output_dir = os.path.join(Data_root, "results", "normalised")
os.makedirs(Output_dir, exist_ok=True)

# Drugs confirmed as poor curve fits (low R², near-100% extreme rate)
flagged_drugs = {
    "GDSC2": [54670067, 864, 124886, 12035],
    "CTRPv2_normalised": [135398744, 252682, 2733487, 6917907],
}

datasets = {
    "GDSC2": os.path.join(Output_dir, "GDSC2.csv"),
    "CTRPv2_normalised": os.path.join(Output_dir, "CTRPv2_normalised.csv"),
}

for name, path in datasets.items():
    df = pd.read_csv(path)
    before_n = len(df)

    df_cleaned = df[~df["pubchem_id"].isin(flagged_drugs[name])]
    after_n = len(df_cleaned)

    out_path = os.path.join(Output_dir, f"{name}_cleaned.csv")
    df_cleaned.to_csv(out_path, index=False)

    print(f"{name}: {before_n} -> {after_n} rows (dropped {before_n - after_n}). Saved to {out_path}")