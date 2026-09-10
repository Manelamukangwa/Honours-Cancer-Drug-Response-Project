"""
Outlier concentration check by drug response-spread band — GDSC2 and CTRPv2.

Purpose
-------
Checks whether severe outliers (via the IQR rule) are concentrated among
low-response-spread ("dead") drugs, or scattered across drugs of all
activity levels — run on both datasets for comparison.

Method
------
1. Flag outliers in the response column using the standard 1.5 x IQR rule.
2. Compute each drug's response spread (SD of response across cell lines).
3. Bin drugs into tertiles (low / mid / high spread), following the same
   "dead / in-between / active" logic used in the reference deck's drug
   banding analysis.
4. Compute the outlier rate within each spread band.

Interpretation
--------------
- Outlier rate much higher in the low-spread ("dead-ish") band
  -> severe outliers are drug-activity driven; dead-drug filtering should
     also reduce kurtosis / tail-heaviness.
- Outlier rate similar across all bands
  -> outliers are not specifically tied to dead drugs; a separate
     outlier-handling step would be needed alongside drug filtering.
- Comparing GDSC2 vs CTRPv2 patterns shows whether this is a general
  property of drug-response data, or specific to one dataset.

Inputs
------
GDSC2.csv               : pubchem_id (or drug id), LN_IC50_curvecurator
CTRPv2_normalised.csv   : pubchem_id, LN_IC50_curvecurator_normalised

Outputs
-------
- <name>_outlier_by_drug_spread.txt : outlier rate summary per spread band
- <name>_drug_spread_bands.csv      : per-drug spread + assigned band
"""

import pandas as pd
import numpy as np
import os

# Folder containing the input CSVs / where outputs are saved
Input_dir = (
    "/Users/manellamukangwa/Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction/results/normalised"
)
Output_dir = Input_dir
os.makedirs(Output_dir, exist_ok=True)

datasets = {
    "GDSC2": {
        "path": os.path.join(Input_dir, "GDSC2.csv"),
        "response_col": "LN_IC50_curvecurator",
        "drug_col": "pubchem_id",
    },
    "CTRPv2_normalised": {
        "path": os.path.join(Input_dir, "CTRPv2_normalised.csv"),
        "response_col": "LN_IC50_curvecurator_normalised",
        "drug_col": "pubchem_id",
    },
}

for name, info in datasets.items():
    df = pd.read_csv(info["path"])
    response_col = info["response_col"]
    drug_col = info["drug_col"]

    df = df.dropna(subset=[response_col, drug_col])

    # --- Step 1: flag overall outliers using IQR rule ---
    q1 = df[response_col].quantile(0.25)
    q3 = df[response_col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df["is_outlier"] = (df[response_col] < lower_bound) | (df[response_col] > upper_bound)

    # --- Step 2: compute per-drug response spread (std of response) ---
    drug_stats = df.groupby(drug_col)[response_col].agg(
        spread="std",
        median="median",
        n="count"
    ).reset_index()

    # --- Step 3: bin drugs into spread bands (tertiles: low/mid/high spread) ---
    drug_stats["spread_band"] = pd.qcut(
        drug_stats["spread"], q=3, labels=["low_spread (dead-ish)", "mid_spread", "high_spread (active)"]
    )

    # --- Step 4: merge band back onto main df, then check outlier rate per band ---
    df = df.merge(drug_stats[[drug_col, "spread_band"]], on=drug_col, how="left")

    outlier_summary = df.groupby("spread_band")["is_outlier"].agg(
        n_points="count",
        n_outliers="sum"
    )
    outlier_summary["outlier_rate"] = outlier_summary["n_outliers"] / outlier_summary["n_points"]

    # --- Save results ---
    results_path = os.path.join(Output_dir, f"{name}_outlier_by_drug_spread.txt")
    with open(results_path, "w") as f:
        f.write(f"Outlier rate by drug response-spread band ({name})\n")
        f.write(f"Overall outlier bounds: [{lower_bound:.4f}, {upper_bound:.4f}]\n\n")
        f.write(outlier_summary.to_string())
        f.write("\n")

    drug_stats.to_csv(os.path.join(Output_dir, f"{name}_drug_spread_bands.csv"), index=False)

    print(f"\n{name}:")
    print(outlier_summary)

print(f"\nSaved all results to {Output_dir}")