"""
Filter out "dead" (low-response-spread) drugs, computed on GDSC2 training
data only, and apply the same drug list to both cleaned datasets.

Purpose
-------
Removes drugs with little/no biological effect (flat response across cell
lines) — the biggest lever for improving cross-dataset agreement, per the
reference deck's analysis. Spread is computed on GDSC2 only, to avoid
leaking CTRPv2 (test) information into the filtering decision.

Inputs
------
GDSC2_cleaned.csv (pubchem_id, LN_IC50_curvecurator)
CTRPv2_normalised_cleaned.csv (pubchem_id, LN_IC50_curvecurator_normalised)

Outputs
-------
GDSC2_filtered.csv, CTRPv2_normalised_filtered.csv
drug_spread_GDSC2.csv (per-drug spread + kept/dropped flag, for reference)
"""

import pandas as pd
import os

Data_root = (
    "/Users/manellamukangwa/Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction"
)
Output_dir = os.path.join(Data_root, "results", "normalised")
os.makedirs(Output_dir, exist_ok=True)

# --- Step 1: compute per-drug response spread on GDSC2 (training) only ---
gdsc2 = pd.read_csv(os.path.join(Output_dir, "GDSC2_cleaned.csv"))
drug_spread = gdsc2.groupby("pubchem_id")["LN_IC50_curvecurator"].std().reset_index()
drug_spread.columns = ["pubchem_id", "spread"]

# --- Step 2: define "dead" drugs as the bottom third by spread ---
threshold = drug_spread["spread"].quantile(1/3)
drug_spread["status"] = drug_spread["spread"].apply(
    lambda s: "dropped (dead)" if s <= threshold else "kept"
)
dead_drug_ids = drug_spread.loc[drug_spread["status"] == "dropped (dead)", "pubchem_id"]

drug_spread.to_csv(os.path.join(Output_dir, "drug_spread_GDSC2.csv"), index=False)
print(f"Spread threshold (bottom third): {threshold:.4f}")
print(f"Dead drugs identified: {len(dead_drug_ids)} of {len(drug_spread)}")

# --- Step 3: apply the same dead-drug list to both cleaned datasets ---
ctrpv2 = pd.read_csv(os.path.join(Output_dir, "CTRPv2_normalised_cleaned.csv"))

gdsc2_filtered = gdsc2[~gdsc2["pubchem_id"].isin(dead_drug_ids)]
ctrpv2_filtered = ctrpv2[~ctrpv2["pubchem_id"].isin(dead_drug_ids)]

gdsc2_filtered.to_csv(os.path.join(Output_dir, "GDSC2_filtered.csv"), index=False)
ctrpv2_filtered.to_csv(os.path.join(Output_dir, "CTRPv2_normalised_filtered.csv"), index=False)

print(f"GDSC2: {len(gdsc2)} -> {len(gdsc2_filtered)} rows")
print(f"CTRPv2: {len(ctrpv2)} -> {len(ctrpv2_filtered)} rows")