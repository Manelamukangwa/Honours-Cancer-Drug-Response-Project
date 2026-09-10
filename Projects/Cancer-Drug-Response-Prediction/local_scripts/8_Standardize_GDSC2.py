"""
Standardize GDSC2 (filtered) training data — z-score standardization,
fit on GDSC2 only, then applied to CTRPv2 for later cross-study evaluation.

Purpose
-------
Puts the training target/features on a standard scale (mean 0, SD 1) for
modeling. Fit exclusively on GDSC2 training data to avoid leakage; the same
fitted scaler is then applied to CTRPv2.

Inputs
------
GDSC2_filtered.csv (LN_IC50_curvecurator)
CTRPv2_normalised_filtered.csv (LN_IC50_curvecurator_normalised)

Outputs
-------
GDSC2_standardised.csv, CTRPv2_standardised.csv
"""

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib

Data_root = (
    "/Users/manellamukangwa/Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction"
)
Output_dir = os.path.join(Data_root, "results", "normalised")
os.makedirs(Output_dir, exist_ok=True)

# --- Load filtered datasets ---
gdsc2 = pd.read_csv(os.path.join(Output_dir, "GDSC2_filtered.csv"))
ctrpv2 = pd.read_csv(os.path.join(Output_dir, "CTRPv2_normalised_filtered.csv"))

# --- Fit scaler on GDSC2 training data only ---
scaler = StandardScaler()
gdsc2["LN_IC50_standardised"] = scaler.fit_transform(gdsc2[["LN_IC50_curvecurator"]])

# --- Apply the same fitted scaler to CTRPv2 ---
ctrpv2["LN_IC50_standardised"] = scaler.transform(
    ctrpv2[["LN_IC50_curvecurator_normalised"]].values
)

# --- Save outputs ---
gdsc2.to_csv(os.path.join(Output_dir, "GDSC2_standardised.csv"), index=False)
ctrpv2.to_csv(os.path.join(Output_dir, "CTRPv2_standardised.csv"), index=False)

# Save the fitted scaler for reuse later (e.g. inverse-transforming predictions)
joblib.dump(scaler, os.path.join(Output_dir, "gdsc2_scaler.pkl"))

print(f"GDSC2 mean/SD used for scaling: {scaler.mean_[0]:.4f} / {scaler.scale_[0]:.4f}")
print("Saved GDSC2_standardised.csv, CTRPv2_standardised.csv, gdsc2_scaler.pkl")