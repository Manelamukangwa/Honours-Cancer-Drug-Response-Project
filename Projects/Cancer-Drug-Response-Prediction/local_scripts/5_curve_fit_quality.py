"""
Curve-fit quality check for flagged extreme-response drugs.

Purpose
-------
For the drugs previously identified with very high extreme-response rates,
pull their curve-fit quality diagnostics (R2, RMSE, SignalQuality) and
compare to the dataset-wide average — to check if poor fits explain the
extreme values.

Inputs
------
GDSC2.csv, CTRPv2_normalised.csv (must contain: pubchem_id, R2, RMSE, SignalQuality)

Outputs
-------
<name>_flagged_drug_fit_quality.txt
"""

import pandas as pd
import os

Data_root = (
    "/Users/manellamukangwa/Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction"
)
Output_dir = os.path.join(Data_root, "results", "normalised")
os.makedirs(Output_dir, exist_ok=True)

# Drugs flagged with very high extreme-response rates (from prior step)
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

    overall_avg = df[["R2", "RMSE", "SignalQuality"]].mean()
    flagged = df[df["pubchem_id"].isin(flagged_drugs[name])]
    flagged_avg = flagged[["R2", "RMSE", "SignalQuality"]].mean()

    out_path = os.path.join(Output_dir, f"{name}_flagged_drug_fit_quality.txt")
    with open(out_path, "w") as f:
        f.write(f"Curve-fit quality: flagged drugs vs dataset average ({name})\n\n")
        f.write("Dataset-wide average:\n")
        f.write(overall_avg.to_string())
        f.write("\n\nFlagged drugs average:\n")
        f.write(flagged_avg.to_string())
        f.write("\n\nFlagged drug count (rows): " + str(len(flagged)))

    print(f"{name}: saved to {out_path}")