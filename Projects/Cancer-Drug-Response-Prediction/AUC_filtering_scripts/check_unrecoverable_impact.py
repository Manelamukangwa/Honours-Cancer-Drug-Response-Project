import pandas as pd


# ============================================================
# Input
# ============================================================

response_file = "/Users/manellamukangwa/Honours-Cancer-Drug-Response-Project/Projects/Cancer-Drug-Response-Prediction/data/cleaned_response/GDSC2_clean.csv"


# ============================================================
# Unrecoverable GDSC2 drugs
# ============================================================

unrecoverable = [
    "BDF00022089a",
    "BDOCA000347a",
    "BDP-00009066",
    "BPD-00008900",
    "CDK9_5038",
    "CDK9_5576",
    "CT7033-2",
    "ERK_2440",
    "ERK_6604",
    "Eg5_9814",
    "GSK2276186C",
    "GSK2830371A",
    "GSK3337463A",
    "HKMTI-1-005",
    "IAP_5620",
    "IGF1R_3801",
    "IRAK4_4710",
    "JAK1_8709",
    "JAK_8517",
    "LMB_AB1",
    "LMB_AB2",
    "LMB_AB3",
    "N25720-51-A1",
    "N27922-53-1",
    "N29087-69-1",
    "N30652-18-1",
    "PAK_5339",
    "PBD-288",
    "TAF1_5496",
    "THR-101",
    "THR-102",
    "THR-103",
    "ULK1_4989",
    "VSP34_8731",
    "VTP-A",
    "VTP-B"
]


# ============================================================
# Load response data
# ============================================================

response = pd.read_csv(response_file)


# ============================================================
# Check affected observations
# ============================================================

drug_column = "pubchem_id"

affected = response[
    response[drug_column]
    .astype(str)
    .str.strip()
    .isin(unrecoverable)
]


# ============================================================
# Results
# ============================================================

print("\n========================================")
print("GDSC2 UNRECOVERABLE FINGERPRINT IMPACT")
print("========================================")

print(f"\nTotal response observations: {len(response)}")
print(f"Affected observations: {len(affected)}")
print(f"Affected unique drugs: {affected[drug_column].nunique()}")

print("\nAffected observations by drug:")

print(
    affected[drug_column]
    .value_counts()
    .sort_index()
)

print("\n========================================")