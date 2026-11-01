import argparse
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument("--fingerprints", required=True)
parser.add_argument("--drug-smiles", required=True)

args = parser.parse_args()


# Load files
fingerprints = pd.read_csv(args.fingerprints)
drug_smiles = pd.read_csv(args.drug_smiles)


# Find drugs with completely missing fingerprints
drug_column = fingerprints.columns[0]
fingerprint_values = fingerprints.iloc[:, 1:]

missing_mask = fingerprint_values.isna() | (
    fingerprint_values.astype(str).apply(lambda x: x.str.strip() == "")
)

missing_drugs = (
    fingerprints.loc[missing_mask.all(axis=1), drug_column]
    .astype(str)
    .str.strip()
)

missing_drugs = missing_drugs[missing_drugs != ""]


# Prepare drug identifiers
pubchem_ids = set(
    drug_smiles["pubchem_id"]
    .dropna()
    .astype(str)
    .str.strip()
)

drug_names = set(
    drug_smiles["drug_name"]
    .dropna()
    .astype(str)
    .str.strip()
)


# Check each missing drug
found = []
not_found = []
no_smiles = []

for drug in missing_drugs:

    match = drug_smiles[
        (drug_smiles["pubchem_id"].astype(str).str.strip() == drug)
        | (drug_smiles["drug_name"].astype(str).str.strip() == drug)
    ]

    if match.empty:
        not_found.append(drug)

    elif match["canonical_smiles"].isna().all() or (
        match["canonical_smiles"].astype(str).str.strip() == ""
    ).all():
        no_smiles.append(drug)

    else:
        found.append(drug)


# Results
print("\n========================================")
print("MISSING FINGERPRINT CHECK")
print("========================================")

print(f"\nTotal missing fingerprints: {len(missing_drugs)}")

print(f"\nFound with a SMILES: {len(found)}")
for drug in found:
    print(f"  {drug}")

print(f"\nFound but SMILES missing: {len(no_smiles)}")
for drug in no_smiles:
    print(f"  {drug}")

print(f"\nNot found in drug_smiles.csv: {len(not_found)}")
for drug in not_found:
    print(f"  {drug}")

print("\n========================================")