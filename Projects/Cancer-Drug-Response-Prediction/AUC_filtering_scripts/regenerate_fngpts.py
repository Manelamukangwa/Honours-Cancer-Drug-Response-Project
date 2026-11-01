import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


parser = argparse.ArgumentParser()

parser.add_argument("--fingerprints", required=True)
parser.add_argument("--drug-smiles", required=True)
parser.add_argument("--output", required=True)

args = parser.parse_args()


# Load files
fingerprints = pd.read_csv(args.fingerprints)
drug_smiles = pd.read_csv(args.drug_smiles)


# Columns
drug_column = fingerprints.columns[0]
fingerprint_columns = fingerprints.columns[1:]


# Morgan fingerprint generator
generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=128
)


# Find rows with completely missing fingerprints
fingerprint_values = fingerprints.iloc[:, 1:]

missing_mask = fingerprint_values.isna() | (
    fingerprint_values.astype(str).apply(lambda x: x.str.strip() == "")
)

missing_rows = missing_mask.all(axis=1)


# Regenerate missing fingerprints
regenerated = 0
unrecoverable = 0

for index in fingerprints.index[missing_rows]:

    drug = str(fingerprints.loc[index, drug_column]).strip()

    match = drug_smiles[
        (drug_smiles["pubchem_id"].astype(str).str.strip() == drug)
        | (drug_smiles["drug_name"].astype(str).str.strip() == drug)
    ]

    if match.empty:
        unrecoverable += 1
        continue

    smiles = match["canonical_smiles"].dropna().astype(str).str.strip()

    smiles = smiles[smiles != ""]

    if smiles.empty:
        unrecoverable += 1
        continue

    mol = Chem.MolFromSmiles(smiles.iloc[0])

    if mol is None:
        unrecoverable += 1
        continue

    fingerprint = generator.GetFingerprint(mol)

    bits = list(fingerprint)

    fingerprints.loc[index, fingerprint_columns] = bits

    regenerated += 1


# Save
fingerprints.to_csv(args.output, index=False)


# Results
print("\n========================================")
print("FINGERPRINT REGENERATION")
print("========================================")

print(f"\nOriginal missing fingerprints: {missing_rows.sum()}")
print(f"Fingerprints regenerated: {regenerated}")
print(f"Still unrecoverable: {unrecoverable}")

print(f"\nOutput:")
print(args.output)

print("\n========================================")