
import argparse
from pathlib import Path

import pandas as pd

DRUG_COLUMN = "pubchem_id"


def load_drugs(path):
    df = pd.read_csv(path, low_memory=False)

    if DRUG_COLUMN not in df.columns:
        raise ValueError(f"Missing {DRUG_COLUMN} in {path}")

    drugs = (
        df[DRUG_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
    )

    drugs = drugs[drugs.ne("")]
    return set(drugs), drugs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--unseen-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    split_root = Path(args.split_dir)
    unseen_root = Path(args.unseen_dir)
    results = []

    for split_dir in sorted(split_root.glob("split_*")):
        if not split_dir.is_dir():
            continue
        train_path = split_dir / "train.csv"
        unseen_path = (
            unseen_root / split_dir.name / "GDSC2_unseen.csv"
        )

        if not train_path.exists() or not unseen_path.exists():
            raise FileNotFoundError(
                f"Missing train or unseen file for {split_dir.name}"
            )

        train_drugs, train_values = load_drugs(train_path)
        unseen_drugs, unseen_values = load_drugs(unseen_path)

        shared = train_drugs & unseen_drugs
        unseen_only = unseen_drugs - train_drugs

        results.append({
            "split": split_dir.name,
            "training_unique_drugs": len(train_drugs),
            "GDSC2_unique_drugs": len(unseen_drugs),
            "shared_drugs": len(shared),
            "GDSC2_drugs_absent_from_training": len(unseen_only),
            "GDSC2_drug_overlap_pct": (
                100 * len(shared) / len(unseen_drugs)
                if unseen_drugs else float("nan")
            ),
            "training_non_numeric_ids": sum(
                not value.isdigit() for value in train_drugs
            ),
            "GDSC2_non_numeric_ids": sum(
                not value.isdigit() for value in unseen_drugs
            ),
        })

    output = pd.DataFrame(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(output.to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()

