"""
Script: Plot LN_IC50 Distributions

Purpose
-------
Create one boxplot of standardised LN_IC50 values for each LCO split.

The script automatically detects the within-study and cross-study
standardised response files in each split directory.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser(
        description="Plot LN_IC50 distributions for each LCO split."
    )

    parser.add_argument(
        "input_dir",
        help="Directory containing split_1, split_2, ..., split_5"
    )

    args = parser.parse_args()
    base_dir = Path(args.input_dir)

    for split in range(1, 6):

        split_dir = base_dir / f"split_{split}"

        files = {
            "Train": split_dir / "train_standardised.csv",
            "Validation": split_dir / "validation_standardised.csv",
            "Test": split_dir / "test_standardised.csv"
        }

        # Automatically find cross-study standardised files
        for file in split_dir.glob("matched_*_standardised.csv"):
            name = file.stem.replace("matched_", "").replace(
                "_standardised", ""
            )
            files[name] = file

        data = []

        for name, file in files.items():

            df = pd.read_csv(file)

            values = pd.to_numeric(
                df["LN_IC50_standardised"],
                errors="coerce"
            ).dropna()

            data.append(values)

        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=files.keys())

        plt.title(f"Standardised LN_IC50 Distribution - LCO Split {split}")
        plt.ylabel("Standardised LN_IC50")
        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()