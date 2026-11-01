"""
Check whether cellosaurus_id and pubchem_id map consistently to
cell-line and drug names within and across response datasets.
Dataset paths are supplied through the terminal.
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Check cell-line and drug identifier mappings."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset paths in LABEL=FILE.csv format."
    )
    args = parser.parse_args()

    frames = []

    for item in args.datasets:
        if "=" not in item:
            parser.error(f"Invalid input '{item}'. Use LABEL=FILE.csv.")

        label, file_path = item.split("=", 1)

        if not Path(file_path).is_file():
            parser.error(f"File not found: {file_path}")

        df = pd.read_csv(file_path, low_memory=False)

        required = [
            "cellosaurus_id",
            "cell_line_name",
            "pubchem_id",
            "drug_name",
        ]
        missing = [col for col in required if col not in df.columns]

        if missing:
            parser.error(f"{label} is missing columns: {missing}")

        df = df[required].copy()
        df["dataset"] = label

        for col in required:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace("", pd.NA)

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    checks = [
        ("cellosaurus_id", "cell_line_name", "cell line"),
        ("pubchem_id", "drug_name", "drug"),
    ]

    for identifier, name, description in checks:
        print("\n" + "=" * 65)
        print(f"CHECKING {description.upper()} MAPPINGS")
        print("=" * 65)

        valid = data.dropna(subset=[identifier, name])

        # Identify identifiers associated with multiple names overall.
        name_counts = valid.groupby(identifier)[name].nunique()
        inconsistent = name_counts[name_counts > 1]

        print(f"Unique identifiers: {valid[identifier].nunique():,}")
        print(
            f"Identifiers mapping to multiple {description} names "
            f"across datasets: {len(inconsistent):,}"
        )

        for identifier_value in inconsistent.index:
            subset = valid[valid[identifier] == identifier_value]
            print(f"\n{identifier}: {identifier_value}")

            mapping = (
                subset.groupby(["dataset", name], dropna=False)
                .size()
                .reset_index(name="records")
            )
            print(mapping.to_string(index=False))

        # Also check whether mappings are inconsistent within a dataset.
        print(f"\nWithin-dataset inconsistencies for {description}s:")

        within_found = False

        for label, subset in valid.groupby("dataset"):
            counts = subset.groupby(identifier)[name].nunique()
            bad_ids = counts[counts > 1]

            if not bad_ids.empty:
                within_found = True
                print(f"\n{label}: {len(bad_ids):,} inconsistent identifiers")

                for identifier_value in bad_ids.index:
                    names = (
                        subset.loc[
                            subset[identifier] == identifier_value, name
                        ]
                        .drop_duplicates()
                        .tolist()
                    )
                    print(f"  {identifier_value}: {names}")

        if not within_found:
            print("None found.")

    print("\nMapping checks completed.")


if __name__ == "__main__":
    main()