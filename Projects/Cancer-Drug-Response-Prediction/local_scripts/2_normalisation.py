"""
Normalisation: GDSC2 vs CTRPv2
--------------------------------

Purpose
-------
Load raw GDSC2 and CTRPv2 drug-response data.

Normalise ONLY CTRPv2 using GDSC2 as the reference
distribution through reference-based standardisation.

GDSC2 remains completely unchanged.

The normalised CTRPv2 response is stored in a new column:

    LN_IC50_curvecurator_normalised

The script then creates ONE final comparison histogram using:

    GDSC2:
        LN_IC50_curvecurator

    CTRPv2:
        LN_IC50_curvecurator_normalised

Outputs are saved to:

    results/normalised

No other preprocessing is performed.
"""


import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    "/Users/manellamukangwa/"
    "Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction/data"
)

# Raw input datasets
GDSC2_INPUT = (
    DATA_ROOT / "raw/GDSC2/GDSC2.csv"
)

CTRPV2_INPUT = (
    DATA_ROOT / "raw/CTRPv2/CTRPv2.csv"
)

# Results directory
RESULTS_OUTPUT = (
    DATA_ROOT.parent / "results/normalised"
)


# ============================================================
# RESPONSE COLUMNS
# ============================================================

RESPONSE_COLUMN = (
    "LN_IC50_curvecurator"
)

NORMALISED_COLUMN = (
    "LN_IC50_curvecurator_normalised"
)


# ============================================================
# OUTPUT FILES
# ============================================================

GDSC2_OUTPUT = (
    RESULTS_OUTPUT / "GDSC2.csv"
)

CTRPV2_OUTPUT = (
    RESULTS_OUTPUT / "CTRPv2_normalised.csv"
)

HISTOGRAM_OUTPUT = (
    RESULTS_OUTPUT
    / "gdsc2_vs_ctrpv2_normalised_distribution.png"
)

SUMMARY_OUTPUT = (
    RESULTS_OUTPUT
    / "03_normalised_response_summary.xlsx"
)


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

RESULTS_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD RAW DATA
# ============================================================

print("Loading raw GDSC2...")
gdsc2 = pd.read_csv(
    GDSC2_INPUT,
    low_memory=False
)

print("Loading raw CTRPv2...")
ctrpv2 = pd.read_csv(
    CTRPV2_INPUT,
    low_memory=False
)


# ============================================================
# CHECK RESPONSE COLUMN
# ============================================================

if RESPONSE_COLUMN not in gdsc2.columns:
    raise KeyError(
        f"Column '{RESPONSE_COLUMN}' "
        "was not found in GDSC2."
    )

if RESPONSE_COLUMN not in ctrpv2.columns:
    raise KeyError(
        f"Column '{RESPONSE_COLUMN}' "
        "was not found in CTRPv2."
    )


# ============================================================
# EXTRACT RESPONSE VALUES
# ============================================================

gdsc2_response = (
    gdsc2[RESPONSE_COLUMN]
    .dropna()
)

ctrpv2_response = (
    ctrpv2[RESPONSE_COLUMN]
    .dropna()
)


# ============================================================
# CALCULATE REFERENCE STATISTICS
# ============================================================

gdsc2_mean = gdsc2_response.mean()
gdsc2_sd = gdsc2_response.std()

ctrpv2_mean = ctrpv2_response.mean()
ctrpv2_sd = ctrpv2_response.std()


print("\n================ ORIGINAL RESPONSE STATISTICS ================\n")

print("GDSC2")
print(f"Mean: {gdsc2_mean:.6f}")
print(f"SD:   {gdsc2_sd:.6f}")

print("\nCTRPv2")
print(f"Mean: {ctrpv2_mean:.6f}")
print(f"SD:   {ctrpv2_sd:.6f}")


# ============================================================
# REFERENCE-BASED STANDARDISATION
# ============================================================
#
# GDSC2 is the reference dataset.
#
# GDSC2 remains unchanged.
#
# CTRPv2 is:
#
# 1. centred using the CTRPv2 mean
# 2. scaled using the CTRPv2 SD
# 3. rescaled using the GDSC2 SD
# 4. shifted using the GDSC2 mean
#
# Formula:
#
# CTRPv2_normalised =
#
# ((CTRPv2 - CTRPv2_mean) / CTRPv2_SD)
# * GDSC2_SD
# + GDSC2_mean
#
# ============================================================

print(
    "\nNormalising CTRPv2 using GDSC2 "
    "as the reference..."
)

ctrpv2[NORMALISED_COLUMN] = (
    (
        ctrpv2[RESPONSE_COLUMN]
        - ctrpv2_mean
    )
    / ctrpv2_sd
) * gdsc2_sd + gdsc2_mean


# ============================================================
# VERIFY NORMALISED RESPONSE
# ============================================================

ctrpv2_normalised_response = (
    ctrpv2[NORMALISED_COLUMN]
    .dropna()
)


ctrpv2_normalised_mean = (
    ctrpv2_normalised_response.mean()
)

ctrpv2_normalised_sd = (
    ctrpv2_normalised_response.std()
)


print(
    "\n================ NORMALISATION RESULT ================\n"
)

print("GDSC2 — unchanged")
print(f"Mean: {gdsc2_mean:.6f}")
print(f"SD:   {gdsc2_sd:.6f}")

print("\nCTRPv2 — normalised")
print(f"Mean: {ctrpv2_normalised_mean:.6f}")
print(f"SD:   {ctrpv2_normalised_sd:.6f}")


# ============================================================
# CREATE ONE FINAL COMPARISON HISTOGRAM
# ============================================================

print(
    "\nCreating final response distribution histogram..."
)

plt.figure(figsize=(10, 6))

plt.hist(
    gdsc2_response,
    bins=50,
    alpha=0.5,
    label="GDSC2"
)

plt.hist(
    ctrpv2_normalised_response,
    bins=50,
    alpha=0.5,
    label="CTRPv2 normalised"
)

plt.xlabel(
    "LN_IC50_curvecurator"
)

plt.ylabel(
    "Number of measurements"
)

plt.title(
    "GDSC2 vs CTRPv2 Normalised Drug-Response Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    HISTOGRAM_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE OUTPUT DATASETS
# ============================================================

print("\nSaving output datasets...")

# GDSC2 is saved unchanged
gdsc2.to_csv(
    GDSC2_OUTPUT,
    index=False
)

# CTRPv2 contains the new normalised column
ctrpv2.to_csv(
    CTRPV2_OUTPUT,
    index=False
)


# ============================================================
# SAVE NORMALISED RESPONSE SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "Dataset": [
            "GDSC2",
            "CTRPv2_normalised"
        ],
        "Response_column": [
            RESPONSE_COLUMN,
            NORMALISED_COLUMN
        ],
        "N": [
            len(gdsc2_response),
            len(ctrpv2_normalised_response)
        ],
        "Mean": [
            gdsc2_response.mean(),
            ctrpv2_normalised_response.mean()
        ],
        "Median": [
            gdsc2_response.median(),
            ctrpv2_normalised_response.median()
        ],
        "SD": [
            gdsc2_response.std(),
            ctrpv2_normalised_response.std()
        ],
        "Min": [
            gdsc2_response.min(),
            ctrpv2_normalised_response.min()
        ],
        "Max": [
            gdsc2_response.max(),
            ctrpv2_normalised_response.max()
        ]
    }
)

summary.to_excel(
    SUMMARY_OUTPUT,
    index=False
)


# ============================================================
# FINAL OUTPUT SUMMARY
# ============================================================

print(
    "\n================ PIPELINE COMPLETE ================\n"
)

print(
    "GDSC2 was kept unchanged."
)

print(
    "CTRPv2 was normalised using GDSC2 "
    "as the reference."
)

print(
    f"\nGDSC2 output:\n{GDSC2_OUTPUT}"
)

print(
    f"\nCTRPv2 output:\n{CTRPV2_OUTPUT}"
)

print(
    f"\nComparison histogram:\n{HISTOGRAM_OUTPUT}"
)

print(
    f"\nNormalised response summary:\n{SUMMARY_OUTPUT}"
)