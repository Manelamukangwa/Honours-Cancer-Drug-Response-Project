"""
Train and evaluate Random Forest and Naive Mean Effects models
using standardised LCO splits.

Gene expression:
    - arcsinh transformed
    - StandardScaler fitted using training cell lines only
    - final scaler fitted using train + validation cell lines
      after hyperparameter selection

Evaluates:
    - within-dataset GDSC2 test performance
    - cross-dataset CTRPv2 performance

The five LCO split sets are created before this script runs.
"""


# ============================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error
)

from scipy.stats import (
    pearsonr,
    spearmanr
)


# ============================================================
# SECTION 2: CONFIGURATION
# ============================================================

Data_root = (
    "/Users/manellamukangwa/Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction"
)

DATASET_A = "GDSC2"
DATASET_B = "CTRPv2"

A_GENE_FILE = os.path.join(
    Data_root,
    "data/raw/GDSC2/gene_expression.csv"
)

A_FP_FILE = os.path.join(
    Data_root,
    "data/processed/GDSC2_fingerprints_128_transposed.csv"
)

A_SPLIT_DIR = os.path.join(
    Data_root,
    "data/processed/lco_splits_gdsc2_standardised"
)

B_GENE_FILE = os.path.join(
    Data_root,
    "data/raw/CTRPv2/gene_expression.csv"
)

B_FP_FILE = os.path.join(
    Data_root,
    "data/processed/CTRPv2_fingerprints_128_transposed.csv"
)

LANDMARK_GENES_FILE = os.path.join(
    Data_root,
    "data/meta/gene_lists/landmark_genes_reduced.csv"
)

CELL_ID = "cellosaurus_id"
DRUG_ID = "pubchem_id"

RESPONSE = "LN_IC50_standardised"

N_SPLITS = 5

OUTPUT_DIR = os.path.join(
    Data_root,
    "results",
    f"{DATASET_A.lower()}_{DATASET_B.lower()}",
    "outputs"
)

NAIVE_DIR = os.path.join(
    OUTPUT_DIR,
    "naive_mean"
)

COMPARISON_DIR = os.path.join(
    OUTPUT_DIR,
    "rf_vs_naive_mean"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    NAIVE_DIR,
    exist_ok=True
)

os.makedirs(
    COMPARISON_DIR,
    exist_ok=True
)


# ============================================================
# SECTION 3: LOAD DATA
# ============================================================

def load_gene_expression(path):

    df = pd.read_csv(
        path,
        low_memory=False
    )

    df[CELL_ID] = (
        df[CELL_ID]
        .astype(str)
        .str.strip()
    )

    # Remove duplicate cell-line IDs
    df = df.drop_duplicates(
        subset=[CELL_ID],
        keep="first"
    )

    return df.set_index(
        CELL_ID
    )


def load_fingerprints(path):

    df = pd.read_csv(
        path,
        low_memory=False
    )

    df[DRUG_ID] = (
        df[DRUG_ID]
        .astype(str)
        .str.strip()
    )

    df = df.drop_duplicates(
        subset=[DRUG_ID],
        keep="first"
    )

    return df.set_index(
        DRUG_ID
    )


def load_split(
    split,
    part
):

    path = os.path.join(
        A_SPLIT_DIR,
        f"split_{split}",
        f"{part}_standardised.csv"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Split file not found: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    df[CELL_ID] = (
        df[CELL_ID]
        .astype(str)
        .str.strip()
    )

    df[DRUG_ID] = (
        df[DRUG_ID]
        .astype(str)
        .str.strip()
    )

    return df


def load_crossstudy_split(
    split
):

    path = os.path.join(
        A_SPLIT_DIR,
        f"split_{split}",
        "crossstudy_standardised.csv"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Cross-study split file not found: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    df[CELL_ID] = (
        df[CELL_ID]
        .astype(str)
        .str.strip()
    )

    df[DRUG_ID] = (
        df[DRUG_ID]
        .astype(str)
        .str.strip()
    )

    return df


# ------------------------------------------------------------
# Load gene expression
# ------------------------------------------------------------

print(
    "Loading GDSC2 gene expression..."
)

A_gene = load_gene_expression(
    A_GENE_FILE
)

print(
    "Loading CTRPv2 gene expression..."
)

B_gene = load_gene_expression(
    B_GENE_FILE
)


# ------------------------------------------------------------
# Load fingerprints
# ------------------------------------------------------------

print(
    "Loading GDSC2 fingerprints..."
)

A_fp = load_fingerprints(
    A_FP_FILE
)

print(
    "Loading CTRPv2 fingerprints..."
)

B_fp = load_fingerprints(
    B_FP_FILE
)


# ============================================================
# SECTION 4: SELECT COMMON LANDMARK GENES
# ============================================================

landmark_genes = (
    pd.read_csv(
        LANDMARK_GENES_FILE
    )
    .iloc[:, 0]
    .astype(str)
    .tolist()
)

genes = [
    gene
    for gene in landmark_genes
    if (
        gene in A_gene.columns
        and
        gene in B_gene.columns
    )
]

if not genes:

    raise ValueError(
        "No common landmark genes were found "
        "between GDSC2 and CTRPv2."
    )

A_gene = A_gene[
    genes
]

B_gene = B_gene[
    genes
]

print(
    f"Common landmark genes used: "
    f"{len(genes)}"
)


# ============================================================
# SECTION 5: ARCSINH TRANSFORMATION
# ============================================================

A_gene = np.arcsinh(
    A_gene.astype(float)
)

B_gene = np.arcsinh(
    B_gene.astype(float)
)


# ============================================================
# SECTION 6: METRICS
# ============================================================

def metrics(
    y_true,
    y_pred
):

    return {
        "R2": r2_score(
            y_true,
            y_pred
        ),

        "Pearson": pearsonr(
            y_true,
            y_pred
        )[0],

        "Spearman": spearmanr(
            y_true,
            y_pred
        )[0],

        "MSE": mean_squared_error(
            y_true,
            y_pred
        )
    }


# ============================================================
# SECTION 7: BUILD MODEL FEATURES
# ============================================================

def build_features(
    response_df,
    gene_df,
    fp_df
):

    df = response_df.merge(
        gene_df.reset_index(),
        on=CELL_ID,
        how="inner"
    )

    df = df.merge(
        fp_df.reset_index(),
        on=DRUG_ID,
        how="inner"
    )

    return df


# ============================================================
# SECTION 8: NAIVE MEAN EFFECTS
# ============================================================

class NaiveMeanEffects:

    def fit(
        self,
        df
    ):

        self.mean = (
            df[RESPONSE]
            .mean()
        )

        cell_means = (
            df
            .groupby(CELL_ID)[RESPONSE]
            .mean()
        )

        drug_means = (
            df
            .groupby(DRUG_ID)[RESPONSE]
            .mean()
        )

        self.cell_effect = (
            cell_means
            -
            self.mean
        )

        self.drug_effect = (
            drug_means
            -
            self.mean
        )

    def predict(
        self,
        df
    ):

        cell_effect = (
            df[CELL_ID]
            .map(self.cell_effect)
            .fillna(0)
        )

        drug_effect = (
            df[DRUG_ID]
            .map(self.drug_effect)
            .fillna(0)
        )

        return (
            self.mean
            +
            cell_effect
            +
            drug_effect
        )


# ============================================================
# SECTION 9: MAIN LCO LOOP
# ============================================================

rf_predictions = []

rf_metrics = []

naive_predictions = []

naive_metrics = []

comparison = []


for split in range(
    N_SPLITS
):

    print(
        f"\n=============================="
    )

    print(
        f"LCO SPLIT {split}"
    )

    print(
        f"=============================="
    )


    # --------------------------------------------------------
    # Load LCO split
    # --------------------------------------------------------

    train = load_split(
        split,
        "train"
    )

    validation = load_split(
        split,
        "validation"
    )

    test = load_split(
        split,
        "test"
    )

    B_response = load_crossstudy_split(
        split
    )


    # ========================================================
    # TRAINING-ONLY GENE EXPRESSION SCALING
    # ========================================================

    train_cells = (
        train[CELL_ID]
        .unique()
    )

    missing_train_cells = [
        cell
        for cell in train_cells
        if cell not in A_gene.index
    ]

    if missing_train_cells:

        raise ValueError(
            f"Split {split} contains "
            f"{len(missing_train_cells)} training "
            f"cell lines without gene expression."
        )

    scaler = StandardScaler()

    scaler.fit(
        A_gene.loc[
            train_cells
        ]
    )

    A_gene_scaled = pd.DataFrame(
        scaler.transform(
            A_gene
        ),
        index=A_gene.index,
        columns=A_gene.columns
    )

    B_gene_scaled = pd.DataFrame(
        scaler.transform(
            B_gene
        ),
        index=B_gene.index,
        columns=B_gene.columns
    )


    # ========================================================
    # BUILD TRAINING DATA
    # ========================================================

    train_data = build_features(
        train,
        A_gene_scaled,
        A_fp
    )

    validation_data = build_features(
        validation,
        A_gene_scaled,
        A_fp
    )

    test_data = build_features(
        test,
        A_gene_scaled,
        A_fp
    )


    if train_data.empty:

        raise ValueError(
            f"Split {split}: training data is empty "
            "after feature matching."
        )

    if validation_data.empty:

        raise ValueError(
            f"Split {split}: validation data is empty "
            "after feature matching."
        )

    if test_data.empty:

        raise ValueError(
            f"Split {split}: test data is empty "
            "after feature matching."
        )


    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in train_data.columns
        if column not in [
            CELL_ID,
            DRUG_ID,
            RESPONSE,
            "LN_IC50_curvecurator",
            "cell_line_name",
            "tissue"
        ]
    ]


    X_train = (
        train_data[
            feature_columns
        ]
    )

    y_train = (
        train_data[RESPONSE]
    )

    X_val = (
        validation_data[
            feature_columns
        ]
    )

    y_val = (
        validation_data[RESPONSE]
    )


    print(
        f"Train rows: {len(y_train)}"
    )

    print(
        f"Validation rows: {len(y_val)}"
    )

    print(
        f"Test rows: {len(test_data)}"
    )


    # ========================================================
    # RANDOM FOREST HYPERPARAMETER TUNING
    # ========================================================

    best_depth = None

    best_mse = np.inf

    for depth in [
        5,
        10,
        30
    ]:

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=depth,
            max_samples=0.2,
            criterion="squared_error",
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        val_pred = model.predict(
            X_val
        )

        mse = mean_squared_error(
            y_val,
            val_pred
        )

        print(
            f"Depth {depth}: "
            f"validation MSE = {mse:.4f}"
        )

        if mse < best_mse:

            best_mse = mse

            best_depth = depth


    print(
        f"Best depth: {best_depth}"
    )


    # ========================================================
    # FINAL TRAIN + VALIDATION MODEL
    # ========================================================

    trainval = pd.concat(
        [
            train,
            validation
        ],
        ignore_index=True
    )


    # --------------------------------------------------------
    # Fit final gene scaler using train + validation only
    # --------------------------------------------------------

    trainval_cells = (
        trainval[CELL_ID]
        .unique()
    )

    missing_trainval_cells = [
        cell
        for cell in trainval_cells
        if cell not in A_gene.index
    ]

    if missing_trainval_cells:

        raise ValueError(
            f"Split {split} contains "
            f"{len(missing_trainval_cells)} "
            f"train/validation cell lines without "
            f"gene expression."
        )


    final_scaler = StandardScaler()

    final_scaler.fit(
        A_gene.loc[
            trainval_cells
        ]
    )


    final_A_gene_scaled = pd.DataFrame(
        final_scaler.transform(
            A_gene
        ),
        index=A_gene.index,
        columns=A_gene.columns
    )


    final_B_gene_scaled = pd.DataFrame(
        final_scaler.transform(
            B_gene
        ),
        index=B_gene.index,
        columns=B_gene.columns
    )


    # --------------------------------------------------------
    # Build final train + validation features
    # --------------------------------------------------------

    trainval_data = build_features(
        trainval,
        final_A_gene_scaled,
        A_fp
    )

    test_data = build_features(
        test,
        final_A_gene_scaled,
        A_fp
    )


    X_trainval = (
        trainval_data[
            feature_columns
        ]
    )

    y_trainval = (
        trainval_data[RESPONSE]
    )


    X_test = (
        test_data[
            feature_columns
        ]
    )

    y_test = (
        test_data[RESPONSE]
    )


    # --------------------------------------------------------
    # Final Random Forest
    # --------------------------------------------------------

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=best_depth,
        max_samples=0.2,
        criterion="squared_error",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_trainval,
        y_trainval
    )


    # ========================================================
    # GDSC2 TEST PREDICTION
    # ========================================================

    rf_test_pred = rf.predict(
        X_test
    )

    test_scores = metrics(
        y_test,
        rf_test_pred
    )

    rf_metrics.append(
        {
            "Split": split,
            "Dataset": DATASET_A,
            "Evaluation": "Test",
            "Best_depth": best_depth,
            **test_scores
        }
    )


    for cell, drug, actual, pred in zip(
        test_data[CELL_ID],
        test_data[DRUG_ID],
        y_test,
        rf_test_pred
    ):

        rf_predictions.append(
            {
                "Split": split,
                "Dataset": DATASET_A,
                "Evaluation": "Test",
                CELL_ID: cell,
                DRUG_ID: drug,
                "Actual": actual,
                "Predicted": pred
            }
        )


    # ========================================================
    # CTRPv2 CROSS-STUDY PREDICTION
    # ========================================================

    B_data = build_features(
        B_response,
        final_B_gene_scaled,
        B_fp
    )


    if B_data.empty:

        print(
            "No usable CTRPv2 rows for this split."
        )

        B_scores = {
            "R2": np.nan,
            "Pearson": np.nan,
            "Spearman": np.nan,
            "MSE": np.nan
        }

    else:

        X_B = B_data[
            feature_columns
        ]

        y_B = B_data[
            RESPONSE
        ]

        rf_B_pred = rf.predict(
            X_B
        )

        B_scores = metrics(
            y_B,
            rf_B_pred
        )


        for cell, drug, actual, pred in zip(
            B_data[CELL_ID],
            B_data[DRUG_ID],
            y_B,
            rf_B_pred
        ):

            rf_predictions.append(
                {
                    "Split": split,
                    "Dataset": DATASET_B,
                    "Evaluation": "Cross-study",
                    CELL_ID: cell,
                    DRUG_ID: drug,
                    "Actual": actual,
                    "Predicted": pred
                }
            )


    rf_metrics.append(
        {
            "Split": split,
            "Dataset": DATASET_B,
            "Evaluation": "Cross-study",
            "Best_depth": best_depth,
            **B_scores
        }
    )


    # ========================================================
    # NAIVE MEAN EFFECTS
    # ========================================================

    naive = NaiveMeanEffects()

    naive.fit(
        trainval
    )


    naive_test_pred = naive.predict(
        test
    )

    naive_B_pred = naive.predict(
        B_response
    )


    naive_test_scores = metrics(
        test[RESPONSE],
        naive_test_pred
    )


    naive_B_scores = metrics(
        B_response[RESPONSE],
        naive_B_pred
    )


    naive_metrics.append(
        {
            "Split": split,
            "Dataset": DATASET_A,
            "Evaluation": "Test",
            **naive_test_scores
        }
    )


    naive_metrics.append(
        {
            "Split": split,
            "Dataset": DATASET_B,
            "Evaluation": "Cross-study",
            **naive_B_scores
        }
    )


    # --------------------------------------------------------
    # Naive GDSC2 predictions
    # --------------------------------------------------------

    for cell, drug, actual, pred in zip(
        test[CELL_ID],
        test[DRUG_ID],
        test[RESPONSE],
        naive_test_pred
    ):

        naive_predictions.append(
            {
                "Split": split,
                "Dataset": DATASET_A,
                "Evaluation": "Test",
                CELL_ID: cell,
                DRUG_ID: drug,
                "Actual": actual,
                "Predicted": pred
            }
        )


    # --------------------------------------------------------
    # Naive CTRPv2 predictions
    # --------------------------------------------------------

    for cell, drug, actual, pred in zip(
        B_response[CELL_ID],
        B_response[DRUG_ID],
        B_response[RESPONSE],
        naive_B_pred
    ):

        naive_predictions.append(
            {
                "Split": split,
                "Dataset": DATASET_B,
                "Evaluation": "Cross-study",
                CELL_ID: cell,
                DRUG_ID: drug,
                "Actual": actual,
                "Predicted": pred
            }
        )


    # ========================================================
    # RF VS NAIVE
    # ========================================================

    comparison.extend(
        [
            {
                "Split": split,
                "Dataset": DATASET_A,
                "Evaluation": "Test",
                "Model": "Random Forest",
                **test_scores
            },

            {
                "Split": split,
                "Dataset": DATASET_A,
                "Evaluation": "Test",
                "Model": "Naive Mean Effects",
                **naive_test_scores
            },

            {
                "Split": split,
                "Dataset": DATASET_B,
                "Evaluation": "Cross-study",
                "Model": "Random Forest",
                **B_scores
            },

            {
                "Split": split,
                "Dataset": DATASET_B,
                "Evaluation": "Cross-study",
                "Model": "Naive Mean Effects",
                **naive_B_scores
            }
        ]
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"\nSplit {split} results:"
    )

    print(
        f"  RF {DATASET_A} R² = "
        f"{test_scores['R2']:.3f}"
    )

    print(
        f"  Naive {DATASET_A} R² = "
        f"{naive_test_scores['R2']:.3f}"
    )

    if not np.isnan(
        B_scores["R2"]
    ):

        print(
            f"  RF {DATASET_B} R² = "
            f"{B_scores['R2']:.3f}"
        )

        print(
            f"  Naive {DATASET_B} R² = "
            f"{naive_B_scores['R2']:.3f}"
        )


# ============================================================
# SECTION 10: SAVE RESULTS
# ============================================================

pd.DataFrame(
    rf_predictions
).to_excel(
    os.path.join(
        OUTPUT_DIR,
        "rf_predictions_all_splits.xlsx"
    ),
    index=False
)


pd.DataFrame(
    rf_metrics
).to_excel(
    os.path.join(
        OUTPUT_DIR,
        "rf_performance_metrics.xlsx"
    ),
    index=False
)


pd.DataFrame(
    naive_predictions
).to_excel(
    os.path.join(
        NAIVE_DIR,
        "naive_mean_predictions_all_splits.xlsx"
    ),
    index=False
)


pd.DataFrame(
    naive_metrics
).to_excel(
    os.path.join(
        NAIVE_DIR,
        "naive_mean_performance_metrics.xlsx"
    ),
    index=False
)


pd.DataFrame(
    comparison
).to_excel(
    os.path.join(
        COMPARISON_DIR,
        "rf_vs_naive_mean_comparison.xlsx"
    ),
    index=False
)


# ============================================================
# SECTION 11: SIMPLE COMPARISON PLOT
# ============================================================

comparison_df = pd.DataFrame(
    comparison
)

mean_r2 = (
    comparison_df
    .groupby(
        [
            "Dataset",
            "Evaluation",
            "Model"
        ]
    )["R2"]
    .mean()
    .reset_index()
)


for evaluation in (
    mean_r2["Evaluation"]
    .unique()
):

    plot_data = mean_r2[
        mean_r2["Evaluation"]
        ==
        evaluation
    ]

    plot_data = plot_data.pivot(
        index="Dataset",
        columns="Model",
        values="R2"
    )

    plot_data.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.ylabel(
        "Mean R²"
    )

    plt.title(
        "Random Forest vs Naive Mean Effects - "
        f"{evaluation}"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# SECTION 12: FINISHED
# ============================================================

print(
    "\nDone."
)

print(
    f"Results saved to: {OUTPUT_DIR}"
)