"""
Train and evaluate a MultiViewXGBoost model and a Naive Mean Effects
baseline for cancer drug-response prediction.

Gene expression is transformed using arcsinh and standardised using a
scaler fitted on the LCO training cell lines only. The same scaler is
then applied to validation, test, and external data.

Response standardisation is performed separately for each LCO split,
using the training response only. The corresponding split-specific
standardised CTRPv2 response is used for cross-study evaluation.

Both models are evaluated on the same GDSC2 LCO test rows and the same
split-specific external CTRPv2 rows using R2, Pearson, Spearman, and MSE.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

Data_root = (
    "/Users/manellamukangwa/Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction/data"
)

Dataset_A_name = "GDSC2"

Dataset_A_gene_expression = (
    f"{Data_root}/raw/GDSC2/gene_expression.csv"
)

Dataset_A_drug_fingerprints = (
    f"{Data_root}/processed/GDSC2_fingerprints_128_transposed.csv"
)

# Uses split-specific standardised LCO folders
Dataset_A_lco_splits_dir = (
    f"{Data_root}/processed/lco_splits_gdsc2_standardised"
)

Dataset_B_name = "CTRPv2"

Dataset_B_gene_expression = (
    f"{Data_root}/raw/CTRPv2/gene_expression.csv"
)

Dataset_B_drug_fingerprints = (
    f"{Data_root}/processed/CTRPv2_fingerprints_128_transposed.csv"
)

Gene_list_path = (
    f"{Data_root}/meta/gene_lists/landmark_genes_reduced.csv"
)

Cell_line_identifier = "cell_line_name"
Drug_identifier = "pubchem_id"

Response_column = "LN_IC50_standardised"
Dataset_B_response_column = "LN_IC50_standardised"

n_splits = 5


# -------------------------------------------------------------------------
# Output folder
# -------------------------------------------------------------------------

Results_root = (
    "/Users/manellamukangwa/"
    "Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction/"
    "results/gds2c/outputs"
)

os.makedirs(Results_root, exist_ok=True)


# -------------------------------------------------------------------------
# MultiViewXGBoost hyperparameters
# -------------------------------------------------------------------------

Hyperparam_grid = [
    {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 10,
        "subsample": 0.8,
        "colsample_bytree": colsample,
        "reg_alpha": reg_alpha,
    }
    for colsample in [0.6, 0.8]
    for reg_alpha in [0, 1]
]

XGB_random_state = 42


# -------------------------------------------------------------------------
# Load gene expression
# -------------------------------------------------------------------------

def load_gene_expression_features(
    gene_expression_csv,
    gene_list_csv
):
    """
    Load gene expression and select the reduced landmark-gene set.
    """

    gene_expression = pd.read_csv(
        gene_expression_csv,
        index_col=Cell_line_identifier
    )

    gene_expression.index = (
        gene_expression.index.astype(str)
    )

    if "cellosaurus_id" in gene_expression.columns:
        gene_expression = gene_expression.drop(
            columns=["cellosaurus_id"]
        )

    gene_list = pd.read_csv(
        gene_list_csv
    )["Symbol"].astype(str).tolist()

    missing_genes = [
        gene
        for gene in gene_list
        if gene not in gene_expression.columns
    ]

    if missing_genes:
        raise ValueError(
            f"{len(missing_genes)} landmark genes are missing. "
            f"First few: {missing_genes[:10]}"
        )

    gene_expression = gene_expression[
        gene_list
    ]

    gene_expression = gene_expression[
        ~gene_expression.index.duplicated(
            keep="first"
        )
    ]

    return {
        cell_line: gene_expression.loc[cell_line].to_numpy(
            dtype=float
        )
        for cell_line in gene_expression.index
    }


# -------------------------------------------------------------------------
# Load drug fingerprints
# -------------------------------------------------------------------------

def load_drug_fingerprint_features(
    fingerprint_csv
):
    """
    Load transposed drug molecular fingerprints.
    """

    fingerprints = pd.read_csv(
        fingerprint_csv,
        index_col=Drug_identifier
    )

    fingerprints.index = (
        fingerprints.index.astype(str)
    )

    return {
        drug: fingerprints.loc[drug].to_numpy(
            dtype=float
        )
        for drug in fingerprints.index
    }


# -------------------------------------------------------------------------
# Gene-expression preprocessing
# -------------------------------------------------------------------------

def arcsinh_transform(
    gene_expression
):
    return {
        cell_line: np.arcsinh(vector)
        for cell_line, vector in gene_expression.items()
    }


def fit_and_scale_gene_expression(
    gene_expression,
    training_cell_lines
):
    """
    Fit StandardScaler using only the LCO training cell lines.
    """

    transformed = arcsinh_transform(
        gene_expression
    )

    training_vectors = [
        transformed[cell_line]
        for cell_line in training_cell_lines
        if cell_line in transformed
    ]

    if not training_vectors:
        raise ValueError(
            "No training cell lines were found for scaling."
        )

    scaler = StandardScaler()

    scaler.fit(
        np.vstack(training_vectors)
    )

    scaled_expression = {
        cell_line: scaler.transform(
            vector.reshape(1, -1)
        )[0]
        for cell_line, vector in transformed.items()
    }

    return scaled_expression, scaler


# -------------------------------------------------------------------------
# Load standardised LCO split
# -------------------------------------------------------------------------

def load_split_part(
    splits_dir,
    split_idx,
    part
):
    """
    Load one of the existing standardised LCO split files.
    """

    path = os.path.join(
        splits_dir,
        f"split_{split_idx}",
        f"{part}_standardised.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Standardised split file not found: {path}"
        )

    return pd.read_csv(
        path,
        dtype={
            Drug_identifier: str,
            Cell_line_identifier: str
        },
        low_memory=False
    )


# -------------------------------------------------------------------------
# Load split-specific standardised cross-study response
# -------------------------------------------------------------------------

def load_crossstudy_split(
    splits_dir,
    split_idx
):
    """
    Load the split-specific standardised CTRPv2 response.
    """

    path = os.path.join(
        splits_dir,
        f"split_{split_idx}",
        "crossstudy_standardised.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Standardised cross-study file not found: {path}"
        )

    return pd.read_csv(
        path,
        dtype={
            Drug_identifier: str,
            Cell_line_identifier: str
        },
        low_memory=False
    )


# -------------------------------------------------------------------------
# Build model input
# -------------------------------------------------------------------------

def build_X_y(
    df,
    gene_expression,
    drug_fingerprints,
    response_column
):
    """
    Combine gene-expression and drug-fingerprint features.
    """

    X_rows = []
    y_rows = []
    kept_idx = []

    for i, row in df.iterrows():

        cell_line = row[
            Cell_line_identifier
        ]

        drug = row[
            Drug_identifier
        ]

        if cell_line not in gene_expression:
            continue

        if drug not in drug_fingerprints:
            continue

        X_rows.append(
            np.concatenate(
                [
                    gene_expression[cell_line],
                    drug_fingerprints[drug]
                ]
            )
        )

        y_rows.append(
            row[response_column]
        )

        kept_idx.append(i)

    if not X_rows:
        return (
            np.empty((0, 0)),
            np.array([], dtype=float),
            df.iloc[0:0].copy()
        )

    X = np.vstack(X_rows)

    y = np.array(
        y_rows,
        dtype=float
    )

    kept_df = df.loc[
        kept_idx
    ].reset_index(
        drop=True
    )

    return X, y, kept_df


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

def compute_metrics(
    y_true,
    y_pred
):
    if len(y_true) < 2:
        return {
            "R2": np.nan,
            "Pearson": np.nan,
            "Spearman": np.nan,
            "MSE": np.nan
        }

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


# -------------------------------------------------------------------------
# Naive Mean Effects Predictor
# -------------------------------------------------------------------------

class NaiveMeanEffectsPredictor:
    """
    Predict response using the overall mean plus cell-line and
    drug-specific effects learned from the training data.
    """

    def fit(self, df):
        self.overall_mean = (
            df[Response_column].mean()
        )

        self.cell_effects = (
            df.groupby(
                Cell_line_identifier
            )[Response_column].mean()
            - self.overall_mean
        )

        self.drug_effects = (
            df.groupby(
                Drug_identifier
            )[Response_column].mean()
            - self.overall_mean
        )

    def predict(self, df):
        cell_effect = (
            df[Cell_line_identifier]
            .map(self.cell_effects)
            .fillna(0)
        )

        drug_effect = (
            df[Drug_identifier]
            .map(self.drug_effects)
            .fillna(0)
        )

        return (
            self.overall_mean
            + cell_effect
            + drug_effect
        )


# -------------------------------------------------------------------------
# Load features
# -------------------------------------------------------------------------

print(
    f"Loading {Dataset_A_name} gene expression..."
)

dataset_a_gene_expr = load_gene_expression_features(
    Dataset_A_gene_expression,
    Gene_list_path
)


print(
    f"Loading {Dataset_A_name} drug fingerprints..."
)

dataset_a_drug_fingerprints = load_drug_fingerprint_features(
    Dataset_A_drug_fingerprints
)


print(
    f"Loading {Dataset_B_name} gene expression..."
)

dataset_b_gene_expr = load_gene_expression_features(
    Dataset_B_gene_expression,
    Gene_list_path
)


print(
    f"Loading {Dataset_B_name} drug fingerprints..."
)

dataset_b_drug_fingerprints = load_drug_fingerprint_features(
    Dataset_B_drug_fingerprints
)


# -------------------------------------------------------------------------
# Results
# -------------------------------------------------------------------------

xgb_within_results = []
xgb_cross_results = []

naive_within_results = []
naive_cross_results = []

xgb_predictions = []
naive_predictions = []


# -------------------------------------------------------------------------
# Main LCO loop
# -------------------------------------------------------------------------

for split_idx in range(n_splits):

    print(
        f"\n================ SPLIT {split_idx} ================"
    )

    # -------------------------------------------------------------
    # Load split-specific standardised responses
    # -------------------------------------------------------------

    train_df = load_split_part(
        Dataset_A_lco_splits_dir,
        split_idx,
        "train"
    )

    validation_df = load_split_part(
        Dataset_A_lco_splits_dir,
        split_idx,
        "validation"
    )

    test_df = load_split_part(
        Dataset_A_lco_splits_dir,
        split_idx,
        "test"
    )

    dataset_b_response = load_crossstudy_split(
        Dataset_A_lco_splits_dir,
        split_idx
    )


    # -------------------------------------------------------------
    # Fit scaler using training cells only
    # -------------------------------------------------------------

    training_cell_lines = train_df[
        Cell_line_identifier
    ].unique()

    scaled_dataset_a_gene_expr, scaler = (
        fit_and_scale_gene_expression(
            dataset_a_gene_expr,
            training_cell_lines
        )
    )


    # -------------------------------------------------------------
    # Transform CTRPv2 gene expression using the same
    # GDSC2 training-fitted scaler
    # -------------------------------------------------------------

    dataset_b_transformed = {
        cell_line: np.arcsinh(vector)
        for cell_line, vector in dataset_b_gene_expr.items()
    }

    dataset_b_scaled_gene_expr = {
        cell_line: scaler.transform(
            vector.reshape(1, -1)
        )[0]
        for cell_line, vector
        in dataset_b_transformed.items()
    }


    # -------------------------------------------------------------
    # Build train and validation matrices
    # -------------------------------------------------------------

    X_train, y_train, _ = build_X_y(
        train_df,
        scaled_dataset_a_gene_expr,
        dataset_a_drug_fingerprints,
        Response_column
    )

    X_val, y_val, _ = build_X_y(
        validation_df,
        scaled_dataset_a_gene_expr,
        dataset_a_drug_fingerprints,
        Response_column
    )


    print(
        f"Train size: {len(y_train)} | "
        f"Validation size: {len(y_val)}"
    )


    # -------------------------------------------------------------
    # Hyperparameter tuning
    # -------------------------------------------------------------

    best_params = None
    best_val_mse = np.inf

    for params in Hyperparam_grid:

        model = xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            random_state=XGB_random_state,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        val_pred = model.predict(
            X_val
        )

        val_mse = mean_squared_error(
            y_val,
            val_pred
        )

        if val_mse < best_val_mse:

            best_val_mse = val_mse
            best_params = params


    print(
        f"Best hyperparameters: {best_params}"
    )


    # -------------------------------------------------------------
    # Train final MultiViewXGBoost
    # -------------------------------------------------------------

    trainval_df = pd.concat(
        [
            train_df,
            validation_df
        ],
        ignore_index=True
    )

    X_trainval, y_trainval, _ = build_X_y(
        trainval_df,
        scaled_dataset_a_gene_expr,
        dataset_a_drug_fingerprints,
        Response_column
    )

    final_model = xgb.XGBRegressor(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        reg_alpha=best_params["reg_alpha"],
        random_state=XGB_random_state,
        n_jobs=-1
    )

    final_model.fit(
        X_trainval,
        y_trainval
    )


    # -------------------------------------------------------------
    # GDSC2 test
    # -------------------------------------------------------------

    X_test, y_test, test_kept_df = build_X_y(
        test_df,
        scaled_dataset_a_gene_expr,
        dataset_a_drug_fingerprints,
        Response_column
    )

    xgb_test_pred = final_model.predict(
        X_test
    )

    xgb_test_metrics = compute_metrics(
        y_test,
        xgb_test_pred
    )

    xgb_test_metrics["split"] = split_idx
    xgb_test_metrics["dataset"] = Dataset_A_name

    xgb_within_results.append(
        xgb_test_metrics
    )

    xgb_predictions.append(
        pd.DataFrame({
            "dataset": Dataset_A_name,
            "split": split_idx,
            "cell_line_id": test_kept_df[
                Cell_line_identifier
            ].values,
            "drug_id": test_kept_df[
                Drug_identifier
            ].values,
            "y_actual": y_test,
            "y_predicted": xgb_test_pred
        })
    )


    # -------------------------------------------------------------
    # CTRPv2 cross-study evaluation
    # -------------------------------------------------------------

    X_dataset_b, y_dataset_b, dataset_b_kept_df = (
        build_X_y(
            dataset_b_response,
            dataset_b_scaled_gene_expr,
            dataset_b_drug_fingerprints,
            Dataset_B_response_column
        )
    )

    if len(y_dataset_b) > 0:

        xgb_dataset_b_pred = final_model.predict(
            X_dataset_b
        )

        xgb_dataset_b_metrics = compute_metrics(
            y_dataset_b,
            xgb_dataset_b_pred
        )

        xgb_predictions.append(
            pd.DataFrame({
                "dataset": Dataset_B_name,
                "split": split_idx,
                "cell_line_id": dataset_b_kept_df[
                    Cell_line_identifier
                ].values,
                "drug_id": dataset_b_kept_df[
                    Drug_identifier
                ].values,
                "y_actual": y_dataset_b,
                "y_predicted": xgb_dataset_b_pred
            })
        )

    else:

        xgb_dataset_b_metrics = {
            "R2": np.nan,
            "Pearson": np.nan,
            "Spearman": np.nan,
            "MSE": np.nan
        }


    xgb_dataset_b_metrics["split"] = split_idx
    xgb_dataset_b_metrics["dataset"] = Dataset_B_name

    xgb_cross_results.append(
        xgb_dataset_b_metrics
    )


    # -------------------------------------------------------------
    # Naive Mean Effects
    # -------------------------------------------------------------

    naive_model = NaiveMeanEffectsPredictor()

    naive_model.fit(
        trainval_df
    )


    # GDSC2 test
    naive_test_pred = naive_model.predict(
        test_kept_df
    )

    naive_test_metrics = compute_metrics(
        test_kept_df[Response_column].values,
        naive_test_pred
    )

    naive_test_metrics["split"] = split_idx
    naive_test_metrics["dataset"] = Dataset_A_name

    naive_within_results.append(
        naive_test_metrics
    )


    naive_predictions.append(
        pd.DataFrame({
            "dataset": Dataset_A_name,
            "split": split_idx,
            "cell_line_id": test_kept_df[
                Cell_line_identifier
            ].values,
            "drug_id": test_kept_df[
                Drug_identifier
            ].values,
            "y_actual": test_kept_df[
                Response_column
            ].values,
            "y_predicted": naive_test_pred
        })
    )


    # CTRPv2
    naive_dataset_b_pred = naive_model.predict(
        dataset_b_kept_df
    )

    naive_dataset_b_metrics = compute_metrics(
        dataset_b_kept_df[
            Dataset_B_response_column
        ].values,
        naive_dataset_b_pred
    )

    naive_dataset_b_metrics["split"] = split_idx
    naive_dataset_b_metrics["dataset"] = Dataset_B_name

    naive_cross_results.append(
        naive_dataset_b_metrics
    )


    naive_predictions.append(
        pd.DataFrame({
            "dataset": Dataset_B_name,
            "split": split_idx,
            "cell_line_id": dataset_b_kept_df[
                Cell_line_identifier
            ].values,
            "drug_id": dataset_b_kept_df[
                Drug_identifier
            ].values,
            "y_actual": dataset_b_kept_df[
                Dataset_B_response_column
            ].values,
            "y_predicted": naive_dataset_b_pred
        })
    )


    print(
        f"{Dataset_A_name} | "
        f"MultiViewXGBoost R2 = "
        f"{xgb_test_metrics['R2']:.4f} | "
        f"Naive R2 = "
        f"{naive_test_metrics['R2']:.4f}"
    )

    print(
        f"{Dataset_B_name} | "
        f"MultiViewXGBoost R2 = "
        f"{xgb_dataset_b_metrics['R2']:.4f} | "
        f"Naive R2 = "
        f"{naive_dataset_b_metrics['R2']:.4f}"
    )


# -------------------------------------------------------------------------
# Save predictions
# -------------------------------------------------------------------------

xgb_predictions_df = pd.concat(
    xgb_predictions,
    ignore_index=True
)

naive_predictions_df = pd.concat(
    naive_predictions,
    ignore_index=True
)


xgb_predictions_path = (
    f"{Results_root}/multiview_xgboost_predictions_all_splits.xlsx"
)

naive_predictions_path = (
    f"{Results_root}/naive_mean_predictions_all_splits.xlsx"
)


xgb_predictions_df.to_excel(
    xgb_predictions_path,
    index=False
)

naive_predictions_df.to_excel(
    naive_predictions_path,
    index=False
)


# -------------------------------------------------------------------------
# Save metrics
# -------------------------------------------------------------------------

xgb_metrics_df = pd.DataFrame(
    xgb_within_results + xgb_cross_results
)

naive_metrics_df = pd.DataFrame(
    naive_within_results + naive_cross_results
)


xgb_metrics_path = (
    f"{Results_root}/multiview_xgboost_performance_metrics.xlsx"
)

naive_metrics_path = (
    f"{Results_root}/naive_mean_performance_metrics.xlsx"
)


xgb_metrics_df.to_excel(
    xgb_metrics_path,
    index=False
)

naive_metrics_df.to_excel(
    naive_metrics_path,
    index=False
)


# -------------------------------------------------------------------------
# Direct model comparison
# -------------------------------------------------------------------------

xgb_comparison = xgb_metrics_df.copy()
xgb_comparison["model"] = "MultiViewXGBoost"

naive_comparison = naive_metrics_df.copy()
naive_comparison["model"] = "Naive Mean Effects"


comparison_df = pd.concat(
    [
        xgb_comparison,
        naive_comparison
    ],
    ignore_index=True
)


comparison_df = comparison_df[
    [
        "dataset",
        "split",
        "model",
        "R2",
        "Pearson",
        "Spearman",
        "MSE"
    ]
]


comparison_path = (
    f"{Results_root}/multiview_xgboost_vs_naive_mean.xlsx"
)


comparison_df.to_excel(
    comparison_path,
    index=False
)


# -------------------------------------------------------------------------
# Print results
# -------------------------------------------------------------------------

print("\nMultiViewXGBoost results:")
print(xgb_metrics_df)

print("\nNaive Mean Effects results:")
print(naive_metrics_df)

print("\nModel comparison:")
print(comparison_df)

print(
    f"\nResults saved to: {Results_root}"
)


# -------------------------------------------------------------------------
# Actual vs predicted plots
# -------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


for ax, dataset_name in zip(
    axes,
    [
        Dataset_A_name,
        Dataset_B_name
    ]
):

    subset = xgb_predictions_df[
        xgb_predictions_df["dataset"] == dataset_name
    ]

    ax.scatter(
        subset["y_actual"],
        subset["y_predicted"],
        alpha=0.2,
        s=8
    )

    limits = [
        subset[
            ["y_actual", "y_predicted"]
        ].min().min(),

        subset[
            ["y_actual", "y_predicted"]
        ].max().max()
    ]

    ax.plot(
        limits,
        limits,
        "r--"
    )

    ax.set_title(f"{dataset_name} - MultiViewXGBoost")

    ax.set_xlabel("Actual")

    ax.set_ylabel("Predicted")


plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------
# Mean R2 comparison
# -------------------------------------------------------------------------

mean_r2 = (
    comparison_df
    .groupby(
        ["dataset", "model"]
    )["R2"]
    .mean()
    .unstack()
)

mean_r2.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.ylabel("Mean R2")

plt.title("MultiViewXGBoost vs Naive Mean Effects")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------
# Mean MSE comparison
# -------------------------------------------------------------------------

mean_mse = (
    comparison_df
    .groupby(
        ["dataset", "model"]
    )["MSE"]
    .mean()
    .unstack()
)

mean_mse.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.ylabel("Mean MSE")

plt.title(
    "MultiViewXGBoost vs Naive Mean Effects"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()
plt.show()