"""
Script 04: Train and evaluate a Random Forest model for cancer drug-response prediction.

Purpose
-------
Trains a Random Forest regression model using cell-line gene-expression features
and drug molecular fingerprints. The model is trained and evaluated using the
existing GDSC2 LCO split sets and is also evaluated on an external CTRPv2
dataset containing cell lines unseen in GDSC2.

For each GDSC2 LCO split:
1. Load the train, validation, and test data.
2. Scale gene-expression features using the training data.
3. Train Random Forest models with different maximum-depth values.
4. Select the best hyperparameters using validation MSE.
5. Retrain the final model using the combined training and validation data.
6. Evaluate the final model on the GDSC2 LCO test set.
7. Evaluate the final model on the external CTRPv2 unseen-cell-line dataset.
8. Calculate R², Pearson correlation, Spearman correlation, and MSE.
9. Save predictions and performance metrics to Excel files.
10. Generate plots comparing model performance.

The script does NOT:
1. Perform LCO splitting.
2. Create or modify the response datasets.
3. Generate gene-expression features.
4. Generate drug fingerprints.
5. Perform LPO, LDO, or LTO splitting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# -------------------------------------------------------------------------
# Configuration and folder layout
# -------------------------------------------------------------------------

Data_root = (
    "/Users/manellamukangwa/Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction/data"
)


# Dataset A: training/reference dataset
Dataset_A_name = "GDSC2"

Dataset_A_gene_expression = (
    f"{Data_root}/raw/GDSC2/gene_expression.csv"
)

Dataset_A_drug_fingerprints = (
    f"{Data_root}/processed/GDSC2_fingerprints_128_transposed.csv"
)

# GDSC2 LCO split sets created by Script 03
Dataset_A_lco_splits_dir = (
    f"{Data_root}/processed/lco_splits_gdsc2"
)


# Dataset B: external/unseen dataset
Dataset_B_name = "CTRPv2"

# CTRPv2 cell lines that are unseen in GDSC2
Dataset_B_response_unseen = (
    f"{Data_root}/processed/CTRPv2_response_unseen_in_GDSC2.csv"
)

Dataset_B_gene_expression = (
    f"{Data_root}/raw/CTRPv2/gene_expression.csv"
)

Dataset_B_drug_fingerprints = (
    f"{Data_root}/processed/CTRPv2_fingerprints_128_transposed.csv"
)


# Gene list used for both datasets
Gene_list_path = (
    f"{Data_root}/meta/gene_lists/landmark_genes_reduced.csv"
)

Cell_line_identifier = "cell_line_name"
Drug_identifier = "pubchem_id"

# GDSC2 uses the standardised response scale
Response_column = "LN_IC50_standardised"

# CTRPv2 uses the standardised response, scaled using the GDSC2-fitted scaler
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

os.makedirs(
    Results_root,
    exist_ok=True
)


# -------------------------------------------------------------------------
# DrEval Random Forest hyperparameters
# -------------------------------------------------------------------------

Hyperparam_grid = [
    {
        "n_estimators": 100,
        "max_depth": d,
        "max_samples": 0.2,
        "criterion": "squared_error"
    }
    for d in [5, 10, 30]
]

RF_random_state = 42


# -------------------------------------------------------------------------
# Cell-line features: gene expression view
# -------------------------------------------------------------------------

def load_gene_expression_features(
    gene_expression_csv,
    gene_list_csv
):

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
    )["Symbol"].tolist()

    missing_genes = [
        g
        for g in gene_list
        if g not in gene_expression.columns
    ]

    if missing_genes:

        raise ValueError(
            f"{len(missing_genes)} genes from list are missing in "
            f"{gene_expression_csv}. First few missing: "
            f"{missing_genes[:10]}"
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
        cl: gene_expression.loc[cl].to_numpy(
            dtype=float
        )
        for cl in gene_expression.index
    }


# -------------------------------------------------------------------------
# Drug features: fingerprint view
# -------------------------------------------------------------------------

def load_drug_fingerprint_features(
    transposed_fingerprint_csv
):

    drug_fingerprint = pd.read_csv(
        transposed_fingerprint_csv,
        index_col=Drug_identifier
    )

    drug_fingerprint.index = (
        drug_fingerprint.index.astype(str)
    )

    return {
        drug: drug_fingerprint.loc[drug].to_numpy(
            dtype=float
        )
        for drug in drug_fingerprint.index
    }


# -------------------------------------------------------------------------
# Response data: existing LCO splits + response tables
# -------------------------------------------------------------------------

def load_split_part(
    splits_dir,
    split_idx,
    part
):

    path = os.path.join(
        splits_dir,
        f"cv_split_{split_idx}_{part}.csv"
    )

    return pd.read_csv(
        path,
        dtype={
            Drug_identifier: str,
            Cell_line_identifier: str
        }
    )


def load_response_table(
    path,
    response_column=Response_column
):

    df = pd.read_csv(
        path,
        dtype={
            Drug_identifier: str,
            Cell_line_identifier: str
        },
        low_memory=False,
    )

    df = df[
        [
            Cell_line_identifier,
            Drug_identifier,
            response_column
        ]
    ].copy()

    df = df.dropna(
        subset=[response_column]
    )

    df = df.drop_duplicates(
        subset=[
            Cell_line_identifier,
            Drug_identifier
        ]
    )

    return df


# -------------------------------------------------------------------------
# Gene expression scaling: arcsinh, then StandardScaler
# -------------------------------------------------------------------------

def arcsinh_transform(
    gene_expression
):

    return {
        cl: np.arcsinh(vec)
        for cl, vec in gene_expression.items()
    }


def fit_and_scale_gene_expression(
    gene_expression,
    fit_cell_lines
):

    arcsinh_expression = arcsinh_transform(
        gene_expression
    )

    fit_matrix = np.vstack([
        arcsinh_expression[cl]
        for cl in fit_cell_lines
        if cl in arcsinh_expression
    ])

    scaler = StandardScaler()

    scaler.fit(
        fit_matrix
    )

    scaled_expression = {
        cl: scaler.transform(
            vec.reshape(1, -1)
        )[0]
        for cl, vec in arcsinh_expression.items()
    }

    return (
        scaled_expression,
        scaler
    )


def scale_with_fitted_Scaler(
    gene_expression,
    scaler
):

    arcsinh_expression = arcsinh_transform(
        gene_expression
    )

    return {
        cl: scaler.transform(
            vec.reshape(1, -1)
        )[0]
        for cl, vec in arcsinh_expression.items()
    }


# -------------------------------------------------------------------------
# Building the input matrix: concatenating the two views
# -------------------------------------------------------------------------

def build_X_y(
    df,
    gene_expression,
    drug_fingerprints,
    response_column=Response_column
):

    X_rows = []
    y_rows = []
    kept_idx = []

    for i, row in df.iterrows():

        cl = row[
            Cell_line_identifier
        ]

        drug = row[
            Drug_identifier
        ]

        if cl not in gene_expression:
            continue

        if drug not in drug_fingerprints:
            continue

        X_rows.append(
            np.concatenate(
                [
                    gene_expression[cl],
                    drug_fingerprints[drug]
                ]
            )
        )

        y_rows.append(
            row[response_column]
        )

        kept_idx.append(i)

    X = (
        np.vstack(X_rows)
        if X_rows
        else np.empty((0,))
    )

    y = np.array(
        y_rows,
        dtype=float
    )

    return (
        X,
        y,
        df.loc[
            kept_idx
        ].reset_index(drop=True)
    )


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

def compute_metrics(
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
        ),
    }


# -------------------------------------------------------------------------
# Load features once
# -------------------------------------------------------------------------

print(
    f"Loading {Dataset_A_name} gene expression features ..."
)

dataset_a_gene_expr = (
    load_gene_expression_features(
        Dataset_A_gene_expression,
        Gene_list_path
    )
)


print(
    f"Loading {Dataset_A_name} drug fingerprints "
    "(already transposed) ..."
)

dataset_a_drug_fingerprints = (
    load_drug_fingerprint_features(
        Dataset_A_drug_fingerprints
    )
)


print(
    f"Loading {Dataset_B_name} gene expression features ..."
)

dataset_b_gene_expr = (
    load_gene_expression_features(
        Dataset_B_gene_expression,
        Gene_list_path
    )
)


print(
    f"Loading {Dataset_B_name} drug fingerprints "
    "(already transposed) ..."
)

dataset_b_drug_fingerprints = (
    load_drug_fingerprint_features(
        Dataset_B_drug_fingerprints
    )
)


print(
    f"Loading {Dataset_B_name} response table "
    "(restricted to cell lines unseen in "
    f"{Dataset_A_name}) ..."
)

dataset_b_response = (
    load_response_table(
        Dataset_B_response_unseen,
        response_column=Dataset_B_response_column
    )
)


# -------------------------------------------------------------------------
# Main loop: train, tune, test, and cross-study evaluate
# -------------------------------------------------------------------------

within_study_results = []
cross_study_results = []

all_predictions = []


for split_idx in range(n_splits):

    print(
        f"\n================= LCO SPLIT {split_idx} ================="
    )

    # -------------------------------------------------------------
    # Load GDSC2 LCO train, validation and test data
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

    train_cell_lines = train_df[
        Cell_line_identifier
    ].unique()


    # -------------------------------------------------------------
    # Scale GDSC2 gene expression using training cell lines
    # -------------------------------------------------------------

    scaled_dataset_a_gene_expr, _ = (
        fit_and_scale_gene_expression(
            dataset_a_gene_expr,
            train_cell_lines
        )
    )


    # -------------------------------------------------------------
    # Build training and validation matrices
    # -------------------------------------------------------------

    X_train, y_train, _ = build_X_y(
        train_df,
        scaled_dataset_a_gene_expr,
        dataset_a_drug_fingerprints
    )

    X_val, y_val, _ = build_X_y(
        validation_df,
        scaled_dataset_a_gene_expr,
        dataset_a_drug_fingerprints
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

        model = RandomForestRegressor(
            random_state=RF_random_state,
            n_jobs=-1,
            **params
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

        print(
            f"  hyperparams {params} -> "
            f"validation MSE = {val_mse:.4f}"
        )

        if val_mse < best_val_mse:

            best_val_mse = val_mse
            best_params = params


    print(
        f"Best hyperparameters: {best_params}"
    )


    # -------------------------------------------------------------
    # Combine GDSC2 train + validation
    # -------------------------------------------------------------

    trainval_df = pd.concat(
        [
            train_df,
            validation_df
        ],
        ignore_index=True
    )

    trainval_cell_lines = trainval_df[
        Cell_line_identifier
    ].unique()


    # -------------------------------------------------------------
    # Refit scaler using GDSC2 train + validation cell lines
    # -------------------------------------------------------------

    final_scaled_dataset_a_gene_expr, final_scaler = (
        fit_and_scale_gene_expression(
            dataset_a_gene_expr,
            trainval_cell_lines
        )
    )


    # -------------------------------------------------------------
    # Build final training matrix
    # -------------------------------------------------------------

    X_trainval, y_trainval, _ = build_X_y(
        trainval_df,
        final_scaled_dataset_a_gene_expr,
        dataset_a_drug_fingerprints
    )


    # -------------------------------------------------------------
    # Train final Random Forest model
    # -------------------------------------------------------------

    final_model = RandomForestRegressor(
        random_state=RF_random_state,
        n_jobs=-1,
        **best_params
    )

    final_model.fit(
        X_trainval,
        y_trainval
    )


    # -------------------------------------------------------------
    # Evaluate on GDSC2 LCO test set
    # -------------------------------------------------------------

    X_test, y_test, test_kept_df = build_X_y(
        test_df,
        final_scaled_dataset_a_gene_expr,
        dataset_a_drug_fingerprints
    )

    test_pred = final_model.predict(
        X_test
    )

    test_metrics = compute_metrics(
        y_test,
        test_pred
    )

    test_metrics["split"] = split_idx

    within_study_results.append(
        test_metrics
    )

    print(
        f"Test metrics (within-study, "
        f"{Dataset_A_name}): {test_metrics}"
    )


    # -------------------------------------------------------------
    # Save GDSC2 test predictions
    # -------------------------------------------------------------

    all_predictions.append(
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
            "y_predicted": test_pred,
        })
    )


    # -------------------------------------------------------------
    # Scale CTRPv2 using the GDSC2-trained scaler
    # -------------------------------------------------------------

    dataset_b_scaled_gene_expression = (
        scale_with_fitted_Scaler(
            dataset_b_gene_expr,
            final_scaler
        )
    )


    # -------------------------------------------------------------
    # Build CTRPv2 unseen-cell-line matrix
    # -------------------------------------------------------------

    X_dataset_b, y_dataset_b, dataset_b_kept_df = (
        build_X_y(
            dataset_b_response,
            dataset_b_scaled_gene_expression,
            dataset_b_drug_fingerprints,
            response_column=Dataset_B_response_column
        )
    )


    # -------------------------------------------------------------
    # Evaluate final GDSC2 model on unseen CTRPv2 cells
    # -------------------------------------------------------------

    if len(y_dataset_b) > 0:

        dataset_b_pred = final_model.predict(
            X_dataset_b
        )

        dataset_b_metrics = compute_metrics(
            y_dataset_b,
            dataset_b_pred
        )

        all_predictions.append(
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
                "y_predicted": dataset_b_pred,
            })
        )

    else:

        dataset_b_metrics = {
            "R2": np.nan,
            "Pearson": np.nan,
            "Spearman": np.nan,
            "MSE": np.nan
        }


    dataset_b_metrics["split"] = split_idx

    cross_study_results.append(
        dataset_b_metrics
    )

    print(
        f"Cross-study metrics "
        f"({Dataset_B_name}, unseen cells): "
        f"{dataset_b_metrics}"
    )


# -------------------------------------------------------------------------
# Combine all predictions
# -------------------------------------------------------------------------

all_predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)


# -------------------------------------------------------------------------
# Build metrics tables
# -------------------------------------------------------------------------

within_metrics_df = pd.DataFrame(
    within_study_results
)

within_metrics_df["dataset"] = Dataset_A_name


cross_metrics_df = pd.DataFrame(
    cross_study_results
)

cross_metrics_df["dataset"] = Dataset_B_name


all_metrics_df = pd.concat(
    [
        within_metrics_df,
        cross_metrics_df
    ],
    ignore_index=True
)


all_metrics_df = all_metrics_df[
    [
        "dataset",
        "split",
        "R2",
        "Pearson",
        "Spearman",
        "MSE"
    ]
]


within_df = within_metrics_df.set_index(
    "split"
)

cross_df = cross_metrics_df.set_index(
    "split"
)


# -------------------------------------------------------------------------
# Save predictions and metrics to Excel
# -------------------------------------------------------------------------

predictions_xlsx_path = (
    f"{Results_root}/rf_predictions_all_splits.xlsx"
)

all_predictions_df.to_excel(
    predictions_xlsx_path,
    index=False
)

print(
    f"\nSaved predictions to: "
    f"{predictions_xlsx_path}"
)


metrics_excel_path = (
    f"{Results_root}/rf_performance_metrics.xlsx"
)

all_metrics_df.to_excel(
    metrics_excel_path,
    index=False
)

print(
    f"Saved performance metrics to: "
    f"{metrics_excel_path}"
)

print(
    all_metrics_df
)


# -------------------------------------------------------------------------
# Plot: Actual vs Predicted scatter
# -------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

for ax, dataset_name in zip(
    axes,
    [Dataset_A_name, Dataset_B_name]
):

    subset = all_predictions_df[
        all_predictions_df["dataset"] == dataset_name
    ]

    ax.scatter(
        subset["y_actual"],
        subset["y_predicted"],
        alpha=0.2,
        s=8
    )

    lims = [
        subset[
            ["y_actual", "y_predicted"]
        ].min().min(),

        subset[
            ["y_actual", "y_predicted"]
        ].max().max()
    ]

    ax.plot(
        lims,
        lims,
        "r--"
    )

    ax.set_title(
        f"{dataset_name} - Actual vs Predicted "
        "(all 5 splits)"
    )

    ax.set_xlabel(
        "Actual"
    )

    ax.set_ylabel(
        "Predicted"
    )

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------
# Plot: Performance across the 5 splits
# -------------------------------------------------------------------------

metrics_to_plot = [
    "R2",
    "Pearson",
    "Spearman",
    "MSE"
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8)
)

axes = axes.flatten()

for ax, metric in zip(
    axes,
    metrics_to_plot
):

    ax.plot(
        within_df.index,
        within_df[metric],
        marker="o",
        label=f"{Dataset_A_name} (within-study)"
    )

    ax.plot(
        cross_df.index,
        cross_df[metric],
        marker="o",
        label=f"{Dataset_B_name} (cross-study)"
    )

    ax.set_title(
        metric
    )

    ax.set_xlabel(
        "LCO split"
    )

    ax.set_ylabel(
        metric
    )

    ax.legend()

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------
# Plot: Dataset A vs Dataset B comparison
# -------------------------------------------------------------------------

comparison_df = pd.DataFrame({

    f"{Dataset_A_name} (within-study)":
        within_df[
            ["R2", "Pearson", "Spearman"]
        ].mean(),

    f"{Dataset_B_name} (cross-study)":
        cross_df[
            ["R2", "Pearson", "Spearman"]
        ].mean(),

})

comparison_df.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title(
    f"Mean performance: {Dataset_A_name} "
    f"(within-study) vs {Dataset_B_name} "
    "(cross-study)"
)

plt.ylabel(
    "Score"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()
plt.show()

print(
    comparison_df
)


# -------------------------------------------------------------------------
# Plot: MSE comparison averaged across the 5 LCO splits
# -------------------------------------------------------------------------

mse_comparison_df = pd.DataFrame({

    f"{Dataset_A_name} (within-study)":
        [within_df["MSE"].mean()],

    f"{Dataset_B_name} (cross-study)":
        [cross_df["MSE"].mean()],

})

mse_comparison_df.plot(
    kind="bar",
    figsize=(6, 5)
)

plt.title(
    f"Mean MSE: {Dataset_A_name} "
    f"(within-study) vs {Dataset_B_name} "
    "(cross-study)"
)

plt.ylabel(
    "MSE"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()
plt.show()

print(
    mse_comparison_df
)