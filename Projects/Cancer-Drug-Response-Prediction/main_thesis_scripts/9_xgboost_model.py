"""
Train and evaluate XGBoost and Naive Mean Effects models for
within-study and cross-study cancer drug-response prediction.

This version does not apply AUC filtering or response standardisation.

For each LCO split:

Within-study evaluation:
    - XGBoost hyperparameters are selected using training and
      validation data.
    - The final XGBoost model is trained on training + validation.
    - The final XGBoost model is evaluated on the within-study
      test set.
    - The Naive Mean Effects model is trained on training +
      validation and evaluated on the within-study test set.

Cross-study evaluation:
    - The same final XGBoost model trained on training + validation
      is evaluated on unseen cross-study datasets.
    - A separate Naive Mean Effects model is trained on training
      data only.
    - These models are evaluated on unseen cross-study datasets.

Cross-study datasets are never used for model training or
hyperparameter selection.

Model inputs:
    - processed gene expression
    - processed drug fingerprints

Feature matching:
    - cellosaurus_id for gene-expression matching
    - pubchem_id for drug-fingerprint matching
    - cell_line_name for Naive Mean Effects

Gene-expression preprocessing is performed before this script.
Drug-fingerprint preprocessing is performed before this script.

Models operate directly on LN_IC50_curvecurator during training
and prediction.

No response standardisation or inverse transformation is performed.
Final evaluation metrics are calculated directly on the original
LN_IC50_curvecurator response scale.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

RANDOM_STATE = 42

CELL_LINE_COLUMN = "cell_line_name"
CELL_ID_COLUMN = "cellosaurus_id"
DRUG_ID_COLUMN = "pubchem_id"

RESPONSE_COLUMN = "LN_IC50_curvecurator"

EXCLUDED_COLUMNS = [
    CELL_ID_COLUMN,
    DRUG_ID_COLUMN,
    RESPONSE_COLUMN,
    CELL_LINE_COLUMN,
    "tissue",
]


# ---------------------------------------------------------------------
# XGBOOST HYPERPARAMETER GRID
# ---------------------------------------------------------------------

HYPERPARAM_GRID = [
    {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 10,
        "subsample": 0.8,
        "colsample_bytree": colsample,
        "reg_alpha": reg_alpha,
        "reg_lambda": 0.1,
    }
    for colsample in [0.6, 0.8]
    for reg_alpha in [0.0, 1.0]
]


# ---------------------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate XGBoost and Naive Mean Effects "
            "models for within-study and cross-study "
            "drug-response prediction without response "
            "standardisation."
        )
    )

    parser.add_argument(
        "--project-root",
        required=True,
        help="Project root directory.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where XGBoost results will be saved.",
    )

    parser.add_argument(
        "--within-study-dataset",
        required=True,
        help="Name of the within-study dataset, e.g. GDSC2.",
    )

    parser.add_argument(
        "--cross-study-datasets",
        nargs="+",
        required=True,
        help=(
            "Names of the cross-study datasets to evaluate, "
            "e.g. GDSC1 CTRPv1 CTRPv2 CCLE."
        ),
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of LCO splits to evaluate.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:

    actual = np.asarray(
        actual,
        dtype=float,
    ).ravel()

    predicted = np.asarray(
        predicted,
        dtype=float,
    ).ravel()

    if len(actual) < 2:
        return {
            "R2": np.nan,
            "Pearson": np.nan,
            "Spearman": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
        }

    pearson_result = pearsonr(
        actual,
        predicted,
    )

    spearman_result = spearmanr(
        actual,
        predicted,
    )

    return {
        "R2": float(
            r2_score(
                actual,
                predicted,
            )
        ),
        "Pearson": float(
            pearson_result.statistic
        ),
        "Spearman": float(
            spearman_result.statistic
        ),
        "MSE": float(
            mean_squared_error(
                actual,
                predicted,
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
    }


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def load_response_data(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Response file not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = [
        CELL_LINE_COLUMN,
        CELL_ID_COLUMN,
        DRUG_ID_COLUMN,
        RESPONSE_COLUMN,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {path}:\n{missing}"
        )

    return df


def load_gene_expression(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Gene-expression file not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if CELL_ID_COLUMN not in df.columns:
        raise ValueError(
            f"{CELL_ID_COLUMN} not found in:\n{path}"
        )

    return df


def load_fingerprints(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Fingerprint file not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if DRUG_ID_COLUMN not in df.columns:
        raise ValueError(
            f"{DRUG_ID_COLUMN} not found in:\n{path}"
        )

    return df


# ---------------------------------------------------------------------
# FEATURE MERGING
# ---------------------------------------------------------------------

def prepare_model_data(
    response_df: pd.DataFrame,
    gene_expression_df: pd.DataFrame,
    fingerprints_df: pd.DataFrame,
) -> pd.DataFrame:

    response_df = response_df.copy()
    gene_expression_df = gene_expression_df.copy()
    fingerprints_df = fingerprints_df.copy()

    # -------------------------------------------------------------
    # Standardise identifier types
    # -------------------------------------------------------------

    response_df[CELL_ID_COLUMN] = (
        response_df[CELL_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    gene_expression_df[CELL_ID_COLUMN] = (
        gene_expression_df[CELL_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    response_df[DRUG_ID_COLUMN] = (
        response_df[DRUG_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    fingerprints_df[DRUG_ID_COLUMN] = (
        fingerprints_df[DRUG_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    # -------------------------------------------------------------
    # Remove duplicate identifiers before merging
    # -------------------------------------------------------------

    gene_expression_df = (
        gene_expression_df
        .drop_duplicates(
            subset=[CELL_ID_COLUMN],
            keep="first",
        )
    )

    fingerprints_df = (
        fingerprints_df
        .drop_duplicates(
            subset=[DRUG_ID_COLUMN],
            keep="first",
        )
    )

    # -------------------------------------------------------------
    # Merge gene expression
    # -------------------------------------------------------------

    merged = response_df.merge(
        gene_expression_df,
        on=CELL_ID_COLUMN,
        how="inner",
        suffixes=("", "_gene"),
    )

    # -------------------------------------------------------------
    # Merge drug fingerprints
    # -------------------------------------------------------------

    merged = merged.merge(
        fingerprints_df,
        on=DRUG_ID_COLUMN,
        how="inner",
        suffixes=("", "_drug"),
    )

    if merged.empty:
        raise ValueError(
            "No observations remained after matching response data "
            "with gene expression and drug fingerprints."
        )

    return merged


# ---------------------------------------------------------------------
# FEATURE MATRIX
# ---------------------------------------------------------------------

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:

    excluded = set(
        EXCLUDED_COLUMNS
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    if not feature_columns:
        raise ValueError(
            "No model features remain after excluding metadata columns."
        )

    return feature_columns


def build_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "The dataset is missing feature columns required by the "
            f"training data. Missing {len(missing)} columns."
        )

    X = df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    y = pd.to_numeric(
        df[RESPONSE_COLUMN],
        errors="coerce",
    )

    valid = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = (
        X.loc[valid]
        .reset_index(drop=True)
    )

    y = (
        y.loc[valid]
        .to_numpy(dtype=float)
    )

    if X.empty:
        raise ValueError(
            "No valid observations remain after feature conversion."
        )

    return X, y


# ---------------------------------------------------------------------
# XGBOOST HYPERPARAMETER TUNING
# ---------------------------------------------------------------------

def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
) -> tuple[dict, pd.DataFrame]:

    results = []

    for params in HYPERPARAM_GRID:

        model = xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            reg_alpha=params["reg_alpha"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        metrics = calculate_metrics(
            y_validation,
            predictions,
        )

        results.append(
            {
                "n_estimators": params["n_estimators"],
                "learning_rate": params["learning_rate"],
                "max_depth": params["max_depth"],
                "subsample": params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "reg_lambda": params["reg_lambda"],
                "reg_alpha": params["reg_alpha"],
                "R2": metrics["R2"],
                "Pearson": metrics["Pearson"],
                "Spearman": metrics["Spearman"],
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
            }
        )

    results_df = pd.DataFrame(
        results
    )

    best_index = results_df[
        "MSE"
    ].idxmin()

    best_params = {
        "n_estimators": int(
            results_df.loc[
                best_index,
                "n_estimators",
            ]
        ),
        "learning_rate": float(
            results_df.loc[
                best_index,
                "learning_rate",
            ]
        ),
        "max_depth": int(
            results_df.loc[
                best_index,
                "max_depth",
            ]
        ),
        "subsample": float(
            results_df.loc[
                best_index,
                "subsample",
            ]
        ),
        "colsample_bytree": float(
            results_df.loc[
                best_index,
                "colsample_bytree",
            ]
        ),
        "reg_alpha": float(
            results_df.loc[
                best_index,
                "reg_alpha",
            ]
        ),
        "reg_lambda": float(
            results_df.loc[
                best_index,
                "reg_lambda",
            ]
        ),
    }

    return (
        best_params,
        results_df,
    )


# ---------------------------------------------------------------------
# FINAL XGBOOST
# ---------------------------------------------------------------------

def train_final_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    best_params: dict,
) -> xgb.XGBRegressor:

    model = xgb.XGBRegressor(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ---------------------------------------------------------------------
# NAIVE MEAN EFFECTS
# ---------------------------------------------------------------------

class NaiveMeanEffects:

    def __init__(
        self,
        cell_line_column: str,
        drug_column: str,
        response_column: str,
    ) -> None:

        self.cell_line_column = cell_line_column
        self.drug_column = drug_column
        self.response_column = response_column

        self.global_mean = 0.0

        self.cell_effects: dict[
            str,
            float,
        ] = {}

        self.drug_effects: dict[
            str,
            float,
        ] = {}

    def fit(
        self,
        df: pd.DataFrame,
    ) -> None:

        data = df.copy()

        data[
            self.response_column
        ] = pd.to_numeric(
            data[
                self.response_column
            ],
            errors="coerce",
        )

        data = data.dropna(
            subset=[
                self.response_column
            ]
        )

        self.global_mean = float(
            data[
                self.response_column
            ].mean()
        )

        cell_means = (
            data.groupby(
                self.cell_line_column
            )[
                self.response_column
            ]
            .mean()
        )

        drug_means = (
            data.groupby(
                self.drug_column
            )[
                self.response_column
            ]
            .mean()
        )

        self.cell_effects = (
            cell_means
            - self.global_mean
        ).to_dict()

        self.drug_effects = (
            drug_means
            - self.global_mean
        ).to_dict()

    def predict(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        predictions = []

        for _, row in df.iterrows():

            cell = row[
                self.cell_line_column
            ]

            drug = row[
                self.drug_column
            ]

            cell_effect = (
                self.cell_effects.get(
                    cell,
                    0.0,
                )
            )

            drug_effect = (
                self.drug_effects.get(
                    drug,
                    0.0,
                )
            )

            prediction = (
                self.global_mean
                + cell_effect
                + drug_effect
            )

            predictions.append(
                prediction
            )

        return np.asarray(
            predictions,
            dtype=float,
        )


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------

def evaluate_model(
    model,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    dict[str, float],
    np.ndarray,
    np.ndarray,
]:

    X, y = build_feature_matrix(
        df,
        feature_columns,
    )

    predictions = model.predict(
        X
    )

    actual = y

    metrics = calculate_metrics(
        actual,
        predictions,
    )

    return (
        metrics,
        actual,
        predictions,
    )


def get_prediction_rows(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:

    rows = df.copy()

    if feature_columns is None:
        valid = pd.to_numeric(
            rows[RESPONSE_COLUMN],
            errors="coerce",
        ).notna()
        return rows.loc[valid].copy()

    X = rows[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    y = pd.to_numeric(
        rows[RESPONSE_COLUMN],
        errors="coerce",
    )
    valid = X.notna().all(axis=1) & y.notna()
    return rows.loc[valid].copy()


def make_prediction_records(
    rows: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    split_number: int,
    model_name: str,
    dataset_name: str,
) -> list[dict]:

    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()

    if len(rows) != len(actual) or len(actual) != len(predicted):
        raise ValueError(
            "Prediction rows, actual responses, and predictions are not aligned."
        )

    records = []
    for (_, row), observed, estimate in zip(
        rows.iterrows(), actual, predicted
    ):
        records.append(
            {
                "split": split_number,
                "model": model_name,
                "dataset": dataset_name,
                CELL_LINE_COLUMN: row.get(CELL_LINE_COLUMN, np.nan),
                CELL_ID_COLUMN: row.get(CELL_ID_COLUMN, np.nan),
                DRUG_ID_COLUMN: row.get(DRUG_ID_COLUMN, np.nan),
                "actual": float(observed),
                "predicted": float(estimate),
            }
        )
    return records


def evaluate_naive_mean_effects(
    model: NaiveMeanEffects,
    df: pd.DataFrame,
) -> tuple[
    dict[str, float],
    np.ndarray,
    np.ndarray,
]:

    valid = pd.to_numeric(
        df[
            RESPONSE_COLUMN
        ],
        errors="coerce",
    ).notna()

    evaluation_df = df.loc[
        valid
    ].copy()

    actual = pd.to_numeric(
        evaluation_df[
            RESPONSE_COLUMN
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    predictions = model.predict(
        evaluation_df
    )

    metrics = calculate_metrics(
        actual,
        predictions,
    )

    return (
        metrics,
        actual,
        predictions,
    )


# ---------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------

def save_actual_vs_predicted_plot(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str,
    output_path: Path,
) -> None:

    plt.figure(
        figsize=(7, 7)
    )

    plt.scatter(
        actual,
        predicted,
        alpha=0.5,
    )

    minimum = min(
        np.min(actual),
        np.min(predicted),
    )

    maximum = max(
        np.max(actual),
        np.max(predicted),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    plt.xlabel(
        "Actual LN IC50"
    )

    plt.ylabel(
        "Predicted LN IC50"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:

    args = parse_arguments()

    project_root = Path(
        args.project_root
    ).resolve()

    output_root = Path(
        args.output_dir
    ).resolve()

    within_study_dataset = (
        args.within_study_dataset
    )

    cross_study_datasets = (
        args.cross_study_datasets
    )

    # -------------------------------------------------------------
    # DATA DIRECTORIES
    # -------------------------------------------------------------

    LCO_root = (
        project_root
        / "local_scripts"
        / "Raw_datasets"
        / "results"
        / "LCO_splits"
        / within_study_dataset
    )

    unseen_cross_study_root = (
        project_root
        / "local_scripts"
        / "Raw_datasets"
        / "results"
        / "LCO_splits"
        / within_study_dataset
    )

    # UPDATED: CTRPv2 within-study gene-expression location
    within_study_gene_expression_root = (
        project_root
        / "local_scripts"
        / "Raw_datasets"
        / "results"
        / "gene_expr_transformed"
    )

    cross_study_gene_expression_root = (
        project_root
        / "data"
        / "gene_expr_transformed"
    )

    within_study_fingerprint_root = (
        project_root
        / "data"
        / "processed"
    )

    cross_study_fingerprint_root = (
        project_root
        / "data"
        / "cross_study_fingerprints"
    )

    # -------------------------------------------------------------
    # LOAD WITHIN-STUDY FEATURES
    # -------------------------------------------------------------

    # UPDATED: filename pattern is
    # gene_expression_CTRPv2_processed.csv
    within_study_gene_expression = (
        load_gene_expression(
            within_study_gene_expression_root
            / (
                f"gene_expression_{within_study_dataset}_"
                f"processed.csv"
            )
        )
    )

    within_study_fingerprints = (
        load_fingerprints(
            within_study_fingerprint_root
            / (
                f"{within_study_dataset}_"
                f"fingerprints_128_transposed.csv"
            )
        )
    )

    # -------------------------------------------------------------
    # LOAD CROSS-STUDY FEATURES
    # -------------------------------------------------------------

    cross_study_gene_expression = {}
    cross_study_fingerprints = {}

    for dataset in cross_study_datasets:

        cross_study_gene_expression[
            dataset
        ] = load_gene_expression(
            cross_study_gene_expression_root
            / (
                f"gene_expression_{dataset}_"
                f"processed.csv"
            )
        )

        cross_study_fingerprints[
            dataset
        ] = load_fingerprints(
            cross_study_fingerprint_root
            / (
                f"{dataset}_"
                f"fingerprints_128_transposed.csv"
            )
        )

    # -------------------------------------------------------------
    # OUTPUT DIRECTORY
    # -------------------------------------------------------------

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metrics = []
    all_prediction_records = []

    # =============================================================
    # PROCESS LCO SPLITS
    # =============================================================

    for split_number in range(
        1,
        args.n_splits + 1,
    ):

        print()
        print("=" * 70)
        print(
            f"PROCESSING LCO SPLIT "
            f"{split_number}"
        )
        print("=" * 70)

        split_dir = (
            LCO_root
            / f"split_{split_number}"
        )

        split_output = (
            output_root
            / f"split_{split_number}"
        )

        split_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------
        # LOAD WITHIN-STUDY RESPONSE DATA
        # ---------------------------------------------------------

        train_df = load_response_data(
            split_dir
            / "train.csv"
        )

        validation_df = load_response_data(
            split_dir
            / "validation.csv"
        )

        test_df = load_response_data(
            split_dir
            / "test.csv"
        )

        print(
            f"{within_study_dataset} train observations: "
            f"{len(train_df)}"
        )

        print(
            f"{within_study_dataset} validation observations: "
            f"{len(validation_df)}"
        )

        print(
            f"{within_study_dataset} test observations: "
            f"{len(test_df)}"
        )

        # ---------------------------------------------------------
        # PREPARE WITHIN-STUDY MODEL DATA
        # ---------------------------------------------------------

        train_model_df = prepare_model_data(
            train_df,
            within_study_gene_expression,
            within_study_fingerprints,
        )

        validation_model_df = prepare_model_data(
            validation_df,
            within_study_gene_expression,
            within_study_fingerprints,
        )

        test_model_df = prepare_model_data(
            test_df,
            within_study_gene_expression,
            within_study_fingerprints,
        )

        # ---------------------------------------------------------
        # DETERMINE FEATURE COLUMNS FROM TRAINING DATA
        # ---------------------------------------------------------

        feature_columns = get_feature_columns(
            train_model_df
        )

        print(
            f"Number of XGBoost features: "
            f"{len(feature_columns)}"
        )

        # ---------------------------------------------------------
        # BUILD TRAINING MATRICES
        # ---------------------------------------------------------

        X_train, y_train = build_feature_matrix(
            train_model_df,
            feature_columns,
        )

        X_validation, y_validation = (
            build_feature_matrix(
                validation_model_df,
                feature_columns,
            )
        )

        print(
            f"Model training observations: "
            f"{len(y_train)}"
        )

        print(
            f"Validation observations: "
            f"{len(y_validation)}"
        )

        # ---------------------------------------------------------
        # XGBOOST HYPERPARAMETER TUNING
        # ---------------------------------------------------------

        (
            best_params,
            tuning_results,
        ) = tune_xgboost(
            X_train,
            y_train,
            X_validation,
            y_validation,
        )

        tuning_results.to_csv(
            split_output
            / "xgboost_hyperparameter_tuning.csv",
            index=False,
        )

        print(
            "Best XGBoost hyperparameters:"
        )

        print(
            best_params
        )

        # ---------------------------------------------------------
        # TRAIN + VALIDATION
        # ---------------------------------------------------------

        train_validation_df = pd.concat(
            [
                train_model_df,
                validation_model_df,
            ],
            ignore_index=True,
        )

        X_train_validation, y_train_validation = (
            build_feature_matrix(
                train_validation_df,
                feature_columns,
            )
        )

        # ---------------------------------------------------------
        # WITHIN-STUDY XGBOOST
        # ---------------------------------------------------------

        xgb_model = train_final_xgboost(
            X_train_validation,
            y_train_validation,
            best_params,
        )

        with open(
            split_output
            / "xgboost_model.pkl",
            "wb",
        ) as file:

            pickle.dump(
                xgb_model,
                file,
            )

        # ---------------------------------------------------------
        # WITHIN-STUDY XGBOOST TEST
        # ---------------------------------------------------------

        (
            xgb_metrics,
            actual,
            predictions,
        ) = evaluate_model(
            xgb_model,
            test_model_df,
            feature_columns,
        )

        all_prediction_records.extend(
            make_prediction_records(
                get_prediction_rows(test_model_df, feature_columns),
                actual,
                predictions,
                split_number,
                "XGBoost",
                f"{within_study_dataset}_test",
            )
        )

        xgb_metrics.update(
            {
                "split": split_number,
                "model": "XGBoost",
                "dataset": (
                    f"{within_study_dataset}_test"
                ),
            }
        )

        all_metrics.append(
            xgb_metrics
        )

        save_actual_vs_predicted_plot(
            actual,
            predictions,
            (
                f"XGBoost - "
                f"{within_study_dataset} Test - "
                f"Split {split_number}"
            ),
            split_output
            / (
                f"XGBoost_"
                f"{within_study_dataset}_test_"
                f"actual_vs_predicted.png"
            ),
        )

        print(
            f"XGBoost - "
            f"{within_study_dataset} test: "
            f"{xgb_metrics}"
        )

        # ---------------------------------------------------------
        # WITHIN-STUDY NAIVE MEAN EFFECTS
        # ---------------------------------------------------------

        naive_model = NaiveMeanEffects(
            cell_line_column=CELL_LINE_COLUMN,
            drug_column=DRUG_ID_COLUMN,
            response_column=RESPONSE_COLUMN,
        )

        naive_model.fit(
            train_validation_df
        )

        with open(
            split_output
            / "naive_mean_effects_model.pkl",
            "wb",
        ) as file:

            pickle.dump(
                naive_model,
                file,
            )

        (
            naive_metrics,
            actual,
            predictions,
        ) = evaluate_naive_mean_effects(
            naive_model,
            test_model_df,
        )

        all_prediction_records.extend(
            make_prediction_records(
                get_prediction_rows(test_model_df),
                actual,
                predictions,
                split_number,
                "Naive Mean Effects",
                f"{within_study_dataset}_test",
            )
        )

        naive_metrics.update(
            {
                "split": split_number,
                "model": "Naive Mean Effects",
                "dataset": (
                    f"{within_study_dataset}_test"
                ),
            }
        )

        all_metrics.append(
            naive_metrics
        )

        save_actual_vs_predicted_plot(
            actual,
            predictions,
            (
                f"Naive Mean Effects - "
                f"{within_study_dataset} Test - "
                f"Split {split_number}"
            ),
            split_output
            / (
                f"Naive_"
                f"{within_study_dataset}_test_"
                f"actual_vs_predicted.png"
            ),
        )

        print(
            f"Naive Mean Effects - "
            f"{within_study_dataset} test: "
            f"{naive_metrics}"
        )

        # ---------------------------------------------------------
        # CROSS-STUDY NAIVE MEAN EFFECTS
        #
        # Training data only.
        # ---------------------------------------------------------

        cross_study_naive_model = NaiveMeanEffects(
            cell_line_column=CELL_LINE_COLUMN,
            drug_column=DRUG_ID_COLUMN,
            response_column=RESPONSE_COLUMN,
        )

        cross_study_naive_model.fit(
            train_model_df
        )

        with open(
            split_output
            / "cross_study_naive_mean_effects_model.pkl",
            "wb",
        ) as file:

            pickle.dump(
                cross_study_naive_model,
                file,
            )

        # =========================================================
        # CROSS-STUDY EVALUATION
        # =========================================================

        for dataset in cross_study_datasets:

            print()
            print(
                f"Cross-study evaluation: "
                f"{dataset}"
            )

            # -----------------------------------------------------
            # LOAD UNSEEN CROSS-STUDY RESPONSE DATA
            #
            # Files are stored in:
            #  local_scripts/Raw_datasets/results/LCO_splits/
            #
            # Filenames are:
            # dataset_unseen.csv
            # -----------------------------------------------------

            cross_study_dir = (
                unseen_cross_study_root
                / f"split_{split_number}"
            )

            cross_study_df = load_response_data(
                cross_study_dir
                / f"{dataset}_unseen.csv"
            )

            # -----------------------------------------------------
            # PREPARE CROSS-STUDY MODEL DATA
            # -----------------------------------------------------

            cross_study_model_df = (
                prepare_model_data(
                    cross_study_df,
                    cross_study_gene_expression[
                        dataset
                    ],
                    cross_study_fingerprints[
                        dataset
                    ],
                )
            )

            # -----------------------------------------------------
            # CROSS-STUDY XGBOOST
            # -----------------------------------------------------

            (
                cross_study_xgb_metrics,
                actual,
                predictions,
            ) = evaluate_model(
                xgb_model,
                cross_study_model_df,
                feature_columns,
            )

            all_prediction_records.extend(
                make_prediction_records(
                    get_prediction_rows(cross_study_model_df, feature_columns),
                    actual,
                    predictions,
                    split_number,
                    "XGBoost",
                    dataset,
                )
            )

            cross_study_xgb_metrics.update(
                {
                    "split": split_number,
                    "model": "XGBoost",
                    "dataset": dataset,
                }
            )

            all_metrics.append(
                cross_study_xgb_metrics
            )

            save_actual_vs_predicted_plot(
                actual,
                predictions,
                (
                    f"XGBoost - {dataset} - "
                    f"Split {split_number}"
                ),
                split_output
                / (
                    f"XGBoost_{dataset}_"
                    f"actual_vs_predicted.png"
                ),
            )

            # -----------------------------------------------------
            # CROSS-STUDY NAIVE MEAN EFFECTS
            # -----------------------------------------------------

            (
                cross_study_naive_metrics,
                actual,
                predictions,
            ) = evaluate_naive_mean_effects(
                cross_study_naive_model,
                cross_study_model_df,
            )

            all_prediction_records.extend(
                make_prediction_records(
                    get_prediction_rows(cross_study_model_df),
                    actual,
                    predictions,
                    split_number,
                    "Naive Mean Effects",
                    dataset,
                )
            )

            cross_study_naive_metrics.update(
                {
                    "split": split_number,
                    "model": "Naive Mean Effects",
                    "dataset": dataset,
                }
            )

            all_metrics.append(
                cross_study_naive_metrics
            )

            save_actual_vs_predicted_plot(
                actual,
                predictions,
                (
                    f"Naive Mean Effects - "
                    f"{dataset} - "
                    f"Split {split_number}"
                ),
                split_output
                / (
                    f"Naive_{dataset}_"
                    f"actual_vs_predicted.png"
                ),
            )

            print(
                f"{dataset} | "
                f"XGBoost R2 = "
                f"{cross_study_xgb_metrics['R2']:.4f} | "
                f"Naive R2 = "
                f"{cross_study_naive_metrics['R2']:.4f}"
            )

    # =============================================================
    # SAVE ALL RESULTS
    # =============================================================

    metrics_df = pd.DataFrame(
        all_metrics
    )

    metrics_df = metrics_df[
        [
            "split",
            "model",
            "dataset",
            "R2",
            "Pearson",
            "Spearman",
            "MSE",
            "RMSE",
        ]
    ]

    metrics_df.to_csv(
        output_root
        / "xgboost_naive_mean_effects_results.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # SAVE INDIVIDUAL ACTUAL AND PREDICTED RESPONSES
    # -------------------------------------------------------------

    predictions_df = pd.DataFrame(
        all_prediction_records
    )

    predictions_df.to_csv(
        output_root / "xgboost_naive_mean_effects_predictions.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # SAVE MEAN +/- SD PERFORMANCE ACROSS LCO SPLITS
    # -------------------------------------------------------------

    metric_columns = [
        "R2",
        "Pearson",
        "Spearman",
        "MSE",
        "RMSE",
    ]

    summary_df = (
        metrics_df
        .groupby(["model", "dataset"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary_df.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in summary_df.columns
    ]

    summary_df.to_excel(
        output_root / "xgboost_naive_mean_effects_performance_mean_sd.xlsx",
        index=False,
    )

    # -------------------------------------------------------------
    # SAVE COMBINED SCATTER PLOTS ACROSS ALL AVAILABLE SPLITS
    # -------------------------------------------------------------

    combined_plot_root = output_root / "combined_scatterplots"
    combined_plot_root.mkdir(parents=True, exist_ok=True)

    for (model_name, dataset_name), group in predictions_df.groupby(
        ["model", "dataset"],
        dropna=False,
    ):
        safe_model = str(model_name).replace(" ", "_")
        safe_dataset = str(dataset_name).replace(" ", "_")
        save_actual_vs_predicted_plot(
            group["actual"].to_numpy(dtype=float),
            group["predicted"].to_numpy(dtype=float),
            f"{model_name} - {dataset_name} - Combined LCO Splits",
            combined_plot_root
            / f"{safe_model}_{safe_dataset}_combined_actual_vs_predicted.png",
        )

    print()
    print("=" * 70)
    print(
        "XGBOOST AND NAIVE MEAN EFFECTS COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Results saved to:"
    )

    print(
        output_root
    )


# ---------------------------------------------------------------------
# RUN SCRIPT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
