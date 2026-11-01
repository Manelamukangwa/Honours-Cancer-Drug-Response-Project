"""
Train and evaluate Random Forest and Naive Mean Effects models
for within-study and cross-study cancer drug-response prediction.

This version does not apply AUC filtering or response standardisation.

For each LCO split:
- Random Forest hyperparameters are selected using training and
  validation data.
- The final within-study Random Forest is trained on training +
  validation data and evaluated on the within-study test set.
- The same final Random Forest, trained on training + validation
  data, is evaluated on unseen cross-study cell lines.
- The within-study Naive Mean Effects model is trained on training +
  validation data and evaluated on the within-study test set.
- The cross-study Naive Mean Effects model is trained on training
  data only and evaluated on unseen cross-study cell lines.

Cross-study datasets are used only for evaluation and are never
used for model training.

All models use the original LN_IC50_curvecurator response values
directly.

No response standardisation or inverse transformation is performed.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
MAX_DEPTH_VALUES = [5, 10, 30]
N_ESTIMATORS = 300
MAX_SAMPLES = 0.2
CRITERION = "squared_error"

CELL_LINE_COLUMN = "cell_line_name"
CELL_ID_COLUMN = "cellosaurus_id"
DRUG_ID_COLUMN = "pubchem_id"

RESPONSE_COLUMN = "LN_IC50_curvecurator"

EXCLUDED_COLUMNS = {
    CELL_LINE_COLUMN,
    CELL_ID_COLUMN,
    DRUG_ID_COLUMN,
    RESPONSE_COLUMN,
    "tissue",
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate Random Forest and Naive Mean "
            "Effects models for within-study and cross-study "
            "drug-response prediction without response "
            "standardisation."
        )
    )

    parser.add_argument(
        "--project-root",
        required=True,
        help="Project root directory containing the data directories.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where model results and figures will be saved.",
    )

    parser.add_argument(
        "--within-study-dataset",
        required=True,
        help="Name of the within-study dataset, e.g. CTRPv2.",
    )

    parser.add_argument(
        "--cross-study-datasets",
        nargs="+",
        required=True,
        help=(
            "Names of the cross-study datasets to evaluate, "
            "e.g. GDSC1 GDSC2 CTRPv1 CCLE."
        ),
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of LCO splits to evaluate.",
    )

    return parser.parse_args()


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)

    pearson = pearsonr(y_true, y_pred)[0]
    spearman = spearmanr(y_true, y_pred)[0]

    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    return {
        "R2": r2,
        "Pearson": pearson,
        "Spearman": spearman,
        "MSE": mse,
        "RMSE": rmse
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_response_data(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Response file not found: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = {
        CELL_LINE_COLUMN,
        CELL_ID_COLUMN,
        DRUG_ID_COLUMN,
        RESPONSE_COLUMN,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing)}"
        )

    return df


def load_gene_expression(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Gene-expression file not found: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if CELL_ID_COLUMN not in df.columns:
        raise ValueError(
            f"{CELL_ID_COLUMN} not found in gene-expression file: {path}"
        )

    return df


def load_fingerprints(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Fingerprint file not found: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if DRUG_ID_COLUMN not in df.columns:
        raise ValueError(
            f"{DRUG_ID_COLUMN} not found in fingerprint file: {path}"
        )

    return df


# ============================================================
# PREPARE MODEL DATA
# ============================================================

def prepare_model_data(
    response_df,
    gene_expression_df,
    fingerprint_df,
):

    response_df = response_df.copy()
    gene_expression_df = gene_expression_df.copy()
    fingerprint_df = fingerprint_df.copy()

    response_df[CELL_ID_COLUMN] = (
        response_df[CELL_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    response_df[DRUG_ID_COLUMN] = (
        response_df[DRUG_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    gene_expression_df[CELL_ID_COLUMN] = (
        gene_expression_df[CELL_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    fingerprint_df[DRUG_ID_COLUMN] = (
        fingerprint_df[DRUG_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    gene_expression_df = gene_expression_df.drop_duplicates(
        subset=[CELL_ID_COLUMN]
    )

    fingerprint_df = fingerprint_df.drop_duplicates(
        subset=[DRUG_ID_COLUMN]
    )

    merged = response_df.merge(
        gene_expression_df,
        on=CELL_ID_COLUMN,
        how="inner",
        suffixes=("", "_gene"),
    )

    merged = merged.merge(
        fingerprint_df,
        on=DRUG_ID_COLUMN,
        how="inner",
        suffixes=("", "_drug"),
    )

    return merged


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_columns(df):

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if not feature_columns:
        raise ValueError(
            "No feature columns were found."
        )

    return feature_columns


def build_feature_matrix(
    df,
    feature_columns,
):

    X = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    y = pd.to_numeric(
        df[RESPONSE_COLUMN],
        errors="coerce",
    )

    valid_rows = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    return X, y


# ============================================================
# RANDOM FOREST TUNING
# ============================================================

def tune_random_forest(
    X_train,
    y_train,
    X_validation,
    y_validation,
):

    results = []

    best_depth = None
    best_r2 = -np.inf

    for max_depth in MAX_DEPTH_VALUES:

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=max_depth,
            max_samples=MAX_SAMPLES,
            criterion=CRITERION,
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
                "max_depth": max_depth,
                **metrics,
            }
        )

        if metrics["R2"] > best_r2:

            best_r2 = metrics["R2"]
            best_depth = max_depth

    return best_depth, pd.DataFrame(results)


# ============================================================
# FINAL RANDOM FOREST
# ============================================================

def train_final_random_forest(
    X_train,
    y_train,
    best_depth,
):

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=best_depth,
        max_samples=MAX_SAMPLES,
        criterion=CRITERION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# NAIVE MEAN EFFECTS
# ============================================================

class NaiveMeanEffects:

    def __init__(
        self,
        cell_line_column,
        drug_column,
        response_column,
    ):

        self.cell_line_column = cell_line_column
        self.drug_column = drug_column
        self.response_column = response_column

        self.global_mean = None
        self.cell_effects = None
        self.drug_effects = None

    def fit(self, df):

        self.global_mean = df[
            self.response_column
        ].mean()

        cell_means = (
            df.groupby(
                self.cell_line_column
            )[self.response_column]
            .mean()
        )

        drug_means = (
            df.groupby(
                self.drug_column
            )[self.response_column]
            .mean()
        )

        self.cell_effects = (
            cell_means - self.global_mean
        )

        self.drug_effects = (
            drug_means - self.global_mean
        )

    def predict(self, df):

        cell_effect = (
            df[self.cell_line_column]
            .map(self.cell_effects)
            .fillna(0)
        )

        drug_effect = (
            df[self.drug_column]
            .map(self.drug_effects)
            .fillna(0)
        )

        predictions = (
            self.global_mean
            + cell_effect
            + drug_effect
        )

        return predictions.to_numpy()


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
):

    predictions = model.predict(X)

    actual = y.to_numpy()

    metrics = calculate_metrics(
        actual,
        predictions,
    )

    return (
        metrics,
        actual,
        predictions,
    )


def evaluate_naive_mean_effects(
    model,
    df,
):

    predictions = model.predict(df)

    actual = pd.to_numeric(
        df[RESPONSE_COLUMN],
        errors="coerce",
    ).to_numpy()

    metrics = calculate_metrics(
        actual,
        predictions,
    )

    return (
        metrics,
        actual,
        predictions,
    )


# ============================================================
# PLOTTING
# ============================================================

def save_prediction_plot(
    y_true,
    predictions,
    title,
    output_path,
):

    plt.figure(figsize=(6, 6))

    plt.scatter(
        y_true,
        predictions,
        alpha=0.5,
    )

    min_value = min(
        np.min(y_true),
        np.min(predictions),
    )

    max_value = max(
        np.max(y_true),
        np.max(predictions),
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
    )

    plt.xlabel(
        "Actual LN IC50"
    )

    plt.ylabel(
        "Predicted LN IC50"
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    project_root = Path(
        args.project_root
    )

    output_dir = Path(
        args.output_dir
    )

    within_study_dataset = (
        args.within_study_dataset
    )

    cross_study_datasets = (
        args.cross_study_datasets
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # DATA DIRECTORIES
    # --------------------------------------------------------

    lco_root = (
        project_root
        / "local_scripts"
        / "Raw_datasets"
        / "results"
        / "LCO_splits"
    )

    within_study_gene_expression_root = (
        project_root
        / "local_scripts"
        / "Raw_datasets"
        / "results"
        / "gene_expression_processed"
    )

    cross_study_gene_expression_root = (
        project_root
        / "local_scripts"
        / "Raw_datasets"
        / "results"
        / "gene_expr_transformed"
    )

    within_study_fingerprint_root = (
        project_root
        / "data"
        / "cross_study_fingerprints"
    )

    cross_study_fingerprint_root = (
        project_root
        / "local_scripts"
        / "Raw_datasets"
        / "results"
        / "cross_study_fingerprints"
    )

    # ========================================================
    # LOAD WITHIN-STUDY FEATURES
    # ========================================================

    within_study_fingerprints = (
        load_fingerprints(
            within_study_fingerprint_root
            / f"{within_study_dataset}_fingerprints_128_transposed.csv"
        )
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    all_results = []
    rf_prediction_rows = []

    # ========================================================
    # LCO SPLITS
    # ========================================================

    for split_number in range(
        1,
        args.n_splits + 1,
    ):

        print(
            f"\n================ SPLIT "
            f"{split_number} ================\n"
        )

        split_dir = (
            lco_root
            / within_study_dataset
            / f"split_{split_number}"
        )

        if not split_dir.exists():
            raise FileNotFoundError(
                f"LCO split directory not found: {split_dir}"
            )

        # ----------------------------------------------------
        # LOAD WITHIN-STUDY DATA
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # LOAD SPLIT-SPECIFIC WITHIN-STUDY
        # GENE-EXPRESSION DATA
        # ----------------------------------------------------

        gene_expression_split_dir = (
            within_study_gene_expression_root
            / within_study_dataset
            / f"split_{split_number}"
        )

        train_gene_expression = (
            load_gene_expression(
                gene_expression_split_dir
                / "train_gene_expression.csv"
            )
        )

        validation_gene_expression = (
            load_gene_expression(
                gene_expression_split_dir
                / "validation_gene_expression.csv"
            )
        )

        test_gene_expression = (
            load_gene_expression(
                gene_expression_split_dir
                / "test_gene_expression.csv"
            )
        )

        # ----------------------------------------------------
        # PREPARE MODEL DATA
        # ----------------------------------------------------

        train_model_df = prepare_model_data(
            train_df,
            train_gene_expression,
            within_study_fingerprints,
        )

        validation_model_df = prepare_model_data(
            validation_df,
            validation_gene_expression,
            within_study_fingerprints,
        )

        test_model_df = prepare_model_data(
            test_df,
            test_gene_expression,
            within_study_fingerprints,
        )

        feature_columns = get_feature_columns(
            train_model_df
        )

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

        X_test, y_test = build_feature_matrix(
            test_model_df,
            feature_columns,
        )

        # ----------------------------------------------------
        # RANDOM FOREST HYPERPARAMETER TUNING
        # ----------------------------------------------------

        best_depth, tuning_results = (
            tune_random_forest(
                X_train,
                y_train,
                X_validation,
                y_validation,
            )
        )

        print(
            f"Best max_depth for split "
            f"{split_number}: {best_depth}"
        )

        tuning_results.to_csv(
            output_dir
            / f"split_{split_number}_rf_tuning.csv",
            index=False,
        )

        # ----------------------------------------------------
        # WITHIN-STUDY FINAL RANDOM FOREST
        # ----------------------------------------------------

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

        rf_model = train_final_random_forest(
            X_train_validation,
            y_train_validation,
            best_depth,
        )

        rf_model_path = (
            output_dir
            / f"split_{split_number}_"
            f"random_forest.pkl"
        )

        with open(
            rf_model_path,
            "wb",
        ) as f:

            pickle.dump(
                rf_model,
                f,
            )

        # ----------------------------------------------------
        # WITHIN-STUDY RANDOM FOREST TEST
        # ----------------------------------------------------

        rf_metrics, actual, predictions = (
            evaluate_model(
                rf_model,
                X_test,
                y_test,
            )
        )

        rf_prediction_data = test_model_df.loc[
            y_test.index
        ]

        rf_prediction_rows.append(
            pd.DataFrame({
                "split": split_number,
                "model": "Random Forest",
                "dataset": f"{within_study_dataset}_test",
                "cell_line_name": rf_prediction_data[
                    CELL_LINE_COLUMN
                ].to_numpy(),
                "cellosaurus_id": rf_prediction_data[
                    CELL_ID_COLUMN
                ].to_numpy(),
                "pubchem_id": rf_prediction_data[
                    DRUG_ID_COLUMN
                ].to_numpy(),
                "actual_response": actual,
                "predicted_response": predictions,
            })
        )

        print(
            f"Random Forest - "
            f"{within_study_dataset} test:",
            rf_metrics,
        )

        all_results.append(
            {
                "split": split_number,
                "model": "Random Forest",
                "dataset": (
                    f"{within_study_dataset}_test"
                ),
                "max_depth": best_depth,
                **rf_metrics,
            }
        )

        save_prediction_plot(
            actual,
            predictions,
            (
                f"Random Forest - "
                f"{within_study_dataset} Test - "
                f"Split {split_number}"
            ),
            output_dir
            / (
                f"split_{split_number}_"
                f"rf_{within_study_dataset}_test.png"
            ),
        )

        # ----------------------------------------------------
        # WITHIN-STUDY NAIVE MEAN EFFECTS
        # ----------------------------------------------------

        naive_model = NaiveMeanEffects(
            cell_line_column=CELL_LINE_COLUMN,
            drug_column=DRUG_ID_COLUMN,
            response_column=RESPONSE_COLUMN,
        )

        naive_model.fit(
            train_validation_df
        )

        naive_model_path = (
            output_dir
            / f"split_{split_number}_"
            f"naive_mean_effects.pkl"
        )

        with open(
            naive_model_path,
            "wb",
        ) as f:

            pickle.dump(
                naive_model,
                f,
            )

        # ----------------------------------------------------
        # CROSS-STUDY NAIVE MEAN EFFECTS
        # ----------------------------------------------------

        cross_study_naive_model = NaiveMeanEffects(
            cell_line_column=CELL_LINE_COLUMN,
            drug_column=DRUG_ID_COLUMN,
            response_column=RESPONSE_COLUMN,
        )

        cross_study_naive_model.fit(
            train_model_df
        )

        cross_study_naive_model_path = (
            output_dir
            / f"split_{split_number}_"
            f"cross_study_naive_mean_effects.pkl"
        )

        with open(
            cross_study_naive_model_path,
            "wb",
        ) as f:

            pickle.dump(
                cross_study_naive_model,
                f,
            )

        # ----------------------------------------------------
        # WITHIN-STUDY NAIVE MEAN EFFECTS TEST
        # ----------------------------------------------------

        naive_prediction_data = test_model_df.loc[
            y_test.index
        ]

        naive_metrics, actual, predictions = (
            evaluate_naive_mean_effects(
                naive_model,
                naive_prediction_data,
            )
        )

        rf_prediction_rows.append(
            pd.DataFrame({
                "split": split_number,
                "model": "Naive Mean Effects",
                "dataset": f"{within_study_dataset}_test",
                "cell_line_name": naive_prediction_data[
                    CELL_LINE_COLUMN
                ].to_numpy(),
                "cellosaurus_id": naive_prediction_data[
                    CELL_ID_COLUMN
                ].to_numpy(),
                "pubchem_id": naive_prediction_data[
                    DRUG_ID_COLUMN
                ].to_numpy(),
                "actual_response": actual,
                "predicted_response": predictions,
            })
        )

        print(
            f"Naive Mean Effects - "
            f"{within_study_dataset} test:",
            naive_metrics,
        )

        all_results.append(
            {
                "split": split_number,
                "model": "Naive Mean Effects",
                "dataset": (
                    f"{within_study_dataset}_test"
                ),
                "max_depth": np.nan,
                **naive_metrics,
            }
        )

        save_prediction_plot(
            actual,
            predictions,
            (
                f"Naive Mean Effects - "
                f"{within_study_dataset} Test - "
                f"Split {split_number}"
            ),
            output_dir
            / (
                f"split_{split_number}_"
                f"naive_{within_study_dataset}_test.png"
            ),
        )

        # ====================================================
        # CROSS-STUDY EVALUATION
        # ====================================================

        for dataset in cross_study_datasets:

            print(
                f"\nEvaluating cross-study dataset: "
                f"{dataset}"
            )

            # ------------------------------------------------
            # LOAD UNSEEN CROSS-STUDY RESPONSE DATA
            # ------------------------------------------------

            cross_study_path = (
                split_dir
                / f"{dataset}_unseen.csv"
            )

            cross_study_df = load_response_data(
                cross_study_path
            )

            # ------------------------------------------------
            # LOAD CROSS-STUDY FEATURES
            # ------------------------------------------------

            cross_study_gene_expression = (
                load_gene_expression(
                    cross_study_gene_expression_root
                    / f"gene_expression_{dataset}_processed.csv"
                )
            )

            cross_study_fingerprints = (
                load_fingerprints(
                    cross_study_fingerprint_root
                    / f"{dataset}_fingerprints_128_transposed.csv"
                )
            )

            # ------------------------------------------------
            # PREPARE CROSS-STUDY DATA
            # ------------------------------------------------

            cross_study_model_df = (
                prepare_model_data(
                    cross_study_df,
                    cross_study_gene_expression,
                    cross_study_fingerprints,
                )
            )

            X_cross_study, y_cross_study = (
                build_feature_matrix(
                    cross_study_model_df,
                    feature_columns,
                )
            )

            # ------------------------------------------------
            # CROSS-STUDY RANDOM FOREST
            # ------------------------------------------------

            cross_study_rf_metrics, actual, predictions = (
                evaluate_model(
                    rf_model,
                    X_cross_study,
                    y_cross_study,
                )
            )

            rf_prediction_data = cross_study_model_df.loc[
                y_cross_study.index
            ]

            rf_prediction_rows.append(
                pd.DataFrame({
                    "split": split_number,
                    "model": "Random Forest",
                    "dataset": dataset,
                    "cell_line_name": rf_prediction_data[
                        CELL_LINE_COLUMN
                    ].to_numpy(),
                    "cellosaurus_id": rf_prediction_data[
                        CELL_ID_COLUMN
                    ].to_numpy(),
                    "pubchem_id": rf_prediction_data[
                        DRUG_ID_COLUMN
                    ].to_numpy(),
                    "actual_response": actual,
                    "predicted_response": predictions,
                })
            )

            print(
                f"Random Forest - {dataset}:",
                cross_study_rf_metrics,
            )

            all_results.append(
                {
                    "split": split_number,
                    "model": "Random Forest",
                    "dataset": dataset,
                    "max_depth": best_depth,
                    **cross_study_rf_metrics,
                }
            )

            save_prediction_plot(
                actual,
                predictions,
                (
                    f"Random Forest - {dataset} - "
                    f"Split {split_number}"
                ),
                output_dir
                / (
                    f"split_{split_number}_"
                    f"rf_{dataset}.png"
                ),
            )

            # ------------------------------------------------
            # CROSS-STUDY NAIVE MEAN EFFECTS
            # ------------------------------------------------

            naive_prediction_data = cross_study_model_df.loc[
                y_cross_study.index
            ]

            naive_metrics, actual, predictions = (
                evaluate_naive_mean_effects(
                    cross_study_naive_model,
                    naive_prediction_data,
                )
            )

            rf_prediction_rows.append(
                pd.DataFrame({
                    "split": split_number,
                    "model": "Naive Mean Effects",
                    "dataset": dataset,
                    "cell_line_name": naive_prediction_data[
                        CELL_LINE_COLUMN
                    ].to_numpy(),
                    "cellosaurus_id": naive_prediction_data[
                        CELL_ID_COLUMN
                    ].to_numpy(),
                    "pubchem_id": naive_prediction_data[
                        DRUG_ID_COLUMN
                    ].to_numpy(),
                    "actual_response": actual,
                    "predicted_response": predictions,
                })
            )

            print(
                f"Naive Mean Effects - {dataset}:",
                naive_metrics,
            )

            all_results.append(
                {
                    "split": split_number,
                    "model": "Naive Mean Effects",
                    "dataset": dataset,
                    "max_depth": np.nan,
                    **naive_metrics,
                }
            )

            save_prediction_plot(
                actual,
                predictions,
                (
                    f"Naive Mean Effects - {dataset} - "
                    f"Split {split_number}"
                ),
                output_dir
                / (
                    f"split_{split_number}_"
                    f"naive_{dataset}.png"
                ),
            )

    # ========================================================
    # SAVE ALL RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        output_dir
        / "model_evaluation_results.csv",
        index=False,
    )

    rf_predictions_df = pd.concat(
        rf_prediction_rows,
        ignore_index=True,
    )

    rf_predictions_df.to_csv(
        output_dir / "model.predictions.csv",
        index=False
    )

    # ========================================================
    # SAVE MEAN ± SD PERFORMANCE TABLE ACROSS LCO SPLITS
    # ========================================================

    metric_columns = [
        "R2",
        "Pearson",
        "Spearman",
        "RMSE",
    ]

    summary_rows = []

    for (model_name, dataset_name), group in results_df.groupby(
        ["model", "dataset"],
        sort=False,
    ):
        row = {
            "Model": model_name,
            "Dataset": (
                f"{within_study_dataset} (within-study)"
                if dataset_name == f"{within_study_dataset}_test"
                else dataset_name
            ),
        }

        for metric in metric_columns:
            mean_value = group[metric].mean()
            sd_value = group[metric].std(ddof=1)

            row[f"Mean {metric} ± SD"] = (
                f"{mean_value:.4f} ± {sd_value:.4f}"
            )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_excel(
        output_dir / "model_performance_mean_sd.xlsx",
        index=False,
    )

    print("\nModel evaluation completed successfully.")


if __name__ == "__main__":
    main()
