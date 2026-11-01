"""
Train and evaluate a Simple Neural Network and Naive Mean Effects
models for within-study and cross-study cancer drug-response prediction.

This version matches the Random Forest pipeline:

- No AUC filtering is performed.
- No response standardisation is performed.
- Original LN_IC50_curvecurator response values are used directly.
- Within-study evaluation uses LCO split sets.
- Neural-network architecture is selected using the validation set.
- The final neural network is trained using training + validation data.
- The same final fitted neural network is used for within-study and
  cross-study evaluation.
- Cross-study datasets are used only for evaluation.
- Naive Mean Effects follows the same training/evaluation logic.

For each LCO split:

Within-study:
    Hyperparameter tuning:
        train -> validation

    Final model:
        train + validation -> test

Cross-study:
    Final model:
        same fitted train + validation model

    Evaluation:
        unseen cross-study cell lines only

All final metrics are calculated on the original
LN_IC50_curvecurator response scale.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

BATCH_SIZE = 16
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
DROPOUT_PROB = 0.3
LEARNING_RATE = 0.001

HIDDEN_LAYER_CONFIGURATIONS = [
    [32, 16, 8, 4],
    [128, 64, 32],
    [64, 64, 32],
    [1024, 128, 64, 16],
]

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
# REPRODUCIBILITY
# ============================================================

def set_seed(
    seed: int = RANDOM_STATE,
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a Simple Neural Network "
            "and Naive Mean Effects models for within-study "
            "and cross-study drug-response prediction "
            "without response standardisation."
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
# NEURAL NETWORK
# ============================================================

class FeedForwardNetwork(nn.Module):
    """
    Simple feed-forward neural network.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        dropout_prob: float = DROPOUT_PROB,
    ) -> None:

        super().__init__()

        layers = []

        previous_dim = input_dim

        for hidden_dim in hidden_layers:

            layers.append(
                nn.Linear(
                    previous_dim,
                    hidden_dim,
                )
            )

            layers.append(
                nn.BatchNorm1d(
                    hidden_dim,
                )
            )

            layers.append(
                nn.Dropout(
                    p=dropout_prob,
                )
            )

            layers.append(
                nn.ReLU()
            )

            previous_dim = hidden_dim

        layers.append(
            nn.Linear(
                previous_dim,
                1,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x).squeeze(-1)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return {
        "R2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "Pearson": float(
            pearsonr(
                y_true,
                y_pred,
            ).statistic
        ),
        "Spearman": float(
            spearmanr(
                y_true,
                y_pred,
            ).statistic
        ),
        "MSE": float(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_response_data(
    path: Path,
):

    if not path.exists():
        raise FileNotFoundError(
            f"Response file not found: {path}"
        )

    data = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = {
        CELL_LINE_COLUMN,
        CELL_ID_COLUMN,
        DRUG_ID_COLUMN,
        RESPONSE_COLUMN,
    }

    missing = (
        required_columns
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns in {path}: "
            f"{sorted(missing)}"
        )

    data[CELL_ID_COLUMN] = (
        data[CELL_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    data[DRUG_ID_COLUMN] = (
        data[DRUG_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    return data


def load_gene_expression(
    path: Path,
):

    if not path.exists():
        raise FileNotFoundError(
            f"Gene-expression file not found: {path}"
        )

    data = pd.read_csv(
        path,
        low_memory=False,
    )

    if CELL_ID_COLUMN not in data.columns:
        raise ValueError(
            f"{CELL_ID_COLUMN} not found in "
            f"gene-expression file: {path}"
        )

    data[CELL_ID_COLUMN] = (
        data[CELL_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    data = data.drop_duplicates(
        subset=[CELL_ID_COLUMN],
        keep="first",
    )

    return data


def load_fingerprints(
    path: Path,
):

    if not path.exists():
        raise FileNotFoundError(
            f"Fingerprint file not found: {path}"
        )

    data = pd.read_csv(
        path,
        low_memory=False,
    )

    if DRUG_ID_COLUMN not in data.columns:
        raise ValueError(
            f"{DRUG_ID_COLUMN} not found in "
            f"fingerprint file: {path}"
        )

    data[DRUG_ID_COLUMN] = (
        data[DRUG_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    data = data.drop_duplicates(
        subset=[DRUG_ID_COLUMN],
        keep="first",
    )

    return data


# ============================================================
# PREPARE MODEL DATA
# ============================================================

def get_feature_columns(
    df,
):

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

    gene_expression_df = (
        gene_expression_df
        .drop_duplicates(
            subset=[CELL_ID_COLUMN],
            keep="first",
        )
    )

    fingerprint_df = (
        fingerprint_df
        .drop_duplicates(
            subset=[DRUG_ID_COLUMN],
            keep="first",
        )
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

    if merged.empty:
        raise ValueError(
            "No rows remained after merging "
            "response, gene-expression, and "
            "fingerprint data."
        )

    feature_columns = get_feature_columns(
        merged
    )

    X = merged[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    y = pd.to_numeric(
        merged[RESPONSE_COLUMN],
        errors="coerce",
    )

    valid_rows = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if X.empty:
        raise ValueError(
            "No valid rows remain after feature "
            "and response conversion."
        )

    return (
        X.to_numpy(dtype=np.float32),
        y.to_numpy(dtype=np.float32),
        merged.loc[valid_rows].reset_index(drop=True),
        feature_columns,
    )


# ============================================================
# TRAIN NEURAL NETWORK
# ============================================================

def train_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
    hidden_layers,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = FeedForwardNetwork(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        dropout_prob=DROPOUT_PROB,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_function = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train,
            dtype=torch.float32,
        ),
    )

    train_batch_size = min(
        BATCH_SIZE,
        len(train_dataset),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
    )

    if X_validation is not None:

        validation_dataset = TensorDataset(
            torch.tensor(
                X_validation,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_validation,
                dtype=torch.float32,
            ),
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

    else:

        validation_loader = None

    best_state = None
    best_validation_loss = float("inf")

    train_losses = []
    validation_losses = []

    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):

        model.train()

        train_loss_total = 0.0
        train_count = 0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(
                batch_x
            )

            loss = loss_function(
                predictions,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            train_loss_total += (
                loss.item()
                * len(batch_x)
            )

            train_count += len(batch_x)

        train_loss = (
            train_loss_total
            / train_count
        )

        train_losses.append(
            train_loss
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if validation_loader is not None:

            model.eval()

            validation_loss_total = 0.0
            validation_count = 0

            with torch.no_grad():

                for batch_x, batch_y in validation_loader:

                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    predictions = model(
                        batch_x
                    )

                    loss = loss_function(
                        predictions,
                        batch_y,
                    )

                    validation_loss_total += (
                        loss.item()
                        * len(batch_x)
                    )

                    validation_count += len(batch_x)

            validation_loss = (
                validation_loss_total
                / validation_count
            )

            validation_losses.append(
                validation_loss
            )

            if validation_loss < best_validation_loss:

                best_validation_loss = (
                    validation_loss
                )

                best_state = copy.deepcopy(
                    model.state_dict()
                )

                epochs_without_improvement = 0

            else:

                epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return (
        model,
        train_losses,
        validation_losses,
    )


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    X,
):

    device = next(
        model.parameters()
    ).device

    model.eval()

    with torch.no_grad():

        predictions = model(
            torch.tensor(
                X,
                dtype=torch.float32,
            ).to(device)
        )

    return (
        predictions
        .cpu()
        .numpy()
    )


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

        self.cell_line_column = (
            cell_line_column
        )

        self.drug_column = (
            drug_column
        )

        self.response_column = (
            response_column
        )

        self.global_mean = None
        self.cell_effects = None
        self.drug_effects = None

    def fit(
        self,
        df,
    ):

        df = df.copy()

        df[self.drug_column] = (
            df[self.drug_column]
            .astype(str)
            .str.strip()
        )

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
            cell_means
            - self.global_mean
        )

        self.drug_effects = (
            drug_means
            - self.global_mean
        )

    def predict(
        self,
        df,
    ):

        df = df.copy()

        df[self.drug_column] = (
            df[self.drug_column]
            .astype(str)
            .str.strip()
        )

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

    predictions = predict(
        model,
        X,
    )

    metrics = calculate_metrics(
        y,
        predictions,
    )

    return (
        metrics,
        y,
        predictions,
    )


def evaluate_naive_mean_effects(
    model,
    df,
):

    predictions = model.predict(
        df
    )

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


def add_prediction_records(
    records,
    model_name,
    dataset_name,
    split_number,
    prediction_df,
    actual,
    predictions,
):
    """Store predictions alongside identifiers from the same model-data rows."""

    if len(prediction_df) != len(actual) or len(actual) != len(predictions):
        raise ValueError(
            f"Prediction-row alignment error for {model_name}, "
            f"{dataset_name}, split {split_number}: "
            f"rows={len(prediction_df)}, actual={len(actual)}, "
            f"predictions={len(predictions)}."
        )

    for row, actual_value, predicted_value in zip(
        prediction_df.to_dict(orient="records"),
        actual,
        predictions,
    ):
        records.append(
            {
                CELL_LINE_COLUMN: row.get(CELL_LINE_COLUMN),
                CELL_ID_COLUMN: row.get(CELL_ID_COLUMN),
                DRUG_ID_COLUMN: row.get(DRUG_ID_COLUMN),
                "dataset": dataset_name,
                "lco_split": split_number,
                "model": model_name,
                "actual_response": float(actual_value),
                "predicted_response": float(predicted_value),
            }
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

    plt.figure(
        figsize=(6, 6)
    )

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

    plt.title(
        title
    )

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

    set_seed()

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

    # ========================================================
    # DATA DIRECTORIES
    # ========================================================

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
        / "processed"
    )

    cross_study_fingerprint_root = (
        project_root
        / "local_scripts"
        / "Raw_datasets"
        / "results"
        / "cross_study_fingerprints"
    )

    # ========================================================
    # LOAD WITHIN-STUDY FINGERPRINTS
    # ========================================================

    within_study_fingerprints = (
        load_fingerprints(
            within_study_fingerprint_root
            / (
                f"{within_study_dataset}"
                f"_fingerprints_128_transposed.csv"
            )
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    all_results = []
    all_prediction_rows = []

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
                f"LCO split directory not found: "
                f"{split_dir}"
            )

        # ----------------------------------------------------
        # LOAD WITHIN-STUDY RESPONSE DATA
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
        # LOAD SPLIT-SPECIFIC GENE EXPRESSION
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

        # ====================================================
        # PREPARE TRAINING DATA
        # ====================================================

        (
            X_train,
            y_train,
            train_model_df,
            feature_columns,
        ) = prepare_model_data(
            train_df,
            train_gene_expression,
            within_study_fingerprints,
        )

        # ====================================================
        # PREPARE VALIDATION DATA
        # ====================================================

        (
            X_validation,
            y_validation,
            validation_model_df,
            validation_feature_columns,
        ) = prepare_model_data(
            validation_df,
            validation_gene_expression,
            within_study_fingerprints,
        )

        # ----------------------------------------------------
        # Ensure feature columns are consistent
        # ----------------------------------------------------

        if feature_columns != validation_feature_columns:

            raise ValueError(
                "Training and validation feature columns "
                "are not identical."
            )

        # ====================================================
        # HYPERPARAMETER TUNING
        # ====================================================

        tuning_results = []

        best_hidden_layers = None
        best_validation_mse = np.inf

        for hidden_layers in HIDDEN_LAYER_CONFIGURATIONS:

            print(
                "\nTesting architecture:",
                hidden_layers,
            )

            set_seed()

            (
                model,
                train_losses,
                validation_losses,
            ) = train_model(
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                hidden_layers=hidden_layers,
            )

            validation_predictions = predict(
                model,
                X_validation,
            )

            metrics = calculate_metrics(
                y_validation,
                validation_predictions,
            )

            best_epoch = (
                int(np.argmin(validation_losses))
                + 1
                if validation_losses
                else len(train_losses)
            )

            tuning_results.append(
                {
                    "hidden_layers": str(
                        hidden_layers
                    ),
                    "dropout_prob": DROPOUT_PROB,
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "max_epochs": MAX_EPOCHS,
                    "best_epoch": best_epoch,
                    "R2": metrics["R2"],
                    "Pearson": metrics["Pearson"],
                    "Spearman": metrics["Spearman"],
                    "MSE": metrics["MSE"],
                }
            )

            print(
                f"Validation MSE: "
                f"{metrics['MSE']:.6f}"
            )

            if metrics["MSE"] < best_validation_mse:

                best_validation_mse = (
                    metrics["MSE"]
                )

                best_hidden_layers = (
                    hidden_layers.copy()
                )

        tuning_results_df = pd.DataFrame(
            tuning_results
        )

        tuning_results_df.to_csv(
            output_dir
            / (
                f"split_{split_number}_"
                f"nn_tuning.csv"
            ),
            index=False,
        )

        with open(
            output_dir
            / (
                f"split_{split_number}_"
                f"best_hyperparameters.json"
            ),
            "w",
        ) as file:

            json.dump(
                {
                    "hidden_layers":
                        best_hidden_layers,
                    "dropout_prob":
                        DROPOUT_PROB,
                    "learning_rate":
                        LEARNING_RATE,
                    "batch_size":
                        BATCH_SIZE,
                    "max_epochs":
                        MAX_EPOCHS,
                    "best_validation_mse":
                        best_validation_mse,
                },
                file,
                indent=4,
            )

        print(
            "\nBest architecture:",
            best_hidden_layers,
        )

        print(
            "Best validation MSE:",
            best_validation_mse,
        )

        # ====================================================
        # WITHIN-STUDY FINAL TRAINING
        # TRAIN + VALIDATION
        # ====================================================

        train_validation_df = pd.concat(
            [
                train_model_df,
                validation_model_df,
            ],
            ignore_index=True,
        )

        # Use the same feature columns for train + validation.
        X_train_validation = (
            train_validation_df[
                feature_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        y_train_validation = pd.to_numeric(
            train_validation_df[
                RESPONSE_COLUMN
            ],
            errors="coerce",
        )

        valid_rows = (
            X_train_validation
            .notna()
            .all(axis=1)
            & y_train_validation.notna()
        )

        X_train_validation = (
            X_train_validation
            .loc[valid_rows]
            .to_numpy(
                dtype=np.float32
            )
        )

        y_train_validation = (
            y_train_validation
            .loc[valid_rows]
            .to_numpy(
                dtype=np.float32
            )
        )

        set_seed()

        (
            final_model,
            final_train_losses,
            _,
        ) = train_model(
            X_train=X_train_validation,
            y_train=y_train_validation,
            X_validation=None,
            y_validation=None,
            hidden_layers=best_hidden_layers,
        )

        final_model_path = (
            output_dir
            / (
                f"split_{split_number}_"
                f"simple_neural_network.pt"
            )
        )

        torch.save(
            final_model.state_dict(),
            final_model_path,
        )

        # ====================================================
        # WITHIN-STUDY TEST
        # ====================================================

        (
            X_test,
            y_test,
            test_model_df,
            test_feature_columns,
        ) = prepare_model_data(
            test_df,
            test_gene_expression,
            within_study_fingerprints,
        )

        if feature_columns != test_feature_columns:

            raise ValueError(
                "Training and test feature columns "
                "are not identical."
            )

        test_metrics, actual, predictions = (
            evaluate_model(
                final_model,
                X_test,
                y_test,
            )
        )

        print(
            f"\nSimple Neural Network - "
            f"{within_study_dataset} test:",
            test_metrics,
        )

        all_results.append(
            {
                "split": split_number,
                "model": "Simple Neural Network",
                "dataset": (
                    f"{within_study_dataset}_test"
                ),
                "max_depth": np.nan,
                **test_metrics,
            }
        )

        add_prediction_records(
            all_prediction_rows,
            "Simple Neural Network",
            f"{within_study_dataset} (within-study)",
            split_number,
            test_model_df,
            actual,
            predictions,
        )

        save_prediction_plot(
            actual,
            predictions,
            (
                f"Simple Neural Network - "
                f"{within_study_dataset} Test - "
                f"Split {split_number}"
            ),
            output_dir
            / (
                f"split_{split_number}_"
                f"nn_{within_study_dataset}_test.png"
            ),
        )

        # ====================================================
        # WITHIN-STUDY NAIVE MEAN EFFECTS
        # TRAIN + VALIDATION
        # ====================================================

        naive_model = NaiveMeanEffects(
            cell_line_column=CELL_LINE_COLUMN,
            drug_column=DRUG_ID_COLUMN,
            response_column=RESPONSE_COLUMN,
        )

        naive_model.fit(
            train_validation_df
        )

        naive_metrics, actual, predictions = (
            evaluate_naive_mean_effects(
                naive_model,
                test_model_df,
            )
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

        add_prediction_records(
            all_prediction_rows,
            "Naive Mean Effects",
            f"{within_study_dataset} (within-study)",
            split_number,
            test_model_df,
            actual,
            predictions,
        )

        naive_model_path = (
            output_dir
            / (
                f"split_{split_number}_"
                f"naive_mean_effects.pkl"
            )
        )

        with open(
            naive_model_path,
            "wb",
        ) as file:

            pickle.dump(
                naive_model,
                file,
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
        # CROSS-STUDY NAIVE MEAN EFFECTS
        #
        # TRAINING DATA ONLY
        # ====================================================

        cross_study_naive_model = (
            NaiveMeanEffects(
                cell_line_column=CELL_LINE_COLUMN,
                drug_column=DRUG_ID_COLUMN,
                response_column=RESPONSE_COLUMN,
            )
        )

        cross_study_naive_model.fit(
            train_model_df
        )

        cross_study_naive_model_path = (
            output_dir
            / (
                f"split_{split_number}_"
                f"cross_study_naive_mean_effects.pkl"
            )
        )

        with open(
            cross_study_naive_model_path,
            "wb",
        ) as file:

            pickle.dump(
                cross_study_naive_model,
                file,
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
            # LOAD CROSS-STUDY GENE EXPRESSION
            # ------------------------------------------------

            cross_study_gene_expression = (
                load_gene_expression(
                    cross_study_gene_expression_root
                    / (
                        f"gene_expression_"
                        f"{dataset}_processed.csv"
                    )
                )
            )

            # ------------------------------------------------
            # LOAD CROSS-STUDY FINGERPRINTS
            # ------------------------------------------------

            cross_study_fingerprints = (
                load_fingerprints(
                    cross_study_fingerprint_root
                    / (
                        f"{dataset}_"
                        f"fingerprints_128_transposed.csv"
                    )
                )
            )

            # ------------------------------------------------
            # PREPARE CROSS-STUDY DATA
            # ------------------------------------------------

            (
                X_cross_study,
                y_cross_study,
                cross_study_model_df,
                cross_study_feature_columns,
            ) = prepare_model_data(
                cross_study_df,
                cross_study_gene_expression,
                cross_study_fingerprints,
            )

            # ------------------------------------------------
            # Ensure cross-study feature order matches
            # training features
            # ------------------------------------------------

            if feature_columns != cross_study_feature_columns:

                raise ValueError(
                    f"Feature columns for {dataset} do not "
                    "match the within-study training features."
                )

            # ------------------------------------------------
            # Cross-study neural network
            # Use the same final fitted model as within-study
            # ------------------------------------------------

            cross_study_nn_metrics, actual, predictions = (
                evaluate_model(
                    final_model,
                    X_cross_study,
                    y_cross_study,
                )
            )

            print(
                f"Simple Neural Network - "
                f"{dataset}:",
                cross_study_nn_metrics,
            )

            all_results.append(
                {
                    "split": split_number,
                    "model": "Simple Neural Network",
                    "dataset": dataset,
                    "max_depth": np.nan,
                    **cross_study_nn_metrics,
                }
            )

            add_prediction_records(
                all_prediction_rows,
                "Simple Neural Network",
                dataset,
                split_number,
                cross_study_model_df,
                actual,
                predictions,
            )

            save_prediction_plot(
                actual,
                predictions,
                (
                    f"Simple Neural Network - "
                    f"{dataset} - "
                    f"Split {split_number}"
                ),
                output_dir
                / (
                    f"split_{split_number}_"
                    f"nn_{dataset}.png"
                ),
            )

            # ------------------------------------------------
            # Cross-study Naive Mean Effects
            # ------------------------------------------------

            naive_metrics, actual, predictions = (
                evaluate_naive_mean_effects(
                    cross_study_naive_model,
                    cross_study_model_df,
                )
            )

            print(
                f"Naive Mean Effects - "
                f"{dataset}:",
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

            add_prediction_records(
                all_prediction_rows,
                "Naive Mean Effects",
                dataset,
                split_number,
                cross_study_model_df,
                actual,
                predictions,
            )

            save_prediction_plot(
                actual,
                predictions,
                (
                    f"Naive Mean Effects - "
                    f"{dataset} - "
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

    # Save individual actual and predicted responses with identifiers.
    predictions_df = pd.DataFrame(all_prediction_rows)
    predictions_df.to_csv(
        output_dir / "individual_predictions.csv",
        index=False,
    )

    # Create one combined plot per model/dataset, pooling the LCO splits.
    for (model_name, dataset_name), group in predictions_df.groupby(
        ["model", "dataset"],
        sort=False,
    ):
        model_prefix = (
            "nn"
            if model_name == "Simple Neural Network"
            else "naive"
        )
        safe_dataset_name = "".join(
            character
            if character.isalnum() or character in ("-", "_")
            else "_"
            for character in dataset_name
        )
        save_prediction_plot(
            group["actual_response"].to_numpy(),
            group["predicted_response"].to_numpy(),
            (
                f"{model_name} - {dataset_name} - "
                f"Combined LCO Splits 1-{args.n_splits}"
            ),
            output_dir
            / (
                f"combined_{model_prefix}_{safe_dataset_name}_"
                f"splits_1-{args.n_splits}.png"
            ),
        )

    # Summarise each performance metric across LCO splits.
    summary_source = results_df.copy()
    summary_source["dataset"] = summary_source["dataset"].replace(
        {
            f"{within_study_dataset}_test":
                f"{within_study_dataset} (within-study)"
        }
    )

    metric_columns = ["R2", "Pearson", "Spearman", "RMSE"]
    summary_rows = []

    for (model_name, dataset_name), group in summary_source.groupby(
        ["model", "dataset"],
        sort=False,
    ):
        summary_row = {
            "Model": model_name,
            "Dataset": dataset_name,
        }

        for metric in metric_columns:
            metric_mean = group[metric].mean()
            metric_sd = group[metric].std(ddof=1)
            summary_row[f"Mean {metric} ± SD"] = (
                f"{metric_mean:.4f} ± {metric_sd:.4f}"
            )

        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "Model",
            "Dataset",
            "Mean R2 ± SD",
            "Mean Pearson ± SD",
            "Mean Spearman ± SD",
            "Mean RMSE ± SD",
        ],
    )

    summary_df.to_excel(
        output_dir / "mean_sd_model_performance.xlsx",
        index=False,
    )

    print(
        "\nModel evaluation completed successfully."
    )
    print(f"Individual predictions saved to: {output_dir / 'individual_predictions.csv'}")
    print(f"Mean ± SD summary saved to: {output_dir / 'mean_sd_model_performance.xlsx'}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()