
"""
Script: Simple Neural Network Drug-Response Prediction

Purpose
-------
Train and evaluate a DRevalPy-style Simple Neural Network using
LCO split sets.

For each LCO split:

Within-study evaluation:
    - Tune the neural network using training and validation data.
    - Select the best architecture using validation MSE.
    - Train the final within-study model using training + validation.
    - Evaluate on the within-study test set.
    - Train and evaluate Naive Mean Effects using training + validation.

Cross-study evaluation:
    - Train a separate neural network using training data only.
    - Train a separate Naive Mean Effects model using training data only.
    - Evaluate these models on unseen cross-study datasets.

Cross-study datasets are used only for evaluation and are never
used for model training or hyperparameter selection.

Response standardisation:
    Response standardisation is performed before this script.
    The split-specific scaler fitted on the training response is
    loaded from scaler.pkl.

    Training and prediction use:
        LN_IC50_standardised

    Final reported metrics use:
        LN_IC50_curvecurator

    Actual and predicted responses are inverse-transformed using
    the split-specific response scaler before calculating final
    R2, Pearson, Spearman, and MSE.

Gene-expression preprocessing:
    Gene-expression files are assumed to already have the arcsinh
    transformation applied.

    This script does NOT apply arcsinh again.

Gene-expression scaling:
    During hyperparameter tuning:
        scaler -> training data only

    During final within-study training:
        scaler -> training + validation data

    During cross-study training:
        scaler -> training data only

    Validation, test, and cross-study data are transformed using
    the appropriate training-fitted scaler and are never used to
    fit that scaler.

Fingerprint handling:
    Molecular fingerprints are generated using the established
    128-bit Morgan fingerprint procedure.

    GDSC1 and GDSC2 use the recovered fingerprint files.
    Response observations for drugs without a valid complete
    128-bit fingerprint are excluded from modelling.

    Original response files are not modified.
"""


# ============================================================
# Imports
# ============================================================

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

CELL_ID = "cellosaurus_id"
CELL_NAME = "cell_line_name"
DRUG_ID = "pubchem_id"

STANDARDISED_RESPONSE = "LN_IC50_standardised"
ORIGINAL_RESPONSE = "LN_IC50_curvecurator"

RANDOM_STATE = 42

BATCH_SIZE = 16
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 5
DROPOUT_PROB = 0.3

LEARNING_RATE = 0.001

HIDDEN_LAYER_CONFIGURATIONS = [
    [32, 16, 8, 4],
    [128, 64, 32],
    [64, 64, 32],
    [1024, 128, 64, 16],
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(
    seed: int = RANDOM_STATE,
) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Neural Network
# ============================================================

class FeedForwardNetwork(nn.Module):
    """
    DRevalPy-style feed-forward neural network.
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
                    hidden_dim
                )
            )

            layers.append(
                nn.Dropout(
                    p=dropout_prob
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
# CSV loading
# ============================================================

def load_csv(
    path: Path,
) -> pd.DataFrame:
    """Load a CSV file."""

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


# ============================================================
# Load gene expression
# ============================================================

def load_gene_expression(
    path: Path,
) -> pd.DataFrame:
    """Load processed gene-expression data."""

    data = load_csv(path)

    if CELL_ID not in data.columns:
        raise ValueError(
            f"Gene-expression file is missing "
            f"{CELL_ID!r}."
        )

    data[CELL_ID] = (
        data[CELL_ID]
        .astype(str)
        .str.strip()
    )

    data = data.drop_duplicates(
        subset=[CELL_ID],
        keep="first",
    )

    return data


# ============================================================
# Load fingerprints
# ============================================================

def load_fingerprints(
    path: Path,
) -> pd.DataFrame:
    """
    Load processed drug fingerprints.

    Only drugs with complete 128-bit fingerprints are retained.
    """

    data = load_csv(path)

    if DRUG_ID not in data.columns:
        raise ValueError(
            f"Fingerprint file is missing "
            f"{DRUG_ID!r}."
        )

    data[DRUG_ID] = (
        data[DRUG_ID]
        .astype(str)
        .str.strip()
    )

    data = data.drop_duplicates(
        subset=[DRUG_ID],
        keep="first",
    )

    fingerprint_columns = [
        column
        for column in data.columns
        if column != DRUG_ID
    ]

    if len(fingerprint_columns) != 128:
        raise ValueError(
            f"Expected 128 fingerprint columns, "
            f"but found {len(fingerprint_columns)} "
            f"in:\n{path}"
        )

    valid_mask = (
        data[fingerprint_columns]
        .notna()
        .all(axis=1)
    )

    invalid_count = int(
        (~valid_mask).sum()
    )

    print(
        f"\nFingerprint file: {path.name}"
    )

    print(
        f"Total drugs: {len(data):,}"
    )

    print(
        f"Valid 128-bit fingerprints: "
        f"{valid_mask.sum():,}"
    )

    print(
        f"Drugs without valid fingerprints: "
        f"{invalid_count:,}"
    )

    data = data.loc[
        valid_mask
    ].copy()

    return data


# ============================================================
# Filter response to valid fingerprints
# ============================================================

def filter_response_to_valid_fingerprints(
    response: pd.DataFrame,
    fingerprints: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only response observations whose drug has
    a valid molecular fingerprint.

    The original response dataframe is not modified.
    """

    response = response.copy()

    response[DRUG_ID] = (
        response[DRUG_ID]
        .astype(str)
        .str.strip()
    )

    valid_drugs = set(
        fingerprints[DRUG_ID]
        .astype(str)
        .str.strip()
    )

    filtered = response[
        response[DRUG_ID].isin(valid_drugs)
    ].copy()

    excluded = len(response) - len(filtered)

    if excluded > 0:
        print(
            f"Fingerprint filtering: "
            f"{excluded:,} response observations excluded."
        )

    if filtered.empty:
        raise ValueError(
            "No response observations remain after "
            "fingerprint filtering."
        )

    return filtered


# ============================================================
# Identify feature columns
# ============================================================

def get_gene_expression_columns(
    gene_expression: pd.DataFrame,
) -> list[str]:
    """Return gene-expression feature columns."""

    return [
        column
        for column in gene_expression.columns
        if column != CELL_ID
    ]


def get_fingerprint_columns(
    fingerprints: pd.DataFrame,
) -> list[str]:
    """Return fingerprint feature columns."""

    return [
        column
        for column in fingerprints.columns
        if column != DRUG_ID
    ]


# ============================================================
# Prepare model data
# ============================================================

def prepare_model_data(
    response: pd.DataFrame,
    gene_expression: pd.DataFrame,
    fingerprints: pd.DataFrame,
    gene_columns: list[str],
    fingerprint_columns: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    response = response.copy()

    response[CELL_ID] = (
        response[CELL_ID]
        .astype(str)
        .str.strip()
    )

    response[DRUG_ID] = (
        response[DRUG_ID]
        .astype(str)
        .str.strip()
    )

    gene_expression = gene_expression.copy()
    fingerprints = fingerprints.copy()

    gene_expression[CELL_ID] = (
        gene_expression[CELL_ID]
        .astype(str)
        .str.strip()
    )

    fingerprints[DRUG_ID] = (
        fingerprints[DRUG_ID]
        .astype(str)
        .str.strip()
    )

    missing_gene_columns = [
        column
        for column in gene_columns
        if column not in gene_expression.columns
    ]

    if missing_gene_columns:
        raise ValueError(
            "Gene-expression data are missing "
            f"{len(missing_gene_columns)} required features."
        )

    missing_fingerprint_columns = [
        column
        for column in fingerprint_columns
        if column not in fingerprints.columns
    ]

    if missing_fingerprint_columns:
        raise ValueError(
            "Fingerprint data are missing "
            f"{len(missing_fingerprint_columns)} required features."
        )

    merged = response.merge(
        gene_expression[
            [CELL_ID] + gene_columns
        ],
        on=CELL_ID,
        how="inner",
    )

    merged = merged.merge(
        fingerprints[
            [DRUG_ID] + fingerprint_columns
        ],
        on=DRUG_ID,
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No rows remained after merging "
            "response, gene-expression, and "
            "fingerprint data."
        )

    missing_target = merged[
        STANDARDISED_RESPONSE
    ].isna().sum()

    if missing_target > 0:
        raise ValueError(
            f"{missing_target:,} rows have missing "
            f"{STANDARDISED_RESPONSE} values."
        )

    X_gene = (
        merged[gene_columns]
        .astype(float)
        .to_numpy()
    )

    X_fingerprint = (
        merged[fingerprint_columns]
        .astype(float)
        .to_numpy()
    )

    y = (
        merged[STANDARDISED_RESPONSE]
        .astype(float)
        .to_numpy()
    )

    cell_names = (
        merged[CELL_NAME]
        .astype(str)
        .to_numpy()
    )

    return (
        X_gene,
        X_fingerprint,
        y,
        cell_names,
    )


# ============================================================
# Fit gene-expression scaler
# ============================================================

def fit_gene_expression_scaler(
    gene_expression: pd.DataFrame,
    training_cell_ids: set[str],
    gene_columns: list[str],
) -> StandardScaler:
    """
    Fit StandardScaler using training cell lines only.
    """

    training_expression = gene_expression[
        gene_expression[CELL_ID].isin(
            training_cell_ids
        )
    ]

    if training_expression.empty:
        raise ValueError(
            "No gene-expression rows were found "
            "for the training cell lines."
        )

    scaler = StandardScaler()

    scaler.fit(
        training_expression[
            gene_columns
        ].astype(float)
    )

    return scaler


# ============================================================
# Transform model input
# ============================================================

def transform_inputs(
    X_gene: np.ndarray,
    X_fingerprint: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Scale gene expression and concatenate fingerprints."""

    if not np.isfinite(X_gene).all():
        nan_count = np.isnan(X_gene).sum()
        inf_count = np.isinf(X_gene).sum()

        raise ValueError(
            "Gene-expression input contains invalid values: "
            f"{nan_count:,} NaN values and "
            f"{inf_count:,} infinite values."
        )

    if not np.isfinite(X_fingerprint).all():
        nan_count = np.isnan(X_fingerprint).sum()
        inf_count = np.isinf(X_fingerprint).sum()

        raise ValueError(
            "Fingerprint input contains invalid values: "
            f"{nan_count:,} NaN values and "
            f"{inf_count:,} infinite values."
        )

    X_gene_scaled = scaler.transform(
        X_gene
    )

    if not np.isfinite(X_gene_scaled).all():
        nan_count = np.isnan(X_gene_scaled).sum()
        inf_count = np.isinf(X_gene_scaled).sum()

        raise ValueError(
            "Scaled gene-expression input contains invalid "
            f"values: {nan_count:,} NaN values and "
            f"{inf_count:,} infinite values."
        )

    X = np.concatenate(
        [
            X_gene_scaled,
            X_fingerprint,
        ],
        axis=1,
    ).astype(np.float32)

    if not np.isfinite(X).all():
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()

        raise ValueError(
            "Final neural-network input contains invalid "
            f"values: {nan_count:,} NaN values and "
            f"{inf_count:,} infinite values."
        )

    return X


# ============================================================
# Inverse-transform response
# ============================================================

def inverse_transform_response(
    values: np.ndarray,
    scaler,
) -> np.ndarray:
    """
    Convert standardised response values back to the original
    LN_IC50_curvecurator scale.
    """

    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1, 1)

    return scaler.inverse_transform(
        values
    ).ravel()


# ============================================================
# Train neural network
# ============================================================

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray | None,
    y_validation: np.ndarray | None,
    hidden_layers: list[int],
) -> tuple[
    FeedForwardNetwork,
    list[float],
    list[float],
]:

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    if len(train_dataset) < BATCH_SIZE:
        train_loader = DataLoader(
            train_dataset,
            batch_size=len(train_dataset),
            shuffle=True,
            drop_last=False,
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
# Predict
# ============================================================

def predict(
    model: FeedForwardNetwork,
    X: np.ndarray,
) -> np.ndarray:

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
# Metrics
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:

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
    }


# ============================================================
# Naive Mean Effects
# ============================================================

class NaiveMeanEffects:
    """
    Baseline model using the global mean plus cell-line
    and drug effects.
    """

    def __init__(self) -> None:

        self.global_mean = 0.0
        self.cell_effects = {}
        self.drug_effects = {}

    def fit(
        self,
        data: pd.DataFrame,
    ) -> None:

        data = data.copy()

        data[DRUG_ID] = (
            data[DRUG_ID]
            .astype(str)
            .str.strip()
        )

        self.global_mean = float(
            data[
                STANDARDISED_RESPONSE
            ].mean()
        )

        global_mean = self.global_mean

        cell_means = (
            data.groupby(CELL_NAME)[
                STANDARDISED_RESPONSE
            ]
            .mean()
        )

        drug_means = (
            data.groupby(DRUG_ID)[
                STANDARDISED_RESPONSE
            ]
            .mean()
        )

        self.cell_effects = (
            cell_means
            .sub(global_mean)
            .to_dict()
        )

        self.drug_effects = (
            drug_means
            .sub(global_mean)
            .to_dict()
        )

    def predict(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:

        data = data.copy()

        data[DRUG_ID] = (
            data[DRUG_ID]
            .astype(str)
            .str.strip()
        )

        predictions = np.full(
            len(data),
            self.global_mean,
            dtype=float,
        )

        for i, row in data.reset_index(
            drop=True
        ).iterrows():

            predictions[i] += (
                self.cell_effects.get(
                    row[CELL_NAME],
                    0.0,
                )
            )

            predictions[i] += (
                self.drug_effects.get(
                    str(row[DRUG_ID]),
                    0.0,
                )
            )

        return predictions


# ============================================================
# Actual vs predicted plot
# ============================================================

def save_actual_vs_predicted_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.5,
    )

    minimum = min(
        y_true.min(),
        y_pred.min(),
    )

    maximum = max(
        y_true.max(),
        y_pred.max(),
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


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--project-root",
        required=True,
        help="Project root directory.",
    )

    parser.add_argument(
        "--within-study-dataset",
        required=True,
        help="Name of the within-study dataset.",
    )

    parser.add_argument(
        "--cross-study-datasets",
        nargs="+",
        required=True,
        help=(
            "Names of the cross-study datasets to evaluate, "
            "e.g. CTRPv1 CTRPv2 GDSC1."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of LCO split sets to evaluate.",
    )

    args = parser.parse_args()

    set_seed()

    project_root = Path(
        args.project_root
    )

    dataset = args.within_study_dataset

    cross_study_datasets = (
        args.cross_study_datasets
    )

    output_dir = Path(
        args.output_dir
    )

    # --------------------------------------------------------
    # Dataset paths
    # --------------------------------------------------------

    standardised_root = (
        project_root
        / "data"
        / "standardised"
    )

    within_study_gene_expression_root = (
        project_root
        / "data"
        / "processed_gene_expression"
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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load within-study gene expression
    # --------------------------------------------------------

    gene_expression = (
        load_gene_expression(
            within_study_gene_expression_root
            / f"{dataset}_gene_expression_processed.csv"
        )
    )

    # --------------------------------------------------------
    # Load within-study fingerprints
    # --------------------------------------------------------

    if dataset in {"GDSC1", "GDSC2"}:

        fingerprint_filename = (
            f"{dataset}_fingerprints_128_recovered.csv"
        )

    else:

        fingerprint_filename = (
            f"{dataset}_fingerprints_128_transposed.csv"
        )

    fingerprints = (
        load_fingerprints(
            within_study_fingerprint_root
            / fingerprint_filename
        )
    )

    gene_columns = (
        get_gene_expression_columns(
            gene_expression
        )
    )

    fingerprint_columns = (
        get_fingerprint_columns(
            fingerprints
        )
    )

    # ========================================================
    # Process each LCO split
    # ========================================================

    for split_number in range(
        1,
        args.n_splits + 1,
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"SIMPLE NEURAL NETWORK — "
            f"{dataset} — SPLIT {split_number}"
        )

        print(
            "=" * 70
        )

        split_dir = (
            standardised_root
            / f"split_{split_number}"
        )

        split_output_dir = (
            output_dir
            / f"split_{split_number}"
        )

        split_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Load split-specific response scaler
        # ----------------------------------------------------

        response_scaler_path = (
            split_dir
            / "scaler.pkl"
        )

        if not response_scaler_path.exists():
            raise FileNotFoundError(
                "Response scaler not found:\n"
                f"{response_scaler_path}"
            )

        response_scaler = joblib.load(
            response_scaler_path
        )

        # ----------------------------------------------------
        # Load response data
        # ----------------------------------------------------

        train = load_csv(
            split_dir
            / "train_standardised.csv"
        )

        validation = load_csv(
            split_dir
            / "validation_standardised.csv"
        )

        test = load_csv(
            split_dir
            / "test_standardised.csv"
        )

        # ----------------------------------------------------
        # Keep only observations with valid fingerprints
        # ----------------------------------------------------

        train = filter_response_to_valid_fingerprints(
            train,
            fingerprints,
        )

        validation = filter_response_to_valid_fingerprints(
            validation,
            fingerprints,
        )

        test = filter_response_to_valid_fingerprints(
            test,
            fingerprints,
        )

        # ====================================================
        # Prepare training and validation data
        # ====================================================

        (
            train_gene,
            train_fp,
            y_train,
            _,
        ) = prepare_model_data(
            train,
            gene_expression,
            fingerprints,
            gene_columns,
            fingerprint_columns,
        )

        (
            validation_gene,
            validation_fp,
            y_validation,
            _,
        ) = prepare_model_data(
            validation,
            gene_expression,
            fingerprints,
            gene_columns,
            fingerprint_columns,
        )

        # ----------------------------------------------------
        # Fit tuning scaler on training cell lines only
        # ----------------------------------------------------

        training_cell_ids = set(
            train[CELL_ID]
            .astype(str)
            .str.strip()
            .unique()
        )

        tuning_scaler = (
            fit_gene_expression_scaler(
                gene_expression,
                training_cell_ids,
                gene_columns,
            )
        )

        X_train = transform_inputs(
            train_gene,
            train_fp,
            tuning_scaler,
        )

        X_validation = transform_inputs(
            validation_gene,
            validation_fp,
            tuning_scaler,
        )

        # ====================================================
        # Hyperparameter tuning
        # ====================================================

        tuning_results = []

        best_hidden_layers = None
        best_validation_mse = float("inf")

        for hidden_layers in HIDDEN_LAYER_CONFIGURATIONS:

            print(
                "\nTesting architecture:",
                hidden_layers,
            )

            set_seed()

            (
                model,
                train_losses,
                _,
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

            tuning_results.append(
                {
                    "hidden_layers": str(
                        hidden_layers
                    ),
                    "dropout_prob": DROPOUT_PROB,
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "max_epochs": MAX_EPOCHS,
                    "best_epoch": len(
                        train_losses
                    ),
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

            if (
                metrics["MSE"]
                < best_validation_mse
            ):

                best_validation_mse = (
                    metrics["MSE"]
                )

                best_hidden_layers = (
                    hidden_layers.copy()
                )

        tuning_df = pd.DataFrame(
            tuning_results
        )

        tuning_df.to_csv(
            split_output_dir
            / "simple_neural_network_tuning.csv",
            index=False,
        )

        # ----------------------------------------------------
        # Save selected hyperparameters
        # ----------------------------------------------------

        with open(
            split_output_dir
            / "best_hyperparameters.json",
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
        # Final within-study model
        # Train + validation
        # ====================================================

        train_validation = pd.concat(
            [
                train,
                validation,
            ],
            ignore_index=True,
        )

        (
            final_gene,
            final_fp,
            y_final,
            _,
        ) = prepare_model_data(
            train_validation,
            gene_expression,
            fingerprints,
            gene_columns,
            fingerprint_columns,
        )

        final_training_cell_ids = set(
            train_validation[CELL_ID]
            .astype(str)
            .str.strip()
            .unique()
        )

        final_scaler = (
            fit_gene_expression_scaler(
                gene_expression,
                final_training_cell_ids,
                gene_columns,
            )
        )

        X_final = transform_inputs(
            final_gene,
            final_fp,
            final_scaler,
        )

        set_seed()

        (
            final_model,
            _,
            _,
        ) = train_model(
            X_train=X_final,
            y_train=y_final,
            X_validation=None,
            y_validation=None,
            hidden_layers=best_hidden_layers,
        )

        with open(
            split_output_dir
            / "gene_expression_scaler.pkl",
            "wb",
        ) as file:

            pickle.dump(
                final_scaler,
                file,
            )

        torch.save(
            final_model.state_dict(),
            split_output_dir
            / "simple_neural_network_model.pt",
        )

        # ====================================================
        # Within-study test

        # Keep only test observations with matching
        # gene-expression data
        # ====================================================

        test[CELL_ID] = (
            test[CELL_ID]
            .astype(str)
            .str.strip()
        )
        test_model_response = (
            test.merge(
                gene_expression[[CELL_ID]],
                on=CELL_ID,
                how="inner",
            )
        )
        
        ( 
            test_gene,
            test_fp,
            y_test,
            _,        
        ) = prepare_model_data(
            test_model_response,
            gene_expression,
            fingerprints,
            gene_columns,
            fingerprint_columns,
        )

        X_test = transform_inputs(
            test_gene,
            test_fp,
            final_scaler,
        )

        test_predictions_standardised = predict(
            final_model,
            X_test,
        )

        y_test_original = (
            inverse_transform_response(
                y_test,
                response_scaler,
            )
        )

        test_predictions_original = (
            inverse_transform_response(
                test_predictions_standardised,
                response_scaler,
            )
        )

        test_metrics = calculate_metrics(
            y_test_original,
            test_predictions_original,
        )

        print(
            f"\n{dataset} TEST:"
        )

        print(
            test_metrics
        )

        # ====================================================
        # Within-study Naive Mean Effects
        # Train + validation
        # ====================================================

        naive_model = (
            NaiveMeanEffects()
        )

        naive_model.fit(
            train_validation
        )

        naive_test_predictions_standardised = (
            naive_model.predict(
                test_model_response
            )
        )

        naive_test_predictions_original = (
            inverse_transform_response(
                naive_test_predictions_standardised,
                response_scaler,
            )
        )

        naive_test_metrics = (
            calculate_metrics(
                y_test_original,
                naive_test_predictions_original,
            )
        )

        # ====================================================
        # Results
        # ====================================================

        results = [
            {
                "split": split_number,
                "model":
                    "Simple Neural Network",
                "dataset":
                    f"{dataset}_test",
                "R2":
                    test_metrics["R2"],
                "Pearson":
                    test_metrics["Pearson"],
                "Spearman":
                    test_metrics["Spearman"],
                "MSE":
                    test_metrics["MSE"],
            },
            {
                "split": split_number,
                "model":
                    "Naive Mean Effects",
                "dataset":
                    f"{dataset}_test",
                "R2":
                    naive_test_metrics["R2"],
                "Pearson":
                    naive_test_metrics["Pearson"],
                "Spearman":
                    naive_test_metrics["Spearman"],
                "MSE":
                    naive_test_metrics["MSE"],
            },
        ]

        save_actual_vs_predicted_plot(
            y_test_original,
            test_predictions_original,
            (
                f"Simple Neural Network — "
                f"{dataset} Test — "
                f"Split {split_number}"
            ),
            split_output_dir
            / f"{dataset}_test_actual_vs_predicted.png",
        )

        # ====================================================
        # Cross-study model
        #
        # Training data ONLY
        # ====================================================

        cross_study_scaler = tuning_scaler

        X_cross_training = transform_inputs(
            train_gene,
            train_fp,
            cross_study_scaler,
        )

        set_seed()

        (
            cross_study_model,
            _,
            _,
        ) = train_model(
            X_train=X_cross_training,
            y_train=y_train,
            X_validation=None,
            y_validation=None,
            hidden_layers=best_hidden_layers,
        )

        torch.save(
            cross_study_model.state_dict(),
            split_output_dir
            / "cross_study_simple_neural_network_model.pt",
        )

        # ----------------------------------------------------
        # Cross-study Naive Mean Effects
        # Training data only
        # ----------------------------------------------------

        cross_study_naive_model = (
            NaiveMeanEffects()
        )

        cross_study_naive_model.fit(
            train
        )

        with open(
            split_output_dir
            / "cross_study_naive_mean_effects_model.pkl",
            "wb",
        ) as file:

            pickle.dump(
                cross_study_naive_model,
                file,
            )

        # ====================================================
        # Cross-study evaluation
        # ====================================================

        for cross_study_dataset in cross_study_datasets:

            print(
                f"\nCross-study dataset: "
                f"{cross_study_dataset}"
            )

            cross_study_response = load_csv(
                split_dir
                / f"{cross_study_dataset}_matched_standardised.csv"
            )

            cross_study_gene_expression = (
                load_gene_expression(
                    cross_study_gene_expression_root
                    / (
                        f"gene_expression_"
                        f"{cross_study_dataset}_processed.csv"
                    )
                )
            )

            if cross_study_dataset in {
                "GDSC1",
                "GDSC2",
            }:

                cross_study_fingerprint_filename = (
                    f"{cross_study_dataset}_"
                    f"fingerprints_128_recovered.csv"
                )

            else:

                cross_study_fingerprint_filename = (
                    f"{cross_study_dataset}_"
                    f"fingerprints_128_transposed.csv"
                )

            cross_study_fingerprint_path = (
                cross_study_fingerprint_root
                / cross_study_fingerprint_filename
            )

            cross_study_fingerprints = (
                load_fingerprints(
                    cross_study_fingerprint_path
                )
            )

            # ------------------------------------------------
            #  Keep the exact observations that have matching
            #  gene-expression and fingerprint data
            # ------------------------------------------------

            cross_study_response = (
                filter_response_to_valid_fingerprints(
                    cross_study_response,
                    cross_study_fingerprints,
                )
            )
            
            cross_study_model_response = (
                cross_study_response.merge(
                    cross_study_gene_expression[[CELL_ID]],
                    on=CELL_ID,
                    how="inner",
                )
            )

            (
                cross_gene,
                cross_fp,
                y_cross,
                _,
            ) = prepare_model_data(
                cross_study_model_response,
                cross_study_gene_expression,
                cross_study_fingerprints,
                gene_columns,
                fingerprint_columns,
            )

            X_cross = transform_inputs(
                cross_gene,
                cross_fp,
                cross_study_scaler,
            )

            # ------------------------------------------------
            # Cross-study neural network prediction
            # ------------------------------------------------

            cross_predictions_standardised = (
                predict(
                    cross_study_model,
                    X_cross,
                )
            )

            y_cross_original = (
                inverse_transform_response(
                    y_cross,
                    response_scaler,
                )
            )

            cross_predictions_original = (
                inverse_transform_response(
                    cross_predictions_standardised,
                    response_scaler,
                )
            )

            cross_metrics = calculate_metrics(
                y_cross_original,
                cross_predictions_original,
            )

            # ------------------------------------------------
            # Cross-study Naive Mean Effects
            # ------------------------------------------------

            naive_cross_predictions_standardised = (
                cross_study_naive_model.predict(
                    cross_study_model_response
                )
            )

            naive_cross_predictions_original = (
                inverse_transform_response(
                    naive_cross_predictions_standardised,
                    response_scaler,
                )
            )

            naive_cross_metrics = (
                calculate_metrics(
                    y_cross_original,
                    naive_cross_predictions_original,
                )
            )

            results.extend(
                [
                    {
                        "split": split_number,
                        "model":
                            "Simple Neural Network",
                        "dataset":
                            f"{cross_study_dataset}_cross_study",
                        "R2":
                            cross_metrics["R2"],
                        "Pearson":
                            cross_metrics["Pearson"],
                        "Spearman":
                            cross_metrics["Spearman"],
                        "MSE":
                            cross_metrics["MSE"],
                    },
                    {
                        "split": split_number,
                        "model":
                            "Naive Mean Effects",
                        "dataset":
                            f"{cross_study_dataset}_cross_study",
                        "R2":
                            naive_cross_metrics["R2"],
                        "Pearson":
                            naive_cross_metrics["Pearson"],
                        "Spearman":
                            naive_cross_metrics["Spearman"],
                        "MSE":
                            naive_cross_metrics["MSE"],
                    },
                ]
            )

            print(
                f"{cross_study_dataset} | "
                f"NN R2 = "
                f"{cross_metrics['R2']:.4f} | "
                f"Naive R2 = "
                f"{naive_cross_metrics['R2']:.4f}"
            )

            save_actual_vs_predicted_plot(
                y_cross_original,
                cross_predictions_original,
                (
                    f"Simple Neural Network — "
                    f"{cross_study_dataset} — "
                    f"Split {split_number}"
                ),
                split_output_dir
                / (
                    f"{cross_study_dataset}_"
                    f"actual_vs_predicted.png"
                ),
            )

        # ====================================================
        # Save split metrics
        # ====================================================

        metrics_df = pd.DataFrame(
            results
        )

        metrics_df.to_csv(
            split_output_dir
            / "metrics.csv",
            index=False,
        )

        print(
            f"\nSplit {split_number} complete."
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SIMPLE NEURAL NETWORK ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()

