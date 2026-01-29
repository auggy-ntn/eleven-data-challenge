"""Utilities for loading model inputs."""

import pandas as pd
import shap
import yaml

from constants.column_names import ACTUAL_TAXI_OUT_SEC
from constants.paths import (
    GOLD_TEST_AIRPORT_DATA_CLEAN_PATH,
    GOLD_TEST_AIRPORT_DATA_PATH,
    GOLD_TRAINING_AIRPORT_DATA_CLEAN_PATH,
    GOLD_TRAINING_AIRPORT_DATA_PATH,
    PARAMS_FILE,
)

TARGET_COLUMN = ACTUAL_TAXI_OUT_SEC


def load_params() -> dict:
    """Load parameters from /params.yaml.

    Returns:
        dict: Parameters dictionary.
    """
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def get_dataset_type(model_type: str) -> str:
    """Get the dataset type for a given model.

    Args:
        model_type (str): Type of model.

    Returns:
        str: Dataset type ('default' or 'clean').
    """
    params = load_params()
    datasets = params.get("datasets", {})
    return datasets.get(model_type, "default")


def load_training_data(model_type: str | None = None):
    """Load training and testing data for model training.

    Args:
        model_type (str | None): Type of model to determine which dataset to load.
            If None, loads the default dataset (with NaN).

    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Training feature matrix.
            - y_train (pd.Series): Training target vector.
            - X_test (pd.DataFrame): Testing feature matrix.
            - y_test (pd.Series): Testing target vector.
    """
    # Determine which dataset to load
    if model_type is not None:
        dataset_type = get_dataset_type(model_type)
    else:
        dataset_type = "default"

    # Select paths based on dataset type
    if dataset_type == "clean":
        train_path = GOLD_TRAINING_AIRPORT_DATA_CLEAN_PATH
        test_path = GOLD_TEST_AIRPORT_DATA_CLEAN_PATH
    else:
        train_path = GOLD_TRAINING_AIRPORT_DATA_PATH
        test_path = GOLD_TEST_AIRPORT_DATA_PATH

    # Load gold datasets
    train_data = pd.read_parquet(train_path)
    test_data = pd.read_parquet(test_path)

    # Split into features and target
    X_train = train_data.drop(columns=[TARGET_COLUMN])
    y_train = train_data[TARGET_COLUMN]

    X_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]

    return X_train, y_train, X_test, y_test


def load_hyperparameter_grid(model_type: str) -> dict:
    """Load hyperparameter grid for a given model type.

    Args:
        model_type (str): Type of model to load hyperparameter grid for.

    Returns:
        dict: Hyperparameter grid for the specified model type.

    Raises:
        KeyError: If hyperparameter grid for model_type is not found.
    """
    params = load_params()
    return params["hyperparameters"][model_type]


def load_nb_optimization_trials() -> int:
    """Load the number of optimization trials for hyperparameter tuning.

    Returns:
        int: Number of optimization trials.
    """
    params = load_params()
    return params["train_models"]["n_trials"]


def load_prediction_data(n_samples: int) -> tuple[pd.DataFrame, pd.Series]:
    """Load data for model prediction and SHAP explanations.

    Args:
        n_samples (int): Number of samples to load for predictions.

    Returns:
        tuple: A tuple containing:
            - X_test (pd.DataFrame): Feature matrix for predictions.
            - y_test (pd.Series): Target vector for predictions.
    """
    test_data = pd.read_parquet(GOLD_TEST_AIRPORT_DATA_PATH)
    test_data = test_data.sample(n=n_samples).reset_index(drop=True)
    X_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]
    return X_test, y_test


def compute_shap_values(
    model, model_type: str, X: pd.DataFrame
) -> tuple[float, pd.DataFrame]:
    """Compute SHAP values for model predictions.

    Args:
        model: Trained model.
        model_type (str): Type of the model.
        X (pd.DataFrame): Feature matrix.

    Returns:
        tuple: A tuple containing:
            - base_value (float): The expected value of the model output.
            - shap_values (pd.DataFrame): SHAP values for each feature.
    """
    if model_type == "xgboost":
        explainer = shap.TreeExplainer(model)

    elif model_type == "linear_regression":
        explainer = shap.LinearExplainer(model, X)

    base_value = explainer.expected_value
    shap_values = explainer.shap_values(X)

    return base_value, shap_values
