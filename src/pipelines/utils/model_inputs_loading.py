"""Utilities for loading model inputs."""

import pandas as pd
import yaml

from constants.paths import (
    GOLD_TEST_AIRPORT_DATA_PATH,
    GOLD_TRAINING_AIRPORT_DATA_PATH,
    PARAMS_FILE,
)

TARGET_COLUMN = "actual_taxi_out_sec"


def load_params() -> dict:
    """Load parameters from /params.yaml.

    Returns:
        dict: Parameters dictionary.
    """
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def load_training_data():
    """Load training and testing data for model training.

    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Training feature matrix.
            - y_train (pd.Series): Training target vector.
            - X_test (pd.DataFrame): Testing feature matrix.
            - y_test (pd.Series): Testing target vector.
    """
    # Load gold datasets
    train_data = pd.read_parquet(GOLD_TRAINING_AIRPORT_DATA_PATH)
    test_data = pd.read_parquet(GOLD_TEST_AIRPORT_DATA_PATH)

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
