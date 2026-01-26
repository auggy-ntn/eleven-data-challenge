"""Utilities for loading model inputs."""

import yaml

from constants.paths import PARAMS_FILE


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
    # Implement your data loading logic here
    pass


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
