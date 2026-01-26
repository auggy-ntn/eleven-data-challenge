"""Pipeline to train models."""

from src.pipelines.models.linear_regression import train_linear_regression_model
from src.pipelines.utils.model_inputs_loading import (
    load_hyperparameter_grid,
    load_nb_optimization_trials,
    load_params,
    load_training_data,
)


def train_models(model_type: str, run_name: str, model_name: str):
    """Train machine learning models for plane taxi time prediction.

    Args:
        model_type (str): Type of model to train.
        run_name (str): Name of the MLflow run.
        model_name (str): Name to log the model under in MLflow.
    """
    # Load training data
    X_train, y_train, X_test, y_test = load_training_data()

    # Train specified model
    if model_type == "linear_regression":
        # Load hyperparameter grid and number of optimization trials
        linear_hyperparameter_grid = load_hyperparameter_grid(model_type)
        nb_optimization_trials = load_nb_optimization_trials()

        # Train Linear Regression model --> Stored and logged in MLflow
        train_linear_regression_model(
            run_name=run_name,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparameter_grid=linear_hyperparameter_grid,
            n_trials=nb_optimization_trials,
        )

    # TODO: Add other model types here - Follow the same structure as above

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    # Execute the pipeline --> Called using DVC
    # Parameters are loaded from /params.yaml
    params = load_params()
    train_params = params["train_models"]

    train_models(
        model_type=train_params["model_type"],
        run_name=train_params["run_name"],
        model_name=train_params["model_name"],
    )
