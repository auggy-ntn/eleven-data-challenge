"""Pipeline to train models."""

from src.pipelines.models.catboost_model import train_catboost_model
from src.pipelines.models.lightgbm_model import train_lightgbm_model
from src.pipelines.models.linear_regression import train_linear_regression_model
from src.pipelines.models.random_forest import train_random_forest_model
from src.pipelines.models.xgboost_model import train_xgboost_model
from src.pipelines.utils.model_inputs_loading import (
    load_hyperparameter_grid,
    load_params,
    load_training_data,
)


def train_model(model_type: str, X_train, y_train, X_test, y_test):
    """Train a single model.

    Args:
        model_type (str): Type of model to train.
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_test: Testing feature matrix.
        y_test: Testing target vector.
    """
    run_name = f"{model_type}_run"
    model_name = f"{model_type}_model"

    if model_type == "linear_regression":
        # Load hyperparameter grid
        hyperparameter_grid = load_hyperparameter_grid(model_type)

        # Train Linear Regression model --> Stored and logged in MLflow
        train_linear_regression_model(
            run_name=run_name,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparameter_grid=hyperparameter_grid,
        )

    elif model_type == "xgboost":
        # Load hyperparameter grid
        hyperparameter_grid = load_hyperparameter_grid(model_type)

        # Train XGBoost model --> Stored and logged in MLflow
        train_xgboost_model(
            run_name=run_name,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparameter_grid=hyperparameter_grid,
        )

    elif model_type == "random_forest":
        # Load hyperparameter grid
        hyperparameter_grid = load_hyperparameter_grid(model_type)

        # Train Random Forest model --> Stored and logged in MLflow
        train_random_forest_model(
            run_name=run_name,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparameter_grid=hyperparameter_grid,
        )

    elif model_type == "lightgbm":
        # Load hyperparameter grid
        hyperparameter_grid = load_hyperparameter_grid(model_type)

        # Train LightGBM model --> Stored and logged in MLflow
        train_lightgbm_model(
            run_name=run_name,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparameter_grid=hyperparameter_grid,
        )

    elif model_type == "catboost":
        # Load hyperparameter grid
        hyperparameter_grid = load_hyperparameter_grid(model_type)

        # Train CatBoost model --> Stored and logged in MLflow
        train_catboost_model(
            run_name=run_name,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparameter_grid=hyperparameter_grid,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_models(model_types: list[str]):
    """Train multiple machine learning models for plane taxi time prediction.

    Args:
        model_types (list[str]): List of model types to train.
    """
    # Train each model with its appropriate dataset
    for model_type in model_types:
        print(f"\n{'=' * 50}")
        print(f"Training {model_type}...")
        print(f"{'=' * 50}\n")

        # Load data specific to this model type (clean or default)
        X_train, y_train, X_test, y_test = load_training_data(model_type)
        print(
            f"Loaded dataset for {model_type}: "
            f"{X_train.shape[0]} train, {X_test.shape[0]} test"
        )

        train_model(model_type, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    # Execute the pipeline --> Called using DVC
    # Parameters are loaded from /params.yaml
    params = load_params()
    train_params = params["train_models"]

    train_models(model_types=train_params["model_types"])
