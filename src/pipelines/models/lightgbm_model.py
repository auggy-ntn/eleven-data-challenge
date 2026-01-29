"""LightGBM Model Implementation.

Defines a training, optimization, and evaluation pipeline for a LightGBM model.
Logs metrics and model artifacts using MLflow.
"""

from dotenv import load_dotenv
from lightgbm import LGBMRegressor
import mlflow
from mlflow.models.signature import infer_signature
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


# Model Optimization Function
def optimize_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparameter_grid: dict | None,
    n_trials: int = 30,
    validation_split: float = 0.2,
    **kwargs,
) -> dict:
    """Optimize a LightGBM model with Optuna using time series split.

    Uses the last portion of training data as validation (time series split).

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.
        hyperparameter_grid (dict | None): Hyperparameter grid for optimization.
            If None, a default search space is used.
        n_trials (int, optional): Number of Optuna trials. Defaults to 30.
        validation_split (float, optional): Fraction of training data to use for
            validation (taken from the end). Defaults to 0.2.
        **kwargs: Additional keyword arguments for LGBMRegressor.

    Returns:
        dict: Best hyperparameters found by Optuna.
    """
    # Time series split: use last portion as validation
    split_idx = int(len(X_train) * (1 - validation_split))
    X_opt_train = X_train.iloc[:split_idx]
    y_opt_train = y_train.iloc[:split_idx]
    X_opt_val = X_train.iloc[split_idx:]
    y_opt_val = y_train.iloc[split_idx:]

    def objective(trial):
        """Internal objective function for Optuna hyperparameter tuning.

        Args:
            trial (optuna.Trial): Optuna trial object for hyperparameter suggestions.

        Returns:
            float: RMSE metric for the trial.
        """
        # Use provided grid or create default search space
        if hyperparameter_grid is not None:
            params = {}
            for key, values in hyperparameter_grid.items():
                if isinstance(values, list):
                    params[key] = trial.suggest_categorical(key, values)
                elif isinstance(values, dict):
                    # Handle range-based parameters
                    if values.get("type") == "int":
                        params[key] = trial.suggest_int(
                            key,
                            values["low"],
                            values["high"],
                            log=values.get("log", False),
                        )
                    elif values.get("type") == "float":
                        params[key] = trial.suggest_float(
                            key,
                            values["low"],
                            values["high"],
                            log=values.get("log", False),
                        )
        else:
            # Default search space for LightGBM
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

        model = LGBMRegressor(
            **params,
            random_state=42,
            device="gpu",
            verbosity=-1,
            **kwargs,
        )
        model.fit(X_opt_train, y_opt_train)
        y_pred = model.predict(X_opt_val)
        optimization_metric = root_mean_squared_error(y_opt_val, y_pred)
        return optimization_metric

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    return best_params


# Model Training and Logging Function
def train_lightgbm_model(
    run_name: str,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hyperparameter_grid: dict | None = None,
    n_trials: int = 30,
    **kwargs,
):
    """Train and log a LightGBM model using MLflow.

    Optimizes hyperparameters using a time series split on training data,
    then trains the final model on the full training set.

    Args:
        run_name (str, optional): Name of the MLflow run.
        model_name (str): Name to log the model under in MLflow.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.
        X_test (pd.DataFrame): Testing feature matrix.
        y_test (pd.Series): Testing target vector.
        hyperparameter_grid (dict | None): Hyperparameter grid for optimization.
            If None, a default search space is used. Defaults to None.
        n_trials (int, optional): Number of Optuna trials. Defaults to 30.
        **kwargs: Additional keyword arguments for LGBMRegressor.
    """
    # Load environment variables for MLflow configuration
    load_dotenv()

    with mlflow.start_run(run_name=run_name):
        # Optimize hyperparameters using time series split on training data only
        best_params = optimize_lightgbm_model(
            X_train,
            y_train,
            hyperparameter_grid,
            n_trials,
            **kwargs,
        )

        # Fit final model with best parameters on FULL training set
        model = LGBMRegressor(
            **best_params,
            random_state=42,
            device="gpu",
            verbosity=-1,
            **kwargs,
        )
        model.fit(X_train, y_train)

        # Evaluate on held-out test set
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        std = np.std(y_test - y_pred)

        # Log parameters
        mlflow.log_params(best_params)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("std", std)

        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.lightgbm.log_model(
            model,
            name=model_name,
            signature=signature,
            pip_requirements=None,
            conda_env=None,
        )

    return None
