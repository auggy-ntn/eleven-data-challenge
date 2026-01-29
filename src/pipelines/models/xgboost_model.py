"""XGBoost Model Implementation.

Defines a training, optimization, and evaluation pipeline for an XGBoost model.
Logs metrics and model artifacts using MLflow.
"""

from dotenv import load_dotenv
import mlflow
from mlflow.models.signature import infer_signature
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from xgboost import XGBRegressor


# Model Optimization Function
def optimize_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hyperparameter_grid: dict | None,
    n_trials: int = 50,
    **kwargs,
) -> dict:
    """Optimize an XGBoost model with Optuna.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.
        X_test (pd.DataFrame): Testing feature matrix.
        y_test (pd.Series): Testing target vector.
        hyperparameter_grid (dict | None): Hyperparameter grid for optimization.
            If None, a default search space is used.
        n_trials (int, optional): Number of Optuna trials. Defaults to 50.
        **kwargs: Additional keyword arguments for XGBRegressor.

    Returns:
        dict: Best hyperparameters found by Optuna.
    """

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
            # Default search space for XGBoost
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

        model = XGBRegressor(
            **params,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            **kwargs,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        optimization_metric = root_mean_squared_error(y_test, y_pred)
        return optimization_metric

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    return best_params


# Model Training and Logging Function
def train_xgboost_model(
    run_name: str,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hyperparameter_grid: dict | None = None,
    n_trials: int = 50,
    **kwargs,
):
    """Train and log an XGBoost model using MLflow.

    Args:
        run_name (str, optional): Name of the MLflow run.
        model_name (str): Name to log the model under in MLflow.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.
        X_test (pd.DataFrame): Testing feature matrix.
        y_test (pd.Series): Testing target vector.
        hyperparameter_grid (dict | None): Hyperparameter grid for optimization.
            If None, a default search space is used. Defaults to None.
        n_trials (int, optional): Number of Optuna trials. Defaults to 50.
        **kwargs: Additional keyword arguments for XGBRegressor.
    """
    # Load environment variables for MLflow configuration
    load_dotenv()

    with mlflow.start_run(run_name=run_name):
        best_params = optimize_xgboost_model(
            X_train,
            y_train,
            X_test,
            y_test,
            hyperparameter_grid,
            n_trials,
            **kwargs,
        )

        # Fit model with best parameters
        model = XGBRegressor(
            **best_params,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            **kwargs,
        )
        model.fit(X_train, y_train)

        # Compute evalutation metrics - TODO: Add more metrics
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # Log parameters
        mlflow.log_params(best_params)

        # Log metrics - TODO: Add more metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mape", mape)

        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.xgboost.log_model(
            model,
            name=model_name,
            signature=signature,
            pip_requirements=None,
            conda_env=None,
        )

    return None
