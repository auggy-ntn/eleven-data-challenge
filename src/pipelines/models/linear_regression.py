"""Linear Regression Model Implementation.

Defines a training, optimization, and evaluation pipeline for a linear regression model.
Logs metrics and model artifacts using MLflow.
"""

from dotenv import load_dotenv
import mlflow
from mlflow.models.signature import infer_signature
import optuna
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


# Model Optimization Function
def optimize_linear_regression_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hyperparameter_grid: dict | None,
    n_trials: int = 50,
    **kwargs,
) -> dict:
    """Optimize a Linear Regression model with Optuna.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.
        X_test (pd.DataFrame): Testing feature matrix.
        y_test (pd.Series): Testing target vector.
        hyperparameter_grid (dict | None): Hyperparameter grid for optimization.
            If None, a default search space is used.
        n_trials (int, optional): Number of Optuna trials. Defaults to 50.
        **kwargs: Additional keyword arguments for LinearRegression.

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
            params = {
                key: trial.suggest_categorical(key, values)
                for key, values in hyperparameter_grid.items()
            }
        else:
            # Default search space for Linear Regression
            params = {
                "fit_intercept": trial.suggest_categorical(
                    "fit_intercept", [True, False]
                ),
            }
        model = LinearRegression(**params, **kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        optimization_metric = root_mean_squared_error(y_test, y_pred)
        return optimization_metric

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    return best_params


# Model Training and Logging Function
def train_linear_regression_model(
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
    """Train and log a Linear Regression model using MLflow.

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
        **kwargs: Additional keyword arguments for LinearRegression.
    """
    # Load environment variables for MLflow configuration
    load_dotenv()

    with mlflow.start_run(run_name=run_name):
        best_params = optimize_linear_regression_model(
            X_train,
            y_train,
            X_test,
            y_test,
            hyperparameter_grid,
            n_trials,
            **kwargs,
        )

        # Fit model with best parameters
        model = LinearRegression(**best_params, **kwargs)
        model.fit(X_train, y_train)

        # Compute evalutation metrics - TODO: Add more metrics
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)

        # Log parameters
        mlflow.log_params(best_params)

        # Log metrics - TODO: Add more metrics
        mlflow.log_metric("rmse", rmse)

        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            name=model_name,
            signature=signature,
            pip_requirements=None,
            conda_env=None,
        )

    return None
