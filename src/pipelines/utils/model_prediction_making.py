"""Model prediction making utils.

Utility functions for loading models from MLflow, computing SHAP values,
and creating dashboard datasets.
"""

from dotenv import load_dotenv
import mlflow
import numpy as np
import pandas as pd
import shap

from constants.column_names import (
    AIRCRAFT_LENGTH,
    DASHBOARD_ACTUAL_TAXI_SEC,
    DASHBOARD_AIRCRAFT_IMPACT,
    DASHBOARD_AIRCRAFT_LENGTH,
    DASHBOARD_DISTANCE_IMPACT,
    DASHBOARD_NO_ENGINES,
    DASHBOARD_PREDICTED_TAXI_SEC,
    DASHBOARD_RUNWAY,
    DASHBOARD_STAND,
    DASHBOARD_TRAFFIC_IMPACT,
    DASHBOARD_WEATHER_IMPACT,
    NO_ENGINES,
    RUNWAY,
    STAND,
)
from constants.feature_categories import (
    AIRCRAFT,
    DISTANCE_CAT,
    TRAFFIC,
    WEATHER,
    get_feature_categories,
)


def load_mlflow_env():
    """Load MLflow environment variables from .env file."""
    load_dotenv()
    mlflow.set_tracking_uri("databricks")


def load_model(run_id: str, model_type: str):
    """Load a model from MLflow given its URI.

    Args:
        run_id (str): The run ID of the model in MLflow.
        model_type (str): The type of the model.

    Returns:
        The loaded model.
    """
    if model_type == "xgboost":
        model_name = "xgboost_model"
        model_uri = f"runs:/{run_id}/{model_name}"
        model = mlflow.xgboost.load_model(model_uri)

    elif model_type == "linear_regression":
        model_name = "linear_regression_model"
        model_uri = f"runs:/{run_id}/{model_name}"
        model = mlflow.sklearn.load_model(model_uri)

    return model


def get_best_run_id(experiment_id: str) -> str:
    """Get the best run ID from an MLflow experiment based on MAE.

    Args:
        experiment_id: The MLflow experiment ID.

    Returns:
        The run ID of the best performing model.
    """
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id], order_by=["start_time DESC"]
    )
    best_run = runs.loc[runs["metrics.mae"].idxmin()]
    return best_run["run_id"]


def compute_shap_values(
    model, model_type: str, X: pd.DataFrame
) -> tuple[float, np.ndarray]:
    """Compute SHAP values for model predictions.

    Args:
        model: Trained model.
        model_type (str): Type of the model.
        X (pd.DataFrame): Feature matrix.

    Returns:
        tuple: A tuple containing:
            - base_value (float): The expected value of the model output.
            - shap_values (np.ndarray): SHAP values for each feature.
    """
    if model_type == "xgboost":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear_regression":
        explainer = shap.LinearExplainer(model, X)

    shap_explanation = explainer(X)
    base_value = explainer.expected_value
    shap_values = shap_explanation.values

    return base_value, shap_values


def compute_category_impacts(
    shap_values_row: list[float],
    feature_names: list[str],
    categories: dict[str, list[str]],
) -> dict[str, float]:
    """Compute SHAP impact per category in seconds.

    Args:
        shap_values_row: SHAP values for a single prediction.
        feature_names: List of feature names corresponding to SHAP values.
        categories: Dictionary mapping category names to lists of feature names.

    Returns:
            Dictionary mapping category names to their SHAP impact in seconds.
    """
    shap_dict = dict(zip(feature_names, shap_values_row, strict=False))
    impacts = {}
    for cat_name, cat_features in categories.items():
        impacts[cat_name] = sum(shap_dict.get(f, 0) for f in cat_features)
    return impacts


def create_dashboard_dataset(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """Create a dataset for dashboard visualization.

    Args:
        X: DataFrame with feature data.
        y_true: Series with true target values.
        y_pred: Array with predicted values.
        shap_values: Array with SHAP values for each feature.

    Returns:
        DataFrame suitable for dashboard visualization with columns:
        - Stand, Runway, Aircraft Length, No. Engines
        - Actual Taxi Time (s), Predicted Taxi Time (s)
        - Weather, Traffic, Distance, Aircraft (SHAP impacts in seconds)
    """
    feature_cols = X.columns.tolist()
    feature_categories = get_feature_categories(feature_cols)

    dashboard_data = []
    for i in range(len(X)):
        row = X.iloc[i]
        impacts = compute_category_impacts(
            shap_values_row=shap_values[i],
            feature_names=feature_cols,
            categories=feature_categories,
        )

        dashboard_data.append(
            {
                DASHBOARD_STAND: f"STAND_{int(row[STAND])}",
                DASHBOARD_RUNWAY: f"RUNWAY_{int(row[RUNWAY])}",
                DASHBOARD_AIRCRAFT_LENGTH: row[AIRCRAFT_LENGTH],
                DASHBOARD_NO_ENGINES: int(row[NO_ENGINES]),
                DASHBOARD_ACTUAL_TAXI_SEC: int(y_true.iloc[i]),
                DASHBOARD_PREDICTED_TAXI_SEC: round(y_pred[i], 1),
                DASHBOARD_WEATHER_IMPACT: round(impacts[WEATHER], 1),
                DASHBOARD_TRAFFIC_IMPACT: round(impacts[TRAFFIC], 1),
                DASHBOARD_DISTANCE_IMPACT: round(impacts[DISTANCE_CAT], 1),
                DASHBOARD_AIRCRAFT_IMPACT: round(impacts[AIRCRAFT], 1),
            }
        )

    return pd.DataFrame(dashboard_data)
