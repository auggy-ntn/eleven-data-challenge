"""Model prediction making utils.

We want to load a trained model from MLflow and use it to make predictions with
SHAP explanations, and create a dataset for the dashboards.
"""

import os

from dotenv import load_dotenv
import mlflow
import numpy as np
import pandas as pd
import shap

from constants.column_names import (
    ACTUAL_TAXI_OUT_SEC,
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
from constants.paths import (
    DASHBOARD_DATA_CSV_PATH,
    DASHBOARD_DATA_PARQUET_PATH,
    GOLD_TEST_AIRPORT_DATA_PATH,
)
from src.utils.logger import logger


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


def prediction_pipeline(
    n_samples: int = 10,
    model_type: str = "xgboost",
    run_id: str | None = None,
    random_seed: int = 42,
):
    """Run the prediction pipeline to create a dashboard dataset.

    This pipeline:
    1. Loads the best model from MLflow (or a specific run)
    2. Samples test data
    3. Makes predictions
    4. Computes SHAP values
    5. Creates and saves a dashboard dataset with SHAP category impacts

    Args:
        n_samples: Number of samples to use for the dashboard dataset.
        model_type: Type of model to load ("xgboost" or "linear_regression").
        run_id: Specific MLflow run ID to load. If None, loads the best model.
        random_seed: Random seed for reproducibility.
    """
    # Load MLflow environment
    logger.info("Loading MLflow environment")
    load_mlflow_env()

    # Get the best model run ID if not specified
    if run_id is None:
        experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
        if experiment_id is None:
            raise ValueError("MLFLOW_EXPERIMENT_ID not set in environment")
        logger.info(f"Finding best model from experiment {experiment_id}")
        run_id = get_best_run_id(experiment_id)

    logger.info(f"Loading model from run {run_id}")
    model = load_model(run_id, model_type)
    logger.info(f"Model loaded: {type(model).__name__}")

    # Load and sample test data
    logger.info(f"Loading test data from {GOLD_TEST_AIRPORT_DATA_PATH}")
    test_data = pd.read_parquet(GOLD_TEST_AIRPORT_DATA_PATH)

    np.random.seed(random_seed)
    sample_indices = np.random.choice(len(test_data), size=n_samples, replace=False)
    test_sample = test_data.iloc[sample_indices].copy().reset_index(drop=True)
    logger.info(f"Sampled {len(test_sample)} flights from {len(test_data)} total")

    # Prepare features and target
    X = test_sample.drop(columns=[ACTUAL_TAXI_OUT_SEC])
    y_true = test_sample[ACTUAL_TAXI_OUT_SEC]

    # Make predictions
    logger.info("Making predictions")
    y_pred = model.predict(X)
    logger.info(f"Predictions complete for {len(y_pred)} samples")

    # Compute SHAP values
    logger.info("Computing SHAP values")
    base_value, shap_values = compute_shap_values(model, model_type, X)
    logger.info(f"SHAP values shape: {shap_values.shape}")
    logger.info(f"Base value: {base_value:.2f} seconds ({base_value / 60:.1f} min)")

    # Create dashboard dataset
    logger.info("Creating dashboard dataset")
    dashboard_df = create_dashboard_dataset(X, y_true, y_pred, shap_values)

    # Save dashboard dataset
    DASHBOARD_DATA_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    dashboard_df.to_csv(DASHBOARD_DATA_CSV_PATH, index=False)
    dashboard_df.to_parquet(DASHBOARD_DATA_PARQUET_PATH, index=False)
    logger.info(f"Dashboard dataset saved to {DASHBOARD_DATA_CSV_PATH}")
    logger.info(f"Dashboard dataset saved to {DASHBOARD_DATA_PARQUET_PATH}")

    return dashboard_df


if __name__ == "__main__":
    prediction_pipeline()
