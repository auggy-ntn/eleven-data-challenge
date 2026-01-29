"""Create predictions pipeline.

This pipeline loads a trained model from MLflow, makes predictions on test data,
computes SHAP explanations, and creates a dashboard dataset.
"""

import os

import numpy as np
import pandas as pd

from constants.column_names import ACTUAL_TAXI_OUT_SEC
from constants.paths import (
    DASHBOARD_DATA_CSV_PATH,
    DASHBOARD_DATA_PARQUET_PATH,
    GOLD_TEST_AIRPORT_DATA_PATH,
    PREDICTIONS_DIR,
)
from src.pipelines.utils.model_prediction_making import (
    compute_shap_values,
    create_dashboard_dataset,
    get_best_run_id,
    load_mlflow_env,
    load_model,
)
from src.utils.logger import logger


def create_predictions(
    n_samples: int = 10,
    model_type: str = "xgboost",
    run_id: str | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
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

    Returns:
        DataFrame with dashboard data.
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
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    dashboard_df.to_csv(DASHBOARD_DATA_CSV_PATH, index=False)
    dashboard_df.to_parquet(DASHBOARD_DATA_PARQUET_PATH, index=False)
    logger.info(f"Dashboard dataset saved to {DASHBOARD_DATA_CSV_PATH}")
    logger.info(f"Dashboard dataset saved to {DASHBOARD_DATA_PARQUET_PATH}")

    return dashboard_df


if __name__ == "__main__":
    logger.info("Starting create predictions pipeline")
    create_predictions()
    logger.info("Finished create predictions pipeline")
