"""Bronze to Silver Pipeline."""

import re

import pandas as pd

from constants.column_names import RUNWAY, STAND
from constants.paths import (
    BRONZE_AC_CHAR_PATH,
    BRONZE_GEOGRAPHIC_DATA_PATH,
    BRONZE_GLOSSARY_PATH,
    BRONZE_TEST_AIRPORT_DATA_PATH,
    BRONZE_TEST_GEOGRAPHIC_DATA_PATH,
    BRONZE_TEST_WEATHER_DATA_PATH,
    BRONZE_TRAINING_AIRPORT_DATA_PATH,
    BRONZE_WEATHER_DATA_PATH,
    SILVER_AC_CHAR_PATH,
    SILVER_DIR,
    SILVER_GEOGRAPHIC_DATA_PATH,
    SILVER_GLOSSARY_PATH,
    SILVER_TEST_AIRPORT_DATA_PATH,
    SILVER_TEST_GEOGRAPHIC_DATA_PATH,
    SILVER_TEST_WEATHER_DATA_PATH,
    SILVER_TRAINING_AIRPORT_DATA_PATH,
    SILVER_WEATHER_DATA_PATH,
)
from src.utils.logger import logger

# Pipeline Steps
# Define your functions here and call them in the bronze_to_silver function.


def read_file(file_path):
    """Read a file based on its extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def convert_object_columns_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns to string to avoid mixed-type issues with parquet."""
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
    return df


def standardize_airport_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize runway and stand column names to lowercase."""
    column_mapping = {}
    for col in df.columns:
        if col.lower() == "runway":
            column_mapping[col] = RUNWAY
        elif col.lower() == "stand":
            column_mapping[col] = STAND
    if column_mapping:
        df = df.rename(columns=column_mapping)
    return df


def transform_runway_stand_values(value: str) -> str:
    """Transform values like 'Runway_0' to 'RUNWAY_1' (uppercase and add 1)."""
    match = re.match(r"([A-Za-z]+)_(\d+)", value)
    if match:
        prefix = match.group(1).upper()
        number = int(match.group(2)) + 1
        return f"{prefix}_{number}"
    return value


def standardize_geographic_values(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize runway and stand values in geographic data."""
    if "runway" in df.columns:
        df["runway"] = df["runway"].apply(transform_runway_stand_values)
    if "stand" in df.columns:
        df["stand"] = df["stand"].apply(transform_runway_stand_values)
    return df


def bronze_to_silver():
    """Pipeline to transform bronze data to silver data.

    This function uses the previously defined functions to read bronze data,
    process it, and write the silver data.

    To add a step to this pipeline, define the function above and call it here.
    """
    # The structure is always the same:

    # 1. Read data from bronze:
    # logger.info(f"Reading XXXX data at {PATH_TO_DATA}")
    # data = pd.read_csv(PATH_TO_DATA)

    # 2. Process data
    # logger.info("Processing XXXX data")
    # processed_data = process_xxxx(data) -> Use your function here

    # 3. Write data to silver
    # logger.info(f"Writing processed data to {PATH_TO_SILVER}")
    # processed_data.to_parquet(PATH_TO_SILVER)

    # Ensure silver directory exists
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Training Airport Data
    # =========================================================================
    logger.info(
        f"Reading training airport data from {BRONZE_TRAINING_AIRPORT_DATA_PATH}"
    )
    training_airport_data = read_file(BRONZE_TRAINING_AIRPORT_DATA_PATH)

    training_airport_data = convert_object_columns_to_string(training_airport_data)
    training_airport_data = standardize_airport_columns(training_airport_data)

    SILVER_TRAINING_AIRPORT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {SILVER_TRAINING_AIRPORT_DATA_PATH}")
    training_airport_data.to_parquet(SILVER_TRAINING_AIRPORT_DATA_PATH, index=False)

    # =========================================================================
    # Test Airport Data
    # =========================================================================
    logger.info(f"Reading test airport data from {BRONZE_TEST_AIRPORT_DATA_PATH}")
    test_airport_data = read_file(BRONZE_TEST_AIRPORT_DATA_PATH)

    test_airport_data = convert_object_columns_to_string(test_airport_data)
    test_airport_data = standardize_airport_columns(test_airport_data)

    SILVER_TEST_AIRPORT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {SILVER_TEST_AIRPORT_DATA_PATH}")
    test_airport_data.to_parquet(SILVER_TEST_AIRPORT_DATA_PATH, index=False)

    # =========================================================================
    # Geographic Data
    # =========================================================================
    logger.info(f"Reading geographic data from {BRONZE_GEOGRAPHIC_DATA_PATH}")
    geographic_data = read_file(BRONZE_GEOGRAPHIC_DATA_PATH)

    geographic_data = convert_object_columns_to_string(geographic_data)
    geographic_data = standardize_geographic_values(geographic_data)

    SILVER_GEOGRAPHIC_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {SILVER_GEOGRAPHIC_DATA_PATH}")
    geographic_data.to_parquet(SILVER_GEOGRAPHIC_DATA_PATH, index=False)

    # =========================================================================
    # Test Geographic Data
    # =========================================================================
    logger.info(f"Reading test geographic data from {BRONZE_TEST_GEOGRAPHIC_DATA_PATH}")
    test_geographic_data = read_file(BRONZE_TEST_GEOGRAPHIC_DATA_PATH)

    test_geographic_data = convert_object_columns_to_string(test_geographic_data)
    test_geographic_data = standardize_geographic_values(test_geographic_data)

    SILVER_TEST_GEOGRAPHIC_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {SILVER_TEST_GEOGRAPHIC_DATA_PATH}")
    test_geographic_data.to_parquet(SILVER_TEST_GEOGRAPHIC_DATA_PATH, index=False)

    # =========================================================================
    # Weather Data
    # =========================================================================
    logger.info(f"Reading weather data from {BRONZE_WEATHER_DATA_PATH}")
    weather_data = read_file(BRONZE_WEATHER_DATA_PATH)

    weather_data = convert_object_columns_to_string(weather_data)

    SILVER_WEATHER_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {SILVER_WEATHER_DATA_PATH}")
    weather_data.to_parquet(SILVER_WEATHER_DATA_PATH, index=False)

    # =========================================================================
    # Test Weather Data
    # =========================================================================
    logger.info(f"Reading test weather data from {BRONZE_TEST_WEATHER_DATA_PATH}")
    test_weather_data = read_file(BRONZE_TEST_WEATHER_DATA_PATH)

    test_weather_data = convert_object_columns_to_string(test_weather_data)

    SILVER_TEST_WEATHER_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {SILVER_TEST_WEATHER_DATA_PATH}")
    test_weather_data.to_parquet(SILVER_TEST_WEATHER_DATA_PATH, index=False)

    # =========================================================================
    # AC Characteristics
    # =========================================================================
    logger.info(f"Reading AC characteristics from {BRONZE_AC_CHAR_PATH}")
    ac_char_data = read_file(BRONZE_AC_CHAR_PATH)

    ac_char_data = convert_object_columns_to_string(ac_char_data)

    SILVER_AC_CHAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {SILVER_AC_CHAR_PATH}")
    ac_char_data.to_parquet(SILVER_AC_CHAR_PATH, index=False)

    # =========================================================================
    # Glossary
    # =========================================================================
    logger.info(f"Reading glossary from {BRONZE_GLOSSARY_PATH}")
    glossary_data = read_file(BRONZE_GLOSSARY_PATH)

    glossary_data = convert_object_columns_to_string(glossary_data)

    SILVER_GLOSSARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {SILVER_GLOSSARY_PATH}")
    glossary_data.to_parquet(SILVER_GLOSSARY_PATH, index=False)


if __name__ == "__main__":
    # Execute the pipeline --> Called using DVC
    logger.info("Starting Bronze to Silver pipeline")
    bronze_to_silver()
    logger.info("Finished Bronze to Silver pipeline")
