"""Path configurations for the project.

This module centralizes all path definitions to ensure consistency across the project.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# Bronze paths
BRONZE_TRAINING_AIRPORT_DATA_PATH = (
    BRONZE_DIR / "0. Airport data" / "training_set_airport_data.csv"
)
BRONZE_TEST_AIRPORT_DATA_PATH = (
    BRONZE_DIR / "3. Test set" / "test_set_airport_data.xlsx"
)
BRONZE_GEOGRAPHIC_DATA_PATH = BRONZE_DIR / "0. Airport data" / "geographic_data.csv"
BRONZE_WEATHER_DATA_PATH = (
    BRONZE_DIR / "2. Weather data" / "training_set_weather_data.csv"
)
BRONZE_TEST_WEATHER_DATA_PATH = (
    BRONZE_DIR / "3. Test set" / "test_set_weather_data.xlsx"
)
BRONZE_AC_CHAR_PATH = BRONZE_DIR / "1. AC characteristics" / "ACchar.xlsx"
BRONZE_GLOSSARY_PATH = BRONZE_DIR / "Glossary.xlsx"
BRONZE_TEST_GEOGRAPHIC_DATA_PATH = BRONZE_DIR / "3. Test set" / "geographic_data.csv"

# Silver paths
SILVER_TRAINING_AIRPORT_DATA_PATH = (
    SILVER_DIR / "0. Airport data" / "training_set_airport_data.parquet"
)
SILVER_TEST_AIRPORT_DATA_PATH = (
    SILVER_DIR / "3. Test set" / "test_set_airport_data.parquet"
)
SILVER_GEOGRAPHIC_DATA_PATH = SILVER_DIR / "0. Airport data" / "geographic_data.parquet"
SILVER_WEATHER_DATA_PATH = (
    SILVER_DIR / "2. Weather data" / "training_set_weather_data.parquet"
)
SILVER_TEST_WEATHER_DATA_PATH = (
    SILVER_DIR / "3. Test set" / "test_set_weather_data.parquet"
)
SILVER_AC_CHAR_PATH = SILVER_DIR / "1. AC characteristics" / "ACchar.parquet"
SILVER_GLOSSARY_PATH = SILVER_DIR / "Glossary.parquet"
SILVER_TEST_GEOGRAPHIC_DATA_PATH = (
    SILVER_DIR / "3. Test set" / "geographic_data.parquet"
)

# Gold paths
GOLD_TRAINING_AIRPORT_DATA_PATH = GOLD_DIR / "training_set_airport_data.parquet"
GOLD_TEST_AIRPORT_DATA_PATH = GOLD_DIR / "test_set_airport_data.parquet"

# Configuration directory
CONFIG_DIR = PROJECT_ROOT / "config"

# DVC params file
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
