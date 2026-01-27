"""Silver to Gold Pipeline."""

import numpy as np
import pandas as pd

from constants.paths import (
    GOLD_DIR,
    GOLD_TEST_AIRPORT_DATA_PATH,
    GOLD_TRAINING_AIRPORT_DATA_PATH,
    SILVER_GEOGRAPHIC_DATA_PATH,
    SILVER_TEST_AIRPORT_DATA_PATH,
    SILVER_TRAINING_AIRPORT_DATA_PATH,
    SILVER_WEATHER_DATA_PATH,
)
from src.utils.logger import logger

# Pipeline Steps
# Define your functions here and call them in the silver_to_gold function.


def haversine_distance(lat1, lng1, lat2, lng2):
    """Calculate the Haversine distance between two points in meters."""
    R = 6371000  # Earth's radius in meters

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lng = np.radians(lng2 - lng1)

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lng / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def create_runway_stand_distances(geographic_df: pd.DataFrame) -> pd.DataFrame:
    """Create a new dataset with runway-stand distances.

    Parameters
    ----------
    geographic_df: pd.DataFrame
        DataFrame containing geographic data.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing runway, stand, and distance between them.
    """
    # Create dataframe with distinct runways and their coordinates
    runways_df = geographic_df[["runway", "Lat_runway", "Lng_runway"]].drop_duplicates()
    runways_df = runways_df.rename(columns={"Lat_runway": "lat", "Lng_runway": "lng"})

    # Create dataframe with distinct stands and their coordinates
    stands_df = geographic_df[["stand", "Lat_stand", "Lng_stand"]].drop_duplicates()
    stands_df = stands_df.rename(columns={"Lat_stand": "lat", "Lng_stand": "lng"})

    # Cycle through all combinations to calculate distances
    distance_records = []
    for _, runway in runways_df.iterrows():
        for _, stand in stands_df.iterrows():
            distance = haversine_distance(
                runway["lat"], runway["lng"], stand["lat"], stand["lng"]
            )
            distance_records.append(
                {
                    "runway": runway["runway"],
                    "stand": stand["stand"],
                    "distance": distance,
                    "log_distance": np.log(distance),
                }
            )

    runway_stand_distances_df = pd.DataFrame(distance_records)

    return runway_stand_distances_df


def add_distance_columns(
    airport_data: pd.DataFrame, runway_stand_distances: pd.DataFrame
) -> pd.DataFrame:
    """Add distance and log_distance columns to airport data.

    Parameters
    ----------
    airport_data: pd.DataFrame
        DataFrame containing airport data with 'runway' and 'stand' columns.
    runway_stand_distances: pd.DataFrame
        DataFrame containing runway-stand distances.

    Returns:
    -------
    pd.DataFrame
        Airport data with distance and log_distance columns added.
    """
    return airport_data.merge(
        runway_stand_distances[["runway", "stand", "distance", "log_distance"]],
        on=["runway", "stand"],
        how="left",
    )


def clean_weather_data(weather_data: pd.DataFrame) -> pd.DataFrame:
    """Clean weather data by dropping unnecessary columns."""
    columns_to_drop = [
        "Flight Number",
        "Aircraft Span",
        "Airport Arrival/Departure",
        "Movement Type",
        "Distance_proxy_m",
        "Log_distance_m",
        "Year",
        "Month",
        "Weekday",
        "Hour",
        "Q_dep_arr",
        "time_hourly",
        "apparentTemperature",
        "dewPoint",
        "windGust",
    ]
    # Only drop columns that exist in the dataframe
    columns_to_drop = [col for col in columns_to_drop if col in weather_data.columns]
    return weather_data.drop(columns=columns_to_drop)


def merge_airport_weather_data(
    airport_data: pd.DataFrame,
    weather_data: pd.DataFrame,
) -> pd.DataFrame:
    """Merge airport data with weather data.

    Parameters
    ----------
    airport_data: pd.DataFrame
        DataFrame containing airport data.
    weather_data: pd.DataFrame
        DataFrame containing weather data.

    Returns:
    -------
    pd.DataFrame
        Merged DataFrame with airport and weather data.
    """
    merge_columns = ["Flight Datetime", "Aircraft Model", "AOBT", "ATOT"]

    # Ensure merge columns have consistent types (convert to string)
    airport_data = airport_data.copy()
    weather_data = weather_data.copy()
    for col in merge_columns:
        if col in airport_data.columns:
            airport_data[col] = airport_data[col].astype(str)
        if col in weather_data.columns:
            weather_data[col] = weather_data[col].astype(str)

    return airport_data.merge(weather_data, on=merge_columns, how="left")


def cyclical_encode(df: pd.DataFrame, column: str, max_value: int) -> pd.DataFrame:
    """Encode a column as cyclical features using sin/cos transformation.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the column to encode.
    column: str
        Name of the column to encode.
    max_value: int
        Maximum value for the cycle (e.g., 24 for hours, 12 for months).

    Returns:
    -------
    pd.DataFrame
        DataFrame with new sin/cos columns added.
    """
    df[f"{column}_sin"] = np.sin(2 * np.pi * df[column] / max_value)
    df[f"{column}_cos"] = np.cos(2 * np.pi * df[column] / max_value)
    return df


def encode_features(
    df: pd.DataFrame,
    datetime_columns: list[str],
    categorical_columns: list[str],
    category_mappings: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Encode features with cyclical encoding for datetime and categorical encoding.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to encode.
    datetime_columns: list[str]
        Columns containing datetime data to cyclically encode.
    categorical_columns: list[str]
        Columns to categorically encode (label encoding).
    category_mappings: dict | None
        Pre-defined category mappings for consistent encoding between train/test.
        If None, categories are learned from the data.

    Returns:
    -------
    tuple[pd.DataFrame, dict]
        Encoded DataFrame and category mappings dict.
    """
    df = df.copy()

    if category_mappings is None:
        category_mappings = {}

    # Cyclical encoding for datetime columns
    for col in datetime_columns:
        if col in df.columns:
            # Parse datetime
            dt = pd.to_datetime(df[col], format="%m/%d/%Y %H:%M", errors="coerce")

            # Create unique column names for this datetime column
            hour_col = f"{col}_hour"
            dow_col = f"{col}_day_of_week"
            month_col = f"{col}_month"
            dom_col = f"{col}_day_of_month"

            # Extract components and encode cyclically
            df[hour_col] = dt.dt.hour
            df = cyclical_encode(df, hour_col, 24)

            df[dow_col] = dt.dt.dayofweek
            df = cyclical_encode(df, dow_col, 7)

            df[month_col] = dt.dt.month
            df = cyclical_encode(df, month_col, 12)

            df[dom_col] = dt.dt.day
            df = cyclical_encode(df, dom_col, 31)

            # Drop intermediate columns and original datetime column
            df = df.drop(columns=[hour_col, dow_col, month_col, dom_col, col])

    # Categorical encoding (label encoding)
    for col in categorical_columns:
        if col in df.columns:
            if col in category_mappings:
                # Use pre-defined categories
                cat_type = pd.CategoricalDtype(categories=category_mappings[col])
                df[col] = df[col].astype(cat_type).cat.codes
            else:
                # Learn categories from data
                cat_col = df[col].astype("category")
                category_mappings[col] = list(cat_col.cat.categories)
                df[col] = cat_col.cat.codes

    # Drop ATOT column
    if "ATOT" in df.columns:
        df = df.drop(columns=["ATOT"])

    return df, category_mappings


def silver_to_gold():
    """Pipeline to transform silver data to gold data.

    This function uses the previously defined functions to read silver data,
    process it, and write the gold data.

    To add a step to this pipeline, define the function above and call it here.
    """
    # The structure is always the same:

    # 1. Read data from silver:
    # logger.info(f"Reading XXXX data at {PATH_TO_DATA}")
    # data = pd.read_csv(PATH_TO_DATA)

    # 2. Process data
    # logger.info("Processing XXXX data")
    # processed_data = process_xxxx(data) -> Use your function here

    # 3. Write data to gold
    # logger.info(f"Writing processed data to {PATH_TO_GOLD}")
    # processed_data.to_parquet(PATH_TO_GOLD)

    logger.info(
        f"Reading training airport data from {SILVER_TRAINING_AIRPORT_DATA_PATH}"
    )
    training_airport_data = pd.read_parquet(SILVER_TRAINING_AIRPORT_DATA_PATH)
    logger.info(f"Reading test airport data from {SILVER_TEST_AIRPORT_DATA_PATH}")
    test_airport_data = pd.read_parquet(SILVER_TEST_AIRPORT_DATA_PATH)
    logger.info(f"Reading geographic data from {SILVER_GEOGRAPHIC_DATA_PATH}")
    geographic_data = pd.read_parquet(SILVER_GEOGRAPHIC_DATA_PATH)
    logger.info(f"Reading weather data from {SILVER_WEATHER_DATA_PATH}")
    training_weather_data = pd.read_parquet(SILVER_WEATHER_DATA_PATH)

    logger.info("Creating runway-stand distances")
    runway_stand_distances = create_runway_stand_distances(geographic_data)
    logger.info("Adding distance columns to training airport data")
    training_airport_data = add_distance_columns(
        training_airport_data, runway_stand_distances
    )
    logger.info("Adding distance columns to test airport data")
    test_airport_data = add_distance_columns(test_airport_data, runway_stand_distances)
    logger.info("Cleaning weather data")
    training_weather_data = clean_weather_data(training_weather_data)
    logger.info("Merging airport and weather data")
    training_airport_data = merge_airport_weather_data(
        training_airport_data, training_weather_data
    )
    test_airport_data = merge_airport_weather_data(
        test_airport_data, training_weather_data
    )
    logger.info("Encoding features (training - learning categories)")
    training_airport_data, category_mappings = encode_features(
        training_airport_data,
        ["Flight Datetime", "AOBT"],
        ["summary", "icon", "precipType"],
    )
    logger.info("Encoding features (test - using training categories)")
    test_airport_data, _ = encode_features(
        test_airport_data,
        ["Flight Datetime", "AOBT"],
        ["summary", "icon", "precipType"],
        category_mappings=category_mappings,
    )

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing training airport data to {GOLD_TRAINING_AIRPORT_DATA_PATH}")
    training_airport_data.to_parquet(GOLD_TRAINING_AIRPORT_DATA_PATH, index=False)
    logger.info(f"Writing test airport data to {GOLD_TEST_AIRPORT_DATA_PATH}")
    test_airport_data.to_parquet(GOLD_TEST_AIRPORT_DATA_PATH, index=False)


if __name__ == "__main__":
    # Execute the pipeline --> Called using DVC
    logger.info("Starting Silver to Gold pipeline")
    silver_to_gold()
    logger.info("Finished Silver to Gold pipeline")
