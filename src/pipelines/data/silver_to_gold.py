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

    logger.info("Creating runway-stand distances")
    runway_stand_distances = create_runway_stand_distances(geographic_data)
    logger.info("Adding distance columns to training airport data")
    training_airport_data = add_distance_columns(
        training_airport_data, runway_stand_distances
    )
    logger.info("Adding distance columns to test airport data")
    test_airport_data = add_distance_columns(test_airport_data, runway_stand_distances)

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
