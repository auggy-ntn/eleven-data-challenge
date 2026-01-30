"""Feature categories used in the model (for SHAP explanations)."""

from constants.column_names import (
    AIRCRAFT_LENGTH,
    DELAY_SECONDS,
    DISTANCE,
    LOG_DISTANCE,
    N_DEP_ARR,
    N_DEP_DEP,
    NO_ENGINES,
    PLANES_10MIN,
    PLANES_30MIN,
    Q_DEP_DEP,
    RUNWAY,
    STAND,
)

# Categories of features for SHAP explanations
WEATHER = "Weather"
TRAFFIC = "Traffic"
DISTANCE_CAT = "Distance"
AIRCRAFT = "Aircraft"


# Weather feature patterns (for matching encoded column names like "summary_Clear")
WEATHER_PATTERNS = [
    "temperature",
    "humidity",
    "windSpeed",
    "visibility",
    "precipAccumulation",
    "ozone",
    "summary_",
    "icon_",
    "precipType_",
]

# Traffic features (exact column names)
TRAFFIC_FEATURES = [
    N_DEP_DEP,
    N_DEP_ARR,
    Q_DEP_DEP,
    PLANES_30MIN,
    PLANES_10MIN,
    DELAY_SECONDS,
]

# Distance features (exact column names)
DISTANCE_FEATURES = [STAND, RUNWAY, DISTANCE, LOG_DISTANCE]

# Aircraft features (exact column names)
AIRCRAFT_FEATURES = [AIRCRAFT_LENGTH, NO_ENGINES]


def get_feature_categories(feature_cols: list[str]) -> dict[str, list[str]]:
    """Get feature categories for SHAP explanations.

    Args:
        feature_cols: List of feature column names from the model.

    Returns:
        Dictionary mapping category names to lists of feature column names.
    """
    return {
        f"{WEATHER}": [
            col
            for col in feature_cols
            if any(pattern in col for pattern in WEATHER_PATTERNS)
        ],
        f"{TRAFFIC}": (
            TRAFFIC_FEATURES
            + [col for col in feature_cols if "hour" in col.lower()]
            + [col for col in feature_cols if "day_of_week" in col.lower()]
            + [col for col in feature_cols if "month" in col.lower()]
        ),
        f"{DISTANCE_CAT}": DISTANCE_FEATURES,
        f"{AIRCRAFT}": AIRCRAFT_FEATURES,
    }
