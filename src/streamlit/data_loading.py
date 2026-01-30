import numpy as np
import pandas as pd


def load_flights():
    """Return a small example dataframe of flights used for the demo."""
    data = [
        ("2019-12-31 20:30", "EJU8071", "A319", "STAND_9", "RUNWAY_1"),
        ("2019-12-31 20:30", "W63310", "A320-100/200", "STAND_103", "RUNWAY_4"),
        ("2019-12-31 20:50", "DY1319", "B737-800 WINGLETS", "STAND_122", "RUNWAY_4"),
        ("2019-12-31 20:50", "DY1337", "B737-800 WINGLETS", "STAND_29", "RUNWAY_1"),
        ("2019-12-31 20:50", "TP1335", "A319", "STAND_32", "RUNWAY_4"),
        ("2019-12-31 21:05", "W61620", "A321-100/200", "STAND_25", "RUNWAY_2"),
        ("2019-12-31 21:25", "EI249", "A320-100/200", "STAND_154", "RUNWAY_3"),
        ("2019-12-31 21:30", "DI7505", "B787-900", "STAND_142", "RUNWAY_2"),
        ("2019-12-31 21:40", "VY7829", "A320 NEO", "STAND_145", "RUNWAY_1"),
        ("2019-12-31 21:55", "W63032", "A321-100/200", "STAND_164", "RUNWAY_3"),
    ]

    df = pd.DataFrame(
        data,
        columns=[
            "Departure time",
            "Flight",
            "Aircraft",
            "Stand",
            "Runway",
        ],
    )

    df["Departure time"] = pd.to_datetime(df["Departure time"])

    return df


def add_placeholder_predictions(df, seed=42):
    """Add a `Predicted taxi time (min)` column with simple placeholder values."""
    np.random.seed(seed)

    base_times = {
        "RUNWAY_1": 14,
        "RUNWAY_2": 17,
        "RUNWAY_3": 19,
        "RUNWAY_4": 21,
    }

    df = df.copy()

    df["Predicted taxi time (min)"] = (
        df["Runway"]
        .apply(lambda r: base_times.get(r, 18) + np.random.normal(0, 1.5))
        .round(1)
    )

    return df
