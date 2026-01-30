import numpy as np
import pandas as pd


def load_flights():
    """Return the demo flights dataset embedded in the source.

    The CSV lines are kept as a list to avoid very long literal lines that
    would fail linters. The function normalises column names to match the
    rest of the app and computes `Predicted taxi time (min)` from the
    provided seconds value.
    """
    csv_lines = [
        (
            "Departure Time,Flight Number,Stand,Runway,Predicted Taxi Time (s),"
            "Weather,Traffic,Distance,Aircraft"
        ),
        "2019-12-31 20:30,EJU8071,STAND_97,RUNWAY_2,1044.1,-34.9,-55.9,-0.5,11.7",
        "2019-12-31 20:30,W63310,STAND_34,RUNWAY_3,1108.2,-53.4,41.8,-1.3,-2.5",
        "2019-12-31 20:50,DY1319,STAND_80,RUNWAY_2,1319.1,8.3,166.7,-0.8,21.2",
        "2019-12-31 20:55,DY1337,STAND_21,RUNWAY_2,855.1,8.7,-217.2,-0.3,-59.8",
        "2019-12-31 20:50,TP1335,STAND_166,RUNWAY_4,1280.1,69.0,-57.5,38.5,106.5",
        "2019-12-31 21:05,W61620,STAND_93,RUNWAY_4,1125.1,21.2,-34.9,2.2,12.9",
        "2019-12-31 21:25,EI249,STAND_26,RUNWAY_4,1328.1,45.6,105.7,0.0,53.1",
        "2019-12-31 21:30,DI7505,STAND_77,RUNWAY_3,1114.2,-32.5,36.4,9.8,-23.1",
        "2019-12-31 21:40,VY7829,STAND_127,RUNWAY_2,1088.9,-31.3,3.7,0.7,-7.8",
        "2019-12-31 21:55, W63032,STAND_152,RUNWAY_1,953.0,-11.9,-179.4,1.7,19.0",
    ]

    csv_text = "\n".join(csv_lines)

    from io import StringIO

    df = pd.read_csv(StringIO(csv_text))

    # Normalise column names expected by the rest of the app
    df = df.rename(
        columns={
            "Departure Time": "Departure time",
            "Flight Number": "Flight",
            "Predicted Taxi Time (s)": "Predicted Taxi Time (s)",
        }
    )

    # Clean Flight values (some entries have extra spaces)
    df["Flight"] = df["Flight"].astype(str).str.strip()

    # Parse datetimes
    df["Departure time"] = pd.to_datetime(df["Departure time"])

    # Convert seconds to minutes and round
    df["Predicted taxi time (min)"] = (
        df["Predicted Taxi Time (s)"].astype(float) / 60.0
    ).round(1)

    return df


def add_placeholder_predictions(df, seed=42):
    """Add placeholder predictions only when no predictions exist.

    If `df` already contains `Predicted taxi time (min)`, the function
    returns the dataframe unchanged. Otherwise it synthesises simple
    placeholder values per-runway.
    """
    if "Predicted taxi time (min)" in df.columns:
        return df

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
