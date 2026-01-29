"""Column names constants."""

# IDENTIFIERS
RUNWAY = "runway"
STAND = "stand"
FLIGHT_NUMBER = "Flight Number"
AIRCRAFT_MODEL = "Aircraft Model"

# DATETIME COLUMNS
FLIGHT_DATETIME = "Flight Datetime"
AOBT = "AOBT"  # Actual Off-Block Time
ATOT = "ATOT"  # Actual Take-Off Time

# TARGET COLUMN
ACTUAL_TAXI_OUT_SEC = "actual_taxi_out_sec"

# GEOGRAPHIC COLUMNS
LAT_RUNWAY = "Lat_runway"
LNG_RUNWAY = "Lng_runway"
LAT_STAND = "Lat_stand"
LNG_STAND = "Lng_stand"

# DISTANCE COLUMNS
DISTANCE = "distance"
LOG_DISTANCE = "log_distance"
DISTANCE_PROXY_M = "Distance_proxy_m"
LOG_DISTANCE_M = "Log_distance_m"

# AIRCRAFT COLUMNS
AIRCRAFT_LENGTH = "Aircraft Length"
AIRCRAFT_SPAN = "Aircraft Span"
NO_ENGINES = "No. Engines"

# COMPUTED FEATURE COLUMNS
DELAY_SECONDS = "delay_seconds"
PLANES_30MIN = "planes_30min"
PLANES_10MIN = "planes_10min"
PRIVATE_FLIGHT = "private_flight"
HELICOPTER = "helicopter"

# TRAFFIC COLUMNS
N_DEP_DEP = "N_dep_dep"
N_DEP_ARR = "N_dep_arr"
Q_DEP_DEP = "Q_dep_dep"
Q_DEP_ARR = "Q_dep_arr"

# WEATHER COLUMNS
SUMMARY = "summary"
ICON = "icon"
PRECIP_TYPE = "precipType"
PRECIP_INTENSITY = "precipIntensity"
PRECIP_PROBABILITY = "precipProbability"
PRESSURE = "pressure"
WIND_BEARING = "windBearing"
WIND_GUST = "windGust"
APPARENT_TEMPERATURE = "apparentTemperature"
DEW_POINT = "dewPoint"
TIME_HOURLY = "time_hourly"
CLOUD_COVER = "cloudCover"
UV_INDEX = "uvIndex"

# METADATA COLUMNS
AIRPORT_ARRIVAL_DEPARTURE = "Airport Arrival/Departure"
MOVEMENT_TYPE = "Movement Type"

# TEMPORAL COLUMNS
YEAR = "Year"
MONTH = "Month"
WEEKDAY = "Weekday"
HOUR = "Hour"


# DASHBOARD DATASET COLUMNS (display names, different from feature column names)
DASHBOARD_STAND = "Stand"
DASHBOARD_RUNWAY = "Runway"
DASHBOARD_ACTUAL_TAXI_SEC = "Actual Taxi Time (s)"
DASHBOARD_PREDICTED_TAXI_SEC = "Predicted Taxi Time (s)"
DASHBOARD_WEATHER_IMPACT = "Weather"
DASHBOARD_TRAFFIC_IMPACT = "Traffic"
DASHBOARD_DISTANCE_IMPACT = "Distance"
DASHBOARD_AIRCRAFT_IMPACT = "Aircraft"
DASHBOARD_AIRCRAFT_LENGTH = "Aircraft Length (m)"
DASHBOARD_NO_ENGINES = "No. Engines"
