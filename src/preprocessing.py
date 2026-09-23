"""
AquaTrend - src/preprocessing.py

Data preprocessing pipeline for the CGWB groundwater dataset.

The dataset is already a cleaned district-level extract built from the raw
CGWB well-level file (see dataset_source.txt for full details of how it was
built). This module performs the remaining preprocessing that a forecasting
model needs: parsing dates, sorting, removing duplicates/outliers, and
engineering time-series features.

Why each feature is created is explained in comments below.
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/groundwater_data.csv"

# Quarterly readings are taken in Jan, May, Aug, Nov. We map each reading
# month to a season that is meaningful for Indian hydrology.
SEASON_MAP = {
    1: "Winter",       # January reading -> post-winter / pre-summer level
    5: "Summer",       # May reading -> peak summer, lowest recharge
    8: "Monsoon",      # August reading -> during monsoon recharge
    11: "Post-Monsoon" # November reading -> right after monsoon recharge
}


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the district-level groundwater CSV."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning steps:
    - Convert date column to datetime
    - Drop exact duplicate rows
    - Drop rows with missing target values
    - Remove physically impossible outliers (negative depth, or an
      implausibly large depth caused by a data-entry error)
    - Sort chronologically within each location (required before creating
      lag / rolling features, and required for a valid chronological
      train/test split later)
    """
    df = df.copy()

    # Date conversion: the model and dashboard need real datetime objects
    # to sort chronologically and to compute month/year/season features.
    df["date"] = pd.to_datetime(df["date"])

    # Duplicate removal: guards against the same location/date row appearing
    # twice (e.g. if the raw extract was re-run).
    df = df.drop_duplicates(subset=["location", "date"])

    # Missing value handling: target must be present. There are no missing
    # values left in this extract (already handled when the CSV was built
    # from the raw CGWB file), but we keep this defensive step so the
    # pipeline still works if a different/raw CGWB file is dropped in later.
    df = df.dropna(subset=["depth_to_water_level_m"])

    # Outlier handling only when justified: depth to water level cannot be
    # negative, and CGWB manual-measurement errors occasionally produce
    # extreme values. 50 m bgl is used as an upper cutoff because it is far
    # beyond the 99.9th percentile of the raw well-level readings and is
    # documented explicitly rather than silently discarded.
    df = df[(df["depth_to_water_level_m"] >= 0) & (df["depth_to_water_level_m"] <= 50)]

    # Sorting: every lag/rolling feature below depends on rows being in
    # correct time order within each location.
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features.

    - year, month: capture long-term trend and calendar position
    - season: groups quarterly readings into a hydrologically meaningful
      season (Winter / Summer / Monsoon / Post-Monsoon), used for the
      seasonal-analysis section of the dashboard
    - quarter_index: a simple 0,1,2,... counter per location used to fit a
      trend line (linear regression slope) for depletion-trend analysis
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(SEASON_MAP)
    df["quarter_index"] = df.groupby("location").cumcount()
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling-average features.

    The source data is QUARTERLY (not monthly), so the hackathon-suggested
    "Lag 1 / Lag 3 / Lag 6 / Lag 12" (months) are adapted to the actual
    granularity of this dataset:

    - lag_1  : value 1 quarter ago  (~3 months)  -> most recent trend
    - lag_2  : value 2 quarters ago (~6 months)  -> half-year trend
    - lag_4  : value 4 quarters ago (~1 year)    -> same-season last year,
               which matters because groundwater is strongly seasonal
               (e.g. comparing this Monsoon reading to last Monsoon reading)
    - rolling_avg_4 : rolling mean of the last 4 quarters (~1 year) ->
               smooths out short-term noise and captures the general level
               the well has been sitting at recently

    All of these are computed strictly within each location's own history
    (groupby("location")) so no information leaks across locations.
    """
    df = df.copy()
    g = df.groupby("location")["depth_to_water_level_m"]

    df["lag_1"] = g.shift(1)
    df["lag_2"] = g.shift(2)
    df["lag_4"] = g.shift(4)
    df["rolling_avg_4"] = g.shift(1).rolling(4).mean().reset_index(level=0, drop=True)

    return df


def build_feature_table(path: str = RAW_PATH) -> pd.DataFrame:
    """Run the full preprocessing pipeline and return the model-ready table."""
    df = load_raw_data(path)
    df = clean_data(df)
    df = add_time_features(df)
    df = add_lag_and_rolling_features(df)

    # Rows at the very start of each location's history do not yet have a
    # full set of lag features (e.g. lag_4 needs 4 prior quarters). These
    # rows cannot be used for supervised training and are dropped only for
    # the MODELLING table (the raw/clean history is still shown in full on
    # the dashboard's historical charts).
    model_df = df.dropna(subset=["lag_1", "lag_2", "lag_4", "rolling_avg_4"]).reset_index(drop=True)

    return model_df


if __name__ == "__main__":
    features = build_feature_table()
    print("Feature table shape:", features.shape)
    print(features.head())
    features.to_csv("results/feature_table_preview.csv", index=False)
