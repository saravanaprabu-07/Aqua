"""
AquaTrend - src/forecasting.py

Recursive multi-step-ahead forecasting for a chosen location.

Because lag features (lag_1, lag_2, lag_4, rolling_avg_4) depend on
previous VALUES, forecasting more than one quarter ahead requires a
recursive approach:
  1. Predict quarter t+1 using the last known real values.
  2. Feed that prediction back in as if it were the new "lag_1" value.
  3. Predict quarter t+2 using the updated lag features.
  4. Repeat for as many future quarters as requested.

This is a standard, simple approach for tree-based time-series forecasting
and is easy to explain: "the model predicts one step, then treats that
prediction as the newest known value to predict the next step".
"""

import pandas as pd
import numpy as np

from preprocessing import load_raw_data, clean_data, add_time_features, SEASON_MAP
from model import FEATURE_COLS, CATEGORICAL_COLS, TARGET_COL


def get_location_history(location: str) -> pd.DataFrame:
    raw = load_raw_data()
    df = clean_data(raw)
    df = add_time_features(df)
    hist = df[df["location"] == location].sort_values("date").reset_index(drop=True)
    return hist


def _next_quarter_date(d: pd.Timestamp) -> pd.Timestamp:
    """CGWB readings are quarterly at Jan/May/Aug/Nov. Step to the next one."""
    order = [1, 5, 8, 11]
    month = d.month
    year = d.year
    idx = order.index(month)
    if idx == len(order) - 1:
        return pd.Timestamp(year=year + 1, month=order[0], day=1)
    return pd.Timestamp(year=year, month=order[idx + 1], day=1)


def forecast_location(model, feature_names, location: str, n_periods: int = 4) -> pd.DataFrame:
    """
    Recursively forecast `n_periods` future quarters for one location.
    Returns a dataframe with columns: date, depth_to_water_level_m, type
    where type is "Forecast" (historical rows are handled separately by
    the caller / dashboard using get_location_history()).
    """
    hist = get_location_history(location)
    if len(hist) < 5:
        raise ValueError(f"Not enough history for location '{location}' to forecast.")

    values = list(hist["depth_to_water_level_m"].values)
    last_date = hist["date"].max()
    quarter_index_start = hist["quarter_index"].max() + 1
    num_wells = hist["num_wells_reporting"].iloc[-1]
    state = hist["STATE"].iloc[0]

    forecasts = []
    current_date = last_date

    for step in range(n_periods):
        current_date = _next_quarter_date(current_date)
        lag_1 = values[-1]
        lag_2 = values[-2]
        lag_4 = values[-4]
        rolling_avg_4 = np.mean(values[-4:])

        row = {
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_4": lag_4,
            "rolling_avg_4": rolling_avg_4,
            "quarter_index": quarter_index_start + step,
            "year": current_date.year,
            "month": current_date.month,
            "num_wells_reporting": num_wells,
            "location": location,
            "season": SEASON_MAP[current_date.month],
        }

        row_df = pd.DataFrame([row])
        row_encoded = pd.get_dummies(row_df, columns=CATEGORICAL_COLS)

        # Ensure all columns the model was trained on are present (missing
        # one-hot columns -> 0), and in the same order.
        for col in feature_names:
            if col not in row_encoded.columns:
                row_encoded[col] = 0
        row_encoded = row_encoded[feature_names]

        pred = model.predict(row_encoded)[0]
        pred = max(pred, 0.0)  # depth cannot be negative

        forecasts.append({
            "location": location,
            "date": current_date,
            "depth_to_water_level_m": pred,
            "type": "Forecast",
        })
        values.append(pred)

    return pd.DataFrame(forecasts)


def historical_and_forecast(model, feature_names, location: str, n_periods: int = 4) -> pd.DataFrame:
    """Convenience function: returns one combined dataframe with a `type`
    column ('Historical' or 'Forecast') so the dashboard can plot both with
    clearly different styles."""
    hist = get_location_history(location)[["location", "date", "depth_to_water_level_m"]].copy()
    hist["type"] = "Historical"

    fc = forecast_location(model, feature_names, location, n_periods=n_periods)

    combined = pd.concat([hist, fc], ignore_index=True).sort_values("date").reset_index(drop=True)
    return combined


if __name__ == "__main__":
    from model import train_and_evaluate
    model, feature_names, _, metrics = train_and_evaluate()
    raw = load_raw_data()
    df = clean_data(raw)
    sample_location = df["location"].iloc[0]
    result = historical_and_forecast(model, feature_names, sample_location, n_periods=4)
    print(f"Forecast preview for {sample_location}:")
    print(result.tail(8))
