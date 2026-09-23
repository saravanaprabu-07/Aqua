"""
AquaTrend - src/model.py

Trains a Random Forest Regressor to predict depth-to-water-level one quarter
ahead, using a strictly CHRONOLOGICAL train/test split (never a random
split, because this is time-series data).

Model choice: Random Forest Regressor
- Handles non-linear relationships between lag features and the target
- Robust to the modest amount of noise/measurement variation in manual
  well readings
- Easy to explain to judges: "it grows many decision trees on the lag and
  seasonal features and averages their predictions"
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import build_feature_table

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_4", "rolling_avg_4",
    "quarter_index", "year", "month",
    "num_wells_reporting",
]
CATEGORICAL_COLS = ["location", "season"]
TARGET_COL = "depth_to_water_level_m"


def encode_categoricals(df: pd.DataFrame):
    """One-hot encode location and season so the tree model can use them."""
    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    feature_names = FEATURE_COLS + [c for c in encoded.columns
                                     if c.startswith("location_") or c.startswith("season_")]
    return encoded, feature_names


def chronological_split(df: pd.DataFrame, test_quarters: int = 8):
    """
    Chronological split (NOT random):
    For every location, the most recent `test_quarters` readings become the
    TEST set and everything earlier becomes the TRAIN set. This mimics the
    real forecasting task: predict the future from the past.
    """
    df = df.sort_values(["location", "date"])
    df["rank_from_end"] = df.groupby("location").cumcount(ascending=False)
    train = df[df["rank_from_end"] >= test_quarters].drop(columns=["rank_from_end"])
    test = df[df["rank_from_end"] < test_quarters].drop(columns=["rank_from_end"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def train_and_evaluate(test_quarters: int = 8, random_state: int = 42):
    df = build_feature_table()
    train_df, test_df = chronological_split(df, test_quarters=test_quarters)

    encoded_all, feature_names = encode_categoricals(df)
    encoded_train = encoded_all.loc[train_df.index]
    encoded_test = encoded_all.loc[test_df.index]

    X_train, y_train = encoded_train[feature_names], encoded_train[TARGET_COL]
    X_test, y_test = encoded_test[feature_names], encoded_test[TARGET_COL]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results = pd.DataFrame({
        "location": test_df["location"].values,
        "date": test_df["date"].values,
        "actual_depth_m": y_test.values,
        "predicted_depth_m": preds,
    })

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_quarters_per_location": test_quarters,
    }

    return model, feature_names, results, metrics


if __name__ == "__main__":
    model, feature_names, results, metrics = train_and_evaluate()
    print("Evaluation metrics (chronological test split):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    results.to_csv("results/model_results.csv", index=False)
    print("\nSaved predictions to results/model_results.csv")
