"""
03_train_evaluate_models.py

Trains and evaluates every model (Naive, Moving Average, Linear Regression,
Gradient Boosting) at all four forecast horizons (15 min, 30 min, 1 hour,
2 hours), adds an 80% prediction interval via quantile regression on the
Gradient Boosting model, and writes a single combined predictions file that
the Streamlit dashboard (see /app) reads directly.

Updated in response to Unified Mentor evaluator feedback (3 Sept 2026,
7.5/10): FEATURE_COLS now includes holiday/long-weekend flags, cyclical time
encodings, and same-time-yesterday/last-week seasonal lags -- see
02_feature_engineering.py for what each one is and why. No other logic in
this file changed; it already picks up whatever is in FEATURE_COLS
automatically at every horizon.

Note on horizon alignment: each row's lag/rolling features are built from
data up to (but not including) that row's own timestamp -- so the "current"
row already represents information from one interval ago relative to the
features. To forecast N * 15 minutes ahead of that reference point, the
target must be shifted by (N - 1), not N. Getting this off by one is an easy
mistake in time-series pipelines (we made it ourselves during development --
see the research paper, Section 6.3) and silently makes every horizon look
15 minutes shorter than it actually is.

Usage:
    python 03_train_evaluate_models.py --input data/ferry_tickets_features.csv --output app/app_predictions.csv

Runtime: roughly 5-8 minutes on a laptop CPU (trains ~12 gradient boosting
models: point + 2 quantile models, at each of 4 horizons).
"""
import argparse
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_holiday", "is_long_weekend",
    "sales_lag_1", "sales_lag_2", "sales_lag_4", "sales_lag_8",
    "redemption_lag_1", "redemption_lag_2", "redemption_lag_4", "redemption_lag_8",
    "sales_lag_96", "sales_lag_672", "redemption_lag_96", "redemption_lag_672",
    "sales_roll_mean_4", "sales_roll_std_4", "sales_roll_max_4", "sales_roll_mean_96",
]
HORIZONS = [("15 min", 1), ("30 min", 2), ("1 hour", 4), ("2 hours", 8)]
TEST_DAYS = 182  # ~6 months held out, time-based split (never shuffle time series data)


def run_horizon(df: pd.DataFrame, label: str, steps: int) -> pd.DataFrame:
    shift_amount = steps - 1  # see module docstring
    d = df.copy()
    d["target"] = d["Sales Count"].shift(-shift_amount) if shift_amount > 0 else d["Sales Count"]
    d = d.dropna(subset=["target"])

    cutoff = d["Timestamp"].max() - pd.Timedelta(days=TEST_DAYS)
    train = d[d["Timestamp"] < cutoff]
    test = d[d["Timestamp"] >= cutoff].copy()

    # Naive & moving average need no training -- they just read existing columns
    test["naive_pred"] = test["sales_lag_1"]
    test["ma_pred"] = test["sales_roll_mean_4"]

    lr = LinearRegression()
    lr.fit(train[FEATURE_COLS], train["target"])
    test["lr_pred"] = np.clip(lr.predict(test[FEATURE_COLS]), 0, None)

    t0 = time.time()
    gbm = GradientBoostingRegressor(n_estimators=60, max_depth=4, learning_rate=0.1, subsample=0.6, random_state=42)
    gbm.fit(train[FEATURE_COLS], train["target"])
    test["gbm_pred"] = np.clip(gbm.predict(test[FEATURE_COLS]), 0, None)

    lo = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=60, max_depth=4,
                                    learning_rate=0.1, subsample=0.6, random_state=42)
    lo.fit(train[FEATURE_COLS], train["target"])
    test["gbm_lower"] = np.clip(lo.predict(test[FEATURE_COLS]), 0, None)

    hi = GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=60, max_depth=4,
                                    learning_rate=0.1, subsample=0.6, random_state=42)
    hi.fit(train[FEATURE_COLS], train["target"])
    test["gbm_upper"] = np.clip(hi.predict(test[FEATURE_COLS]), 0, None)

    test["gbm_lower"] = np.minimum(test["gbm_lower"], test["gbm_pred"])
    test["gbm_upper"] = np.maximum(test["gbm_upper"], test["gbm_pred"])
    test["horizon"] = label

    print(f"\n--- {label} (shift={shift_amount}) trained in {time.time()-t0:.0f}s, {len(test)} test rows ---")
    for name, col in [("Naive", "naive_pred"), ("Moving avg", "ma_pred"),
                       ("Linear reg", "lr_pred"), ("Gradient boosting", "gbm_pred")]:
        mae = mean_absolute_error(test["target"], test[col])
        rmse = mean_squared_error(test["target"], test[col]) ** 0.5
        print(f"  {name:18s} MAE={mae:6.2f}  RMSE={rmse:6.2f}")
    coverage = ((test["target"] >= test["gbm_lower"]) & (test["target"] <= test["gbm_upper"])).mean()
    width = (test["gbm_upper"] - test["gbm_lower"]).mean()
    print(f"  GBM 80% interval coverage: {coverage*100:.1f}%   avg width: {width:.1f}")

    cols = ["Timestamp", "horizon", "target", "naive_pred", "ma_pred", "lr_pred", "gbm_pred", "gbm_lower", "gbm_upper"]
    return test[cols]


def main(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    results = [run_horizon(df, label, steps) for label, steps in HORIZONS]
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"\nSaved combined predictions -> {output_path}  (shape: {combined.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/ferry_tickets_features.csv")
    parser.add_argument("--output", default="app/app_predictions.csv")
    args = parser.parse_args()
    main(args.input, args.output)
