"""
02_feature_engineering.py

Turns the cleaned 15-minute ticket time series into a supervised-learning
table: lag features, rolling statistics, and temporal encodings -- all built
using only information available strictly BEFORE the row being predicted, to
avoid data leakage.

Updated in response to Unified Mentor evaluator feedback (3 Sept 2026,
7.5/10): added holiday/long-weekend flags, cyclical time encodings, and
same-time-yesterday / same-time-last-week seasonal lags. See the module
docstring in 03_train_evaluate_models.py and the research paper for why the
seasonal lags in particular target the evaluator's "longer horizons need
better accuracy" point directly: at a 2-hour horizon, the most recent 15/30/60
-minute lags are much weaker signals than they are at a 15-minute horizon,
but "what happened here exactly 24 hours / 7 days ago" stays equally
informative regardless of how far ahead you're forecasting.

Weather and special-event data are NOT included here: this project has no
access to a historical weather feed or an events calendar for Toronto Island
Park matched to this timestamp range, and fabricating either would be worse
than omitting them. See the research paper / README for this noted as
follow-up work, consistent with how ARIMA/SARIMA/Prophet were already
documented as out-of-scope for this development environment.

Usage:
    python 02_feature_engineering.py --input data/ferry_tickets_cleaned.csv --output data/ferry_tickets_features.csv
"""
import argparse
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import DateOffset, Easter, Day
from dateutil.relativedelta import MO
import numpy as np


class CanadaOntarioHolidays(AbstractHolidayCalendar):
    """Federal + Ontario statutory holidays likely to drive park/ferry tourism.

    Computed algorithmically (not a hand-typed date list) so it's correct for
    any year in the dataset's 2015-2025 range without manually checking moving
    holidays like Easter, Family Day, or Labour Day each year.
    """
    rules = [
        Holiday("New Year's Day", month=1, day=1),
        Holiday("Family Day", month=2, day=1, offset=DateOffset(weekday=MO(3))),  # 3rd Monday of Feb
        Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)]),
        Holiday("Victoria Day", month=5, day=24, offset=DateOffset(weekday=MO(-1))),  # Monday preceding May 25
        Holiday("Canada Day", month=7, day=1),
        Holiday("Civic Holiday", month=8, day=1, offset=DateOffset(weekday=MO(1))),  # 1st Monday of Aug
        Holiday("Labour Day", month=9, day=1, offset=DateOffset(weekday=MO(1))),  # 1st Monday of Sep
        Holiday("Thanksgiving", month=10, day=1, offset=DateOffset(weekday=MO(2))),  # 2nd Monday of Oct
        Holiday("Christmas Day", month=12, day=25),
        Holiday("Boxing Day", month=12, day=26),
    ]


def build_features(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # --- Temporal encodings ---
    df["hour"] = df["Timestamp"].dt.hour
    df["day_of_week"] = df["Timestamp"].dt.dayofweek  # 0 = Monday
    df["month"] = df["Timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # --- Cyclical encodings ---
    # Raw integers treat e.g. hour 23 and hour 0 as far apart, and month 12 /
    # month 1 as far apart, when they're actually adjacent. Sin/cos encoding
    # fixes this, which matters most for Linear Regression (tree models can
    # partially work around it via splits, but this removes the need to).
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Holiday / long-weekend flags ---
    cal = CanadaOntarioHolidays()
    holiday_dates = set(cal.holidays(df["Timestamp"].min(), df["Timestamp"].max()).date)
    df["is_holiday"] = df["Timestamp"].dt.date.isin(holiday_dates).astype(int)
    # Long weekend: the holiday-Monday itself, or the Fri/Sat/Sun immediately
    # before a Monday holiday -- this is the actual tourism-driving window for
    # an island park, not just the calendar holiday date in isolation.
    monday_holidays = {d for d in holiday_dates if pd.Timestamp(d).dayofweek == 0}
    long_weekend_dates = set(holiday_dates)
    for d in monday_holidays:
        ts = pd.Timestamp(d)
        long_weekend_dates.update({(ts - pd.Timedelta(days=i)).date() for i in (1, 2, 3)})
    df["is_long_weekend"] = df["Timestamp"].dt.date.isin(long_weekend_dates).astype(int)

    # --- Lag features: 15 / 30 / 60 / 120 minutes prior (unchanged) ---
    for lag in [1, 2, 4, 8]:
        df[f"sales_lag_{lag}"] = df["Sales Count"].shift(lag)
        df[f"redemption_lag_{lag}"] = df["Redemption Count"].shift(lag)

    # --- NEW: seasonal lags -- same time yesterday / same time last week ---
    # These target the evaluator's longer-horizon-accuracy note directly: at a
    # 2-hour horizon the 15-120 minute lags above are much weaker signals, but
    # "what happened at this exact time yesterday/last week" is not weakened
    # by forecast horizon at all.
    df["sales_lag_96"] = df["Sales Count"].shift(96)    # 24 hours (96 x 15-min)
    df["sales_lag_672"] = df["Sales Count"].shift(672)  # 7 days
    df["redemption_lag_96"] = df["Redemption Count"].shift(96)
    df["redemption_lag_672"] = df["Redemption Count"].shift(672)

    # --- Rolling stats over the trailing hour, using ONLY past values ---
    # shift(1) first so the current row's own value is never included.
    df["sales_roll_mean_4"] = df["Sales Count"].shift(1).rolling(4).mean()
    df["sales_roll_std_4"] = df["Sales Count"].shift(1).rolling(4).std()
    df["sales_roll_max_4"] = df["Sales Count"].shift(1).rolling(4).max()
    # --- NEW: trailing 24h mean -- a smoothed daily baseline ---
    df["sales_roll_mean_96"] = df["Sales Count"].shift(1).rolling(96).mean()

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Rows before dropping early no-history rows: {before}")
    print(f"Rows after: {len(df)}  (dropped {before - len(df)}, "
          f"{(before - len(df)) / before * 100:.2f}% -- the new 7-day lag needs a full week of history)")
    print(f"Holidays found in range: {len(holiday_dates)}  |  long-weekend days: {len(long_weekend_dates)}")

    df.to_csv(output_path, index=False)
    print(f"Saved feature table -> {output_path}  (shape: {df.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/ferry_tickets_cleaned.csv")
    parser.add_argument("--output", default="data/ferry_tickets_features.csv")
    args = parser.parse_args()
    build_features(args.input, args.output)
