import pandas as pd
import numpy as np
import holidays

def add_datetime_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df["hour"] = df[ts_col].dt.hour
    df["day"] = df[ts_col].dt.day
    df["weekday"] = df[ts_col].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5,6]).astype(int)
    df["month"] = df[ts_col].dt.month
    df["year"] = df[ts_col].dt.year
    # season
    df["season"] = ((df["month"]%12 + 3)//3)

    # holidays (UK)
    try:
        uk_holidays = holidays.UnitedKingdom()
        df["is_holiday"] = df[ts_col].dt.date.apply(lambda d: 1 if d in uk_holidays else 0)
    except Exception:
        df["is_holiday"] = 0

    # peak hour flag
    df["is_peak"] = df["hour"].isin(range(17,21)).astype(int)
    return df

def add_lags_and_rolls(df: pd.DataFrame, value_col: str = "consumption", lags: int = 3) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by="timestamp")
    for lag in range(1, lags+1):
        df[f"lag_{lag}"] = df[value_col].shift(lag)
    df[f"roll_mean_{lags}"] = df[value_col].rolling(window=lags).mean()
    df[f"roll_std_{lags}"] = df[value_col].rolling(window=lags).std().fillna(0)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df
