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
    """Add chronological lag and rolling features without mixing meter histories."""
    df = df.copy()
    if value_col not in df.columns:
        raise ValueError(f"{value_col!r} column is required for lag features")

    sort_columns = ["timestamp"]
    group_columns = []
    if "meter_id" in df.columns:
        sort_columns.insert(0, "meter_id")
        group_columns = ["meter_id"]
    df = df.sort_values(by=sort_columns)

    values = (
        df.groupby(group_columns, observed=False)[value_col]
        if group_columns
        else df[value_col]
    )
    for lag in range(1, lags+1):
        df[f"lag_{lag}"] = values.shift(lag)

    if group_columns:
        df[f"roll_mean_{lags}"] = values.transform(
            lambda series: series.rolling(window=lags, min_periods=1).mean()
        )
        df[f"roll_std_{lags}"] = values.transform(
            lambda series: series.rolling(window=lags, min_periods=1).std().fillna(0)
        )
    else:
        df[f"roll_mean_{lags}"] = values.rolling(window=lags, min_periods=1).mean()
        df[f"roll_std_{lags}"] = values.rolling(window=lags, min_periods=1).std().fillna(0)

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df
