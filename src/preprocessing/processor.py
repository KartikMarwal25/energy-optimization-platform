from typing import Tuple
import pandas as pd
import numpy as np
from src.feature_engineering.features import add_datetime_features, add_lags_and_rolls
from config import settings
import os


def preprocess_pipeline(df: pd.DataFrame, persist: bool = True) -> pd.DataFrame:
    """Clean meter data and create time-series features while preserving kWh values.

    ``consumption`` deliberately remains in its original units.  Forecasts,
    recommendations, and executive reports display that field to users, so
    scaling it would make their values meaningless.
    """
    df = df.copy()

    # Basic datetime handling
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"])
    else:
        # try to infer from first column
        df.columns = [c.lower() for c in df.columns]
        if df.shape[1] >= 2:
            df.rename(columns={df.columns[0]: "meter_id", df.columns[1]: "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Required columns
    if "consumption" not in df.columns and "energy" in df.columns:
        df["consumption"] = df["energy"]

    # Missing values
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
    df["consumption"] = df["consumption"].ffill().bfill().fillna(0)

    # Feature engineering
    df = add_datetime_features(df, ts_col="timestamp")
    df = add_lags_and_rolls(df, value_col="consumption")

    if persist:
        os.makedirs(os.path.dirname(settings.DATA_PROCESSED), exist_ok=True)
        df.to_csv(settings.DATA_PROCESSED, index=False)
    return df
