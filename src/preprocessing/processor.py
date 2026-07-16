from typing import Tuple
import pandas as pd
import numpy as np
from src.feature_engineering.features import add_datetime_features, add_lags_and_rolls
from sklearn.preprocessing import StandardScaler
from config import settings
import os


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Complete preprocessing pipeline: cleaning, datetime parsing, feature engineering, scaling."""
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
    df["consumption"] = df["consumption"].ffill().bfill().fillna(0)

    # Feature engineering
    df = add_datetime_features(df, ts_col="timestamp")
    df = add_lags_and_rolls(df, value_col="consumption")

    # Scaling numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])

    os.makedirs(os.path.dirname(settings.DATA_PROCESSED), exist_ok=True)
    df.to_csv(settings.DATA_PROCESSED, index=False)
    return df
