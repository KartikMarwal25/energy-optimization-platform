import pandas as pd
import numpy as np
import os
import glob
from typing import List, Optional
import logging
import pandas as pd
import numpy as np
from config import settings

logger = logging.getLogger(__name__)


def _detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive match
    cols_lower = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def _read_and_standardize(file):
    def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
        meter_col = _detect_column(df, ["LCLid", "meter_id", "meter", "id"])

        ts_col = _detect_column(
            df,
            ["timestamp", "date", "day", "datetime", "localminute", "reading_timestamp", "tstp"]
        )

        val_col = _detect_column(
            df,
            [
                "energy(kWh/hh)",
                "energy(kwh/hh)",
                "energy_sum",
                "energy_mean",
                "energy_median",
                "consumption",
                "energy",
                "value",
                "usage",
                "kwh",
            ]
        )

        rename_map = {}
        if meter_col:
            rename_map[meter_col] = "meter_id"
        if ts_col:
            rename_map[ts_col] = "timestamp"
        if val_col:
            rename_map[val_col] = "consumption"

        if rename_map:
            df = df.rename(columns=rename_map)

        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "consumption" in df.columns:
            df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")

        return df

    # Try a normal read first, fall back to chunked or python-engine reads on failure
    try:
        df = pd.read_csv(file, low_memory=False, skipinitialspace=True)
        return _downcast_dataframe(_standardize_df(df))
    except (MemoryError, pd.errors.ParserError, OSError) as e:
        logger.warning(f"Primary read failed for {file}: {e}; attempting streaming/chunked read")
        parts = []
        try:
            for chunk in pd.read_csv(file, chunksize=200000, iterator=True, low_memory=True, skipinitialspace=True):
                try:
                    parts.append(_standardize_df(chunk))
                except Exception as se:
                    logger.exception(f"Failed to standardize chunk from {file}: {se}")
            if parts:
                return _downcast_dataframe(pd.concat(parts, ignore_index=True))
        except Exception as ce:
            logger.warning(f"Chunked read failed for {file}: {ce}; trying python engine fallback")
            try:
                df = pd.read_csv(file, engine="python", low_memory=True, skipinitialspace=True)
                return _downcast_dataframe(_standardize_df(df))
            except Exception as fe:
                logger.exception(f"Failed to read {file} with fallback engines: {fe}")
                raise


def _list_block_files(folder: str) -> list[str]:
    return sorted(glob.glob(os.path.join(folder, "**", "block_*.csv"), recursive=True))


def _downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object"]).columns:
        if col == "meter_id":
            df[col] = df[col].astype("category")
        elif df[col].nunique(dropna=False) < 1000:
            df[col] = df[col].astype("category")
    return df


def _estimate_block_folder_size(folder: str) -> int:
    files = _list_block_files(folder)
    return sum(os.path.getsize(f) for f in files if os.path.exists(f))


def _concat_block_folder(folder: str) -> pd.DataFrame:
    files = _list_block_files(folder)
    if not files:
        return pd.DataFrame()
    parts = []
    merged_parts = []
    for f in files:
        try:
            df = _read_and_standardize(f)
            parts.append(df)
        except Exception as e:
            logger.exception(f"Failed to read block file {f}: {e}")
        if len(parts) >= 20:
            merged_parts.append(_downcast_dataframe(pd.concat(parts, ignore_index=True)))
            parts = []
    if parts:
        merged_parts.append(_downcast_dataframe(pd.concat(parts, ignore_index=True)))
    if not merged_parts:
        return pd.DataFrame()
    return _downcast_dataframe(pd.concat(merged_parts, ignore_index=True))


def _has_required_columns(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    if "timestamp" not in df.columns or "meter_id" not in df.columns:
        return False
    if "consumption" in df.columns:
        return True
    return _detect_column(
        df,
        [
            "consumption",
            "energy",
            "energy_sum",
            "energy_mean",
            "energy_median",
            "energy(kWh/hh)",
            "energy(kwh/hh)",
            "value",
            "usage",
            "kwh",
        ],
    ) is not None


def ensure_data_present():
    os.makedirs("data/raw/daily_dataset", exist_ok=True)
    os.makedirs("data/raw/halfhourly_dataset", exist_ok=True)
    os.makedirs("data/raw/hhblock_dataset", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def _load_auxiliary(file: str) -> Optional[pd.DataFrame]:
    if file and os.path.exists(file):
        try:
            return pd.read_csv(file, low_memory=False, skipinitialspace=True)
        except UnicodeDecodeError:
            try:
                return pd.read_csv(file, low_memory=False, skipinitialspace=True, encoding="latin-1")
            except Exception:
                logger.exception(f"Failed to load auxiliary file {file}")
        except Exception:
            logger.exception(f"Failed to load auxiliary file {file}")
    return None


def load_raw_data(force_reload: bool = False) -> pd.DataFrame:
    """Professional loader that aggregates block files and auxiliary tables.

    Behavior:
    - If a single-file raw export exists at `config.settings.DATA_RAW` and force_reload is False,
      it will be loaded directly (backwards compatibility).
    - Otherwise the loader will read all `block_*.csv` files under the raw subfolders,
      concatenate them, and join household, weather, ACORN and bank-holiday information when available.
    - The final processed dataset is saved to `data/processed/processed_energy_data.parquet` and
      also to the path configured by `settings.DATA_PROCESSED`.
    """
    ensure_data_present()

    processed_path = os.path.join("data/processed", "processed_energy_data.parquet")
    # If already processed and not forcing reload, return it
    if os.path.exists(processed_path) and not force_reload:
        try:
            df = pd.read_parquet(processed_path)
            logger.info(f"Loaded processed dataset from {processed_path}")
            return df
        except Exception:
            logger.warning("Failed to load processed parquet; reprocessing from raw files.")

    # Backwards compatibility: single CSV
    if os.path.exists(settings.DATA_RAW) and not force_reload:
        try:
            df = pd.read_csv(settings.DATA_RAW, low_memory=False, skipinitialspace=True, parse_dates=True)
            logger.info(f"Loaded single-file raw dataset from {settings.DATA_RAW}")
        except Exception:
            logger.exception("Failed to read single-file raw dataset; falling back to block loader.")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    # If empty, load block datasets
    if df.empty:
        root = os.path.join("data", "raw")
        block_candidates = [
            ("daily", os.path.join(root, "daily_dataset")),
            ("hhblock", os.path.join(root, "hhblock_dataset")),
            ("halfhourly", os.path.join(root, "halfhourly_dataset")),
        ]
        # choose the smallest valid dataset first to avoid out-of-memory processing
        candidate_sizes = [
            (name, folder, _estimate_block_folder_size(folder))
            for name, folder in block_candidates
        ]
        candidate_sizes = sorted(candidate_sizes, key=lambda x: x[2])

        for name, folder, size in candidate_sizes:
            if size == 0:
                continue
            logger.info(f"Evaluating block dataset {name} ({size} bytes)")
            candidate = _concat_block_folder(folder)
            if _has_required_columns(candidate):
                df = candidate
                logger.info(f"Loaded block dataset from {name}")
                break

        if df.empty:
            # No compatible block files present — try single-file path again and raise clear error if missing
            if os.path.exists(settings.DATA_RAW):
                df = pd.read_csv(settings.DATA_RAW, low_memory=False, skipinitialspace=True, parse_dates=True)
            else:
                raise FileNotFoundError(f"Raw data not found at {settings.DATA_RAW} and no compatible block files present under data/raw/")

    if not df.empty:
        df = _downcast_dataframe(df)

    # Standardize column names
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        # Try to find a timestamp-like column
        ts = _detect_column(df,[
        "timestamp",
        "date",
        "day",
        "datetime",
        "localminute",
        "tstp"
    ])
        if ts:
            df = df.rename(columns={ts: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Ensure meter id
    mid = _detect_column(df, ["LCLid", "meter_id", "meter", "id"])
    if mid and mid != "meter_id":
        df = df.rename(columns={mid: "meter_id"})

    # Ensure consumption numeric
    if "consumption" not in df.columns:
        val = _detect_column(
    df,
    [
        "consumption",
        "energy",
        "energy_sum",
        "energy_mean",
        "energy_median",
        "energy(kWh/hh)",
        "energy(kwh/hh)",
        "value",
        "usage",
        "kwh"
    ]
)
        if val:
            df = df.rename(columns={val: "consumption"})
    if "consumption" in df.columns:
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")

    # Load auxiliary tables
    root = os.path.join("data", "raw")
    households = _load_auxiliary(os.path.join(root, "informations_households.csv"))
    weather_hourly = _load_auxiliary(os.path.join(root, "weather_hourly_darksky.csv"))
    weather_daily = _load_auxiliary(os.path.join(root, "weather_daily_darksky.csv"))
    acorn = _load_auxiliary(os.path.join(root, "acorn_details.csv"))
    bank_holidays = _load_auxiliary(os.path.join(root, "uk_bank_holidays.csv"))

    large_dataset = len(df) > 2_000_000
    if large_dataset:
        logger.info("Large dataset detected; skipping household, hourly weather, and ACORN merges to reduce memory usage.")

    # Merge household information if available
    if not large_dataset and households is not None and "meter_id" in df.columns:
        # try common id names
        hid = _detect_column(households, ["LCLid", "meter_id", "id"])
        if hid and hid != "meter_id":
            households = households.rename(columns={hid: "meter_id"})
        try:
            df = df.merge(households, on="meter_id", how="left", sort=False)
        except Exception:
            logger.exception("Failed to merge household info; continuing without it.")

    # Merge hourly weather on rounded hour timestamp
    if not large_dataset and weather_hourly is not None and "timestamp" in df.columns:
        wh = weather_hourly.copy()
        # detect timestamp col
        wts = _detect_column(wh, ["timestamp", "time", "datetime", "date"]) or "timestamp"
        if wts in wh.columns:
            wh[wts] = pd.to_datetime(wh[wts], errors="coerce")
            wh["ts_hour"] = wh[wts].dt.floor("h")
            df["ts_hour"] = df["timestamp"].dt.floor("h")
            # merge by ts_hour; if weather has station id etc, keep first
            try:
                df = df.merge(wh.drop(columns=[wts]), on="ts_hour", how="left", sort=False)
            except Exception:
                logger.exception("Failed to merge hourly weather")

    # Merge daily weather
    if weather_daily is not None and "timestamp" in df.columns:
        wd = weather_daily.copy()
        wts = _detect_column(wd, ["date", "timestamp", "day"]) or "date"
        if wts in wd.columns:
            wd[wts] = pd.to_datetime(wd[wts], errors="coerce").dt.date
            df["ts_date"] = df["timestamp"].dt.date
            try:
                df = df.merge(wd, left_on="ts_date", right_on=wts, how="left")
            except Exception:
                logger.exception("Failed to merge daily weather")

    # Merge ACORN details if available
    if acorn is not None:
        # try to find acorn key in df or household
        acol = _detect_column(acorn, ["acorn_id", "acorn_category", "acorn"]) or None
        # prefer merge by area code if households had such a column
        area_cols = [c for c in ["LSOA", "lsoa", "area_code"] if c in df.columns]
        if area_cols and acol is not None:
            try:
                df = df.merge(acorn, left_on=area_cols[0], right_on=acol, how="left")
            except Exception:
                logger.exception("Failed to merge ACORN by area code")

    # Bank holidays
    if bank_holidays is not None and "timestamp" in df.columns:
        bh = bank_holidays.copy()
        date_col = _detect_column(bh, ["date", "holiday_date"]) or None
        if date_col and date_col in bh.columns:
            bh[date_col] = pd.to_datetime(bh[date_col], errors="coerce").dt.date
            df["is_holiday"] = df["timestamp"].dt.date.isin(bh[date_col]).astype(int)
        else:
            df["is_holiday"] = 0
    else:
        if "is_holiday" not in df.columns:
            df["is_holiday"] = 0

    # Feature engineering: hour, day, weekday, month, year, season, is_weekend, is_peak
    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.day
        df["weekday"] = df["timestamp"].dt.weekday
        df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
        df["month"] = df["timestamp"].dt.month
        df["year"] = df["timestamp"].dt.year
        df["season"] = ((df["month"] % 12 + 3) // 3)
        df["is_peak"] = df["hour"].isin(range(17, 21)).astype(int)

    # Lags and rolling features per meter
    if "meter_id" in df.columns and "consumption" in df.columns and "timestamp" in df.columns:
        df = df.sort_values(["meter_id", "timestamp"]) 
        df["lag_1"] = df.groupby("meter_id", observed=False)["consumption"].shift(1)
        df["lag_2"] = df.groupby("meter_id", observed=False)["consumption"].shift(2)
        df["lag_3"] = df.groupby("meter_id", observed=False)["consumption"].shift(3)
        df["roll_mean_3"] = df.groupby("meter_id", observed=False)["consumption"].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        df["roll_std_3"] = df.groupby("meter_id", observed=False)["consumption"].rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

    # Final cleanup: fill NA numeric with 0, drop temporary cols
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Save processed
    try:
        df.to_parquet(processed_path, index=False)
        # also save a CSV for compatibility
        try:
            df.to_csv(settings.DATA_PROCESSED, index=False)
        except Exception:
            logger.exception("Failed to write settings.DATA_PROCESSED CSV")
        logger.info(f"Saved processed dataset to {processed_path}")
    except Exception:
        logger.exception("Failed to save processed parquet; continuing without saving")

    return df
