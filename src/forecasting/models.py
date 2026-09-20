from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from config import settings
from src.database.db import Database
import datetime


class ModelManager:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.models: Dict[str, Any] = {}
        self.results = pd.DataFrame()
        os.makedirs(settings.MODEL_DIR, exist_ok=True)
        self.db = Database(settings.DB_PATH)

    def _prepare(self, target_col: str = "consumption"):
        df = self.df.dropna()
        X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
        y = df[target_col]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def _train_model(self, name: str, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        model_path = os.path.join(settings.MODEL_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
        metadata = {
            "model": name,
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "path": model_path,
            "trained_at": datetime.datetime.utcnow().isoformat()
        }
        self.db.save_model_metadata(metadata)
        return metadata

    def train_and_compare(self, target_col: str = "consumption") -> pd.DataFrame:
        X_train, X_test, y_train, y_test = self._prepare(target_col)
        candidates = {}
        # core candidates
        candidates["LinearRegression"] = LinearRegression()
        candidates["RandomForest"] = RandomForestRegressor(n_estimators=100, random_state=42)

        # optional gradient boosters if available
        try:
            import xgboost as xgb
            candidates["XGBoost"] = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100)
        except Exception:
            pass
        try:
            import lightgbm as lgb
            candidates["LightGBM"] = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        except Exception:
            pass
        try:
            from catboost import CatBoostRegressor
            candidates["CatBoost"] = CatBoostRegressor(verbose=0, random_state=42)
        except Exception:
            pass

        results = []
        for name, model in candidates.items():
            try:
                meta = self._train_model(name, model, X_train, X_test, y_train, y_test)
                results.append(meta)
                self.models[name] = model
            except Exception as e:
                # continue on failure
                continue

        self.results = pd.DataFrame(results)
        return self.results

    def select_best(self) -> str:
        if self.results.empty:
            return ""
        best_row = self.results.sort_values('rmse').iloc[0]
        return best_row['model']

    def forecast_consumer(
        self,
        consumer_id: str,
        horizon_days: int = 30,
        method: str = "linear_trend",
    ) -> pd.DataFrame:
        """Forecast one consumer's daily consumption with a trend or mean baseline."""
        if horizon_days < 1:
            raise ValueError("horizon_days must be at least 1")
        if method not in {"linear_trend", "mean"}:
            raise ValueError("method must be 'linear_trend' or 'mean'")
        df = self.df.copy()
        if 'meter_id' not in df.columns:
            raise ValueError('meter_id column required for consumer forecast')
        dfc = df[df['meter_id'] == consumer_id].dropna(subset=['consumption', 'timestamp']).copy()
        if dfc.empty:
            return pd.DataFrame()
        # daily sum
        dfc['date'] = pd.to_datetime(dfc['timestamp']).dt.date
        daily = dfc.groupby('date')['consumption'].sum().reset_index()
        daily['date'] = pd.to_datetime(daily['date'])
        daily = daily.sort_values('date')
        daily['t'] = (daily['date'] - daily['date'].min()).dt.days

        y = daily['consumption'].to_numpy()
        last_date = daily['date'].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)]
        if method == "mean" or len(y) < 3:
            mean_val = float(y.mean())
            uncertainty = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
            out = pd.DataFrame({'date': future_dates, 'forecast': [mean_val] * horizon_days})
            out['lower'] = out['forecast'] - 1.96 * uncertainty
            out['upper'] = out['forecast'] + 1.96 * uncertainty
            return out

        from sklearn.linear_model import LinearRegression
        X = daily[['t']].to_numpy()
        model = LinearRegression()
        model.fit(X, y)
        preds_train = model.predict(X)
        resid = y - preds_train
        resid_std = float(resid.std(ddof=1)) if len(resid) > 1 else 0.0

        last_t = int(daily['t'].max())
        future_t = [[last_t + i] for i in range(1, horizon_days+1)]
        y_pred = model.predict(future_t)

        df_out = pd.DataFrame({
            'date': future_dates,
            'forecast': y_pred
        })
        # 95% CI using residual std (approximate)
        df_out['lower'] = df_out['forecast'] - 1.96 * resid_std
        df_out['upper'] = df_out['forecast'] + 1.96 * resid_std
        return df_out
