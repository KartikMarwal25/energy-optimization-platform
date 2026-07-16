import pandas as pd
import joblib
import os
from config import settings


class Explainability:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def explain_model(self, model_path: str, nsamples: int = 100) -> dict:
        try:
            import shap
        except Exception as e:
            raise RuntimeError("SHAP is required for explainability. Install shap and retry.") from e

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        model = joblib.load(model_path)
        # build numeric feature matrix from the dataframe
        X_raw = self.df.select_dtypes(include=['number', 'bool']).copy()
        X_raw = X_raw.fillna(0)

        # try to discover expected feature names from the model
        expected_features = None
        try:
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
            # xgboost/LightGBM boosters
            elif hasattr(model, 'get_booster'):
                try:
                    booster = model.get_booster()
                    if hasattr(booster, 'feature_names') and booster.feature_names is not None:
                        expected_features = list(booster.feature_names)
                except Exception:
                    expected_features = None
            elif hasattr(model, 'feature_names'):
                expected_features = list(model.feature_names)
        except Exception:
            expected_features = None

        notes = []
        if expected_features:
            # ensure all expected features are present; add missing as zeros
            missing = [f for f in expected_features if f not in X_raw.columns]
            extra = [c for c in X_raw.columns if c not in expected_features]
            if missing:
                for f in missing:
                    X_raw[f] = 0
                notes.append(f"Added missing features with zeros: {missing}")
            if extra:
                X_raw = X_raw.drop(columns=extra)
                notes.append(f"Dropped unexpected features: {extra}")
            # reorder to expected feature order
            X = X_raw[expected_features]
        else:
            X = X_raw

        # Final safety: ensure DataFrame columns are strings and aligned
        X.columns = [str(c) for c in X.columns]

        # Create SHAP explainer and compute values
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X.iloc[:nsamples])
        return {"shap_values": shap_values, "features": X.columns.tolist(), "notes": notes}
