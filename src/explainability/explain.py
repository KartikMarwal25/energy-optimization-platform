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
        X = self.df.select_dtypes(include=['number']).fillna(0)
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X.iloc[:nsamples])
        return {"shap_values": shap_values, "features": X.columns.tolist()}
