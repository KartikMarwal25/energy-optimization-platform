import joblib
import os
from typing import Any

def save_model(model: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str) -> Any:
    return joblib.load(path)
