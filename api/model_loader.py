import json
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data" / "processed"


class ModelBundle:
    """Loads the production model + supporting artifacts once at startup."""

    def __init__(self) -> None:
        self.model = joblib.load(MODEL_DIR / "lightgbm_production.pkl")
        self.encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")

        with open(DATA_DIR / "selected_feature_names.json") as f:
            self.feature_names: list[str] = json.load(f)

        with open(MODEL_DIR / "optimal_threshold.json") as f:
            self.threshold: float = json.load(f)["optimal_threshold"]

    def _build_row(self, features: dict) -> list:
        row = []
        for col in self.feature_names:
            value = features.get(col)
            encoder = self.encoders.get(col)
            if encoder is not None:
                value_str = str(value)
                if value_str not in encoder.classes_:
                    value_str = encoder.classes_[0]
                value = encoder.transform([value_str])[0]
            row.append(value)
        return row

    def predict_proba(self, features: dict) -> float:
        row = self._build_row(features)
        X = pd.DataFrame([row], columns=self.feature_names)
        return float(self.model.predict_proba(X)[:, 1][0])


model_bundle = ModelBundle()
