import io
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from azure.storage.blob import BlobServiceClient

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data" / "processed"

BLOB_CONTAINER = "models"


def _download_blob(client: BlobServiceClient, blob_name: str) -> bytes:
    blob = client.get_container_client(BLOB_CONTAINER).get_blob_client(blob_name)
    return blob.download_blob().readall()


class ModelBundle:
    """Loads the production model + supporting artifacts once at startup.

    If AZURE_STORAGE_CONNECTION_STRING is set, downloads artifacts from
    Azure Blob Storage. Otherwise falls back to local files (development).
    """

    def __init__(self) -> None:
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

        if conn_str:
            print("Loading model artifacts from Azure Blob Storage...")
            client = BlobServiceClient.from_connection_string(conn_str)

            self.model = joblib.load(
                io.BytesIO(_download_blob(client, "lightgbm_production.pkl"))
            )
            self.encoders = joblib.load(
                io.BytesIO(_download_blob(client, "label_encoders.pkl"))
            )
            threshold_data = json.loads(
                _download_blob(client, "optimal_threshold.json")
            )
            self.threshold: float = threshold_data["optimal_threshold"]

            self.feature_names: list[str] = json.loads(
                _download_blob(client, "selected_feature_names.json")
            )
        else:
            print("Loading model artifacts from local files...")
            self.model = joblib.load(MODEL_DIR / "lightgbm_production.pkl")
            self.encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")

            with open(MODEL_DIR / "optimal_threshold.json") as f:
                self.threshold = json.load(f)["optimal_threshold"]

            with open(DATA_DIR / "selected_feature_names.json") as f:
                self.feature_names = json.load(f)

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
