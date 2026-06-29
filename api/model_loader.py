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


def _load_from_registry():
    """Returns (model, encoders) for whichever version is tagged 'production'
    in Azure ML's native model registry, or None if the workspace env vars
    aren't set or nothing's been promoted yet — caller falls back to the
    fixed-filename Blob Storage path or local files in that case.

    Uses azure-ai-ml's MLClient directly (not MLflow's Model Registry
    feature) — see src/lifecycle/registry.py's module docstring for why:
    MLflow's registry, against Azure ML, goes through a compatibility shim
    with real gaps (a 404 on a newer API, and a strict source-URI format
    that rejects anything outside Azure ML's own asset system). Azure ML's
    own SDK talks to the same asset system with no shim in between.
    """
    required_env = (
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_RESOURCE_GROUP",
        "AZURE_ML_WORKSPACE",
    )
    if not all(os.getenv(v) for v in required_env):
        return None

    import tempfile
    from pathlib import Path as _Path

    from src.lifecycle.registry import download_model, get_production_version

    production_version = get_production_version()
    if production_version is None:
        return None

    print(f"Loading from Azure ML registry, version {production_version.version}...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        download_model(production_version, tmp_dir)
        model_path = next(_Path(tmp_dir).rglob("lightgbm_production.pkl"))
        encoders_path = next(_Path(tmp_dir).rglob("label_encoders.pkl"))
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
    return model, encoders


class ModelBundle:
    """Loads the production model + supporting artifacts once at startup.

    Tries the Azure ML model registry first (whichever version is tagged
    'production' — see src/lifecycle/registry.py). Falls back to Azure Blob
    Storage if AZURE_STORAGE_CONNECTION_STRING is set, then to local files
    for development.
    """

    def __init__(self) -> None:
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        client = (
            BlobServiceClient.from_connection_string(conn_str) if conn_str else None
        )

        registry_result = _load_from_registry()

        if registry_result is not None:
            self.model, self.encoders = registry_result
        elif client is not None:
            print("Loading model artifacts from Azure Blob Storage...")
            self.model = joblib.load(
                io.BytesIO(_download_blob(client, "lightgbm_production.pkl"))
            )
            self.encoders = joblib.load(
                io.BytesIO(_download_blob(client, "label_encoders.pkl"))
            )
        else:
            print("Loading model artifacts from local files...")
            self.model = joblib.load(MODEL_DIR / "lightgbm_production.pkl")
            self.encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")

        # Threshold and feature list aren't part of the registered model
        # version (they don't change per-version the way the model does) —
        # always read from the fixed-filename blobs/local files, regardless
        # of which path the model itself came from above.
        if client is not None:
            threshold_data = json.loads(
                _download_blob(client, "optimal_threshold.json")
            )
            self.threshold: float = threshold_data["optimal_threshold"]
            self.feature_names: list[str] = json.loads(
                _download_blob(client, "selected_feature_names.json")
            )
        else:
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
