"""Phase 3 — model registry using Azure ML's native SDK (azure-ai-ml), not
MLflow's Model Registry feature.

Why: MLflow's Model Registry, when pointed at Azure ML, goes through a
compatibility shim that translates generic MLflow registry calls into Azure
ML's own asset system — and that shim has real gaps, found by testing both:
mlflow.sklearn.log_model() hits a 404 on a newer MLflow API endpoint Azure ML
hasn't implemented; mlflow.create_model_version() rejects any `source` that
isn't Azure ML's own internal azureml://artifacts/... URI format (rejects a
plain Blob Storage path, even though that file genuinely exists and is
downloadable). Azure ML's own SDK talks to the same underlying asset system
directly, with no translation layer — the first-party, fully-supported path
once committing to Azure ML specifically, which is what this project does.

MLflow is still used elsewhere (src/training/train.py) for run-level
metrics/params — that part of the API has been reliable throughout. Only
model *registration* moves to azure-ai-ml.
"""

import os
import shutil
import tempfile

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

MODEL_NAME = "fraud-lightgbm"
STAGE_TAG_KEY = "stage"


def _get_client() -> MLClient:
    return MLClient(
        DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )


def register_candidate(
    model_path: str, encoders_path: str, model_name: str = MODEL_NAME
) -> str:
    """Uploads the model + encoders (the SDK handles the upload itself, no
    separate Blob Storage step needed) and registers them as a new version,
    tagged 'candidate'.

    Never tags 'production' directly — only promote_to_production() (called
    from evaluation logic, after evaluate_candidate() passes) can do that.
    This is the boundary SKILL.md calls "the only gate."
    """
    client = _get_client()
    with tempfile.TemporaryDirectory() as tmp_dir:
        shutil.copy(model_path, tmp_dir)
        shutil.copy(encoders_path, tmp_dir)
        model = Model(
            path=tmp_dir,
            type=AssetTypes.CUSTOM_MODEL,
            name=model_name,
            description="LightGBM fraud model (joblib model + label encoders)",
            tags={STAGE_TAG_KEY: "candidate"},
        )
        registered = client.models.create_or_update(model)
    return registered.version


def promote_to_production(version: str, model_name: str = MODEL_NAME) -> None:
    """The only function that should ever set stage=production.

    Demotes whatever version is currently tagged production to 'archived'
    first, so exactly one version is ever tagged production at a time.
    """
    client = _get_client()
    for v in client.models.list(name=model_name):
        if v.tags.get(STAGE_TAG_KEY) == "production":
            archived = client.models.get(name=model_name, version=v.version)
            archived.tags[STAGE_TAG_KEY] = "archived"
            client.models.create_or_update(archived)

    promoted = client.models.get(name=model_name, version=version)
    promoted.tags[STAGE_TAG_KEY] = "production"
    client.models.create_or_update(promoted)


def get_production_version(model_name: str = MODEL_NAME):
    """Returns the Azure ML Model asset currently tagged production, or None
    if nothing has been promoted yet."""
    client = _get_client()
    for v in client.models.list(name=model_name):
        if v.tags.get(STAGE_TAG_KEY) == "production":
            return client.models.get(name=model_name, version=v.version)
    return None


def download_model(version_asset, download_dir: str) -> None:
    """Downloads a model version's files into download_dir (the SDK creates
    its own subfolder structure inside it — caller should search for the
    expected filenames rather than assume an exact nested path)."""
    client = _get_client()
    client.models.download(
        name=version_asset.name,
        version=version_asset.version,
        download_path=download_dir,
    )
