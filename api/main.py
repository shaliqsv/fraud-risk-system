from fastapi import FastAPI, HTTPException

from api.model_loader import model_bundle
from api.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(title="Fraud Risk API", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    return MetadataResponse(
        model_name="lightgbm_production",
        n_features=len(model_bundle.feature_names),
        threshold=model_bundle.threshold,
        feature_names=model_bundle.feature_names,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    missing = set(model_bundle.feature_names) - set(request.features.keys())
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing {len(missing)} required feature(s), "
            f"e.g. {sorted(missing)[:5]}",
        )

    proba = model_bundle.predict_proba(request.features)
    return PredictionResponse(
        fraud_probability=proba,
        is_fraud=proba >= model_bundle.threshold,
        threshold_used=model_bundle.threshold,
    )
