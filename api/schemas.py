from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, float | int | str | None] = Field(
        ...,
        description="Mapping of feature name -> value. Must contain all "
        "412 features the model was trained on (see /metadata).",
    )


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold_used: float


class HealthResponse(BaseModel):
    status: str


class MetadataResponse(BaseModel):
    model_name: str
    n_features: int
    threshold: float
    feature_names: list[str]
