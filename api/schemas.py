from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    transaction_id: str | int | None = Field(
        None,
        description="Caller-supplied identifier, logged alongside the prediction "
        "so a fraud label arriving later can be joined back to it. "
        "Not used in scoring.",
    )
    features: dict[str, float | int | str | None] = Field(
        ...,
        description="Mapping of feature name -> value. Must contain all "
        "features the model was trained on (see /metadata for the count and names).",
    )


class PredictionResponse(BaseModel):
    transaction_id: str | int | None
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
