import os
import uuid

import psycopg2
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.model_loader import model_bundle

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "lightgbm_production"
    assert data["n_features"] == len(model_bundle.feature_names)
    assert len(data["feature_names"]) == data["n_features"]


def test_predict_missing_features():
    response = client.post("/predict", json={"features": {"TransactionAmt": 100.0}})
    assert response.status_code == 422


def test_predict_full_request():
    # Build a request with every required feature set to 0 — not a realistic
    # transaction, but enough to confirm the model runs end to end.
    features = dict.fromkeys(model_bundle.feature_names, 0)
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["threshold_used"] == model_bundle.threshold
    assert data["is_fraud"] == (data["fraud_probability"] >= model_bundle.threshold)


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
def test_predict_logs_prediction():
    transaction_id = f"test-{uuid.uuid4()}"
    features = dict.fromkeys(model_bundle.feature_names, 0)

    response = client.post(
        "/predict", json={"transaction_id": transaction_id, "features": features}
    )
    assert response.status_code == 200
    data = response.json()

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT fraud_probability, is_fraud, threshold_used "
                "FROM predictions WHERE transaction_id = %s",
                (transaction_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    assert row is not None, "Expected a logged row for this transaction_id"
    logged_proba, logged_is_fraud, logged_threshold = row
    assert logged_proba == pytest.approx(data["fraud_probability"])
    assert logged_is_fraud == data["is_fraud"]
    assert logged_threshold == data["threshold_used"]
