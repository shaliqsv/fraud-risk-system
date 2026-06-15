FROM python:3.11-slim

# LightGBM's compiled booster needs libgomp at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so they're cached unless pyproject/uv.lock change
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

# App code
COPY api/ ./api/

# Model artifacts (v1: baked into the image — see notebooks/07_final_model_training.ipynb)
COPY models/lightgbm_production.pkl models/label_encoders.pkl models/optimal_threshold.json ./models/
COPY data/processed/selected_feature_names.json ./data/processed/selected_feature_names.json

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
