# fraud-risk-system

A production-grade machine learning system for real-time fraud risk scoring, featuring model training pipelines, feature engineering, drift monitoring, explainability, and a FastAPI inference API.

## Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Copy and configure environment variables
cp .env.example .env

# Install pre-commit hooks
uv run pre-commit install
```

## Project Structure

```
fraud-risk-system/
├── src/
│   ├── features/        # Feature engineering and transformation
│   ├── training/        # Model training and evaluation
│   ├── monitoring/      # Data drift and model performance monitoring
│   ├── lifecycle/       # Model registration, promotion, and deployment
│   └── explainability/  # SHAP-based model explanations
├── api/                 # FastAPI inference service
├── notebooks/           # Exploratory analysis and prototyping
├── tests/               # Unit and integration tests
├── docker/              # Dockerfiles and compose configs
├── mlflow/              # MLflow server configuration
└── data/
    ├── raw/             # Source data (gitignored)
    └── processed/       # Transformed datasets
```

## Usage

**Start the MLflow tracking server:**
```bash
mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlartifacts
```

**Run training pipeline:**
```bash
uv run python -m src.training.train
```

**Start the inference API:**
```bash
uv run uvicorn api.main:app --reload
```

**Run tests:**
```bash
uv run pytest tests/
```
