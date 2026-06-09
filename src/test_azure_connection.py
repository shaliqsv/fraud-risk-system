import os

from dotenv import load_dotenv

import mlflow

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("fraud-risk-test")

with mlflow.start_run():
    mlflow.log_param("test_param", "day3")
    mlflow.log_metric("test_metric", 0.99)
    print("Successfully logged to Azure ML!")
