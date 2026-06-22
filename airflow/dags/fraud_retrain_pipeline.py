"""Phase 10 — automates the manual leak-fix-and-redeploy cycle from earlier this
session: feature_engineering -> check_drift -> (retrain -> evaluate -> deploy,
if drifted).

Honesty note: there's no genuinely new incoming labeled data in this project — the
"new" data used throughout is the held-out Kaggle test set (no labels), the same
stand-in 08_monitoring.ipynb uses. In a real deployment, feature_engineering would
pull a fresh extract that includes newly-confirmed fraud labels from the weeks
since the last training run.
"""

import pendulum

from airflow.decorators import dag, task

PROJECT_ROOT = "/opt/airflow/project"
CANDIDATE_DIR = f"{PROJECT_ROOT}/data/airflow_candidate"


@dag(
    dag_id="fraud_retrain_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["fraud-risk", "phase10"],
)
def fraud_retrain_pipeline():
    @task
    def feature_engineering() -> dict:
        """Builds train/val (train-only-fit) and a 'new data' set (production-fit),
        mirroring src/features/build_test_features.py — but writes to CANDIDATE_DIR
        instead of the canonical data/processed/ files.
        """
        import os
        import sys

        sys.path.insert(0, PROJECT_ROOT)
        os.chdir(PROJECT_ROOT)

        import joblib
        import pandas as pd

        from src.features.build_features import (
            fit_pipeline,
            fit_production_target_encodings,
            merge_raw,
            transform,
        )

        print("Loading raw train data...")
        train_transaction = pd.read_csv("data/raw/train_transaction.csv")
        train_identity = pd.read_csv("data/raw/train_identity.csv")
        raw_train = merge_raw(train_transaction, train_identity)

        print("Fitting pipeline (train-only target encoding)...")
        artifacts, train_df, val_df = fit_pipeline(raw_train)

        print("Loading raw 'new' data (held-out test set stands in for new traffic)...")
        test_transaction = pd.read_csv("data/raw/test_transaction.csv")
        test_identity = pd.read_csv("data/raw/test_identity.csv")
        test_identity = test_identity.rename(columns=lambda c: c.replace("-", "_"))
        raw_new = merge_raw(test_transaction, test_identity)

        print("Computing production (full-data-fit) target encodings...")
        production_maps = fit_production_target_encodings(train_df, val_df)
        production_artifacts = {**artifacts, **production_maps}

        print("Scoring 'new' data through the fitted pipeline...")
        new_df = transform(raw_new, production_artifacts)

        os.makedirs(CANDIDATE_DIR, exist_ok=True)
        train_df.to_parquet(f"{CANDIDATE_DIR}/train_features.parquet")
        val_df.to_parquet(f"{CANDIDATE_DIR}/val_features.parquet")
        new_df.to_parquet(f"{CANDIDATE_DIR}/new_features.parquet")
        joblib.dump(
            production_artifacts, f"{CANDIDATE_DIR}/feature_pipeline_artifacts.pkl"
        )
        print(f"Saved candidate features to {CANDIDATE_DIR}")

        return {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "new_rows": len(new_df),
        }

    feature_engineering()


fraud_retrain_pipeline()
