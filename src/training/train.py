"""Phase 2 — extracted from notebooks/07_final_model_training.ipynb.

Retrains the production model on a given train/val data window. This is the
function training_dag (Phase 5) would call, weekly or on-demand — not the
Optuna search or the baseline model comparison (notebooks 03/04), which were
one-time decisions, not something to redo on every retrain cycle.
"""

import json
import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier, log_evaluation
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder

import mlflow
from src.features.build_features import fit_production_target_encodings
from src.lifecycle.registry import register_candidate

# best_iteration_ from lightgbm_final.pkl's early-stopped run (notebooks/
# 04_hyperparameter_tuning.ipynb, Chapter 4). Not derived dynamically here —
# redoing early stopping on every retrain would need a held-out split, which
# defeats the point of training on the full combined window.
DEFAULT_N_ESTIMATORS = 569


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    selected_features: list[str],
    best_params: dict,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> dict:
    """Retrain on train_df + val_df combined — mirrors notebook 07.

    Refits email/card4_6 target encodings on the full combined window (no
    held-out split left to leak into once val joins training), label-encodes
    the remaining object columns, then trains LightGBM for a fixed
    n_estimators — no early stopping, since there's nothing left to validate
    against.

    Returns a dict with the trained model, fitted label encoders, and a few
    metrics worth logging (not a true validation score — see train_auc_pr's
    docstring note below).
    """
    full_df = (
        pd.concat([train_df, val_df], axis=0)
        .sort_values("TransactionDT")
        .reset_index(drop=True)
    )

    production_maps = fit_production_target_encodings(train_df, val_df)
    full_df["email_target_encoded"] = full_df["P_emaildomain"].map(
        production_maps["email_fraud_map"]
    )
    full_df["card4_6_target_encoded"] = full_df["card4_6"].map(
        production_maps["card_fraud_map"]
    )

    X_full = full_df[selected_features].copy()
    y_full = full_df["isFraud"]

    obj_cols = X_full.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in obj_cols:
        le = LabelEncoder()
        X_full[col] = le.fit_transform(X_full[col].astype(str))
        encoders[col] = le

    params = {
        **best_params,
        "n_estimators": n_estimators,
        "is_unbalance": True,
        "metric": "average_precision",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = LGBMClassifier(**params)
    model.fit(X_full, y_full, callbacks=[log_evaluation(100)])

    # In-sample AUC-PR — NOT a true validation metric, since there's no
    # held-out data left once val joins training. Logged only as a sanity
    # check that the model fit something sensible; it will read optimistic.
    # Real performance estimates: lightgbm_final.pkl's validated AUC-PR
    # (from the early-stopped run this reuses hyperparameters from), or
    # src/lifecycle/evaluate.py once fresh labeled holdout data exists
    # (Phase 4 — label join).
    train_auc_pr = average_precision_score(y_full, model.predict_proba(X_full)[:, 1])

    return {
        "model": model,
        "encoders": encoders,
        "n_rows": len(full_df),
        "fraud_rate": float(y_full.mean()),
        "trees_built": model.n_estimators_,
        "train_auc_pr": float(train_auc_pr),
        "params": params,
    }


def main() -> None:
    """Runnable entry point: uv run python -m src.training.train

    Uses the same default file paths notebook 07 used (current train/val
    parquet, the saved Optuna study, the saved feature list). train_model()
    above takes plain DataFrames, so it can be called with any data window —
    this main() is just the "default window = whatever's on disk right now"
    convenience wrapper.
    """
    load_dotenv()
    os.chdir(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    train_df = pd.read_parquet("data/processed/train_features.parquet")
    val_df = pd.read_parquet("data/processed/val_features.parquet")
    with open("data/processed/selected_feature_names.json") as f:
        selected_features = json.load(f)
    study = joblib.load("models/optuna_study_lgbm.pkl")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="production_retrain"):
        result = train_model(
            train_df=train_df,
            val_df=val_df,
            selected_features=selected_features,
            best_params=study.best_params,
        )

        mlflow.log_params(result["params"])
        mlflow.log_metric("train_auc_pr", result["train_auc_pr"])
        mlflow.log_metric("n_rows", result["n_rows"])
        mlflow.log_metric("fraud_rate", result["fraud_rate"])
        mlflow.log_metric("trees_built", result["trees_built"])

        joblib.dump(result["model"], "models/lightgbm_production.pkl")
        joblib.dump(result["encoders"], "models/label_encoders.pkl")
        print(f"Saved models/lightgbm_production.pkl ({result['trees_built']} trees)")
        print("Saved models/label_encoders.pkl")

        # Register via Azure ML's native SDK (azure-ai-ml), not MLflow's
        # Model Registry feature — see registry.py's module docstring for
        # why. The SDK handles uploading the model files itself. Never
        # tagged 'production' here — only promote_to_production()
        # (src/lifecycle/registry.py), called after evaluate_candidate()
        # passes, can do that.
        version = register_candidate(
            model_path="models/lightgbm_production.pkl",
            encoders_path="models/label_encoders.pkl",
        )
        mlflow.set_tag("registered_model_version", version)
        print(f"Registered as candidate, version {version}")


if __name__ == "__main__":
    main()
