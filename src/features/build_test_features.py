"""Verify build_features.py reproduces train_features.parquet /
val_features.parquet, then apply the same fitted pipeline to the held-out
Kaggle test set for Phase 6 monitoring.

Run from repo root: uv run python -m src.features.build_test_features
"""

import os

import joblib
import pandas as pd

from src.features.build_features import (
    fit_pipeline,
    fit_production_target_encodings,
    merge_raw,
    transform,
)

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def verify(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    existing_train = pd.read_parquet("data/processed/train_features.parquet")
    existing_val = pd.read_parquet("data/processed/val_features.parquet")

    shared_cols = [c for c in existing_train.columns if c in train_df.columns]
    print(
        f"Comparing {len(shared_cols)} shared columns against the "
        "currently saved parquet files..."
    )

    train_diff = (
        train_df[shared_cols]
        .reset_index(drop=True)
        .compare(existing_train[shared_cols].reset_index(drop=True))
    )
    val_diff = (
        val_df[shared_cols]
        .reset_index(drop=True)
        .compare(existing_val[shared_cols].reset_index(drop=True))
    )

    print(f"train_df differing cells: {train_diff.shape[0]}")
    print(f"val_df differing cells:   {val_diff.shape[0]}")
    if not train_diff.empty:
        print(
            f"Differing columns: {sorted(set(train_diff.columns.get_level_values(0)))}"
        )


def main() -> None:
    print("Loading raw train data...")
    train_transaction = pd.read_csv("data/raw/train_transaction.csv")
    train_identity = pd.read_csv("data/raw/train_identity.csv")
    raw_train = merge_raw(train_transaction, train_identity)

    print("Fitting pipeline + reproducing train/val split...")
    artifacts, train_df, val_df = fit_pipeline(raw_train)

    print("\n=== Comparing corrected pipeline vs currently saved train/val parquet ===")
    verify(train_df, val_df)

    print(
        "\nOverwriting train_features.parquet / val_features.parquet "
        "with corrected data..."
    )
    train_df.to_parquet("data/processed/train_features.parquet")
    val_df.to_parquet("data/processed/val_features.parquet")
    print("Saved corrected data/processed/train_features.parquet")
    print("Saved corrected data/processed/val_features.parquet")

    print("\nLoading raw test data...")
    test_transaction = pd.read_csv("data/raw/test_transaction.csv")
    test_identity = pd.read_csv("data/raw/test_identity.csv")
    # Known Kaggle quirk: test_identity.csv uses "id-01" style names,
    # train_identity.csv uses "id_01" — must align before merging.
    test_identity = test_identity.rename(columns=lambda c: c.replace("-", "_"))
    raw_test = merge_raw(test_transaction, test_identity)

    # Score the test set with the *production* target encodings (fit on
    # train+val combined), not the train-only ones used for train_df/val_df —
    # the deployed lightgbm_production.pkl was retrained on the full-data
    # version (notebooks/07_final_model_training.ipynb), so that's what
    # actually has to be replayed here for the comparison to be meaningful.
    production_maps = fit_production_target_encodings(train_df, val_df)
    production_artifacts = {**artifacts, **production_maps}

    print("Applying fitted pipeline to test set (production target encodings)...")
    test_df = transform(raw_test, production_artifacts)
    print(f"test_df shape: {test_df.shape}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(production_artifacts, "models/feature_pipeline_artifacts.pkl")
    print("Saved models/feature_pipeline_artifacts.pkl")

    test_df.to_parquet("data/processed/test_features.parquet")
    print("Saved data/processed/test_features.parquet")


if __name__ == "__main__":
    main()
