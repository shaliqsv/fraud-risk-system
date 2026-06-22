"""Reusable feature engineering pipeline, mirroring
notebooks/02_feature_engineering.ipynb.

The notebook computed several statistics (IQR fence, percentile thresholds,
imputation medians, target-encoding maps, label encoders) inline and never
persisted them. This module splits that logic into fit/transform so the exact
pipeline used to build train_features.parquet / val_features.parquet can be
replayed on new raw data (e.g. the held-out Kaggle test set) for monitoring.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LOW_CARD_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "M1",
    "M2",
    "M3",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]


def merge_raw(transaction_df: pd.DataFrame, identity_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(transaction_df, identity_df, on="TransactionID", how="left")
    df["has_identity"] = (
        df["TransactionID"].isin(identity_df["TransactionID"]).astype(int)
    )
    return df


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pure-formula features — no fitted state, safe to apply to any raw data."""
    df = df.copy()
    df["log_TransactionAmt"] = np.log1p(df["TransactionAmt"])
    df["day_of_week"] = (df["TransactionDT"] // (24 * 3600)) % 7
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["hour_is_risky"] = df["hour"].isin([5, 6, 7, 8, 9]).astype(int)
    df["card4_6"] = df["card4"].fillna("unknown") + "_" + df["card6"].fillna("unknown")
    df["M4_encoded"] = df["M4"].map({"M0": 0, "M1": 1, "M2": 2})
    # NOTE: M4_encoded is already a column by this point, and its name starts
    # with "M" — it gets swept into m_cols below and double-counted alongside
    # M4 itself. That's a quirk of the original notebook, preserved here
    # deliberately so results match train_features.parquet exactly.
    m_cols = [col for col in df.columns if col.startswith("M")]
    df["M_null_count"] = df[m_cols].isnull().sum(axis=1)
    return df


def fit_global_artifacts(df: pd.DataFrame) -> dict:
    """Fit on the full raw merged set, BEFORE the time-based split.

    Matches notebook chapters 1, 6, 7 — those ran on the full 590,540-row
    train+val combined set, before the chapter 8 split. Reusing that same
    full set for fitting keeps test-set scoring consistent with the
    production model, which was also retrained on the full set.

    email_target_encoded and card4_6_target_encoded are deliberately NOT fit
    here. The original train_features.parquet/val_features.parquet on disk
    were found to contain a leaky version of email_target_encoded (fit on
    the full train+val set, computed in notebook chapter 4) instead of the
    train-only version chapter 9's prose describes — a case of the
    notebook's saved cell output drifting from its current code. This
    pipeline fits both target encodings correctly, on train_df only, in
    fit_train_only_artifacts below. A separate full-data-fit version (matching
    notebooks/07_final_model_training.ipynb's production retrain) is available
    via fit_production_target_encodings, for scoring data with the deployed
    model rather than reproducing train_features.parquet/val_features.parquet.
    """
    Q1 = df["TransactionAmt"].quantile(0.25)
    Q3 = df["TransactionAmt"].quantile(0.75)
    amt_outlier_fence = Q3 + 1.5 * (Q3 - Q1)

    c_cols = [col for col in df.columns if col.startswith("C")]
    c_75th = df[c_cols].quantile(0.75)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_medians = df[numeric_cols].median()

    return {
        "amt_outlier_fence": amt_outlier_fence,
        "c_cols": c_cols,
        "c_75th": c_75th,
        "numeric_cols": numeric_cols,
        "numeric_medians": numeric_medians,
    }


def apply_global_artifacts(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = df.copy()
    df["is_outlier"] = (df["TransactionAmt"] > artifacts["amt_outlier_fence"]).astype(
        int
    )
    df["high_velocity"] = (df[artifacts["c_cols"]] > artifacts["c_75th"]).sum(axis=1)

    numeric_cols = [c for c in artifacts["numeric_cols"] if c in df.columns]
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    df[numeric_cols] = df[numeric_cols].fillna(
        artifacts["numeric_medians"][numeric_cols]
    )
    df[cat_cols] = df[cat_cols].fillna("unknown")
    return df


def fit_train_only_artifacts(train_df: pd.DataFrame) -> dict:
    """Fit on train_df only, AFTER the time split — matches notebook chapter 9's
    intent: target encoding and low-cardinality label encoding must be fit on
    train only to avoid leaking val/test fraud rates into the encoding.
    """
    email_fraud_map = train_df.groupby("P_emaildomain")["isFraud"].mean()
    card_fraud_map = train_df.groupby("card4_6")["isFraud"].mean()
    train_fraud_mean = train_df["isFraud"].mean()

    label_encoders = {}
    for col in LOW_CARD_COLS:
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        label_encoders[col] = le

    return {
        "email_fraud_map": email_fraud_map,
        "card_fraud_map": card_fraud_map,
        "train_fraud_mean": train_fraud_mean,
        "label_encoders": label_encoders,
    }


def fit_production_target_encodings(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> dict:
    """Refit email/card4_6 target encoding on train+val combined — matches
    notebooks/07_final_model_training.ipynb's production retrain, which trains
    on the full labeled set with no held-out split left to leak into.

    The deployed lightgbm_production.pkl was trained on *this* version, not
    the train-only version fit_train_only_artifacts produces — so scoring the
    held-out test set (or any new data) with that model must use this version
    too, to actually reflect what the deployed model sees in production.
    Merge the result into a fit_pipeline artifacts dict before calling
    transform(), overriding its train-only email_fraud_map/card_fraud_map.
    """
    full_df = pd.concat([train_df, val_df], axis=0)
    return {
        "email_fraud_map": full_df.groupby("P_emaildomain")["isFraud"].mean(),
        "card_fraud_map": full_df.groupby("card4_6")["isFraud"].mean(),
        "train_fraud_mean": full_df["isFraud"].mean(),
    }


def apply_train_only_artifacts(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = df.copy()
    df["email_target_encoded"] = (
        df["P_emaildomain"]
        .map(artifacts["email_fraud_map"])
        .fillna(artifacts["train_fraud_mean"])
    )
    df["card4_6_target_encoded"] = (
        df["card4_6"]
        .map(artifacts["card_fraud_map"])
        .fillna(artifacts["train_fraud_mean"])
    )

    for col, le in artifacts["label_encoders"].items():
        values = df[col].astype(str)
        known = set(le.classes_)
        # Unseen categories fall back to the first known class — same
        # strategy api/model_loader.py uses at inference time, so test-set
        # scoring matches what the live API would actually do.
        values = values.where(values.isin(known), le.classes_[0])
        df[col] = le.transform(values)

    return df


def fit_pipeline(raw_train_df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Full fit, replicating notebooks/02_feature_engineering.ipynb end to end.

    raw_train_df must be merge_raw(train_transaction, train_identity) — the
    full 590,540-row combined set, *before* the chapter 8 time split.
    Returns (artifacts, train_df, val_df) so callers can verify train_df/val_df
    against the existing parquet files.
    """
    base = add_base_features(raw_train_df)
    global_artifacts = fit_global_artifacts(base)
    base = apply_global_artifacts(base, global_artifacts)

    split_point = base["TransactionDT"].quantile(0.75)
    train_df = base[base["TransactionDT"] <= split_point].copy()
    val_df = base[base["TransactionDT"] > split_point].copy()

    train_only_artifacts = fit_train_only_artifacts(train_df)
    train_df = apply_train_only_artifacts(train_df, train_only_artifacts)
    val_df = apply_train_only_artifacts(val_df, train_only_artifacts)

    artifacts = {**global_artifacts, **train_only_artifacts, "split_point": split_point}
    return artifacts, train_df, val_df


def transform(raw_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Apply a previously fitted pipeline to new raw data (e.g. the test set)."""
    base = add_base_features(raw_df)
    base = apply_global_artifacts(base, artifacts)
    return apply_train_only_artifacts(base, artifacts)
