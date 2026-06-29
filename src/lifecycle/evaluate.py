"""Phase 2 — extracted from notebooks/06_champion_challenger.ipynb.

That notebook ran a thorough one-time check: a 4-fold walk-forward comparison
between two *different feature sets* (412 vs 443 features), with a paired
t-test, to validate the feature-selection decision made in
04_hyperparameter_tuning.ipynb. It found the two were statistically
indistinguishable (paired t-test p=0.847).

evaluate_candidate() below answers a different, narrower, repeatable
question: "is this freshly retrained candidate model good enough to replace
the current production model?" Both models share the same 410-feature
schema (that question is already settled), so this is a single holdout
comparison, not a multi-fold study — evaluation_dag needs a fast, decisive
answer every time training_dag completes, not a 20-minute re-validation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix

# Cost matrix from notebooks/04_threshold_optimization.ipynb.
TP_VALUE = 200
FP_COST = -5
FN_COST = -250

# Notebook 06 found champion/challenger AUC-PR differences of ~0.003 across
# folds were noise (paired t-test p=0.847) — a candidate within this much of
# production shouldn't be rejected as "worse," that's within normal variance.
DEFAULT_MAX_AUC_PR_REGRESSION = 0.01


def _profit(y_true: pd.Series, y_pred: np.ndarray) -> int:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return int(tp * TP_VALUE + fp * FP_COST + fn * FN_COST)


def evaluate_candidate(
    candidate_model,
    production_model,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    threshold: float,
    max_auc_pr_regression: float = DEFAULT_MAX_AUC_PR_REGRESSION,
) -> dict:
    """Score candidate vs. production on the same holdout, decide pass/fail.

    X_holdout must already be fully prepared (label-encoded, same 410-column
    schema both models expect) — this function only scores and compares, it
    doesn't do feature engineering or encoding itself.

    Pass criteria: candidate's AUC-PR must not fall more than
    max_auc_pr_regression below production's. This is deliberately a "don't
    regress" bar, not "must strictly improve" — mirrors notebook 06's finding
    that small differences between reasonable configs are noise, not signal.
    """
    candidate_proba = candidate_model.predict_proba(X_holdout)[:, 1]
    production_proba = production_model.predict_proba(X_holdout)[:, 1]

    candidate_auc_pr = average_precision_score(y_holdout, candidate_proba)
    production_auc_pr = average_precision_score(y_holdout, production_proba)
    auc_pr_delta = candidate_auc_pr - production_auc_pr

    candidate_profit = _profit(y_holdout, (candidate_proba >= threshold).astype(int))
    production_profit = _profit(y_holdout, (production_proba >= threshold).astype(int))

    passed = auc_pr_delta >= -max_auc_pr_regression

    return {
        "passed": passed,
        "candidate_auc_pr": float(candidate_auc_pr),
        "production_auc_pr": float(production_auc_pr),
        "auc_pr_delta": float(auc_pr_delta),
        "candidate_profit": candidate_profit,
        "production_profit": production_profit,
    }
