---
name: mlops-fraud-architecture
description: Reference for this project's intended MLOps architecture for the fraud detection model — the boundary between the online serving path (FastAPI + Docker + Azure) and the offline pipeline path (Airflow DAGs for data, training, evaluation, monitoring, deployment). Consult this BEFORE writing, editing, or restructuring any Airflow DAG, retraining script, evaluation/promotion logic, monitoring job, or deployment pipeline in this repo. If a change would blur the line between serving and training, or would add training/retraining logic into the FastAPI service, stop and re-read this file.
---

# Fraud Detection MLOps Architecture

## The core rule

There are two loops in this system and they must stay separate:

- **Online serving loop** (FastAPI + Docker + Azure): real-time only. Loads whatever model is tagged `production` in the registry, scores requests, logs predictions. Never trains. Never decides if a model is "good."
- **Offline pipeline loop** (Airflow): batch only. Owns data prep, training, evaluation, monitoring, and triggering deployment. Never serves live traffic.

The only thing connecting the two loops is the **model registry** (e.g. MLflow Model Registry). The serving app reads from it. The pipelines write to it. There is no other coupling — no shared training code inside the FastAPI service, no inference code inside Airflow tasks.

If you're asked to "make retraining work" or "improve monitoring," the fix lives in Airflow DAGs and the registry tagging logic — not in the FastAPI app.

## The five DAGs

### 1. `data_feature_pipeline_dag`
- **Schedule:** daily (adjust to transaction volume)
- **Tasks:** ingest raw transactions → validate schema/quality → compute features → write to feature store → join newly arrived fraud labels (chargebacks, manual review outcomes) onto previously logged predictions
- **Output:** a labeled dataset consumed by both `evaluation_dag` (as holdout) and `monitoring_dag` (as ground truth for real performance)
- **Note:** fraud labels arrive with delay. This DAG is what backfills them. Don't expect same-day ground truth.

### 2. `training_dag`
- **Schedule:** weekly, plus on-demand trigger from `monitoring_dag`
- **Tasks:** pull training window from feature store → train candidate model → log run + artifact to registry tagged `candidate` (never `production` directly)
- **Triggers on completion:** `evaluation_dag`

### 3. `evaluation_dag`
- **Trigger:** fires immediately when `training_dag` completes (TriggerDagRunOperator or dataset-aware scheduling)
- **Tasks:** score `candidate` vs current `production` model (champion/challenger) on freshest labeled holdout → check against defined thresholds (precision/recall at operating point, cost-weighted metric reflecting false-positive vs false-negative asymmetry) → if it passes, retag model `production` in registry; if not, alert and halt
- **This is the gate.** Nothing gets promoted without passing through here.
- **Triggers on promotion:** `deployment_dag`

### 4. `monitoring_dag`
- **Schedule:** hourly or daily, fully independent of the training cycle
- **Tasks:** read logged predictions + newly joined labels → compute data drift, prediction drift, and real performance decay → alert if thresholds breached → optionally trigger `training_dag` out-of-cycle
- **Guard rail:** include a cooldown so a noisy signal can't cause retrain storms (e.g. don't re-trigger training more than once per N hours regardless of how many alerts fire)

### 5. `deployment_dag`
- **Trigger:** fires on the registry promotion event from `evaluation_dag`
- **Tasks:** build Docker image with new artifact → push to Azure Container Registry → roll out (canary or rolling update) → smoke test `/health` and `/predict` → shift full traffic only after smoke test passes
- **Alternative:** this step can live in CI/CD instead of Airflow. Pick one and be consistent — don't split it across both.

## Dependency graph

```
data_feature_pipeline_dag (daily)
        │
        ▼
training_dag (weekly / on-demand) ──► evaluation_dag ──(pass)──► deployment_dag
        ▲                                   │
        │                              (fail: alert, halt)
        │
monitoring_dag (hourly/daily, independent) ──(drift/decay detected)──┘
```

## Things to check before approving a change

- Does this change add training, model-fitting, or evaluation logic to the FastAPI service? → wrong, move it to a DAG.
- Does this change make a DAG call the live `/predict` endpoint to serve traffic? → wrong, DAGs consume logged/batch data, not live requests.
- Does a new model get tagged `production` anywhere outside `evaluation_dag`? → wrong, that's the only gate.
- Does `monitoring_dag` retrain anything directly? → wrong, it only triggers `training_dag`; it never trains itself.
- Is there a cooldown/guard on auto-triggered retraining? → required, to prevent retrain storms from noisy drift signals.
