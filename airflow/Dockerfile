FROM apache/airflow:3.0.2

# LightGBM's compiled booster needs the OpenMP runtime at the OS level — same
# fix as docker/Dockerfile, needed again here because this is a different base
# image (apache/airflow vs python:3.11-slim) that independently lacks it.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Pinned to the exact versions in ../uv.lock — a retraining DAG computing
# slightly different floats than the notebooks it's meant to replicate would be
# a silent, hard-to-debug failure mode.
RUN pip install --no-cache-dir \
    pandas==2.3.3 \
    numpy==2.4.4 \
    scikit-learn==1.8.0 \
    lightgbm==4.6.0 \
    joblib==1.5.3
