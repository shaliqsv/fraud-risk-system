"""Best-effort prediction logging to Postgres.

A logging failure here must never break the prediction response — /predict's
job is to score transactions, not to guarantee log delivery. Every failure
path below logs a warning/exception and returns, rather than raising.
"""

import json
import logging
import os

import psycopg2
import psycopg2.pool

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

_pool: psycopg2.pool.SimpleConnectionPool | None = None
if DATABASE_URL:
    try:
        _pool = psycopg2.pool.SimpleConnectionPool(1, 5, DATABASE_URL)
    except Exception:
        logger.exception("Failed to create database connection pool")
        _pool = None
else:
    logger.warning("DATABASE_URL not set — prediction logging disabled")


def log_prediction(
    transaction_id: str | int | None,
    features: dict,
    fraud_probability: float,
    is_fraud: bool,
    threshold_used: float,
) -> None:
    if _pool is None:
        return

    conn = None
    try:
        conn = _pool.getconn()
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions
                    (transaction_id, features, fraud_probability,
                     is_fraud, threshold_used)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    str(transaction_id) if transaction_id is not None else None,
                    json.dumps(features),
                    fraud_probability,
                    is_fraud,
                    threshold_used,
                ),
            )
    except Exception:
        logger.exception("Failed to log prediction to database")
    finally:
        if conn is not None:
            _pool.putconn(conn)
