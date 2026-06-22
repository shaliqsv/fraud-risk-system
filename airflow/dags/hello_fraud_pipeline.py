"""Phase 10 — trivial DAG to confirm Airflow mechanics before building the real
pipeline."""

import pendulum

from airflow.decorators import dag, task


@dag(
    dag_id="hello_fraud_pipeline",
    schedule=None,  # manual trigger only — no automatic schedule yet
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["fraud-risk", "phase10"],
)
def hello_fraud_pipeline():
    @task
    def say_hello() -> str:
        print("Hello from the fraud-risk-system Airflow pipeline!")
        return "hello-done"

    @task
    def say_goodbye(message: str) -> None:
        print(f"Received from upstream task: {message}")
        print("Goodbye!")

    say_goodbye(say_hello())


hello_fraud_pipeline()
