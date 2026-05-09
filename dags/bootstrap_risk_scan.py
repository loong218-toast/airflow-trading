# bootstrap_risk_scan.py

from __future__ import annotations

import logging
import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import dag, task

PROJECT_ROOT = Path("/opt/airflow/airflow-trading")
DAGS_ROOT = PROJECT_ROOT / "dags"

if str(DAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(DAGS_ROOT))

logger = logging.getLogger("airflow.task")


@dag(
    dag_id="bootstrap_risk_scan",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "loong", "retries": 0, "retry_delay": timedelta(minutes=10)},
    tags=["trade_ml", "bootstrap", "risk", "drawdown"],
)
def bootstrap_risk_scan_dag():
    @task()
    def run_bootstrap_risk_task() -> dict[str, str]:
        from post_grid.trade_ml_bootstrap_risk import main as run_bootstrap_risk

        run_bootstrap_risk()
        logger.info("bootstrap_risk_scan finished successfully")
        return {"status": "ok"}

    run_bootstrap_risk_task()


bootstrap_risk_scan_dag = bootstrap_risk_scan_dag()