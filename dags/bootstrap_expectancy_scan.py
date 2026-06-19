from __future__ import annotations

import logging
import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import dag, task

PROJECT_ROOT = Path("/opt/airflow/airflow-trading")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("airflow.task")


@dag(
    dag_id="expectancy_bootstrap_scan",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "loong", "retries": 0, "retry_delay": timedelta(minutes=10)},
    tags=["expectancy", "bootstrap", "mt5", "candlestick"],
)
def expectancy_bootstrap_scan_dag():
    @task()
    def run_bootstrap_task() -> dict[str, str]:
        from research.bootstrap_expectancy_analysis import run_bootstrap_expectancy_scan

        return run_bootstrap_expectancy_scan()

    @task()
    def run_candlestick_task() -> dict[str, str]:
        from research.candlestick_trade_chart import run_candlestick_trade_chart

        return run_candlestick_trade_chart()

    bootstrap_result = run_bootstrap_task()
    candlestick_result = run_candlestick_task()

    bootstrap_result >> candlestick_result


expectancy_bootstrap_scan_dag = expectancy_bootstrap_scan_dag()