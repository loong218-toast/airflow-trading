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
    tags=["expectancy", "bootstrap", "mt5", "candlestick", "trailing_tp", "timer"],
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

    @task()
    def run_trailing_compare_task() -> dict[str, str]:
        from research.trailing_tp_compare_runner import run_trailing_tp_compare

        return run_trailing_tp_compare()

    @task()
    def run_trade_timer_compare_task() -> dict[str, str]:
        from research.trade_timer_compare_runner import run_trade_timer_compare

        return run_trade_timer_compare()

    bootstrap_result = run_bootstrap_task()
    candlestick_result = run_candlestick_task()
    trailing_result = run_trailing_compare_task()
    timer_result = run_trade_timer_compare_task()

    bootstrap_result >> candlestick_result >> [trailing_result, timer_result]


expectancy_bootstrap_scan_dag = expectancy_bootstrap_scan_dag()
