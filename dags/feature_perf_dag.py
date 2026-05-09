# feature_perf_dag.py
from __future__ import annotations

import logging
import sys
import os
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import dag, task, Variable

# Path setup
PROJECT_ROOT = Path("/opt/airflow/airflow-trading")
# Ensure the folder containing 'post_grid' is in the path
DAGS_PATH = str(PROJECT_ROOT / "dags")
if DAGS_PATH not in sys.path:
    sys.path.insert(0, DAGS_PATH)

logger = logging.getLogger("airflow.task")

default_args = {
    "owner": "loong",
    "retries": 0,
    "retry_delay": timedelta(minutes=10),
}

@dag(
    dag_id="feature_performance_analysis",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["feature", "performance", "baseline_vs_signal"],
)
def feature_perf_pipeline():

    @task()
    def run_feature_perf_task():
        # Add path again inside the worker to be safe
        if DAGS_PATH not in sys.path:
            sys.path.insert(0, DAGS_PATH)
            
        from post_grid.feature_perf import main as run_analysis
        
        # We don't need to manually set SESSION_NAME here anymore 
        # because feature_perf.main() handles it internally now.
        
        try:
            run_analysis()
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Feature Performance Analysis failed: {e}")
            raise

    run_feature_perf_task()

feature_perf_dag = feature_perf_pipeline()