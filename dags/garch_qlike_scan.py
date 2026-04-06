from __future__ import annotations

import json
import logging
import os
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import Variable, dag, task

logger = logging.getLogger("airflow.task")

DATA_LAKE_ROOT = Variable.get(
    "DATA_LAKE_ROOT",
    default=os.getenv("DATA_LAKE_ROOT", "/opt/airflow/airflow-trading/data_lake"),
)
RUN_CONFIG_PATH = Path(
    Variable.get(
        "RUN_CONFIG_PATH",
        default=os.getenv("RUN_CONFIG_PATH", "/opt/airflow/airflow-trading/garch_run_config.json"),
    )
)

GARCH_SCAN_ROOT = Path(DATA_LAKE_ROOT) / "garch_qlike_scan"

GARCH_POOL_NAME = Variable.get("GARCH_SCAN_POOL_NAME", default="heavy_compute_pool")

default_args = {
    "owner": "loong",
    "retries": 0,
    "retry_delay": timedelta(minutes=10),
}

MAX_ACTIVE_TASKS = 2


def _load_json_file_strict(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required config file not found: {path}")
    txt = path.read_text(encoding="utf8")
    cfg = json.loads(txt)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a JSON object/dict")
    return cfg


@dag(
    dag_id="garch_qlike_scan",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    fail_fast=True,
    max_active_tasks=MAX_ACTIVE_TASKS,
    max_active_runs=2,
    catchup=False,
    default_args=default_args,
    tags=["garch", "qlike", "volatility"],
)
def garch_qlike_scan_pipeline():

    @task()
    def init_session_task():
        from datetime import datetime, timezone

        run_cfg = _load_json_file_strict(RUN_CONFIG_PATH)

        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_dir = GARCH_SCAN_ROOT / f"session_{now}"
        session_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = session_dir / "run_config.json"
        snapshot_path.write_text(json.dumps(run_cfg, indent=2, default=str), encoding="utf8")

        logger.info("🚀 GARCH QLIKE scan session: %s", session_dir)
        logger.info("💾 Wrote run_config snapshot: %s", snapshot_path)
        logger.info("📅 grid_start=%s", run_cfg.get("grid_start_date"))
        logger.info("📅 grid_end=%s", run_cfg.get("grid_end_date"))
        logger.info("🎯 timeframes=%s", run_cfg.get("garch_timeframes", [5, 15]))
        logger.info("🎯 horizons=%s", run_cfg.get("garch_horizons_hours", [1, 4, 24]))
        logger.info("🎯 max_active_tasks=%s", MAX_ACTIVE_TASKS)

        return str(session_dir)

    @task()
    def prepare_cache_task(session_dir: str):
        from research.run_garch_scan import prepare_timeframe_cache

        session_dir = Path(session_dir)
        run_cfg = _load_json_file_strict(session_dir / "run_config.json")

        logger.info("📥 Preparing timeframe cache...")
        out = prepare_timeframe_cache(
            session_dir=str(session_dir),
            run_cfg=run_cfg,
            force_rebuild=bool(run_cfg.get("garch_force_rebuild_cache", False)),
        )
        logger.info("✅ Cache manifest: %s", out.get("manifest_path"))
        return out

    @task()
    def build_jobs_task(session_dir: str):
        from research.run_garch_scan import build_scan_jobs

        jobs = build_scan_jobs(session_dir=str(session_dir))
        logger.info("📋 Built %d jobs", len(jobs))
        return jobs

    @task(pool=GARCH_POOL_NAME, priority_weight=3)
    def worker_scan_task(job: dict, session_dir: str):
        from research.run_garch_scan import run_scan_job
        from airflow.sdk import get_current_context

        context = get_current_context()
        ti = context["ti"]
        map_index = getattr(ti, "map_index", -1)

        os.environ["AIRFLOW_MAP_INDEX"] = str(map_index)
        logger.info("🧵 worker map_index=%s job=%s", map_index, job.get("job_id"))

        return run_scan_job(job=job, session_dir=session_dir)

    @task()
    def combine_results_task(session_dir: str):
        from research.run_garch_scan import combine_scan_outputs

        logger.info("🧹 Combining GARCH scan outputs for %s", session_dir)
        return combine_scan_outputs(session_dir=session_dir)

    # FLOW
    s_path = init_session_task()
    cache_ready = prepare_cache_task(s_path)
    jobs = build_jobs_task(s_path)

    # keep ordering explicit
    cache_ready >> jobs

    results = worker_scan_task.partial(session_dir=s_path).expand(job=jobs)
    combined = combine_results_task(s_path)

    results >> combined


garch_qlike_scan_dag = garch_qlike_scan_pipeline()