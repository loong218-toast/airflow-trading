import logging
from airflow.sdk import Variable, dag, task
import pendulum
import os
import json
from datetime import timedelta
from typing import List
from pathlib import Path

# DAG config via Airflow Variables (UI -> Admin -> Variables)
# - 'DATA_LAKE_ROOT' (optional) : path root for data lake (default: /opt/airflow/airflow-trading/data_lake)
# - 'GRID_RESUME_IF_POSSIBLE' (optional) : "true"/"false"
# - The full run configuration should be stored in the Variable named 'run_config' (JSON string).
logger = logging.getLogger("airflow.task")

DATA_LAKE_ROOT = Variable.get("DATA_LAKE_ROOT", default=os.getenv("DATA_LAKE_ROOT", "/opt/airflow/airflow-trading/data_lake"))
GRID_RESUME_IF_POSSIBLE = Variable.get("GRID_RESUME_IF_POSSIBLE", default="true").lower() == "true"

PERF_LOCAL_PATH = Path("/opt/airflow/airflow-trading/performance_config_local.json")
PERF_CLOUD_PATH = Path("/opt/airflow/airflow-trading/cloud_performance_config_cloud.json")
RUN_CONFIG_PATH = Path("/opt/airflow/airflow-trading/run_config.json")

default_args = {
    "owner": "loong",
    "retries": 0,
    "retry_delay": timedelta(minutes=10)
}

@dag(
    dag_id="grid_search",
    schedule=None,  # run manually or trigger from UI / another DAG
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    fail_fast=True,
    max_active_tasks=4,  # <--- CPU CORES (e.g., 4 or 8)
    max_active_runs=1,   # <--- DON'T ALLOW TWO GRID SEARCHES AT ONCE
    catchup=False,
    default_args=default_args,
    tags=["grid", "opt"]
)
def grid_search_pipeline():

    def _load_json_file_strict(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Required config file not found: {path}")
        txt = path.read_text(encoding="utf8")
        cfg = json.loads(txt)
        if not isinstance(cfg, dict):
            raise ValueError(f"Config at {path} must be a JSON object/dict")
        return cfg

    def apply_performance_and_run_config():
        """
        Strict loader: chooses which performance config to apply based on Airflow Variable PERFORMANCE_PROFILE.
        Sets environment variables and returns the merged run_config dict.
        """
        from airflow.sdk import Variable  # local import so module import doesn't require Airflow at test-time
        profile = Variable.get("PERFORMANCE_PROFILE", default="local").strip().lower()
        if profile not in ("local", "cloud"):
            raise ValueError("PERFORMANCE_PROFILE must be 'local' or 'cloud'")

        perf_path = PERF_LOCAL_PATH if profile == "local" else PERF_CLOUD_PATH
        perf_cfg = _load_json_file_strict(perf_path)

        # load base run_config if present (this is the run_params)

        run_cfg = _load_json_file_strict(RUN_CONFIG_PATH)


        # apply perf keys to os.environ AND to run_cfg so code sees them
        # We only set str env vars for primitive types (int/float/str/bool)
        for k, v in perf_cfg.items():
            if isinstance(v, (int, float, str, bool)):
                os.environ[k] = str(v)
            # Keep the perf key in run_cfg as well so grid logic can read it
            run_cfg[k] = v

        # return merged config for immediate use (and write snapshot later if needed)
        return run_cfg

    @task()
    def init_session_task():
        """Creates the session folder and generates batch configs, with resume safety checks."""

        from etl.session import resolve_or_create_session
        from etl.grid import generate_configs

        # parse current run config  
        current_cfg = apply_performance_and_run_config()
        cur_start = current_cfg.get("grid_start_date")
        cur_end = current_cfg.get("grid_end_date")
        pair = current_cfg.get("pair", "unknown")

        perf_map = {
            "CACHE_USE_STREAMING_MERGE": "CACHE_USE_STREAMING_MERGE",
            "CACHE_FLUSH_ROWS": "CACHE_FLUSH_ROWS",
            "CACHE_MAX_INMEM_ROWS": "CACHE_MAX_INMEM_ROWS",
            "CACHE_TMP_DIR": "CACHE_TMP_DIR",
            "DATA_LAKE_ROOT": "DATA_LAKE_ROOT"
        }
        for vkey in ("CACHE_USE_STREAMING_MERGE", "CACHE_FLUSH_ROWS", "CACHE_MAX_INMEM_ROWS", "CACHE_TMP_DIR"):
            if vkey in current_cfg:
                os.environ[vkey] = str(current_cfg[vkey])
                logger.info(f"SET ENV: {vkey}={current_cfg[vkey]}")

        logger.info("**************************************************")
        logger.info(f"🚀 STARTING GRID SEARCH FOR: {pair}")
        logger.info(f"📅 TARGET START DATE: {cur_start}")
        logger.info(f"📅 TARGET END DATE:   {cur_end}")
        logger.info(f"📂 RESUME MODE:       {GRID_RESUME_IF_POSSIBLE}")
        logger.info("**************************************************")

        # Resolve or create session dir (may return an existing session when resuming)
        session_dir = resolve_or_create_session(DATA_LAKE_ROOT, resume_if_possible=GRID_RESUME_IF_POSSIBLE)
        session_dir = Path(session_dir)
        logger.info(f"📍 SESSION DIRECTORY: {session_dir}")

        # === RESUME SAFETY CHECK ===
        # Look for a run_config snapshot written when the session was created.
        snapshot_path = session_dir / "run_config.json"
        if snapshot_path.exists():
            try:
                existing_cfg = json.loads(snapshot_path.read_text(encoding="utf8"))
            except Exception as e:
                raise RuntimeError(f"Failed to read existing session run_config snapshot: {e}")

            # Compare the two date window values (normalize to ISO strings)
            prev_start = existing_cfg.get("grid_start_date")
            prev_end = existing_cfg.get("grid_end_date")

            # If resume is enabled and the previous session window differs -> fail early
            if GRID_RESUME_IF_POSSIBLE and (prev_start != cur_start or prev_end != cur_end):
                logger.error("❌ Session resume blocked: previous session grid window differs from current run_config.")
                logger.error(f"   previous session: start={prev_start} end={prev_end}")
                logger.error(f"   current run_config: start={cur_start} end={cur_end}")
                logger.error("   To continue you can:")
                logger.error("     - Set GRID_RESUME_IF_POSSIBLE=false (Airflow Variable) to force a new session, or")
                logger.error("     - Delete or move the previous session directory, or")
                logger.error("     - Update run_config to match the previous session window.")
                # Abort by raising an error so Airflow marks the DAG run as failed.
                raise ValueError("Session resume prevented due to mismatched grid_start/grid_end. See logs for details.")
            else:
                logger.info("✔️ Existing session run_config snapshot matches current dates (resume OK).")
        else:
            # No snapshot found -> this is a fresh session: persist the run_config snapshot for future resume checks
            try:
                snapshot_path.write_text(json.dumps(current_cfg, indent=2, default=str), encoding="utf8")
                logger.info(f"💾 Wrote run_config snapshot to session: {snapshot_path.name}")
            except Exception as e:
                logger.warning(f"Could not write run_config snapshot to {snapshot_path}: {e}")

        # Generate config batches (uses current run_config)
        p = Path(session_dir)
        generate_configs(p)

        return str(session_dir)

    @task()
    def prepare_task(session_dir: str):
        """
        Prepare session-local base_data.parquet by calling etl.session.prepare_base_data.
        This wrapper builds DB URI from env and passes run_cfg loaded from RUN_CONFIG_PATH.
        """

        from etl.session import prepare_base_data as _prepare_base_data

        session_dir = Path(session_dir)
        # build DB URI
        db_user = os.getenv('POSTGRES_USER')
        db_pass = os.getenv('POSTGRES_PASSWORD')
        db_host = os.getenv('POSTGRES_HOST', 'postgres')
        db_db = os.getenv('POSTGRES_DB', 'airflow')

        db_uri = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/{db_db}"

        # load run_config (strict)
        run_cfg = _load_json_file_strict(RUN_CONFIG_PATH)

        # call the session helper (no transforms inside; data already prepared in DB)
        from etl.session import prepare_base_data as _prepare_base_data  # local import to avoid top-level cycles
        res = _prepare_base_data(session_dir=str(session_dir), db_uri=db_uri, run_cfg=run_cfg, force=False, make_full=False)

        logger.info("✅ Base data ready: %d rows -> %s", int(res.get("rows", 0)), res.get("path"))
        return res.get("path")

    @task()
    def list_pending_task(session_dir: str):
        """Finds only the BATCH files that haven't been completed yet."""
        from etl.grid import list_pending_config_paths
        # This now looks for batch_*.json instead of cfg_*.json
        pending_batches = list_pending_config_paths(Path(session_dir))
        logger.info(f"📋 Found {len(pending_batches)} batches to process.")
        return pending_batches

    @task()
    def count_batches(batches: list[str]) -> int:
        return len(batches)

    @task(pool="heavy_compute_pool", priority_weight=3)
    def worker_compute_task(cfg_path: str, session_dir: str, total_count: int):
        from etl.grid import compute_config_and_save
        from airflow.sdk import get_current_context

        # --- GET WORKER IDENTITY ---
        context = get_current_context()
        ti = context['ti']
        worker_id = getattr(ti, "map_index", -1)
        
        # Set environment variables so the cache system can see them
        os.environ["AIRFLOW_MAP_INDEX"] = str(worker_id)
        os.environ["TOTAL_WORKER_COUNT"] = str(total_count)
        
        logger.info(f"🧵 Worker {worker_id}/{total_count} processing {cfg_path}")
        
        # This calls the actual logic in etl/grid.py
        # Ensure compute_config_and_save uses the unique-part logic we discussed
        return compute_config_and_save(cfg_path, session_dir)

    @task()
    def combine_results_task(session_dir: str, dependencies):
        from etl.grid import combine_results_to_master
        # We include 'dependencies' in the arguments to force Airflow to wait for workers
        logger.info(f"🧹 Combining results for {session_dir}...")
        return combine_results_to_master(session_dir)

    # --- EXECUTION FLOW ---
    # 1. Setup
    s_path = init_session_task()
    p_data = prepare_task(s_path)
    pending_batches = list_pending_task(s_path)

    # 1. Capture the total number of tasks for the workers
    # We use a helper task or just pass the length of the list
    total_count = count_batches(pending_batches)

    # 2. Fan-out
    # We pass 'total_count' as a partial so every worker knows the goal line
    compute_results = worker_compute_task.partial(
        session_dir=s_path, 
        total_count=total_count
    ).expand(
        cfg_path=pending_batches
    )
    
    # Explicitly ensure prep is done before mapping
    p_data >> compute_results

    # compute_results >> join
    combine_results_task(s_path, dependencies=compute_results)

grid_search_dag = grid_search_pipeline()