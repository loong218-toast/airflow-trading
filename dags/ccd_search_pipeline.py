from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import get_current_context
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Variable, dag, task

logger = logging.getLogger("airflow.task")

DATA_LAKE_ROOT = Variable.get(
    "DATA_LAKE_ROOT",
    default=os.getenv("DATA_LAKE_ROOT", "/opt/airflow/airflow-trading/data_lake"),
)
CCD_RESUME_IF_POSSIBLE = Variable.get("CCD_RESUME_IF_POSSIBLE", default="false").lower() == "true"

PERF_LOCAL_PATH = Path("/opt/airflow/airflow-trading/performance_config_local.json")
PERF_CLOUD_PATH = Path("/opt/airflow/airflow-trading/cloud_performance_config_cloud.json")
RUN_CONFIG_PATH = Path("/opt/airflow/airflow-trading/run_config.json")


def _load_json_file_strict(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required config file not found: {path}")
    txt = path.read_text(encoding="utf8")
    cfg = json.loads(txt)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a JSON object/dict")
    return cfg


def apply_performance_and_run_config() -> dict:
    """
    Load the run config and overlay the selected performance profile.

    The same snapshot is reused across CCD cycle runs.
    """
    profile = Variable.get("PERFORMANCE_PROFILE", default="local").strip().lower()
    if profile not in ("local", "cloud"):
        raise ValueError("PERFORMANCE_PROFILE must be 'local' or 'cloud'")

    perf_path = PERF_LOCAL_PATH if profile == "local" else PERF_CLOUD_PATH
    perf_cfg = _load_json_file_strict(perf_path)
    run_cfg = _load_json_file_strict(RUN_CONFIG_PATH)

    for k, v in perf_cfg.items():
        if isinstance(v, (int, float, str, bool)):
            os.environ[k] = str(v)
        run_cfg[k] = v

    run_cfg["search_mode"] = str(run_cfg.get("search_mode", "cyclic_coordinate_descent"))
    return run_cfg

def _build_next_cycle_payload(
    *,
    session_dir: Path,
    current_cycle: int,
    max_cycles: int,
    current_run_id: str,
    dag_conf: dict,
    run_cfg: dict,
    result: dict,
    state: dict,
    cycle_seconds: float,
    avg_cycle_seconds: float,
) -> dict:
    fatal_reasons = {
        "empty_master_df",
        "no_pending_batches",
        "no_candidate_keys_for_block",
        "max_cycles_reached",
    }

    reason = str(result.get("reason", "") or "")
    next_cycle = int(current_cycle) + 1
    next_cycle_exists = next_cycle < int(max_cycles)

    should_trigger_next = next_cycle_exists and reason not in fatal_reasons
    if reason == "no_incumbent_found":
        should_trigger_next = next_cycle_exists

    progress = dict(state.get("progress") or {})
    remaining_cycles = max(0, int(max_cycles) - next_cycle)
    eta_seconds = float(avg_cycle_seconds) * remaining_cycles

    ts = datetime.now().strftime("%H%M%S")
    next_run_id = f"ccd_search__{session_dir.name}__cycle_{next_cycle:04d}__{ts}"

    next_conf = {
        "session_dir": str(session_dir),
        "cycle_idx": next_cycle,
        "max_cycles": int(max_cycles),
        "root_run_id": str(dag_conf.get("root_run_id") or current_run_id),
        "parent_run_id": current_run_id,
        "search_mode": str(run_cfg.get("search_mode", "cyclic_coordinate_descent")),
        "bootstrap": False,
    }

    return {
        "cycle_idx": int(current_cycle),
        "cycle_display": int(current_cycle) + 1,
        "max_cycles": int(max_cycles),
        "max_cycle_display": int(max_cycles),
        "done": True,
        "selected": bool(result.get("selected", False)),
        "cycle_seconds": float(cycle_seconds),
        "avg_cycle_seconds": float(avg_cycle_seconds),
        "remaining_cycles": int(remaining_cycles),
        "eta_seconds": float(eta_seconds),
        "loss_delta": progress.get("loss_delta"),
        "candidate_loss": progress.get("candidate_loss"),
        "incumbent_loss_before": progress.get("incumbent_loss_before"),
        "best_seen_loss": progress.get("best_seen_loss"),
        "active_block": state.get("active_block"),
        "block_refine_idx": state.get("block_refine_idx"),
        "search_scale": state.get("search_scale"),
        "should_trigger_next": bool(should_trigger_next),
        "next_run_id": next_run_id if should_trigger_next else None,
        "next_conf": next_conf if should_trigger_next else None,
        "reason": reason,
    }


@dag(
    dag_id="ccd_search",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    fail_fast=True,
    max_active_tasks=2,
    max_active_runs=1,
    catchup=False,
    default_args={"owner": "loong", "retries": 0, "retry_delay": timedelta(minutes=10)},
    tags=["ccd", "opt"],
)
def ccd_search_pipeline():
    @task()
    def init_session_task():
        """
        Resolve the session directory.

        First run creates or resumes the session from local state.
        Self-triggered cycle runs reuse the same session_dir from dag_run.conf.
        """
        from common.session import resolve_or_create_session

        current_cfg = apply_performance_and_run_config()
        cur_start = current_cfg.get("grid_start_date")
        cur_end = current_cfg.get("grid_end_date")
        pair = current_cfg.get("pair", "unknown")

        ctx = get_current_context()
        dag_run = ctx.get("dag_run")
        dag_conf = dict(getattr(dag_run, "conf", {}) or {})
        session_dir_hint = dag_conf.get("session_dir")

        for vkey in ("CACHE_USE_STREAMING_MERGE", "CACHE_FLUSH_ROWS", "CACHE_MAX_INMEM_ROWS", "CACHE_TMP_DIR"):
            if vkey in current_cfg:
                os.environ[vkey] = str(current_cfg[vkey])
                logger.info("SET ENV: %s=%s", vkey, current_cfg[vkey])

        logger.info("**************************************************")
        logger.info("🚀 STARTING CCD SEARCH FOR: %s", pair)
        logger.info("📅 TARGET START DATE: %s", cur_start)
        logger.info("📅 TARGET END DATE:   %s", cur_end)
        logger.info("📂 RESUME MODE:       %s", CCD_RESUME_IF_POSSIBLE)
        logger.info("**************************************************")

        if session_dir_hint:
            session_dir = Path(str(session_dir_hint))
            session_dir.mkdir(parents=True, exist_ok=True)
            logger.info("📍 REUSING SESSION DIRECTORY FROM DAG CONF: %s", session_dir)
        else:
            session_dir = Path(resolve_or_create_session(DATA_LAKE_ROOT, resume_if_possible=CCD_RESUME_IF_POSSIBLE))
            logger.info("📍 CREATED/RESUMED SESSION DIRECTORY: %s", session_dir)

        snapshot_path = session_dir / "run_config.json"
        if snapshot_path.exists():
            existing_cfg = json.loads(snapshot_path.read_text(encoding="utf8"))
            prev_start = existing_cfg.get("grid_start_date")
            prev_end = existing_cfg.get("grid_end_date")
            if CCD_RESUME_IF_POSSIBLE and (prev_start != cur_start or prev_end != cur_end):
                raise ValueError("Session resume prevented due to mismatched grid_start/grid_end.")
            logger.info("✔️ Existing session run_config snapshot matches current dates (resume OK).")
        else:
            snapshot_path.write_text(json.dumps(current_cfg, indent=2, default=str), encoding="utf8")
            logger.info("💾 Wrote run_config snapshot to session: %s", snapshot_path.name)

        # Shared session cleanup before the CCD cycle begins.
        # Batch masters from older runs are removed only at the cycle boundary.
        if bool(current_cfg.get("coord_descent", {}).get("clean_old_batch_masters", True)):
            from research.ccd_maintenance import cleanup_old_ccd_batch_masters
            cleanup_old_ccd_batch_masters(session_dir)

        return str(session_dir)

    @task()
    def prepare_task(session_dir: str):
        """
        Build the base market dataset once for the session.

        The file path is reused across self-triggered cycle runs.
        """
        from common.session import prepare_base_data as _prepare_base_data

        session_dir = Path(session_dir)
        db_user = os.getenv("POSTGRES_USER")
        db_pass = os.getenv("POSTGRES_PASSWORD")
        db_host = os.getenv("POSTGRES_HOST", "postgres")
        db_db = os.getenv("POSTGRES_DB", "airflow")
        db_uri = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/{db_db}"

        session_snapshot = session_dir / "run_config.json"
        with open(session_snapshot, "r", encoding="utf8") as f:
            run_cfg = json.load(f)

        res = _prepare_base_data(
            session_dir=str(session_dir),
            db_uri=db_uri,
            run_cfg=run_cfg,
            force=False,
            make_full=False,
        )
        logger.info("✅ Base data ready: %d rows -> %s", int(res.get("rows", 0)), res.get("path"))
        return res.get("path")

    @task()
    def prepare_features_task(session_dir: str):
        """
        Precompute the reusable feature cache once.

        The cache is reused across all CCD cycle runs for the same session.
        """
        from common.feature_prep import prepare_feature_cache

        session_dir = Path(session_dir)
        session_snapshot = session_dir / "run_config.json"
        if not session_snapshot.exists():
            raise RuntimeError(f"Missing run_config snapshot at {session_snapshot}")

        run_cfg = json.loads(session_snapshot.read_text(encoding="utf8"))
        base_path = Path(DATA_LAKE_ROOT) / "base_data_full" / "base_data_full.parquet"
        if not base_path.exists():
            raise FileNotFoundError(f"CRITICAL: Base data not found at {base_path}.")

        ref_path = session_dir / "prepared_main_ref.json"
        if ref_path.exists():
            try:
                ref = json.loads(ref_path.read_text(encoding="utf8"))
                cached_path = Path(ref.get("parquet_path", ""))
                if cached_path.exists() and not bool(run_cfg.get("force_rebuild_cache", False)):
                    logger.info("✅ Prepared feature cache already present: %s", cached_path)
                    return str(cached_path)
            except Exception:
                pass

        prepared_path, manifest = prepare_feature_cache(
            base_path=base_path,
            run_cfg=run_cfg,
            force_rebuild=bool(run_cfg.get("force_rebuild_cache", False)),
        )

        ref = {
            "parquet_path": str(prepared_path),
            "manifest_key": manifest["cache_key"],
            "lookback_map": run_cfg.get("lookback_map", {}),
            "ma_cols": run_cfg.get("ma_cols", []),
            "stoch_cols": run_cfg.get("stoch_cols", []),
            "bbw_cols": run_cfg.get("bbw_cols", []),
        }
        ref_path.write_text(json.dumps(ref, indent=2), encoding="utf8")
        logger.info("✅ Prepared feature cache ready: %s", prepared_path)
        return str(prepared_path)

    @task(pool="heavy_compute_pool")
    def cleanup_ccd_session_task(session_dir: str):
        """
        Session-level cleanup for stale batch masters.

        Serialized execution prevents file conflicts across workers.
        """
        from pathlib import Path
        from research.ccd_maintenance import cleanup_old_ccd_batch_masters

        cleanup_old_ccd_batch_masters(Path(session_dir))
        return session_dir

    @task(pool="heavy_compute_pool")
    def run_ccd_loop_task(session_dir: str):
        """
        Single CCD cycle executor.

        The recursive loop is implemented by self-triggered DAG runs.
        One DAG run advances one CCD cycle, records progress, and prepares the
        next run payload when additional cycles remain.
        """
        from research.ccd import (
        ensure_coord_descent_state,
        load_compact_master_metrics,
        pick_next_incumbent,
        save_coord_descent_state,
        )
        from research.ccd_config import generate_configs, list_pending_config_paths
        from research.ccd_maintenance import (
            cleanup_old_ccd_batch_masters,
            cleanup_ccd_cycle_artifacts,
        )
        from research.merge_utils import combine_results_to_master
        from research.grid import compute_config_and_save

        ctx = get_current_context()
        current_run_id = str(ctx["dag_run"].run_id)
        dag_conf = dict(getattr(ctx["dag_run"], "conf", {}) or {})

        session_dir = Path(session_dir)
        run_cfg = json.loads((session_dir / "run_config.json").read_text(encoding="utf8"))
        coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})
        max_cycles = int(coord_cfg.get("max_cycles", 10) or 10)

        state = ensure_coord_descent_state(session_dir, run_cfg)

        # DAG payload is the source of truth for the current cycle.
        current_cycle = int(dag_conf.get("cycle_idx", state.get("cycle_idx", 0)) or 0)

        # Keep persisted state aligned with the DAG run before heavy work starts.
        state["cycle_idx"] = current_cycle
        save_coord_descent_state(session_dir, state)

        if current_cycle >= max_cycles:
            logger.info("CCD stop | cycle_idx=%d reached max_cycles=%d", current_cycle, max_cycles)
            return {
                "cycle_idx": current_cycle,
                "cycle_display": current_cycle + 1,
                "max_cycles": max_cycles,
                "max_cycle_display": max_cycles,
                "done": True,
                "should_trigger_next": False,
                "next_run_id": None,
                "next_conf": None,
                "reason": "max_cycles_reached",
            }

        cycle_start = time.monotonic()

        logger.info("CCD cycle start | cycle_idx=%d/%d | active_block=%s | resume=%s",
                    current_cycle, max_cycles - 1, state.get("active_block"), bool(CCD_RESUME_IF_POSSIBLE))

        cleanup_old_ccd_batch_masters(session_dir)
        cleanup_ccd_cycle_artifacts(session_dir, dry_run=False)

        logger.info("CCD step | generating configs")
        generate_configs(session_dir, run_cfg)

        pending_batches = list_pending_config_paths(session_dir)
        logger.info("CCD step | pending_batches=%d", len(pending_batches))

        if not pending_batches:
            logger.warning("CCD cycle produced no pending batches.")
            return {
                "cycle_idx": current_cycle,
                "cycle_display": current_cycle + 1,
                "max_cycles": max_cycles,
                "max_cycle_display": max_cycles,
                "done": True,
                "should_trigger_next": False,
                "next_run_id": None,
                "next_conf": None,
                "reason": "no_pending_batches",
            }

        for cfg_path in pending_batches:
            logger.info("CCD step | computing %s", Path(cfg_path).name)
            compute_config_and_save(cfg_path, str(session_dir))
            logger.info("CCD step | finished %s", Path(cfg_path).name)

        logger.info("CCD step | merging batch masters")
        combine_results_to_master(session_dir)

        merged_master = session_dir / "master_metrics.parquet"
        master_df = load_compact_master_metrics(session_dir, master_path=merged_master)

        logger.info("CCD step | picking next incumbent")
        result = pick_next_incumbent(session_dir=session_dir, run_cfg=run_cfg, master_df=master_df)
        state = result.get("state", state)

        # Keep the state aligned with the latest cycle number before saving.
        state["cycle_idx"] = current_cycle
        save_coord_descent_state(session_dir, state)

        cycle_seconds = time.monotonic() - cycle_start

        history = list(state.get("history", []))
        cycle_durations = [
            float(x.get("cycle_seconds", 0.0))
            for x in history
            if isinstance(x, dict) and x.get("cycle_seconds") is not None
        ]
        avg_cycle_seconds = sum(cycle_durations) / len(cycle_durations) if cycle_durations else cycle_seconds

        payload = _build_next_cycle_payload(
            session_dir=session_dir,
            current_cycle=current_cycle,
            max_cycles=max_cycles,
            current_run_id=current_run_id,
            dag_conf=dag_conf,
            run_cfg=run_cfg,
            result=result,
            state=state,
            cycle_seconds=cycle_seconds,
            avg_cycle_seconds=avg_cycle_seconds,
        )

        logger.info(
            "CCD cycle %d/%d | active_block=%s | refine=%s/%s | scale=%.3f | selected=%s | loss_delta=%s | best_seen_loss=%s | remaining=%d | avg_cycle=%.1fs | eta=%.1fs",
            payload["cycle_display"],
            payload["max_cycle_display"],
            state.get("active_block"),
            state.get("block_refine_idx"),
            state.get("block_refine_rounds"),
            state.get("search_scale"),
            bool(result.get("selected", False)),
            payload["loss_delta"],
            payload["best_seen_loss"],
            payload["remaining_cycles"],
            avg_cycle_seconds,
            payload["eta_seconds"],
        )

        logger.info(
            "CCD decision debug | cycle=%s | reason=%s | selected=%s | incumbent_regime_id=%s | should_trigger_next=%s | next_cycle_exists=%s",
            current_cycle,
            payload["reason"],
            bool(result.get("selected", False)),
            state.get("incumbent", {}).get("regime_id"),
            payload["should_trigger_next"],
            (current_cycle + 1) < max_cycles,
        )

        return payload

    @task.branch
    def choose_cycle_branch(cycle_result: dict) -> str:
        """
        Branch selection for the CCD chain.

        The next-cycle trigger is selected while additional cycles remain.
        The final equity task is selected on the terminal cycle.
        """
        if cycle_result.get("should_trigger_next"):
            return "trigger_next_cycle_task"
        return "finalize_selected_equity_task"

    @task()
    def finalize_selected_equity_task(session_dir: str):
        """
        Recompute only the chosen incumbent once and write final equity to:
        session_dir/equity_partitioned/

        The chosen incumbent is read from coord_descent_state.json after CCD state
        has been updated by the most recent cycle run.
        """
        from uuid import uuid4

        from research.ccd import load_coord_descent_state
        from research.grid import compute_config_and_save

        session_dir = Path(session_dir)
        state = load_coord_descent_state(session_dir)

        incumbent = dict(state.get("incumbent") or {})
        regime_cfg = dict(incumbent.get("regime_cfg") or {})
        regime_id = incumbent.get("regime_id", None)

        if regime_id is None:
            return {
                "selected": False,
                "reason": "no_incumbent_found",
            }

        session_snapshot = session_dir / "run_config.json"
        if not session_snapshot.exists():
            raise RuntimeError(f"Missing run_config snapshot at {session_snapshot}")

        run_cfg = json.loads(session_snapshot.read_text(encoding="utf8"))

        # Finalize batch writes equity into the final folder instead of the CCD search folder.
        finalize_batch_id = int(run_cfg.get("FINALIZE_BATCH_ID", 999999))
        temp_batch_path = session_dir / "results" / f"_finalize_selected_{uuid4().hex}.json"

        final_regime = dict(regime_cfg)
        final_regime["regime_id"] = int(regime_id)

        temp_payload = {
            "batch_id": finalize_batch_id,
            "search_mode": "finalize_selected",
            "coord_descent": {
                "active_block": None,
                "cycle_idx": int(state.get("cycle_idx", 0) or 0),
                "profile_count": len(state.get("profiles", [])),
                "state_path": str(session_dir / "coord_descent_state.json"),
            },
            "regimes": [final_regime],
        }

        temp_batch_path.write_text(json.dumps(temp_payload, indent=2, default=str), encoding="utf8")

        try:
            result = compute_config_and_save(str(temp_batch_path), str(session_dir))
        finally:
            try:
                temp_batch_path.unlink(missing_ok=True)
            except Exception:
                pass

        return {
            "selected": True,
            "regime_id": int(regime_id),
            "result": result,
        }

    s_path = init_session_task()
    p_data = prepare_task(s_path)
    f_data = prepare_features_task(s_path)
    cleanup_task = cleanup_ccd_session_task(s_path)

    cycle_result = run_ccd_loop_task(s_path)
    cycle_branch = choose_cycle_branch(cycle_result)

    trigger_next_cycle_task = TriggerDagRunOperator(
        task_id="trigger_next_cycle_task",
        trigger_dag_id="ccd_search",
        trigger_run_id="{{ (task_instance.xcom_pull(task_ids='run_ccd_loop_task') or {}).get('next_run_id', 'next_run_placeholder') }}",
        conf="{{ (task_instance.xcom_pull(task_ids='run_ccd_loop_task') or {}).get('next_conf', {}) | tojson }}",
        wait_for_completion=False,
    )

    finalize_equity = finalize_selected_equity_task(s_path)

    p_data >> f_data >> cleanup_task >> cycle_result >> cycle_branch
    cycle_branch >> trigger_next_cycle_task
    cycle_branch >> finalize_equity


ccd_search_dag = ccd_search_pipeline()