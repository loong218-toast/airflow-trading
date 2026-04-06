import os
import json
import time
import shutil
import logging
import datetime
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from etl.db import get_engine
from etl.transform import build_df_main_from_5m_polars, load_candles_from_db_polars

from research.io_utils import (
    _ensure_full_lake_dir,
    FULL_LAKE_DIR,
    MANIFEST_FILE,
    atomic_write_parquet,
)

# Constants - ensure these match your environment
DEFAULT_DATA_LAKE_ROOT = os.getenv("DATA_LAKE_ROOT", "/opt/airflow/airflow-trading/data_lake")
SESSION_PREFIX = "Opt_Session"

logger = logging.getLogger(__name__)

# --- Helper: Parquet Checkers ---

def _parquet_has_rows(p: Path) -> bool:
    """Fail-safe check if a parquet file contains data."""
    try:
        if not p.exists(): return False
        meta = pq.ParquetFile(str(p)).metadata
        return getattr(meta, "num_rows", 0) > 0
    except Exception:
        return True # Safe: don't delete if we can't be sure

def _any_parquet_with_rows_under(dirpath: Path) -> bool:
    """Recursive check for any data-filled parquets in a directory."""
    if not dirpath.exists(): return False
    # Check top-level then recursive subfolders
    for p in list(dirpath.glob("*.parquet")) + list(dirpath.rglob("*.parquet")):
        if _parquet_has_rows(p): return True
    return False

# --- Session Logic ---

def is_session_finished(session_dir: Path) -> bool:
    """
    Checks if the final consolidation has completed.
    In the streaming architecture, master_metrics.parquet is the final artifact.
    """
    master_path = session_dir / "master_metrics.parquet"
    
    # If the master file exists and isn't a 0-byte ghost file, we are done.
    if master_path.exists():
        try:
            if pq.ParquetFile(str(master_path)).metadata.num_rows > 0:
                return True
        except Exception:
            # If the file is corrupted or currently being written, don't skip yet
            return False

    return False

def prune_old_sessions(base_root: Optional[str] = None, keep: int = 10) -> Dict[str, Any]:
    """
    Keeps only the 'keep' most recent Opt_Session folders.
    Deletes the oldest ones regardless of whether they are 'finished'.
    """
    base = Path(base_root or DEFAULT_DATA_LAKE_ROOT)
    if not base.exists(): 
        return {"error": "Root missing"}

    # 1. Get all session directories
    sessions = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(SESSION_PREFIX)]
    
    # 2. Sort by modification time (newest first)
    # Using os.path.getmtime is often more reliable across Docker volumes
    sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # 3. Identify folders to delete (everything after the first 'keep' items)
    to_delete = sessions[keep:]
    deleted_names = []

    for s in to_delete:
        try:
            logger.info(f"Pruning old session: {s.name}")
            shutil.rmtree(s, ignore_errors=True)
            deleted_names.append(s.name)
        except Exception as e:
            logger.error(f"Failed to delete {s.name}: {e}")

    return {
        "deleted_count": len(deleted_names),
        "deleted_items": deleted_names,
        "total_remaining": len(sessions) - len(deleted_names)
    }

def resolve_or_create_session(base_root: str, resume_if_possible: bool = True) -> Path:
    """Finds the last incomplete session or starts a fresh one."""
    base = Path(base_root)
    base.mkdir(parents=True, exist_ok=True)

    prune_old_sessions(base_root=base_root, keep=10)

    sessions = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith(SESSION_PREFIX)],
                      key=lambda p: p.stat().st_mtime)

    if resume_if_possible and sessions:
        last = sessions[-1]
        # If results aren't finished, we reuse it
        if not is_session_finished(last):
            return last

    ts = time.strftime("%Y%m%d_%H%M%S")
    candidate = base / f"{SESSION_PREFIX}_{ts}_01"
    candidate.mkdir(parents=True)
    (candidate / "configs").mkdir()
    (candidate / "results").mkdir()
    return candidate


def prepare_base_data(session_dir: str, db_uri: str, run_cfg: Dict[str, Any], force: bool = False, make_full: bool = False) -> Dict[str, Any]:
    """
    Load or build a single global base_data_full.parquet in FULL_LAKE_DIR and return its path & rows.
    - If FULL_LAKE_DIR/base_data_full.parquet exists and has rows and not force -> reuse.
    - Else if make_full True -> build full file from DB and write to FULL_LAKE_DIR/base_data_full.parquet.
    - Else -> build the minimal padded slice for grid window and write to FULL_LAKE_DIR/base_data_full.parquet.
    NOTE: does NOT write per-session base files anymore.
    """
    s_dir = Path(session_dir)
    s_dir.mkdir(parents=True, exist_ok=True)

    # Ensure FULL_LAKE_DIR exists (comes from research.io_utils)
    base_full_dir = Path(FULL_LAKE_DIR)
    base_full_dir.mkdir(parents=True, exist_ok=True)
    global_base_file = base_full_dir / "base_data_full.parquet"

    # session-local slice file
    session_base_file = Path(session_dir) / "base_data_full.parquet"

    target_base_file = global_base_file if make_full else session_base_file

    

    if not isinstance(run_cfg, dict) or not run_cfg:
        raise ValueError("prepare_base_data MUST receive a valid run_cfg dictionary from the session snapshot.")

    # dates
    if not run_cfg.get("grid_start_date") or not run_cfg.get("grid_end_date"):
        raise ValueError("run_cfg must include 'grid_start_date' and 'grid_end_date'")

    grid_start = pd.to_datetime(run_cfg["grid_start_date"], utc=True)
    grid_end = pd.to_datetime(run_cfg["grid_end_date"], utc=True)

    padding_days = run_cfg.get("padded_days", 15) 
    padded_start = grid_start - pd.Timedelta(days=padding_days)

    if target_base_file.exists() and not force:
        try:
            if _parquet_has_rows(target_base_file):
                logger.info("Reusing existing base file: %s", target_base_file)
                df_main = pl.read_parquet(str(target_base_file))
                return {"status": "reused", "path": str(target_base_file), "rows": int(df_main.height)}
        except Exception as e:
            logger.debug("Base file exists but could not be read: %s (will rebuild)", e)

    # If we reach here, build the base file (either full or sliced)
    engine = get_engine(db_uri)
    pair = run_cfg.get("pair", "XXBTZUSD")
    market_type = run_cfg.get("market_type", "spot")

    grid_start = pd.to_datetime(run_cfg["grid_start_date"], utc=True)
    grid_end = pd.to_datetime(run_cfg["grid_end_date"], utc=True)
    
    # If requested to build full historical base (rare), read entire table range
    if make_full:
        logger.info("Building global full base_data (make_full=True). This may take time.")
        query = """
            SELECT * FROM df_main
            WHERE pair = %s AND market_type = %s
            ORDER BY time ASC
        """
        df_main = pl.read_database(query=query, connection=engine, execute_options={"parameters":[pair, market_type]})
    else:
        # Build only the padded window needed (fast)
        logger.info("Building global base_data window slice: %s -> %s (padded start %s)", padded_start.isoformat(), grid_end.isoformat(), padded_start.isoformat())
        query = """
            SELECT * FROM df_main
            WHERE pair = %s
              AND market_type = %s
              AND time >= %s
              AND time <= %s
            ORDER BY time ASC
        """
        df_main = pl.read_database(
            query=query,
            connection=engine,
            execute_options={"parameters":[pair, market_type, padded_start.to_pydatetime(), grid_end.to_pydatetime()]}
        )

    if df_main is None or df_main.height == 0:
        raise RuntimeError("Failed to build base data for requested window; no rows returned.")

    # dedupe & ensure time range
    if "time_ns" in df_main.columns:
        df_main = df_main.unique(subset=["time_ns"])
    if "time" in df_main.columns:
        df_main = df_main.filter(pl.col("time").is_between(padded_start, grid_end, closed="left"))

    logger.info("Enforcing chronological sort order on df_main...")
    df_main = df_main.sort("time")

    # atomically write the global base file
    atomic_write_parquet(df_main, global_base_file)

    actual_start = df_main["time"][0]
    actual_end = df_main["time"][-1]
    logger.info(f"✅ Verified Range: {actual_start} TO {actual_end} (Rows: {df_main.height})")
    return {"status": "written", "path": str(target_base_file), "rows": int(df_main.height)}