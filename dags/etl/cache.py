# etl/cache.py
from __future__ import annotations

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import lru_cache

import polars as pl

from etl.schema import get_schema, enforce_schema

_LOG = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cache_settings() -> Dict[str, Any]:
    data_lake_root = os.getenv("DATA_LAKE_ROOT", "/opt/airflow/airflow-trading/data_lake")
    default_cache_root = Path(data_lake_root) / "cache"

    def _int_env(k: str, default: int) -> int:
        v = os.getenv(k)
        try:
            return int(v) if v is not None else int(default)
        except Exception:
            return int(default)

    def _bool_env(k: str, default: int) -> bool:
        v = os.getenv(k)
        try:
            return bool(int(v)) if v is not None else bool(int(default))
        except Exception:
            return bool(int(default))

    def _str_env(k: str, default: str) -> str:
        return os.getenv(k, default)

    return {
        "DEFAULT_CACHE_ROOT": default_cache_root,
        "CACHE_FLUSH_ROWS": _int_env("CACHE_FLUSH_ROWS", 50000),
        "CACHE_MAX_INMEM_ROWS": _int_env("CACHE_MAX_INMEM_ROWS", 20000),
        "CACHE_USE_STREAMING_MERGE": _bool_env("CACHE_USE_STREAMING_MERGE", 1),
        "TMP_DIR": Path(_str_env("CACHE_TMP_DIR", str(default_cache_root / "tmp_cache"))),
        "PARTS_FLUSH_THRESHOLD": _int_env("CACHE_PARTS_FLUSH_THRESHOLD", 8),
        "GRID_RESUME_IF_POSSIBLE": _str_env("GRID_RESUME_IF_POSSIBLE", "true").lower() not in ("0", "false", "no"),
    }


def _get_cache_root() -> Path:
    return _cache_settings()["DEFAULT_CACHE_ROOT"]


def _get_tmp_dir() -> Path:
    p = _cache_settings()["TMP_DIR"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_cache_flush_rows() -> int:
    return _cache_settings()["CACHE_FLUSH_ROWS"]


def _get_cache_max_inmem_rows() -> int:
    return _cache_settings()["CACHE_MAX_INMEM_ROWS"]


def _get_cache_use_streaming_merge() -> bool:
    return bool(_cache_settings()["CACHE_USE_STREAMING_MERGE"])


def _get_parts_flush_threshold() -> int:
    return int(_cache_settings()["PARTS_FLUSH_THRESHOLD"])


def _resume_enabled() -> bool:
    return bool(_cache_settings()["GRID_RESUME_IF_POSSIBLE"])


def _era_dir_base(months: int, kind: str, era_label: str) -> Path:
    base = _get_cache_root() / f"{int(months)}mo" / kind / f"era_{era_label}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _target_path(kind: str, months: int, era_label: str, regime_id: str) -> Path:
    base = _era_dir_base(months, kind, era_label)
    if kind == "signals":
        return base / f"config_{regime_id}.parquet"
    else:
        return base / f"config_{regime_id}_combos.parquet"


def _worker_part_path(regime_id: str, worker_id: str) -> Path:
    tmp = _get_tmp_dir()
    return tmp / f"config_{regime_id}_batch_{worker_id}.parquet"


def _list_worker_parts_for_config(regime_id: str) -> List[Path]:
    tmp = _get_tmp_dir()
    return sorted(tmp.glob(f"config_{regime_id}_batch_*.parquet"))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_atomic(src_path: Path, dest_path: Path) -> None:
    os.replace(str(src_path), str(dest_path))


# helper to union columns and cast to a target schema (if provided)
def _union_columns_and_cast(dfs: List[pl.DataFrame], kind: str) -> pl.DataFrame:
    """
    Unions multiple DataFrames and forces the final result to match the schema.py definition.
    Uses diagonal concat, then enforce_schema(..., strict=True) to guarantee column order/dtypes.
    """
    if not dfs:
        # Return empty DF with correct schema if no data
        return enforce_schema(None, kind)

    combined = pl.concat(dfs, how="diagonal")
    return enforce_schema(combined, kind, strict=True)


# -----------------------
# Atomic merge (streaming-only)
# -----------------------
def _atomic_merge_files(inputs: List[str], out_path: Path) -> None:
    """
    Streaming (lazy) merge of multiple Parquet files into out_path. Uses a tmp file and atomic replace.
    This avoids loading files into memory.
    """
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    if not inputs:
        raise ValueError("No input files provided for merge")

    tmp_out = out_path.with_suffix(".tmp.parquet")
    try:
        # lazy concat and write out
        pl.scan_parquet(inputs).sink_parquet(str(tmp_out), compression="snappy")
        os.replace(str(tmp_out), str(out_path))
    except Exception as e:
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        _LOG.error("Streaming merge failed for %s. Error: %s", out_path, e)
        raise


# -----------------------
# Unified load function
# -----------------------
def load_cached(kind: str, months: int, era_label: str, regime_id: str) -> Optional[pl.DataFrame]:
    """
    Load cached parquet for either 'signals' or 'backtest'.
    Returns None if file not present.
    Always enforces the canonical schema (strict).
    """
    if kind not in ("signals", "backtest"):
        raise ValueError(f"Unknown kind: {kind}")

    p = _target_path(kind, months, era_label, regime_id)
    if not p.exists():
        return None

    try:
        # 1. Peek at the file's metadata (very fast)
        file_schema = pl.read_parquet_schema(str(p))
        target_schema = get_schema(kind)
        
        # 2. Check: Does the file have every column we currently need?
        required_cols = set(target_schema.keys())
        existing_cols = set(file_schema.keys())
        
        missing_from_file = required_cols - existing_cols
        
        if missing_from_file:
            _LOG.info(
                "Cache Invalidation for %s: Missing new columns %s. Re-generating...", 
                p.name, missing_from_file
            )
            # Delete so the worker knows it must generate a fresh version
            p.unlink(missing_ok=True)
            return None

        # 3. Load if valid
        df = pl.read_parquet(str(p))
        # Ensure correct dtypes in case of historical float/int shifts
        df = enforce_schema(df, kind, strict=True)
        return df

    except Exception as exc:
        _LOG.warning("Cache read error for %s: %s", p, exc)
        p.unlink(missing_ok=True)
        return None

def stage_for_flush(kind: str, months: int, era_label: str, regime_id: str, df_new: pl.DataFrame) -> None:
    """
    WORKER-ONLY WRITE: Each worker writes a unique file. No reading, no merging.
    """
    if df_new is None or df_new.height == 0:
        return

    # 1. Identify unique worker via Airflow environment variables
    worker_id = str(os.getenv("AIRFLOW_MAP_INDEX", "0"))
    
    # 2. Define unique path: config_{id}_batch_{worker}.parquet
    tmp_dir = _get_tmp_dir()
    worker_part = tmp_dir / f"config_{regime_id}_batch_{worker_id}.parquet"

    # 3. Ensure schema is canonical before saving
    from etl.schema import enforce_schema
    df_to_write = enforce_schema(df_new, kind, strict=True)

    try:
        # 4. Atomic Write (Overwrite any existing file for THIS specific worker/config only)
        df_to_write.write_parquet(str(worker_part), compression="snappy")
        _LOG.debug("Worker %s staged %d rows for cfg %s", worker_id, df_to_write.height, regime_id)
    except Exception as e:
        _LOG.error("Worker %s failed to stage cache for cfg %s: %s", worker_id, regime_id, e)
        # We raise here because if staging fails, the final merge will produce incomplete results
        raise


def flush_all_buffers() -> None:
    # For worker-only design flush is a no-op; merging of worker parts into final target should be
    # handled by the coordinator/merge step (outside this module).
    _LOG.debug("flush_all_buffers called (no-op in worker-only design).")


def inspect_cache_root() -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    base = _get_cache_root()
    for cat in sorted(base.glob("*mo")):
        cat_name = cat.name
        out[cat_name] = {}
        for kind in ("signals", "backtest"):
            base_k = cat / kind
            if not base_k.exists():
                out[cat_name][kind] = 0
                continue
            count = sum(1 for _ in base_k.rglob("*.parquet"))
            out[cat_name][kind] = count
    try:
        tmp = _get_tmp_dir()
        tmp_count = len(list(tmp.glob("config_*_batch_*.parquet")))
        out["tmp_parts"] = {"count": tmp_count}
    except Exception:
        pass
    return out

    # --- Compatibility Aliases ---

def load_signals_cached(months: int, era_label: str, regime_id: str) -> Optional[pl.DataFrame]:
    """Compatibility wrapper for load_cached."""
    return load_cached("signals", months, era_label, regime_id)

def load_backtest_cached(months: int, era_label: str, regime_id: str) -> Optional[pl.DataFrame]:
    """Compatibility wrapper for load_cached."""
    return load_cached("backtest", months, era_label, regime_id)

__all__ = [
    "load_cached", 
    "load_signals_cached", 
    "load_backtest_cached", 
    "stage_for_flush", 
    "inspect_cache_root"
]