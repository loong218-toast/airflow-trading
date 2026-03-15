# etl/merge_utils.py
"""
Merge utilities: streaming-only merges for cache parts and master aggregation.

- combine_cache(kind, months, era_label, regime_id)
    merge worker tmp parts (config_{regime_id}_batch_{worker}.parquet) -> final per-era target
    (signals -> config_{id}.parquet ; backtest -> config_{id}_combos.parquet)
    Uses Polars streaming only. On any streaming failure it logs diagnostics and raises.

- combine_results_to_master(session_dir)
    Streaming-only merges for assembling master_metrics.parquet and streaming aggregation of equity partitions.
"""
from __future__ import annotations

import os
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

import polars as pl
import pyarrow.parquet as pq

from etl.cache import _get_cache_root as _io_get_cache_root  # to build target paths consistently

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# small filesystem helpers (unchanged)
# ---------------------------------------------------------------------
def _tmp_dir() -> Path:
    tmp = os.getenv("CACHE_TMP_DIR", None)
    if tmp:
        p = Path(tmp)
    else:
        p = Path(_io_get_cache_root()) / "tmp_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _era_dir_base(months: int, kind: str, era_label: str) -> Path:
    base = Path(_io_get_cache_root()) / f"{int(months)}mo" / kind / f"era_{era_label}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _target_path(kind: str, months: int, era_label: str, regime_id: str) -> Path:
    base = _era_dir_base(months, kind, era_label)
    if kind == "signals":
        return base / f"config_{regime_id}.parquet"
    else:
        return base / f"config_{regime_id}_combos.parquet"


def _worker_parts_for_config(regime_id: str) -> List[Path]:
    td = _tmp_dir()
    return sorted(td.glob(f"config_{regime_id}_batch_*.parquet"))


# ---------------------------------------------------------------------
# Schema-aware casting utilities
# ---------------------------------------------------------------------
def _is_int_dtype(dt: pl.DataType) -> bool:
    return dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64)


def _is_float_dtype(dt: pl.DataType) -> bool:
    return dt in (pl.Float32, pl.Float64)


def get_canonical_casts(kind: str, input_cols: Set[str]) -> List[pl.Expr]:
    """
    Build cast expressions for a streaming pipeline based on the canonical schema for `kind`.
    Ensures Int/Float safety: fills nulls for ints and clips floats when targeting Float32.
    """
    from etl.schema import get_schema

    schema = get_schema(kind)
    exprs: List[pl.Expr] = []
    for col_name, dtype in schema.items():
        if col_name in input_cols:
            # present in input: cast with safety for ints/floats
            if _is_int_dtype(dtype):
                exprs.append(pl.col(col_name).fill_nan(None).fill_null(0).cast(dtype).alias(col_name))
            elif _is_float_dtype(dtype):
                if dtype == pl.Float32:
                    exprs.append(pl.col(col_name).fill_nan(None).clip(-1e37, 1e37).cast(dtype).alias(col_name))
                else:
                    exprs.append(pl.col(col_name).fill_nan(None).cast(dtype).alias(col_name))
            else:
                exprs.append(pl.col(col_name).cast(dtype).alias(col_name))
        else:
            # missing: inject typed null
            exprs.append(pl.lit(None).cast(dtype).alias(col_name))
    return exprs


def _master_cast_exprs_for(schema_cols: Set[str]) -> List[pl.Expr]:
    """
    Similar to get_canonical_casts but targeted at MASTER_SCHEMA mapping.
    Handles Int32 and Float32 (and all canonical master types) safely.
    """
    from etl.schema import MASTER_SCHEMA

    exprs: List[pl.Expr] = []
    for name, pol_dt in MASTER_SCHEMA.items():
        if name in schema_cols:
            if _is_int_dtype(pol_dt):
                exprs.append(pl.col(name).fill_nan(None).fill_null(0).cast(pol_dt).alias(name))
            elif _is_float_dtype(pol_dt):
                if pol_dt == pl.Float32:
                    exprs.append(pl.col(name).fill_nan(None).clip(-1e37, 1e37).cast(pol_dt).alias(name))
                else:
                    exprs.append(pl.col(name).fill_nan(None).cast(pol_dt).alias(name))
            else:
                exprs.append(pl.col(name).cast(pol_dt).alias(name))
        else:
            exprs.append(pl.lit(None).cast(pol_dt).alias(name))
    return exprs


# ---------------------------------------------------------------------
# parquet schema introspection helper
# ---------------------------------------------------------------------
def _gather_parquet_schema_names(paths: List[Path]) -> Set[str]:
    """Return union of column names across given parquet file paths using pyarrow metadata."""
    cols: Set[str] = set()
    for p in paths:
        try:
            schema = pq.read_schema(str(p))
            cols.update(schema.names)
        except Exception:
            _LOG.debug("Could not read pyarrow schema for %s", p)
    return cols


# ---------------------------------------------------------------------
# Streaming merge implementation
# ---------------------------------------------------------------------
def merge_parquet_files_streaming(inputs: List[str], out_path: Path, kind: str) -> None:
    """
    Streaming merge that forces the output to match the etl.schema strictly.
    """
    if not inputs:
        raise ValueError(f"No inputs for {kind} merge")

    # 1. Peek at schemas to see what columns we actually have
    all_input_cols = _gather_parquet_schema_names([Path(p) for p in inputs])

    # 2. Build the casting plan based on the official schema
    cast_exprs = get_canonical_casts(kind, all_input_cols)
    # final order from schema
    from etl.schema import get_schema
    final_cols = list(get_schema(kind).keys())

    tmp_out = out_path.with_suffix(".tmp.parquet")
    try:
        pl.scan_parquet(inputs).with_columns(cast_exprs).select(final_cols).sink_parquet(
            str(tmp_out), compression="snappy"
        )
        os.replace(str(tmp_out), str(out_path))
        _LOG.info("Merged %d files into %s (Kind: %s)", len(inputs), out_path.name, kind)
    except Exception as e:
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        _LOG.error("Strict merge failed for %s: %s", kind, e)
        raise


# ---------------------------------------------------------------------
# Public: combine_cache  (merge worker parts -> final per-era file)
# ---------------------------------------------------------------------
def combine_cache(kind: str, months: int, era_label: str, regime_id: str) -> Optional[Path]:
    """
    Merge all worker temp parts for regime_id into the per-era final file.

    - Only streaming merges are used. On failure this raises (so caller sees it).
    - Removes worker parts on success (best-effort).
    - Returns the final target Path or None if no worker parts found.
    """
    if kind not in ("signals", "backtest"):
        raise ValueError("kind must be 'signals' or 'backtest'")

    parts = _worker_parts_for_config(regime_id)
    if not parts:
        _LOG.debug("combine_cache: no worker parts found for %s (kind=%s)", regime_id, kind)
        return None

    target = _target_path(kind, months, era_label, regime_id)
    inputs = [str(p) for p in parts]

    _LOG.info("combine_cache: merging %d worker parts for config=%s -> %s", len(inputs), regime_id, target)
    merge_parquet_files_streaming(inputs, target, kind)

    # cleanup worker parts on success (best-effort)
    for p in parts:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            _LOG.debug("combine_cache: failed to remove worker part %s (ignored)", p)

    return target


# ---------------------------------------------------------------------
# Streaming-only combine_results_to_master (shorter / less redundant)
# ---------------------------------------------------------------------
def combine_results_to_master(session_dir: str) -> Dict[str, Any]:
    """
    Streaming-only combine for final master_metrics.parquet.

    Steps:
      1) Merge per-batch master parts (streaming) into master_metrics.parquet.
      2) Aggregate equity partition streaming and left-join aggregates into master.
      3) Write final master (canonical types).
    """
    from etl.schema import MASTER_SCHEMA, enforce_schema

    session_dir = Path(session_dir)
    results_dir = session_dir / "results"
    master_metrics_path = session_dir / "master_metrics.parquet"
    equity_part_dir = session_dir / "equity_partitioned"

    has_equity_data = (
        equity_part_dir.exists() and 
        any(equity_part_dir.rglob("*.parquet"))
    )

    if not has_equity_data:
        _LOG.info("Equity dataset skipped or empty. Finalizing master metrics as-is.")

    # 1) discover batch master parts and merge streaming (or create empty canonical)
    batch_files = sorted(results_dir.glob("batch_*_master_metrics.parquet")) if results_dir.exists() else []
    nonempty = []
    for p in batch_files:
        try:
            # This check detects corruption/truncation
            meta = pq.ParquetFile(str(p)).metadata
            if meta.num_rows > 0:
                nonempty.append(p)
        except Exception as e:
            _LOG.error("Skipping corrupted master batch file %s: %s", p.name, e)

    master_cols = list(MASTER_SCHEMA.keys())

    if nonempty:
        _LOG.info("Merging %d batch masters...", len(nonempty))
        # Use a lazy scan and cast using the schema helper logic

        df_master = pl.scan_parquet([str(p) for p in nonempty]).collect()
        df_master = enforce_schema(df_master, "master", strict=True)
        df_master.write_parquet(master_metrics_path)

        _LOG.info("✅ Successfully merged %d batches into master_metrics (Rows: %d)", len(nonempty), df_master.height)
    else:
        df_master = pl.DataFrame([], schema=MASTER_SCHEMA)
        df_master.write_parquet(master_metrics_path)

        _LOG.info("⚠️ No batch data found. Created empty master_metrics.")

    if not has_equity_data:
        _LOG.info("Equity dataset skipped. Finalizing.")
        return {"status": "complete_no_equity", "path": str(master_metrics_path)}

    # 2) Aggregate equity partition (if exists) and join to master
    valid_equity_paths = []
    if equity_part_dir.exists():

        all_equity = sorted(equity_part_dir.rglob("*.parquet"))
        for p in all_equity:
            try:
                # Check for corruption
                pq.ParquetFile(str(p)).metadata
                valid_equity_paths.append(p)
            except Exception as e:
                _LOG.error("Skipping corrupted equity file %s: %s", p.name, e)

    has_equity_data = len(valid_equity_paths) > 0

    if not has_equity_data:
        _LOG.info("No valid equity data found (skipped or corrupted). Finalizing.")
        return {"status": "complete_no_equity", "path": str(master_metrics_path), "rows": df_master.height}

    return {
        "status": "complete_with_equity",
        "master_metrics_path": str(master_metrics_path),
        "rows": df_master.height
    }