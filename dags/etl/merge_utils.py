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
    from etl.schema import MASTER_SCHEMA

    session_dir = Path(session_dir)
    results_dir = session_dir / "results"
    master_metrics_path = session_dir / "master_metrics.parquet"
    equity_part_dir = session_dir / "equity_partitioned"

    # 1) discover batch master parts and merge streaming (or create empty canonical)
    batch_files = sorted(results_dir.glob("batch_*_master_metrics.parquet")) if results_dir.exists() else []
    nonempty = []
    for p in batch_files:
        try:
            if p.exists() and pq.ParquetFile(str(p)).metadata.num_rows > 0:
                nonempty.append(p)
        except Exception:
            # If metadata can't be read, still include for attempt
            nonempty.append(p)

    master_cols = list(MASTER_SCHEMA.keys())

    if nonempty:
        _LOG.info("combine_results_to_master: merging %d batch masters -> %s", len(nonempty), master_metrics_path.name)
        try:
            schema_cols = _gather_parquet_schema_names(nonempty)
            casts = _master_cast_exprs_for(schema_cols)
            pl.scan_parquet([str(p) for p in nonempty]).with_columns(casts).select(master_cols).sink_parquet(
                str(master_metrics_path), compression="snappy"
            )
        except Exception as e:
            _LOG.exception("combine_results_to_master: streaming merge of batch masters FAILED: %s", e)
            # diagnostics: list schemas for debugging
            try:
                schemas = []
                for p in nonempty:
                    try:
                        s = pq.read_schema(str(p))
                        schemas.append((str(p), list(s.names)))
                    except Exception as ex:
                        schemas.append((str(p), f"schema-read-failed: {ex}"))
                _LOG.error("Master batch file schemas: %s", schemas)
            except Exception:
                _LOG.exception("Failed to gather master batch schemas for diagnostics.")
            raise RuntimeError("Streaming merge of batch masters failed; aborting.") from e
    else:
        # write an empty canonical master
        empty_df = pl.DataFrame({k: [] for k in MASTER_SCHEMA.keys()})
        try:
            empty_df = empty_df.select([pl.lit(None).cast(dt).alias(k) for k, dt in MASTER_SCHEMA.items()])
        except Exception:
            pass
        empty_df.write_parquet(str(master_metrics_path), compression="snappy")
        _LOG.info("combine_results_to_master: wrote empty master_metrics.parquet (no batch files).")

    # 2) Aggregate equity partition (if exists) and join to master
    final_master = pl.read_parquet(str(master_metrics_path))
    if equity_part_dir.exists():
        try:
            _LOG.info("combine_results_to_master: aggregating equity partition (streaming).")
            equity_paths = sorted(Path(equity_part_dir).rglob("*.parquet"))
            if not equity_paths:
                _LOG.info("No equity parquet files found under equity_partitioned.")
            else:
                # inspect columns present across equity files to decide grouping strategy
                equity_cols = _gather_parquet_schema_names(equity_paths)
                has_sl_tp = ("SL" in equity_cols) and ("TP" in equity_cols)
                # choose grouping keys: prefer full (regime_id, SL, TP, side, era_int) else fallback to (regime_id, side, era_int)
                preferred_keys = ["regime_id", "SL", "TP", "side", "era_int"]
                fallback_keys = ["regime_id", "side", "era_int"]
                group_keys = preferred_keys if has_sl_tp else fallback_keys

                # streaming aggregation
                q = pl.scan_parquet([str(p) for p in equity_paths])
                # ensure pnl_pct/equity exist so aggregations don't fail
                if "pnl_pct" not in equity_cols:
                    _LOG.warning("pnl_pct missing from equity files; win_pos will be computed as 0.")
                    q = q.with_columns(pl.lit(0.0).alias("pnl_pct"))
                if "equity" not in equity_cols and "balance" in equity_cols:
                    q = q.with_columns(pl.col("balance").alias("equity"))

                agg_exprs = [
                    pl.count().alias("total_pos"),
                    (pl.col("pnl_pct") > 0).sum().alias("win_pos"),
                    pl.col("equity").last().alias("balance"),
                ]
                equity_agg = q.group_by(group_keys).agg(agg_exprs).collect()

                # join into master: ensure regime_id is always part of the join key
                master_join_on = ["regime_id", "era_int", "side"]
                # if equity_agg includes SL/TP and master has SL/TP, join on those also
                if has_sl_tp and set(["SL", "TP"]).issubset(set(final_master.columns)):
                    master_join_on = ["regime_id", "SL", "TP", "side", "era_int"]
                    # ensure equity_agg has the same order/names
                else:
                    # fallback: equity_agg aggregated without SL/TP -> collapse aggregates per regime/side/era
                    # To avoid duplicate columns and mismatch, drop SL/TP from equity_agg if present
                    for c in ("SL", "TP"):
                        if c in equity_agg.columns:
                            equity_agg = equity_agg.drop(c)

                # drop existing aggregate cols on master (to replace with aggregated results)
                for c in ("total_pos", "win_pos", "balance"):
                    if c in final_master.columns:
                        final_master = final_master.drop(c)

                joined = final_master.join(equity_agg, on=master_join_on, how="left")

                # fill missing aggregate values and cast to canonical master types
                # create expressions for casting and fill defaults
                from etl.schema import MASTER_SCHEMA
                cast_exprs = []
                for k, dt in MASTER_SCHEMA.items():
                    if k in joined.columns:
                        # ensure aggregate columns filled with defaults
                        if k == "total_pos" or k == "win_pos":
                            cast_exprs.append(pl.col(k).fill_null(0).cast(dt).alias(k))
                        elif k == "balance" or k == "max_drawdown":
                            cast_exprs.append(pl.col(k).fill_null(pl.lit(float("nan"))).cast(dt).alias(k))
                        else:
                            cast_exprs.append(pl.col(k).cast(dt).alias(k))
                    else:
                        # inject nulls if missing
                        cast_exprs.append(pl.lit(None).cast(dt).alias(k))

                final_master = joined.with_columns(cast_exprs).select(list(MASTER_SCHEMA.keys()))
        except Exception as e:
            _LOG.exception("combine_results_to_master: equity aggregation/join FAILED: %s", e)
            raise RuntimeError("Equity aggregation/join failed (streaming) - aborting.") from e

    # 3) Write final master metrics (overwrite) - canonical types already applied
    try:
        final_master.write_parquet(str(master_metrics_path), compression="snappy")
    except Exception as e:
        _LOG.exception("combine_results_to_master: failed writing final master: %s", e)
        raise RuntimeError("Failed to write final master_metrics.parquet") from e

    # counts (best-effort)
    metrics_rows = 0
    try:
        if master_metrics_path.exists():
            metrics_rows = int(pq.ParquetFile(str(master_metrics_path)).metadata.num_rows)
    except Exception:
        metrics_rows = 0

    equity_rows = 0
    if Path(equity_part_dir).exists():
        try:
            q = pl.scan_parquet(f"{equity_part_dir}/**/*.parquet")
            equity_rows = int(q.select(pl.count()).collect().item())
        except Exception:
            equity_rows = 0

    _LOG.info("combine_results_to_master: complete: master_metrics rows=%d, equity_rows=%d", metrics_rows, equity_rows)
    return {
        "metrics_rows": metrics_rows,
        "equity_rows": equity_rows,
        "master_metrics_path": str(master_metrics_path),
        "equity_partition_dir": str(equity_part_dir),
    }