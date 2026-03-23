from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from collections import defaultdict
import re

import polars as pl
import pyarrow.parquet as pq

from etl.cache import _get_cache_root as _io_get_cache_root
from etl.schema import MASTER_SCHEMA, enforce_schema, get_schema

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# small filesystem helpers
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
    return base / f"config_{regime_id}_combos.parquet"


def _worker_parts_for_config(kind: str, months: int, era_label: str, regime_id: str) -> List[Path]:
    tmp = _tmp_dir()
    pattern = f"{kind}_config_{regime_id}_era_{era_label}_batch_*.parquet"
    return sorted(tmp.glob(pattern))


def _equity_tmp_dir(session_dir: Path) -> Path:
    tmp_dir = session_dir / "equity_partitioned" / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


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
    """
    schema = get_schema(kind)
    exprs: List[pl.Expr] = []

    for col_name, dtype in schema.items():
        if col_name in input_cols:
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
            exprs.append(pl.lit(None).cast(dtype).alias(col_name))

    return exprs


def _master_cast_exprs_for(schema_cols: Set[str]) -> List[pl.Expr]:
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

    all_input_cols = _gather_parquet_schema_names([Path(p) for p in inputs])
    cast_exprs = get_canonical_casts(kind, all_input_cols)
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
# Public: combine_cache
# ---------------------------------------------------------------------
def combine_cache(kind: str, months: int, era_label: str, regime_id: str) -> Optional[Path]:
    """
    Merge all worker temp parts for one (kind, months, era_label, regime_id)
    into the final per-era cache file.
    """
    if kind not in ("signals", "backtest"):
        raise ValueError("kind must be 'signals' or 'backtest'")

    parts = _worker_parts_for_config(kind, months, era_label, regime_id)
    if not parts:
        _LOG.debug(
            "combine_cache: no worker parts found for kind=%s regime=%s era=%s",
            kind, regime_id, era_label
        )
        return None

    target = _target_path(kind, months, era_label, regime_id)
    inputs = [str(p) for p in parts]

    _LOG.info(
        "combine_cache: merging %d worker parts for kind=%s cfg=%s era=%s -> %s",
        len(inputs), kind, regime_id, era_label, target
    )
    merge_parquet_files_streaming(inputs, target, kind)

    for p in parts:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            _LOG.debug("combine_cache: failed to remove worker part %s (ignored)", p)

    return target


import re
from collections import defaultdict

_EQUITY_SHARD_RE = re.compile(
    r"^equity_era_int=(?P<era>\d+)_batch=(?P<batch>\d+)_(?:worker|task)=(?P<unit>\d+)\.parquet$"
)

def _equity_tmp_dir(session_dir: Path) -> Path:
    tmp_dir = session_dir / "equity_partitioned" / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def _equity_parts_for_partition(
    session_dir: Path,
    partition_key: str,
    batch_id: Optional[int] = None,
) -> List[Path]:
    tmp_dir = _equity_tmp_dir(session_dir)

    if batch_id is None:
        candidates = sorted(tmp_dir.glob(f"equity_era_int={partition_key}_batch=*.parquet"))
    else:
        candidates = sorted(tmp_dir.glob(f"equity_era_int={partition_key}_batch={int(batch_id)}_*.parquet"))

    parts: List[Path] = []
    for p in candidates:
        m = _EQUITY_SHARD_RE.match(p.name)
        if not m:
            _LOG.debug("Skipping non-matching equity shard: %s", p.name)
            continue
        parts.append(p)

    return parts


def combine_equity_parts(
    session_dir: Path,
    partition_key: str,
    batch_id: Optional[int] = None,
) -> Optional[Path]:
    """
    Merge worker/task equity shards for one era into a single final equity parquet.
    """
    parts = _equity_parts_for_partition(session_dir, partition_key, batch_id=batch_id)
    if not parts:
        _LOG.debug(
            "combine_equity_parts: no equity shards found for era=%s batch=%s",
            partition_key, batch_id
        )
        return None

    final_dir = session_dir / "equity_partitioned" / f"era_int={partition_key}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"equity_era_int={partition_key}.parquet"

    _LOG.info(
        "combine_equity_parts: merging %d shards for era=%s batch=%s -> %s",
        len(parts), partition_key, batch_id, final_path
    )

    merge_parquet_files_streaming([str(p) for p in parts], final_path, kind="equity")

    merged_schema = pl.read_parquet_schema(str(final_path))
    missing = set(get_schema("equity").keys()) - set(merged_schema.keys())
    if missing:
        raise RuntimeError(f"Post-merge: missing columns in final equity {missing}")

    manifest = {"parts": [str(p) for p in parts], "final": str(final_path)}
    final_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))

    for p in parts:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            _LOG.debug("combine_equity_parts: failed to remove shard %s (ignored)", p)

    return final_path


def combine_all_equity_parts(
    session_dir: str | Path,
    batch_id: Optional[int] = None,
) -> List[Path]:
    """
    Merge every era found in session_dir/equity_partitioned/_tmp.
    Safe for low RAM because each era is merged separately with streaming.
    """
    session_dir = Path(session_dir)
    tmp_dir = _equity_tmp_dir(session_dir)

    if batch_id is None:
        candidates = sorted(tmp_dir.glob("equity_era_int=*_batch=*.parquet"))
    else:
        candidates = sorted(tmp_dir.glob(f"equity_era_int=*_batch={int(batch_id)}_*.parquet"))

    eras: Dict[str, List[Path]] = defaultdict(list)

    for p in candidates:
        m = _EQUITY_SHARD_RE.match(p.name)
        if not m:
            continue
        era = m.group("era")
        if batch_id is not None and int(m.group("batch")) != int(batch_id):
            continue
        eras[era].append(p)

    outputs: List[Path] = []
    for era in sorted(eras.keys()):
        out = combine_equity_parts(session_dir=session_dir, partition_key=era, batch_id=batch_id)
        if out is not None:
            outputs.append(out)

    _LOG.info("combine_all_equity_parts: merged %d era files", len(outputs))
    return outputs

# ---------------------------------------------------------------------
# Master merge
# ---------------------------------------------------------------------
def combine_results_to_master(session_dir: str) -> Dict[str, Any]:
    """
    Streaming-only combine for final master_metrics.parquet.
    This function only handles master metrics.
    Equity merging is handled separately by combine_equity_parts().
    """
    session_dir = Path(session_dir)
    results_dir = session_dir / "results"
    master_metrics_path = session_dir / "master_metrics.parquet"

    batch_files = sorted(results_dir.glob("batch_*_master_metrics.parquet")) if results_dir.exists() else []
    nonempty = []
    for p in batch_files:
        try:
            meta = pq.ParquetFile(str(p)).metadata
            if meta.num_rows > 0:
                nonempty.append(p)
        except Exception as e:
            _LOG.error("Skipping corrupted master batch file %s: %s", p.name, e)

    if nonempty:
        _LOG.info("Merging %d batch masters...", len(nonempty))
        tmp_master = master_metrics_path.with_suffix(".tmp.parquet")
        try:
            pl.scan_parquet([str(p) for p in nonempty]).sink_parquet(str(tmp_master), compression="snappy")
            os.replace(str(tmp_master), str(master_metrics_path))
        except Exception:
            if tmp_master.exists():
                tmp_master.unlink(missing_ok=True)
            raise

        _LOG.info("✅ Successfully merged %d batches into master_metrics", len(nonempty))
    else:
        df_master = pl.DataFrame([], schema=MASTER_SCHEMA)
        df_master.write_parquet(master_metrics_path)
        _LOG.info("⚠️ No batch data found. Created empty master_metrics.")
        return {"status": "complete_no_batch_data", "path": str(master_metrics_path), "rows": 0}

    df_master = pl.read_parquet(master_metrics_path)
    df_master = enforce_schema(df_master, "master", strict=True)
    df_master.write_parquet(master_metrics_path)

    return {"status": "complete_master_only", "path": str(master_metrics_path), "rows": df_master.height}