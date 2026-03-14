# etl/io_utils.py
import os
import time
import tempfile
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import math
import datetime

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from etl.master_io_utils import (
    buffer_master_row,
    _atomic_write_parquet,
    merge_parquet_files_fast,
    _flush_master_rows_buffer,
    # cast_master_polars_df used as helper
    cast_master_polars_df,
)
from etl.schema import DF_MAIN_SCHEMA, MASTER_SCHEMA, enforce_schema

logger = logging.getLogger(__name__)

# full lake partitioning root (year/month)
FULL_LAKE_DIR = Path(os.getenv("DATA_LAKE_ROOT", "/opt/airflow/airflow-trading/data_lake")) / "base_data_full"
MANIFEST_FILE = FULL_LAKE_DIR / "_manifest.json"

# small helpers
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _io_err_log_path() -> Path:
    p = Path(os.getenv("CACHE_TMP_DIR", "/opt/airflow/airflow-trading/data_lake/cache/tmp_cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p / "io_utils_errors.log"

def _record_io_error(exc: Exception, ctx: str = ""):
    try:
        p = _io_err_log_path()
        with open(p, "a", encoding="utf8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} ctx={ctx}\n")
            import traceback
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=fh)
            fh.write("\n\n")
    except Exception:
        logger.exception("Failed to write io_utils_errors.log")

# ---- Partition helpers ----
def _ensure_full_lake_dir():
    _ensure_dir(FULL_LAKE_DIR)

def _partition_path_for(yr: int, mo: int) -> Path:
    return FULL_LAKE_DIR / f"year={yr}" / f"month={mo:02d}"

def _list_partition_files_for_month(yr: int, mo: int) -> List[Path]:
    p = _partition_path_for(yr, mo)
    if not p.exists():
        return []
    return sorted([x for x in p.glob("*.parquet") if x.suffix in (".parquet", "") or x.name.endswith(".parquet")])

def _get_year_month_iter(start_dt: datetime.datetime, end_dt: datetime.datetime) -> List[Tuple[int,int]]:
    out = []
    cur = datetime.date(start_dt.year, start_dt.month, 1)
    last = datetime.date(end_dt.year, end_dt.month, 1)
    while cur <= last:
        out.append((cur.year, cur.month))
        if cur.month == 12:
            cur = datetime.date(cur.year + 1, 1, 1)
        else:
            cur = datetime.date(cur.year, cur.month + 1, 1)
    return out

# ---- Write partition (atomic) ----
def _write_partition(df: pl.DataFrame, year: int, month: int) -> Path:
    d = _partition_path_for(year, month)
    _ensure_dir(d)
    fname = f"part_{int(time.time())}_{uuid.uuid4().hex}.parquet"
    tmp = d / (fname + ".tmp")
    out = d / fname
    if "time_ns" in df.columns:
        df = df.sort("time_ns")
    df.write_parquet(str(tmp), compression="snappy")
    os.replace(str(tmp), str(out))
    return out

# ---- Update full lake incremental (skip overlapping months) ----
def update_full_lake_incremental(df_new: pl.DataFrame) -> List[Path]:
    if df_new is None or df_new.height == 0:
        return []

    _ensure_full_lake_dir()

    # read manifest first to know last_ns (if exists)
    last_ns = 0
    try:
        if MANIFEST_FILE.exists():
            with open(MANIFEST_FILE, "r", encoding="utf8") as fh:
                last_ns = int(json.load(fh).get("last_time_ns", 0) or 0)
    except Exception:
        last_ns = 0

    if "time" not in df_new.columns:
        raise RuntimeError("df_new must contain 'time' column")

    tmp = df_new.with_columns([
        pl.col("time").dt.year().alias("_yr"),
        pl.col("time").dt.month().alias("_mo")
    ])
    parts_written = []
    for (yr, mo), sub in tmp.groupby(["_yr", "_mo"]):
        yr_i = int(yr)
        mo_i = int(mo)
        pdir = _partition_path_for(yr_i, mo_i)
        # skip if manifest last_ns indicates no need to write this month
        try:
            min_ns = int(sub.select(pl.col("time_ns")).min()["time_ns"])
        except Exception:
            min_ns = None
        if pdir.exists() and any(pdir.glob("*.parquet")) and last_ns and min_ns and min_ns <= last_ns:
            logger.debug("Skipping existing partition %s/%02d (already has data till manifest)", yr_i, mo_i)
            continue
        sub = sub.drop(["_yr", "_mo"])
        out = _write_partition(sub, yr_i, mo_i)
        parts_written.append(out)

    # update manifest with new max time_ns
    try:
        last_ns_new = int(df_new.select(pl.col("time_ns")).max()["time_ns"])
    except Exception:
        last_ns_new = None
    if last_ns_new:
        tmpm = FULL_LAKE_DIR / f".manifest.tmp.{uuid.uuid4().hex}"
        try:
            with open(tmpm, "w", encoding="utf8") as fh:
                fh.write(json.dumps({"last_time_ns": int(last_ns_new), "updated": int(time.time())}))
            os.replace(str(tmpm), str(MANIFEST_FILE))
        except Exception as e:
            logger.warning("Could not update manifest: %s", e)

    return parts_written

# ---- Read slice from full lake (only relevant partition files) ----
def read_slice_from_full_lake(start_dt: datetime.datetime, end_dt: datetime.datetime, cols: List[str]) -> pl.DataFrame:
    _ensure_full_lake_dir()

    if start_dt.tzinfo is not None:
        start_dt = start_dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    if end_dt.tzinfo is not None:
        end_dt = end_dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)

    months = _get_year_month_iter(start_dt, end_dt)
    files = []
    for (yr, mo) in months:
        files += _list_partition_files_for_month(yr, mo)
    if not files:
        return pl.DataFrame({}, schema={})

    existing_cols = set()
    for f in files:
        try:
            sch = pq.read_schema(str(f))
            existing_cols.update(sch.names)
        except Exception:
            pass

    sel_cols = [c for c in cols if c in existing_cols]
    # always ensure we can filter on time/time_ns
    for extra in ("time", "time_ns"):
        if extra in existing_cols and extra not in sel_cols:
            sel_cols.append(extra)

    if not sel_cols:
        raise RuntimeError("No requested columns present in full lake partitions")

    files_str = [str(f) for f in files]
    try:
        q = pl.scan_parquet(files_str)
        df = q.select(sel_cols).filter((pl.col("time") >= start_dt) & (pl.col("time") < end_dt)).collect().unique(subset=["time_ns"])
        # attempt to cast to DF_MAIN_SCHEMA for safety if "time" present
        try:
            df = enforce_schema(df, "df_main")
        except Exception:
            # not fatal; return as-is
            pass
        # final select to return only requested cols (if present)
        final_cols = [c for c in cols if c in df.columns]
        if final_cols:
            df = df.select(final_cols)
        return df
    except Exception as e:
        _record_io_error(e, "read_slice_from_full_lake.collect")
        logger.exception("Failed to read slice from full lake: %s", e)
        raise

# ---- Atomic pyarrow table write ----
def _write_pa_table_atomic(table: pa.Table, out_path: Path):
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{uuid.uuid4().hex}")
    try:
        pq.write_table(table, str(tmp), compression="snappy")
        os.replace(str(tmp), str(out_path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

# ---- Merge helper (fast / chunked fallback) ----
def merge_parquet_files_fast(paths: List[Path], out_path: Path, chunk_size: int = 64):
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    if not paths:
        raise ValueError("No input paths to merge_parquet_files_fast")

    srcs = [str(p) for p in paths]
    try:
        pl.scan_parquet(srcs).sink_parquet(str(out_path), compression="snappy")
        return
    except Exception as e_stream:
        logger.warning("Streaming merge failed (%s). Falling back to chunked collect.", e_stream)
        _record_io_error(e_stream, "streaming_merge_failed")

    tmp_chunks: List[Path] = []
    try:
        for i in range(0, len(srcs), chunk_size):
            chunk = srcs[i: i + chunk_size]
            tmp_chunk = out_path.with_name(f"{out_path.stem}.chunk{i:04d}.parquet")
            df_chunk = pl.scan_parquet(chunk).collect()
            # If merging master-like files try to align to canonical master schema
            try:
                df_chunk = cast_master_polars_df(df_chunk)
            except Exception:
                pass
            fd, tmp = tempfile.mkstemp(prefix=tmp_chunk.name, dir=str(tmp_chunk.parent))
            os.close(fd)
            try:
                df_chunk.write_parquet(tmp, compression="snappy")
                os.replace(tmp, str(tmp_chunk))
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
            tmp_chunks.append(tmp_chunk)
        pl.scan_parquet([str(p) for p in tmp_chunks]).sink_parquet(str(out_path), compression="snappy")
    except Exception as e_chunk:
        _record_io_error(e_chunk, "chunked_merge_failed")
        logger.exception("Chunked merge failed: %s", e_chunk)
        raise
    finally:
        for p in tmp_chunks:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

# ---- Read helpers ----
def read_master_parquet_casted(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(str(path))
    try:
        df = enforce_schema(df, "master")
    except Exception:
        pass
    return df

# small helpers for schema mapping (pyarrow -> polars)
def _pa_to_polars_dtype(pa_type: pa.DataType):
    if pa.types.is_int8(pa_type): return pl.Int8
    if pa.types.is_int16(pa_type): return pl.Int16
    if pa.types.is_int32(pa_type): return pl.Int32
    if pa.types.is_int64(pa_type): return pl.Int64
    if pa.types.is_float32(pa_type): return pl.Float32
    if pa.types.is_float64(pa_type): return pl.Float64
    if pa.types.is_boolean(pa_type): return pl.Boolean
    if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type): return pl.Utf8
    return pl.Utf8

__all__ = [
    "FULL_LAKE_DIR", "MANIFEST_FILE",
    "_ensure_full_lake_dir", "update_full_lake_incremental",
    "read_slice_from_full_lake", "_write_pa_table_atomic",
    "merge_parquet_files_fast", "_record_io_error", "read_master_parquet_casted"
]