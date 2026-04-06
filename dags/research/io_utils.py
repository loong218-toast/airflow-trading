# research/io_utils.py
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

from common.schema import DF_MAIN_SCHEMA, MASTER_SCHEMA, enforce_schema

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

def atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    os.close(fd)
    tmp_p = Path(tmp)

    try:
        df.write_parquet(str(tmp_p), compression="snappy")
        os.replace(str(tmp_p), str(path))
    finally:
        if tmp_p.exists():
            try:
                tmp_p.unlink()
            except Exception:
                pass


__all__ = [
    "FULL_LAKE_DIR",
    "atomic_write_parquet",
    "read_slice_from_full_lake",
    "_ensure_full_lake_dir"
]