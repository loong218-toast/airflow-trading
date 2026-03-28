from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

from etl.schema import MASTER_SCHEMA, enforce_schema, get_schema

logger = logging.getLogger(__name__)

master_metrics_schema = MASTER_SCHEMA


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _io_err_log_path() -> Path:
    p = Path(os.getenv("CACHE_TMP_DIR", "/opt/airflow/airflow-trading/data_lake/cache/tmp_cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p / "master_io_errors.log"


def _record_io_error(exc: Exception, ctx: str = "") -> None:
    try:
        p = _io_err_log_path()
        with open(p, "a", encoding="utf8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} ctx={ctx}\n")
            import traceback
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=fh)
            fh.write("\n\n")
    except Exception:
        logger.exception("Failed to write master_io_errors.log")


def _worker_id() -> str:
    return os.getenv("AIRFLOW_MAP_INDEX", os.getenv("MAP_INDEX", "0"))


def _master_schema_fields() -> List[str]:
    return list(master_metrics_schema.keys())


def cast_master_polars_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cast a Polars DataFrame to the canonical master schema.
    Keeps canonical column order and drops extras.
    """
    exprs = []
    for name, pol_dt in master_metrics_schema.items():
        if name in df.columns:
            exprs.append(pl.col(name).cast(pol_dt).alias(name))
        else:
            exprs.append(pl.lit(None).cast(pol_dt).alias(name))
    return df.select(exprs)


def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    """
    Atomically write a Polars DataFrame to `path`.
    """
    path = Path(path)
    _ensure_dir(path.parent)

    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    os.close(fd)
    tmp_p = Path(tmp)

    try:
        df = cast_master_polars_df(df)
        df.write_parquet(str(tmp_p), compression="snappy")
        os.replace(str(tmp_p), str(path))
    finally:
        if tmp_p.exists():
            try:
                tmp_p.unlink()
            except Exception:
                pass


_master_rows_buffer: List[Dict[str, Any]] = []
_MASTER_FLUSH_ROWS = int(os.getenv("MASTER_ROWS_FLUSH", "2000"))
_MASTER_LOCK = threading.Lock()


def buffer_master_row(results_dir: Path, batch_id: int, row: Dict[str, Any], flush_immediate: bool = False) -> None:
    """
    Buffer a canonical master-row and periodically flush it to a worker-local parquet part.
    """
    canonical = {name: row.get(name, None) for name in _master_schema_fields()}
    part_dir = Path(results_dir) / "master_parts" / f"batch_{batch_id:04d}"
    _ensure_dir(part_dir)

    wid = _worker_id()
    stamp = int(time.time() * 1000)

    if flush_immediate:
        out_path = part_dir / f"part_w{wid}_{stamp}_{uuid.uuid4().hex}.parquet"
        try:
            df = pl.DataFrame([canonical], schema=master_metrics_schema)
            df = cast_master_polars_df(df)
            _atomic_write_parquet(df, out_path)
            logger.debug("Wrote single master part -> %s", out_path)
        except Exception as e:
            _record_io_error(e, "write_single_master_part")
            logger.exception("Failed to write immediate master part")
            raise
        return

    with _MASTER_LOCK:
        _master_rows_buffer.append(canonical)
        buf_to_flush = None
        if len(_master_rows_buffer) >= _MASTER_FLUSH_ROWS:
            buf_to_flush = list(_master_rows_buffer)
            _master_rows_buffer.clear()

    if buf_to_flush:
        out_path = part_dir / f"master_part_w{wid}_{stamp}_{uuid.uuid4().hex}.parquet"
        try:
            df = pl.DataFrame(buf_to_flush, schema=master_metrics_schema)
            df = cast_master_polars_df(df)
            _atomic_write_parquet(df, out_path)
            logger.debug("Flushed %d master rows -> %s", len(buf_to_flush), out_path)
        except Exception as e:
            _record_io_error(e, "flush_master_rows")
            logger.exception("Failed to flush master rows buffer")
            raise


def _flush_master_rows_buffer(results_dir: Path, batch_id: int) -> None:
    """
    Force-flush any buffered master rows to disk at worker exit.
    """
    global _master_rows_buffer
    with _MASTER_LOCK:
        if not _master_rows_buffer:
            return
        buf = list(_master_rows_buffer)
        _master_rows_buffer.clear()

    part_dir = Path(results_dir) / "master_parts" / f"batch_{batch_id:04d}"
    _ensure_dir(part_dir)

    wid = _worker_id()
    stamp = int(time.time() * 1000)
    out_path = part_dir / f"master_part_w{wid}_{stamp}_{uuid.uuid4().hex}.parquet"

    try:
        df = pl.DataFrame(buf, schema=master_metrics_schema)
        df = cast_master_polars_df(df)
        _atomic_write_parquet(df, out_path)
        logger.debug("Flushed %d master rows -> %s", len(buf), out_path)
    except Exception as e:
        _record_io_error(e, "flush_master_rows_final")
        logger.exception("Failed to flush master rows buffer (final)")
        raise

__all__ = [
    "master_metrics_schema",
    "buffer_master_row",
    "_flush_master_rows_buffer",
    "_atomic_write_parquet",
    "cast_master_polars_df",
]