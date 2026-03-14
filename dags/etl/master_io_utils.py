# etl/master_io_utils.py
"""
Polars-based Master-level IO utilities.

- master_metrics_schema: canonical schema (dict of polars dtypes) is imported from etl.schema
- buffer_master_row(results_dir, batch_id, row, flush_immediate=False)
- _flush_master_rows_buffer(results_dir, batch_id)
- _atomic_write_parquet(df, path)
- merge_parquet_files_fast(paths, out_path, chunk_size=64)  # streaming-first, bounded chunk fallback

Workers should write only their worker-unique parts via buffer_master_row. Merging is performed by the combine step.
"""
from __future__ import annotations

import os
import time
import tempfile
import uuid
import threading
import logging
from pathlib import Path
from typing import List, Dict, Any

import polars as pl

from etl.schema import MASTER_SCHEMA  # centralized Polars dtype dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Canonical master metrics schema (polars) - imported
# ---------------------------------------------------------------------
master_metrics_schema = MASTER_SCHEMA  # alias for backward compatibility


def _master_schema_fields() -> List[str]:
    return list(master_metrics_schema.keys())


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _io_err_log_path() -> Path:
    p = Path(os.getenv("CACHE_TMP_DIR", "/opt/airflow/airflow-trading/data_lake/cache/tmp_cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p / "master_io_errors.log"


def _record_io_error(exc: Exception, ctx: str = ""):
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
    """Return worker id from environment (AIRFLOW_MAP_INDEX preferred)."""
    return os.getenv("AIRFLOW_MAP_INDEX", os.getenv("MAP_INDEX", "0"))


# ---------------------------------------------------------------------
# Atomic polars dataframe write
# ---------------------------------------------------------------------
def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    """
    Atomically write a Polars DataFrame to `path` using a temp file + os.replace.
    """
    path = Path(path)
    _ensure_dir(path.parent)
    # create temp file in same directory to ensure atomic replace works
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


# ---------------------------------------------------------------------
# Casting / normalization
# ---------------------------------------------------------------------
def cast_master_polars_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cast a Polars DataFrame to canonical master_metrics_schema.
    - Keeps canonical column order.
    - Adds missing columns with nulls of the appropriate polars dtype.
    - Drops extras.
    """
    exprs = []
    for name, pol_dt in master_metrics_schema.items():
        if name in df.columns:
            exprs.append(pl.col(name).cast(pol_dt).alias(name))
        else:
            exprs.append(pl.lit(None).cast(pol_dt).alias(name))
    # select in canonical order; drop extras
    return df.select(exprs)


# ---------------------------------------------------------------------
# Master rows buffer & atomic flush (worker-local)
# ---------------------------------------------------------------------
_master_rows_buffer: List[Dict[str, Any]] = []
_MASTER_FLUSH_ROWS = int(os.getenv("MASTER_ROWS_FLUSH", "2000"))
_MASTER_LOCK = threading.Lock()


def buffer_master_row(results_dir: Path, batch_id: int, row: Dict[str, Any], flush_immediate: bool = False) -> None:
    """
    Buffer a canonical master-row (dict). Periodically flushes to
    results_dir/master_parts/batch_{batch_id:04d}/ as atomic Polars parquet parts.

    Each flush writes parts with a worker-unique filename so merges can be
    performed later by a combiner step (workers do not merge).
    """
    canonical = {name: row.get(name, None) for name in _master_schema_fields()}
    part_dir = Path(results_dir) / "master_parts" / f"batch_{batch_id:04d}"
    _ensure_dir(part_dir)

    wid = _worker_id()
    stamp = int(time.time() * 1000)

    if flush_immediate:
        out_path = part_dir / f"part_w{wid}_{stamp}_{uuid.uuid4().hex}.parquet"
        try:
            df = pl.DataFrame([canonical], schema=MASTER_SCHEMA)
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
            df = pl.DataFrame(buf_to_flush, schema=MASTER_SCHEMA)
            df = cast_master_polars_df(df)
            _atomic_write_parquet(df, out_path)
            logger.debug("Flushed %d master rows -> %s", len(buf_to_flush), out_path)
        except Exception as e:
            _record_io_error(e, "flush_master_rows")
            logger.exception("Failed to flush master rows buffer")
            raise


def _flush_master_rows_buffer(results_dir: Path, batch_id: int) -> None:
    """
    Force-flush any buffered master rows to disk. Should be called at worker exit.
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
        df = pl.DataFrame(buf, schema=MASTER_SCHEMA)
        df = cast_master_polars_df(df)
        _atomic_write_parquet(df, out_path)
        logger.debug("Flushed %d master rows -> %s", len(buf), out_path)
    except Exception as e:
        _record_io_error(e, "flush_master_rows_final")
        logger.exception("Failed to flush master rows buffer (final)")
        raise


# ---------------------------------------------------------------------
# Merge helper (streaming primary, bounded chunk fallback)
# ---------------------------------------------------------------------
def merge_parquet_files_fast(paths: List[Path], out_path: Path, chunk_size: int = 64) -> None:
    """
    Merge a list of parquet files into out_path using Polars streaming when possible.
    If streaming fails due to schema inconsistencies, this function will try a
    bounded chunk approach that keeps memory bounded by `chunk_size` files per iteration.
    """
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    if not paths:
        raise ValueError("No input paths to merge_parquet_files_fast")

    srcs = [str(p) for p in paths]

    # Try streaming first (low memory)
    try:
        pl.scan_parquet(srcs).sink_parquet(str(out_path), compression="snappy")
        return
    except Exception as e_stream:
        logger.warning("Streaming merge failed (%s). Falling back to bounded chunked merge.", e_stream)
        _record_io_error(e_stream, "streaming_merge_failed")

    tmp_chunks: List[Path] = []
    try:
        for i in range(0, len(srcs), chunk_size):
            chunk = srcs[i: i + chunk_size]
            tmp_chunk = out_path.with_name(f"{out_path.stem}.chunk{i:04d}.parquet")
            # collect chunk (bounded size)
            df_chunk = pl.scan_parquet(chunk).collect()
            # cast chunk to canonical schema to help subsequent combine succeed
            df_chunk = cast_master_polars_df(df_chunk)
            # atomic write chunk
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
        # combine intermediate chunks via streaming
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


# ---------------------------------------------------------------------
# Helpers for consumer code
# ---------------------------------------------------------------------
def list_master_part_paths(results_dir: Path, batch_id: int) -> List[Path]:
    part_dir = Path(results_dir) / "master_parts" / f"batch_{batch_id:04d}"
    if not part_dir.exists():
        return []
    return sorted(part_dir.glob("*.parquet"))


def inspect_master_part_schema(path: Path) -> Dict[str, str]:
    """
    Return a simple mapping of column->dtype string for a given master part file.
    This will read the file header via Polars (lightweight for small parts).
    """
    try:
        df = pl.read_parquet(str(path), n_rows=0)
        return {c: str(df.schema[c]) for c in df.columns}
    except Exception as e:
        _record_io_error(e, f"inspect_schema_{path}")
        logger.debug("inspect_master_part_schema failed for %s: %s", path, e)
        return {}


# Public API
__all__ = [
    "master_metrics_schema",
    "buffer_master_row",
    "_flush_master_rows_buffer",
    "_atomic_write_parquet",
    "merge_parquet_files_fast",
    "list_master_part_paths",
    "inspect_master_part_schema",
    "cast_master_polars_df",
]