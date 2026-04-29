# trade_ml_stager.py

from __future__ import annotations

import gc
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from common.schema import enforce_schema, get_schema
from research.merge_utils import _merge_parquet_files_skip_bad

logger = logging.getLogger(__name__)

_TRADE_ML_SHARD_RE = re.compile(
    r"^trade_ml_era_int=(?P<era>\d+)_batch=(?P<batch>\d+)"
    r"_(?:worker|task)=(?P<worker>[^.]+)\.parquet$"
)


# -----------------------------------------------------------------------------
# Small filesystem helpers
# -----------------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _cleanup_path(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        logger.debug("cleanup failed for %s: %s", path, e)


def _normalize_worker_id() -> str:
    worker = os.getenv("AIRFLOW_MAP_INDEX", os.getenv("MAP_INDEX", "0"))
    worker = str(worker).strip() or "0"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", worker)


# -----------------------------------------------------------------------------
# Writer state
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class _ShardKey:
    session_dir: str
    batch_id: int
    era_int: int
    worker_id: str


@dataclass
class _WriterState:
    path: Path
    writer: Optional[pq.ParquetWriter] = None
    schema_arrow: Optional[pa.Schema] = None


_WRITERS: Dict[_ShardKey, _WriterState] = {}


def _tmp_dir(session_dir: Path) -> Path:
    p = Path(session_dir) / "trade_ml_partitioned" / "_tmp"
    _ensure_dir(p)
    return p


def _part_path(session_dir: str | Path, batch_id: int, era_int: int, worker_id: Optional[str] = None) -> Path:
    session_dir = Path(session_dir)
    tmp_dir = _tmp_dir(session_dir)
    worker_token = _normalize_worker_id() if worker_id is None else re.sub(r"[^A-Za-z0-9_.-]+", "_", str(worker_id))
    return tmp_dir / (
        f"trade_ml_era_int={int(era_int)}"
        f"_batch={int(batch_id):04d}"
        f"_worker={worker_token}.parquet"
    )


def _shard_key(session_dir: str | Path, batch_id: int, era_int: int, worker_id: Optional[str] = None) -> _ShardKey:
    return _ShardKey(
        session_dir=str(Path(session_dir).resolve()),
        batch_id=int(batch_id),
        era_int=int(era_int),
        worker_id=_normalize_worker_id() if worker_id is None else re.sub(r"[^A-Za-z0-9_.-]+", "_", str(worker_id)),
    )


def _get_or_create_state(
    session_dir: str | Path,
    batch_id: int,
    era_int: int,
    first_df: pl.DataFrame,
    worker_id: Optional[str] = None,
) -> _WriterState:
    key = _shard_key(session_dir, batch_id=batch_id, era_int=era_int, worker_id=worker_id)

    state = _WRITERS.get(key)
    if state is not None and state.writer is not None:
        return state

    out_path = _part_path(session_dir, batch_id=batch_id, era_int=era_int, worker_id=worker_id)
    _ensure_dir(out_path.parent)

    if out_path.exists():
        try:
            out_path.unlink()
        except Exception:
            pass

    if first_df is None or first_df.is_empty():
        raise RuntimeError(
            f"Cannot open trade_ml writer for empty batch: era_int={era_int}, batch_id={batch_id}"
        )

    schema_arrow = first_df.to_arrow().schema
    writer = pq.ParquetWriter(str(out_path), schema_arrow, compression="snappy")
    state = _WriterState(path=out_path, writer=writer, schema_arrow=schema_arrow)
    _WRITERS[key] = state
    return state


def _append_df(
    session_dir: str | Path,
    batch_id: int,
    era_int: int,
    df_chunk: pl.DataFrame,
    worker_id: Optional[str] = None,
) -> Path:
    if df_chunk is None or df_chunk.is_empty():
        raise ValueError("df_chunk must not be empty")

    state = _get_or_create_state(
        session_dir=session_dir,
        batch_id=batch_id,
        era_int=era_int,
        first_df=df_chunk,
        worker_id=worker_id,
    )
    assert state.writer is not None

    state.writer.write_table(df_chunk.to_arrow())
    return state.path


def flush_trade_ml_writers(session_dir: str | Path | None = None) -> None:
    """
    Close any open trade_ml writers.

    If session_dir is provided, only close writers belonging to that session.
    """
    session_filter = None if session_dir is None else str(Path(session_dir).resolve())

    items = list(_WRITERS.items())
    for key, state in items:
        if session_filter is not None and key.session_dir != session_filter:
            continue

        writer = state.writer
        if writer is None:
            continue

        try:
            writer.close()
        except Exception:
            logger.exception("flush_trade_ml_writers: failed to close writer for %s", state.path)

        if key in _WRITERS:
            _WRITERS[key].writer = None

    if session_filter is None:
        _WRITERS.clear()
    else:
        for key in [k for k in _WRITERS.keys() if k.session_dir == session_filter]:
            _WRITERS.pop(key, None)


def close_trade_ml_writers(session_dir: str | Path | None = None) -> None:
    flush_trade_ml_writers(session_dir=session_dir)


def _schema_cast_trade_ml(df: pl.DataFrame) -> pl.DataFrame:
    if df is None or df.is_empty():
        return df
    return enforce_schema(df, "trade_ml", strict=True)


# -----------------------------------------------------------------------------
# Stager
# -----------------------------------------------------------------------------
class TradeMLStager:
    """
    One temp parquet per (era_int, batch_id, worker_id).

    Compared with the older approach, this class:
    - keeps one writer open per shard
    - writes in larger chunks
    - avoids creating a new parquet file for every fragment
    - keeps gc.collect() out of the inner write loop
    """

    def __init__(
        self,
        session_dir: str | Path,
        batch_id: int,
        flush_rows: int,
        partition_by: str = "era_int",
        worker_id: Optional[str] = None,
    ):
        self.session_dir = Path(session_dir)
        self.batch_id = int(batch_id)
        self.flush_rows = int(flush_rows)
        self.partition_by = str(partition_by)
        self.worker_id = _normalize_worker_id() if worker_id is None else re.sub(r"[^A-Za-z0-9_.-]+", "_", str(worker_id))

        self.session_dir.mkdir(parents=True, exist_ok=True)
        _tmp_dir(self.session_dir)

        self._writers: Dict[str, pq.ParquetWriter] = {}
        self._paths: Dict[str, Path] = {}

    def _ensure_writer(self, part_key: str, first_df: pl.DataFrame) -> pq.ParquetWriter:
        writer = self._writers.get(part_key)
        if writer is not None:
            return writer

        out_path = _part_path(self.session_dir, self.batch_id, int(part_key), self.worker_id)
        self._paths[part_key] = out_path

        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass

        if first_df is None or first_df.height == 0:
            raise RuntimeError(f"Cannot open trade_ml writer for empty part_key={part_key}")

        writer = pq.ParquetWriter(str(out_path), first_df.to_arrow().schema, compression="snappy")
        self._writers[part_key] = writer
        return writer

    def stage(self, part_key: str, df_small: pl.DataFrame) -> None:
        """
        Append trade ML rows immediately, chunked by flush_rows.
        """
        if df_small is None or df_small.height == 0:
            return

        df_canonical = _schema_cast_trade_ml(df_small)
        if df_canonical is None or df_canonical.height == 0:
            return

        n = int(df_canonical.height)
        step = max(1, int(self.flush_rows))

        writer = self._ensure_writer(str(part_key), df_canonical)
        for start in range(0, n, step):
            chunk = df_canonical.slice(start, step)
            if chunk.is_empty():
                continue
            writer.write_table(chunk.to_arrow())

    def flush(self, part_key: str) -> None:
        """
        No buffer to flush because writes are immediate.
        Kept as a no-op interface method.
        """
        return

    def flush_all(self) -> None:
        for part_key, writer in list(self._writers.items()):
            try:
                writer.close()
            except Exception:
                logger.exception("TradeMLStager.flush_all: failed to close writer for %s", part_key)
        self._writers = {}
        self._paths = {}
        gc.collect()


# -----------------------------------------------------------------------------
# Shard inspection
# -----------------------------------------------------------------------------
def _trade_ml_parts_for_partition(session_dir: Path, partition_key: str, batch_id: Optional[int] = None) -> List[Path]:
    tmp_dir = _tmp_dir(session_dir)

    if batch_id is None:
        candidates = sorted(tmp_dir.glob(f"trade_ml_era_int={partition_key}_batch=*.parquet"))
    else:
        candidates = sorted(tmp_dir.glob(f"trade_ml_era_int={partition_key}_batch={int(batch_id):04d}_*.parquet"))

    parts: List[Path] = []
    for p in candidates:
        if _TRADE_ML_SHARD_RE.match(p.name):
            parts.append(p)

    return parts


def _open_parquet(path: Path) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(str(path))
    except Exception as e:
        raise RuntimeError(f"Invalid parquet file: {path}") from e

def combine_trade_ml_parts(
    session_dir: str | Path,
    partition_key: str,
    batch_id: Optional[int] = None,
    batch_size: int = 65_536,
    stage_size: int = 256,  # kept only for compatibility; not used here
) -> Optional[Path]:
    """
    Merge worker trade_ml shards for one era into one final parquet file.

    Output layout:
      trade_ml_partitioned/era_int=<era>/
        trade_ml_era_int=<era>.parquet
        trade_ml_era_int=<era>.manifest.json
    """
    session_dir = Path(session_dir)

    flush_trade_ml_writers(session_dir=session_dir)

    parts = _trade_ml_parts_for_partition(session_dir, partition_key, batch_id=batch_id)
    if not parts:
        return None

    final_dir = session_dir / "trade_ml_partitioned" / f"era_int={partition_key}"
    final_dir.mkdir(parents=True, exist_ok=True)

    final_path = final_dir / f"trade_ml_era_int={partition_key}.parquet"
    tmp_path = final_path.with_suffix(".tmp.parquet")

    _cleanup_path(final_path)
    _cleanup_path(tmp_path)

    result = _merge_parquet_files_skip_bad(
        [str(p) for p in parts],
        tmp_path,
        kind="trade_ml",
        batch_size=batch_size,
    )

    os.replace(str(tmp_path), str(final_path))

    manifest = {
        "partition_key": str(partition_key),
        "batch_id": None if batch_id is None else int(batch_id),
        "final": str(final_path),
        "part_count": int(len(parts)),
        "merged_files": int(result["merged_files"]),
        "skipped_files": int(result["skipped_files"]),
    }
    manifest_path = final_dir / f"trade_ml_era_int={partition_key}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")

    for p in parts:
        try:
            p.unlink(missing_ok=True)
        except Exception as e:
            logger.debug("failed to delete shard %s: %s", p, e)

    logger.info(
        "combine_trade_ml_parts: era=%s merged=%d skipped=%d",
        partition_key,
        result["merged_files"],
        result["skipped_files"],
    )
    return final_path

def combine_all_trade_ml_parts(session_dir: str | Path, batch_id: Optional[int] = None) -> List[Path]:
    """
    Merge every era partition for trade_ml.
    """
    session_dir = Path(session_dir)
    flush_trade_ml_writers(session_dir=session_dir)

    tmp_dir = _tmp_dir(session_dir)

    if batch_id is None:
        candidates = sorted(tmp_dir.glob("trade_ml_era_int=*_batch=*.parquet"))
    else:
        candidates = sorted(tmp_dir.glob(f"trade_ml_era_int=*_batch={int(batch_id):04d}_*.parquet"))

    eras: Dict[str, List[Path]] = {}
    for p in candidates:
        m = _TRADE_ML_SHARD_RE.match(p.name)
        if not m:
            logger.debug("Skipping non-matching trade_ml shard name: %s", p.name)
            continue
        if batch_id is not None and int(m.group("batch")) != int(batch_id):
            continue
        era = m.group("era")
        eras.setdefault(era, []).append(p)

    outputs: List[Path] = []
    for era in sorted(eras.keys()):
        out = combine_trade_ml_parts(session_dir=session_dir, partition_key=era, batch_id=batch_id)
        if out is not None:
            outputs.append(out)

    return outputs


__all__ = [
    "TradeMLStager",
    "flush_trade_ml_writers",
    "close_trade_ml_writers",
    "combine_trade_ml_parts",
    "combine_all_trade_ml_parts",
]
