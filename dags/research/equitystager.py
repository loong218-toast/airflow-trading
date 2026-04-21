from __future__ import annotations

import gc
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable, Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from common.schema import enforce_schema

logger = logging.getLogger(__name__)

_EQUITY_SHARD_RE = re.compile(
    r"^equity_era_int=(?P<era>\d+)_batch=(?P<batch>\d+)_(?:worker|task)=(?P<unit>\d+)\.parquet$"
)

def _equity_tmp_dir(session_dir: Path) -> Path:
    tmp_dir = session_dir / "equity_partitioned" / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir

# -----------------------------------------------------------------------------
# Equity staging
# -----------------------------------------------------------------------------
@dataclass
class _EquityWriterState:
    path: Path
    writer: Optional[pq.ParquetWriter] = None


class EquityStager:
    """
    One temp parquet per (partition_key, batch_id, worker_id).

    It keeps a writer open per partition and appends row groups to the same file
    instead of creating many tiny parquet fragments.
    """

    def __init__(
        self,
        equity_part_base: Path,
        batch_id: int,
        flush_rows: int,
        partition_by: str,
        max_total_rows: int = 200_000,
        max_partitions: int = 128,
        tmp_dir: Optional[Path] = None,
    ):
        self.base = Path(equity_part_base)
        self.batch_id = int(batch_id)
        self.flush_rows = int(flush_rows)
        self.partition_by = str(partition_by)

        self.max_total_rows = int(max_total_rows)
        self.max_partitions = int(max_partitions)

        self._tmp_dir = Path(tmp_dir) if tmp_dir is not None else (self.base / "_tmp")
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self.base.mkdir(parents=True, exist_ok=True)

        # Open writer per partition key and keep it open until flush_all().
        self._writers: Dict[str, pq.ParquetWriter] = {}
        self._paths: Dict[str, Path] = {}

    def _worker_id(self) -> str:
        return os.getenv("AIRFLOW_MAP_INDEX", os.getenv("MAP_INDEX", "0"))

    def _part_path(self, part_key: str) -> Path:
        worker_id = self._worker_id()
        return self._tmp_dir / (
            f"equity_{self.partition_by}={part_key}"
            f"_batch={self.batch_id}"
            f"_task={worker_id}.parquet"
        )

    def _ensure_writer(self, part_key: str, first_table: Optional[pl.DataFrame] = None) -> pq.ParquetWriter:
        writer = self._writers.get(part_key)
        if writer is not None:
            return writer

        out_path = self._part_path(part_key)
        self._paths[part_key] = out_path

        # Fresh run or stale partial file: overwrite cleanly.
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass

        if first_table is None or first_table.height == 0:
            raise RuntimeError(f"Cannot open ParquetWriter for empty equity batch: part_key={part_key}")

        writer = pq.ParquetWriter(
            str(out_path),
            first_table.to_arrow().schema,
            compression="snappy",
        )
        self._writers[part_key] = writer
        return writer

    def _append_df(self, part_key: str, df_chunk: pl.DataFrame) -> None:
        if df_chunk is None or df_chunk.height == 0:
            return

        writer = self._ensure_writer(part_key, df_chunk)
        writer.write_table(df_chunk.to_arrow())

    def stage(self, part_key: str, df_small: pl.DataFrame) -> None:
        """
        Append equity rows immediately, chunked by flush_rows to keep memory low.
        """
        if df_small is None or df_small.height == 0:
            return

        try:
            df_canonical = enforce_schema(df_small, "equity", strict=True)
        except Exception:
            logger.exception("EquityStager.stage: enforce_schema failed for part %s; skipping fragment", part_key)
            return

        if df_canonical.height == 0:
            return

        n = int(df_canonical.height)
        step = max(1, int(self.flush_rows))

        for start in range(0, n, step):
            chunk = df_canonical.slice(start, step)
            try:
                self._append_df(part_key, chunk)
            except Exception:
                logger.exception("EquityStager.stage: write failed for part %s", part_key)
                raise

    def flush(self, part_key: str) -> None:
        """
        No buffer to flush because writes are immediate.
        Kept as a no-op interface method.
        """
        return

    def flush_all(self) -> None:
        """
        Close all open Parquet writers.
        """
        for part_key, writer in list(self._writers.items()):
            try:
                writer.close()
            except Exception:
                logger.exception("EquityStager.flush_all: failed to close writer for %s", part_key)

        self._writers = {}
        self._paths = {}


# -----------------------------------------------------------------------------
# Equity preview helpers
# -----------------------------------------------------------------------------
# These are intentionally kept in this module so grid.py can stay smaller.
# If the low-level numba kernels live elsewhere, import them here and wire them in.
def compute_equity_preview(
    closed_entry_prices_arr: np.ndarray,
    closed_exit_prices_arr: np.ndarray,
    closed_entry_idxs: np.ndarray,
    closed_exit_idxs: np.ndarray,
    main_close_arr: np.ndarray,
    main_time_ns_arr: np.ndarray,
    main_spread_arr: np.ndarray,
    main_funding_arr: np.ndarray,
    sl_val: float,
    tp_val: float,
    side_flag: int,
    run_cfg: dict,
    pnl_fn: Callable[..., np.ndarray],
    equity_gate_fn: Callable[[np.ndarray, float, float], Tuple[np.ndarray, float, bool]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """
    Convert closed-trade exits into equity curve inputs.

    The kernel-specific math is injected so this file stays isolated from grid.py.
    """
    pnl_pct_arr = pnl_fn(
        entry_prices=np.asarray(closed_entry_prices_arr, dtype=np.float64),
        exit_prices=np.asarray(closed_exit_prices_arr, dtype=np.float64),
        spread_arr=np.asarray(main_spread_arr[closed_entry_idxs], dtype=np.float64)
        if main_spread_arr is not None else np.zeros(closed_entry_idxs.shape[0], dtype=np.float64),
        funding_raw=np.asarray(main_funding_arr[closed_entry_idxs], dtype=np.float64)
        if main_funding_arr is not None else np.zeros(closed_entry_idxs.shape[0], dtype=np.float64),
        entry_idxs=np.asarray(closed_entry_idxs, dtype=np.int64),
        exit_idxs=np.asarray(closed_exit_idxs, dtype=np.int64),
        main_time_ns=np.asarray(main_time_ns_arr, dtype=np.int64),
        side_flag=int(side_flag),
        risk_pct=float(run_cfg.get("risk_pct", 0.005)),
        sl_val=float(sl_val),
        sl_tp_in_pct=bool(run_cfg.get("sl_tp_in_pct", True)),
        spread_is_percent=bool(run_cfg.get("spread_is_percent", True)),
        funding_period_hours=int(run_cfg.get("funding_period_hours", 8)),
        funding_is_per_hour_bool=bool(run_cfg.get("funding_is_per_hour", False)),
        trading_fee_rate=float(run_cfg.get("trading_fees", 0.0004) or 0.0004),
    )

    if pnl_pct_arr is None or pnl_pct_arr.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            0.0,
            False,
        )

    pnl_pct_arr = np.nan_to_num(pnl_pct_arr, nan=0.0, posinf=1e37, neginf=-1e37)
    max_dd_threshold = float(run_cfg.get("max_dd_threshold", 0.5))
    equity_arr, max_dd, breached = equity_gate_fn(pnl_pct_arr, 100.0, max_dd_threshold)
    equity_arr = np.nan_to_num(equity_arr, nan=0.0, posinf=1e37, neginf=-1e37)

    exit_times_ns = np.asarray(main_time_ns_arr[closed_exit_idxs], dtype=np.int64)
    return pnl_pct_arr, exit_times_ns, equity_arr, float(max_dd), bool(breached)


def stage_equity_from_preview(
    pnl_pct_arr: np.ndarray,
    exit_times_ns: np.ndarray,
    equity_arr: np.ndarray,
    closed_entry_idxs: np.ndarray,
    closed_exit_idxs: np.ndarray,
    sl_val: float,
    tp_val: float,
    regime_id: int,
    signal_layer: int,
    era_int: int,
    side_flag: int,
    stager: EquityStager,
    max_dd: float,
    signal_scope: str = "",
) -> Tuple[float, int, float]:
    """
    Stage equity rows only after the DD gate already passed.
    Returns final_balance, win_pos, max_dd.
    """
    if equity_arr is None or equity_arr.size == 0:
        return 100.0, 0, float(max_dd)

    pnl_pct_arr = np.nan_to_num(pnl_pct_arr, nan=0.0, posinf=1e37, neginf=-1e37)
    equity_arr = np.nan_to_num(equity_arr, nan=0.0, posinf=1e37, neginf=-1e37)

    win_pos = int(np.count_nonzero(pnl_pct_arr > 0.0))
    final_balance = float(equity_arr[-1]) if equity_arr.size else 100.0

    equity_df = pl.DataFrame(
        {
            "regime_id": [int(regime_id)],
            "signal_layer": [int(signal_layer)],
            "signal_scope": [str(signal_scope or "")],
            "signal_scope_id": [str(signal_scope or "")],
            "era_int": [int(era_int)],
            "side": [int(side_flag)],
            "SL": [float(sl_val)],
            "TP": [float(tp_val)],
            "time_ns": pl.Series([exit_times_ns.astype(np.int64)], dtype=pl.List(pl.Int64)),
            "entry_idx": pl.Series([closed_entry_idxs.astype(np.int64)], dtype=pl.List(pl.Int64)),
            "exit_idx": pl.Series([closed_exit_idxs.astype(np.int64)], dtype=pl.List(pl.Int64)),
            "pnl_pct": pl.Series([pnl_pct_arr.astype(np.float32)], dtype=pl.List(pl.Float32)),
            "equity": pl.Series([equity_arr.astype(np.float32)], dtype=pl.List(pl.Float32)),
        }
    )

    try:
        equity_df = enforce_schema(equity_df, "equity", strict=True)
    except Exception as e:
        logger.exception("stage_equity_from_preview: failed to build equity DataFrame: %s", e)
        return final_balance, win_pos, float(max_dd)

    meta = {
        "rows": int(equity_df.height),
        "cols": list(equity_df.columns),
    }
    if meta["rows"] == 0:
        logger.error("stage_equity_from_preview: refusing to stage empty equity fragment")
        return final_balance, win_pos, float(max_dd)

    partition_val = str(regime_id) if getattr(stager, "partition_by", "") == "regime_id" else str(era_int)
    try:
        stager.stage(partition_val, equity_df)
    except Exception:
        logger.exception("stage_equity_from_preview: stager.stage failed for partition %s", partition_val)

    return final_balance, win_pos, float(max_dd)

def _equity_parts_for_partition(session_dir: Path, partition_key: str, batch_id: Optional[int] = None) -> List[Path]:
    tmp_dir = _equity_tmp_dir(session_dir)
    if batch_id is None:
        candidates = sorted(tmp_dir.glob(f"equity_era_int={partition_key}_batch=*.parquet"))
    else:
        candidates = sorted(tmp_dir.glob(f"equity_era_int={partition_key}_batch={int(batch_id)}_*.parquet"))

    parts: List[Path] = []
    for p in candidates:
        if _EQUITY_SHARD_RE.match(p.name):
            parts.append(p)
    return parts


def _append_parquet_file(writer: pq.ParquetWriter, path: Path) -> bool:
    try:
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches():
            writer.write_table(batch.to_table())
        return True
    except Exception as e:
        _LOG.error("Skipping corrupt parquet shard: %s | %s", path, e)
        return False

def combine_equity_parts(session_dir: Path, partition_key: str, batch_id: Optional[int] = None) -> Optional[Path]:
    parts = _equity_parts_for_partition(session_dir, partition_key, batch_id=batch_id)
    if not parts:
        return None

    total_parts = len(parts)
    _LOG.info("Starting merge of %d shards for era %s", total_parts, partition_key)

    final_dir = session_dir / "equity_partitioned" / f"era_int={partition_key}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"equity_era_int={partition_key}.parquet"
    tmp_path = final_path.with_suffix(".tmp.parquet")

    _cleanup_path(final_path)
    _cleanup_path(tmp_path)
    writer = None
    merged = 0

    try:
        for idx, p in enumerate(parts):
            if idx % 25 == 0 and idx > 0:
                _LOG.info("[%s] Merged %d/%d shards...", partition_key, idx, total_parts)
            try:
                pf = pq.ParquetFile(str(p))
            except Exception as e:
                _LOG.error("Skipping corrupt parquet shard: %s | %s", p, e)
                continue

            if writer is None:
                writer = pq.ParquetWriter(str(final_path.with_suffix(".tmp.parquet")), pf.schema_arrow, compression="snappy")

            try:
                for batch in pf.iter_batches():
                    # Wrap the single batch in a list to create a Table
                    table = pa.Table.from_batches([batch]) 
                    writer.write_table(table)
                merged += 1
            except Exception as e:
                _LOG.error("Skipping bad parquet shard during merge: %s | %s", p, e)
                continue

        if writer is None or merged == 0:
            return None

        writer.close()
        os.replace(str(final_path.with_suffix(".tmp.parquet")), str(final_path))

        manifest = {
            "parts": [str(p) for p in parts],
            "final": str(final_path),
            "merged_files": merged,
        }
        final_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))

        for p in parts:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

        return final_path

    except Exception:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        _cleanup_path(final_path.with_suffix(".tmp.parquet"))
        raise


def combine_all_equity_parts(session_dir: str | Path, batch_id: Optional[int] = None) -> List[Path]:
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
        if batch_id is not None and int(m.group("batch")) != int(batch_id):
            continue
        eras[m.group("era")].append(p)

    outputs: List[Path] = []
    for era in sorted(eras.keys()):
        out = combine_equity_parts(session_dir=session_dir, partition_key=era, batch_id=batch_id)
        if out is not None:
            outputs.append(out)

    return outputs




__all__ = [
    "EquityStager",
    "compute_equity_preview",
    "stage_equity_from_preview",
]
