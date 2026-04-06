from __future__ import annotations
import logging
import os
import time
from datetime import timedelta
import json
import math
import gc
import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
import polars as pl
import pandas as pd
import pyarrow.parquet as pq
import secrets
from itertools import product

import numba as _numba
from numba import njit
_numba_threads = int(os.getenv("NUMBA_NUM_THREADS", os.getenv("NUMBA_NUM_THREADS_OVERRIDE", "1")))
try:
    _numba.set_num_threads(_numba_threads)
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", str(_numba_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_numba_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_numba_threads))

import psutil
from airflow.exceptions import AirflowFailException

from etl.transform import build_df_main_from_5m_polars, load_candles_from_db_polars

from common.feature_prep import (
    load_prepared_feature_ref, 
    precompute_all_possible_features, 
    prepare_feature_cache
    )
from common.feature_helpers import (
    generate_filtered_signals, 
    get_regime_amp_index_for_indices
    )

from research.backtest import (
    backtest_signals_sl_tp_rets,
    fast_compound_equity_gate,
    compute_pnl_pct_vectorized,
    compute_max_consecutive_losses
)

from common.cache import (
    load_signals_cached,
    stage_for_flush,
    load_backtest_cached,
    flush_all_buffers,
    load_global_signals_cached,
    stage_global_signals
)

from research.io_utils import FULL_LAKE_DIR
from research.merge_utils import combine_cache, merge_batch_master_parts, combine_results_to_master
from research.master_io_utils import (
    buffer_master_row,
    _flush_master_rows_buffer,
)

from common.profiling import maybe_profile

from research.grid_row_builders import _make_master_row, build_trade_ml_rows_from_backtest

from common.schema import enforce_schema, get_schema, cast_to_schema, classify_fragment

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------- utilities ----------------
def heartbeat_log(tag: str, extra: dict | None = None):
    proc = psutil.Process()
    mem = proc.memory_info().rss
    cpu = psutil.cpu_percent(interval=None)
    extra = extra or {}
    logger.info(f"💓 {tag} | cpu={cpu:.1f}% mem={mem//1024//1024}MB {extra}")

from pathlib import Path

def _get_cgroup_memory_limit_bytes() -> int | None:
    paths = [
        Path("/sys/fs/cgroup/memory.max"),                 # cgroup v2
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes") # cgroup v1
    ]

    for p in paths:
        try:
            if p.exists():
                txt = p.read_text().strip()
                if txt == "max":
                    return None
                val = int(txt)
                if val > 0 and val < 1 << 60:
                    return val
        except Exception:
            pass

    return None


def _get_process_rss_mb() -> float:
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _get_memory_guard_mb() -> float:
    limit_bytes = _get_cgroup_memory_limit_bytes()
    if limit_bytes is not None:
        return float(limit_bytes) / (1024.0 * 1024.0)
    try:
        return float(psutil.virtual_memory().total) / (1024.0 * 1024.0)
    except Exception:
        return 0.0

def _split_period_windows_from_pl(
    df: pl.DataFrame,
    months: int,
    min_dt: Optional[datetime.datetime] = None,
    max_dt: Optional[datetime.datetime] = None,
    include_partial_tail: bool = False,
) -> List[tuple]:
    import datetime as _dt

    def _parse(dt_in):
        if dt_in is None:
            return None
        if isinstance(dt_in, str):
            s = dt_in
            if s.endswith("Z"):
                s = s[:-1]
            try:
                parsed = _dt.datetime.fromisoformat(s)
            except Exception:
                try:
                    parsed = _dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
                except Exception:
                    return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=_dt.timezone.utc)
            return parsed.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        if isinstance(dt_in, _dt.datetime):
            d = dt_in
            if d.tzinfo is not None:
                return d.astimezone(_dt.timezone.utc).replace(tzinfo=None)
            return d
        return None

    if (min_dt is None or max_dt is None) and (df is None or df.height == 0):
        return []

    if min_dt is None or max_dt is None:
        try:
            first = df[0, "time"]
            last = df[-1, "time"]
            if hasattr(first, "to_pydatetime"):
                min_dt = min_dt or first.to_pydatetime()
            else:
                min_dt = min_dt or first
            if hasattr(last, "to_pydatetime"):
                max_dt = max_dt or last.to_pydatetime()
            else:
                max_dt = max_dt or last
        except Exception:
            return []

    min_dt = _parse(min_dt)
    max_dt = _parse(max_dt)
    if min_dt is None or max_dt is None:
        return []

    start = _dt.datetime(min_dt.year, min_dt.month, 1)
    windows = []
    cur = start

    while True:
        m = cur.month - 1 + int(months)
        y = cur.year + (m // 12)
        mm = (m % 12) + 1
        end = _dt.datetime(year=y, month=mm, day=1)

        if end > max_dt:
            if include_partial_tail and cur < max_dt:
                windows.append((cur, max_dt))
            break

        windows.append((cur, end))
        cur = end

    return windows

class EquityStager:
    """
    One temp parquet per (partition_key, batch_id, worker_id).
    Appends row groups to the same file instead of creating many files.
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

        # Open writer per partition key and keep it open until flush_all()
        self._writers: Dict[str, pq.ParquetWriter] = {}
        self._paths: Dict[str, Path] = {}

    def _worker_id(self) -> str:
        return os.getenv("AIRFLOW_MAP_INDEX", "0")

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

        # Fresh run or stale partial file: overwrite cleanly
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass

        if first_table is None or first_table.height == 0:
            raise RuntimeError(f"Cannot open ParquetWriter for empty equity batch: part_key={part_key}")

        arrow_table = first_table.to_arrow()
        writer = pq.ParquetWriter(
            str(out_path),
            arrow_table.schema,
            compression="snappy",
        )
        self._writers[part_key] = writer
        return writer

    def _append_df(self, part_key: str, df_chunk: pl.DataFrame):
        if df_chunk is None or df_chunk.height == 0:
            return

        writer = self._ensure_writer(part_key, df_chunk)
        writer.write_table(df_chunk.to_arrow())


    def stage(self, part_key: str, df_small: pl.DataFrame):
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

    def flush(self, part_key: str):
        """
        No buffer to flush because writes are immediate.
        Kept as a no-op interface method.
        """
        return

    def flush_all(self):
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

# SL/TP helpers (unchanged)
def _expand_sl_tp(run_cfg: Dict) -> Tuple[List[float], List[float]]:
    def _get_vals(key):
        if f"{key}_values" in run_cfg:
            return [float(x) for x in run_cfg[f"{key}_values"]]
        if f"{key}_range" in run_cfg:
            r = run_cfg[f"{key}_range"]
            mn = float(r["min"]); mx = float(r["max"]); step = float(r["step"])
            vals = []
            v = mn
            while v <= mx + 1e-9:
                vals.append(round(float(v), 8))
                v += step
            return vals
        raise ValueError(f"Missing '{key}_range' or '{key}_values' in run_config.")
    sl_vals = _get_vals("sl")
    tp_vals = _get_vals("tp")
    return sl_vals, tp_vals

def _prune_by_min_rr(sl_vals: List[float], tp_vals: List[float], min_rr: float) -> List[tuple]:
    """
    Return list of (SL, TP) pairs where TP >= min_rr * SL.

    Implementation notes:
    - Avoids O(n*m) nested Python loops and large boolean matrices by sorting TP once
      and using np.searchsorted to find the first acceptable TP for each SL.
    - Filters out non-positive SL/TP values (they cannot satisfy a positive RR).
    - If min_rr <= 0, returns the Cartesian product (behaves like no pruning).
    """
    import numpy as np

    if not sl_vals or not tp_vals:
        return []

    sl_arr = np.asarray(sl_vals, dtype=np.float32)
    tp_arr = np.asarray(tp_vals, dtype=np.float32)

    # Fast path: if min_rr <= 0, nothing to prune (return full cartesian product)
    if float(min_rr) <= 0.0:
        return [(float(s), float(t)) for s in sl_arr for t in tp_arr]

    # Exclude non-positive values (can't satisfy positive RR)
    sl_arr = sl_arr[sl_arr > 0.0]
    tp_arr = tp_arr[tp_arr > 0.0]
    if sl_arr.size == 0 or tp_arr.size == 0:
        return []

    # Sort TP once and use searchsorted to find the first TP >= threshold for each SL
    tp_sorted = np.sort(tp_arr)
    combos: List[tuple] = []
    rr = float(min_rr)

    for s in sl_arr:
        threshold = rr * float(s)
        pos = int(np.searchsorted(tp_sorted, threshold, side="left"))
        if pos < tp_sorted.size:
            # extend with only acceptable TP values (keeps memory usage minimal)
            # convert to Python floats for downstream compatibility
            combos.extend([(float(s), float(t)) for t in tp_sorted[pos:]])

    return combos

def _partition_list(items: List, chunk_size: int) -> List[List]:
    if chunk_size <= 0:
        return [items]
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def _to_numpy_ensure(arr_series: pl.Series, dtype):
    try:
        a = arr_series.to_numpy(allow_copy=False)
    except RuntimeError:
        a = arr_series.to_numpy(allow_copy=True)
    if a.dtype != dtype:
        try:
            return a.astype(dtype, copy=False)
        except Exception:
            return a.astype(dtype, copy=True)
    return a

def prepare_worker_data(session_dir: Path, run_cfg: dict):
    from common.feature_prep import load_prepared_feature_ref

    session_dir = Path(session_dir)

    ref = load_prepared_feature_ref(session_dir)
    prepared_path = Path(ref["parquet_path"])
    if not prepared_path.exists():
        raise RuntimeError(f"Prepared feature parquet missing at {prepared_path}")

    df_main = pl.read_parquet(str(prepared_path))

    run_cfg["lookback_map"] = dict(ref.get("lookback_map", {}))
    run_cfg["ma_cols"] = list(ref.get("ma_cols", []))
    run_cfg["stoch_cols"] = list(ref.get("stoch_cols", []))
    run_cfg["bbw_cols"] = list(ref.get("bbw_cols", []))
    run_cfg["feature_cache_path"] = str(prepared_path)

    base_cols = {
        "pair", "market_type", "time", "time_ns", "open", "high", "low",
        "close", "volume", "funding_rate", "spread", "era_int", "idx",
    }

    return {
        "df_main": df_main,
        "base_cols": base_cols,
        "lookback_map": run_cfg["lookback_map"],
        "ma_cols": run_cfg["ma_cols"],
        "stoch_cols": run_cfg["stoch_cols"],
        "bbw_cols": run_cfg["bbw_cols"],
        "main_close_arr": _to_numpy_ensure(df_main["close"], np.float32),
        "main_high_arr": _to_numpy_ensure(df_main["high"], np.float32),
        "main_low_arr": _to_numpy_ensure(df_main["low"], np.float32),
        "main_time_ns_arr": _to_numpy_ensure(df_main["time_ns"], np.int64),
        "main_spread_arr": _to_numpy_ensure(df_main["spread"], np.float32),
        "main_funding_arr": _to_numpy_ensure(df_main["funding_rate"], np.float32),
    }

def _get_or_generate_global_signals(
    regime_id: int,
    regime_cfg: dict,
    run_cfg: dict,
    data_ctx: dict,
    months: int,
) -> pl.DataFrame:
    """
    Build one global signal set for the full date range of the regime.
    Cache it under era_label='GLOBAL' so it reuses your existing cache layout.
    """
    force_rebuild_cache = bool(run_cfg.get("force_rebuild_cache", False))
    df_main: pl.DataFrame = data_ctx["df_main"]

    if not force_rebuild_cache:
        try:
            cached = load_global_signals_cached(months, str(regime_id))
            if cached is not None and not cached.is_empty():
                return enforce_schema(cached, "signals", strict=False)
        except Exception as e:
            logger.debug("global signal cache read failed cfg=%s: %s", regime_id, e)

    df_signals = generate_filtered_signals(
        df_main,
        {**run_cfg, **regime_cfg, "regime_id": regime_id},
        df_main=df_main,
    )

    if not df_signals.is_empty():
        try:
            signals_for_cache = df_signals.select(["idx", "time_ns", "side", "regime_id"])
            signals_for_cache = enforce_schema(signals_for_cache, "signals", strict=True)
            stage_global_signals(months, str(regime_id), signals_for_cache)
        except Exception as e:
            logger.debug("stage_for_flush(global signals) failed cfg=%s: %s", regime_id, e)

    return enforce_schema(df_signals, "signals", strict=False) if not df_signals.is_empty() else pl.DataFrame([], schema=get_schema("signals"))


def _build_era_registry(
    windows,
    global_signals: pl.DataFrame,
):
    """
    Precompute era boundaries and per-era signal stats.
    """
    if global_signals is None or global_signals.is_empty():
        sig_times_all = np.asarray([], dtype=np.int64)
        sig_sides_all = np.asarray([], dtype=np.int8)
    else:
        sig_times_all = global_signals["time_ns"].to_numpy(allow_copy=True).astype(np.int64)
        sig_sides_all = global_signals["side"].to_numpy(allow_copy=True).astype(np.int8)

    out = []
    for (start, end) in windows:
        if hasattr(start, "tzinfo") and start.tzinfo is not None:
            start = start.replace(tzinfo=None)
        if hasattr(end, "tzinfo") and end.tzinfo is not None:
            end = end.replace(tzinfo=None)

        era_label = start.strftime("%Y-%m")
        try:
            era_int = int(start.strftime("%Y%m%d"))
        except Exception:
            era_int = 0

        start_ns = np.datetime64(start).astype("datetime64[ns]").astype(np.int64)
        end_ns = np.datetime64(end).astype("datetime64[ns]").astype(np.int64)

        mask_all = (sig_times_all >= start_ns) & (sig_times_all < end_ns)
        buy_mask = mask_all & (sig_sides_all == 1)
        sell_mask = mask_all & (sig_sides_all == -1)

        out.append({
            "era_label": era_label,
            "era_int": era_int,
            "start_ns": int(start_ns),
            "end_ns": int(end_ns),
            "buy_sig_n": int(buy_mask.sum()),
            "sell_sig_n": int(sell_mask.sum()),
            "buy_sig_min_ns": int(sig_times_all[buy_mask].min()) if buy_mask.any() else -1,
            "buy_sig_max_ns": int(sig_times_all[buy_mask].max()) if buy_mask.any() else -1,
            "sell_sig_min_ns": int(sig_times_all[sell_mask].min()) if sell_mask.any() else -1,
            "sell_sig_max_ns": int(sig_times_all[sell_mask].max()) if sell_mask.any() else -1,
            "buy_sig_mask": buy_mask,
            "sell_sig_mask": sell_mask,
        })

    return out

@njit(cache=True)
def _numba_pnl_from_actual_exits(
    entry_prices,
    exit_prices,
    spread_arr,
    funding_raw,
    entry_idxs,
    exit_idxs,
    main_time_ns,
    side_flag,
    risk_pct,
    sl_val,
    sl_tp_in_pct,
    spread_is_percent,
    funding_period_hours,
):
    n = entry_prices.shape[0]
    pnl_out = np.empty(n, dtype=np.float64)

    risk = float(risk_pct)
    ns_to_hours = 1.0 / (1e9 * 3600.0)
    f_period = float(max(1, funding_period_hours))
    side = 1 if int(side_flag) >= 0 else -1

    for i in range(n):
        ep = float(entry_prices[i])
        xp = float(exit_prices[i])
        ep_safe = ep if ep != 0.0 else 1.0

        h_hours = (main_time_ns[exit_idxs[i]] - main_time_ns[entry_idxs[i]]) * ns_to_hours
        f_fee = (funding_raw[i] / f_period) * h_hours

        spread = float(spread_arr[i])
        if spread_is_percent:
            spr_pct = spread
        else:
            spr_pct = (spread / ep_safe) * 100.0

        if sl_tp_in_pct:
            raw_sl = float(sl_val)
        else:
            raw_sl = (float(sl_val) / ep_safe) * 100.0

        eff_sl = max(raw_sl + spr_pct, 0.01)
        pos_size = min(risk / (eff_sl / 100.0), 100.0)

        if side == 1:
            trade_ret = (xp - ep) / ep_safe
        else:
            trade_ret = (ep - xp) / ep_safe

        pnl_out[i] = max(min((pos_size * trade_ret) - f_fee, 1e10), -1.0)

    return pnl_out


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    pnl_pct_arr = _numba_pnl_from_actual_exits(
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
    equity_arr, max_dd, breached = fast_compound_equity_gate(pnl_pct_arr, 100.0, max_dd_threshold)
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
    era_int: int,
    side_flag: int,
    stager,
    max_dd: float,
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

    equity_data = {
        "regime_id": int(regime_id),
        "era_int": int(era_int),
        "side": int(side_flag),
        "SL": float(sl_val),
        "TP": float(tp_val),
        "time_ns": exit_times_ns.astype(np.int64),
        "entry_idx": closed_entry_idxs.astype(np.int64),
        "exit_idx": closed_exit_idxs.astype(np.int64),
        "pnl_pct": pnl_pct_arr.astype(np.float32),
        "equity": equity_arr.astype(np.float32),
    }

    try:
        equity_df = cast_to_schema(equity_data, "equity")
    except Exception as e:
        logger.exception("stage_equity_from_preview: failed to build equity DataFrame: %s", e)
        return final_balance, win_pos, float(max_dd)

    meta = classify_fragment(equity_df, "equity")
    if not meta.get("is_like", False):
        logger.error(
            "stage_equity_from_preview: refusing to stage fragment; not equity-like. meta=%s sample=%s",
            meta,
            equity_df.head(1).to_dicts() if equity_df.height else None,
        )
        return final_balance, win_pos, float(max_dd)

    partition_val = str(regime_id) if getattr(stager, "partition_by", "") == "regime_id" else str(era_int)
    try:
        stager.stage(partition_val, equity_df)
    except Exception:
        logger.exception("stage_equity_from_preview: stager.stage failed for partition %s", partition_val)

    return final_balance, win_pos, float(max_dd)

def _run_backtest_grid(
    regime_id: int,
    regime_cfg: dict,
    run_cfg: dict,
    data_ctx: dict,
    df_signals: pl.DataFrame,
    months: int,
    era_label: str,
    era_int: int,
    results_dir: Path,
    batch_id: int,
    stager,
    max_dd_threshold: float,
    current_idx: int,
    total_in_batch: int,
    combos: List[Tuple[float, float]],
):
    df_main = data_ctx["df_main"]
    main_close_arr = data_ctx["main_close_arr"]
    main_high_arr = data_ctx["main_high_arr"]
    main_low_arr = data_ctx["main_low_arr"]
    main_time_ns_arr = data_ctx["main_time_ns_arr"]
    main_spread_arr = data_ctx["main_spread_arr"]
    main_funding_arr = data_ctx["main_funding_arr"]

    use_limit_entry = bool(regime_cfg.get("use_limit_entry", True))
    limit_order_expiry_h = int(regime_cfg.get("limit_order_expiry_h", 0) or 0)
    trade_window_interval = int(regime_cfg.get("trade_window_interval", 0) or 0)

    if df_signals is not None and not df_signals.is_empty():
        buys = df_signals.filter(pl.col("side") == 1)
        sells = df_signals.filter(pl.col("side") == -1)
    else:
        buys = pl.DataFrame([], schema=get_schema("signals"))
        sells = pl.DataFrame([], schema=get_schema("signals"))

    bucket_map = {}
    if not bool(run_cfg.get("force_rebuild_cache", False)):
        try:
            bucket_df = load_backtest_cached(months, era_label, str(regime_id))
            if bucket_df is not None and not bucket_df.is_empty():
                for row in bucket_df.iter_rows(named=True):
                    key = (
                        int(row["sig_n"]),
                        int(row["sig_min_ns"]),
                        int(row["sig_max_ns"]),
                        round(float(row["SL"]), 6),
                        round(float(row["TP"]), 6),
                        int(row["side"]),
                        int(row["exit_window_h"]),
                        int(bool(row.get("use_trailing_sl", False))),
                        round(float(row.get("trailing_sl_pct", 0.0)), 6),
                        int(row.get("trailing_sl_interval", 0)),
                        int(bool(row.get("trailing_sl_stop_at_pos", True))),
                        int(bool(row.get("use_limit_entry", True))),
                        int(row.get("limit_order_expiry_h", 0)),
                        int(row.get("trade_window_interval", 0)),
                    )
                    bucket_map[key] = row
        except Exception as e:
            logger.debug("load_backtest_cached failed for cfg=%s era=%s: %s", regime_id, era_label, e)
            bucket_map = {}

    logger.info(
        "DEBUG: era=%s regime=%s total_signals=%d",
        era_label,
        regime_id,
        int(df_signals.height) if df_signals is not None else 0,
    )

    dd_pass = 0
    dd_fail = 0
    cache_hit = 0
    cache_miss = 0
    staged_equity = 0
    master_rows_written = 0
    empty_rows_written = 0

    for sl_val, tp_val in combos:
        for side_df_pl, side_flag in ((buys, 1), (sells, -1)):
            total_pos = int(side_df_pl.height) if side_df_pl is not None and not side_df_pl.is_empty() else 0

            if total_pos == 0:
                master_row_raw = _make_empty_master_row(
                    regime_id=regime_id,
                    era_int=era_int,
                    side_flag=side_flag,
                    sl_val=sl_val,
                    tp_val=tp_val,
                    regime_cfg=regime_cfg,
                )
                buffer_master_row(results_dir, batch_id, master_row_raw)
                empty_rows_written += 1
                master_rows_written += 1
                continue

            sig_idxs = side_df_pl["idx"].to_numpy(allow_copy=True).astype(np.int64)
            sig_times_ns = side_df_pl["time_ns"].to_numpy(allow_copy=True).astype(np.int64)
            sig_n = int(sig_idxs.size)
            sig_min_ns = int(sig_times_ns.min()) if sig_n > 0 else -1
            sig_max_ns = int(sig_times_ns.max()) if sig_n > 0 else -1

            entry_idx = None
            entry_price_arr = None
            exit_idx = None
            rets = None

            key = (
                sig_n,
                sig_min_ns,
                sig_max_ns,
                round(float(sl_val), 6),
                round(float(tp_val), 6),
                int(side_flag),
                int(regime_cfg.get("exit_window_h", 0)),
                int(bool(regime_cfg.get("use_trailing_sl", False))),
                round(float(regime_cfg.get("trailing_sl_pct", 0.0)), 6),
                int(regime_cfg.get("trailing_sl_interval", 0)),
                int(bool(regime_cfg.get("trailing_sl_stop_at_pos", True))),
                int(use_limit_entry),
                int(limit_order_expiry_h),
                int(trade_window_interval),
            )

            hit = bucket_map.get(key)
            if hit is not None:
                entry_idx = np.asarray(hit.get("entry_idx", []), dtype=np.int64)
                entry_price_arr = np.asarray(hit.get("entry_price", []), dtype=np.float64)
                exit_idx = np.asarray(hit.get("exit_idx", []), dtype=np.int64)
                exit_price_arr = np.asarray(hit.get("exit_price", []), dtype=np.float64)
                rets = np.asarray(hit.get("ret", []), dtype=np.float64)
                exit_reason_arr = np.asarray(hit.get("exit_reason", []), dtype=np.int8)

                if use_limit_entry and entry_price_arr.size != entry_idx.size:
                    hit = None
                    entry_idx = None
                    entry_price_arr = None
                    exit_idx = None
                    exit_price_arr = None
                    rets = None
                    exit_reason_arr = None
                    cache_miss += 1
                else:
                    if exit_price_arr.size != entry_price_arr.size:
                        if side_flag == 1:
                            exit_price_arr = entry_price_arr * (1.0 + rets)
                        else:
                            exit_price_arr = entry_price_arr * (1.0 - rets)

                    if exit_reason_arr.size != entry_idx.size:
                        exit_reason_arr = np.zeros(entry_idx.size, dtype=np.int8)

                    cache_hit += 1
            else:
                cache_miss += 1

            if (sig_idxs < 0).any() or (sig_idxs >= len(main_close_arr)).any():
                logger.error("OOB Index: cfg=%s era=%s sig_idxs out of range", regime_id, era_label)
                continue

            if entry_idx is None:
                try:
                    res = backtest_signals_sl_tp_rets(
                        main_close_arr=main_close_arr,
                        main_high_arr=main_high_arr,
                        main_low_arr=main_low_arr,
                        main_time_ns_arr=main_time_ns_arr,
                        sig_idxs=sig_idxs,
                        sl=float(sl_val),
                        tp=float(tp_val),
                        sl_tp_in_pct=bool(run_cfg.get("sl_tp_in_pct", True)),
                        exit_window_h=int(regime_cfg.get("exit_window_h", 0)),
                        limit_order_expiry_h=limit_order_expiry_h,
                        trade_window_interval=trade_window_interval,
                        base_minutes=int(run_cfg.get("BASE_MINUTES", 5)),
                        spread=float(run_cfg.get("BTC_SETTINGS", {}).get("spread", 0.0)),
                        conservative_sl_first=bool(run_cfg.get("conservative_sl_first", True)),
                        side_flag=side_flag,
                        use_limit_entry=use_limit_entry,
                        use_trailing_sl=bool(regime_cfg.get("use_trailing_sl", False)),
                        trailing_sl_pct=float(regime_cfg.get("trailing_sl_pct", 0.0)),
                        trailing_sl_interval=int(regime_cfg.get("trailing_sl_interval", 0)),
                        trailing_sl_stop_at_pos=bool(regime_cfg.get("trailing_sl_stop_at_pos", True)),
                    )

                    entry_idx = np.asarray(res.get("entry_idx", []), dtype=np.int64)
                    entry_price_arr = np.asarray(res.get("entry_price", []), dtype=np.float64)
                    exit_idx = np.asarray(res.get("exit_idx", []), dtype=np.int64)
                    exit_price_arr = np.asarray(res.get("exit_price", []), dtype=np.float64)
                    rets = np.asarray(res.get("rets", []), dtype=np.float64)
                    exit_reason_arr = np.asarray(res.get("exit_reason", []), dtype=np.int8)

                    filled_pos = int(entry_idx.size)

                    res_df = pl.DataFrame(
                        {
                            "sig_n": [sig_n],
                            "sig_min_ns": [sig_min_ns],
                            "sig_max_ns": [sig_max_ns],
                            "SL": [float(sl_val)],
                            "TP": [float(tp_val)],
                            "side": [int(side_flag)],
                            "regime_id": [int(regime_id)],
                            "exit_window_h": [int(regime_cfg.get("exit_window_h", 0))],
                            "use_trailing_sl": [bool(regime_cfg.get("use_trailing_sl", False))],
                            "trailing_sl_pct": [float(regime_cfg.get("trailing_sl_pct", 0.0))],
                            "trailing_sl_interval": [int(regime_cfg.get("trailing_sl_interval", 0))],
                            "trailing_sl_stop_at_pos": [bool(regime_cfg.get("trailing_sl_stop_at_pos", True))],
                            "use_limit_entry": [bool(use_limit_entry)],
                            "limit_order_expiry_h": [int(limit_order_expiry_h)],
                            "trade_window_interval": [int(trade_window_interval)],
                            "entry_idx": pl.Series([entry_idx], dtype=pl.List(pl.Int64)),
                            "entry_price": pl.Series([entry_price_arr], dtype=pl.List(pl.Float32)),
                            "exit_idx": pl.Series([exit_idx], dtype=pl.List(pl.Int64)),
                            "ret": pl.Series([rets], dtype=pl.List(pl.Float32)),
                            "exit_price": pl.Series([exit_price_arr], dtype=pl.List(pl.Float32)),
                            "exit_reason": pl.Series([exit_reason_arr], dtype=pl.List(pl.Int8)),
                        }
                    ).pipe(enforce_schema, "backtest", strict=True)

                    stage_for_flush("backtest", months, era_label, str(regime_id), res_df)
                except Exception as e:
                    logger.error("Backtest kernel error cfg=%s era=%s: %s", regime_id, era_label, e)
                    continue
            else:
                filled_pos = int(entry_idx.size)

            if entry_idx.size == 0:
                master_row_raw = _make_empty_master_row(
                    regime_id=regime_id,
                    era_int=era_int,
                    side_flag=side_flag,
                    sl_val=sl_val,
                    tp_val=tp_val,
                    regime_cfg=regime_cfg,
                )
                buffer_master_row(results_dir, batch_id, master_row_raw)
                empty_rows_written += 1
                master_rows_written += 1
                continue

            mask_closed = exit_idx >= 0
            if not mask_closed.any():
                master_row_raw = _make_empty_master_row(
                    regime_id=regime_id,
                    era_int=era_int,
                    side_flag=side_flag,
                    sl_val=sl_val,
                    tp_val=tp_val,
                    regime_cfg=regime_cfg,
                )
                buffer_master_row(results_dir, batch_id, master_row_raw)
                master_rows_written += 1
                continue

            closed_rets = rets[mask_closed]
            max_consecutive_losses = compute_max_consecutive_losses(closed_rets)
            closed_entry_idxs = entry_idx[mask_closed].astype(np.int64)
            closed_exit_idxs = exit_idx[mask_closed].astype(np.int64)
            closed_entry_prices = entry_price_arr[mask_closed].astype(np.float64)
            closed_exit_prices = exit_price_arr[mask_closed].astype(np.float64)

            pnl_pct_arr, exit_times_ns, equity_arr, current_max_dd, breached = compute_equity_preview(
                closed_rets=closed_rets,
                closed_entry_idxs=closed_entry_idxs,
                closed_exit_idxs=closed_exit_idxs,
                entry_prices_arr=closed_entry_prices,
                main_close_arr=main_close_arr,
                main_time_ns_arr=main_time_ns_arr,
                main_spread_arr=main_spread_arr,
                main_funding_arr=main_funding_arr,
                sl_val=float(sl_val),
                tp_val=float(tp_val),
                run_cfg=run_cfg,
            )

            if breached:
                dd_fail += 1
                final_balance = float(equity_arr[-1]) if equity_arr.size else 100.0
                win_pos = int(np.count_nonzero(closed_rets > 0.0)) if closed_rets.size else 0

                master_row_raw = _make_master_row(
                    regime_id=regime_id,
                    era_int=era_int,
                    side_flag=side_flag,
                    sl_val=sl_val,
                    tp_val=tp_val,
                    total_pos=filled_pos,
                    win_pos=win_pos,
                    balance=final_balance,
                    max_dd=current_max_dd,
                    max_consecutive_losses=max_consecutive_losses,
                    regime_cfg=regime_cfg,
                )
                buffer_master_row(results_dir, batch_id, master_row_raw)
                master_rows_written += 1
                continue

            dd_pass += 1

            final_balance, win_pos, max_dd = stage_equity_from_preview(
                pnl_pct_arr=pnl_pct_arr,
                exit_times_ns=exit_times_ns,
                equity_arr=equity_arr,
                closed_entry_idxs=closed_entry_idxs,
                closed_exit_idxs=closed_exit_idxs,
                sl_val=float(sl_val),
                tp_val=float(tp_val),
                regime_id=regime_id,
                era_int=era_int,
                side_flag=side_flag,
                stager=stager,
                max_dd=current_max_dd,
            )
            staged_equity += 1

            master_row_raw = _make_master_row(
                regime_id=regime_id,
                era_int=era_int,
                side_flag=side_flag,
                sl_val=sl_val,
                tp_val=tp_val,
                total_pos=filled_pos,
                win_pos=win_pos,
                balance=final_balance,
                max_dd=max_dd,
                max_consecutive_losses=max_consecutive_losses,
                regime_cfg=regime_cfg,
            )
            buffer_master_row(results_dir, batch_id, master_row_raw)

    return {
        "dd_pass": dd_pass,
        "dd_fail": dd_fail,
        "cache_hit": cache_hit,
        "cache_miss": cache_miss,
        "staged_equity": staged_equity,
        "master_rows_written": master_rows_written,
        "empty_rows_written": empty_rows_written,
    }
    
def process_era_combos(
    regime_id: int,
    regime_cfg: dict,
    run_cfg: dict,
    data_ctx: dict,
    results_dir: Path,
    batch_id: int,
    stager,
    max_dd_threshold: float,
    current_idx: int,
    total_in_batch: int,
) -> Tuple[int, int]:
    """
    Global signals once, global trade simulation once per combo/side,
    then slice the global trade arrays by era.
    """

    required_keys = (
        "df_main",
        "main_time_ns_arr",
        "main_close_arr",
        "main_high_arr",
        "main_low_arr",
        "main_spread_arr",
        "main_funding_arr",
        "lookback_map",
    )
    missing = [k for k in required_keys if k not in data_ctx]
    if missing:
        raise RuntimeError(f"data_ctx missing required keys: {missing}")

    df_main: pl.DataFrame = data_ctx["df_main"]
    main_time_ns_arr: np.ndarray = data_ctx["main_time_ns_arr"]
    main_close_arr: np.ndarray = data_ctx["main_close_arr"]
    main_high_arr: np.ndarray = data_ctx["main_high_arr"]
    main_low_arr: np.ndarray = data_ctx["main_low_arr"]
    main_spread_arr: np.ndarray = data_ctx["main_spread_arr"]
    main_funding_arr: np.ndarray = data_ctx["main_funding_arr"]
    lookback_map: dict = data_ctx["lookback_map"]

    months = int(run_cfg["sl_tp_interval_months"])
    grid_start = pd.to_datetime(run_cfg["grid_start_date"]).to_pydatetime()
    grid_end = pd.to_datetime(run_cfg["grid_end_date"]).to_pydatetime()

    sl_vals, tp_vals = _expand_sl_tp({**run_cfg, **regime_cfg})
    combos = _prune_by_min_rr(sl_vals, tp_vals, float(run_cfg.get("min_rr", 3.0)))

    windows = _split_period_windows_from_pl(
        df_main,
        months,
        min_dt=grid_start,
        max_dt=grid_end,
        include_partial_tail=False,
    )

    logger.debug("process_era_combos: regime=%s windows=%d", regime_id, len(windows))

    global_signals = _get_or_generate_global_signals(
        regime_id=regime_id,
        regime_cfg=regime_cfg,
        run_cfg=run_cfg,
        data_ctx=data_ctx,
        months=months,
    )

    era_registry = _build_era_registry(windows, global_signals)

    bucket_maps_by_era = {}
    if not bool(run_cfg.get("force_rebuild_cache", False)):
        for era in era_registry:
            era_label = era["era_label"]
            bucket_map = {}
            try:
                bucket_df = load_backtest_cached(months, era_label, str(regime_id))
                if bucket_df is not None and not bucket_df.is_empty():
                    for row in bucket_df.iter_rows(named=True):
                        key = (
                            int(row["sig_n"]),
                            int(row["sig_min_ns"]),
                            int(row["sig_max_ns"]),
                            round(float(row["SL"]), 6),
                            round(float(row["TP"]), 6),
                            int(row["side"]),
                            int(row["exit_window_h"]),
                            int(bool(row.get("use_trailing_sl", False))),
                            round(float(row.get("trailing_sl_pct", 0.0)), 6),
                            int(row.get("trailing_sl_interval", 0)),
                            int(bool(row.get("trailing_sl_stop_at_pos", True))),
                            int(bool(row.get("use_limit_entry", True))),
                            int(row.get("limit_order_expiry_h", 0)),
                            int(row.get("trade_window_interval", 0)),
                        )
                        bucket_map[key] = row
            except Exception as e:
                logger.debug("load_backtest_cached failed cfg=%s era=%s: %s", regime_id, era_label, e)
            bucket_maps_by_era[era_label] = bucket_map

    if global_signals is not None and not global_signals.is_empty():
        global_buys = global_signals.filter(pl.col("side") == 1)
        global_sells = global_signals.filter(pl.col("side") == -1)
    else:
        global_buys = pl.DataFrame([], schema=get_schema("signals"))
        global_sells = pl.DataFrame([], schema=get_schema("signals"))

    buy_sig_idxs_all = (
        global_buys["idx"].to_numpy(allow_copy=True).astype(np.int64)
        if not global_buys.is_empty()
        else np.asarray([], dtype=np.int64)
    )
    buy_sig_times_all = (
        global_buys["time_ns"].to_numpy(allow_copy=True).astype(np.int64)
        if not global_buys.is_empty()
        else np.asarray([], dtype=np.int64)
    )

    sell_sig_idxs_all = (
        global_sells["idx"].to_numpy(allow_copy=True).astype(np.int64)
        if not global_sells.is_empty()
        else np.asarray([], dtype=np.int64)
    )
    sell_sig_times_all = (
        global_sells["time_ns"].to_numpy(allow_copy=True).astype(np.int64)
        if not global_sells.is_empty()
        else np.asarray([], dtype=np.int64)
    )

    era_master_rows_written = {era["era_label"]: 0 for era in era_registry}

    dd_pass = 0
    dd_fail = 0
    cache_hit = 0
    cache_miss = 0
    staged_equity = 0
    master_rows_written = 0
    empty_rows_written = 0

    use_limit_entry = bool(regime_cfg.get("use_limit_entry", True))
    limit_order_expiry_h = int(regime_cfg.get("limit_order_expiry_h", 0) or 0)
    trade_window_interval = int(regime_cfg.get("trade_window_interval", 0) or 0)

    for sl_val, tp_val in combos:
        for side_flag, side_sig_idxs_all, side_sig_times_all in (
            (1, buy_sig_idxs_all, buy_sig_times_all),
            (-1, sell_sig_idxs_all, sell_sig_times_all),
        ):
            if side_sig_idxs_all.size == 0:
                for era in era_registry:
                    era_label = era["era_label"]
                    master_row_raw = _make_empty_master_row(
                        regime_id=regime_id,
                        era_int=era["era_int"],
                        side_flag=side_flag,
                        sl_val=sl_val,
                        tp_val=tp_val,
                        regime_cfg=regime_cfg,
                    )
                    buffer_master_row(results_dir, batch_id, master_row_raw)
                    era_master_rows_written[era_label] += 1
                    empty_rows_written += 1
                    master_rows_written += 1
                continue

            try:
                res = backtest_signals_sl_tp_rets(
                    main_close_arr=main_close_arr,
                    main_high_arr=main_high_arr,
                    main_low_arr=main_low_arr,
                    main_time_ns_arr=main_time_ns_arr,
                    sig_idxs=side_sig_idxs_all,
                    sl=float(sl_val),
                    tp=float(tp_val),
                    sl_tp_in_pct=bool(run_cfg.get("sl_tp_in_pct", True)),
                    exit_window_h=int(regime_cfg.get("exit_window_h", 0)),
                    limit_order_expiry_h=limit_order_expiry_h,
                    trade_window_interval=trade_window_interval,
                    base_minutes=int(run_cfg.get("BASE_MINUTES", 5)),
                    spread=float(run_cfg.get("BTC_SETTINGS", {}).get("spread", 0.0)),
                    conservative_sl_first=bool(run_cfg.get("conservative_sl_first", True)),
                    side_flag=side_flag,
                    use_limit_entry=use_limit_entry,
                    use_trailing_sl=bool(regime_cfg.get("use_trailing_sl", False)),
                    trailing_sl_pct=float(regime_cfg.get("trailing_sl_pct", 0.0)),
                    trailing_sl_interval=int(regime_cfg.get("trailing_sl_interval", 0)),
                    trailing_sl_stop_at_pos=bool(regime_cfg.get("trailing_sl_stop_at_pos", True)),
                )
            except Exception as e:
                logger.error(
                    "Global backtest kernel error cfg=%s side=%s SL=%.4f TP=%.4f: %s",
                    regime_id,
                    side_flag,
                    float(sl_val),
                    float(tp_val),
                    e,
                )
                continue

            entry_idx_all = np.asarray(res.get("entry_idx", []), dtype=np.int64)
            entry_price_all = np.asarray(res.get("entry_price", []), dtype=np.float64)
            exit_idx_all = np.asarray(res.get("exit_idx", []), dtype=np.int64)
            exit_price_all = np.asarray(res.get("exit_price", []), dtype=np.float64)
            rets_all = np.asarray(res.get("rets", []), dtype=np.float64)
            exit_reason_all = np.asarray(res.get("exit_reason", []), dtype=np.int8)

            if exit_price_all.size != entry_price_all.size:
                if side_flag == 1:
                    exit_price_all = entry_price_all * (1.0 + rets_all)
                else:
                    exit_price_all = entry_price_all * (1.0 - rets_all)

            if exit_reason_all.size != entry_idx_all.size:
                exit_reason_all = np.zeros(entry_idx_all.size, dtype=np.int8)

            if entry_idx_all.size == 0:
                for era in era_registry:
                    era_label = era["era_label"]
                    master_row_raw = _make_empty_master_row(
                        regime_id=regime_id,
                        era_int=era["era_int"],
                        side_flag=side_flag,
                        sl_val=sl_val,
                        tp_val=tp_val,
                        regime_cfg=regime_cfg,
                    )
                    buffer_master_row(results_dir, batch_id, master_row_raw)
                    era_master_rows_written[era_label] += 1
                    empty_rows_written += 1
                    master_rows_written += 1
                continue

            entry_time_ns_all = main_time_ns_arr[entry_idx_all]
            closed_mask_all = exit_idx_all >= 0

            for era in era_registry:
                era_label = era["era_label"]
                start_ns = era["start_ns"]
                end_ns = era["end_ns"]
                bucket_map = bucket_maps_by_era.get(era_label, {})

                if side_flag == 1:
                    sig_n = era["buy_sig_n"]
                    sig_min_ns = era["buy_sig_min_ns"]
                    sig_max_ns = era["buy_sig_max_ns"]
                else:
                    sig_n = era["sell_sig_n"]
                    sig_min_ns = era["sell_sig_min_ns"]
                    sig_max_ns = era["sell_sig_max_ns"]

                if sig_n == 0:
                    master_row_raw = _make_empty_master_row(
                        regime_id=regime_id,
                        era_int=era_int,
                        side_flag=side_flag,
                        sl_val=sl_val,
                        tp_val=tp_val,
                        regime_cfg=regime_cfg,
                    )
                    buffer_master_row(results_dir, batch_id, master_row_raw)
                    era_master_rows_written[era_label] += 1
                    empty_rows_written += 1
                    master_rows_written += 1
                    continue

                entry_idx = None
                entry_price_arr = None
                exit_idx = None
                rets = None

                key = (
                    int(sig_n),
                    int(sig_min_ns),
                    int(sig_max_ns),
                    round(float(sl_val), 6),
                    round(float(tp_val), 6),
                    int(side_flag),
                    int(regime_cfg.get("exit_window_h", 0)),
                    int(bool(regime_cfg.get("use_trailing_sl", False))),
                    round(float(regime_cfg.get("trailing_sl_pct", 0.0)), 6),
                    int(regime_cfg.get("trailing_sl_interval", 0)),
                    int(bool(regime_cfg.get("trailing_sl_stop_at_pos", True))),
                    int(use_limit_entry),
                    int(limit_order_expiry_h),
                    int(trade_window_interval),
                )

                hit = bucket_map.get(key)
                if hit is not None:
                    entry_idx = np.asarray(hit.get("entry_idx", []), dtype=np.int64)
                    entry_price_arr = np.asarray(hit.get("entry_price", []), dtype=np.float64)
                    exit_idx = np.asarray(hit.get("exit_idx", []), dtype=np.int64)
                    exit_price_arr = np.asarray(hit.get("exit_price", []), dtype=np.float64)
                    rets = np.asarray(hit.get("ret", []), dtype=np.float64)
                    exit_reason_arr = np.asarray(hit.get("exit_reason", []), dtype=np.int8)

                    if use_limit_entry and entry_price_arr.size != entry_idx.size:
                        hit = None
                        entry_idx = None
                        entry_price_arr = None
                        exit_idx = None
                        exit_price_arr = None
                        rets = None
                        exit_reason_arr = None
                        cache_miss += 1
                    else:
                        if exit_price_arr.size != entry_price_arr.size:
                            if side_flag == 1:
                                exit_price_arr = entry_price_arr * (1.0 + rets)
                            else:
                                exit_price_arr = entry_price_arr * (1.0 - rets)

                        if exit_reason_arr.size != entry_idx.size:
                            exit_reason_arr = np.zeros(entry_idx.size, dtype=np.int8)

                        cache_hit += 1
                else:
                    cache_miss += 1

                if hit is None:
                    cache_miss += 1
                    trade_mask = closed_mask_all & (entry_time_ns_all >= start_ns) & (entry_time_ns_all < end_ns)

                    if not trade_mask.any():
                        master_row_raw = _make_empty_master_row(
                            regime_id=regime_id,
                            era_int=era_int,
                            side_flag=side_flag,
                            sl_val=sl_val,
                            tp_val=tp_val,
                            regime_cfg=regime_cfg,
                        )
                        buffer_master_row(results_dir, batch_id, master_row_raw)
                        era_master_rows_written[era_label] += 1
                        empty_rows_written += 1
                        master_rows_written += 1
                        continue

                    entry_idx = entry_idx_all[trade_mask].astype(np.int64)
                    entry_price_arr = entry_price_all[trade_mask].astype(np.float64)
                    exit_idx = exit_idx_all[trade_mask].astype(np.int64)
                    exit_price_arr = exit_price_all[trade_mask].astype(np.float64)
                    rets = rets_all[trade_mask].astype(np.float64)
                    exit_reason_arr = exit_reason_all[trade_mask].astype(np.int8)

                    try:
                        res_df = pl.DataFrame(
                            {
                                "sig_n": [sig_n],
                                "sig_min_ns": [sig_min_ns],
                                "sig_max_ns": [sig_max_ns],
                                "SL": [float(sl_val)],
                                "TP": [float(tp_val)],
                                "side": [int(side_flag)],
                                "regime_id": [int(regime_id)],
                                "exit_window_h": [int(regime_cfg.get("exit_window_h", 0))],
                                "use_trailing_sl": [bool(regime_cfg.get("use_trailing_sl", False))],
                                "trailing_sl_pct": [float(regime_cfg.get("trailing_sl_pct", 0.0))],
                                "trailing_sl_interval": [int(regime_cfg.get("trailing_sl_interval", 0))],
                                "trailing_sl_stop_at_pos": [bool(regime_cfg.get("trailing_sl_stop_at_pos", True))],
                                "use_limit_entry": [bool(use_limit_entry)],
                                "limit_order_expiry_h": [int(limit_order_expiry_h)],
                                "trade_window_interval": [int(trade_window_interval)],
                                "entry_idx": pl.Series([entry_idx], dtype=pl.List(pl.Int64)),
                                "entry_price": pl.Series([entry_price_arr], dtype=pl.List(pl.Float32)),
                                "exit_idx": pl.Series([exit_idx], dtype=pl.List(pl.Int64)),
                                "ret": pl.Series([rets], dtype=pl.List(pl.Float32)),
                                "exit_price": pl.Series([exit_price_arr], dtype=pl.List(pl.Float32)),
                                "exit_reason": pl.Series([exit_reason_arr], dtype=pl.List(pl.Int8)),
                            }
                        ).pipe(enforce_schema, "backtest", strict=True)

                        stage_for_flush("backtest", months, era_label, str(regime_id), res_df)
                    except Exception as e:
                        logger.error("Backtest cache write failed cfg=%s era=%s: %s", regime_id, era_label, e)

                if entry_idx is None or entry_idx.size == 0:
                    master_row_raw = _make_empty_master_row(
                        regime_id=regime_id,
                        era_int=era["era_int"],
                        side_flag=side_flag,
                        sl_val=sl_val,
                        tp_val=tp_val,
                        regime_cfg=regime_cfg,
                    )
                    buffer_master_row(results_dir, batch_id, master_row_raw)
                    era_master_rows_written[era_label] += 1
                    empty_rows_written += 1
                    master_rows_written += 1
                    continue

                mask_closed = exit_idx >= 0
                if not mask_closed.any():
                    master_row_raw = _make_empty_master_row(
                        regime_id=regime_id,
                        era_int=era["era_int"],
                        side_flag=side_flag,
                        sl_val=sl_val,
                        tp_val=tp_val,
                        regime_cfg=regime_cfg,
                    )
                    buffer_master_row(results_dir, batch_id, master_row_raw)
                    era_master_rows_written[era_label] += 1
                    empty_rows_written += 1
                    master_rows_written += 1
                    continue

                closed_rets = rets[mask_closed]
                closed_entry_idxs = entry_idx[mask_closed].astype(np.int64)
                closed_exit_idxs = exit_idx[mask_closed].astype(np.int64)
                closed_entry_prices = entry_price_arr[mask_closed].astype(np.float64)
                closed_exit_prices = exit_price_arr[mask_closed].astype(np.float64)

                pnl_pct_arr, exit_times_ns, equity_arr, current_max_dd, breached = compute_equity_preview(
                    closed_entry_prices_arr=closed_entry_prices,
                    closed_exit_prices_arr=closed_exit_prices,
                    closed_entry_idxs=closed_entry_idxs,
                    closed_exit_idxs=closed_exit_idxs,
                    main_close_arr=main_close_arr,
                    main_time_ns_arr=main_time_ns_arr,
                    main_spread_arr=main_spread_arr,
                    main_funding_arr=main_funding_arr,
                    sl_val=float(sl_val),
                    tp_val=float(tp_val),
                    side_flag=side_flag,
                    run_cfg=run_cfg,
                )

                if breached:
                    dd_fail += 1
                    final_balance = float(equity_arr[-1]) if equity_arr.size else 100.0
                    win_pos = int(np.count_nonzero(closed_rets > 0.0)) if closed_rets.size else 0

                    master_row_raw = _make_master_row(
                        regime_id=regime_id,
                        era_int=era["era_int"],
                        side_flag=side_flag,
                        sl_val=sl_val,
                        tp_val=tp_val,
                        total_pos=int(entry_idx.size),
                        win_pos=win_pos,
                        balance=final_balance,
                        max_dd=current_max_dd,
                        max_consecutive_losses=compute_max_consecutive_losses(closed_rets),
                        regime_cfg=regime_cfg,
                    )
                    buffer_master_row(results_dir, batch_id, master_row_raw)
                    era_master_rows_written[era_label] += 1
                    master_rows_written += 1
                    continue

                dd_pass += 1

                final_balance, win_pos, max_dd = stage_equity_from_preview(
                    pnl_pct_arr=pnl_pct_arr,
                    exit_times_ns=exit_times_ns,
                    equity_arr=equity_arr,
                    closed_entry_idxs=closed_entry_idxs,
                    closed_exit_idxs=closed_exit_idxs,
                    sl_val=float(sl_val),
                    tp_val=float(tp_val),
                    regime_id=regime_id,
                    era_int=era["era_int"],
                    side_flag=side_flag,
                    stager=stager,
                    max_dd=current_max_dd,
                )
                staged_equity += 1

                master_row_raw = _make_master_row(
                    regime_id=regime_id,
                    era_int=era["era_int"],
                    side_flag=side_flag,
                    sl_val=sl_val,
                    tp_val=tp_val,
                    total_pos=int(entry_idx.size),
                    win_pos=win_pos,
                    balance=final_balance,
                    max_dd=max_dd,
                    max_consecutive_losses=compute_max_consecutive_losses(closed_rets),
                    regime_cfg=regime_cfg,
                )
                buffer_master_row(results_dir, batch_id, master_row_raw)

    empty_era_streak = 0
    max_empty_era_streak = int(run_cfg.get("MAX_EMPTY_ERA_STREAK", 2))
    for era in era_registry:
        era_label = era["era_label"]
        if era_master_rows_written.get(era_label, 0) == 0:
            empty_era_streak += 1
            if empty_era_streak >= max_empty_era_streak:
                error_msg = f"❌ CRITICAL FAILURE: Regime {regime_id} killed. Hit max empty streak of {max_empty_era_streak}."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        else:
            empty_era_streak = 0

    processed_inc = sum(1 for era in era_registry if era_master_rows_written.get(era["era_label"], 0) > 0)
    skipped_inc = len(era_registry) - processed_inc

    return {
        "processed_inc": processed_inc,
        "skipped_inc": skipped_inc,
        "dd_pass": dd_pass,
        "dd_fail": dd_fail,
        "cache_hit": cache_hit,
        "cache_miss": cache_miss,
        "staged_equity": staged_equity,
        "master_rows_written": master_rows_written,
        "empty_rows_written": empty_rows_written,
    }
    
def compute_config_and_save(batch_path: str, session_dir: str, compute_backtest: bool = True) -> Dict:
    try:
        _HAS_PSUTIL = True
        _PROC = psutil.Process()
    except Exception:
        _HAS_PSUTIL = False
        _PROC = None

    batch_path = Path(batch_path)
    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise AirflowFailException(f"FATAL: session_dir NOT FOUND at {session_dir}")

    session_snapshot = session_dir / "run_config.json"
    if not session_snapshot.exists():
        raise AirflowFailException(f"FATAL: session run_config.json missing at {session_snapshot}")

    with open(session_snapshot, "r", encoding="utf8") as f:
        run_cfg = json.load(f)

    logger.info("✅ Loaded session-specific run_config from snapshot.")

    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(batch_path, "r", encoding="utf8") as f:
        batch_data = json.load(f)

    regimes = batch_data["regimes"]
    batch_id = int(batch_data.get("batch_id", 0))
    logger.info("🧵 Worker started: %s | Regimes to process: %d (batch_id=%d)", batch_path.name, len(regimes), batch_id)

    prepared = prepare_worker_data(session_dir, run_cfg)
    data_ctx = prepared  # keep the entire prepared dict intact; pass around as context

    if "lookback_map" not in data_ctx or not isinstance(data_ctx["lookback_map"], dict):
        raise RuntimeError("prepare_worker_data did not return a valid 'lookback_map'. Ensure precompute produced it.")

    max_dd_threshold = float(run_cfg.get("max_dd_threshold", 0.20))
    flush_rows = int(run_cfg.get("PARQUET_FLUSH_ROWS", 100_000))
    partition_by = str(run_cfg.get("EQUITY_PARTITION_BY", "era_int"))

    # Use session-local tmp dir so coordinator can find parts (session_dir/equity_partitioned/_tmp)
    equity_base = session_dir / "equity_partitioned"
    equity_tmp = equity_base / "_tmp"
    stager = EquityStager(
        equity_part_base=equity_base,
        batch_id=batch_id,
        flush_rows=flush_rows,
        partition_by=partition_by,
        tmp_dir=equity_tmp,
    )

    processed_count = 0
    skipped_count = 0

    container_limit_mb = _get_memory_guard_mb()
    rss_mb = _get_process_rss_mb()

    if container_limit_mb > 0:
        # keep some headroom for Polars / PyArrow / Python allocator spikes
        soft_cap_mb = container_limit_mb * 0.80
        if rss_mb >= soft_cap_mb:
            logger.error("OOM RISK: process RSS %.1f MB is near container limit %.1f MB", rss_mb, container_limit_mb)
            raise AirflowFailException(
                f"Process memory exceeded safe threshold: {rss_mb:.1f} MB / {container_limit_mb:.1f} MB"
            )

    try:
        total_in_batch = len(regimes)
        for idx, regime_cfg in enumerate(regimes):
            if _HAS_PSUTIL and _PROC:
                mem_info = psutil.virtual_memory()
                if mem_info.percent > 92.0:
                    logger.error("OOM RISK: System Memory at %s%%", mem_info.percent)
                    raise AirflowFailException(f"System memory threshold exceeded ({mem_info.percent}%).")

            gc_every_n = int(run_cfg.get("GC_EVERY_N_REGIMES", 0) or 0)
            gc_rss_limit_mb = int(run_cfg.get("GC_RSS_LIMIT_MB", 0) or 0)

            if gc_every_n > 0 and idx % gc_every_n == 0:
                try:
                    rss_mb = _get_process_rss_mb()
                    if gc_rss_limit_mb > 0 and rss_mb >= gc_rss_limit_mb:
                        gc.collect()
                except Exception:
                    gc.collect()

            if idx % 10 == 0:
                heartbeat_log(f"batch{batch_id}_progress", {"idx": idx, "regime_id": str(regime_cfg.get("regime_id"))})

            # normalize regime types
            try:
                regime_cfg["regime_id"] = int(regime_cfg.get("regime_id"))
            except Exception:
                logger.debug("regime_id not int-convertible: %s", regime_cfg.get("regime_id"))
            regime_cfg["ma_int"] = int(regime_cfg.get("ma_int", 0) or 0)
            regime_cfg["ma_reversion"] = bool(regime_cfg.get("ma_reversion", False))
            lb_val = regime_cfg.get("entry_lookback_units", 0)
            regime_cfg["entry_lookback_units"] = lb_val if isinstance(lb_val, list) else int(lb_val or 0)
            regime_cfg["exit_window_h"] = int(regime_cfg.get("exit_window_h", 0) or 0)
            regime_cfg["use_stochastic"] = bool(regime_cfg.get("use_stochastic", False))

            regime_cfg["bbw_periods"] = int(regime_cfg.get("bbw_periods", 0) or 0)
            regime_cfg["bbw_std"] = float(regime_cfg.get("bbw_std", 0.0) or 0.0)
            regime_cfg["use_bbw"] = bool(regime_cfg.get("use_bbw", False))
            regime_cfg["bbw_thresholds"] = int(regime_cfg.get("bbw_thresholds", 0) or 0)

            regime_cfg["use_trailing_sl"] = bool(regime_cfg.get("use_trailing_sl", False))
            regime_cfg["trailing_sl_pct"] = float(regime_cfg.get("trailing_sl_pct", 0.0) or 0.0)
            regime_cfg["trailing_sl_interval"] = int(regime_cfg.get("trailing_sl_interval", 0) or 0)
            regime_cfg["trailing_sl_stop_at_pos"] = bool(regime_cfg.get("trailing_sl_stop_at_pos", True))

            regime_cfg["use_limit_entry"] = bool(regime_cfg.get("use_limit_entry", True))
            regime_cfg["limit_order_expiry_h"] = int(regime_cfg.get("limit_order_expiry_h", 0) or 0)

            regime_cfg["trade_window_interval"] = int(regime_cfg.get("trade_window_interval", 0) or 0)

            regime_id = regime_cfg.get("regime_id")
            summary_path = results_dir / f"cfg_{regime_id}_summary.json"
            if summary_path.exists():
                skipped_count += 1
                continue

            if processed_count % 5 == 0 or processed_count == 0:
                logger.info("📑 [Batch %d] Progress: %d/%d (%.1f%%) ...", batch_id, idx, total_in_batch, (idx / total_in_batch) * 100.0)

            with maybe_profile(f"regime_{regime_id}_process"):
                result = process_era_combos(
                    regime_id=regime_id,
                    regime_cfg=regime_cfg,
                    run_cfg=run_cfg,
                    data_ctx=data_ctx,
                    results_dir=results_dir,
                    batch_id=batch_id,
                    stager=stager,
                    max_dd_threshold=max_dd_threshold,
                    current_idx=idx,
                    total_in_batch=total_in_batch,
                )

            if isinstance(result, dict):
                p_inc = int(result.get("processed_inc", 0))
                s_inc = int(result.get("skipped_inc", 0))
            else:
                p_inc, s_inc = result

            processed_count += p_inc
            skipped_count += s_inc

            final_output = {"regime_id": regime_id, "regime_params": regime_cfg, "combos_tested": 0, "results": []}
            with open(results_dir / f"cfg_{regime_id}_summary.json", "w", encoding="utf8") as fh:
                json.dump(final_output, fh, indent=2)

    except Exception as e:
        logger.error("Error processing batch %s: %s", batch_path.name, e)
        raise
    finally:
        try:
            # spill staged equity parts to tmp files (workers do not merge final files)
            stager.flush_all()
            _flush_master_rows_buffer(results_dir, batch_id)
        except Exception as e:
            logger.debug("Flush local stage failed: %s", e)

        try:
            flush_all_buffers()
        except Exception:
            logger.debug("flush_all_buffers() failed")

        try:
            merge_batch_master_parts(session_dir, batch_id)
        except Exception as e:
            logger.exception("Failed to merge master parts for batch %s: %s", batch_id, e)
            raise

    logger.info("🏁 Batch %s finished. Processed: %d, Skipped: %d", batch_path.name, processed_count, skipped_count)
    return {"batch": batch_path.name, "processed": processed_count, "skipped": skipped_count}