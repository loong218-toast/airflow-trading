from __future__ import annotations

import datetime
import gc
import json
import logging
import math
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import numpy as np
import pandas as pd
import polars as pl
import psutil
import pyarrow.parquet as pq
from airflow.exceptions import AirflowFailException
from numba import njit

from common.cache import (
    flush_all_buffers,
    load_backtest_cached,
    load_global_signals_cached,
    stage_for_flush,
    stage_global_signals,
    build_scope_cache_id
)
from common.feature_helpers import (
    build_signal_json,
    generate_filtered_signals,
    signal_scope_text,
)

from common.feature_prep import (
    load_prepared_feature_ref,
    precompute_all_possible_features,
    prepare_feature_cache,
)
from common.profiling import maybe_profile
from common.schema import cast_to_schema, classify_fragment, enforce_schema, get_schema
from etl.transform import build_df_main_from_5m_polars, load_candles_from_db_polars
from research.backtest import (
    backtest_signals_sl_tp_rets,
    compute_max_consecutive_losses,
    fast_compound_equity_gate,
)
from research.grid_row_builders import (
    _make_empty_master_row,
    _make_master_row,
    build_trade_ml_rows_from_backtest,
)
from research.io_utils import FULL_LAKE_DIR
from research.master_io_utils import _flush_master_rows_buffer, buffer_master_row
from research.trade_ml_stager import TradeMLStager
from research.merge_utils import combine_cache, combine_results_to_master, merge_batch_master_parts
from research.equitystager import EquityStager, stage_equity_from_preview

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

def _summary_path_for_regime(results_dir: Path, regime_cfg: dict) -> Path:
    """
    One summary file per canonical regime.

    The file name is based on the canonical regime signature only so it stays
    stable even if regime_id changes due to reordering or dedupe.
    """
    results_dir = Path(results_dir)
    regime_key = str(regime_cfg.get("regime_key") or _regime_signature(regime_cfg))
    digest = hashlib.sha1(regime_key.encode("utf8")).hexdigest()[:16]
    return results_dir / f"cfg_{digest}_summary.json"

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

def _safe_token(value: Any, default: str = "x") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
    return text or default


def _scope_cache_id(months: int, signal_layer: int, signal_scope_id: str, signal_json: str) -> str:
    """
    Cache key for global signals.

    This should depend on the signal definition only, not on SL/TP,
    because the global signal set is independent of backtest sizing.
    """
    scope = _safe_token(signal_scope_id, "scope")
    payload = str(signal_json or "")
    digest = hashlib.sha1(payload.encode("utf8")).hexdigest()[:16]
    return f"m{int(months)}__l{int(signal_layer)}__{scope}__{digest}"

def _get_or_generate_global_signals(
    regime_id: int,
    regime_cfg: dict,
    run_cfg: dict,
    data_ctx: dict,
    months: int,
    signal_layer: int,
    signal_scope_id: str,
) -> pl.DataFrame:
    """
    Build one global signal set for the full date range of the regime.
    Cache it under a scope-aware key so single-slot, pair-slot, and full-scope
    regimes do not collide.
    """
    force_rebuild_cache = bool(run_cfg.get("force_rebuild_cache", False))
    df_main: pl.DataFrame = data_ctx["df_main"]

    cache_id = _scope_cache_id(months, signal_layer, signal_scope_id, run_cfg.get("signal_json", ""))

    if not force_rebuild_cache:
        try:
            cached = load_global_signals_cached(months, cache_id)
            if cached is not None and not cached.is_empty():
                cached = enforce_schema(cached, "signals", strict=False)
                logger.info(
                    "global_signals cache hit cfg=%s rows=%d buys=%d sells=%d",
                    cache_id,
                    cached.height,
                    int(cached.filter(pl.col("side") == 1).height),
                    int(cached.filter(pl.col("side") == -1).height),
                )
                return cached
        except Exception as e:
            logger.debug("global signal cache read failed cfg=%s: %s", cache_id, e)

    df_signals, stats = generate_filtered_signals(
        df_main,
        {**run_cfg, **regime_cfg, "regime_id": regime_id},
        df_context=df_main,
        return_stats=True,
    )

    logger.info(
        "global_signals stats cfg=%s input=%d final=%d buys=%d sells=%d ma=%s stoch=%s lookback=%s bbw=%s",
        cache_id,
        int(stats.get("input_rows", 0)),
        int(stats.get("final_total_signals", 0)),
        int(stats.get("final_buy_signals", 0)),
        int(stats.get("final_sell_signals", 0)),
        bool(stats.get("ma_enabled", False)),
        bool(stats.get("stochastic_enabled", False)),
        bool(stats.get("lookback_enabled", False)),
        bool(stats.get("bbw_enabled", False)),
    )

    if not df_signals.is_empty():
        try:
            signals_for_cache = df_signals.select(["idx", "time_ns", "side", "regime_id"])
            signals_for_cache = enforce_schema(signals_for_cache, "signals", strict=True)
            stage_global_signals(months, cache_id, signals_for_cache)
        except Exception as e:
            logger.debug("stage_for_flush(global signals) failed cfg=%s: %s", cache_id, e)

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
        })

    return out

def _backtest_cache_key(
    sig_n: int,
    sig_min_ns: int,
    sig_max_ns: int,
    sl_val: float,
    tp_val: float,
    side_flag: int,
    exit_window_h: int,
    use_trailing_sl: bool,
    trailing_sl_pct: float,
    trailing_sl_interval: int,
    trailing_sl_stop_at_pos: bool,
    use_limit_entry: bool,
    limit_order_expiry_bars: int,
    trade_window_interval: int,
    trade_overlap: bool,
    trade_flip_on_entry: bool,
    signal_layer: int,
    signal_scope_id: str,
) -> tuple:
    # This key must match the cached backtest parquet row shape exactly.
    # If any parameter changes, the cache entry should miss and recompute.
    return (
        int(sig_n),
        int(sig_min_ns),
        int(sig_max_ns),
        round(float(sl_val), 6),
        round(float(tp_val), 6),
        int(side_flag),
        int(exit_window_h),
        int(bool(use_trailing_sl)),
        round(float(trailing_sl_pct), 6),
        int(trailing_sl_interval),
        int(bool(trailing_sl_stop_at_pos)),
        int(bool(use_limit_entry)),
        int(limit_order_expiry_bars),
        int(trade_window_interval),
        int(bool(trade_overlap)),
        int(bool(trade_flip_on_entry)),
        int(signal_layer),
        str(signal_scope_id or ""),
    )

def process_era_combos(
    regime_id: int,
    regime_cfg: dict,
    run_cfg: dict,
    data_ctx: dict,
    session_dir: Path,
    results_dir: Path,
    batch_id: int,
    equity_stager,
    trade_ml_stager,
    signal_layer: int,
    signal_scope_id: str,
    max_dd_threshold: float,
    current_idx: int,
    total_in_batch: int,
) -> Dict[str, Any]:
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

    signal_layer = int(signal_layer)
    signal_scope_id = str(signal_scope_id or regime_cfg.get("signal_scope_id") or "").strip().lower()

    if signal_scope_id.startswith("baseline:") and signal_layer != 0:
        logger.warning(
            "process_era_combos: forcing baseline regime to signal_layer=0 cfg=%s scope=%s",
            regime_id,
            signal_scope_id,
        )
        signal_layer = 0

    if not signal_scope_id and signal_layer == 0:
        signal_scope_id = "baseline:all_buy"

    months = int(run_cfg["sl_tp_interval_months"])
    grid_start = pd.to_datetime(run_cfg["grid_start_date"]).to_pydatetime()
    grid_end = pd.to_datetime(run_cfg["grid_end_date"]).to_pydatetime()

    # One regime should now represent one SL/TP combo.
    sl_val = float(regime_cfg.get("SL", run_cfg.get("SL", 0.2)) or 0.2)
    tp_val = float(regime_cfg.get("TP", run_cfg.get("TP", 6.0)) or 6.0)
    combos = [(sl_val, tp_val)]

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
        signal_layer=signal_layer,
        signal_scope_id=signal_scope_id,
    )

    if global_signals is None or global_signals.is_empty():
        logger.warning(
            "process_era_combos: regime=%s produced zero global signals. "
            "That usually means the signal config or signal filters are too restrictive.",
            regime_id,
        )
    else:
        total_g = int(global_signals.height)
        buy_g = int(global_signals.filter(pl.col("side") == 1).height)
        sell_g = int(global_signals.filter(pl.col("side") == -1).height)
        logger.info(
            "process_era_combos: regime=%s global_signals=%d buys=%d sells=%d",
            regime_id,
            total_g,
            buy_g,
            sell_g,
        )

    era_registry = _build_era_registry(windows, global_signals)

    if global_signals is not None and not global_signals.is_empty():
        all_sig_idxs_all = global_signals["idx"].to_numpy(allow_copy=True).astype(np.int64)
        all_sig_times_all = global_signals["time_ns"].to_numpy(allow_copy=True).astype(np.int64)
        all_sig_sides_all = global_signals["side"].to_numpy(allow_copy=True).astype(np.int8)
    else:
        all_sig_idxs_all = np.asarray([], dtype=np.int64)
        all_sig_times_all = np.asarray([], dtype=np.int64)
        all_sig_sides_all = np.asarray([], dtype=np.int8)

    era_master_rows_written = {era["era_label"]: 0 for era in era_registry}

    dd_pass = 0
    dd_fail = 0
    cache_hit = 0
    cache_miss = 0
    staged_equity = 0
    staged_trade_ml = 0
    master_rows_written = 0
    empty_rows_written = 0

    use_limit_entry = bool(regime_cfg.get("use_limit_entry", True))
    trade_flip_on_entry = bool(regime_cfg.get("trade_flip_on_entry", False))
    trade_overlap = bool(regime_cfg.get("trade_overlap", True))
    limit_order_expiry_bars = int(regime_cfg.get("limit_order_expiry_bars", 0) or 0)
    trade_window_interval = int(regime_cfg.get("trade_window_interval", 0) or 0)

    # Baseline should behave like a true control case.
    # If you keep limit-entry on here, the "all bars" baseline can get starved by pending-order cancellation.
    is_baseline = signal_scope_id.startswith("baseline:")
    use_limit_entry_eff = False if is_baseline else use_limit_entry

    logger.info(
        "process_era_combos: regime=%s combos=%d eras=%d side_scope=%s baseline=%s",
        regime_id,
        len(combos),
        len(era_registry),
        signal_scope_id,
        is_baseline,
    )

    try:
        for sl_val, tp_val in combos:
            res = None
            entry_idx_all = None
            entry_price_all = None
            exit_idx_all = None
            exit_price_all = None
            rets_all = None
            exit_reason_all = None

            try:
                if all_sig_idxs_all.size == 0:
                    for era in era_registry:
                        era_label = era["era_label"]
                        master_row_raw = _make_empty_master_row(
                            regime_id=regime_id,
                            signal_layer=signal_layer,
                            era_int=era["era_int"],
                            sl_val=sl_val,
                            tp_val=tp_val,
                            regime_cfg=regime_cfg,
                            run_cfg=run_cfg,
                            signal_scope_id=signal_scope_id,
                        )
                        buffer_master_row(results_dir, batch_id, master_row_raw)
                        era_master_rows_written[era_label] += 1
                        empty_rows_written += 1
                        master_rows_written += 1
                    continue

                # --- FIX: Determine side flag from signals ---
                # If signals are all one side, use that side. 
                # If mixed, we usually use 0 or handle accordingly based on your kernel's spec.
                unique_sides = np.unique(all_sig_sides_all)
                if unique_sides.size == 1:
                    effective_side_flag = int(unique_sides[0])
                else:
                    # If mixed (like in regime 27), the kernel usually needs a specific flag
                    # to process both, or 1 to indicate "use the side in the array"
                    effective_side_flag = 1 

                res = backtest_signals_sl_tp_rets(
                    main_close_arr=main_close_arr,
                    main_high_arr=main_high_arr,
                    main_low_arr=main_low_arr,
                    main_time_ns_arr=main_time_ns_arr,
                    sig_idxs=all_sig_idxs_all,
                    sl=float(sl_val),
                    tp=float(tp_val),
                    sl_tp_in_pct=bool(run_cfg.get("sl_tp_in_pct", True)),
                    spread_is_percent=bool(run_cfg.get("spread_is_percent", True)),
                    exit_window_h=int(regime_cfg.get("exit_window_h", 0)),
                    limit_order_expiry_bars=limit_order_expiry_bars,
                    trade_window_interval=trade_window_interval,
                    base_minutes=int(run_cfg.get("BASE_MINUTES", 5)),
                    spread=float(run_cfg.get("BTC_SETTINGS", {}).get("spread", 0.0)),
                    conservative_sl_first=bool(run_cfg.get("conservative_sl_first", True)),
                    side_flag=effective_side_flag,
                    use_limit_entry=use_limit_entry_eff,
                    trade_overlap=trade_overlap,
                    trade_flip_on_entry=trade_flip_on_entry,
                    use_trailing_sl=bool(regime_cfg.get("use_trailing_sl", False)),
                    trailing_sl_pct=float(regime_cfg.get("trailing_sl_pct", 0.0)),
                    trailing_sl_interval=int(regime_cfg.get("trailing_sl_interval", 0)),
                    trailing_sl_stop_at_pos=bool(regime_cfg.get("trailing_sl_stop_at_pos", True)),
                )
            except Exception as e:
                logger.error(
                    "Global backtest kernel error cfg=%s SL=%.4f TP=%.4f: %s",
                    regime_id,
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
            side_all = np.asarray(res.get("side", []), dtype=np.int8)

            if exit_price_all.size != entry_price_all.size and entry_price_all.size == rets_all.size:
                exit_price_all = entry_price_all * (1.0 + rets_all)

            if exit_reason_all.size != entry_idx_all.size:
                exit_reason_all = np.zeros(entry_idx_all.size, dtype=np.int8)

            if entry_idx_all.size == 0:
                for era in era_registry:
                    era_label = era["era_label"]
                    master_row_raw = _make_empty_master_row(
                        regime_id=regime_id,
                        signal_layer=signal_layer,
                        era_int=era["era_int"],
                        sl_val=sl_val,
                        tp_val=tp_val,
                        regime_cfg=regime_cfg,
                        run_cfg=run_cfg,
                        signal_scope_id=signal_scope_id,
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

                sig_n = era["buy_sig_n"] + era["sell_sig_n"]

                if sig_n == 0:
                    master_row_raw = _make_empty_master_row(
                        regime_id=regime_id,
                        signal_layer=signal_layer,
                        era_int=era["era_int"],
                        sl_val=sl_val,
                        tp_val=tp_val,
                        regime_cfg=regime_cfg,
                        run_cfg=run_cfg,
                        signal_scope_id=signal_scope_id,
                    )
                    buffer_master_row(results_dir, batch_id, master_row_raw)
                    era_master_rows_written[era_label] += 1
                    empty_rows_written += 1
                    master_rows_written += 1
                    continue

                trade_mask = closed_mask_all & (entry_time_ns_all >= start_ns) & (entry_time_ns_all < end_ns)

                if not trade_mask.any():
                    master_row_raw = _make_empty_master_row(
                        regime_id=regime_id,
                        signal_layer=signal_layer,
                        era_int=era["era_int"],
                        sl_val=sl_val,
                        tp_val=tp_val,
                        regime_cfg=regime_cfg,
                        run_cfg=run_cfg,
                        signal_scope_id=signal_scope_id,
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
                trade_sides = side_all[trade_mask] if side_all.size == side_all.shape[0] else np.full(trade_mask.sum(), 1, dtype=np.int8)

                closed_rets = rets
                closed_entry_idxs = entry_idx.astype(np.int64)
                closed_exit_idxs = exit_idx.astype(np.int64)

                pnl_pct_arr = np.nan_to_num(
                    closed_rets.astype(np.float64),
                    nan=0.0,
                    posinf=1e37,
                    neginf=-1e37,
                )
                exit_times_ns = np.asarray(main_time_ns_arr[closed_exit_idxs], dtype=np.int64)

                equity_arr, current_max_dd, breached = fast_compound_equity_gate(
                    pnl_pct_arr,
                    100.0,
                    float(max_dd_threshold),
                )
                equity_arr = np.nan_to_num(equity_arr, nan=0.0, posinf=1e37, neginf=-1e37)

                # Stage trade ML regardless of drawdown gate.
                # This keeps the trade dataset complete even when an era breaches max DD.
                try:
                    era_mask = (all_sig_times_all >= start_ns) & (all_sig_times_all < end_ns)
                    trade_sig_idxs = all_sig_idxs_all[era_mask] if era_mask.any() else np.asarray([], dtype=np.int64)
                    trade_sig_sides = all_sig_sides_all[era_mask] if era_mask.any() else np.asarray([], dtype=np.int8)

                    trade_ml_df = build_trade_ml_rows_from_backtest(
                        df_main=df_main,
                        main_close_arr=main_close_arr,
                        main_high_arr=main_high_arr,
                        main_low_arr=main_low_arr,
                        main_time_ns_arr=main_time_ns_arr,
                        sig_idxs=trade_sig_idxs,
                        side_arr=trade_sig_sides,
                        sl_val=sl_val,
                        tp_val=tp_val,
                        use_limit_entry=use_limit_entry_eff,
                        limit_order_expiry_bars=limit_order_expiry_bars,
                        trade_window_interval=trade_window_interval,
                        regime_id=regime_id,
                        era_int=era["era_int"],
                        backtest_res=res,
                        regime_cfg=regime_cfg,
                        run_cfg=run_cfg,
                        signal_layer=signal_layer,
                        signal_scope_id=signal_scope_id,
                    )

                    if trade_ml_df is not None and not trade_ml_df.is_empty():
                        trade_ml_stager.stage(str(era["era_int"]), trade_ml_df)
                        staged_trade_ml += 1
                except Exception as e:
                    logger.error(
                        "Trade ML staging error cfg=%s era=%s SL=%.4f TP=%.4f: %s",
                        regime_id,
                        era_label,
                        float(sl_val),
                        float(tp_val),
                        e,
                    )

                if breached:
                    dd_fail += 1
                    final_balance = float(equity_arr[-1]) if equity_arr.size else 100.0
                    win_pos = int(np.count_nonzero(closed_rets > 0.0)) if closed_rets.size else 0

                    master_row_raw = _make_master_row(
                        regime_id=regime_id,
                        signal_layer=signal_layer,
                        era_int=era["era_int"],
                        sl_val=sl_val,
                        tp_val=tp_val,
                        total_pos=int(entry_idx.size),
                        win_pos=win_pos,
                        balance=final_balance,
                        max_dd=current_max_dd,
                        max_consecutive_losses=compute_max_consecutive_losses(closed_rets),
                        regime_cfg=regime_cfg,
                        run_cfg=run_cfg,
                        signal_scope_id=signal_scope_id,
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
                    closed_side_arr=trade_sides,
                    sl_val=float(sl_val),
                    tp_val=float(tp_val),
                    regime_id=regime_id,
                    signal_layer=signal_layer,
                    era_int=era["era_int"],
                    stager=equity_stager,
                    max_dd=current_max_dd,
                    signal_scope=signal_scope_id,
                )
                staged_equity += 1

                master_row_raw = _make_master_row(
                    regime_id=regime_id,
                    signal_layer=signal_layer,
                    era_int=era["era_int"],
                    sl_val=sl_val,
                    tp_val=tp_val,
                    total_pos=int(entry_idx.size),
                    win_pos=win_pos,
                    balance=final_balance,
                    max_dd=max_dd,
                    max_consecutive_losses=compute_max_consecutive_losses(closed_rets),
                    regime_cfg=regime_cfg,
                    run_cfg=run_cfg,
                    signal_scope_id=signal_scope_id,
                )
                buffer_master_row(results_dir, batch_id, master_row_raw)

                era_master_rows_written[era_label] += 1
                master_rows_written += 1

                del entry_idx, entry_price_arr, exit_idx, exit_price_arr, rets, exit_reason_arr
                del closed_rets, closed_entry_idxs, closed_exit_idxs, pnl_pct_arr, exit_times_ns, equity_arr

            del res, entry_idx_all, entry_price_all, exit_idx_all, exit_price_all, rets_all, exit_reason_all

    except Exception:
        logger.error("Error processing regime %s in batch %s", regime_id, batch_id)
        raise
    finally:
        try:
            #equity_stager.flush_all()
            #trade_ml_stager.flush_all()
            _flush_master_rows_buffer(results_dir, batch_id)
        except Exception as e:
            logger.debug("Flush local stage failed: %s", e)

        try:
            flush_all_buffers()
        except Exception:
            logger.debug("flush_all_buffers() failed")

        try:
            del global_signals
            del all_sig_idxs_all, all_sig_times_all, all_sig_sides_all
        except Exception:
            pass

        gc.collect()

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
        "staged_trade_ml": staged_trade_ml,
        "master_rows_written": master_rows_written,
        "empty_rows_written": empty_rows_written,
    }

def _normalize_regime_cfg_for_grid(regime_cfg: dict, run_cfg: dict) -> dict:
    """
    Grid-search regime normalization.

    The working source of truth is nested signal_structure.
    signal_json is only the compact serialized signature written out for analysis.
    """
    out = deepcopy(regime_cfg or {})

    signal_scope_id = str(out.get("signal_scope_id") or "").strip().lower()
    signal_structure = out.get("signal_structure")

    if signal_scope_id.startswith("baseline:"):
        out["signal_layer"] = 0
        if not isinstance(signal_structure, dict) or not signal_structure:
            signal_structure = deepcopy(run_cfg.get("signal_structure") or {})
            out["signal_structure"] = signal_structure
    else:
        if not isinstance(signal_structure, dict) or not signal_structure:
            raise ValueError(
                "signal_structure is required in grid search regimes. "
                "Legacy flat signal keys are no longer supported."
            )

    out.pop("signal_scope", None)

    if not signal_scope_id:
        if int(out.get("signal_layer", 0) or 0) == 0:
            signal_scope_id = "baseline:all_buy"
        else:
            signal_scope_id = ""
    out["signal_scope_id"] = signal_scope_id

    out["pair"] = str(out.get("pair", run_cfg.get("pair", "")) or "")
    out["BASE_MINUTES"] = int(out.get("BASE_MINUTES", run_cfg.get("BASE_MINUTES", 5)) or 5)

    out["sl_tp_in_pct"] = bool(out.get("sl_tp_in_pct", run_cfg.get("sl_tp_in_pct", True)))
    out["min_rr"] = float(out.get("min_rr", run_cfg.get("min_rr", 3.0)) or 3.0)
    out["sl_tp_interval_months"] = int(out.get("sl_tp_interval_months", run_cfg.get("sl_tp_interval_months", 3)) or 3)

    out["SL"] = float(out.get("SL", run_cfg.get("SL", 0.2)) or 0.2)
    out["TP"] = float(out.get("TP", run_cfg.get("TP", 6.0)) or 6.0)

    out["trade_overlap"] = bool(out.get("trade_overlap", True))
    out["trade_flip_on_entry"] = bool(out.get("trade_flip_on_entry", False))
    out["use_trailing_sl"] = bool(out.get("use_trailing_sl", False))
    out["trailing_sl_pct"] = float(out.get("trailing_sl_pct", 0.0) or 0.0)
    out["trailing_sl_interval"] = int(out.get("trailing_sl_interval", 0) or 0)
    out["trailing_sl_stop_at_pos"] = bool(out.get("trailing_sl_stop_at_pos", True))

    out["use_limit_entry"] = bool(out.get("use_limit_entry", True))
    out["limit_order_expiry_bars"] = int(out.get("limit_order_expiry_bars", 0) or 0)
    out["trade_window_interval"] = int(out.get("trade_window_interval", 0) or 0)

    out["exit_window_h"] = int(out.get("exit_window_h", 24) or 24)
    return out


def compute_config_and_save(
    batch_path: str,
    session_dir: str,
    total_batches: int | None = None,
    compute_backtest: bool = True,
) -> Dict:
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

    if isinstance(run_cfg.get("signal_structure"), dict) and not run_cfg.get("signal_json"):
        run_cfg["signal_json"] = build_signal_json(run_cfg)

    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(batch_path, "r", encoding="utf8") as f:
        batch_data = json.load(f)

    regimes = batch_data["regimes"]
    batch_id = int(batch_data.get("batch_id", 0))

    batch_pos = batch_id + 1
    batch_total = int(total_batches or 0)
    batch_tag = f"({batch_pos}/{batch_total})" if batch_total > 0 else f"(batch {batch_pos})"

    logger.info(
        "🧵 %s started: %s | regimes=%d",
        batch_tag,
        batch_path.name,
        len(regimes),
    )

    prepared = prepare_worker_data(session_dir, run_cfg)
    data_ctx = prepared

    def _mem_snapshot(tag: str) -> None:
        try:
            rss_mb = _get_process_rss_mb()
        except Exception:
            rss_mb = -1.0

        try:
            sys_pct = psutil.virtual_memory().percent
        except Exception:
            sys_pct = -1.0

        logger.info("🧠 %s | rss=%.1fMB sys=%.1f%%", tag, rss_mb, sys_pct)

    if "lookback_map" not in data_ctx or not isinstance(data_ctx["lookback_map"], dict):
        raise RuntimeError("prepare_worker_data did not return a valid 'lookback_map'.")

    max_dd_threshold = float(run_cfg.get("max_dd_threshold", 0.20))
    flush_rows = int(run_cfg.get("PARQUET_FLUSH_ROWS", 100_000))
    partition_by = str(run_cfg.get("EQUITY_PARTITION_BY", "era_int"))

    equity_base = session_dir / "equity_partitioned"
    equity_tmp = equity_base / "_tmp"
    equity_stager = EquityStager(
        equity_part_base=equity_base,
        batch_id=batch_id,
        flush_rows=flush_rows,
        partition_by=partition_by,
        tmp_dir=equity_tmp,
    )

    trade_ml_stager = TradeMLStager(
        session_dir=session_dir,
        batch_id=batch_id,
        flush_rows=flush_rows,
        partition_by="era_int",
    )

    processed_regimes = 0
    skipped_regimes = 0

    container_limit_mb = _get_memory_guard_mb()
    rss_mb = _get_process_rss_mb()

    if container_limit_mb > 0:
        soft_cap_mb = container_limit_mb * 0.80
        if rss_mb >= soft_cap_mb:
            raise AirflowFailException(
                f"Process memory exceeded safe threshold: {rss_mb:.1f} MB / {container_limit_mb:.1f} MB"
            )

    try:
        total_in_batch = len(regimes)

        for idx, regime_cfg in enumerate(regimes):
            gc_every_n = int(run_cfg.get("GC_EVERY_N_REGIMES", 0) or 0)
            gc_rss_limit_mb = int(run_cfg.get("GC_RSS_LIMIT_MB", 0) or 0)

            _mem_snapshot(f"before regime {idx + 1}/{total_in_batch}")

            if _HAS_PSUTIL and _PROC:
                mem_info = psutil.virtual_memory()
                if mem_info.percent > 92.0:
                    raise AirflowFailException(f"System memory threshold exceeded ({mem_info.percent}%).")

            if idx % 10 == 0 or idx + 1 == total_in_batch:
                logger.info(
                    "📍 %s regime %d/%d",
                    batch_tag,
                    idx + 1,
                    total_in_batch,
                )
                heartbeat_log(
                    f"batch{batch_id}_progress",
                    {"idx": idx, "regime_id": str(regime_cfg.get("regime_id"))},
                )

            regime_cfg = _normalize_regime_cfg_for_grid(regime_cfg, run_cfg)
            signal_layer = int(regime_cfg.get("signal_layer", 0) or 0)
            signal_scope_id = str(regime_cfg.get("signal_scope_id") or "").strip().lower()
            if not signal_scope_id and signal_layer == 0:
                signal_scope_id = "baseline:all_buy"

            regime_cfg["signal_layer"] = signal_layer
            regime_cfg["signal_scope_id"] = signal_scope_id

            try:
                regime_cfg["regime_id"] = int(regime_cfg.get("regime_id"))
            except Exception:
                logger.debug("regime_id not int-convertible: %s", regime_cfg.get("regime_id"))

            regime_id = regime_cfg.get("regime_id")
            summary_path = _summary_path_for_regime(results_dir, regime_cfg)
            if summary_path.exists():
                skipped_regimes += 1
                continue

            with maybe_profile(f"regime_{regime_id}_process"):
                result = process_era_combos(
                    regime_id=regime_id,
                    regime_cfg=regime_cfg,
                    run_cfg=run_cfg,
                    data_ctx=data_ctx,
                    session_dir=session_dir,
                    results_dir=results_dir,
                    batch_id=batch_id,
                    equity_stager=equity_stager,
                    trade_ml_stager=trade_ml_stager,
                    signal_layer=signal_layer,
                    signal_scope_id=signal_scope_id,
                    max_dd_threshold=max_dd_threshold,
                    current_idx=idx,
                    total_in_batch=total_in_batch,
                )

            processed_regimes += 1
            del result

            try:
                _flush_master_rows_buffer(results_dir, batch_id)
            except Exception as e:
                logger.debug("Flush master rows after regime failed: %s", e)

            if gc_every_n > 0 and (idx + 1) % gc_every_n == 0:
                try:
                    rss_mb = _get_process_rss_mb()
                    if gc_rss_limit_mb > 0 and rss_mb >= gc_rss_limit_mb:
                        gc.collect()
                except Exception:
                    gc.collect()

            _mem_snapshot(f"after regime {idx + 1}/{total_in_batch}")

            final_output = {
                "regime_id": regime_id,
                "regime_params": regime_cfg,
                "signal_json": regime_cfg.get("signal_json"),
                "combos_tested": 0,
                "results": [],
            }
            with open(summary_path, "w", encoding="utf8") as fh:
                json.dump(final_output, fh, indent=2)

    except Exception as e:
        logger.error("Error processing batch %s: %s", batch_path.name, e)
        raise
    finally:
        try:
            equity_stager.flush_all()
            trade_ml_stager.flush_all()
            _flush_master_rows_buffer(results_dir, batch_id)
        except Exception as e:
            logger.debug("Flush local stage failed: %s", e)

        try:
            flush_all_buffers()
        except Exception:
            logger.debug("flush_all_buffers() failed")

    logger.info(
        "✅ %s finished: regimes_done=%d skipped_regimes=%d",
        batch_tag,
        processed_regimes,
        skipped_regimes,
    )
    return {
        "batch": batch_path.name,
        "processed": processed_regimes,
        "skipped": skipped_regimes,
    }