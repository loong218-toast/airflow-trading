# research/grid_row_builders.py

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import polars as pl

from common.feature_helpers import normalize_timeframe
from common.schema import enforce_schema

from common.casting import (
    _as_bool,
    _as_int,
    _as_float,
    _as_list,
    _as_int_list,
    _as_float_list,
    _as_threshold_pairs,
    _ordered_unique_ints,
    _ordered_unique_floats,
    _ordered_unique_strings,
    _as_str,
)

def _single_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value

def _json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _signal_structure_source(regime_cfg: dict) -> dict:
    signal_block = regime_cfg.get("signal_structure")
    if isinstance(signal_block, dict) and signal_block:
        return deepcopy(signal_block)
    return {}

def _scalarize(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[0]
    return value

def _build_signal_json(regime_cfg: dict, run_cfg: Optional[dict] = None) -> dict:
    """
    Compact nested payload used by master rows, analysis, and surrogate-free
    debugging.

    This is intentionally the same signal contract consumed by feature_prep.py
    and live signal filtering.
    """
    signal_block = _signal_structure_source(regime_cfg)
    out: dict = {
        "version": 3,
        "signals": {},
    }

    if not isinstance(signal_block, dict):
        return out

    for signal_name, signal_cfg in signal_block.items():
        if not isinstance(signal_cfg, dict):
            continue

        enabled = signal_cfg.get("enabled", True)
        enabled = _single_value(enabled)
        if isinstance(enabled, (list, tuple)):
            enabled = enabled[0] if enabled else True
        enabled = _as_bool(enabled, True)
        if not enabled:
            continue

        tf_map = signal_cfg.get("by_timeframe", {})
        if not isinstance(tf_map, dict):
            continue

        signal_out: dict = {}
        for tf, tf_cfg in tf_map.items():
            if not isinstance(tf_cfg, dict):
                continue
            cfg = deepcopy(tf_cfg)
            cfg["timeframe"] = normalize_timeframe(tf)
            signal_out[cfg["timeframe"]] = cfg

        if signal_out:
            out["signals"][signal_name] = signal_out

    return out


def _make_empty_master_row(
    regime_id: int,
    era_int: int,
    side_flag: int,
    sl_val: float,
    tp_val: float,
    regime_cfg: dict,
    run_cfg: Optional[dict] = None,
) -> dict:
    """
    Canonical empty master row.

    This must match common/schema.py -> MASTER_SCHEMA exactly, so downstream
    code never has to branch on "empty trade set" special cases.
    """
    signal_json = _build_signal_json(regime_cfg, run_cfg=run_cfg)

    return {
        "regime_id": int(regime_id),
        "era_int": int(era_int),
        "side": int(side_flag),
        "exit_window_h": int(_scalarize(regime_cfg.get("exit_window_h", 0)) or 0),
        "SL": float(sl_val),
        "TP": float(tp_val),
        "use_trailing_sl": bool(_as_bool(_scalarize(regime_cfg.get("use_trailing_sl", False)), False)),
        "trailing_sl_pct": float(_scalarize(regime_cfg.get("trailing_sl_pct", 0.0)) or 0.0),
        "trailing_sl_interval": int(_scalarize(regime_cfg.get("trailing_sl_interval", 0)) or 0),
        "trailing_sl_stop_at_pos": bool(_as_bool(_scalarize(regime_cfg.get("trailing_sl_stop_at_pos", True)), True)),
        "use_limit_entry": bool(_as_bool(_scalarize(regime_cfg.get("use_limit_entry", True)), True)),
        "limit_order_expiry_bars": int(_scalarize(regime_cfg.get("limit_order_expiry_bars", 0)) or 0),
        "trade_window_interval": int(_scalarize(regime_cfg.get("trade_window_interval", 0)) or 0),
        "total_pos": 0,
        "win_pos": 0,
        "balance": 100.0,
        "max_drawdown": 0.0,
        "max_consecutive_losses": 0,
        "signal_json": _json_compact(signal_json),
    }


def _make_master_row(
    regime_id: int,
    era_int: int,
    side_flag: int,
    sl_val: float,
    tp_val: float,
    total_pos: int,
    win_pos: int,
    balance: float,
    max_dd: float,
    max_consecutive_losses: int,
    regime_cfg: dict,
    run_cfg: Optional[dict] = None,
) -> dict:
    """
    Populated master row.

    The downstream master table uses:
    - regime selection
    - era-level analysis
    - time-bucket analysis
    - survival/robustness checks
    - future signal_json reconstruction
    """
    row = _make_empty_master_row(
        regime_id=regime_id,
        era_int=era_int,
        side_flag=side_flag,
        sl_val=sl_val,
        tp_val=tp_val,
        regime_cfg=regime_cfg,
        run_cfg=run_cfg,
    )

    row.update(
        {
            "total_pos": int(total_pos),
            "win_pos": int(win_pos),
            "balance": float(balance),
            "max_drawdown": float(max_dd),
            "max_consecutive_losses": int(max_consecutive_losses),
        }
    )
    return row


def _empty_range_array(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=np.float32)


def build_trade_ml_rows_from_backtest(
    df_main: pl.DataFrame,
    main_close_arr: np.ndarray,
    main_high_arr: np.ndarray,
    main_low_arr: np.ndarray,
    main_time_ns_arr: np.ndarray,
    sig_idxs: np.ndarray,
    side_flag: int,
    sl_val: float,
    tp_val: float,
    use_limit_entry: bool,
    limit_order_expiry_bars: int,
    trade_window_interval: int,
    regime_id: int,
    era_int: int,
    backtest_res: Dict,
    regime_cfg: dict,
    run_cfg: dict,
) -> pl.DataFrame:
    """
    Build trade-level training rows from a completed backtest.

    Legacy amp-index logic has been removed on purpose.
    The range columns remain in the schema, but they are left as NaN so the
    grid branch stays compatible without carrying hidden feature dependencies.
    """
    sig_idxs = np.asarray(sig_idxs, dtype=np.int64)
    n = sig_idxs.shape[0]

    if n == 0:
        return pl.DataFrame(
            {
                "regime_id": pl.Series([], dtype=pl.Int32),
                "era_int": pl.Series([], dtype=pl.Int64),
                "side": pl.Series([], dtype=pl.Int8),
                "SL": pl.Series([], dtype=pl.Float32),
                "TP": pl.Series([], dtype=pl.Float32),
                "SL_hit": pl.Series([], dtype=pl.Float32),
                "TP_hit": pl.Series([], dtype=pl.Float32),
                "use_limit_entry": pl.Series([], dtype=pl.Boolean),
                "limit_order_expiry_bars": pl.Series([], dtype=pl.Int32),
                "trade_window_interval": pl.Series([], dtype=pl.Int32),
                "signal_idx": pl.Series([], dtype=pl.Int64),
                "signal_time_ns": pl.Series([], dtype=pl.Int64),
                "signal_price": pl.Series([], dtype=pl.Float32),
                "order_idx": pl.Series([], dtype=pl.Int64),
                "order_time_ns": pl.Series([], dtype=pl.Int64),
                "order_price": pl.Series([], dtype=pl.Float32),
                "order_mode": pl.Series([], dtype=pl.Int8),
                "fill_status": pl.Series([], dtype=pl.Int8),
                "entry_idx": pl.Series([], dtype=pl.Int64),
                "entry_time_ns": pl.Series([], dtype=pl.Int64),
                "entry_price": pl.Series([], dtype=pl.Float32),
                "exit_idx": pl.Series([], dtype=pl.Int64),
                "exit_time_ns": pl.Series([], dtype=pl.Int64),
                "exit_price": pl.Series([], dtype=pl.Float32),
                "exit_reason": pl.Series([], dtype=pl.Int8),
                "fill_delay_bars": pl.Series([], dtype=pl.Int32),
                "pnl_pct": pl.Series([], dtype=pl.Float32),
            }
        )

    signal_time_ns = np.asarray(main_time_ns_arr[sig_idxs], dtype=np.int64)
    signal_price = np.asarray(main_close_arr[sig_idxs], dtype=np.float64)

    order_mode = np.full(n, 1 if use_limit_entry else 0, dtype=np.int8)

    if use_limit_entry:
        if int(side_flag) == 1:
            order_price = 0.5 * (signal_price + np.asarray(main_low_arr[sig_idxs], dtype=np.float64))
        else:
            order_price = 0.5 * (signal_price + np.asarray(main_high_arr[sig_idxs], dtype=np.float64))
    else:
        order_price = signal_price.copy()

    fill_status = np.zeros(n, dtype=np.int8)
    entry_idx = np.full(n, -1, dtype=np.int64)
    entry_time_ns = np.full(n, -1, dtype=np.int64)
    entry_price = np.full(n, np.nan, dtype=np.float64)

    exit_idx = np.full(n, -1, dtype=np.int64)
    exit_time_ns = np.full(n, -1, dtype=np.int64)
    exit_price = np.full(n, np.nan, dtype=np.float64)
    exit_reason = np.full(n, -2, dtype=np.int8)

    pnl_pct = np.full(n, np.nan, dtype=np.float64)
    fill_delay_bars = np.full(n, -1, dtype=np.int32)
    sl_hit = np.full(n, np.nan, dtype=np.float64)
    tp_hit = np.full(n, np.nan, dtype=np.float64)

    filled_signal_idx = np.asarray(backtest_res.get("signal_idx", []), dtype=np.int64)
    filled_entry_idx = np.asarray(backtest_res.get("entry_idx", []), dtype=np.int64)
    filled_entry_price = np.asarray(backtest_res.get("entry_price", []), dtype=np.float64)
    filled_exit_idx = np.asarray(backtest_res.get("exit_idx", []), dtype=np.int64)
    filled_exit_price = np.asarray(backtest_res.get("exit_price", []), dtype=np.float64)
    filled_rets = np.asarray(backtest_res.get("rets", []), dtype=np.float64)
    filled_exit_reason = np.asarray(backtest_res.get("exit_reason", []), dtype=np.int8)
    filled_sl_hit = np.asarray(backtest_res.get("SL_hit", []), dtype=np.float64)
    filled_tp_hit = np.asarray(backtest_res.get("TP_hit", []), dtype=np.float64)

    filled_map = {int(s): i for i, s in enumerate(filled_signal_idx)}

    for i in range(n):
        sidx = int(sig_idxs[i])
        j = filled_map.get(sidx, -1)
        if j < 0:
            continue

        fill_status[i] = 1
        entry_idx[i] = int(filled_entry_idx[j])
        entry_price[i] = float(filled_entry_price[j])
        exit_idx[i] = int(filled_exit_idx[j])
        exit_price[i] = float(filled_exit_price[j])
        exit_reason[i] = int(filled_exit_reason[j])
        pnl_pct[i] = float(filled_rets[j])

        if j < filled_sl_hit.size:
            sl_hit[i] = float(filled_sl_hit[j])
        if j < filled_tp_hit.size:
            tp_hit[i] = float(filled_tp_hit[j])

        if 0 <= entry_idx[i] < main_time_ns_arr.shape[0]:
            entry_time_ns[i] = int(main_time_ns_arr[entry_idx[i]])

        if 0 <= exit_idx[i] < main_time_ns_arr.shape[0]:
            exit_time_ns[i] = int(main_time_ns_arr[exit_idx[i]])

        if entry_idx[i] >= 0:
            fill_delay_bars[i] = int(entry_idx[i] - sidx)

    df = pl.DataFrame(
        {
            "regime_id": np.full(n, int(regime_id), dtype=np.int32),
            "era_int": np.full(n, int(era_int), dtype=np.int64),
            "side": np.full(n, int(side_flag), dtype=np.int8),
            "SL": np.full(n, float(sl_val), dtype=np.float32),
            "TP": np.full(n, float(tp_val), dtype=np.float32),
            "SL_hit": sl_hit.astype(np.float32),
            "TP_hit": tp_hit.astype(np.float32),
            "use_limit_entry": np.full(n, bool(use_limit_entry), dtype=bool),
            "limit_order_expiry_bars": np.full(n, int(limit_order_expiry_bars), dtype=np.int32),
            "trade_window_interval": np.full(n, int(trade_window_interval), dtype=np.int32),
            "signal_idx": sig_idxs.astype(np.int64),
            "signal_time_ns": signal_time_ns.astype(np.int64),
            "signal_price": signal_price.astype(np.float32),
            "order_idx": sig_idxs.astype(np.int64),
            "order_time_ns": signal_time_ns.astype(np.int64),
            "order_price": order_price.astype(np.float32),
            "order_mode": order_mode.astype(np.int8),
            "fill_status": fill_status.astype(np.int8),
            "entry_idx": entry_idx.astype(np.int64),
            "entry_time_ns": entry_time_ns.astype(np.int64),
            "entry_price": entry_price.astype(np.float32),
            "exit_idx": exit_idx.astype(np.int64),
            "exit_time_ns": exit_time_ns.astype(np.int64),
            "exit_price": exit_price.astype(np.float32),
            "exit_reason": exit_reason.astype(np.int8),
            "fill_delay_bars": fill_delay_bars.astype(np.int32),
            "pnl_pct": pnl_pct.astype(np.float32),
        }
    )

    return enforce_schema(df, "trade_ml", strict=False)