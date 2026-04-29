# grid_row_builders.py

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import polars as pl

import logging

from common.feature_helpers import (
    build_signal_json,
    compact_json,
    signal_scope_text,
)
from common.schema import enforce_schema, get_schema

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

def _scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value

def _make_empty_master_row(
    regime_id: int,
    signal_layer: int,
    era_int: int,
    sl_val: float,
    tp_val: float,
    regime_cfg: dict,
    run_cfg: Optional[dict] = None,
    signal_scope_id: str = "",
) -> dict:
    signal_json = build_signal_json(regime_cfg, run_cfg=run_cfg)
    scope_id = str(signal_scope_id or regime_cfg.get("signal_scope_id") or "").strip().lower()

    return {
        "regime_id": int(regime_id),
        "signal_layer": int(signal_layer),
        "signal_scope_id": scope_id,
        "era_int": int(era_int),
        "exit_window_h": int(_scalar(regime_cfg.get("exit_window_h", 0)) or 0),
        "SL": float(sl_val),
        "TP": float(tp_val),
        "use_trailing_sl": bool(_as_bool(_scalar(regime_cfg.get("use_trailing_sl", False)), False)),
        "trailing_sl_pct": float(_scalar(regime_cfg.get("trailing_sl_pct", 0.0)) or 0.0),
        "trailing_sl_interval": int(_scalar(regime_cfg.get("trailing_sl_interval", 0)) or 0),
        "trailing_sl_stop_at_pos": bool(_as_bool(_scalar(regime_cfg.get("trailing_sl_stop_at_pos", True)), True)),
        "use_limit_entry": bool(_as_bool(_scalar(regime_cfg.get("use_limit_entry", True)), True)),
        "limit_order_expiry_bars": int(_scalar(regime_cfg.get("limit_order_expiry_bars", 0)) or 0),
        "trade_overlap": bool(_as_bool(_scalar(regime_cfg.get("trade_overlap", True)), True)),
        "trade_flip_on_entry": bool(_as_bool(_scalar(regime_cfg.get("trade_flip_on_entry", False)), False)),
        "trade_window_interval": int(_scalar(regime_cfg.get("trade_window_interval", 0)) or 0),
        "total_pos": 0,
        "win_pos": 0,
        "balance": 100.0,
        "max_drawdown": 0.0,
        "max_consecutive_losses": 0,
        "signal_json": compact_json(signal_json),
    }


def _make_master_row(
    regime_id: int,
    signal_layer: int,
    era_int: int,
    sl_val: float,
    tp_val: float,
    total_pos: int,
    win_pos: int,
    balance: float,
    max_dd: float,
    max_consecutive_losses: int,
    regime_cfg: dict,
    run_cfg: Optional[dict] = None,
    signal_scope_id: str = "",
) -> dict:
    """
    Populated master row.
    """
    row = _make_empty_master_row(
        regime_id=regime_id,
        signal_layer=signal_layer,
        era_int=era_int,
        sl_val=sl_val,
        tp_val=tp_val,
        regime_cfg=regime_cfg,
        run_cfg=run_cfg,
        signal_scope_id=signal_scope_id,
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

logger = logging.getLogger(__name__)


def build_trade_ml_rows_from_backtest(
    df_main: pl.DataFrame,
    main_close_arr: np.ndarray,
    main_high_arr: np.ndarray,
    main_low_arr: np.ndarray,
    main_time_ns_arr: np.ndarray,
    sig_idxs: np.ndarray,
    side_arr: Optional[np.ndarray],
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
    signal_layer: int = 0,
    signal_scope_id: str = "",
) -> pl.DataFrame:
    sig_idxs = np.asarray(sig_idxs, dtype=np.int64)
    if side_arr is None:
        side_arr = np.full(sig_idxs.shape[0], 1, dtype=np.int8)
    else:
        side_arr = np.asarray(side_arr, dtype=np.int8)

    n = min(sig_idxs.shape[0], side_arr.shape[0])
    if n == 0:
        return pl.DataFrame(schema=get_schema("trade_ml"))

    sig_idxs = sig_idxs[:n]
    side_arr = side_arr[:n]

    signal_scope_id = str(signal_scope_id or "")
    signal_layer = int(signal_layer)

    signal_time_ns = np.asarray(main_time_ns_arr[sig_idxs], dtype=np.int64)
    signal_price = np.asarray(main_close_arr[sig_idxs], dtype=np.float64)

    order_mode = np.full(n, 1 if use_limit_entry else 0, dtype=np.int8)

    if use_limit_entry:
        order_price = signal_price.copy()
        buy_mask = side_arr == 1
        sell_mask = side_arr == -1

        if np.any(buy_mask):
            order_price[buy_mask] = 0.5 * (
                signal_price[buy_mask]
                + np.asarray(main_low_arr[sig_idxs[buy_mask]], dtype=np.float64)
            )
        if np.any(sell_mask):
            order_price[sell_mask] = 0.5 * (
                signal_price[sell_mask]
                + np.asarray(main_high_arr[sig_idxs[sell_mask]], dtype=np.float64)
            )
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

    filled_signal_idx = np.asarray(
        backtest_res.get("signal_idx", backtest_res.get("signal_idxs", backtest_res.get("signal_index", []))),
        dtype=np.int64,
    )
    filled_side = np.asarray(backtest_res.get("side", []), dtype=np.int8)
    filled_entry_idx = np.asarray(backtest_res.get("entry_idx", []), dtype=np.int64)
    filled_entry_price = np.asarray(backtest_res.get("entry_price", []), dtype=np.float64)
    filled_exit_idx = np.asarray(backtest_res.get("exit_idx", []), dtype=np.int64)
    filled_exit_price = np.asarray(backtest_res.get("exit_price", []), dtype=np.float64)
    filled_rets = np.asarray(backtest_res.get("rets", backtest_res.get("pnl_pct", [])), dtype=np.float64)
    filled_exit_reason = np.asarray(backtest_res.get("exit_reason", []), dtype=np.int8)
    filled_sl_hit = np.asarray(backtest_res.get("SL_hit", []), dtype=np.float64)
    filled_tp_hit = np.asarray(backtest_res.get("TP_hit", []), dtype=np.float64)

    if filled_signal_idx.size == 0 or filled_side.size == 0:
        logger.warning(
            "trade_ml builder: empty filled stream | regime=%s era=%s layer=%s scope=%s sigs=%d keys=%s",
            regime_id,
            era_int,
            signal_layer,
            signal_scope_id,
            int(n),
            list(backtest_res.keys()),
        )
        return pl.DataFrame(schema=get_schema("trade_ml"))

    m = min(
        filled_signal_idx.size,
        filled_side.size,
        filled_entry_idx.size,
        filled_entry_price.size,
        filled_exit_idx.size,
        filled_exit_price.size,
        filled_rets.size,
        filled_exit_reason.size,
    )
    if m == 0:
        return pl.DataFrame(schema=get_schema("trade_ml"))

    # Key fix: match on (signal_idx, side), not signal_idx alone.
    filled_map = {
        (int(filled_signal_idx[i]), int(filled_side[i])): i
        for i in range(m)
    }

    keep_rows = []
    unmatched = 0

    for i in range(n):
        key = (int(sig_idxs[i]), int(side_arr[i]))
        j = filled_map.get(key, -1)
        if j < 0:
            unmatched += 1
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
            fill_delay_bars[i] = int(entry_idx[i] - sig_idxs[i])

        keep_rows.append(i)

    if not keep_rows:
        logger.warning(
            "trade_ml builder: all rows dropped after matching | regime=%s era=%s layer=%s scope=%s sigs=%d filled=%d unmatched=%d",
            regime_id,
            era_int,
            signal_layer,
            signal_scope_id,
            int(n),
            int(m),
            int(unmatched),
        )
        return pl.DataFrame(schema=get_schema("trade_ml"))

    keep_rows = np.asarray(keep_rows, dtype=np.int64)

    df = pl.DataFrame(
        {
            "regime_id": np.full(keep_rows.size, int(regime_id), dtype=np.int32),
            "signal_layer": np.full(keep_rows.size, int(signal_layer), dtype=np.int8),
            "signal_scope_id": np.full(keep_rows.size, signal_scope_id, dtype=object),
            "era_int": np.full(keep_rows.size, int(era_int), dtype=np.int64),
            "side": side_arr[keep_rows].astype(np.int8),
            "SL": np.full(keep_rows.size, float(sl_val), dtype=np.float32),
            "TP": np.full(keep_rows.size, float(tp_val), dtype=np.float32),
            "SL_hit": sl_hit[keep_rows].astype(np.float32),
            "TP_hit": tp_hit[keep_rows].astype(np.float32),
            "signal_idx": sig_idxs[keep_rows].astype(np.int64),
            "signal_time_ns": signal_time_ns[keep_rows].astype(np.int64),
            "signal_price": signal_price[keep_rows].astype(np.float32),
            "order_idx": sig_idxs[keep_rows].astype(np.int64),
            "order_time_ns": signal_time_ns[keep_rows].astype(np.int64),
            "order_price": order_price[keep_rows].astype(np.float32),
            "order_mode": order_mode[keep_rows].astype(np.int8),
            "fill_status": fill_status[keep_rows].astype(np.int8),
            "entry_idx": entry_idx[keep_rows].astype(np.int64),
            "entry_time_ns": entry_time_ns[keep_rows].astype(np.int64),
            "entry_price": entry_price[keep_rows].astype(np.float32),
            "exit_idx": exit_idx[keep_rows].astype(np.int64),
            "exit_time_ns": exit_time_ns[keep_rows].astype(np.int64),
            "exit_price": exit_price[keep_rows].astype(np.float32),
            "exit_reason": exit_reason[keep_rows].astype(np.int8),
            "fill_delay_bars": fill_delay_bars[keep_rows].astype(np.int32),
            "pnl_pct": pnl_pct[keep_rows].astype(np.float32),
        }
    )

    df = df.filter(pl.col("fill_status") == 1)
    return enforce_schema(df, "trade_ml", strict=False)