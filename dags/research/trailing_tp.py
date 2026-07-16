# trailing_tp.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrailingExitParams:
    use_trailing_exit: bool
    trailing_exit_activation_r: float
    trailing_exit_pct: float
    trailing_exit_interval: int


def normalize_float_list(values: Any, default: Sequence[float]) -> List[float]:
    if values is None:
        return [float(x) for x in default]
    if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        out = [float(x) for x in values]
        return out or [float(x) for x in default]
    return [float(values)]


def normalize_int_list(values: Any, default: Sequence[int]) -> List[int]:
    if values is None:
        return [int(x) for x in default]
    if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        out = [int(x) for x in values]
        return out or [int(x) for x in default]
    return [int(values)]


def iter_trailing_params(
    use_trailing_exit: bool,
    trailing_exit_activation_r: float,
    trailing_exit_pct_values: Any,
    trailing_exit_interval_values: Any,
    default_pct: Sequence[float] = (0.2,),
    default_interval: Sequence[int] = (3,),
) -> List[TrailingExitParams]:
    pct_values = normalize_float_list(trailing_exit_pct_values, default_pct)
    interval_values = normalize_int_list(trailing_exit_interval_values, default_interval)

    if not use_trailing_exit:
        return [
            TrailingExitParams(
                False,
                float(trailing_exit_activation_r),
                float(pct_values[0]),
                int(interval_values[0]),
            )
        ]

    out: List[TrailingExitParams] = []
    for pct in pct_values:
        for interval in interval_values:
            out.append(
                TrailingExitParams(
                    use_trailing_exit=True,
                    trailing_exit_activation_r=float(trailing_exit_activation_r),
                    trailing_exit_pct=float(pct),
                    trailing_exit_interval=max(1, int(interval)),
                )
            )
    return out


def _move_pct(side: int, entry_price: float, exit_price: float) -> float:
    if not np.isfinite(entry_price) or entry_price <= 0.0 or not np.isfinite(exit_price):
        return np.nan
    if int(side) == 1:
        return ((exit_price - entry_price) / entry_price) * 100.0
    return ((entry_price - exit_price) / entry_price) * 100.0


def simulate_adverse_trailing_tp(
    *,
    side: int,
    entry_price: float,
    sl_price: float,
    target_price: float,
    future_high: np.ndarray,
    future_low: np.ndarray,
    future_close: np.ndarray,
    future_time_ns: np.ndarray,
    activation_r: float,
    trailing_tp_pct: float,
    trailing_tp_interval: int,
) -> Dict[str, Any]:
    """
    Dynamic TP on adverse excursion.

    BUY:
    - activate when price moves down toward SL by activation_r of the SL distance
    - TP line starts at target_price
    - every trailing_tp_interval candles after activation, TP line moves downward
      toward SL by trailing_tp_pct of the SL distance

    SELL is the mirror version.
    """
    n = int(min(len(future_high), len(future_low), len(future_close), len(future_time_ns)))
    nan_out = {
        "trailing_triggered": False,
        "trailing_tp_activation_idx": None,
        "trailing_tp_idx": None,
        "trailing_tp_price": np.nan,
        "trailing_tp_time_ns": None,
        "trailing_tp_line": np.nan,
        "trailing_tp_updates": 0,
        "trailing_tp_moved_r": np.nan,
        "move_pct": np.nan,
        # compatibility keys
        "trailing_activation_idx": None,
        "trailing_exit_idx": None,
        "trailing_exit_price": np.nan,
        "trailing_exit_time_ns": None,
        "trailing_stop_price": np.nan,
        "trailing_stop_updates": 0,
    }

    if n <= 0:
        return nan_out
    if not np.isfinite(entry_price) or entry_price <= 0.0:
        return nan_out
    if not np.isfinite(sl_price) or not np.isfinite(target_price):
        return nan_out

    future_high = np.asarray(future_high[:n], dtype=np.float64)
    future_low = np.asarray(future_low[:n], dtype=np.float64)
    future_close = np.asarray(future_close[:n], dtype=np.float64)
    future_time_ns = np.asarray(future_time_ns[:n], dtype=np.int64)

    sl_distance = abs(float(entry_price) - float(sl_price))
    if sl_distance <= 0.0:
        return nan_out

    activation_r = float(activation_r)
    trailing_tp_pct = float(trailing_tp_pct)
    trailing_tp_interval = max(1, int(trailing_tp_interval))

    activation_distance = sl_distance * activation_r
    tp_step_distance = sl_distance * trailing_tp_pct

    side = int(side)
    if side == 1:
        activation_level = float(entry_price) - activation_distance
    else:
        activation_level = float(entry_price) + activation_distance

    activated = False
    activation_idx = -1
    tp_line = float(target_price)
    tp_updates = 0
    bars_since_activation = 0

    for i in range(n):
        high_px = float(future_high[i])
        low_px = float(future_low[i])

        if not activated:
            if side == 1:
                triggered = low_px <= activation_level
            else:
                triggered = high_px >= activation_level

            if not triggered:
                continue

            activated = True
            activation_idx = i
            tp_line = float(target_price)
            tp_updates = 1
            bars_since_activation = 0
        else:
            bars_since_activation += 1
            if bars_since_activation % trailing_tp_interval == 0:
                if side == 1:
                    tp_line = max(float(sl_price), float(tp_line) - tp_step_distance)
                else:
                    tp_line = min(float(sl_price), float(tp_line) + tp_step_distance)
                tp_updates += 1

        if side == 1:
            tp_hit = high_px >= tp_line
        else:
            tp_hit = low_px <= tp_line

        if tp_hit:
            exit_price = float(tp_line)
            exit_time_ns = int(future_time_ns[i])
            move_pct = float(_move_pct(side, entry_price, exit_price))
            moved_r = max(0.0, (float(target_price) - exit_price) / sl_distance) if side == 1 else max(0.0, (exit_price - float(target_price)) / sl_distance)
            return {
                "trailing_triggered": True,
                "trailing_tp_activation_idx": int(activation_idx),
                "trailing_tp_idx": int(i),
                "trailing_tp_price": exit_price,
                "trailing_tp_time_ns": exit_time_ns,
                "trailing_tp_line": float(tp_line),
                "trailing_tp_updates": int(tp_updates),
                "trailing_tp_moved_r": float(moved_r),
                "move_pct": move_pct,
                # compatibility keys
                "trailing_activation_idx": int(activation_idx),
                "trailing_exit_idx": int(i),
                "trailing_exit_price": exit_price,
                "trailing_exit_time_ns": exit_time_ns,
                "trailing_stop_price": float(tp_line),
                "trailing_stop_updates": int(tp_updates),
            }

    return nan_out


simulate_adverse_trailing_exit = simulate_adverse_trailing_tp
