# backtest.py

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numba import njit, set_num_threads

# -------------------------
# runtime threading control
# -------------------------
_numba_threads = int(os.getenv("NUMBA_NUM_THREADS", os.getenv("NUMBA_NUM_THREADS_OVERRIDE", "1")))
try:
    set_num_threads(_numba_threads)
except Exception as e:
    raise RuntimeError(f"Failed to set numba thread count to {_numba_threads}") from e

os.environ.setdefault("OMP_NUM_THREADS", str(_numba_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_numba_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_numba_threads))

# -------------------------
# small helpers
# -------------------------
def _funding_per_hour_from_series(
    funding_arr: np.ndarray,
    funding_period_hours: int,
    funding_rate_unit: str = "per_period",
) -> np.ndarray:
    funding = np.asarray(funding_arr, dtype=np.float64)
    if funding_rate_unit == "per_hour":
        return funding
    return funding / float(max(1, int(funding_period_hours)))


def _hours_to_bars(hours: Optional[int], base_minutes: int) -> int:
    if hours is None:
        return 0
    try:
        h = int(hours)
    except Exception:
        return 0
    if h <= 0:
        return 0
    return int((h * 60) / max(1, int(base_minutes)))


def _expiry_to_bars(limit_order_expiry_bars: Optional[int]) -> int:
    """
    New source of truth:
    limit_order_expiry_bars only.

    No fallback to hours.
    """
    if limit_order_expiry_bars is None:
        return 0
    try:
        bars = int(limit_order_expiry_bars)
    except Exception:
        return 0
    return max(0, bars)


def _trade_hit_pct_arrays(
    entry_price_arr: np.ndarray,
    exit_price_arr: np.ndarray,
    exit_reason_arr: np.ndarray,
    side_arr: np.ndarray,
    tp_val: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trade-level hit fields.

    SL_hit:
      - positive = the trade was stopped out at a loss
      - negative = trailing SL protected profit and closed beyond entry

    TP_hit:
      - configured TP percentage when TP is the exit reason
      - NaN otherwise
    """
    entry_price_arr = np.asarray(entry_price_arr, dtype=np.float64)
    exit_price_arr = np.asarray(exit_price_arr, dtype=np.float64)
    exit_reason_arr = np.asarray(exit_reason_arr, dtype=np.int8)
    side_arr = np.asarray(side_arr, dtype=np.int8)

    n = min(entry_price_arr.size, exit_price_arr.size, exit_reason_arr.size, side_arr.size)
    sl_hit = np.full(n, np.nan, dtype=np.float64)
    tp_hit = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return sl_hit, tp_hit

    entry_price_arr = entry_price_arr[:n]
    exit_price_arr = exit_price_arr[:n]
    exit_reason_arr = exit_reason_arr[:n]
    side_arr = side_arr[:n]

    tp_target = abs(float(tp_val)) if np.isfinite(tp_val) else np.nan

    for i in range(n):
        ep = float(entry_price_arr[i])
        xp = float(exit_price_arr[i])
        if not np.isfinite(ep) or ep <= 0.0 or not np.isfinite(xp):
            continue

        if exit_reason_arr[i] == -1:
            move_pct = abs((xp - ep) / ep) * 100.0
            side = int(side_arr[i])

            profitable_stop = (side == 1 and xp > ep) or (side == -1 and xp < ep)
            sl_hit[i] = -move_pct if profitable_stop else move_pct

        elif exit_reason_arr[i] == 1:
            tp_hit[i] = tp_target

    return sl_hit, tp_hit


def _apply_trade_window_interval(
    signal_idx: np.ndarray,
    entry_idx: np.ndarray,
    exit_idx: np.ndarray,
    entry_price: np.ndarray,
    exit_price: np.ndarray,
    rets: np.ndarray,
    exit_reason: np.ndarray,
    tp_hit_price: np.ndarray,
    sl_hit_price: np.ndarray,
    side_out: np.ndarray,
    trade_window_bars: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep only trades that start after the previous accepted trade has exited
    plus the configured cooldown window.

    trade_window_bars = 0 => no filtering.
    """
    if trade_window_bars <= 0 or signal_idx.size == 0:
        return (
            signal_idx,
            entry_idx,
            entry_price,
            exit_idx,
            exit_price,
            rets,
            exit_reason,
            tp_hit_price,
            sl_hit_price,
            side_out,
        )

    keep_idx = []
    last_exit_idx = -1

    for i in range(signal_idx.shape[0]):
        cur_entry_idx = int(entry_idx[i])

        if last_exit_idx >= 0 and cur_entry_idx <= (last_exit_idx + trade_window_bars):
            continue

        keep_idx.append(i)

        cur_exit_idx = int(exit_idx[i])
        last_exit_idx = cur_exit_idx if cur_exit_idx >= 0 else cur_entry_idx

    if not keep_idx:
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_f64 = np.empty(0, dtype=np.float64)
        empty_i8 = np.empty(0, dtype=np.int8)
        return empty_i64, empty_i64, empty_f64, empty_i64, empty_f64, empty_f64, empty_i8, empty_f64, empty_f64, empty_i8

    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    return (
        signal_idx[keep_idx],
        entry_idx[keep_idx],
        entry_price[keep_idx],
        exit_idx[keep_idx],
        exit_price[keep_idx],
        rets[keep_idx],
        exit_reason[keep_idx],
        tp_hit_price[keep_idx],
        sl_hit_price[keep_idx],
        side_out[keep_idx],
    )


@njit(cache=True)
def _max_consecutive_losses_numba(rets):
    max_streak = 0
    streak = 0

    for i in range(rets.shape[0]):
        r = rets[i]

        if not np.isfinite(r):
            r = -1.0

        if r < 0.0:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0

    return max_streak


def compute_max_consecutive_losses(rets: np.ndarray) -> int:
    arr = np.asarray(rets, dtype=np.float64)
    if arr.size == 0:
        return 0
    return int(_max_consecutive_losses_numba(arr))


@njit(cache=True)
def fast_compound_equity_gate(pnl_arr, start_equity=100.0, max_dd_threshold=-1.0):
    n = pnl_arr.shape[0]
    equity = np.empty(n, dtype=np.float64)
    cur = start_equity
    running_max = cur
    max_dd = 0.0

    for i in range(n):
        pnl = pnl_arr[i]

        if not np.isfinite(pnl):
            pnl = -1.0

        if pnl <= -0.999999:
            pnl = -0.999999

        cur = cur * (1.0 + pnl)

        if cur > 1e37:
            cur = 1e37

        if cur < 0.0001 or not np.isfinite(cur):
            cur = 0.0001

        equity[i] = cur

        if cur > running_max:
            running_max = cur

        if running_max > 0.0001:
            dd = (running_max - cur) / running_max
        else:
            dd = 0.0

        if dd > max_dd:
            max_dd = dd

        if max_dd_threshold > 0.0 and max_dd >= max_dd_threshold:
            return equity[: i + 1], max_dd, True

    return equity, max_dd, False


# -------------------------
# stateful pending-order resolver
# -------------------------
@njit(cache=True)
def _resolve_entries_stateful(
    close_f64,
    high_f64,
    low_f64,
    signal_idx_i64,
    side_i8,
    limit_window_bars_i64,
    use_limit_entry_bool,
):
    """
    Universal rule:
      - only one pending limit order can exist at a time
      - a newer signal cancels the old pending order if it has not filled yet

    If use_limit_entry_bool=False:
      - direct entry on the signal bar
      - no pending order logic
    """
    n_rows = close_f64.shape[0]
    n_signals = signal_idx_i64.shape[0]

    signal_out = np.empty(n_signals, dtype=np.int64)
    entry_idx_out = np.empty(n_signals, dtype=np.int64)
    entry_price_out = np.empty(n_signals, dtype=np.float64)
    side_out = np.empty(n_signals, dtype=np.int8)

    k = 0

    if not use_limit_entry_bool:
        for i in range(n_signals):
            signal_idx = int(signal_idx_i64[i])
            if signal_idx < 0 or signal_idx >= n_rows:
                continue

            side = int(side_i8[i])
            if side != 1 and side != -1:
                continue

            signal_out[k] = signal_idx
            entry_idx_out[k] = signal_idx
            entry_price_out[k] = close_f64[signal_idx]
            side_out[k] = side
            k += 1

        return signal_out[:k], entry_idx_out[:k], entry_price_out[:k], side_out[:k]

    pending_active = False
    pending_signal_idx = -1
    pending_side = 0
    pending_limit_price = 0.0
    pending_scan_start = 0
    pending_expiry_idx = -1

    for i in range(n_signals):
        signal_idx = int(signal_idx_i64[i])
        if signal_idx < 0 or signal_idx >= n_rows:
            continue

        side = int(side_i8[i])
        if side != 1 and side != -1:
            continue

        # Check whether the current pending order fills before the next signal arrives.
        # This is the only place where a pending order can become a trade.
        if pending_active:
            scan_end = signal_idx - 1
            if pending_expiry_idx < scan_end:
                scan_end = pending_expiry_idx

            if scan_end >= pending_scan_start:
                j = pending_scan_start
                filled = False
                fill_idx = -1

                while j <= scan_end:
                    if pending_side == 1:
                        if low_f64[j] <= pending_limit_price:
                            filled = True
                            fill_idx = j
                            break
                    else:
                        if high_f64[j] >= pending_limit_price:
                            filled = True
                            fill_idx = j
                            break
                    j += 1

                if filled:
                    signal_out[k] = pending_signal_idx
                    entry_idx_out[k] = fill_idx
                    entry_price_out[k] = pending_limit_price
                    side_out[k] = pending_side
                    k += 1

            # New signal always cancels the older pending order.
            pending_active = False

        # Place the new pending order from this signal candle.
        signal_close = float(close_f64[signal_idx])
        if side == 1:
            limit_price = 0.5 * (signal_close + float(low_f64[signal_idx]))
        else:
            limit_price = 0.5 * (signal_close + float(high_f64[signal_idx]))

        pending_active = True
        pending_signal_idx = signal_idx
        pending_side = side
        pending_limit_price = limit_price
        pending_scan_start = signal_idx + 1
        pending_expiry_idx = signal_idx + int(max(0, limit_window_bars_i64))

    # Flush the last pending order.
    if pending_active and pending_scan_start < n_rows:
        scan_end = pending_expiry_idx
        if scan_end >= n_rows:
            scan_end = n_rows - 1

        if scan_end >= pending_scan_start:
            j = pending_scan_start
            while j <= scan_end:
                if pending_side == 1:
                    if low_f64[j] <= pending_limit_price:
                        signal_out[k] = pending_signal_idx
                        entry_idx_out[k] = j
                        entry_price_out[k] = pending_limit_price
                        side_out[k] = pending_side
                        k += 1
                        break
                else:
                    if high_f64[j] >= pending_limit_price:
                        signal_out[k] = pending_signal_idx
                        entry_idx_out[k] = j
                        entry_price_out[k] = pending_limit_price
                        side_out[k] = pending_side
                        k += 1
                        break
                j += 1

    return signal_out[:k], entry_idx_out[:k], entry_price_out[:k], side_out[:k]


# -------------------------
# overlap / flip mode
# -------------------------
@njit(cache=True)
def _apply_trade_overlap_mode_numba(
    signal_idx,
    entry_idx,
    entry_price,
    exit_idx,
    exit_price,
    rets,
    exit_reason,
    tp_hit_price,
    sl_hit_price,
    side_out,
    trade_overlap_bool,
    trade_flip_on_entry_bool,
):
    """
    trade_overlap=True:
      - keep current behavior
      - trades may overlap

    trade_overlap=False and trade_flip_on_entry=False:
      - hard block
      - if a trade is still open, later trades do not count until it closes

    trade_overlap=False and trade_flip_on_entry=True:
      - flip mode
      - a new filled trade cuts the previous one off at the new entry price
      - then the new trade becomes the active trade
    """
    if trade_overlap_bool or signal_idx.size == 0:
        return (
            signal_idx,
            entry_idx,
            entry_price,
            exit_idx,
            exit_price,
            rets,
            exit_reason,
            tp_hit_price,
            sl_hit_price,
            side_out,
        )

    n = signal_idx.shape[0]
    keep_signal = np.empty(n, dtype=np.int64)
    keep_entry_idx = np.empty(n, dtype=np.int64)
    keep_entry_price = np.empty(n, dtype=np.float64)
    keep_exit_idx = np.empty(n, dtype=np.int64)
    keep_exit_price = np.empty(n, dtype=np.float64)
    keep_rets = np.empty(n, dtype=np.float64)
    keep_exit_reason = np.empty(n, dtype=np.int8)
    keep_tp_hit_price = np.empty(n, dtype=np.float64)
    keep_sl_hit_price = np.empty(n, dtype=np.float64)
    keep_side_out = np.empty(n, dtype=np.int8)

    k = 0
    active_open = False
    active_exit_idx = -1

    for i in range(n):
        cur_signal_idx = int(signal_idx[i])
        cur_entry_idx = int(entry_idx[i])
        cur_entry_price = float(entry_price[i])
        cur_exit_idx = int(exit_idx[i])
        cur_exit_price = float(exit_price[i])
        cur_rets = float(rets[i])
        cur_exit_reason = int(exit_reason[i])
        cur_tp_hit_price = float(tp_hit_price[i])
        cur_sl_hit_price = float(sl_hit_price[i])
        cur_side = int(side_out[i])

        if cur_entry_idx < 0 or cur_exit_idx < 0:
            continue

        # First accepted trade or a trade that starts after the active one is already closed.
        if (not active_open) or (cur_entry_idx > active_exit_idx):
            keep_signal[k] = cur_signal_idx
            keep_entry_idx[k] = cur_entry_idx
            keep_entry_price[k] = cur_entry_price
            keep_exit_idx[k] = cur_exit_idx
            keep_exit_price[k] = cur_exit_price
            keep_rets[k] = cur_rets
            keep_exit_reason[k] = cur_exit_reason
            keep_tp_hit_price[k] = cur_tp_hit_price
            keep_sl_hit_price[k] = cur_sl_hit_price
            keep_side_out[k] = cur_side
            active_open = True
            active_exit_idx = cur_exit_idx
            k += 1
            continue

        # Here, the new trade arrives while the active trade is still open.
        if not trade_flip_on_entry_bool:
            # Hard block mode: discard the new trade entirely.
            continue

        # Flip mode:
        # cut the previous accepted trade at the new entry price and replace it.
        prev = k - 1
        prev_side = int(keep_side_out[prev])
        prev_entry_price = float(keep_entry_price[prev])

        keep_exit_idx[prev] = cur_entry_idx
        keep_exit_price[prev] = cur_entry_price
        keep_exit_reason[prev] = 0
        keep_tp_hit_price[prev] = np.nan
        keep_sl_hit_price[prev] = np.nan

        if prev_side == 1:
            keep_rets[prev] = (cur_entry_price - prev_entry_price) / prev_entry_price
        else:
            keep_rets[prev] = (prev_entry_price - cur_entry_price) / prev_entry_price

        keep_signal[k] = cur_signal_idx
        keep_entry_idx[k] = cur_entry_idx
        keep_entry_price[k] = cur_entry_price
        keep_exit_idx[k] = cur_exit_idx
        keep_exit_price[k] = cur_exit_price
        keep_rets[k] = cur_rets
        keep_exit_reason[k] = cur_exit_reason
        keep_tp_hit_price[k] = cur_tp_hit_price
        keep_sl_hit_price[k] = cur_sl_hit_price
        keep_side_out[k] = cur_side

        active_open = True
        active_exit_idx = cur_exit_idx
        k += 1

    return (
        keep_signal[:k],
        keep_entry_idx[:k],
        keep_entry_price[:k],
        keep_exit_idx[:k],
        keep_exit_price[:k],
        keep_rets[:k],
        keep_exit_reason[:k],
        keep_tp_hit_price[:k],
        keep_sl_hit_price[:k],
        keep_side_out[:k],
    )


@njit(cache=True)
def _compute_trade_exits_from_fills(
    close_f64,
    high_f64,
    low_f64,
    entry_idx_i64,
    entry_price_f64,
    side_i8,
    sl_f64,
    tp_f64,
    sl_tp_in_pct_bool,
    exit_window_bars_i64,
    spread_f64,
    spread_is_percent_bool,
    conservative_sl_first_bool,
    use_trailing_sl_bool,
    trailing_sl_pct_f64,
    trailing_sl_interval_i64,
    trailing_sl_stop_at_pos_bool,
):
    """
    Independent trade exit simulation from actual fill points.
    """
    n_rows = close_f64.shape[0]
    n = entry_idx_i64.shape[0]

    exit_idx_out = np.empty(n, dtype=np.int64)
    exit_price_out = np.empty(n, dtype=np.float64)
    rets_out = np.empty(n, dtype=np.float64)
    exit_reason_out = np.empty(n, dtype=np.int8)  # 1=TP, -1=SL, 0=time exit
    tp_hit_price_out = np.empty(n, dtype=np.float64)
    sl_hit_price_out = np.empty(n, dtype=np.float64)

    for i in range(n):
        fill_idx = int(entry_idx_i64[i])
        side = int(side_i8[i])
        fill_price = float(entry_price_f64[i])

        if fill_idx < 0 or fill_idx >= n_rows or (side != 1 and side != -1) or not np.isfinite(fill_price) or fill_price <= 0.0:
            exit_idx_out[i] = -1
            exit_price_out[i] = np.nan
            rets_out[i] = np.nan
            exit_reason_out[i] = 0
            tp_hit_price_out[i] = np.nan
            sl_hit_price_out[i] = np.nan
            continue

        # TP/SL setup
        if sl_tp_in_pct_bool:
            if side == 1:
                tp_price = fill_price * (1.0 + tp_f64 / 100.0)
                sl_price = fill_price * (1.0 - sl_f64 / 100.0)
            else:
                tp_price = fill_price * (1.0 - tp_f64 / 100.0)
                sl_price = fill_price * (1.0 + sl_f64 / 100.0)
        else:
            if side == 1:
                tp_price = fill_price + tp_f64
                sl_price = fill_price - sl_f64
            else:
                tp_price = fill_price - tp_f64
                sl_price = fill_price + sl_f64

        # Spread is applied consistently in price space here.
        if spread_f64 != 0.0:
            if spread_is_percent_bool:
                spread_abs = fill_price * (spread_f64 / 100.0)
            else:
                spread_abs = spread_f64

            half_spread = spread_abs / 2.0

            if side == 1:
                tp_price_adj = tp_price - half_spread
                sl_price_adj = sl_price - half_spread
            else:
                tp_price_adj = tp_price + half_spread
                sl_price_adj = sl_price + half_spread
        else:
            tp_price_adj = tp_price
            sl_price_adj = sl_price

        base_sl_dist = abs(sl_price_adj - fill_price)

        exit_at = -1
        exit_price = 0.0
        exit_reason = 0
        tp_hit_price = np.nan
        sl_hit_price = np.nan

        max_j = n_rows - 1
        if exit_window_bars_i64 >= 0:
            tmp = fill_idx + int(exit_window_bars_i64)
            if tmp < max_j:
                max_j = tmp

        j = fill_idx + 1

        while j <= max_j:
            hi = high_f64[j]
            lo = low_f64[j]

            current_sl_price = sl_price_adj
            if use_trailing_sl_bool and trailing_sl_interval_i64 > 0 and trailing_sl_pct_f64 > 0.0:
                closed_bars = j - fill_idx - 1
                if closed_bars >= trailing_sl_interval_i64:
                    steps = closed_bars // trailing_sl_interval_i64
                    move = base_sl_dist * trailing_sl_pct_f64 * float(steps)

                    if side == 1:
                        current_sl_price = sl_price_adj + move
                        if trailing_sl_stop_at_pos_bool and current_sl_price > fill_price:
                            current_sl_price = fill_price
                    else:
                        current_sl_price = sl_price_adj - move
                        if trailing_sl_stop_at_pos_bool and current_sl_price < fill_price:
                            current_sl_price = fill_price

            if side == 1:
                hit_sl = lo <= current_sl_price
                hit_tp = hi >= tp_price_adj

                if hit_sl and hit_tp:
                    if conservative_sl_first_bool:
                        exit_price = current_sl_price
                        exit_reason = -1
                        sl_hit_price = current_sl_price
                    else:
                        exit_price = tp_price_adj
                        exit_reason = 1
                        tp_hit_price = tp_price_adj
                    exit_at = j
                    break

                if hit_tp:
                    exit_price = tp_price_adj
                    exit_reason = 1
                    tp_hit_price = tp_price_adj
                    exit_at = j
                    break

                if hit_sl:
                    exit_price = current_sl_price
                    exit_reason = -1
                    sl_hit_price = current_sl_price
                    exit_at = j
                    break

            else:
                hit_sl = hi >= current_sl_price
                hit_tp = lo <= tp_price_adj

                if hit_sl and hit_tp:
                    if conservative_sl_first_bool:
                        exit_price = current_sl_price
                        exit_reason = -1
                        sl_hit_price = current_sl_price
                    else:
                        exit_price = tp_price_adj
                        exit_reason = 1
                        tp_hit_price = tp_price_adj
                    exit_at = j
                    break

                if hit_tp:
                    exit_price = tp_price_adj
                    exit_reason = 1
                    tp_hit_price = tp_price_adj
                    exit_at = j
                    break

                if hit_sl:
                    exit_price = current_sl_price
                    exit_reason = -1
                    sl_hit_price = current_sl_price
                    exit_at = j
                    break

            j += 1

        if exit_at == -1:
            if exit_window_bars_i64 >= 0:
                last_idx = min(fill_idx + int(exit_window_bars_i64), n_rows - 1)
            else:
                last_idx = n_rows - 1

            if last_idx > fill_idx:
                exit_price = close_f64[last_idx]
                exit_reason = 0
                exit_at = last_idx
            else:
                exit_price = fill_price
                exit_reason = 0
                exit_at = fill_idx

        if side == 1:
            ret = (exit_price - fill_price) / fill_price
        else:
            ret = (fill_price - exit_price) / fill_price

        exit_idx_out[i] = exit_at
        exit_price_out[i] = exit_price
        rets_out[i] = ret
        exit_reason_out[i] = exit_reason
        tp_hit_price_out[i] = tp_hit_price
        sl_hit_price_out[i] = sl_hit_price

    return exit_idx_out, exit_price_out, rets_out, exit_reason_out, tp_hit_price_out, sl_hit_price_out


def backtest_from_arrays(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    signal_idx_arr: np.ndarray,
    side_arr: np.ndarray,
    sl: float = 0.0,
    tp: float = 0.0,
    sl_tp_in_pct: bool = True,
    limit_order_expiry_bars: int = 0,
    exit_window_h: Optional[int] = None,
    trade_window_interval: Optional[int] = 0,
    base_minutes: int = 5,
    spread: float = 0.0,
    spread_is_percent: bool = True,
    conservative_sl_first: bool = True,
    use_limit_entry: bool = True,
    trade_overlap: bool = True,
    trade_flip_on_entry: bool = False,
    use_trailing_sl: bool = False,
    trailing_sl_pct: float = 0.0,
    trailing_sl_interval: int = 0,
    trailing_sl_stop_at_pos: bool = True,
) -> Dict:
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    signal_idx_arr = np.asarray(signal_idx_arr, dtype=np.int64)
    side_arr = np.asarray(side_arr, dtype=np.int8)

    # Keep the signal stream deterministic before any stateful logic.
    if signal_idx_arr.size:
        order = np.argsort(signal_idx_arr, kind="mergesort")
        signal_idx_arr = signal_idx_arr[order]
        side_arr = side_arr[order]

    limit_window_bars = _expiry_to_bars(limit_order_expiry_bars)
    exit_window_bars = _hours_to_bars(exit_window_h, base_minutes)
    trade_window_bars = max(0, _hours_to_bars(trade_window_interval, base_minutes))

    # Step 1: resolve entries from the raw signal stream.
    # This is where the universal single-pending-order rule lives.
    signal_idx, entry_idx, entry_price, side_out = _resolve_entries_stateful(
        close_f64=close,
        high_f64=high,
        low_f64=low,
        signal_idx_i64=signal_idx_arr,
        side_i8=side_arr,
        limit_window_bars_i64=int(limit_window_bars),
        use_limit_entry_bool=bool(use_limit_entry),
    )

    if signal_idx.size == 0:
        return {
            "signal_idx": np.array([], dtype=np.int64),
            "entry_idx": np.array([], dtype=np.int64),
            "entry_price": np.array([], dtype=np.float64),
            "exit_idx": np.array([], dtype=np.int64),
            "exit_price": np.array([], dtype=np.float64),
            "rets": np.array([], dtype=np.float64),
            "exit_reason": np.array([], dtype=np.int8),
            "side": np.array([], dtype=np.int8),
            "SL_hit": np.array([], dtype=np.float64),
            "TP_hit": np.array([], dtype=np.float64),
            "max_consecutive_losses": 0,
        }

    # Step 2: compute each trade's natural exit from its actual fill.
    (
        exit_idx,
        exit_price,
        rets,
        exit_reason,
        tp_hit_price,
        sl_hit_price,
    ) = _compute_trade_exits_from_fills(
        close_f64=close,
        high_f64=high,
        low_f64=low,
        entry_idx_i64=entry_idx,
        entry_price_f64=entry_price,
        side_i8=side_out,
        sl_f64=float(sl),
        tp_f64=float(tp),
        sl_tp_in_pct_bool=bool(sl_tp_in_pct),
        exit_window_bars_i64=int(exit_window_bars),
        spread_f64=float(spread),
        spread_is_percent_bool=bool(spread_is_percent),
        conservative_sl_first_bool=bool(conservative_sl_first),
        use_trailing_sl_bool=bool(use_trailing_sl),
        trailing_sl_pct_f64=float(trailing_sl_pct),
        trailing_sl_interval_i64=int(trailing_sl_interval),
        trailing_sl_stop_at_pos_bool=bool(trailing_sl_stop_at_pos),
    )

    # Step 3: apply overlap policy.
    # trade_overlap=True  -> current behavior
    # trade_overlap=False -> only one active trade at a time
    #   - trade_flip_on_entry=False: later trades are blocked until the active trade closes
    #   - trade_flip_on_entry=True : a new filled trade closes the active trade at the new entry
    (
        signal_idx,
        entry_idx,
        entry_price,
        exit_idx,
        exit_price,
        rets,
        exit_reason,
        tp_hit_price,
        sl_hit_price,
        side_out,
    ) = _apply_trade_overlap_mode_numba(
        signal_idx=signal_idx,
        entry_idx=entry_idx,
        entry_price=entry_price,
        exit_idx=exit_idx,
        exit_price=exit_price,
        rets=rets,
        exit_reason=exit_reason,
        tp_hit_price=tp_hit_price,
        sl_hit_price=sl_hit_price,
        side_out=side_out,
        trade_overlap_bool=bool(trade_overlap),
        trade_flip_on_entry_bool=bool(trade_flip_on_entry),
    )

    # Step 4: optional research-style cooldown filter.
    # This is still post-hoc and separate from trade_overlap.
    if trade_window_bars > 0 and signal_idx.size > 0:
        (
            signal_idx,
            entry_idx,
            entry_price,
            exit_idx,
            exit_price,
            rets,
            exit_reason,
            tp_hit_price,
            sl_hit_price,
            side_out,
        ) = _apply_trade_window_interval(
            signal_idx=signal_idx,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            entry_price=entry_price,
            exit_price=exit_price,
            rets=rets,
            exit_reason=exit_reason,
            tp_hit_price=tp_hit_price,
            sl_hit_price=sl_hit_price,
            side_out=side_out,
            trade_window_bars=trade_window_bars,
        )

    sl_hit, tp_hit = _trade_hit_pct_arrays(
        entry_price_arr=entry_price,
        exit_price_arr=exit_price,
        exit_reason_arr=exit_reason,
        side_arr=side_out,
        tp_val=tp,
    )

    max_consecutive_losses = compute_max_consecutive_losses(rets)

    return {
        "signal_idx": signal_idx,
        "entry_idx": entry_idx,
        "entry_price": entry_price,
        "exit_idx": exit_idx,
        "exit_price": exit_price,
        "rets": rets,
        "exit_reason": exit_reason,
        "side": side_out,
        "SL_hit": sl_hit,
        "TP_hit": tp_hit,
        "max_consecutive_losses": max_consecutive_losses,
    }


def backtest_signals_sl_tp_rets(
    main_close_arr: np.ndarray,
    main_high_arr: np.ndarray,
    main_low_arr: np.ndarray,
    main_time_ns_arr: np.ndarray,
    sig_idxs: np.ndarray,
    sl: float = 0.0,
    tp: float = 0.0,
    sl_tp_in_pct: bool = True,
    exit_window_h: Optional[int] = None,
    limit_order_expiry_bars: int = 0,
    trade_window_interval: Optional[int] = 0,
    base_minutes: int = 5,
    spread: float = 0.0,
    spread_is_percent: bool = True,
    conservative_sl_first: bool = True,
    side_flag: Optional[int] = None,          # Change to Optional
    side_arr: Optional[np.ndarray] = None,   # Add this parameter
    use_limit_entry: bool = True,
    trade_overlap: bool = True,
    trade_flip_on_entry: bool = False,
    use_trailing_sl: bool = False,
    trailing_sl_pct: float = 0.0,
    trailing_sl_interval: int = 0,
    trailing_sl_stop_at_pos: bool = True,
) -> Dict:
    if sig_idxs is None or sig_idxs.size == 0:
        return {
            "signal_idx": np.array([], dtype=np.int64),
            "entry_idx": np.array([], dtype=np.int64),
            "entry_price": np.array([], dtype=np.float64),
            "exit_idx": np.array([], dtype=np.int64),
            "exit_price": np.array([], dtype=np.float64),
            "rets": np.array([], dtype=np.float64),
            "exit_reason": np.array([], dtype=np.int8),
            "side": np.array([], dtype=np.int8),
            "SL_hit": np.array([], dtype=np.float64),
            "TP_hit": np.array([], dtype=np.float64),
            "max_consecutive_losses": 0,
        }

    # FIX: Use the provided side_arr if available, otherwise fallback to side_flag
    if side_arr is None:
        effective_side = 1 if side_flag is None else side_flag
        side_arr = np.full(sig_idxs.shape[0], effective_side, dtype=np.int8)
    else:
        side_arr = np.asarray(side_arr, dtype=np.int8)

    return backtest_from_arrays(
        close=main_close_arr,
        high=main_high_arr,
        low=main_low_arr,
        signal_idx_arr=sig_idxs.astype(np.int64),
        side_arr=side_arr,
        sl=sl,
        tp=tp,
        sl_tp_in_pct=sl_tp_in_pct,
        limit_order_expiry_bars=limit_order_expiry_bars,
        exit_window_h=exit_window_h,
        trade_window_interval=trade_window_interval,
        base_minutes=base_minutes,
        spread=spread,
        spread_is_percent=spread_is_percent,
        conservative_sl_first=conservative_sl_first,
        use_limit_entry=use_limit_entry,
        trade_overlap=trade_overlap,
        trade_flip_on_entry=trade_flip_on_entry,
        use_trailing_sl=use_trailing_sl,
        trailing_sl_pct=trailing_sl_pct,
        trailing_sl_interval=trailing_sl_interval,
        trailing_sl_stop_at_pos=trailing_sl_stop_at_pos,
    )