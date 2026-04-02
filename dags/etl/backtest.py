"""
Backtest numeric kernels for close/high/low driven simulation.

- Supports direct entry or limit-entry mode.
- Limit-entry mode:
  - signal candle only places a pending order
  - fill happens only on future candles
  - trade starts at the filled limit price
- Memory-conscious: accept numpy arrays, use np.asarray for cheap view/cast.
- Numba kernels compiled with cache.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from numba import njit

from etl.feature_helpers import get_ma_price_gaps_for_indices


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
    if hours is None or int(hours) <= 0:
        return -1
    return int((int(hours) * 60) / max(1, int(base_minutes)))

def _apply_trade_window_interval(
    signal_idx: np.ndarray,
    entry_idx: np.ndarray,
    exit_idx: np.ndarray,
    entry_price: np.ndarray,
    exit_price: np.ndarray,
    rets: np.ndarray,
    exit_reason: np.ndarray,
    trade_window_bars: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep only trades that start after the previous accepted trade has exited
    plus the configured cooldown window.

    trade_window_bars = 0 => no filtering.
    """
    if trade_window_bars <= 0 or signal_idx.size == 0:
        return signal_idx, entry_idx, entry_price, exit_idx, exit_price, rets, exit_reason

    keep_idx = []
    last_exit_idx = -1

    for i in range(signal_idx.shape[0]):
        cur_entry_idx = int(entry_idx[i])

        # cooldown measured from previous accepted trade's exit
        if last_exit_idx >= 0 and cur_entry_idx <= (last_exit_idx + trade_window_bars):
            continue

        keep_idx.append(i)

        cur_exit_idx = int(exit_idx[i])
        last_exit_idx = cur_exit_idx if cur_exit_idx >= 0 else cur_entry_idx

    if not keep_idx:
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_f64 = np.empty(0, dtype=np.float64)
        empty_i8 = np.empty(0, dtype=np.int8)
        return empty_i64, empty_i64, empty_f64, empty_i64, empty_f64, empty_f64, empty_i8

    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    return (
        signal_idx[keep_idx],
        entry_idx[keep_idx],
        entry_price[keep_idx],
        exit_idx[keep_idx],
        exit_price[keep_idx],
        rets[keep_idx],
        exit_reason[keep_idx],
    )

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
# pnl kernel (numba)
# -------------------------
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

        # Risk-based sizing:
        # SL hit -> loss ~= risk_pct
        # TP hit -> gain scales by TP/SL through the actual exit price
        pos_size = risk / (eff_sl / 100.0)

        if side == 1:
            trade_ret = (xp - ep) / ep_safe
        else:
            trade_ret = (ep - xp) / ep_safe

        pnl = (pos_size * trade_ret) - f_fee

        if not np.isfinite(pnl):
            pnl = -1.0
        if pnl <= -0.999999:
            pnl = -0.999999

        pnl_out[i] = pnl

    return pnl_out


def compute_pnl_pct_vectorized(
    closed_masked_rets: np.ndarray,
    closed_entry_idxs: np.ndarray,
    closed_exit_idxs: np.ndarray,
    main_close_arr: np.ndarray,
    main_time_ns_arr: np.ndarray,
    spread_arr: Optional[np.ndarray],
    funding_arr: Optional[np.ndarray],
    entry_prices_arr: Optional[np.ndarray],
    sl_val: float = 0.0,
    tp_val: float = 0.0,
    risk_pct: float = 0.005,
    sl_tp_in_pct: bool = True,
    funding_period_hours: int = 8,
    funding_rate_unit: str = "per_period",
    spread_is_percent: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    n = closed_masked_rets.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    entry_prices = (
        np.asarray(entry_prices_arr, dtype=np.float64)
        if entry_prices_arr is not None
        else np.asarray(main_close_arr[closed_entry_idxs], dtype=np.float64)
    )

    spreads = (
        np.asarray(spread_arr[closed_entry_idxs], dtype=np.float64)
        if spread_arr is not None
        else np.zeros(n, dtype=np.float64)
    )

    funding_raw = (
        np.asarray(funding_arr[closed_entry_idxs], dtype=np.float64)
        if funding_arr is not None
        else np.zeros(n, dtype=np.float64)
    )

    pnl_pct = _numba_pnl_kernel_optimized(
        np.asarray(closed_masked_rets, dtype=np.float64),
        entry_prices,
        spreads,
        funding_raw,
        np.asarray(closed_entry_idxs, dtype=np.int64),
        np.asarray(closed_exit_idxs, dtype=np.int64),
        np.asarray(main_time_ns_arr, dtype=np.int64),
        float(risk_pct),
        float(sl_val),
        float(tp_val),
        bool(sl_tp_in_pct),
        bool(spread_is_percent),
        int(funding_period_hours),
    )

    exit_times_ns = np.asarray(main_time_ns_arr[closed_exit_idxs], dtype=np.int64)
    return pnl_pct, exit_times_ns


# -------------------------
# backtest kernel
# -------------------------
@njit(cache=True)
def _backtest_kernel_limit_ohlc(
    close_f64,
    high_f64,
    low_f64,
    signal_idx_i64,
    side_i8,
    sl_f64,
    tp_f64,
    sl_tp_in_pct_bool,
    limit_window_bars_i64,
    exit_window_bars_i64,
    spread_f64,
    conservative_sl_first_bool,
    use_limit_entry_bool,
    use_trailing_sl_bool,
    trailing_sl_pct_f64,
    trailing_sl_interval_i64,
    trailing_sl_stop_at_pos_bool,
):
    n_rows = close_f64.shape[0]
    n_signals = signal_idx_i64.shape[0]

    signal_out = np.empty(n_signals, dtype=np.int64)
    entry_idx_out = np.empty(n_signals, dtype=np.int64)
    entry_price_out = np.empty(n_signals, dtype=np.float64)
    exit_idx_out = np.empty(n_signals, dtype=np.int64)
    exit_price_out = np.empty(n_signals, dtype=np.float64)
    rets_out = np.empty(n_signals, dtype=np.float64)
    exit_reason_out = np.empty(n_signals, dtype=np.int8)  # 1=TP, -1=SL, 0=time exit

    k = 0

    for i in range(n_signals):
        signal_idx = int(signal_idx_i64[i])
        if signal_idx < 0 or signal_idx >= n_rows:
            continue

        side = int(side_i8[i])
        signal_close = float(close_f64[signal_idx])

        fill_idx = -1
        fill_price = 0.0

        if use_limit_entry_bool:
            if side == 1:
                limit_price = 0.5 * (signal_close + float(low_f64[signal_idx]))
            else:
                limit_price = 0.5 * (signal_close + float(high_f64[signal_idx]))

            max_limit_j = n_rows - 1
            if limit_window_bars_i64 >= 0:
                tmp = signal_idx + int(limit_window_bars_i64)
                if tmp < max_limit_j:
                    max_limit_j = tmp

            j = signal_idx + 1
            while j <= max_limit_j:
                hi = high_f64[j]
                lo = low_f64[j]

                if side == 1:
                    if lo <= limit_price:
                        fill_idx = j
                        fill_price = limit_price
                        break
                else:
                    if hi >= limit_price:
                        fill_idx = j
                        fill_price = limit_price
                        break

                j += 1

            if fill_idx == -1:
                continue
        else:
            fill_idx = signal_idx
            fill_price = signal_close

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

        if spread_f64 != 0.0:
            if side == 1:
                tp_price_adj = tp_price - (spread_f64 / 2.0)
                sl_price_adj = sl_price - (spread_f64 / 2.0)
            else:
                tp_price_adj = tp_price + (spread_f64 / 2.0)
                sl_price_adj = sl_price + (spread_f64 / 2.0)
        else:
            tp_price_adj = tp_price
            sl_price_adj = sl_price

        base_sl_dist = abs(sl_price_adj - fill_price)

        exit_at = -1
        exit_price = 0.0
        exit_reason = 0

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
                # Only tighten after fully closed candles.
                # For bar j, the last fully closed candle is j - 1.
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
                    else:
                        exit_price = tp_price_adj
                        exit_reason = 1
                    exit_at = j
                    break

                if hit_tp:
                    exit_price = tp_price_adj
                    exit_reason = 1
                    exit_at = j
                    break

                if hit_sl:
                    exit_price = current_sl_price
                    exit_reason = -1
                    exit_at = j
                    break

            else:
                hit_sl = hi >= current_sl_price
                hit_tp = lo <= tp_price_adj

                if hit_sl and hit_tp:
                    if conservative_sl_first_bool:
                        exit_price = current_sl_price
                        exit_reason = -1
                    else:
                        exit_price = tp_price_adj
                        exit_reason = 1
                    exit_at = j
                    break

                if hit_tp:
                    exit_price = tp_price_adj
                    exit_reason = 1
                    exit_at = j
                    break

                if hit_sl:
                    exit_price = current_sl_price
                    exit_reason = -1
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

        signal_out[k] = signal_idx
        entry_idx_out[k] = fill_idx
        entry_price_out[k] = fill_price
        exit_idx_out[k] = exit_at
        exit_price_out[k] = exit_price
        rets_out[k] = ret
        exit_reason_out[k] = exit_reason
        k += 1

    return (
        signal_out[:k],
        entry_idx_out[:k],
        entry_price_out[:k],
        exit_idx_out[:k],
        exit_price_out[:k],
        rets_out[:k],
        exit_reason_out[:k],
    )


# -------------------------
# grid kernel
# -------------------------
@njit(parallel=False, cache=True)
def _grid_kernel_numba(
    closes_f32,
    n_min_f32,
    n_max_f32,
    t_min_ns_i64,
    t_max_ns_i64,
    s_vals_f32,
    t_vals_f32,
    breakeven_after_f32,
    min_rr_f32,
    tie_mode_i32,
    position_type_i32,
):
    n_rows = closes_f32.shape[0]
    ns = s_vals_f32.shape[0]
    nt = t_vals_f32.shape[0]
    good = np.zeros(ns * nt, dtype=np.int32)
    be = np.zeros(ns * nt, dtype=np.int32)

    for i in range(ns):
        sl_val = s_vals_f32[i]

        if sl_val <= 0.0001:
            continue

        min_tp_allowed = sl_val * min_rr_f32

        for j in range(nt):
            tp_val = t_vals_f32[j]
            if tp_val < min_tp_allowed:
                continue

            count_good = 0
            count_be = 0

            for r in range(n_rows):
                if t_min_ns_i64[r] < 0:
                    continue

                cp = closes_f32[r]

                if position_type_i32 == 1:
                    t_tp = t_max_ns_i64[r]
                    t_sl = t_min_ns_i64[r]
                    dist_tp = n_max_f32[r] - cp
                    dist_sl = cp - n_min_f32[r]
                else:
                    t_tp = t_min_ns_i64[r]
                    t_sl = t_max_ns_i64[r]
                    dist_tp = cp - n_min_f32[r]
                    dist_sl = n_max_f32[r] - cp

                hit_tp = dist_tp >= tp_val
                hit_sl = dist_sl >= sl_val

                tp_wins = False
                if hit_tp and hit_sl:
                    if tie_mode_i32 == 1:
                        tp_wins = (t_tp <= t_sl)
                    else:
                        tp_wins = (t_tp < t_sl)
                elif hit_tp:
                    tp_wins = True

                if tp_wins:
                    count_good += 1
                elif dist_tp >= (tp_val - breakeven_after_f32):
                    count_be += 1

            idx = i * nt + j
            good[idx] = count_good
            be[idx] = count_be

    return good, be


# -------------------------
# precompute windows
# -------------------------
@njit(cache=True)
def _compute_windows_numba(main_close_f64, main_time_ns_i64, entry_idxs_i64, steps_i32):
    Nsig = entry_idxs_i64.shape[0]
    n_min = np.empty(Nsig, dtype=np.float64)
    n_max = np.empty(Nsig, dtype=np.float64)
    t_min_ns = np.empty(Nsig, dtype=np.int64)
    t_max_ns = np.empty(Nsig, dtype=np.int64)
    closes = np.empty(Nsig, dtype=np.float64)

    n_rows = main_close_f64.shape[0]
    for i in range(Nsig):
        pos = int(entry_idxs_i64[i])
        if pos < 0:
            pos = 0
        if pos >= n_rows:
            pos = n_rows - 1

        closes[i] = main_close_f64[pos]

        start = pos + 1
        end = pos + 1 + int(steps_i32)
        if start >= n_rows or start >= end:
            n_min[i] = 0.0
            n_max[i] = 0.0
            t_min_ns[i] = -1
            t_max_ns[i] = -1
            continue

        if end > n_rows:
            end = n_rows

        min_val = main_close_f64[start]
        max_val = main_close_f64[start]
        min_time = main_time_ns_i64[start]
        max_time = main_time_ns_i64[start]

        for k in range(start + 1, end):
            val = main_close_f64[k]
            t = main_time_ns_i64[k]
            if val < min_val:
                min_val = val
                min_time = t
            if val > max_val:
                max_val = val
                max_time = t

        n_min[i] = min_val
        n_max[i] = max_val
        t_min_ns[i] = min_time
        t_max_ns[i] = max_time

    return closes, n_min, n_max, t_min_ns, t_max_ns


def precompute_kernel_arrays(df_main, side_df, exit_window_h: int, base_minutes: int = 5):
    """
    Build arrays required by the numba kernel for side signals.
    """
    main_close = df_main["close"].to_numpy().astype(np.float64)
    main_time_ns = df_main["time_ns"].to_numpy().astype(np.int64)
    sig_time_ns = side_df["time_ns"].to_numpy().astype(np.int64)
    entry_prices = side_df["close"].to_numpy().astype(np.float64)

    steps = max(1, int((exit_window_h * 60) / base_minutes))

    entry_idxs = np.searchsorted(main_time_ns, sig_time_ns, side="left")
    entry_idxs = np.clip(entry_idxs, 0, max(0, main_time_ns.shape[0] - 1)).astype(np.int64)

    cl, mi, ma, t_mi, t_ma = _compute_windows_numba(main_close, main_time_ns, entry_idxs, np.int32(steps))
    return cl.astype(np.float64), mi.astype(np.float32), ma.astype(np.float32), t_mi, t_ma, entry_prices, entry_idxs


# -------------------------
# low-level array entrypoint
# -------------------------
def backtest_from_arrays(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    signal_idx_arr: np.ndarray,
    side_arr: np.ndarray,
    sl: float = 0.0,
    tp: float = 0.0,
    sl_tp_in_pct: bool = True,
    limit_order_expiry_h: Optional[int] = None,
    exit_window_h: Optional[int] = None,
    trade_window_interval: Optional[int] = 0,
    base_minutes: int = 5,
    spread: float = 0.0,
    conservative_sl_first: bool = True,
    use_limit_entry: bool = True,
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

    limit_window_bars = _hours_to_bars(limit_order_expiry_h, base_minutes)
    exit_window_bars = _hours_to_bars(exit_window_h, base_minutes)
    trade_window_bars = max(0, _hours_to_bars(trade_window_interval, base_minutes))

    signal_idx, entry_idx, entry_price, exit_idx, exit_price, rets, exit_reason = _backtest_kernel_limit_ohlc(
        close,
        high,
        low,
        signal_idx_arr,
        side_arr,
        float(sl),
        float(tp),
        bool(sl_tp_in_pct),
        int(limit_window_bars),
        int(exit_window_bars),
        float(spread),
        bool(conservative_sl_first),
        bool(use_limit_entry),
        bool(use_trailing_sl),
        float(trailing_sl_pct),
        int(trailing_sl_interval),
        bool(trailing_sl_stop_at_pos),
    )

    if trade_window_bars > 0 and signal_idx.size > 0:
        signal_idx, entry_idx, entry_price, exit_idx, exit_price, rets, exit_reason = _apply_trade_window_interval(
            signal_idx=signal_idx,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            entry_price=entry_price,
            exit_price=exit_price,
            rets=rets,
            exit_reason=exit_reason,
            trade_window_bars=trade_window_bars,
        )

    return {
        "signal_idx": signal_idx,
        "entry_idx": entry_idx,
        "entry_price": entry_price,
        "exit_idx": exit_idx,
        "exit_price": exit_price,
        "rets": rets,
        "exit_reason": exit_reason,
    }


# -------------------------
# public wrapper
# -------------------------
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
    limit_order_expiry_h: Optional[int] = None,
    trade_window_interval: Optional[int] = 0,
    base_minutes: int = 5,
    spread: float = 0.0,
    conservative_sl_first: bool = True,
    side_flag: int = 1,
    use_limit_entry: bool = True,
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
        }

    side_arr = np.full(sig_idxs.shape[0], side_flag, dtype=np.int8)

    return backtest_from_arrays(
        close=main_close_arr,
        high=main_high_arr,
        low=main_low_arr,
        signal_idx_arr=sig_idxs.astype(np.int64),
        side_arr=side_arr,
        sl=sl,
        tp=tp,
        sl_tp_in_pct=sl_tp_in_pct,
        limit_order_expiry_h=limit_order_expiry_h,
        exit_window_h=exit_window_h,
        trade_window_interval=trade_window_interval,
        base_minutes=base_minutes,
        spread=spread,
        conservative_sl_first=conservative_sl_first,
        use_limit_entry=use_limit_entry,
        use_trailing_sl=use_trailing_sl,
        trailing_sl_pct=trailing_sl_pct,
        trailing_sl_interval=trailing_sl_interval,
        trailing_sl_stop_at_pos=trailing_sl_stop_at_pos,
    )