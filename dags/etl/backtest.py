# etl/backtest.py
"""
Backtest numeric kernels focused on close-price only.

- Uses close series only for hit detection (no high/low).
- Memory-conscious: accept numpy arrays, use np.asarray for cheap view/cast.
- Numba kernels compiled with cache; the per-signal kernel avoids parfor compile pitfalls.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from numba import njit, prange

from etl.feature_helpers import get_ma_price_gaps_for_indices


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
    regime_cfg: dict,
) -> dict:
    return {
        "balance": float(balance),
        "SL": float(sl_val),
        "TP": float(tp_val),
        "win_pos": int(win_pos),
        "total_pos": int(total_pos),
        "side": int(side_flag),
        "exit_window_h": int(regime_cfg.get("exit_window_h", 0)),
        "era_int": int(era_int),
        "regime_id": int(regime_id),
        "ma_int": int(regime_cfg.get("ma_int", 0)),
        "ma_reversion": bool(regime_cfg.get("ma_reversion", False)),
        "entry_lookback_units": int(regime_cfg.get("entry_lookback_units", 0)),
        "use_bbw": bool(regime_cfg.get("use_bbw", False)),
        "bbw_periods": int(regime_cfg.get("bbw_periods", 0)),
        "bbw_std": float(regime_cfg.get("bbw_std", 0.0)),
        "bbw_thresholds": int(regime_cfg.get("bbw_thresholds", 0)),
        "use_stochastic": bool(regime_cfg.get("use_stochastic", False)),
        "stoch_key": str(regime_cfg.get("stoch_key", "OFF")),
        "use_sl_decay": bool(regime_cfg.get("use_sl_decay", False)),
        "sl_decay_pct": float(regime_cfg.get("sl_decay_pct", 0.0)),
        "sl_decay_interval": int(regime_cfg.get("sl_decay_interval", 0)),
        "sl_decay_stop_at_pos": bool(regime_cfg.get("sl_decay_stop_at_pos", True)),
        "max_drawdown": float(max_dd),
    }

# -------------------------
# small helpers
# -------------------------
def _funding_per_hour_from_series(funding_arr: np.ndarray, funding_period_hours: int, funding_rate_unit: str = "per_period") -> np.ndarray:
    funding = np.asarray(funding_arr, dtype=np.float64)
    if funding_rate_unit == "per_hour":
        return funding
    return funding / float(max(1, int(funding_period_hours)))

# -------------------------
# fast equity (numba)
# -------------------------
@njit(cache=True)
def fast_compound_equity(pnl_arr, start_equity=100.0):
    n = pnl_arr.shape[0]
    equity = np.empty(n, dtype=np.float64)
    cur = start_equity
    running_max = cur
    max_dd = 0.0
    
    for i in range(n):
        pnl = pnl_arr[i]
        # Safety check for numerical explosions (common in BTC high-leverage)
        if not (pnl > -2.0 and pnl < 2.0): 
            pnl = -1.0 
             
        cur = cur * (1.0 + pnl)

        if cur > 1e37:
            cur = 1e37
        
        # Floor logic to prevent equity from "disappearing" or becoming negative
        if cur < 0.0001 or not np.isfinite(cur):
            cur = 0.0001
            
        equity[i] = cur
        
        if cur > running_max:
            running_max = cur
        
        # DIVIDE-BY-ZERO GUARD:
        # If running_max is the floor value, DD is effectively 0 or 1.
        if running_max > 0.0001:
            dd = (running_max - cur) / running_max
        else:
            dd = 0.0
            
        if dd > max_dd:
            max_dd = dd
            
    return equity, max_dd
@njit(cache=True)
def fast_compound_equity_gate(pnl_arr, start_equity=100.0, max_dd_threshold=-1.0):
    n = pnl_arr.shape[0]
    equity = np.empty(n, dtype=np.float64)
    cur = start_equity
    running_max = cur
    max_dd = 0.0

    for i in range(n):
        pnl = pnl_arr[i]

        if not (pnl > -2.0 and pnl < 2.0):
            pnl = -1.0

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
# pnl kernel (numba) - parallel for batch PnL conversion
# -------------------------
@njit(parallel=False, cache=True)
def _numba_pnl_kernel_optimized(
    rets, 
    entry_prices, 
    spread_arr, 
    funding_raw,           # Change: Pass raw funding
    entry_idxs,            # Change: Pass idxs to calc hours inside
    exit_idxs,             # Change: Pass idxs to calc hours inside
    main_time_ns,          # Change: Pass full time array
    risk_pct, 
    sl_val, 
    tp_val, 
    sl_tp_in_pct,
    spread_is_percent,
    funding_period_hours
):
    n = rets.shape[0]
    pnl_out = np.empty(n, dtype=np.float64)
    risk = float(risk_pct)
    
    # Pre-calculate constant
    ns_to_hours = 1.0 / (1e9 * 3600.0)
    f_period = float(max(1, funding_period_hours))

    for i in range(n):
        r = rets[i]
        ep = entry_prices[i]
        ep_safe = ep if ep != 0 else 1.0
        
        # Calculate holding hours inside the loop (zero allocation)
        h_hours = (main_time_ns[exit_idxs[i]] - main_time_ns[entry_idxs[i]]) * ns_to_hours
        
        # Calculate funding fee
        f_fee = (funding_raw[i] / f_period) * h_hours
        
        spread = spread_arr[i]
        if spread_is_percent:
            spr_pct = spread
        else:
            spr_pct = (spread / ep_safe) * 100.0

        raw_sl = float(sl_val) if sl_tp_in_pct else (float(sl_val) / ep_safe) * 100.0
        raw_tp = float(tp_val) if sl_tp_in_pct else (float(tp_val) / ep_safe) * 100.0

        eff_sl = max(raw_sl + spr_pct, 0.01)
        eff_tp = max(raw_tp - spr_pct, 0.0)

        pos_size = min(risk / (eff_sl / 100.0), 100.0)
        
        if r > 0:
            val = (pos_size * (eff_tp / 100.0)) - f_fee
        else:
            val = -risk - f_fee
            
        # CLIP: prevent individual PnL entries from being junk
        pnl_out[i] = max(min(val, 1e10), -1.0)
            
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
    spread_is_percent: bool = True
) -> Tuple[np.ndarray, np.ndarray]:

    n = closed_masked_rets.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    # SCHEMA MATCH: Force float64 promotion for funding and price math
    # even if input_arr was downcast to float32 in worker prep.
    entry_prices = (entry_prices_arr if entry_prices_arr is not None 
                    else main_close_arr[closed_entry_idxs]).astype(np.float64)
    
    spreads = (spread_arr[closed_entry_idxs] if spread_arr is not None 
               else np.zeros(n, dtype=np.float64)).astype(np.float64)
    
    # Ensure tiny funding rates are treated as float64
    funding_raw = (funding_arr[closed_entry_idxs] if funding_arr is not None 
                   else np.zeros(n, dtype=np.float64)).astype(np.float64)
    
    # Single Numba call - all heavy lifting moved inside
    pnl_pct = _numba_pnl_kernel_optimized(
        closed_masked_rets.astype(np.float64), # Precision promotion, 
        entry_prices, 
        spreads, 
        funding_raw,
        closed_entry_idxs,
        closed_exit_idxs,
        main_time_ns_arr,
        float(risk_pct), 
        float(sl_val), 
        float(tp_val), 
        bool(sl_tp_in_pct), 
        bool(spread_is_percent),
        funding_period_hours
    )

    exit_times_ns = main_time_ns_arr[closed_exit_idxs]
    return pnl_pct, exit_times_ns

# -------------------------
# backtest kernel (numba) - CLOSE-only per-signal (no parallel transform)
# -------------------------
# etl/backtest.py
@njit(cache=True)
def _backtest_kernel_close_only(
    close_f64,
    entry_idx_i64, entry_price_f64, side_i8,
    sl_f64, tp_f64, sl_tp_in_pct_bool,
    window_bars_i64, spread_f64,
    conservative_sl_first_bool, treat_no_hit_as_loss_bool,
    use_sl_decay_bool, sl_decay_pct_f64, sl_decay_interval_i64, sl_decay_stop_at_pos_bool
):
    n_rows = close_f64.shape[0]
    n_signals = entry_idx_i64.shape[0]
    rets = np.zeros(n_signals, dtype=np.float64)
    exit_idx = -1 * np.ones(n_signals, dtype=np.int64)

    for i in range(n_signals):
        entry_idx = int(entry_idx_i64[i])
        if entry_idx < 0 or entry_idx >= n_rows:
            rets[i] = 0.0
            exit_idx[i] = -1
            continue

        entry_price = float(entry_price_f64[i])
        side = int(side_i8[i])

        # base SL / TP
        if sl_tp_in_pct_bool:
            if side == 1:
                tp_price = entry_price * (1.0 + tp_f64 / 100.0)
                sl_price = entry_price * (1.0 - sl_f64 / 100.0)
            else:
                tp_price = entry_price * (1.0 - tp_f64 / 100.0)
                sl_price = entry_price * (1.0 + sl_f64 / 100.0)
        else:
            if side == 1:
                tp_price = entry_price + tp_f64
                sl_price = entry_price - sl_f64
            else:
                tp_price = entry_price - tp_f64
                sl_price = entry_price + sl_f64

        # spread adjustment
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

        base_sl_dist = abs(sl_price_adj - entry_price)

        exit_at = -1
        ret = 0.0

        max_j = n_rows - 1
        if window_bars_i64 >= 0:
            tmp = entry_idx + int(window_bars_i64)
            if tmp < max_j:
                max_j = tmp

        j = entry_idx + 1

        if conservative_sl_first_bool:
            if side == 1:
                while j <= max_j:
                    c = close_f64[j]

                    current_sl_price = sl_price_adj
                    if use_sl_decay_bool and sl_decay_interval_i64 > 0 and sl_decay_pct_f64 > 0.0:
                        elapsed = j - entry_idx
                        if elapsed >= sl_decay_interval_i64:
                            steps = elapsed // sl_decay_interval_i64
                            move = base_sl_dist * sl_decay_pct_f64 * float(steps)
                            current_sl_price = sl_price_adj + move
                            if sl_decay_stop_at_pos_bool and current_sl_price > entry_price:
                                current_sl_price = entry_price

                    if c <= current_sl_price:
                        exit_at = j
                        ret = (current_sl_price - entry_price) / entry_price
                        break
                    if c >= tp_price_adj:
                        exit_at = j
                        ret = (tp_price_adj - entry_price) / entry_price
                        break
                    j += 1
            else:
                while j <= max_j:
                    c = close_f64[j]

                    current_sl_price = sl_price_adj
                    if use_sl_decay_bool and sl_decay_interval_i64 > 0 and sl_decay_pct_f64 > 0.0:
                        elapsed = j - entry_idx
                        if elapsed >= sl_decay_interval_i64:
                            steps = elapsed // sl_decay_interval_i64
                            move = base_sl_dist * sl_decay_pct_f64 * float(steps)
                            current_sl_price = sl_price_adj - move
                            if sl_decay_stop_at_pos_bool and current_sl_price < entry_price:
                                current_sl_price = entry_price

                    if c >= current_sl_price:
                        exit_at = j
                        ret = (entry_price - current_sl_price) / entry_price
                        break
                    if c <= tp_price_adj:
                        exit_at = j
                        ret = (entry_price - tp_price_adj) / entry_price
                        break
                    j += 1
        else:
            if side == 1:
                while j <= max_j:
                    c = close_f64[j]

                    current_sl_price = sl_price_adj
                    if use_sl_decay_bool and sl_decay_interval_i64 > 0 and sl_decay_pct_f64 > 0.0:
                        elapsed = j - entry_idx
                        if elapsed >= sl_decay_interval_i64:
                            steps = elapsed // sl_decay_interval_i64
                            move = base_sl_dist * sl_decay_pct_f64 * float(steps)
                            current_sl_price = sl_price_adj + move
                            if sl_decay_stop_at_pos_bool and current_sl_price > entry_price:
                                current_sl_price = entry_price

                    if c >= tp_price_adj:
                        exit_at = j
                        ret = (tp_price_adj - entry_price) / entry_price
                        break
                    if c <= current_sl_price:
                        exit_at = j
                        ret = (current_sl_price - entry_price) / entry_price
                        break
                    j += 1
            else:
                while j <= max_j:
                    c = close_f64[j]

                    current_sl_price = sl_price_adj
                    if use_sl_decay_bool and sl_decay_interval_i64 > 0 and sl_decay_pct_f64 > 0.0:
                        elapsed = j - entry_idx
                        if elapsed >= sl_decay_interval_i64:
                            steps = elapsed // sl_decay_interval_i64
                            move = base_sl_dist * sl_decay_pct_f64 * float(steps)
                            current_sl_price = sl_price_adj - move
                            if sl_decay_stop_at_pos_bool and current_sl_price < entry_price:
                                current_sl_price = entry_price

                    if c <= tp_price_adj:
                        exit_at = j
                        ret = (entry_price - tp_price_adj) / entry_price
                        break
                    if c >= current_sl_price:
                        exit_at = j
                        ret = (entry_price - current_sl_price) / entry_price
                        break
                    j += 1

        if exit_at == -1:
            last_idx = min(entry_idx + int(window_bars_i64), n_rows - 1)
            if last_idx > entry_idx:
                c_last = close_f64[last_idx]
                if side == 1:
                    ret = (c_last - entry_price) / entry_price
                else:
                    ret = (entry_price - c_last) / entry_price
                exit_at = last_idx
            else:
                ret = 0.0
                exit_at = entry_idx

        rets[i] = ret
        exit_idx[i] = exit_at

    return rets, exit_idx

# -------------------------
# grid kernel (numba) - pre-check tp<sl*min_rr outside inner row loop
# (kept parallel=True for heavy aggregate checks)
# -------------------------
@njit(parallel=False, cache=True)
def _grid_kernel_numba(closes_f32, n_min_f32, n_max_f32,
                        t_min_ns_i64, t_max_ns_i64,
                        s_vals_f32, t_vals_f32,
                        breakeven_after_f32,
                        min_rr_f32,
                        tie_mode_i32, position_type_i32):
    n_rows = closes_f32.shape[0]
    ns = s_vals_f32.shape[0]
    nt = t_vals_f32.shape[0]
    good = np.zeros(ns * nt, dtype=np.int32)
    be = np.zeros(ns * nt, dtype=np.int32)

    for i in range(ns):
        sl_val = s_vals_f32[i]
        
        # GUARD: Prevent zero SL from creating infinite RR or math errors
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
                # Skip invalid windows
                if t_min_ns_i64[r] < 0:
                    continue
                
                # Current Close Price
                cp = closes_f32[r]
                
                if position_type_i32 == 1: # Long
                    t_tp = t_max_ns_i64[r]
                    t_sl = t_min_ns_i64[r]
                    dist_tp = n_max_f32[r] - cp
                    dist_sl = cp - n_min_f32[r]
                else: # Short
                    t_tp = t_min_ns_i64[r]
                    t_sl = t_max_ns_i64[r]
                    dist_tp = cp - n_min_f32[r]
                    dist_sl = n_max_f32[r] - cp

                hit_tp = dist_tp >= tp_val
                hit_sl = dist_sl >= sl_val
                
                tp_wins = False
                if hit_tp and hit_sl:
                    if tie_mode_i32 == 1: # Conservative: SL wins on tie
                        tp_wins = (t_tp <= t_sl)
                    else: # Aggressive: TP wins on tie
                        tp_wins = (t_tp < t_sl)
                elif hit_tp:
                    tp_wins = True
                
                if tp_wins:
                    count_good += 1
                elif dist_tp >= (tp_val - breakeven_after_f32):
                    count_be += 1
            
            # Store results using a flat index
            idx = i * nt + j
            good[idx] = count_good
            be[idx] = count_be
            
    return good, be

# -------------------------
# precompute windows: Numba for close-only windows
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

    Returns:
        (closes_f64, n_min_f32, n_max_f32, t_min_ns_i64, t_max_ns_i64, entry_prices_f64, entry_idxs_i64)
    Notes:
      - df_main and side_df should be Polars DataFrames; use zero-copy `to_numpy(zero_copy_only=True)` upstream to avoid copies.
      - This function does minimal conversion to numpy arrays.
    """
    main_close = df_main['close'].to_numpy().astype(np.float64)
    main_time_ns = df_main['time_ns'].to_numpy().astype(np.int64)
    sig_time_ns = side_df['time_ns'].to_numpy().astype(np.int64)
    entry_prices = side_df['close'].to_numpy().astype(np.float64)

    steps = max(1, int((exit_window_h * 60) / base_minutes))

    entry_idxs = np.searchsorted(main_time_ns, sig_time_ns, side='left')
    entry_idxs = np.clip(entry_idxs, 0, max(0, main_time_ns.shape[0] - 1)).astype(np.int64)

    cl, mi, ma, t_mi, t_ma = _compute_windows_numba(main_close, main_time_ns, entry_idxs, np.int32(steps))
    return cl.astype(np.float64), mi.astype(np.float32), ma.astype(np.float32), t_mi, t_ma, entry_prices, entry_idxs

# -------------------------
# low-level array entrypoint - numeric (close-only)
# -------------------------
def backtest_from_arrays(
    close: np.ndarray,
    entry_idx_arr: np.ndarray,
    entry_price_arr: np.ndarray,
    side_arr: np.ndarray,
    sl: float = 0.0,
    tp: float = 0.0,
    sl_tp_in_pct: bool = True,
    window_bars: int = -1,
    spread: float = 0.0,
    conservative_sl_first: bool = True,
    treat_no_hit_as_loss: bool = True,
    use_sl_decay: bool = False,
    sl_decay_pct: float = 0.0,
    sl_decay_interval: int = 0,
    sl_decay_stop_at_pos: bool = True,
) -> Dict:
    close = np.asarray(close, dtype=np.float64)
    entry_idx_arr = np.asarray(entry_idx_arr, dtype=np.int64)
    entry_price_arr = np.asarray(entry_price_arr, dtype=np.float64)
    side_arr = np.asarray(side_arr, dtype=np.int8)

    rets, exit_idx = _backtest_kernel_close_only(
        close,
        entry_idx_arr, entry_price_arr, side_arr,
        float(sl), float(tp), bool(sl_tp_in_pct),
        int(window_bars), float(spread),
        bool(conservative_sl_first), bool(treat_no_hit_as_loss),
        bool(use_sl_decay), float(sl_decay_pct), int(sl_decay_interval), bool(sl_decay_stop_at_pos),
    )

    return {"rets": rets, "exit_idx": exit_idx, "entry_idx": entry_idx_arr}

# -------------------------
# public wrapper (compatible signature)
# -------------------------
def backtest_signals_sl_tp_rets(
    main_close_arr: np.ndarray,
    main_time_ns_arr: np.ndarray,
    sig_idxs: np.ndarray,
    sl: float = 0.0,
    tp: float = 0.0,
    sl_tp_in_pct: bool = True,
    exit_window_h: Optional[int] = None,
    base_minutes: int = 5,
    spread: float = 0.0,
    conservative_sl_first: bool = True,
    treat_no_hit_as_loss: bool = True,
    side_flag: int = 1,
    use_sl_decay: bool = False,
    sl_decay_pct: float = 0.0,
    sl_decay_interval: int = 0,
    sl_decay_stop_at_pos: bool = True,
) -> Dict:
    if sig_idxs is None or sig_idxs.size == 0:
        return {
            "rets": np.array([], dtype=np.float64),
            "exit_idx": np.array([], dtype=np.int64),
            "entry_idx": np.array([], dtype=np.int64),
        }

    entry_price_arr = np.asarray(main_close_arr)[sig_idxs].astype(np.float64)
    side_arr = np.full(sig_idxs.shape[0], side_flag, dtype=np.int8)

    if exit_window_h is None or exit_window_h <= 0:
        window_bars = -1
    else:
        window_bars = int((exit_window_h * 60) / base_minutes)

    return backtest_from_arrays(
        close=main_close_arr,
        entry_idx_arr=sig_idxs.astype(np.int64),
        entry_price_arr=entry_price_arr,
        side_arr=side_arr,
        sl=sl,
        tp=tp,
        sl_tp_in_pct=sl_tp_in_pct,
        window_bars=window_bars,
        spread=spread,
        conservative_sl_first=conservative_sl_first,
        treat_no_hit_as_loss=treat_no_hit_as_loss,
        use_sl_decay=use_sl_decay,
        sl_decay_pct=sl_decay_pct,
        sl_decay_interval=sl_decay_interval,
        sl_decay_stop_at_pos=sl_decay_stop_at_pos,
    )

# -------------------------
# helper: warmup - compile numba functions ahead of time
# -------------------------
def warmup_numba_kernels():
    """
    Run kernels with tiny arrays to trigger numba compilation in advance.
    Call once before spawning worker processes.
    """
    n = 32
    close = np.ones(n, dtype=np.float64)
    entry_idxs = np.zeros(4, dtype=np.int64)
    entry_prices = np.ones(4, dtype=np.float64)
    sides = np.ones(4, dtype=np.int8)

    _ = _backtest_kernel_close_only(
        close,
        entry_idxs,
        entry_prices,
        sides,
        1.0,    # sl_f64
        2.0,    # tp_f64
        True,   # sl_tp_in_pct_bool
        10,     # window_bars_i64
        0.0,    # spread_f64
        True,   # conservative_sl_first_bool
        True,   # treat_no_hit_as_loss_bool
        False,  # use_sl_decay_bool
        0.0,    # sl_decay_pct_f64
        0,      # sl_decay_interval_i64
        True,   # sl_decay_stop_at_pos_bool
    )

    _ = _compute_windows_numba(close, np.arange(n, dtype=np.int64), entry_idxs, np.int32(4))

    _ = _grid_kernel_numba(
        close.astype(np.float32),
        close.astype(np.float32),
        close.astype(np.float32),
        np.arange(n, dtype=np.int64),
        np.arange(n, dtype=np.int64),
        np.array([1.0], dtype=np.float32),
        np.array([2.0], dtype=np.float32),
        np.float32(0.0),
        np.float32(1.0),
        np.int32(0),
        np.int32(1),
    )
