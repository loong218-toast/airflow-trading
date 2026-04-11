# research/grid_row_builders.py

import polars as pl
import numpy as np
from typing import Dict


def _make_empty_master_row(
    regime_id: int,
    era_int: int,
    side_flag: int,
    sl_val: float,
    tp_val: float,
    regime_cfg: dict,
    sl_hit: float = np.nan,
    tp_hit: float = np.nan,
) -> dict:
    return _make_master_row(
        regime_id=regime_id,
        era_int=era_int,
        side_flag=side_flag,
        sl_val=sl_val,
        tp_val=tp_val,
        total_pos=0,
        win_pos=0,
        balance=100.0,
        max_dd=0.0,
        max_consecutive_losses=0,
        regime_cfg=regime_cfg,
        sl_hit=sl_hit,
        tp_hit=tp_hit,
    )
    
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
    sl_hit: float = np.nan,
    tp_hit: float = np.nan,
) -> dict:
    return {
        "balance": float(balance),
        "SL": float(sl_val),
        "TP": float(tp_val),
        "SL_hit": float(sl_hit),
        "TP_hit": float(tp_hit),
        "win_pos": int(win_pos),
        "total_pos": int(total_pos),
        "side": int(side_flag),
        "exit_window_h": int(regime_cfg.get("exit_window_h", 0)),
        "limit_order_expiry_h": int(regime_cfg.get("limit_order_expiry_h", 0)),
        "use_limit_entry": bool(regime_cfg.get("use_limit_entry", False)),
        "trade_window_interval": int(regime_cfg.get("trade_window_interval", 0)),
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
        "use_trailing_sl": bool(regime_cfg.get("use_trailing_sl", False)),
        "trailing_sl_pct": float(regime_cfg.get("trailing_sl_pct", 0.0)),
        "trailing_sl_interval": int(regime_cfg.get("trailing_sl_interval", 0)),
        "trailing_sl_stop_at_pos": bool(regime_cfg.get("trailing_sl_stop_at_pos", True)),
        "max_consecutive_losses": int(max_consecutive_losses),
        "max_drawdown": float(max_dd),
    }

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
    limit_order_expiry_h: int,
    trade_window_interval: int,
    regime_id: int,
    era_int: int,
    backtest_res: Dict,
    regime_cfg: dict,
    run_cfg: dict,
) -> pl.DataFrame:
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
                "limit_order_expiry_h": pl.Series([], dtype=pl.Int32),
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

                "rng_24h_entry": pl.Series([], dtype=pl.Float32),
                "rng_72h_entry": pl.Series([], dtype=pl.Float32),
                "rng_1w_entry": pl.Series([], dtype=pl.Float32),
                "rng_1m_entry": pl.Series([], dtype=pl.Float32),
            }
        )

    signal_time_ns = np.asarray(main_time_ns_arr[sig_idxs], dtype=np.int64)
    signal_price = np.asarray(main_close_arr[sig_idxs], dtype=np.float64)

    order_mode = np.full(n, 1 if use_limit_entry else 0, dtype=np.int8)

    if use_limit_entry:
        if int(side_flag) == 1:
            order_price = 0.5 * (
                signal_price + np.asarray(main_low_arr[sig_idxs], dtype=np.float64)
            )
        else:
            order_price = 0.5 * (
                signal_price + np.asarray(main_high_arr[sig_idxs], dtype=np.float64)
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

    tp_hit_price = np.full(n, np.nan, dtype=np.float64)
    sl_hit_price = np.full(n, np.nan, dtype=np.float64)

    filled_tp_hit = np.asarray(backtest_res.get("TP_hit", []), dtype=np.float64)
    filled_sl_hit = np.asarray(backtest_res.get("SL_hit", []), dtype=np.float64)

    filled_signal_idx = np.asarray(backtest_res.get("signal_idx", []), dtype=np.int64)
    filled_entry_idx = np.asarray(backtest_res.get("entry_idx", []), dtype=np.int64)
    filled_entry_price = np.asarray(backtest_res.get("entry_price", []), dtype=np.float64)
    filled_exit_idx = np.asarray(backtest_res.get("exit_idx", []), dtype=np.int64)
    filled_exit_price = np.asarray(backtest_res.get("exit_price", []), dtype=np.float64)
    filled_rets = np.asarray(backtest_res.get("rets", []), dtype=np.float64)
    filled_exit_reason = np.asarray(backtest_res.get("exit_reason", []), dtype=np.int8)

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

        if 0 <= entry_idx[i] < main_time_ns_arr.shape[0]:
            entry_time_ns[i] = int(main_time_ns_arr[entry_idx[i]])

        if 0 <= exit_idx[i] < main_time_ns_arr.shape[0]:
            exit_time_ns[i] = int(main_time_ns_arr[exit_idx[i]])

        if entry_idx[i] >= 0:
            fill_delay_bars[i] = int(entry_idx[i] - sidx)

        if j < filled_tp_hit.shape[0]:
            tp_hit_price[i] = float(filled_tp_hit[j])
        if j < filled_sl_hit.shape[0]:
            sl_hit_price[i] = float(filled_sl_hit[j])

    feature_entry_idxs = np.where(fill_status == 1, entry_idx, sig_idxs).astype(np.int64)

    filled_mask = fill_status == 1

    df = pl.DataFrame(
        {
            "regime_id": np.full(n, int(regime_id), dtype=np.int32),
            "era_int": np.full(n, int(era_int), dtype=np.int64),
            "side": np.full(n, int(side_flag), dtype=np.int8),
            "SL": np.full(n, float(sl_val), dtype=np.float32),
            "TP": np.full(n, float(tp_val), dtype=np.float32),

            "use_limit_entry": np.full(n, bool(use_limit_entry), dtype=bool),
            "limit_order_expiry_h": np.full(n, int(limit_order_expiry_h), dtype=np.int32),
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

            "rng_24h_entry": np.asarray(rng_24h_entry, dtype=np.float32),
            "rng_72h_entry": np.asarray(rng_72h_entry, dtype=np.float32),
            "rng_1w_entry": np.asarray(rng_1w_entry, dtype=np.float32),
            "rng_1m_entry": np.asarray(rng_1m_entry, dtype=np.float32),
        }
    )

    return df