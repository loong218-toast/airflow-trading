from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl

from research.expectancy_config import (
    AFTER_LOSS_SKIP_TRADES_LIST,
    AFTER_LOSS_TRIGGER_COUNT_LIST,
    CACHE_FILE,
    ENTRY_WINDOW_END_HOUR_MYT,
    ENTRY_WINDOW_LABEL,
    ENTRY_WINDOW_START_HOUR_MYT,
    HORIZON_HOURS_LIST,
    INSTRUMENT,
    MA_PERIOD_BARS,
    MA_TYPE,
    MT5_SYMBOL,
    MYT,
    OUTPUT_DIR,
    PAIR,
    RANDOM_ENTRY_SEED,
    RISK_PCT,
    SEQUENTIAL_SWITCH_ON_LOSS,
    SIMULATION_MODES,
    TRADES_PER_HOUR,
    SL_PCT_LIST,
    TARGET_PCT_LIST,
    USE_AFTER_LOSS_FILTER,
    USE_ENTRY_TIME_WINDOW,
    USE_MA_FILTER,
    _market_tag,
    _net_expectancy_risk_pct,
    _safe_name,
    get_columns_to_remove,
    sample_entry_price_nudged,
)

logger = logging.getLogger(__name__)

MINUTE_NS = 60 * 1_000_000_000
HOUR_NS = 60 * MINUTE_NS

PLOT_BINS = 8
ENTRY_RESET_TOL_PCT = 0.0
PLOT_MAX_R = 2.0
PLOT_BIN_EDGES = np.linspace(0.0, PLOT_MAX_R, PLOT_BINS + 1)


def _import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except Exception:
        sns = None

    return plt, sns


def _safe_div(n: float, d: float) -> Optional[float]:
    if d == 0:
        return None
    return float(n / d)


def _replace_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def _save_figure(fig, out: Path) -> Path:
    _replace_file(out)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    return out


def clear_previous_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "expectancy_scan_*.csv",
        "expectancy_heatmap_*.png",
        "mae_distribution_*.png",
        "mfe_distribution_*.png",
        "mae_win_distribution_*.png",
        "mae_loss_distribution_*.png",
        "mfe_win_distribution_*.png",
        "mfe_loss_distribution_*.png",
    ):
        for p in OUTPUT_DIR.glob(pattern):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass


def _weighted_percentile_from_hist(hist: np.ndarray, percentile: float) -> Optional[float]:
    total = int(hist.sum())
    if total <= 0:
        return None

    target = total * (percentile / 100.0)
    cum = np.cumsum(hist)
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = max(0, min(idx, hist.size - 1))
    return float((PLOT_BIN_EDGES[idx] + PLOT_BIN_EDGES[idx + 1]) / 2.0)


def _bin_update(hist: np.ndarray, value: float) -> None:
    if not np.isfinite(value):
        return
    idx = int(np.searchsorted(PLOT_BIN_EDGES, value, side="right") - 1)
    if 0 <= idx < hist.size:
        hist[idx] += 1


def _utc_to_myt_hours(time_ns: np.ndarray) -> np.ndarray:
    dt_utc = pd.to_datetime(time_ns, unit="ns", utc=True)
    dt_myt = dt_utc.tz_convert(MYT)
    return dt_myt.hour.to_numpy(dtype=np.int8, copy=False)


def _entry_window_mask_fast(
    hours: np.ndarray,
    use_entry_time_window: bool,
    start_hour: int,
    end_hour: int,
) -> np.ndarray:
    if not use_entry_time_window or start_hour == end_hour:
        return np.ones(len(hours), dtype=bool)

    if start_hour < end_hour:
        return (hours >= start_hour) & (hours < end_hour)
    return (hours >= start_hour) | (hours < end_hour)


def _build_allowed_entry_indices(
    time_ns: np.ndarray,
    use_entry_time_window: bool = USE_ENTRY_TIME_WINDOW,
    start_hour: int = ENTRY_WINDOW_START_HOUR_MYT,
    end_hour: int = ENTRY_WINDOW_END_HOUR_MYT,
    trades_per_hour: Optional[int] = TRADES_PER_HOUR,
) -> set[int]:
    if time_ns.size == 0:
        return set()

    myt_hours = _utc_to_myt_hours(time_ns)
    entry_mask = _entry_window_mask_fast(
        myt_hours,
        use_entry_time_window=use_entry_time_window,
        start_hour=start_hour,
        end_hour=end_hour,
    )

    eligible_idx = np.flatnonzero(entry_mask)
    if eligible_idx.size == 0:
        return set()

    if trades_per_hour is None or int(trades_per_hour) <= 0:
        return set(int(i) for i in eligible_idx.tolist())

    dt_utc = pd.to_datetime(time_ns[eligible_idx], unit="ns", utc=True)
    df_tmp = pd.DataFrame({"idx": eligible_idx, "hour_group": dt_utc.floor("h")})

    rng = np.random.default_rng(RANDOM_ENTRY_SEED)
    selected: set[int] = set()

    for _, grp in df_tmp.groupby("hour_group", sort=True):
        idxs = grp["idx"].to_numpy(dtype=int, copy=False)
        if idxs.size <= int(trades_per_hour):
            selected.update(idxs.tolist())
        else:
            chosen = rng.choice(idxs, size=int(trades_per_hour), replace=False)
            selected.update(chosen.tolist())

    return selected


def _new_stats() -> Dict[str, Any]:
    return {
        "anchors_total": 0,
        "trades_taken_count": 0,
        "skipped_after_loss_count": 0,
        "loss_streak_trigger_count": 0,
        "target_first_count": 0,
        "sl_first_count": 0,
        "censored_count": 0,
        "target_first_then_sl_count": 0,
        "target_minutes_sum": 0.0,
        "target_minutes_count": 0,
        "sl_minutes_sum": 0.0,
        "sl_minutes_count": 0,
        "first_event_minutes_sum": 0.0,
        "first_event_minutes_count": 0,
        "forced_exit_r_sum": 0.0,
        "forced_exit_r_count": 0,
        "forced_exit_r_positive_count": 0,
        "entry_revisit_mae_sum": 0.0,
        "entry_revisit_mae_count": 0,
        "entry_revisit_mfe_sum": 0.0,
        "entry_revisit_mfe_count": 0,
        "entry_revisit_mae_win_sum": 0.0,
        "entry_revisit_mae_win_count": 0,
        "entry_revisit_mae_loss_sum": 0.0,
        "entry_revisit_mae_loss_count": 0,
        "entry_revisit_mfe_win_sum": 0.0,
        "entry_revisit_mfe_win_count": 0,
        "entry_revisit_mfe_loss_sum": 0.0,
        "entry_revisit_mfe_loss_count": 0,
        "entry_revisit_mae_hist": np.zeros(PLOT_BINS, dtype=np.int64),
        "entry_revisit_mfe_hist": np.zeros(PLOT_BINS, dtype=np.int64),
        "entry_revisit_mae_win_hist": np.zeros(PLOT_BINS, dtype=np.int64),
        "entry_revisit_mae_loss_hist": np.zeros(PLOT_BINS, dtype=np.int64),
        "entry_revisit_mfe_win_hist": np.zeros(PLOT_BINS, dtype=np.int64),
        "entry_revisit_mfe_loss_hist": np.zeros(PLOT_BINS, dtype=np.int64),
    }


def _compute_ma_values(close: np.ndarray) -> np.ndarray:
    s = pd.Series(close, dtype="float64")
    ma_type = MA_TYPE.lower().strip()
    if ma_type == "ema":
        ma = s.ewm(span=MA_PERIOD_BARS, adjust=False, min_periods=MA_PERIOD_BARS).mean()
    elif ma_type == "sma":
        ma = s.rolling(window=MA_PERIOD_BARS, min_periods=MA_PERIOD_BARS).mean()
    else:
        raise ValueError("MA_TYPE must be 'ema' or 'sma'")
    return ma.shift(1).to_numpy(dtype=np.float64, copy=False)


def _normalize_after_loss_trigger_count_list(values: Optional[Sequence[int]]) -> List[int]:
    if values is None:
        values = AFTER_LOSS_TRIGGER_COUNT_LIST

    out: List[int] = []
    seen = set()
    for value in values:
        v = int(value)
        if v <= 0:
            raise ValueError("AFTER_LOSS_TRIGGER_COUNT_LIST values must be > 0")
        if v in seen:
            continue
        seen.add(v)
        out.append(v)

    return out or [int(AFTER_LOSS_TRIGGER_COUNT_LIST[0])]


def _normalize_after_loss_skip_trades_list(values: Optional[Sequence[Optional[int]]]) -> List[Optional[int]]:
    if values is None:
        values = AFTER_LOSS_SKIP_TRADES_LIST

    out: List[Optional[int]] = []
    seen = set()
    for value in values:
        v = None if value is None else int(value)
        if v is not None and v < 0:
            raise ValueError("AFTER_LOSS_SKIP_TRADES_LIST values must be >= 0 or None")
        key = "__none__" if v is None else v
        if key in seen:
            continue
        seen.add(key)
        out.append(v)

    return out or [AFTER_LOSS_SKIP_TRADES_LIST[0]]


def _ma_pass(side: int, price: float, ma_value: float) -> bool:
    if not np.isfinite(ma_value):
        return True
    return price < ma_value if side == 1 else price > ma_value


def _path_entry_revisit_excursion_stats(
    side: int,
    entry_price: float,
    highs: np.ndarray,
    lows: np.ndarray,
    tp_ratio: float,
    sl_ratio: float,
    reset_tol_pct: float = ENTRY_RESET_TOL_PCT,
    exclude_last_bar: bool = True,
) -> Dict[str, float]:
    nan_out = {"entry_revisit_mae_r": np.nan, "entry_revisit_mfe_r": np.nan, "had_reset": False}

    if not np.isfinite(entry_price) or entry_price <= 0.0:
        return nan_out
    if highs.size < 2 or lows.size < 2:
        return nan_out

    if exclude_last_bar and highs.size > 1 and lows.size > 1:
        highs = highs[:-1]
        lows = lows[:-1]

    if highs.size == 0 or lows.size == 0:
        return nan_out

    eligible_mae_raw: List[float] = []
    eligible_mfe_raw: List[float] = []

    for i in range(len(highs)):
        if side == 1:
            reset_level = entry_price * (1.0 - reset_tol_pct)
            has_reset = bool(np.any(highs[i:] >= reset_level))
            current_mae_raw = entry_price - float(lows[i])
            current_mfe_raw = float(highs[i]) - entry_price
        else:
            reset_level = entry_price * (1.0 + reset_tol_pct)
            has_reset = bool(np.any(lows[i:] <= reset_level))
            current_mae_raw = float(highs[i]) - entry_price
            current_mfe_raw = entry_price - float(lows[i])

        if has_reset:
            eligible_mae_raw.append(max(0.0, current_mae_raw))
            eligible_mfe_raw.append(max(0.0, current_mfe_raw))

    if not eligible_mae_raw:
        return nan_out

    entry_revisit_mae_r = float((max(eligible_mae_raw) / entry_price) / sl_ratio) if sl_ratio > 0 else np.nan
    entry_revisit_mfe_r = float((max(eligible_mfe_raw) / entry_price) / tp_ratio) if tp_ratio > 0 else np.nan

    return {
        "entry_revisit_mae_r": entry_revisit_mae_r,
        "entry_revisit_mfe_r": entry_revisit_mfe_r,
        "had_reset": True,
    }


def load_df_from_cache(cache_file: Path = CACHE_FILE) -> pl.DataFrame:
    if not cache_file.exists():
        return pl.DataFrame()

    try:
        if cache_file.suffix.lower() in {".parquet", ".pq"}:
            df_pd = pd.read_parquet(cache_file)
        else:
            df_pd = pd.read_csv(cache_file)
    except Exception:
        return pl.DataFrame()

    if df_pd.empty:
        return pl.DataFrame()

    needed = ["time_ns", "open", "high", "low", "close"]
    if any(c not in df_pd.columns for c in needed):
        return pl.DataFrame()

    df_pd = df_pd[needed].copy()
    for col in needed:
        df_pd[col] = pd.to_numeric(df_pd[col], errors="coerce")

    df_pd = (
        df_pd.dropna(subset=needed)
        .drop_duplicates(subset=["time_ns"])
        .sort_values("time_ns")
        .reset_index(drop=True)
    )

    return (
        pl.from_pandas(df_pd)
        .with_columns(
            [
                pl.col("time_ns").cast(pl.Int64),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
            ]
        )
        .sort("time_ns")
    )


def _strategy_mode_list() -> List[tuple[str, bool, bool]]:
    out = [("all_trades", False, False)]
    if USE_MA_FILTER:
        out.append(("ma_filter", True, False))
    if USE_AFTER_LOSS_FILTER:
        out.append(("after_loss", False, True))
    if USE_MA_FILTER and USE_AFTER_LOSS_FILTER:
        out.append(("ma_after_loss", True, True))
    return out


def _group_keys_from_df(summary_df: pd.DataFrame) -> List[str]:
    keys: List[str] = ["filter_mode"]
    if "sim_mode" in summary_df.columns and summary_df["sim_mode"].nunique(dropna=True) > 1:
        keys.append("sim_mode")
    return keys


def _build_trade_candidates(
    time_ns: np.ndarray,
    open_px: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ma_values: np.ndarray,
    horizons: Sequence[int],
    target_ratio: float,
    sl_ratio: float,
    apply_ma_filter: bool,
    conservative_sl_first: bool,
) -> Dict[int, List[Dict[str, Any]]]:
    n = int(close.shape[0])
    candidates_by_side: Dict[int, List[Dict[str, Any]]] = {1: [], -1: []}

    horizon_end_by_h: Dict[int, np.ndarray] = {}
    for h in horizons:
        horizon_end_by_h[h] = np.searchsorted(time_ns, time_ns + np.int64(h * HOUR_NS), side="right")

    for anchor_idx in range(n - 1):
        anchor_price = sample_entry_price_nudged(
            open_px=float(open_px[anchor_idx]),
            high_px=float(high[anchor_idx]),
            low_px=float(low[anchor_idx]),
            close_px=float(close[anchor_idx]),
            anchor_idx=anchor_idx,
        )
        if not np.isfinite(anchor_price) or anchor_price <= 0.0:
            continue

        ma_value = float(ma_values[anchor_idx]) if anchor_idx < len(ma_values) else np.nan

        future_high_full = high[anchor_idx + 1 :]
        future_low_full = low[anchor_idx + 1 :]
        future_time_full = time_ns[anchor_idx + 1 :]

        for side in (1, -1):
            if apply_ma_filter and not _ma_pass(side, anchor_price, ma_value):
                continue

            for h in horizons:
                sidx = int(horizon_end_by_h[h][anchor_idx])
                if sidx <= anchor_idx + 1:
                    continue

                local_len = sidx - (anchor_idx + 1)
                if local_len <= 0:
                    continue

                future_high_local = future_high_full[:local_len]
                future_low_local = future_low_full[:local_len]
                future_time_local = future_time_full[:local_len]

                if side == 1:
                    target_price = anchor_price * (1.0 + target_ratio)
                    sl_price = anchor_price * (1.0 - sl_ratio)
                    target_hits = np.flatnonzero(future_high_local >= target_price)
                    sl_hits = np.flatnonzero(future_low_local <= sl_price)
                else:
                    target_price = anchor_price * (1.0 - target_ratio)
                    sl_price = anchor_price * (1.0 + sl_ratio)
                    target_hits = np.flatnonzero(future_low_local <= target_price)
                    sl_hits = np.flatnonzero(future_high_local >= sl_price)

                target_rel = int(target_hits[0]) if target_hits.size else -1
                sl_rel = int(sl_hits[0]) if sl_hits.size else -1
                target_in = target_rel != -1
                sl_in = sl_rel != -1

                if not target_in and not sl_in:
                    exit_idx = local_len - 1
                    exit_price = float(close[anchor_idx + 1 + exit_idx])
                    forced_pnl_pct = (
                        (exit_price - anchor_price) / anchor_price
                        if side == 1
                        else (anchor_price - exit_price) / anchor_price
                    )
                    forced_exit_r = forced_pnl_pct / sl_ratio

                    trade_result = "censored"
                    target_first = False
                    target_first_then_sl = False
                    target_min = None
                    sl_min = None
                    first_event_min = None
                    path_highs = future_high_local[: exit_idx + 1]
                    path_lows = future_low_local[: exit_idx + 1]
                else:
                    if target_in and sl_in:
                        if target_rel < sl_rel:
                            target_first = True
                        elif sl_rel < target_rel:
                            target_first = False
                        else:
                            target_first = not conservative_sl_first
                    elif target_in:
                        target_first = True
                    else:
                        target_first = False

                    trade_result = "target" if target_first else "sl"
                    target_first_then_sl = bool(target_first and sl_in and sl_rel > target_rel)

                    if target_first:
                        first_rel = target_rel
                        target_min = (int(future_time_local[target_rel]) - int(time_ns[anchor_idx])) / MINUTE_NS
                        sl_min = None
                        first_event_min = float(target_min)
                    else:
                        first_rel = sl_rel
                        target_min = None
                        sl_min = (int(future_time_local[sl_rel]) - int(time_ns[anchor_idx])) / MINUTE_NS
                        first_event_min = float(sl_min)

                    forced_exit_r = None
                    exit_idx = int(first_rel)
                    path_highs = future_high_local[: exit_idx + 1]
                    path_lows = future_low_local[: exit_idx + 1]

                entry_revisit_stats = _path_entry_revisit_excursion_stats(
                    side=side,
                    entry_price=anchor_price,
                    highs=np.asarray(path_highs, dtype=np.float64),
                    lows=np.asarray(path_lows, dtype=np.float64),
                    tp_ratio=target_ratio,
                    sl_ratio=sl_ratio,
                    reset_tol_pct=ENTRY_RESET_TOL_PCT,
                    exclude_last_bar=(trade_result in {"target", "sl"}),
                )

                entry_revisit_mae_r = entry_revisit_stats["entry_revisit_mae_r"]
                entry_revisit_mfe_r = entry_revisit_stats["entry_revisit_mfe_r"]

                candidates_by_side[side].append(
                    {
                        "anchor_idx": int(anchor_idx),
                        "horizon_hours": int(h),
                        "side": int(side),
                        "result": trade_result,
                        "target_first": bool(target_first),
                        "target_first_then_sl": bool(target_first_then_sl),
                        "target_minutes": None if target_min is None else float(target_min),
                        "sl_minutes": None if sl_min is None else float(sl_min),
                        "first_event_minutes": None if first_event_min is None else float(first_event_min),
                        "forced_exit_r": None if forced_exit_r is None else float(forced_exit_r),
                        "entry_revisit_mae_r": float(entry_revisit_mae_r) if np.isfinite(entry_revisit_mae_r) else np.nan,
                        "entry_revisit_mfe_r": float(entry_revisit_mfe_r) if np.isfinite(entry_revisit_mfe_r) else np.nan,
                    }
                )

    for side in (1, -1):
        candidates_by_side[side].sort(key=lambda x: (x["anchor_idx"], x["horizon_hours"]))

    return candidates_by_side


def _apply_after_loss_policy(
    candidates: List[Dict[str, Any]],
    trigger_count: int,
    skip_trades: Optional[int],
) -> List[Dict[str, Any]]:
    loss_streak = 0
    cooldown_remaining = 0
    stopped = False

    for c in candidates:
        c["taken"] = False
        c["trigger_fired"] = False

        if stopped:
            continue

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        c["taken"] = True

        if c["result"] == "sl":
            loss_streak += 1
            if loss_streak >= trigger_count:
                c["trigger_fired"] = True
                loss_streak = 0
                if skip_trades is None:
                    stopped = True
                else:
                    cooldown_remaining = int(skip_trades)
        else:
            loss_streak = 0

    return candidates


def _update_stats_from_dict(s: Dict[str, Any], c: Dict[str, Any], sl_ratio: float) -> None:
    s["anchors_total"] += 1
    s["trades_taken_count"] += 1

    if c.get("skipped_after_loss"):
        s["skipped_after_loss_count"] += 1
    if c.get("trigger_fired"):
        s["loss_streak_trigger_count"] += 1

    res = c["result"]
    if res == "target":
        s["target_first_count"] += 1
        if c.get("target_first_then_sl"):
            s["target_first_then_sl_count"] += 1
        val = c.get("target_minutes")
        if val is not None:
            s["target_minutes_sum"] += val
            s["target_minutes_count"] += 1
    elif res == "sl":
        s["sl_first_count"] += 1
        val = c.get("sl_minutes")
        if val is not None:
            s["sl_minutes_sum"] += val
            s["sl_minutes_count"] += 1
    else:
        s["censored_count"] += 1
        val = c.get("forced_exit_r")
        if val is not None:
            s["forced_exit_r_sum"] += val
            s["forced_exit_r_count"] += 1
            if val > 0:
                s["forced_exit_r_positive_count"] += 1

    val_first = c.get("first_event_minutes")
    if val_first is not None:
        s["first_event_minutes_sum"] += val_first
        s["first_event_minutes_count"] += 1

    mae_r = c.get("entry_revisit_mae_r")
    mfe_r = c.get("entry_revisit_mfe_r")

    if mae_r is not None and np.isfinite(mae_r):
        _bin_update(s["entry_revisit_mae_hist"], mae_r)
        s["entry_revisit_mae_sum"] += mae_r
        s["entry_revisit_mae_count"] += 1

    if mfe_r is not None and np.isfinite(mfe_r):
        _bin_update(s["entry_revisit_mfe_hist"], mfe_r)
        s["entry_revisit_mfe_sum"] += mfe_r
        s["entry_revisit_mfe_count"] += 1

    if res == "target":
        if mae_r is not None and np.isfinite(mae_r):
            _bin_update(s["entry_revisit_mae_win_hist"], mae_r)
            s["entry_revisit_mae_win_sum"] += mae_r
            s["entry_revisit_mae_win_count"] += 1
        if mfe_r is not None and np.isfinite(mfe_r):
            _bin_update(s["entry_revisit_mfe_win_hist"], mfe_r)
            s["entry_revisit_mfe_win_sum"] += mfe_r
            s["entry_revisit_mfe_win_count"] += 1
    elif res == "sl":
        if mae_r is not None and np.isfinite(mae_r):
            _bin_update(s["entry_revisit_mae_loss_hist"], mae_r)
            s["entry_revisit_mae_loss_sum"] += mae_r
            s["entry_revisit_mae_loss_count"] += 1
        if mfe_r is not None and np.isfinite(mfe_r):
            _bin_update(s["entry_revisit_mfe_loss_hist"], mfe_r)
            s["entry_revisit_mfe_loss_sum"] += mfe_r
            s["entry_revisit_mfe_loss_count"] += 1


def _update_stats_sequential(
    s: Dict[str, Any],
    trade: Dict[str, Any],
    res: str,
    exit_idx: int,
    time_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tp_ratio: float,
    sl_ratio: float,
) -> None:
    s["anchors_total"] += 1
    s["trades_taken_count"] += 1

    entry_idx = trade["entry_idx"]

    path_highs = high[entry_idx + 1 : exit_idx + 1]
    path_lows = low[entry_idx + 1 : exit_idx + 1]

    revisit = _path_entry_revisit_excursion_stats(
        side=trade["side"],
        entry_price=trade["entry_px"],
        highs=path_highs,
        lows=path_lows,
        tp_ratio=tp_ratio,
        sl_ratio=sl_ratio,
        exclude_last_bar=(res in {"target", "sl"}),
    )

    mae_r = revisit["entry_revisit_mae_r"]
    mfe_r = revisit["entry_revisit_mfe_r"]

    if np.isfinite(mae_r):
        _bin_update(s["entry_revisit_mae_hist"], mae_r)
        s["entry_revisit_mae_sum"] += mae_r
        s["entry_revisit_mae_count"] += 1
    if np.isfinite(mfe_r):
        _bin_update(s["entry_revisit_mfe_hist"], mfe_r)
        s["entry_revisit_mfe_sum"] += mfe_r
        s["entry_revisit_mfe_count"] += 1

    dur = (time_ns[exit_idx] - trade["entry_ns"]) / MINUTE_NS
    s["first_event_minutes_sum"] += dur
    s["first_event_minutes_count"] += 1

    if res == "target":
        future_end = int(np.searchsorted(time_ns, trade["exp"], side="left"))
        if future_end > exit_idx + 1:
            if trade["side"] == 1:
                later_sl = bool(np.any(low[exit_idx + 1 : future_end] <= trade["sl"]))
            else:
                later_sl = bool(np.any(high[exit_idx + 1 : future_end] >= trade["sl"]))
        else:
            later_sl = False
        if later_sl:
            trade["target_first_then_sl"] = True

        s["target_first_count"] += 1
        if trade.get("target_first_then_sl"):
            s["target_first_then_sl_count"] += 1
        s["target_minutes_sum"] += dur
        s["target_minutes_count"] += 1

        if np.isfinite(mae_r):
            _bin_update(s["entry_revisit_mae_win_hist"], mae_r)
            s["entry_revisit_mae_win_sum"] += mae_r
            s["entry_revisit_mae_win_count"] += 1
        if np.isfinite(mfe_r):
            _bin_update(s["entry_revisit_mfe_win_hist"], mfe_r)
            s["entry_revisit_mfe_win_sum"] += mfe_r
            s["entry_revisit_mfe_win_count"] += 1

    elif res == "sl":
        s["sl_first_count"] += 1
        s["sl_minutes_sum"] += dur
        s["sl_minutes_count"] += 1

        if np.isfinite(mae_r):
            _bin_update(s["entry_revisit_mae_loss_hist"], mae_r)
            s["entry_revisit_mae_loss_sum"] += mae_r
            s["entry_revisit_mae_loss_count"] += 1
        if np.isfinite(mfe_r):
            _bin_update(s["entry_revisit_mfe_loss_hist"], mfe_r)
            s["entry_revisit_mfe_loss_sum"] += mfe_r
            s["entry_revisit_mfe_loss_count"] += 1

    else:
        s["censored_count"] += 1
        pnl = (
            (close[exit_idx] - trade["entry_px"]) / trade["entry_px"]
            if trade["side"] == 1
            else ((trade["entry_px"] - close[exit_idx]) / trade["entry_px"])
        )
        forced_exit_r = pnl / sl_ratio
        s["forced_exit_r_sum"] += forced_exit_r
        s["forced_exit_r_count"] += 1
        if forced_exit_r > 0:
            s["forced_exit_r_positive_count"] += 1

    if trade.get("first_event_minutes") is not None:
        s["first_event_minutes_sum"] += float(trade["first_event_minutes"])
        s["first_event_minutes_count"] += 1


def _finalize_row(
    s: Dict[str, Any],
    instr: str,
    sym: str,
    pair: str,
    mode: str,
    tp: float,
    sl: float,
    h: int,
    trig: Optional[int],
    skip: Optional[int],
    sim_mode: str,
    entry_window_start_hour: int,
    entry_window_end_hour: int,
    use_entry_time_window: bool,
) -> Dict[str, Any]:
    taken = s["trades_taken_count"] or 1
    t_count = s["target_first_count"]
    s_count = s["sl_first_count"]
    net_expectancy = _net_expectancy_risk_pct(
        (t_count / taken) * 100.0,
        tp,
        (s_count / taken) * 100.0,
        sl,
        RISK_PCT,
    )

    return {
        "instrument": instr,
        "source_symbol": sym,
        "pair": pair,
        "filter_mode": mode,
        "sim_mode": sim_mode,
        "use_entry_time_window": bool(use_entry_time_window),
        "entry_window_start_hour": int(entry_window_start_hour),
        "entry_window_end_hour": int(entry_window_end_hour),
        "entry_window_label": f"{entry_window_start_hour:02d}:00-{entry_window_end_hour:02d}:00 MYT",
        "target_pct": float(tp),
        "sl_pct": float(sl),
        "horizon_hours": int(h),
        "after_loss_trigger_count": trig,
        "after_loss_skip_trades": skip,
        "anchors_total": int(s["anchors_total"]),
        "trades_taken_count": int(s["trades_taken_count"]),
        "skipped_after_loss_count": int(s["skipped_after_loss_count"]),
        "loss_streak_trigger_count": int(s["loss_streak_trigger_count"]),
        "target_first_count": int(t_count),
        "sl_first_count": int(s_count),
        "censored_count": int(s["censored_count"]),
        "target_first_then_sl_count": int(s["target_first_then_sl_count"]),
        "target_first_rate_pct": (t_count / taken) * 100.0,
        "sl_first_rate_pct": (s_count / taken) * 100.0,
        "net_expectancy_pct": float(net_expectancy),
        "net_expectancy_risk_pct": float(net_expectancy),
        "target_minutes_mean": _safe_div(s["target_minutes_sum"], s["target_minutes_count"]),
        "sl_minutes_mean": _safe_div(s["sl_minutes_sum"], s["sl_minutes_count"]),
        "first_event_minutes_mean": _safe_div(s["first_event_minutes_sum"], s["first_event_minutes_count"]),
        "forced_exit_r_mean": _safe_div(s["forced_exit_r_sum"], s["forced_exit_r_count"]),
        "forced_exit_r_positive_count": int(s["forced_exit_r_positive_count"]),
        "mae_r_mean": _safe_div(s["entry_revisit_mae_sum"], s["entry_revisit_mae_count"]),
        "mfe_r_mean": _safe_div(s["entry_revisit_mfe_sum"], s["entry_revisit_mfe_count"]),
        "mae_r_win_mean": _safe_div(s["entry_revisit_mae_win_sum"], s["entry_revisit_mae_win_count"]),
        "mae_r_loss_mean": _safe_div(s["entry_revisit_mae_loss_sum"], s["entry_revisit_mae_loss_count"]),
        "mfe_r_win_mean": _safe_div(s["entry_revisit_mfe_win_sum"], s["entry_revisit_mfe_win_count"]),
        "mfe_r_loss_mean": _safe_div(s["entry_revisit_mfe_loss_sum"], s["entry_revisit_mfe_loss_count"]),
        "entry_revisit_mae_hist": s["entry_revisit_mae_hist"].tolist(),
        "entry_revisit_mfe_hist": s["entry_revisit_mfe_hist"].tolist(),
        "entry_revisit_mae_win_hist": s["entry_revisit_mae_win_hist"].tolist(),
        "entry_revisit_mae_loss_hist": s["entry_revisit_mae_loss_hist"].tolist(),
        "entry_revisit_mfe_win_hist": s["entry_revisit_mfe_win_hist"].tolist(),
        "entry_revisit_mfe_loss_hist": s["entry_revisit_mfe_loss_hist"].tolist(),
    }


def analyze_target_sl_survival(
    df: pl.DataFrame,
    target_pct_list: Sequence[float],
    sl_pct_list: Sequence[float],
    horizon_hours_list: Sequence[int],
    instrument: str,
    source_symbol: str,
    pair: str,
    use_entry_time_window: bool = USE_ENTRY_TIME_WINDOW,
    entry_window_start_hour: int = ENTRY_WINDOW_START_HOUR_MYT,
    entry_window_end_hour: int = ENTRY_WINDOW_END_HOUR_MYT,
    trades_per_hour: Optional[int] = TRADES_PER_HOUR,
    use_ma_filter: bool = USE_MA_FILTER,
    use_after_loss_filter: bool = USE_AFTER_LOSS_FILTER,
    after_loss_trigger_count_list: Optional[Sequence[int]] = AFTER_LOSS_TRIGGER_COUNT_LIST,
    after_loss_skip_trades_list: Optional[Sequence[Optional[int]]] = AFTER_LOSS_SKIP_TRADES_LIST,
) -> pd.DataFrame:
    if df.is_empty():
        return pd.DataFrame()

    time_ns = df.get_column("time_ns").to_numpy()
    open_px = df.get_column("open").to_numpy()
    high = df.get_column("high").to_numpy()
    low = df.get_column("low").to_numpy()
    close = df.get_column("close").to_numpy()
    n = len(close)

    ma_values = _compute_ma_values(close)

    allowed_entry_indices = _build_allowed_entry_indices(
        time_ns=time_ns,
        use_entry_time_window=use_entry_time_window,
        start_hour=entry_window_start_hour,
        end_hour=entry_window_end_hour,
        trades_per_hour=trades_per_hour,
    )

    dt_utc = pd.to_datetime(time_ns, unit="ns", utc=True)
    utc_dates = pd.Series(dt_utc.date)
    unique_dates = utc_dates.unique()

    rng = np.random.default_rng(RANDOM_ENTRY_SEED)
    daily_bias_map = {day: int(rng.choice([1, -1])) for day in unique_dates}
    bias_array = utc_dates.map(daily_bias_map).to_numpy(dtype=np.int8)

    horizons = sorted({int(h) for h in horizon_hours_list if int(h) > 0})
    target_list = sorted({float(x) for x in target_pct_list if float(x) > 0.0})
    sl_list = sorted({float(x) for x in sl_pct_list if float(x) > 0.0})

    summary_rows: List[Dict[str, Any]] = []

    for sim_mode in SIMULATION_MODES:
        logger.info(">>> Mode: %s", sim_mode)

        for filter_mode, apply_ma_f, apply_al_f in _strategy_mode_list():
            if apply_al_f:
                trig_loop = _normalize_after_loss_trigger_count_list(after_loss_trigger_count_list)
                skip_loop = _normalize_after_loss_skip_trades_list(after_loss_skip_trades_list)
            else:
                trig_loop = [None]
                skip_loop = [None]

            for target_pct in target_list:
                t_ratio = target_pct / 100.0

                for sl_pct in sl_list:
                    sl_ratio = sl_pct / 100.0

                    base_cands: Optional[Dict[int, List[Dict[str, Any]]]] = None
                    if sim_mode == "overlapping":
                        base_cands = _build_trade_candidates(
                            time_ns=time_ns,
                            open_px=open_px,
                            high=high,
                            low=low,
                            close=close,
                            ma_values=ma_values,
                            horizons=horizons,
                            target_ratio=t_ratio,
                            sl_ratio=sl_ratio,
                            apply_ma_filter=apply_ma_f,
                            conservative_sl_first=True,
                        )

                    for trig_count in trig_loop:
                        for skip_val in skip_loop:
                            stats_map = {h: _new_stats() for h in horizons}

                            if sim_mode == "overlapping":
                                assert base_cands is not None
                                for side in (1, -1):
                                    side_cands = [c.copy() for c in base_cands[side]]
                                    if apply_al_f:
                                        _apply_after_loss_policy(side_cands, int(trig_count), skip_val)
                                    else:
                                        for c in side_cands:
                                            c["taken"] = True

                                    for c in side_cands:
                                        if c["anchor_idx"] not in allowed_entry_indices:
                                            continue
                                        if c.get("taken"):
                                            _update_stats_from_dict(stats_map[c["horizon_hours"]], c, sl_ratio)

                            else:
                                for h in horizons:
                                    s = stats_map[h]
                                    active_trade = None
                                    cooldown = 0
                                    streak = 0
                                    last_side = 0
                                    h_ns = np.int64(h * HOUR_NS)

                                    for i in range(n - 1):
                                        if active_trade is not None:
                                            side = active_trade["side"]
                                            is_sl = (low[i] <= active_trade["sl"]) if side == 1 else (high[i] >= active_trade["sl"])
                                            is_tp = (high[i] >= active_trade["tp"]) if side == 1 else (low[i] <= active_trade["tp"])

                                            if is_sl or is_tp or (time_ns[i] >= active_trade["exp"]):
                                                res = "sl" if is_sl else ("target" if is_tp else "censored")
                                                _update_stats_sequential(
                                                    s,
                                                    active_trade,
                                                    res,
                                                    i,
                                                    time_ns,
                                                    high,
                                                    low,
                                                    close,
                                                    t_ratio,
                                                    sl_ratio,
                                                )

                                                if res == "sl":
                                                    streak += 1
                                                    last_side = side
                                                    if apply_al_f and streak >= int(trig_count):
                                                        cooldown = int(skip_val) if skip_val is not None else 999999
                                                        streak = 0
                                                else:
                                                    streak = 0
                                                    last_side = 0
                                                active_trade = None
                                            continue

                                        if i not in allowed_entry_indices:
                                            continue

                                        if apply_al_f and cooldown > 0:
                                            cooldown -= 1
                                            s["skipped_after_loss_count"] += 1
                                            continue

                                        if sim_mode == "sequential_random":
                                            side = int(bias_array[i])
                                        else:
                                            side = -last_side if (last_side != 0 and SEQUENTIAL_SWITCH_ON_LOSS) else 1

                                        entry_px = sample_entry_price_nudged(open_px[i], high[i], low[i], close[i], i)
                                        if apply_ma_f and not _ma_pass(side, entry_px, ma_values[i]):
                                            continue

                                        active_trade = {
                                            "side": side,
                                            "entry_px": entry_px,
                                            "entry_ns": time_ns[i],
                                            "entry_idx": i,
                                            "exp": time_ns[i] + h_ns,
                                            "tp": entry_px * (1.0 + (side * t_ratio)),
                                            "sl": entry_px * (1.0 - (side * sl_ratio)),
                                        }

                            for h, s in stats_map.items():
                                summary_rows.append(
                                    _finalize_row(
                                        s,
                                        instrument,
                                        source_symbol,
                                        pair,
                                        filter_mode,
                                        target_pct,
                                        sl_pct,
                                        h,
                                        trig_count,
                                        skip_val,
                                        sim_mode,
                                        entry_window_start_hour,
                                        entry_window_end_hour,
                                        use_entry_time_window,
                                    )
                                )

    return pd.DataFrame(summary_rows)


def build_expectancy_summary_from_cache(
    cache_file: Path = CACHE_FILE,
    instrument: str = INSTRUMENT,
    source_symbol: str = MT5_SYMBOL,
    pair: str = PAIR,
    target_pct_list: Sequence[float] = TARGET_PCT_LIST,
    sl_pct_list: Sequence[float] = SL_PCT_LIST,
    horizon_hours_list: Sequence[int] = HORIZON_HOURS_LIST,
    use_entry_time_window: bool = USE_ENTRY_TIME_WINDOW,
    entry_window_start_hour: int = ENTRY_WINDOW_START_HOUR_MYT,
    entry_window_end_hour: int = ENTRY_WINDOW_END_HOUR_MYT,
    trades_per_hour: Optional[int] = TRADES_PER_HOUR,
    use_ma_filter: bool = USE_MA_FILTER,
    use_after_loss_filter: bool = USE_AFTER_LOSS_FILTER,
    after_loss_trigger_count_list: Optional[Sequence[int]] = AFTER_LOSS_TRIGGER_COUNT_LIST,
    after_loss_skip_trades_list: Optional[Sequence[Optional[int]]] = AFTER_LOSS_SKIP_TRADES_LIST,
) -> pd.DataFrame:
    df = load_df_from_cache(cache_file)
    if df.is_empty():
        return pd.DataFrame()

    summary_df = analyze_target_sl_survival(
        df=df,
        target_pct_list=target_pct_list,
        sl_pct_list=sl_pct_list,
        horizon_hours_list=horizon_hours_list,
        instrument=instrument,
        source_symbol=source_symbol,
        pair=pair,
        use_entry_time_window=use_entry_time_window,
        entry_window_start_hour=entry_window_start_hour,
        entry_window_end_hour=entry_window_end_hour,
        trades_per_hour=trades_per_hour,
        use_ma_filter=use_ma_filter,
        use_after_loss_filter=use_after_loss_filter,
        after_loss_trigger_count_list=after_loss_trigger_count_list,
        after_loss_skip_trades_list=after_loss_skip_trades_list,
    )

    if summary_df.empty:
        return summary_df

    sort_cols = [
        "instrument",
        "filter_mode",
        "sim_mode",
        "target_pct",
        "sl_pct",
        "after_loss_trigger_count",
        "after_loss_skip_trades",
        "horizon_hours",
    ]
    if "entry_window_start_hour" in summary_df.columns:
        sort_cols = [
            "instrument",
            "filter_mode",
            "entry_window_start_hour",
            "target_pct",
            "sl_pct",
            "after_loss_trigger_count",
            "after_loss_skip_trades",
            "horizon_hours",
        ]

    summary_df = summary_df.round(4).sort_values(sort_cols).reset_index(drop=True)
    return summary_df


def save_expectancy_summary_csv(
    summary_df: pd.DataFrame,
    output_file: Optional[Path] = None,
    instrument: str = INSTRUMENT,
) -> Path:
    if output_file is None:
        output_file = OUTPUT_DIR / f"expectancy_scan_{_safe_name(instrument)}_{_safe_name(ENTRY_WINDOW_LABEL)}.csv"

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _replace_file(output_file)

    export_df = summary_df.drop(columns=get_columns_to_remove(), errors="ignore")
    export_df.to_csv(output_file, index=False)
    return output_file


def save_net_expectancy_tp_sl_plot(
    summary_df: pd.DataFrame,
    instrument: str,
    pair: str,
    symbol: str,
) -> List[Path]:
    plt, sns = _import_plotting()
    if summary_df.empty:
        return []

    market = _market_tag(instrument, pair, symbol)
    saved: List[Path] = []

    if "filter_mode" not in summary_df.columns:
        return []

    group_keys = _group_keys_from_df(summary_df)
    groups = [(k, g.copy()) for k, g in summary_df.groupby(group_keys, sort=True)] if group_keys else [("ALL", summary_df.copy())]

    for key, group in groups:
        if group.empty or "net_expectancy_pct" not in group.columns:
            continue

        horizons = sorted(group["horizon_hours"].dropna().unique().tolist())
        if not horizons:
            continue

        vmin = float(group["net_expectancy_pct"].min())
        vmax = float(group["net_expectancy_pct"].max())
        label = _safe_name(str(key))

        fig, axes = plt.subplots(
            1,
            len(horizons),
            figsize=(7.5 * len(horizons), 6.5),
            squeeze=False,
            constrained_layout=True,
        )
        axes_row = axes[0]
        first_mappable = None

        for ax, h in zip(axes_row, horizons):
            sub = group[group["horizon_hours"] == h].copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            pivot = (
                sub.pivot_table(
                    index="sl_pct",
                    columns="target_pct",
                    values="net_expectancy_pct",
                    aggfunc="mean",
                ).sort_index(ascending=False)
            )

            if sns is not None:
                hm = sns.heatmap(
                    pivot,
                    ax=ax,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    annot=False,
                    linewidths=0.5,
                    linecolor="white",
                    cbar=False,
                    square=True,
                )
                mappable = hm
            else:
                data = pivot.to_numpy(dtype=float)
                im = ax.imshow(data, aspect="auto", origin="upper", vmin=vmin, vmax=vmax, cmap="viridis")
                ax.set_xticks(np.arange(len(pivot.columns)))
                ax.set_xticklabels([f"{x:g}" for x in pivot.columns], rotation=0, fontsize=8)
                ax.set_yticks(np.arange(len(pivot.index)))
                ax.set_yticklabels([f"{x:g}" for x in pivot.index], rotation=0, fontsize=8)
                mappable = im

            if first_mappable is None:
                first_mappable = mappable.collections[0] if sns is not None and getattr(mappable, "collections", None) else mappable

            ax.set_title(f"{market} | {key} | {int(h)}h", fontsize=12, pad=10)
            ax.set_xlabel("Target %")
            ax.set_ylabel("SL %")
            ax.tick_params(axis="x", labelrotation=0, labelsize=8)
            ax.tick_params(axis="y", labelrotation=0, labelsize=8)

        if first_mappable is not None:
            fig.colorbar(first_mappable, ax=axes_row.tolist(), shrink=0.9, pad=0.02, label="Net expectancy (%)")

        out = OUTPUT_DIR / f"expectancy_heatmap_{market}_{label}.png"
        _save_figure(fig, out)
        plt.close(fig)
        saved.append(out)

    return saved


def _accumulate_histograms(summary_df: pd.DataFrame, hist_col: str) -> np.ndarray:
    total_hist = np.zeros(PLOT_BINS, dtype=np.float64)
    if hist_col not in summary_df.columns:
        return total_hist

    for v in summary_df[hist_col].dropna():
        if isinstance(v, str):
            try:
                arr = np.fromstring(v.strip("[]"), sep=",")
                if arr.size == PLOT_BINS:
                    total_hist += arr
                continue
            except Exception:
                continue

        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == PLOT_BINS:
            total_hist += np.asarray(v, dtype=np.float64)

    return total_hist


def save_mae_mfe_distribution_plots(
    summary_df: pd.DataFrame,
    instrument: str,
    pair: str,
    symbol: str,
) -> List[Path]:
    plt, _ = _import_plotting()
    if summary_df.empty:
        return []

    market = _market_tag(instrument, pair, symbol)
    saved: List[Path] = []

    specs = [
        ("entry_revisit_mae_hist", "Entry-revisit MAE (R)", f"mae_distribution_{market}.png"),
        ("entry_revisit_mfe_hist", "Entry-revisit MFE (R)", f"mfe_distribution_{market}.png"),
        ("entry_revisit_mae_win_hist", "Entry-revisit MAE win-only (R)", f"mae_win_distribution_{market}.png"),
        ("entry_revisit_mae_loss_hist", "Entry-revisit MAE loss-only (R)", f"mae_loss_distribution_{market}.png"),
        ("entry_revisit_mfe_win_hist", "Entry-revisit MFE win-only (R)", f"mfe_win_distribution_{market}.png"),
        ("entry_revisit_mfe_loss_hist", "Entry-revisit MFE loss-only (R)", f"mfe_loss_distribution_{market}.png"),
    ]

    centers = (PLOT_BIN_EDGES[:-1] + PLOT_BIN_EDGES[1:]) / 2.0
    width = (PLOT_BIN_EDGES[1] - PLOT_BIN_EDGES[0]) * 0.9

    for hist_col, y_label, fname in specs:
        total_hist = _accumulate_histograms(summary_df, hist_col)
        if total_hist.sum() <= 0:
            continue

        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        ax.bar(centers, total_hist, width=width)
        ax.set_title(f"{market} | {y_label}")
        ax.set_xlabel(y_label)
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.25)

        out = OUTPUT_DIR / fname
        _save_figure(fig, out)
        plt.close(fig)
        saved.append(out)

    return saved


def save_expectancy_plots(
    summary_df: pd.DataFrame,
    instrument: str = INSTRUMENT,
    pair: str = PAIR,
    symbol: str = MT5_SYMBOL,
) -> List[Path]:
    plot_files: List[Path] = []
    plot_files.extend(
        save_mae_mfe_distribution_plots(
            summary_df=summary_df,
            instrument=instrument,
            pair=pair,
            symbol=symbol,
        )
    )
    plot_files.extend(
        save_net_expectancy_tp_sl_plot(
            summary_df=summary_df,
            instrument=instrument,
            pair=pair,
            symbol=symbol,
        )
    )
    return plot_files


def run_expectancy_scan() -> dict[str, str]:
    clear_previous_outputs()

    source_symbol = MT5_SYMBOL

    logger.info("Starting expectancy scan")
    logger.info("Cache file: %s", CACHE_FILE)
    logger.info("Entry window: %s", ENTRY_WINDOW_LABEL)

    summary_df = build_expectancy_summary_from_cache(
        cache_file=CACHE_FILE,
        instrument=INSTRUMENT,
        source_symbol=source_symbol,
        pair=PAIR,
        target_pct_list=TARGET_PCT_LIST,
        sl_pct_list=SL_PCT_LIST,
        horizon_hours_list=HORIZON_HOURS_LIST,
        use_entry_time_window=USE_ENTRY_TIME_WINDOW,
        entry_window_start_hour=ENTRY_WINDOW_START_HOUR_MYT,
        entry_window_end_hour=ENTRY_WINDOW_END_HOUR_MYT,
        trades_per_hour=TRADES_PER_HOUR,
        use_ma_filter=USE_MA_FILTER,
        use_after_loss_filter=USE_AFTER_LOSS_FILTER,
        after_loss_trigger_count_list=AFTER_LOSS_TRIGGER_COUNT_LIST,
        after_loss_skip_trades_list=AFTER_LOSS_SKIP_TRADES_LIST,
    )

    if summary_df.empty:
        logger.warning("No summary produced.")
        return {}

    output_file = save_expectancy_summary_csv(summary_df=summary_df, instrument=INSTRUMENT)
    plot_files = save_expectancy_plots(
        summary_df=summary_df,
        instrument=INSTRUMENT,
        pair=PAIR,
        symbol=source_symbol,
    )

    logger.info("Saved summary CSV to: %s", output_file)
    logger.info("Saved plot count: %d", len(plot_files))
    for p in plot_files:
        logger.info("Saved plot: %s", p)

    return {
        "summary_file": str(output_file),
        "cache_file": str(CACHE_FILE),
        "plots": [str(p) for p in plot_files],
    }


if __name__ == "__main__":
    run_expectancy_scan()
