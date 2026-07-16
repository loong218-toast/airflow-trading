# expectancy_analysis.py

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
    ENTRY_WINDOW_START_HOUR_MYT,
    HORIZON_HOURS_LIST,
    HORIZON_LABEL,
    INSTRUMENT,
    MA_PERIOD_BARS,
    MA_TYPE,
    MIX_BUY_SELL,
    MT5_SYMBOL,
    OUTPUT_DIR,
    PAIR,
    RANDOM_ENTRY_SEED,
    LOOKBACK_DAYS,
    get_lookback_days_list,
    RISK_PCT,
    SL_PCT_LIST,
    SPREAD_PCT,
    TARGET_PCT_LIST,
    TRADES_PER_HOUR,
    USE_AFTER_LOSS_FILTER,
    USE_DAILY_SIDE_BIAS,
    USE_ENTRY_TIME_WINDOW,
    USE_MAE_MFE_STATS,
    USE_MA_FILTER,
    _market_tag,
    _net_expectancy_risk_pct,
    _safe_name,
    get_columns_to_remove,
    sample_entry_price_nudged,
)

if USE_MAE_MFE_STATS:
    from research.expectancy_mae_mfe import (
        PLOT_BINS,
        PLOT_BIN_EDGES,
        PLOT_MAX_R,
        ENTRY_RESET_TOL_PCT,
        _accumulate_histograms,
        _path_pre_resolution_excursion_stats,
        save_mae_mfe_distribution_plots,
        make_mae_mfe_stats,
    )
else:
    PLOT_BINS = np.array([], dtype=np.float64)
    PLOT_BIN_EDGES = np.array([], dtype=np.float64)
    PLOT_MAX_R = 0.0
    ENTRY_RESET_TOL_PCT = 0.0

    def _accumulate_histograms(*args, **kwargs):
        return None

    def _path_pre_resolution_excursion_stats(*args, **kwargs):
        return {"entry_revisit_mae_r": np.nan, "entry_revisit_mfe_r": np.nan}

    def save_mae_mfe_distribution_plots(*args, **kwargs):
        return []

    def make_mae_mfe_stats():
        return {}

logger = logging.getLogger(__name__)

MINUTE_NS = 60 * 1_000_000_000
HOUR_NS = 60 * MINUTE_NS



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
        "expectancy_resolved_heatmap_*.png",
        "mae_distribution_*.png",
        "mfe_distribution_*.png",
        "mae_win_distribution_*.png",
        "mae_loss_distribution_*.png",
        "mfe_win_distribution_*.png",
        "mfe_loss_distribution_*.png",
    ):
        for p in OUTPUT_DIR.rglob(pattern):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass

def _horizon_output_dir(horizon_hours: int, lookback_days: int) -> Path:
    from research.expectancy_config import INSTRUMENT, OUTPUT_BASE_DIR
    out = OUTPUT_BASE_DIR / INSTRUMENT / f"lookback_{int(lookback_days)}d" / f"horizon{int(horizon_hours)}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _horizon_output_part(horizon_hours: int) -> str:
    return f"horizon{int(horizon_hours)}"

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


def _log_scan_progress(done: int, total: int, label: str) -> None:
    if total <= 0:
        return
    if done == 1 or done == total or done % 50 == 0:
        pct = (done / total) * 100.0
        logger.info("%s %d/%d (%.1f%%)", label, done, total, pct)


def _parse_time_ns_array(time_ns: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(time_ns, unit="ns", utc=True)


def _utc_hour_of(time_ns: np.ndarray) -> np.ndarray:
    return _parse_time_ns_array(time_ns).hour.to_numpy(dtype=np.int8, copy=False)


def _utc_date_of(time_ns: np.ndarray) -> np.ndarray:
    return _parse_time_ns_array(time_ns).date

def _build_market_time_ns(time_ns: np.ndarray) -> np.ndarray:
    if time_ns.size < 2:
        return time_ns.astype(np.int64, copy=False)

    diffs = np.diff(time_ns).astype(np.int64, copy=False)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return time_ns.astype(np.int64, copy=False)

    bar_ns = int(np.median(positive_diffs))

    # only compress long market-closure gaps
    gap_threshold = max(2 * HOUR_NS, bar_ns * 3)
    closure_gaps = np.where(diffs > gap_threshold, diffs - bar_ns, 0)

    cumulative = np.concatenate(([0], np.cumsum(closure_gaps, dtype=np.int64)))
    return time_ns.astype(np.int64, copy=False) - cumulative


def _effective_entry_window_label() -> str:
    if not USE_ENTRY_TIME_WINDOW:
        return "24H"
    return f"{ENTRY_WINDOW_START_HOUR_MYT:02d}:00-{ENTRY_WINDOW_END_HOUR_MYT:02d}:00 MYT"


def _group_key_label(key: Any) -> str:
    if isinstance(key, tuple):
        key = "_".join(str(x) for x in key)
    return _safe_name(str(key))


def _side_mode_list() -> list[tuple[str, Optional[int]]]:
    if MIX_BUY_SELL:
        return [("mixed", None)]
    return [("buy", 1), ("sell", -1)]


def _suffix_part(value: Optional[str]) -> str:
    if not value:
        return ""
    return f"_{_safe_name(value)}"


def _output_window_part() -> str:
    return _safe_name(_effective_entry_window_label())


def _new_stats() -> Dict[str, Any]:
    stats = {
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
    }

    if USE_MAE_MFE_STATS:
        stats.update(make_mae_mfe_stats())
    else:
        stats.update(
            {
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
                "entry_revisit_mae_hist": np.zeros(0, dtype=np.int64),
                "entry_revisit_mfe_hist": np.zeros(0, dtype=np.int64),
                "entry_revisit_mae_win_hist": np.zeros(0, dtype=np.int64),
                "entry_revisit_mae_loss_hist": np.zeros(0, dtype=np.int64),
                "entry_revisit_mfe_win_hist": np.zeros(0, dtype=np.int64),
                "entry_revisit_mfe_loss_hist": np.zeros(0, dtype=np.int64),
            }
        )

    return stats


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

def _spread_adjusted_barriers(
    side: int,
    entry_price: float,
    target_ratio: float,
    sl_ratio: float,
    spread_pct: float,
) -> tuple[float, float]:
    # CONVERT spread_pct (0.007) to a ratio (0.00007)
    spread_ratio = float(spread_pct) / 100.0 
    
    if side == 1:
        target_price = entry_price * (1.0 + target_ratio + spread_ratio)
        sl_price = entry_price * (1.0 - sl_ratio + spread_ratio)
    else:
        target_price = entry_price * (1.0 - target_ratio - spread_ratio)
        sl_price = entry_price * (1.0 + sl_ratio - spread_ratio)
    return target_price, sl_price


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
    if "side_mode" in summary_df.columns and summary_df["side_mode"].nunique(dropna=True) > 1:
        keys.append("side_mode")
    if "vol_category" in summary_df.columns and summary_df["vol_category"].nunique(dropna=True) > 1:
        keys.append("vol_category")
    return keys


def _build_allowed_entry_indices(
    time_ns: np.ndarray,
    use_entry_time_window: bool = USE_ENTRY_TIME_WINDOW,
    start_hour: int = ENTRY_WINDOW_START_HOUR_MYT,
    end_hour: int = ENTRY_WINDOW_END_HOUR_MYT,
    trades_per_hour: Optional[int] = TRADES_PER_HOUR,
) -> set[int]:
    if time_ns.size == 0:
        return set()

    hours_utc = _utc_hour_of(time_ns)
    if not use_entry_time_window or start_hour == end_hour:
        entry_mask = np.ones(len(hours_utc), dtype=bool)
    elif start_hour < end_hour:
        entry_mask = (hours_utc >= start_hour) & (hours_utc < end_hour)
    else:
        entry_mask = (hours_utc >= start_hour) | (hours_utc < end_hour)

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


def save_net_expectancy_tp_sl_plot(
    summary_df: pd.DataFrame,
    instrument: str,
    pair: str,
    symbol: str,
    lookback_days: int = LOOKBACK_DAYS,
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

    metric_specs = [
        ("net_expectancy_pct", "Net expectancy (%)", "expectancy_heatmap"),
        ("expectancy_resolved_pct", "Resolved expectancy (%)", "expectancy_resolved_heatmap"),
    ]

    for key, group in groups:
        if group.empty:
            continue

        label = _group_key_label(key)

        for h in sorted(group["horizon_hours"].dropna().unique().tolist()):
            sub_h = group[group["horizon_hours"] == h].copy()
            if sub_h.empty:
                continue

            for metric_col, colorbar_label, file_prefix in metric_specs:
                if metric_col not in sub_h.columns:
                    continue

                metric_df = sub_h[sub_h[metric_col].notna()].copy()
                if metric_df.empty:
                    continue

                pivot = (
                    metric_df.pivot_table(
                        index="sl_pct",
                        columns="target_pct",
                        values=metric_col,
                        aggfunc="mean",
                    )
                    .sort_index(ascending=False)
                )

                if pivot.empty:
                    continue

                vmin = float(np.nanmin(pivot.to_numpy(dtype=float)))
                vmax = float(np.nanmax(pivot.to_numpy(dtype=float)))

                fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)

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
                    mappable = hm.collections[0] if getattr(hm, "collections", None) else hm
                else:
                    data = pivot.to_numpy(dtype=float)
                    mappable = ax.imshow(data, aspect="auto", origin="upper", vmin=vmin, vmax=vmax, cmap="viridis")
                    ax.set_xticks(np.arange(len(pivot.columns)))
                    ax.set_xticklabels([f"{x:g}" for x in pivot.columns], rotation=0, fontsize=8)
                    ax.set_yticks(np.arange(len(pivot.index)))
                    ax.set_yticklabels([f"{x:g}" for x in pivot.index], rotation=0, fontsize=8)

                ax.set_title(f"{market} | {key} | {int(h)}h", fontsize=12, pad=10)
                ax.set_xlabel("Target %")
                ax.set_ylabel("SL %")
                ax.tick_params(axis="x", labelrotation=0, labelsize=8)
                ax.tick_params(axis="y", labelrotation=0, labelsize=8)
                fig.colorbar(mappable, ax=ax, shrink=0.9, pad=0.02, label=colorbar_label)

                out_dir = _horizon_output_dir(int(h), lookback_days)
                out = out_dir / f"{file_prefix}_{market}_{label}.png"
                _save_figure(fig, out)
                plt.close(fig)
                saved.append(out)

    return saved



def _build_trade_candidates(
    time_ns: np.ndarray,
    market_time_ns: np.ndarray,
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
    allowed_anchor_mask: Optional[np.ndarray] = None,
    spread_pct: float = 0.0,
) -> Dict[int, List[Dict[str, Any]]]:
    n = int(close.shape[0])
    candidates_by_side: Dict[int, List[Dict[str, Any]]] = {1: [], -1: []}
    need_mae_mfe = USE_MAE_MFE_STATS

    horizon_end_by_h: Dict[int, np.ndarray] = {}
    for h in horizons:
        horizon_end_by_h[h] = np.searchsorted(
            market_time_ns,
            market_time_ns + np.int64(h * HOUR_NS),
            side="right",
        )

    for anchor_idx in range(n - 1):
        if allowed_anchor_mask is not None and (
            anchor_idx >= len(allowed_anchor_mask) or not bool(allowed_anchor_mask[anchor_idx])
        ):
            continue

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

            target_price, sl_price = _spread_adjusted_barriers(
                side=side,
                entry_price=anchor_price,
                target_ratio=target_ratio,
                sl_ratio=sl_ratio,
                spread_pct=spread_pct,
            )

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
                    target_hits = np.flatnonzero(future_high_local >= target_price)
                    sl_hits = np.flatnonzero(future_low_local <= sl_price)
                else:
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
                    # holding time = full horizon in 5m candles -> minutes
                    first_event_min = float(local_len * 5.0)
                    exit_idx = local_len - 1
                    path_highs = future_high_local[: exit_idx + 1] if need_mae_mfe else None
                    path_lows = future_low_local[: exit_idx + 1] if need_mae_mfe else None
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
                    path_highs = future_high_local[: exit_idx + 1] if need_mae_mfe else None
                    path_lows = future_low_local[: exit_idx + 1] if need_mae_mfe else None

                if need_mae_mfe:
                    entry_revisit_stats = _path_pre_resolution_excursion_stats(
                        side=side,
                        entry_price=anchor_price,
                        highs=np.asarray(path_highs, dtype=np.float64),
                        lows=np.asarray(path_lows, dtype=np.float64),
                        tp_ratio=target_ratio,
                        sl_ratio=sl_ratio,
                    )
                    entry_revisit_mae_r = entry_revisit_stats["entry_revisit_mae_r"]
                    entry_revisit_mfe_r = entry_revisit_stats["entry_revisit_mfe_r"]

                    candidate = {
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
                else:
                    candidate = {
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
                    }

                candidates_by_side[side].append(candidate)

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
    s["trades_taken_count"] += 1
    res = c["result"]

    if res == "target":
        s["target_first_count"] += 1
        if c.get("target_first_then_sl"):
            s["target_first_then_sl_count"] += 1
        if c.get("target_minutes") is not None:
            s["target_minutes_sum"] += c["target_minutes"]
            s["target_minutes_count"] += 1
    elif res == "sl":
        s["sl_first_count"] += 1
        if c.get("sl_minutes") is not None:
            s["sl_minutes_sum"] += c["sl_minutes"]
            s["sl_minutes_count"] += 1
    else:
        s["censored_count"] += 1
        if c.get("forced_exit_r") is not None:
            s["forced_exit_r_sum"] += c["forced_exit_r"]
            s["forced_exit_r_count"] += 1
            if c["forced_exit_r"] > 0:
                s["forced_exit_r_positive_count"] += 1

    # record holding time for every trade
    if c.get("first_event_minutes") is not None:
        s["first_event_minutes_sum"] += c["first_event_minutes"]
        s["first_event_minutes_count"] += 1

    if not USE_MAE_MFE_STATS:
        return

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
    market_time_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tp_ratio: float,
    sl_ratio: float,
) -> None:
    s["anchors_total"] += 1
    s["trades_taken_count"] += 1

    entry_idx = trade["entry_idx"]

    dur = (time_ns[exit_idx] - trade["entry_ns"]) / MINUTE_NS
    s["first_event_minutes_sum"] += dur
    s["first_event_minutes_count"] += 1

    if USE_MAE_MFE_STATS:
        path_highs = high[entry_idx + 1 : exit_idx + 1]
        path_lows = low[entry_idx + 1 : exit_idx + 1]

        revisit = _path_pre_resolution_excursion_stats(
            side=trade["side"],
            entry_price=trade["entry_px"],
            highs=path_highs,
            lows=path_lows,
            tp_ratio=tp_ratio,
            sl_ratio=sl_ratio,
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
    else:
        mae_r = np.nan
        mfe_r = np.nan

    if res == "target":
        future_end = int(np.searchsorted(market_time_ns, trade["market_exp"], side="left"))
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

        if USE_MAE_MFE_STATS:
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

        if USE_MAE_MFE_STATS:
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
            else (trade["entry_px"] - close[exit_idx]) / trade["entry_px"]
        )
        forced_exit_r = pnl / sl_ratio
        s["forced_exit_r_sum"] += forced_exit_r
        s["forced_exit_r_count"] += 1
        if forced_exit_r > 0:
            s["forced_exit_r_positive_count"] += 1



def _finalize_row(
    s: Dict[str, Any],
    instr: str,
    sym: str,
    pair: str,
    mode: str,
    tp: float,
    sl: float,
    spread_pct: float,
    h: int,
    trig: Optional[int],
    skip: Optional[int],
    sim_mode: str,
    side_mode: str,
    lookback_days: int = LOOKBACK_DAYS,
) -> Dict[str, Any]:

    taken = s["trades_taken_count"] or 1
    t_count = s["target_first_count"]
    s_count = s["sl_first_count"]
    resolved_count = t_count + s_count
    censored_count = int(s["censored_count"])

    resolved_expectancy_pct = np.nan
    if resolved_count > 0:
        resolved_expectancy_pct = _net_expectancy_risk_pct(
            (t_count / resolved_count) * 100.0,
            tp,
            (s_count / resolved_count) * 100.0,
            sl,
            RISK_PCT,
            spread_pct=0.0,
        )

    resolved_rate = resolved_count / taken
    expectancy_resolved_pct = (
        float(resolved_expectancy_pct) * resolved_rate
        if np.isfinite(resolved_expectancy_pct)
        else np.nan
    )

    forced_exit_r_mean = _safe_div(s["forced_exit_r_sum"], s["forced_exit_r_count"])
    forced_exit_rate = censored_count / taken
    spread_cost_pct = (float(spread_pct) / float(sl)) * (RISK_PCT * 100.0) if sl > 0 and spread_pct > 0 else 0.0

    net_expectancy = expectancy_resolved_pct if np.isfinite(expectancy_resolved_pct) else 0.0
    if forced_exit_r_mean is not None and np.isfinite(forced_exit_r_mean):
        net_expectancy += forced_exit_rate * ((forced_exit_r_mean * RISK_PCT * 100.0) - spread_cost_pct)

    avg_holding_time_minutes = _safe_div(s["first_event_minutes_sum"], s["first_event_minutes_count"])
    avg_holding_time = _safe_div(avg_holding_time_minutes, 5.0) if avg_holding_time_minutes is not None else None
    net_expectancy_rate = _safe_div(net_expectancy, avg_holding_time) if avg_holding_time is not None else None

    row = {
        "instrument": instr,
        "source_symbol": sym,
        "pair": pair,
        "lookback_d": int(lookback_days),
        "filter_mode": mode,
        "sim_mode": sim_mode,
        "side_mode": side_mode,
        "use_entry_time_window": bool(USE_ENTRY_TIME_WINDOW),
        "entry_window_start_hour": int(ENTRY_WINDOW_START_HOUR_MYT),
        "entry_window_end_hour": int(ENTRY_WINDOW_END_HOUR_MYT),
        "entry_window_label": _effective_entry_window_label(),
        "target_pct": float(tp),
        "sl_pct": float(sl),
        "spread_pct": float(spread_pct),
        "horizon_hours": int(h),
        "after_loss_trigger_count": trig,
        "after_loss_skip_trades": skip,
        "anchors_total": int(s["anchors_total"]),
        "trades_taken_count": int(s["trades_taken_count"]),
        "skipped_after_loss_count": int(s["skipped_after_loss_count"]),
        "loss_streak_trigger_count": int(s["loss_streak_trigger_count"]),
        "target_first_count": int(t_count),
        "sl_first_count": int(s_count),
        "censored_count": censored_count,
        "target_first_then_sl_count": int(s["target_first_then_sl_count"]),
        "resolved_trades_count": int(resolved_count),
        "resolved_trades_pct": resolved_rate * 100.0,
        "censored_trades_pct": forced_exit_rate * 100.0,
        "target_first_rate_pct": (t_count / taken) * 100.0,
        "sl_first_rate_pct": (s_count / taken) * 100.0,
        "expectancy_resolved_pct": float(expectancy_resolved_pct),
        "net_expectancy_pct": float(net_expectancy),
        "net_expectancy_rate": np.nan if net_expectancy_rate is None else float(net_expectancy_rate),
        "avg_holding_time": np.nan if avg_holding_time is None else float(avg_holding_time),
        "target_minutes_mean": _safe_div(s["target_minutes_sum"], s["target_minutes_count"]),
        "sl_minutes_mean": _safe_div(s["sl_minutes_sum"], s["sl_minutes_count"]),
        "first_event_minutes_mean": avg_holding_time_minutes,
        "forced_exit_r_mean": np.nan if forced_exit_r_mean is None else float(forced_exit_r_mean),
        "forced_exit_r_positive_count": int(s["forced_exit_r_positive_count"]),
    }

    if USE_MAE_MFE_STATS:
        row.update(
            {
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
        )

    return row




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
    use_daily_side_bias: bool = USE_DAILY_SIDE_BIAS,
    use_utc_for_bias: bool = True,
    spread_pct: float = SPREAD_PCT,
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    if df.is_empty():
        return pd.DataFrame()

    from research.expectancy_config import SIMULATION_MODES

    time_ns = df.get_column("time_ns").to_numpy()
    open_px = df.get_column("open").to_numpy()
    high = df.get_column("high").to_numpy()
    low = df.get_column("low").to_numpy()
    close = df.get_column("close").to_numpy()
    ma_values = _compute_ma_values(close)
    n = int(close.shape[0])

    rng = np.random.default_rng(RANDOM_ENTRY_SEED)

    valid_entry_indices = _build_allowed_entry_indices(
        time_ns=time_ns,
        use_entry_time_window=use_entry_time_window,
        start_hour=entry_window_start_hour,
        end_hour=entry_window_end_hour,
        trades_per_hour=trades_per_hour,
    )

    valid_entry_mask = np.zeros(n, dtype=bool)
    if valid_entry_indices:
        idx_arr = np.fromiter(valid_entry_indices, dtype=np.int64, count=len(valid_entry_indices))
        idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < n)]
        valid_entry_mask[idx_arr] = True

    dt_utc = pd.to_datetime(time_ns, unit="ns", utc=True)
    date_series = dt_utc.date
    if use_daily_side_bias:
        bias_dates = date_series if use_utc_for_bias else date_series
        unique_days = pd.Index(bias_dates).unique()
        daily_bias_map = {day: int(rng.choice([1, -1])) for day in unique_days}
        bias_array = np.array([daily_bias_map[day] for day in bias_dates], dtype=np.int8)
    else:
        bias_array = None

    horizons = sorted({int(h) for h in horizon_hours_list if int(h) > 0})
    target_list = sorted({float(x) for x in target_pct_list if float(x) > 0.0})
    sl_list = sorted({float(x) for x in sl_pct_list if float(x) > 0.0})

    summary_rows: List[Dict[str, Any]] = []

    market_time_ns = _build_market_time_ns(time_ns)

    for side_mode, fixed_side in _side_mode_list():
        logger.info(">>> Side mode: %s", side_mode)

        for sim_mode in SIMULATION_MODES:
            logger.info(">>> Mode: %s", sim_mode)

            for filter_mode, apply_ma_f, apply_al_f in _strategy_mode_list():
                trig_loop = _normalize_after_loss_trigger_count_list(after_loss_trigger_count_list) if apply_al_f else [None]
                skip_loop = _normalize_after_loss_skip_trades_list(after_loss_skip_trades_list) if apply_al_f else [None]

                for target_pct in target_list:
                    t_ratio = target_pct / 100.0
                    for sl_pct in sl_list:
                        sl_ratio = sl_pct / 100.0

                        base_cands = None
                        if sim_mode in {"overlapping", "overlapping_random"}:
                            allowed_anchor_mask = valid_entry_mask if sim_mode == "overlapping_random" else None
                            base_cands = _build_trade_candidates(
                                time_ns=time_ns,
                                market_time_ns=market_time_ns,
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
                                allowed_anchor_mask=allowed_anchor_mask,
                                spread_pct=spread_pct,
                            )

                        for trig_count in trig_loop:
                            for skip_val in skip_loop:
                                stats_map = {h: _new_stats() for h in horizons}

                                if sim_mode in {"overlapping", "overlapping_random"}:
                                    if base_cands is None:
                                        continue

                                    for side in (1, -1):
                                        if fixed_side is not None and side != int(fixed_side):
                                            continue

                                        side_cands = [c.copy() for c in base_cands[side]]

                                        if apply_al_f:
                                            _apply_after_loss_policy(side_cands, int(trig_count), skip_val)
                                        else:
                                            for c in side_cands:
                                                c["taken"] = True

                                        for c in side_cands:
                                            s = stats_map[c["horizon_hours"]]
                                            if c.get("taken"):
                                                _update_stats_from_dict(s, c, sl_ratio)

                                else:
                                    for h in horizons:
                                        s = stats_map[h]
                                        active_t = None
                                        cooldown_idx = 0
                                        loss_streak = 0
                                        last_lost_side = 0
                                        h_ns = np.int64(h * HOUR_NS)

                                        for i in range(n - 1):
                                            if active_t is not None:
                                                side = active_t["side"]
                                                is_sl = (low[i] <= active_t["sl"]) if side == 1 else (high[i] >= active_t["sl"])
                                                is_tp = (high[i] >= active_t["tp"]) if side == 1 else (low[i] <= active_t["tp"])

                                                if is_sl or is_tp or (market_time_ns[i] >= active_t["market_exp"]):
                                                    res = "sl" if is_sl else ("target" if is_tp else "censored")
                                                    _update_stats_sequential(
                                                        s,
                                                        active_t,
                                                        res,
                                                        i,
                                                        time_ns,
                                                        market_time_ns,
                                                        high,
                                                        low,
                                                        close,
                                                        t_ratio,
                                                        sl_ratio,
                                                        spread_pct,
                                                    )
                                                    if res == "sl":
                                                        loss_streak += 1
                                                        last_lost_side = side
                                                        if apply_al_f and loss_streak >= int(trig_count):
                                                            cooldown_idx = i + (int(skip_val) if skip_val is not None else 999999)
                                                            loss_streak = 0
                                                    else:
                                                        loss_streak = 0
                                                        last_lost_side = 0
                                                    active_t = None
                                                continue

                                            if i < cooldown_idx:
                                                continue
                                            if not valid_entry_mask[i]:
                                                continue
                                            if apply_al_f and cooldown_idx > i:
                                                s["skipped_after_loss_count"] += 1
                                                continue

                                            if fixed_side is not None:
                                                check_order = [int(fixed_side)]
                                            elif sim_mode == "sequential_random":
                                                if bias_array is not None:
                                                    check_order = [int(bias_array[i])]
                                                else:
                                                    check_order = [int(rng.choice([1, -1]))]
                                            else:
                                                check_order = [1, -1]
                                                if last_lost_side != 0 and True:
                                                    check_order = ([-1, 1] if last_lost_side == 1 else [1, -1])

                                            entry_px = sample_entry_price_nudged(open_px[i], high[i], low[i], close[i], i)
                                            if not np.isfinite(entry_px):
                                                continue

                                            chosen_side = None
                                            for side in check_order:
                                                if apply_ma_f and not _ma_pass(side, entry_px, ma_values[i]):
                                                    continue
                                                chosen_side = int(side)
                                                break

                                            if chosen_side is None:
                                                continue

                                            active_t = {
                                                "side": chosen_side,
                                                "entry_px": entry_px,
                                                "entry_ns": time_ns[i],
                                                "entry_idx": i,
                                                "tp": _spread_adjusted_barriers(
                                                    side=chosen_side,
                                                    entry_price=entry_px,
                                                    target_ratio=t_ratio,
                                                    sl_ratio=sl_ratio,
                                                    spread_pct=spread_pct,
                                                )[0],
                                                "sl": _spread_adjusted_barriers(
                                                    side=chosen_side,
                                                    entry_price=entry_px,
                                                    target_ratio=t_ratio,
                                                    sl_ratio=sl_ratio,
                                                    spread_pct=spread_pct,
                                                )[1],
                                                    "market_exp": market_time_ns[i] + h_ns
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
                                            spread_pct,
                                            h,
                                            trig_count,
                                            skip_val,
                                            sim_mode,
                                            side_mode,
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
    use_daily_side_bias: bool = USE_DAILY_SIDE_BIAS,
    use_utc_for_bias: bool = True,
    spread_pct: float = SPREAD_PCT,
    lookback_days: int = LOOKBACK_DAYS,
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
        use_daily_side_bias=use_daily_side_bias,
        use_utc_for_bias=use_utc_for_bias,
        spread_pct=spread_pct,
        lookback_days=lookback_days,
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

    summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True)

    round_map = {c: 4 for c in summary_df.select_dtypes(include="number").columns}
    round_map["net_expectancy_rate"] = 8

    summary_df = summary_df.round(round_map)
    
    return summary_df


def save_expectancy_summary_csv(
    summary_df: pd.DataFrame,
    output_file: Optional[Path] = None,
    instrument: str = INSTRUMENT,
    suffix: Optional[str] = None,
    lookback_days: int = LOOKBACK_DAYS,
) -> Path:
    summary_df["lookback_d"] = int(lookback_days)

    preferred_cols = [
        "instrument",
        "source_symbol",
        "pair",
        "lookback_d",
        "filter_mode",
        "sim_mode",
        "side_mode",
        "use_entry_time_window",
        "entry_window_start_hour",
        "entry_window_end_hour",
        "entry_window_label",
        "target_pct",
        "sl_pct",
        "spread_pct",
        "horizon_hours",
        "after_loss_trigger_count",
        "after_loss_skip_trades",
        "anchors_total",
        "trades_taken_count",
        "skipped_after_loss_count",
        "loss_streak_trigger_count",
        "target_first_count",
        "sl_first_count",
        "censored_count",
        "target_first_then_sl_count",
        "resolved_trades_count",
        "resolved_trades_pct",
        "censored_trades_pct",
        "target_first_rate_pct",
        "sl_first_rate_pct",
        "expectancy_resolved_pct",
        "net_expectancy_pct",
        "net_expectancy_rate",
        "avg_holding_time",
        "target_minutes_mean",
        "sl_minutes_mean",
        "first_event_minutes_mean",
        "forced_exit_r_mean",
        "forced_exit_r_positive_count",
    ]

    summary_df = summary_df.reindex(
        columns=[c for c in preferred_cols if c in summary_df.columns]
        + [c for c in summary_df.columns if c not in preferred_cols]
    )

    window_part = _output_window_part()
    cols_to_remove = get_columns_to_remove()

    if output_file is None:
        output_file = OUTPUT_DIR / f"expectancy_scan_{_safe_name(instrument)}_{window_part}_lookback_{int(lookback_days)}d{_suffix_part(suffix)}.csv"

    output_file = Path(output_file)
    _replace_file(output_file)
    summary_df.drop(columns=cols_to_remove, errors="ignore").to_csv(
        output_file,
        index=False,
        float_format="%.8f",
    )

    if "side_mode" in summary_df.columns and "horizon_hours" in summary_df.columns:
        for (side_mode, h_val), group_df in summary_df.groupby(["side_mode", "horizon_hours"], sort=True):
            h_int = int(h_val)
            side_dir = _horizon_output_dir(h_int, lookback_days)
            side_filename = (
                f"expectancy_scan_{_safe_name(instrument)}horizon{h_int}_"
                f"{window_part}_lookback_{int(lookback_days)}d_{_safe_name(str(side_mode))}.csv"
            )
            side_file_path = side_dir / side_filename

            _replace_file(side_file_path)
            group_df.drop(columns=cols_to_remove, errors="ignore").to_csv(
                side_file_path,
                index=False,
                float_format="%.8f",
            )

    return output_file


def save_expectancy_plots(
    summary_df: pd.DataFrame,
    instrument: str = INSTRUMENT,
    pair: str = PAIR,
    symbol: str = MT5_SYMBOL,
    lookback_days: int = LOOKBACK_DAYS,
) -> List[Path]:
    plot_files: List[Path] = []

    if USE_MAE_MFE_STATS:
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
            lookback_days=lookback_days,
        )
    )
    return plot_files


def run_expectancy_scan(lookback_days: int = LOOKBACK_DAYS) -> dict[str, str]:
    clear_previous_outputs()

    source_symbol = MT5_SYMBOL

    logger.info("Starting expectancy scan")
    logger.info("Cache file: %s", CACHE_FILE)
    logger.info("Entry window label: %s", _effective_entry_window_label())
    logger.info("Use daily bias: %s", USE_DAILY_SIDE_BIAS)
    logger.info("Use sequential mode(s): %s", ", ".join(SIMULATION_MODES))

    summary_df = build_expectancy_summary_from_cache(
        cache_file=CACHE_FILE,
        instrument=INSTRUMENT,
        source_symbol=MT5_SYMBOL,
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
        use_daily_side_bias=USE_DAILY_SIDE_BIAS,
        use_utc_for_bias=True,
        spread_pct=SPREAD_PCT,
        lookback_days=lookback_days,
    )

    if summary_df.empty:
        return {}

    output_file = save_expectancy_summary_csv(
        summary_df=summary_df,
        instrument=INSTRUMENT,
        lookback_days=lookback_days,
    )
    plot_files = save_expectancy_plots(
        summary_df=summary_df,
        instrument=INSTRUMENT,
        pair=PAIR,
        symbol=MT5_SYMBOL,
        lookback_days=lookback_days,
    )

    logger.info("Saved summary CSV to: %s", output_file)
    logger.info("Saved plot count: %d", len(plot_files))
    for p in plot_files:
        logger.info("Saved plot: %s", p)

    return {
        "lookback_days": str(lookback_days),
        "summary_file": str(output_file),
        "cache_file": str(CACHE_FILE),
        "plots": [str(p) for p in plot_files],
    }


def run_expectancy_scan_all_lookbacks() -> list[dict[str, str]]:
    outputs = []
    for lookback_days in get_lookback_days_list():
        outputs.append(run_expectancy_scan(lookback_days=lookback_days))
    return outputs

if __name__ == "__main__":
    run_expectancy_scan_all_lookbacks()
