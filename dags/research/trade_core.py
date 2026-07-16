# trade_core.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from .exploration_config import ExplorationConfig
except ImportError:
    from exploration_config import ExplorationConfig

try:
    from .trailing_tp import simulate_adverse_trailing_tp
except ImportError:
    from trailing_tp import simulate_adverse_trailing_tp

MINUTE_NS = 60 * 1_000_000_000
HOUR_NS = 60 * MINUTE_NS


def safe_name(value: str) -> str:
    text = str(value).strip()
    out: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", ".", "*"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "value"


def _config_first_or_value(value: Any, default: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        if len(value) > 0:
            return value[0]
        return default
    return default if value is None else value


def replace_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def load_cache_df(cache_file: Path) -> pd.DataFrame:
    if not cache_file.exists():
        return pd.DataFrame()

    try:
        if cache_file.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(cache_file)
        else:
            df = pd.read_csv(cache_file)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    needed = ["time_ns", "open", "high", "low", "close"]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()

    df = df[needed].copy()
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.dropna(subset=needed)
        .drop_duplicates(subset=["time_ns"])
        .sort_values("time_ns")
        .reset_index(drop=True)
    )


def bootstrap_mean_samples(values: np.ndarray, n_bootstrap: int, block_size: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return np.asarray([], dtype=np.float64)

    n_bootstrap = max(1, int(n_bootstrap))
    block_size = max(1, min(int(block_size), n))
    rng = np.random.default_rng(int(seed))

    if n == 1:
        return np.full(n_bootstrap, values[0], dtype=np.float64)

    starts = np.arange(0, n - block_size + 1, dtype=np.int64)
    if starts.size == 0:
        starts = np.arange(0, n, dtype=np.int64)
        block_size = 1

    out = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample: list[float] = []
        while len(sample) < n:
            start = int(rng.choice(starts))
            sample.extend(values[start : start + block_size].tolist())
        out[i] = float(np.mean(sample[:n]))
    return out


def describe_numeric_series(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "count": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "count": float(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def _compute_ma_values(close: np.ndarray, ma_type: str, ma_period_bars: int) -> np.ndarray:
    s = pd.Series(close, dtype="float64")
    ma_type = str(ma_type).lower().strip()
    if ma_type == "ema":
        ma = s.ewm(span=int(ma_period_bars), adjust=False, min_periods=int(ma_period_bars)).mean()
    elif ma_type == "sma":
        ma = s.rolling(window=int(ma_period_bars), min_periods=int(ma_period_bars)).mean()
    else:
        raise ValueError("ma_type must be 'ema' or 'sma'")
    return ma.shift(1).to_numpy(dtype=np.float64, copy=False)


def _ma_pass(side: int, price: float, ma_value: float) -> bool:
    if not np.isfinite(ma_value):
        return True
    return price < ma_value if int(side) == 1 else price > ma_value


def sample_entry_price_nudged(
    open_px: float,
    high_px: float,
    low_px: float,
    close_px: float,
    anchor_idx: int,
    randomize_entry_price: bool,
    entry_nudge_max_fraction: float,
    entry_nudge_clip_to_candle: bool,
    seed: int,
) -> float:
    if not randomize_entry_price:
        return float(close_px)

    if not all(np.isfinite(x) for x in (open_px, high_px, low_px, close_px)):
        return np.nan

    lo, hi = float(low_px), float(high_px)
    if hi < lo:
        lo, hi = hi, lo

    candle_range = max(hi - lo, 0.0)
    if candle_range <= 0.0:
        return float(close_px)

    rng = np.random.default_rng(int(seed) + (int(anchor_idx) + 1) * 1_000_003)
    max_nudge = candle_range * float(entry_nudge_max_fraction)
    entry = float(close_px) + float(rng.uniform(-max_nudge, max_nudge))

    if entry_nudge_clip_to_candle:
        entry = min(max(entry, lo), hi)

    return float(entry)


def _bucket_start_hour_from_time_ns_myt(time_ns: int, bucket_hours: int) -> int:
    ts_utc = pd.Timestamp(int(time_ns), unit="ns", tz="UTC")
    ts_myt = ts_utc.tz_convert("Asia/Kuala_Lumpur")
    return int((ts_myt.hour // int(bucket_hours)) * int(bucket_hours))


def _bucket_label_from_start_hour(start_hour: int, bucket_hours: int) -> str:
    return f"{start_hour:02d}:00-{start_hour + int(bucket_hours):02d}:00 MYT"


def _market_time_ns(time_ns: np.ndarray) -> np.ndarray:
    if time_ns.size < 2:
        return time_ns.astype(np.int64, copy=False)

    diffs = np.diff(time_ns).astype(np.int64, copy=False)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return time_ns.astype(np.int64, copy=False)

    bar_ns = int(np.median(positive_diffs))
    gap_threshold = max(2 * HOUR_NS, bar_ns * 3)
    closure_gaps = np.where(diffs > gap_threshold, diffs - bar_ns, 0)
    cumulative = np.concatenate(([0], np.cumsum(closure_gaps, dtype=np.int64)))
    return time_ns.astype(np.int64, copy=False) - cumulative


def _trade_move_pct(side: int, entry_price: float, exit_price: float) -> float:
    if not np.isfinite(entry_price) or entry_price <= 0.0 or not np.isfinite(exit_price):
        return np.nan
    if int(side) == 1:
        return ((exit_price - entry_price) / entry_price) * 100.0
    return ((entry_price - exit_price) / entry_price) * 100.0


def _spread_adjusted_barriers(
    side: int,
    entry_price: float,
    target_ratio: float,
    sl_ratio: float,
    spread_pct: float,
) -> tuple[float, float]:
    spread_ratio = float(spread_pct) / 100.0
    if int(side) == 1:
        target_price = entry_price * (1.0 + target_ratio + spread_ratio)
        sl_price = entry_price * (1.0 - sl_ratio + spread_ratio)
    else:
        target_price = entry_price * (1.0 - target_ratio - spread_ratio)
        sl_price = entry_price * (1.0 + sl_ratio - spread_ratio)
    return target_price, sl_price


def _trade_timer_expiry_ns(entry_time_ns: int, trade_timer_minutes: int) -> int:
    return int(entry_time_ns + int(trade_timer_minutes) * MINUTE_NS)


def _resolve_trade_from_window(
    *,
    symbol: str,
    side: int,
    volume: float,
    anchor_idx: int,
    future_start: int,
    future_end: int,
    time_ns: np.ndarray,
    open_px: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_price: float,
    target_ratio: float,
    sl_ratio: float,
    spread_pct: float,
    risk_pct: float,
    use_trailing_tp: bool = False,
    trailing_tp_activation_r: float = 0.6,
    trailing_tp_pct: float = 0.2,
    trailing_tp_interval: int = 3,
    use_trade_timer: bool = False,
    trade_timer_minutes: int = 0,
) -> Dict[str, Any]:
    future_end = int(min(len(time_ns), max(int(future_start) + 1, int(future_end))))
    future_high = high[future_start:future_end]
    future_low = low[future_start:future_end]
    future_close = close[future_start:future_end]
    future_time = time_ns[future_start:future_end]

    target_price, sl_price = _spread_adjusted_barriers(
        side=side,
        entry_price=entry_price,
        target_ratio=target_ratio,
        sl_ratio=sl_ratio,
        spread_pct=spread_pct,
    )

    timer_expiry_ns = None
    timer_abs_idx = None
    timer_triggerable = False
    if bool(use_trade_timer) and int(trade_timer_minutes) > 0:
        timer_expiry_ns = _trade_timer_expiry_ns(int(time_ns[anchor_idx]), int(trade_timer_minutes))
        timer_abs_idx = int(np.searchsorted(time_ns, timer_expiry_ns, side="left"))
        if int(future_start) <= timer_abs_idx < int(future_end):
            timer_triggerable = True
            future_end = min(future_end, timer_abs_idx + 1)
            future_high = high[future_start:future_end]
            future_low = low[future_start:future_end]
            future_close = close[future_start:future_end]
            future_time = time_ns[future_start:future_end]

    if int(side) == 1:
        target_hits = np.flatnonzero(future_high >= target_price)
        sl_hits = np.flatnonzero(future_low <= sl_price)
    else:
        target_hits = np.flatnonzero(future_low <= target_price)
        sl_hits = np.flatnonzero(future_high >= sl_price)

    target_rel = int(target_hits[0]) if target_hits.size else -1
    sl_rel = int(sl_hits[0]) if sl_hits.size else -1

    if target_rel == -1 and sl_rel == -1:
        if timer_triggerable and timer_abs_idx is not None and timer_abs_idx < len(time_ns):
            exit_type = "timer"
            exit_abs_idx = int(timer_abs_idx)
            exit_rel = int(exit_abs_idx - future_start)
            exit_price = float(close[exit_abs_idx])
            exit_time_ns = int(time_ns[exit_abs_idx])
        else:
            exit_type = "censored"
            exit_rel = int(len(future_close) - 1)
            exit_abs_idx = int(future_start + exit_rel)
            exit_price = float(future_close[exit_rel])
            exit_time_ns = int(future_time[exit_rel])
    else:
        if target_rel != -1 and sl_rel != -1:
            target_first = target_rel < sl_rel
        elif target_rel != -1:
            target_first = True
        else:
            target_first = False

        if target_first:
            exit_type = "tp"
            exit_rel = target_rel
        else:
            exit_type = "sl"
            exit_rel = sl_rel

        exit_abs_idx = int(future_start + exit_rel)
        exit_price = float(future_close[exit_rel])
        exit_time_ns = int(future_time[exit_rel])

    trailing_candidate = None
    if use_trailing_tp:
        trailing_candidate = simulate_adverse_trailing_tp(
            side=side,
            entry_price=entry_price,
            sl_price=sl_price,
            target_price=target_price,
            future_high=future_high,
            future_low=future_low,
            future_close=future_close,
            future_time_ns=future_time,
            activation_r=trailing_tp_activation_r,
            trailing_tp_pct=trailing_tp_pct,
            trailing_tp_interval=trailing_tp_interval,
        )

        if trailing_candidate.get("trailing_triggered"):
            trailing_tp_idx = int(trailing_candidate["trailing_tp_idx"])
            if trailing_tp_idx < int(exit_rel):
                exit_type = "trailing"
                exit_rel = trailing_tp_idx
                exit_abs_idx = int(future_start + exit_rel)
                exit_price = float(trailing_candidate["trailing_tp_price"])
                exit_time_ns = int(trailing_candidate["trailing_tp_time_ns"])

    move_pct = _trade_move_pct(side, entry_price, exit_price)
    trade_r = (move_pct / (float(sl_ratio) * 100.0)) if np.isfinite(move_pct) and sl_ratio > 0 else np.nan
    trade_expectancy_pct = float(trade_r * float(risk_pct) * 100.0) if np.isfinite(trade_r) else np.nan

    row = {
        "instrument": symbol,
        "source_symbol": symbol,
        "pair": symbol,
        "anchor_idx": int(anchor_idx),
        "time_ns": int(time_ns[anchor_idx]),
        "entry_time_myt": pd.Timestamp(int(time_ns[anchor_idx]), unit="ns", tz="UTC").tz_convert("Asia/Kuala_Lumpur"),
        "side": int(side),
        "side_label": "buy" if int(side) == 1 else "sell",
        "entry_price": float(entry_price),
        "target_price": float(target_price),
        "sl_price": float(sl_price),
        "exit_type": str(exit_type),
        "exit_time_ns": int(exit_time_ns),
        "exit_price": float(exit_price),
        "trade_r": float(trade_r) if np.isfinite(trade_r) else np.nan,
        "trade_expectancy_pct": float(trade_expectancy_pct) if np.isfinite(trade_expectancy_pct) else np.nan,
        "move_pct": float(move_pct) if np.isfinite(move_pct) else np.nan,
        "exit_time_myt": pd.Timestamp(int(exit_time_ns), unit="ns", tz="UTC").tz_convert("Asia/Kuala_Lumpur"),
        "entry_bucket_start_hour": 0,
        "entry_bucket_label": "ALL HOURS",
        "volume": float(volume),
        "trailing_tp_triggered": bool(trailing_candidate.get("trailing_triggered", False)) if trailing_candidate else False,
        "trailing_tp_updates": int(trailing_candidate.get("trailing_tp_updates", 0)) if trailing_candidate else 0,
        "trailing_tp_moved_r": (
            float(trailing_candidate.get("trailing_tp_moved_r", np.nan))
            if trailing_candidate and trailing_candidate.get("trailing_triggered", False)
            else np.nan
        ),
        "use_trade_timer": bool(use_trade_timer),
        "trade_timer_minutes": int(trade_timer_minutes),
        "trade_timer_expiry_ns": int(timer_expiry_ns) if timer_expiry_ns is not None else np.nan,
        "timer_triggered": bool(exit_type == "timer"),
        "timer_close": bool(exit_type == "timer"),
    }
    return row


def _entry_window_mask(
    time_ns: np.ndarray,
    use_entry_time_window: bool,
    start_hour: int,
    end_hour: int,
) -> np.ndarray:
    if time_ns.size == 0:
        return np.zeros(0, dtype=bool)

    hours_utc = pd.to_datetime(time_ns, unit="ns", utc=True).hour.to_numpy(dtype=np.int8, copy=False)
    if not use_entry_time_window or int(start_hour) == int(end_hour):
        return np.ones(len(hours_utc), dtype=bool)

    start_hour = int(start_hour)
    end_hour = int(end_hour)
    if start_hour < end_hour:
        return (hours_utc >= start_hour) & (hours_utc < end_hour)
    return (hours_utc >= start_hour) | (hours_utc < end_hour)


def _select_indices_per_hour(
    time_ns: np.ndarray,
    eligible_mask: np.ndarray,
    entries_per_hour: int,
    seed: int,
) -> set[int]:
    eligible_idx = np.flatnonzero(eligible_mask)
    if eligible_idx.size == 0:
        return set()

    if int(entries_per_hour) <= 0:
        return set(int(i) for i in eligible_idx.tolist())

    dt_utc = pd.to_datetime(time_ns[eligible_idx], unit="ns", utc=True)
    df_tmp = pd.DataFrame({"idx": eligible_idx, "hour_group": dt_utc.floor("h")})

    rng = np.random.default_rng(int(seed))
    selected: set[int] = set()

    for _, grp in df_tmp.groupby("hour_group", sort=True):
        idxs = grp["idx"].to_numpy(dtype=int, copy=False)
        if idxs.size <= int(entries_per_hour):
            selected.update(idxs.tolist())
        else:
            chosen = rng.choice(idxs, size=int(entries_per_hour), replace=False)
            selected.update(chosen.tolist())

    return selected


def _choose_side_for_anchor(
    *,
    anchor_idx: int,
    side_mode: str,
    simulation_mode: str,
    seed: int,
    time_ns: np.ndarray,
    use_daily_side_bias: bool,
    use_utc_for_bias: bool,
) -> int:
    side_mode = str(side_mode).lower().strip()
    if side_mode == "buy":
        return 1
    if side_mode == "sell":
        return -1

    if simulation_mode == "sequential_random" and use_daily_side_bias:
        dt = pd.to_datetime(time_ns, unit="ns", utc=True)
        dates = dt.date
        bias_dates = dates if use_utc_for_bias else dates
        unique_days = pd.Index(bias_dates).unique()
        rng = np.random.default_rng(int(seed))
        daily_bias_map = {day: int(rng.choice([1, -1])) for day in unique_days}
        return int(daily_bias_map[bias_dates[anchor_idx]])

    rng = np.random.default_rng(int(seed) + (int(anchor_idx) + 1) * 1_000_003)
    return int(rng.choice([1, -1]))


def build_trade_universe(df: pd.DataFrame, config: ExplorationConfig) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    needed = ["time_ns", "open", "high", "low", "close"]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()

    df = df[needed].copy()
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=needed)
        .drop_duplicates(subset=["time_ns"])
        .sort_values("time_ns")
        .reset_index(drop=True)
    )

    if df.empty or len(df) < 2:
        return pd.DataFrame()

    time_ns = df["time_ns"].to_numpy(dtype=np.int64, copy=False)
    open_px = df["open"].to_numpy(dtype=np.float64, copy=False)
    high = df["high"].to_numpy(dtype=np.float64, copy=False)
    low = df["low"].to_numpy(dtype=np.float64, copy=False)
    close = df["close"].to_numpy(dtype=np.float64, copy=False)
    n = int(len(df))

    ma_values = _compute_ma_values(close, config.ma_type, config.ma_period_bars)
    market_time_ns = _market_time_ns(time_ns)
    horizon_ns = np.int64(int(config.horizon_hours) * HOUR_NS)

    target_ratio = float(config.target_pct) / 100.0
    sl_ratio = float(config.sl_pct) / 100.0

    eligible_mask = _entry_window_mask(
        time_ns=time_ns,
        use_entry_time_window=bool(config.use_entry_time_window),
        start_hour=int(getattr(config, "entry_window_start_hour_myt", 15)),
        end_hour=int(getattr(config, "entry_window_end_hour_myt", 23)),
    )

    simulation_mode = str(getattr(config, "simulation_mode", "overlapping_random")).lower().strip()
    side_mode = str(getattr(config, "side_mode", "mixed")).lower().strip()
    entries_per_hour = int(getattr(config, "entries_per_hour", 1))
    use_daily_side_bias = bool(getattr(config, "use_daily_side_bias", False))
    use_utc_for_bias = bool(getattr(config, "use_utc_for_bias", True))

    use_trade_timer = bool(_config_first_or_value(getattr(config, "use_trade_timer", False), False))
    trade_timer_minutes = int(_config_first_or_value(getattr(config, "trade_timer_minutes", 0), 0))

    use_trailing_tp = bool(_config_first_or_value(getattr(config, "use_trailing_tp", False), False))
    trailing_tp_activation_r = float(_config_first_or_value(getattr(config, "trailing_tp_activation_r", 0.6), 0.6))
    trailing_tp_pct = float(_config_first_or_value(getattr(config, "trailing_tp_distance_r", 0.2), 0.2))
    trailing_tp_interval = int(_config_first_or_value(getattr(config, "trailing_tp_interval", 3), 3))

    rows: list[dict[str, Any]] = []
    attempted_count = 0
    resolved_count = 0
    censored_count = 0
    timer_count = 0

    def _make_row(anchor_idx: int, side: int) -> Optional[dict[str, Any]]:
        nonlocal attempted_count, resolved_count, censored_count, timer_count

        if anchor_idx < 0 or anchor_idx >= n - 1:
            return None

        anchor_price = sample_entry_price_nudged(
            open_px=float(open_px[anchor_idx]),
            high_px=float(high[anchor_idx]),
            low_px=float(low[anchor_idx]),
            close_px=float(close[anchor_idx]),
            anchor_idx=anchor_idx,
            randomize_entry_price=bool(config.randomize_entry_price),
            entry_nudge_max_fraction=float(config.entry_nudge_max_fraction),
            entry_nudge_clip_to_candle=bool(config.entry_nudge_clip_to_candle),
            seed=int(config.bootstrap_seed),
        )
        if not np.isfinite(anchor_price) or anchor_price <= 0.0:
            return None

        ma_value = float(ma_values[anchor_idx]) if anchor_idx < len(ma_values) else np.nan
        if bool(config.use_ma_filter) and not _ma_pass(side, anchor_price, ma_value):
            return None

        future_start = anchor_idx + 1
        horizon_end_abs = int(np.searchsorted(market_time_ns, market_time_ns[anchor_idx] + horizon_ns, side="right"))
        future_end = min(n, max(future_start + 1, horizon_end_abs))

        entry_window_row = _resolve_trade_from_window(
            symbol=str(config.instrument),
            side=side,
            volume=1.0,
            anchor_idx=anchor_idx,
            future_start=future_start,
            future_end=future_end,
            time_ns=time_ns,
            open_px=open_px,
            high=high,
            low=low,
            close=close,
            entry_price=float(anchor_price),
            target_ratio=target_ratio,
            sl_ratio=sl_ratio,
            spread_pct=float(config.spread_pct),
            risk_pct=float(config.risk_pct),
            use_trailing_tp=bool(getattr(config, "use_trailing_tp", False)),
            trailing_tp_activation_r=float(_config_first_or_value(getattr(config, "trailing_tp_activation_r", 0.6), 0.6)),
            trailing_tp_pct=float(_config_first_or_value(getattr(config, "trailing_tp_distance_r", 0.2), 0.2)),
            trailing_tp_interval=int(_config_first_or_value(getattr(config, "trailing_tp_interval", 3), 3)),
            use_trade_timer=use_trade_timer,
            trade_timer_minutes=trade_timer_minutes,
        )

        if not entry_window_row:
            return None

        entry_window_row["ma_value"] = float(ma_value) if np.isfinite(ma_value) else np.nan
        entry_window_row["use_ma_filter"] = bool(config.use_ma_filter)
        entry_window_row["target_pct"] = float(config.target_pct)
        entry_window_row["sl_pct"] = float(config.sl_pct)
        entry_window_row["spread_pct"] = float(config.spread_pct)
        entry_window_row["horizon_hours"] = int(config.horizon_hours)
        entry_window_row["volume"] = 1.0
        entry_window_row["entry_bucket_start_hour"] = 0
        entry_window_row["entry_bucket_label"] = "ALL HOURS"

        attempted_count += 1
        if entry_window_row["exit_type"] == "censored":
            censored_count += 1
        else:
            resolved_count += 1
        if entry_window_row["exit_type"] == "timer":
            timer_count += 1

        return entry_window_row

    if simulation_mode == "sequential_random":
        active: Optional[dict[str, Any]] = None
        seed = int(config.bootstrap_seed)

        for i in range(n - 1):
            if active is not None:
                side = int(active["side"])
                timer_hit = (
                    active.get("timer_exp") is not None
                    and int(time_ns[i]) >= int(active["timer_exp"])
                )
                is_sl = (low[i] <= active["sl_price"]) if side == 1 else (high[i] >= active["sl_price"])
                is_tp = (high[i] >= active["target_price"]) if side == 1 else (low[i] <= active["target_price"])

                if is_sl or is_tp or timer_hit or (market_time_ns[i] >= active["market_exp"]):
                    if is_sl:
                        exit_type = "sl"
                    elif is_tp:
                        exit_type = "tp"
                    elif timer_hit:
                        exit_type = "timer"
                    else:
                        exit_type = "censored"

                    exit_price = float(close[i])
                    move_pct = _trade_move_pct(side, float(active["entry_price"]), exit_price)
                    trade_r = (move_pct / (float(sl_ratio) * 100.0)) if np.isfinite(move_pct) and sl_ratio > 0 else np.nan
                    active.update(
                        {
                            "exit_type": exit_type,
                            "exit_time_ns": int(time_ns[i]),
                            "exit_time_myt": pd.Timestamp(int(time_ns[i]), unit="ns", tz="UTC").tz_convert("Asia/Kuala_Lumpur"),
                            "exit_price": exit_price,
                            "move_pct": float(move_pct) if np.isfinite(move_pct) else np.nan,
                            "trade_r": float(trade_r) if np.isfinite(trade_r) else np.nan,
                            "trade_expectancy_pct": float(trade_r * config.risk_pct * 100.0) if np.isfinite(trade_r) else np.nan,
                            "timer_triggered": bool(exit_type == "timer"),
                            "timer_close": bool(exit_type == "timer"),
                        }
                    )
                    rows.append(active)
                    active = None
                continue

            if not bool(eligible_mask[i]):
                continue

            side = _choose_side_for_anchor(
                anchor_idx=i,
                side_mode=side_mode,
                simulation_mode=simulation_mode,
                seed=seed,
                time_ns=time_ns,
                use_daily_side_bias=use_daily_side_bias,
                use_utc_for_bias=use_utc_for_bias,
            )

            anchor_price = sample_entry_price_nudged(
                open_px=float(open_px[i]),
                high_px=float(high[i]),
                low_px=float(low[i]),
                close_px=float(close[i]),
                anchor_idx=i,
                randomize_entry_price=bool(config.randomize_entry_price),
                entry_nudge_max_fraction=float(config.entry_nudge_max_fraction),
                entry_nudge_clip_to_candle=bool(config.entry_nudge_clip_to_candle),
                seed=seed,
            )
            if not np.isfinite(anchor_price) or anchor_price <= 0.0:
                continue

            ma_value = float(ma_values[i]) if i < len(ma_values) else np.nan
            if bool(config.use_ma_filter) and not _ma_pass(side, anchor_price, ma_value):
                continue

            target_price, sl_price = _spread_adjusted_barriers(
                side=side,
                entry_price=float(anchor_price),
                target_ratio=target_ratio,
                sl_ratio=sl_ratio,
                spread_pct=float(config.spread_pct),
            )

            active = {
                "instrument": str(config.instrument),
                "source_symbol": str(config.mt5_symbol),
                "pair": str(config.pair),
                "anchor_idx": int(i),
                "time_ns": int(time_ns[i]),
                "entry_time_myt": pd.Timestamp(int(time_ns[i]), unit="ns", tz="UTC").tz_convert("Asia/Kuala_Lumpur"),
                "side": int(side),
                "side_label": "buy" if int(side) == 1 else "sell",
                "entry_price": float(anchor_price),
                "target_price": float(target_price),
                "sl_price": float(sl_price),
                "exit_type": None,
                "exit_time_ns": None,
                "exit_time_myt": None,
                "exit_price": None,
                "trade_r": np.nan,
                "trade_expectancy_pct": np.nan,
                "move_pct": np.nan,
                "entry_bucket_start_hour": 0,
                "entry_bucket_label": "ALL HOURS",
                "volume": 1.0,
                "market_exp": int(market_time_ns[i] + horizon_ns),
                "ma_value": float(ma_value) if np.isfinite(ma_value) else np.nan,
                "use_ma_filter": bool(config.use_ma_filter),
                "target_pct": float(config.target_pct),
                "sl_pct": float(config.sl_pct),
                "spread_pct": float(config.spread_pct),
                "horizon_hours": int(config.horizon_hours),
                "use_trade_timer": bool(use_trade_timer),
                "trade_timer_minutes": int(trade_timer_minutes),
                "timer_exp": (
                    int(time_ns[i] + int(trade_timer_minutes) * MINUTE_NS)
                    if bool(use_trade_timer) and int(trade_timer_minutes) > 0
                    else None
                ),
                "timer_triggered": False,
                "timer_close": False,
            }

        if active is not None:
            side = int(active["side"])
            exit_idx = n - 1
            exit_price = float(close[exit_idx])
            move_pct = _trade_move_pct(side, float(active["entry_price"]), exit_price)
            trade_r = (move_pct / (float(sl_ratio) * 100.0)) if np.isfinite(move_pct) and sl_ratio > 0 else np.nan
            active.update(
                {
                    "exit_type": "censored",
                    "exit_time_ns": int(time_ns[exit_idx]),
                    "exit_time_myt": pd.Timestamp(int(time_ns[exit_idx]), unit="ns", tz="UTC").tz_convert("Asia/Kuala_Lumpur"),
                    "exit_price": exit_price,
                    "move_pct": float(move_pct) if np.isfinite(move_pct) else np.nan,
                    "trade_r": float(trade_r) if np.isfinite(trade_r) else np.nan,
                    "trade_expectancy_pct": float(trade_r * config.risk_pct * 100.0) if np.isfinite(trade_r) else np.nan,
                }
            )
            rows.append(active)

    else:
        allowed_indices: set[int]
        if simulation_mode == "overlapping_random":
            allowed_indices = _select_indices_per_hour(
                time_ns=time_ns,
                eligible_mask=eligible_mask,
                entries_per_hour=entries_per_hour,
                seed=int(config.bootstrap_seed),
            )
        else:
            allowed_indices = set(int(i) for i in np.flatnonzero(eligible_mask).tolist())

        seed = int(config.bootstrap_seed)
        for anchor_idx in sorted(allowed_indices):
            side = _choose_side_for_anchor(
                anchor_idx=anchor_idx,
                side_mode=side_mode,
                simulation_mode=simulation_mode,
                seed=seed,
                time_ns=time_ns,
                use_daily_side_bias=use_daily_side_bias,
                use_utc_for_bias=use_utc_for_bias,
            )
            row = _make_row(anchor_idx, side)
            if row is not None:
                rows.append(row)

    trade_df = pd.DataFrame(rows)
    trade_df.attrs["attempted_count"] = int(attempted_count)
    trade_df.attrs["resolved_count"] = int(resolved_count)
    trade_df.attrs["censored_count"] = int(censored_count)
    trade_df.attrs["timer_count"] = int(timer_count)

    if trade_df.empty:
        return trade_df

    sort_cols = [c for c in ["time_ns", "side", "anchor_idx"] if c in trade_df.columns]
    if sort_cols:
        trade_df = trade_df.sort_values(sort_cols).reset_index(drop=True)
    return trade_df


def save_trade_universe_csv(trade_df: pd.DataFrame, output_file: Path) -> Path:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    replace_file(output_file)
    trade_df.to_csv(output_file, index=False)
    return output_file
