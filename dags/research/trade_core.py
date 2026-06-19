from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .exploration_config import ExplorationConfig
except ImportError:
    from exploration_config import ExplorationConfig


MINUTE_NS = 60 * 1_000_000_000
HOUR_NS = 60 * MINUTE_NS


def safe_name(value: str) -> str:
    text = str(value).strip()
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "value"


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
        sample = []
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

    n = len(df)
    time_ns = df["time_ns"].to_numpy(dtype=np.int64, copy=False)
    open_px = df["open"].to_numpy(dtype=np.float64, copy=False)
    high = df["high"].to_numpy(dtype=np.float64, copy=False)
    low = df["low"].to_numpy(dtype=np.float64, copy=False)
    close = df["close"].to_numpy(dtype=np.float64, copy=False)

    rng_selection = np.random.default_rng(int(config.bootstrap_seed))
    hour_groups = time_ns // HOUR_NS
    unique_hours = np.unique(hour_groups)

    selected_indices_map: dict[int, int] = {}
    picks_per_hour = max(1, int(config.entries_per_hour))

    for h_val in unique_hours:
        indices_in_hour = np.where((hour_groups == h_val) & (np.arange(n) < n - 1))[0]
        if len(indices_in_hour) == 0:
            continue

        n_to_pick = min(picks_per_hour, len(indices_in_hour))
        chosen_idxs = rng_selection.choice(indices_in_hour, size=n_to_pick, replace=False)

        for idx in chosen_idxs:
            side = int(rng_selection.choice([1, -1]))
            selected_indices_map[int(idx)] = side

    horizon_ns = np.int64(int(config.horizon_hours) * HOUR_NS)
    horizon_end = np.searchsorted(time_ns, time_ns + horizon_ns, side="right")
    ma_values = _compute_ma_values(close, config.ma_type, config.ma_period_bars)

    rows: list[dict[str, Any]] = []
    attempted_count = 0
    resolved_count = 0
    censored_count = 0

    target_ratio = float(config.target_pct) / 100.0
    sl_ratio = float(config.sl_pct) / 100.0

    for anchor_idx in sorted(selected_indices_map.keys()):
        side = int(selected_indices_map[anchor_idx])

        anchor_price = sample_entry_price_nudged(
            open_px=float(open_px[anchor_idx]),
            high_px=float(high[anchor_idx]),
            low_px=float(low[anchor_idx]),
            close_px=float(close[anchor_idx]),
            anchor_idx=anchor_idx,
            randomize_entry_price=config.randomize_entry_price,
            entry_nudge_max_fraction=config.entry_nudge_max_fraction,
            entry_nudge_clip_to_candle=config.entry_nudge_clip_to_candle,
            seed=int(config.bootstrap_seed),
        )
        if not np.isfinite(anchor_price) or anchor_price <= 0.0:
            continue

        ma_value = float(ma_values[anchor_idx]) if anchor_idx < len(ma_values) else np.nan
        if config.use_ma_filter and not _ma_pass(side, anchor_price, ma_value):
            continue

        bucket_start_hour = (
            _bucket_start_hour_from_time_ns_myt(int(time_ns[anchor_idx]), int(config.entry_bucket_hours))
            if config.use_entry_bucket_hours
            else 0
        )
        bucket_label = (
            _bucket_label_from_start_hour(bucket_start_hour, int(config.entry_bucket_hours))
            if config.use_entry_bucket_hours
            else "ALL HOURS"
        )

        horizon_end_abs = int(horizon_end[anchor_idx])
        if horizon_end_abs <= anchor_idx + 1:
            continue

        future_high = high[anchor_idx + 1 : horizon_end_abs]
        future_low = low[anchor_idx + 1 : horizon_end_abs]
        future_time = time_ns[anchor_idx + 1 : horizon_end_abs]

        if side == 1:
            target_price = anchor_price * (1.0 + target_ratio)
            sl_price = anchor_price * (1.0 - sl_ratio)
            target_hits = np.flatnonzero(future_high >= target_price)
            sl_hits = np.flatnonzero(future_low <= sl_price)
        else:
            target_price = anchor_price * (1.0 - target_ratio)
            sl_price = anchor_price * (1.0 + sl_ratio)
            target_hits = np.flatnonzero(future_low <= target_price)
            sl_hits = np.flatnonzero(future_high >= sl_price)

        target_rel = int(target_hits[0]) if target_hits.size else -1
        sl_rel = int(sl_hits[0]) if sl_hits.size else -1

        attempted_count += 1

        if target_rel == -1 and sl_rel == -1:
            censored_count += 1
            continue

        resolved_count += 1

        if target_rel != -1 and sl_rel != -1:
            target_first = target_rel < sl_rel
        elif target_rel != -1:
            target_first = True
        else:
            target_first = False

        if target_first:
            trade_r = float(target_ratio / sl_ratio) if sl_ratio > 0 else np.nan
            exit_type = "target"
            exit_rel = target_rel
        else:
            trade_r = -1.0
            exit_type = "sl"
            exit_rel = sl_rel

        exit_time_ns = int(future_time[exit_rel]) if 0 <= exit_rel < len(future_time) else np.nan
        holding_minutes = (
            (exit_time_ns - int(time_ns[anchor_idx])) / MINUTE_NS
            if np.isfinite(exit_time_ns)
            else np.nan
        )

        entry_dt_myt = pd.Timestamp(int(time_ns[anchor_idx]), unit="ns", tz="UTC").tz_convert(config.timezone)

        rows.append(
            {
                "instrument": config.instrument,
                "source_symbol": config.mt5_symbol,
                "pair": config.pair,
                "anchor_idx": int(anchor_idx),
                "time_ns": int(time_ns[anchor_idx]),
                "entry_time_myt": entry_dt_myt,
                "side": int(side),
                "side_label": "buy" if side == 1 else "sell",
                "entry_price": float(anchor_price),
                "target_pct": float(config.target_pct),
                "sl_pct": float(config.sl_pct),
                "horizon_hours": int(config.horizon_hours),
                "use_ma_filter": bool(config.use_ma_filter),
                "ma_value": float(ma_value) if np.isfinite(ma_value) else np.nan,
                "exit_type": str(exit_type),
                "trade_r": float(trade_r) if np.isfinite(trade_r) else np.nan,
                "trade_expectancy_pct": float(trade_r * config.risk_pct * 100.0) if np.isfinite(trade_r) else np.nan,
                "holding_minutes": float(holding_minutes) if np.isfinite(holding_minutes) else np.nan,
                "entry_bucket_start_hour": int(bucket_start_hour),
                "entry_bucket_label": str(bucket_label),
            }
        )

    trade_df = pd.DataFrame(rows)
    trade_df.attrs["attempted_count"] = int(attempted_count)
    trade_df.attrs["resolved_count"] = int(resolved_count)
    trade_df.attrs["censored_count"] = int(censored_count)

    if trade_df.empty:
        return trade_df

    return trade_df.dropna(subset=["trade_r"]).sort_values(["time_ns", "side", "anchor_idx"]).reset_index(drop=True)
