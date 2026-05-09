# bootstrap_expectancy_analysis.py

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl

# -----------------------------------------------------------------------------
# Make paths work cleanly across both Windows Local and Airflow Docker Environments
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve()

# 1. Explicit check for your Airflow Docker setup structure
if Path("/opt/airflow/airflow-trading").exists():
    PROJECT_ROOT = Path("/opt/airflow/airflow-trading")
elif Path("/opt/airflow").exists():
    PROJECT_ROOT = Path("/opt/airflow")
else:
    # 2. Windows local execution structural fallback
    PROJECT_ROOT = next(
        (p for p in HERE.parents if (p / "dags").exists() or (p / "data_lake").exists()),
        HERE.parents[2],
    )

DAGS_ROOT = PROJECT_ROOT / "dags" if (PROJECT_ROOT / "dags").exists() else Path("/opt/airflow/dags")
if str(DAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(DAGS_ROOT))

from research.bootstrap_core import (  # noqa: E402
    bootstrap_mean_samples,
    describe_numeric_series,
    safe_name,
    save_histogram,
)

# =========================
# MANUAL SETTINGS
# =========================

INSTRUMENT = "AUDJPY"
PAIR = "AUDJPY"
MT5_SYMBOL = "AUDJPY"

TARGET_PCT = 0.15
SL_PCT = 0.20
HORIZON_HOURS = 24

BASE_DIR = PROJECT_ROOT
CACHE_DIR = BASE_DIR / "data_lake" / "cache"
OUTPUT_DIR = BASE_DIR / "data_lake" / "Saved_results" / "expectancy_checks" / INSTRUMENT / "bootstrap"
CACHE_FILE = CACHE_DIR / f"{MT5_SYMBOL}_m5_cache.csv"

BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_BLOCK_SIZE = 8
BOOTSTRAP_SEED = 14121212

RISK_PCT = 0.005
USE_MA_FILTER = False
MA_TYPE = "ema"
MA_PERIOD_BARS = 32

RANDOMIZE_ENTRY_PRICE = True
ENTRY_NUDGE_MAX_FRACTION = 0.12
ENTRY_NUDGE_CLIP_TO_CANDLE = True

USE_ENTRY_BUCKET_HOURS = False
ENTRY_BUCKET_HOURS = 4

SAVE_TRADE_UNIVERSE_CSV = False
SAVE_BOOTSTRAP_SAMPLES_CSV = False

MYT = pd.Timestamp.now(tz="UTC").tz_convert("Asia/Kuala_Lumpur").tz

logger = logging.getLogger(__name__)

MINUTE_NS = 60 * 1_000_000_000
HOUR_NS = 60 * MINUTE_NS


# =========================
# Helpers
# =========================

def _bucket_start_hour_from_time_ns_myt(time_ns: int, bucket_hours: int = ENTRY_BUCKET_HOURS) -> int:
    ts_utc = pd.Timestamp(int(time_ns), unit="ns", tz="UTC")
    ts_myt = ts_utc.tz_convert("Asia/Kuala_Lumpur")
    return int((ts_myt.hour // bucket_hours) * bucket_hours)


def _bucket_label_from_start_hour(start_hour: int, bucket_hours: int = ENTRY_BUCKET_HOURS) -> str:
    return f"{start_hour:02d}:00-{start_hour + bucket_hours:02d}:00 MYT"


def _rng_for_anchor(anchor_idx: int) -> np.random.Generator:
    return np.random.default_rng(BOOTSTRAP_SEED + (anchor_idx + 1) * 1_000_003)


def sample_entry_price_nudged(
    open_px: float,
    high_px: float,
    low_px: float,
    close_px: float,
    anchor_idx: int,
) -> float:
    if not RANDOMIZE_ENTRY_PRICE:
        return float(close_px)

    if not all(np.isfinite(x) for x in (open_px, high_px, low_px, close_px)):
        return np.nan

    lo, hi = float(low_px), float(high_px)
    if hi < lo:
        lo, hi = hi, lo

    candle_range = max(hi - lo, 0.0)
    if candle_range <= 0.0:
        return float(close_px)

    rng = _rng_for_anchor(anchor_idx)
    max_nudge = candle_range * float(ENTRY_NUDGE_MAX_FRACTION)
    entry = float(close_px) + float(rng.uniform(-max_nudge, max_nudge))

    if ENTRY_NUDGE_CLIP_TO_CANDLE:
        entry = min(max(entry, lo), hi)

    return float(entry)


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


def _ma_pass(side: int, price: float, ma_value: float) -> bool:
    if not np.isfinite(ma_value):
        return True
    return price < ma_value if side == 1 else price > ma_value


def _replace_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def _load_cache_df(cache_file: Path = CACHE_FILE) -> pd.DataFrame:
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

    df = (
        df.dropna(subset=needed)
        .drop_duplicates(subset=["time_ns"])
        .sort_values("time_ns")
        .reset_index(drop=True)
    )
    return df


def _moving_block_bootstrap_sample(
    values: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = int(values.shape[0])
    if n <= 0:
        return np.asarray([], dtype=np.float64)

    block_size = max(1, min(int(block_size), n))
    if n == 1:
        return values.copy()

    starts = np.arange(0, n - block_size + 1, dtype=np.int64)
    if starts.size == 0:
        return values.copy()

    out: List[float] = []
    while len(out) < n:
        start = int(rng.choice(starts))
        block = values[start : start + block_size]
        out.extend(block.tolist())

    return np.asarray(out[:n], dtype=np.float64)


# =========================
# Trade universe construction
# =========================

def build_trade_universe(
    df: pd.DataFrame,
    target_pct: float = TARGET_PCT,
    sl_pct: float = SL_PCT,
    horizon_hours: int = HORIZON_HOURS,
    use_ma_filter: bool = USE_MA_FILTER,
) -> pd.DataFrame:
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
    
    # --- RANDOM SELECTION LOGIC ---
    # 1. Group indices by Hour (Unix timestamp / HOUR_NS)
    # 2. Pick 2 random indices per hour
    # 3. Assign 1 (Buy) or -1 (Sell) randomly to each
    rng_selection = np.random.default_rng(BOOTSTRAP_SEED)
    hour_groups = time_ns // HOUR_NS
    unique_hours = np.unique(hour_groups)
    
    # dictionary: { anchor_idx: side }
    selected_indices_map = {}
    
    for h_val in unique_hours:
        # Indices in this hour (excluding the very last bar of the dataset)
        indices_in_hour = np.where((hour_groups == h_val) & (np.arange(n) < n - 1))[0]
        
        if len(indices_in_hour) == 0:
            continue
            
        # Pick up to 2 indices
        n_to_pick = min(2, len(indices_in_hour))
        chosen_idxs = rng_selection.choice(indices_in_hour, size=n_to_pick, replace=False)
        
        for idx in chosen_idxs:
            # Randomly choose Buy (1) or Sell (-1)
            side = rng_selection.choice([1, -1])
            selected_indices_map[int(idx)] = int(side)
    # ------------------------------

    horizon_end = np.searchsorted(time_ns, time_ns + np.int64(horizon_hours * HOUR_NS), side="right")
    ma_values = _compute_ma_values(close)

    rows: List[Dict[str, Any]] = []
    attempted_count = 0
    resolved_count = 0
    censored_count = 0
    target_ratio = float(target_pct) / 100.0
    sl_ratio = float(sl_pct) / 100.0

    # Only iterate through the indices we randomly selected
    for anchor_idx in sorted(selected_indices_map.keys()):
        side = selected_indices_map[anchor_idx]
        
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

        # MA Filter check
        if use_ma_filter and not _ma_pass(side, anchor_price, ma_value):
            continue

        bucket_start_hour = (
            _bucket_start_hour_from_time_ns_myt(int(time_ns[anchor_idx]), ENTRY_BUCKET_HOURS)
            if USE_ENTRY_BUCKET_HOURS
            else 0
        )
        bucket_label = (
            _bucket_label_from_start_hour(bucket_start_hour, ENTRY_BUCKET_HOURS)
            if USE_ENTRY_BUCKET_HOURS
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

        rows.append(
            {
                "instrument": INSTRUMENT,
                "source_symbol": MT5_SYMBOL,
                "pair": PAIR,
                "anchor_idx": int(anchor_idx),
                "time_ns": int(time_ns[anchor_idx]),
                "side": int(side),
                "entry_price": float(anchor_price),
                "target_pct": float(target_pct),
                "sl_pct": float(sl_pct),
                "horizon_hours": int(horizon_hours),
                "use_ma_filter": bool(use_ma_filter),
                "ma_value": float(ma_value) if np.isfinite(ma_value) else np.nan,
                "exit_type": str(exit_type),
                "trade_r": float(trade_r) if np.isfinite(trade_r) else np.nan,
                "trade_expectancy_pct": float(trade_r * RISK_PCT * 100.0) if np.isfinite(trade_r) else np.nan,
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

    trade_df = trade_df.dropna(subset=["trade_r"]).reset_index(drop=True)
    trade_df = trade_df.sort_values(["time_ns", "side", "anchor_idx"]).reset_index(drop=True)
    return trade_df


# =========================
# Bootstrap summary / plot
# =========================

def build_summary(
    trade_df: pd.DataFrame,
    bootstrap_samples_r: np.ndarray,
) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()

    attempted_count = int(trade_df.attrs.get("attempted_count", len(trade_df)))
    resolved_count = int(trade_df.attrs.get("resolved_count", len(trade_df)))
    censored_count = int(trade_df.attrs.get("censored_count", max(0, attempted_count - resolved_count)))

    resolved_rate_pct = float((resolved_count / attempted_count) * 100.0) if attempted_count > 0 else np.nan
    censored_rate_pct = float((censored_count / attempted_count) * 100.0) if attempted_count > 0 else np.nan
    bootstrap_used_count = int(len(trade_df))
    bootstrap_used_pct = float((bootstrap_used_count / attempted_count) * 100.0) if attempted_count > 0 else np.nan

    observed_mean_r = float(trade_df["trade_r"].mean())
    observed_median_r = float(trade_df["trade_r"].median())
    observed_std_r = float(trade_df["trade_r"].std(ddof=1)) if len(trade_df) > 1 else 0.0

    win_rate_pct = float((trade_df["trade_r"] > 0).mean() * 100.0)
    loss_rate_pct = float((trade_df["trade_r"] < 0).mean() * 100.0)
    flat_rate_pct = float((trade_df["trade_r"] == 0).mean() * 100.0)

    if bootstrap_samples_r.size > 0:
        b_mean = float(np.mean(bootstrap_samples_r))
        b_median = float(np.median(bootstrap_samples_r))
        b_std = float(np.std(bootstrap_samples_r, ddof=1)) if bootstrap_samples_r.size > 1 else 0.0
        b_p05 = float(np.percentile(bootstrap_samples_r, 5))
        b_p10 = float(np.percentile(bootstrap_samples_r, 10))
        b_p25 = float(np.percentile(bootstrap_samples_r, 25))
        b_p75 = float(np.percentile(bootstrap_samples_r, 75))
        b_p90 = float(np.percentile(bootstrap_samples_r, 90))
        b_p95 = float(np.percentile(bootstrap_samples_r, 95))
        prob_negative_pct = float((bootstrap_samples_r < 0).mean() * 100.0)
    else:
        b_mean = b_median = b_std = b_p05 = b_p10 = b_p25 = b_p75 = b_p90 = b_p95 = prob_negative_pct = np.nan

    row = {
        "instrument": INSTRUMENT,
        "source_symbol": MT5_SYMBOL,
        "pair": PAIR,
        "target_pct": float(TARGET_PCT),
        "sl_pct": float(SL_PCT),
        "horizon_hours": int(HORIZON_HOURS),
        "risk_pct": float(RISK_PCT),
        "use_ma_filter": bool(USE_MA_FILTER),
        "randomize_entry_price": bool(RANDOMIZE_ENTRY_PRICE),
        "use_entry_bucket_hours": bool(USE_ENTRY_BUCKET_HOURS),
        "n_trades": int(len(trade_df)),
        "attempted_count": attempted_count,
        "resolved_count": resolved_count,
        "censored_count": censored_count,
        "resolved_rate_pct": resolved_rate_pct,
        "censored_rate_pct": censored_rate_pct,
        "bootstrap_used_count": bootstrap_used_count,
        "bootstrap_used_pct": bootstrap_used_pct,
        "observed_mean_r": observed_mean_r,
        "observed_median_r": observed_median_r,
        "observed_std_r": observed_std_r,
        "observed_expectancy_pct": observed_mean_r * RISK_PCT * 100.0,
        "observed_median_expectancy_pct": observed_median_r * RISK_PCT * 100.0,
        "win_rate_pct": win_rate_pct,
        "loss_rate_pct": loss_rate_pct,
        "flat_rate_pct": flat_rate_pct,
        "bootstrap_rounds": int(BOOTSTRAP_ROUNDS),
        "bootstrap_block_size": int(BOOTSTRAP_BLOCK_SIZE),
        "bootstrap_mean_r": b_mean,
        "bootstrap_median_r": b_median,
        "bootstrap_std_r": b_std,
        "bootstrap_p05_r": b_p05,
        "bootstrap_p10_r": b_p10,
        "bootstrap_p25_r": b_p25,
        "bootstrap_p75_r": b_p75,
        "bootstrap_p90_r": b_p90,
        "bootstrap_p95_r": b_p95,
        "bootstrap_prob_negative_pct": prob_negative_pct,
        "bootstrap_mean_expectancy_pct": b_mean * RISK_PCT * 100.0 if np.isfinite(b_mean) else np.nan,
        "bootstrap_median_expectancy_pct": b_median * RISK_PCT * 100.0 if np.isfinite(b_median) else np.nan,
        "bootstrap_p05_expectancy_pct": b_p05 * RISK_PCT * 100.0 if np.isfinite(b_p05) else np.nan,
        "bootstrap_p25_expectancy_pct": b_p25 * RISK_PCT * 100.0 if np.isfinite(b_p25) else np.nan,
        "bootstrap_p75_expectancy_pct": b_p75 * RISK_PCT * 100.0 if np.isfinite(b_p75) else np.nan,
        "bootstrap_p95_expectancy_pct": b_p95 * RISK_PCT * 100.0 if np.isfinite(b_p95) else np.nan,
    }
    return pd.DataFrame([row])


def run_bootstrap_expectancy_scan() -> Dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_cache_df(CACHE_FILE)
    if df.empty:
        raise RuntimeError(f"No cache data available at {CACHE_FILE}")

    logger.info("Using instrument: %s", INSTRUMENT)
    logger.info("Cache file: %s", CACHE_FILE)
    logger.info("Selected setup: TP=%.4f%% SL=%.4f%% Horizon=%dh", TARGET_PCT, SL_PCT, HORIZON_HOURS)

    trade_df = build_trade_universe(
        df=df,
        target_pct=TARGET_PCT,
        sl_pct=SL_PCT,
        horizon_hours=HORIZON_HOURS,
        use_ma_filter=USE_MA_FILTER,
    )
    if trade_df.empty:
        raise RuntimeError("No trades were generated for the selected configuration.")

    if SAVE_TRADE_UNIVERSE_CSV:
        trade_df.to_csv(OUTPUT_DIR / "trade_universe.csv", index=False)

    trade_r = trade_df["trade_r"].to_numpy(dtype=np.float64, copy=False)
    bootstrap_samples_r = bootstrap_mean_samples(
        values=trade_r,
        n_bootstrap=BOOTSTRAP_ROUNDS,
        block_size=BOOTSTRAP_BLOCK_SIZE,
        seed=BOOTSTRAP_SEED,
    )

    summary_df = build_summary(trade_df, bootstrap_samples_r)
    if summary_df.empty:
        raise RuntimeError("No summary produced.")

    prefix = (
        f"{safe_name(INSTRUMENT)}_"
        f"tp_{str(TARGET_PCT).replace('.', 'p')}_"
        f"sl_{str(SL_PCT).replace('.', 'p')}_"
        f"h_{int(HORIZON_HOURS)}_"
        f"{'ma' if USE_MA_FILTER else 'no_ma'}"
    )

    summary_file = OUTPUT_DIR / f"{prefix}_summary.csv"
    plot_file = OUTPUT_DIR / f"{prefix}_hist.png"

    _replace_file(summary_file)
    summary_df.to_csv(summary_file, index=False)

    observed = float(summary_df.iloc[0]["observed_mean_r"])
    b_median = float(summary_df.iloc[0]["bootstrap_median_r"])
    b_p05 = float(summary_df.iloc[0]["bootstrap_p05_r"])
    b_p95 = float(summary_df.iloc[0]["bootstrap_p95_r"])

    save_histogram(
        values=bootstrap_samples_r,
        out_png=plot_file,
        title=(
            f"{INSTRUMENT} | TP {TARGET_PCT:.4g}% | SL {SL_PCT:.4g}% | "
            f"{int(HORIZON_HOURS)}h | bootstrap expectancy"
        ),
        xlabel="Bootstrap sample mean expectancy in R",
        observed=observed,
        extra_vlines=[b_median, b_p05, b_p95],
        bins=40,
    )

    if SAVE_BOOTSTRAP_SAMPLES_CSV:
        pd.DataFrame({"bootstrap_sample_mean_r": bootstrap_samples_r}).to_csv(
            OUTPUT_DIR / f"{prefix}_bootstrap_samples.csv",
            index=False,
        )

    logger.info("Saved summary: %s", summary_file)
    logger.info("Saved plot: %s", plot_file)

    return {
        "summary_file": str(summary_file),
        "plot_file": str(plot_file),
        "cache_file": str(CACHE_FILE),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_bootstrap_expectancy_scan()