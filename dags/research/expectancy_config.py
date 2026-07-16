# expectancy_config.py

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

PROJECT_ROOT = Path(os.getenv("AIRFLOW_TRADING_ROOT", "/opt/airflow/airflow-trading"))
BASE_DIR = PROJECT_ROOT

if load_dotenv is not None:
    load_dotenv()

# research/expectancy_config.py

LOOKBACK_DAYS = 90
LOOKBACK_DAYS_LIST = [15, 30, 90]

def get_lookback_days_list(values: Optional[Sequence[int]] = None) -> List[int]:
    source = LOOKBACK_DAYS_LIST if values is None else values

    out: List[int] = []
    seen = set()
    for value in source:
        v = int(value)
        if v <= 0:
            raise ValueError("LOOKBACK_DAYS values must be > 0")
        if v in seen:
            continue
        seen.add(v)
        out.append(v)

    return out or [LOOKBACK_DAYS]

RISK_PCT = 0.005

MIX_BUY_SELL = False  # True = one mixed scan, False = split into buy-only and sell-only outputs

USE_MA_FILTER = False
MA_TYPE = "ema"  # ema or sma
MA_PERIOD_BARS = 96  # 32 MA 96 = 15m 384 = 1h

# After-loss cooldown / stop logic:
# - trigger_count list: how many consecutive SLs before the rule activates
# - skip_trades list: how many future signals to skip after activation
# - use None to stop trading for the rest of the sample after the trigger
USE_AFTER_LOSS_FILTER = False

# Example sweep:
# AFTER_LOSS_TRIGGER_COUNT_LIST = [2, 3, 4]
# AFTER_LOSS_SKIP_TRADES_LIST = [None, 5, 10]
AFTER_LOSS_TRIGGER_COUNT_LIST: List[int] = [1]
AFTER_LOSS_SKIP_TRADES_LIST: List[Optional[int]] = [1]

RANDOMIZE_ENTRY_PRICE = True
RANDOM_ENTRY_SEED = 11121212
ENTRY_NUDGE_MAX_FRACTION = 0.1
ENTRY_NUDGE_CLIP_TO_CANDLE = True

# Simulation modes:
# - overlapping: evaluate every eligible entry independently
# - overlapping_random: same as overlapping, but entries are randomly sampled per hour using TRADES_PER_HOUR
# - sequential_flip: take one trade at a time and favor the opposite side after a loss
# - sequential_random: take one trade at a time and use a daily random side bias

SIMULATION_MODES = ["overlapping_random"]
#SIMULATION_MODES = ["sequential_random"]
# SIMULATION_MODES = ["overlapping", "sequential_flip", "sequential_random"]
SEQUENTIAL_SWITCH_ON_LOSS = False  # If true, prefers opposite side after a loss

TRADES_PER_HOUR = 2

# Entry window filter in Malaysia time.
# Bars inside this window are eligible to become entries.
USE_ENTRY_TIME_WINDOW = False
ENTRY_WINDOW_START_HOUR_MYT = 15
ENTRY_WINDOW_END_HOUR_MYT = 23

def get_horizon_label(horizon_hours_list: list[int]) -> str:
    horizons = sorted({int(h) for h in horizon_hours_list if int(h) > 0})
    if not horizons:
        return "horizonNA"
    if len(horizons) == 1:
        return f"horizon{horizons[0]}"
    return "horizon" + "-".join(str(h) for h in horizons)

def _default_utc_window(days_back: Optional[int] = None) -> tuple[str, str]:
    days_back = LOOKBACK_DAYS if days_back is None else int(days_back)
    end_dt = datetime.now(timezone.utc).replace(microsecond=0)
    start_dt = end_dt - timedelta(days=days_back)
    return (
        start_dt.isoformat().replace("+00:00", "Z"),
        end_dt.isoformat().replace("+00:00", "Z"),
    )

def get_entry_window_label(
    use_entry_time_window: bool = USE_ENTRY_TIME_WINDOW,
    start_hour: int = ENTRY_WINDOW_START_HOUR_MYT,
    end_hour: int = ENTRY_WINDOW_END_HOUR_MYT,
) -> str:
    if not use_entry_time_window:
        return "24H"
    return f"{start_hour:02d}:00-{end_hour:02d}:00 MYT"

ENTRY_WINDOW_LABEL = get_entry_window_label()

MYT = timezone(timedelta(hours=8), name="MYT")

USE_MAE_MFE_STATS = False

USE_DAILY_SIDE_BIAS = False
DAILY_BIAS_USE_MYT_DATE = True

GRID_START_DATE, GRID_END_DATE = _default_utc_window(LOOKBACK_DAYS)

INSTRUMENT = "UK100"  # BTC, UK100, AUDJPY, USDCHF, XAUUSD
INSTRUMENT_CONFIG = {
    "BTC": {
        "pair": "BTC",
        "mt5_symbol": "BTCUSD",
        "spread_pct": 0.001,
        "tp_range": {"min": 0.6, "max": 4.0, "step": 0.2},
        "sl_range": {"min": 0.6, "max": 4.0, "step": 0.2},
        "horizon_hours_list": [120],
    },
    "UK100": {
        "pair": "UK100",
        "mt5_symbol": "FTSE100",
        "spread_pct": 0.007,
        "tp_range": {"min": 0.05, "max": 0.70, "step": 0.02},
        "sl_range": {"min": 0.05, "max": 0.70, "step": 0.02},
        "horizon_hours_list": [12, 48],
    },
    "AUDJPY": {
        "pair": "AUDJPY",
        "mt5_symbol": "AUDJPY",
        "spread_pct": 0.007,
        "tp_range": {"min": 0.05, "max": 1.10, "step": 0.05},
        "sl_range": {"min": 0.05, "max": 1.10, "step": 0.05},
        "horizon_hours_list": [48],
    },
    "USDCHF": {
        "pair": "USDCHF",
        "mt5_symbol": "USDCHF",
        "spread_pct": 0.005,
        "tp_range": {"min": 0.05, "max": 0.50, "step": 0.02},
        "sl_range": {"min": 0.05, "max": 0.50, "step": 0.02},
        "horizon_hours_list": [48],
    },
    "XAUUSD": {
        "pair": "XAUUSD",
        "mt5_symbol": "XAUUSD",
        "spread_pct": 0.02,
        "tp_range": {"min": 0.2, "max": 4.2, "step": 0.1},
        "sl_range": {"min": 0.2, "max": 4.2, "step": 0.1},
        "horizon_hours_list": [48],
    },
    "EURJPY": {
        "pair": "EURJPY",
        "mt5_symbol": "EURJPY",
        "spread_pct": 0.01,
        "tp_range": {"min": 0.05, "max": 0.90, "step": 0.05},
        "sl_range": {"min": 0.05, "max": 0.90, "step": 0.05},
        "horizon_hours_list": [48],
    },
}

if INSTRUMENT not in INSTRUMENT_CONFIG:
    raise ValueError(f"Unknown INSTRUMENT={INSTRUMENT!r}. Add it to INSTRUMENT_CONFIG.")

CFG = INSTRUMENT_CONFIG[INSTRUMENT]
PAIR = CFG["pair"]
MT5_SYMBOL = CFG["mt5_symbol"]
SPREAD_PCT = float(CFG.get("spread_pct", 0.0))
HORIZON_HOURS_LIST = CFG["horizon_hours_list"]
HORIZON_LABEL = get_horizon_label(HORIZON_HOURS_LIST)

MT5_CHUNK_DAYS = 30

MT5_PATH = os.getenv("MT5_PATH")
MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_TIMEOUT = int(os.getenv("MT5_TIMEOUT", "60000"))
MT5_PORTABLE = os.getenv("MT5_PORTABLE", "false").strip().lower() in {"1", "true", "yes", "y", "on"}

OUTPUT_BASE_DIR = BASE_DIR / "data_lake" / "Saved_results" / "expectancy_checks"
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = OUTPUT_BASE_DIR / INSTRUMENT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(os.getenv("CACHE_DIR", str(BASE_DIR / "data_lake" / "cache")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = CACHE_DIR / f"{MT5_SYMBOL}_m5_cache.csv"
SUMMARY_FILE = OUTPUT_DIR / f"expectancy_scan_{INSTRUMENT}.csv"


def _float_range_inclusive(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if stop < start:
        raise ValueError("stop must be >= start")

    values: List[float] = []
    cur = float(start)
    eps = step / 10_000.0

    while cur <= stop + eps:
        values.append(round(cur, 10))
        cur += step

    return values


TARGET_PCT_LIST = _float_range_inclusive(CFG["tp_range"]["min"], CFG["tp_range"]["max"], CFG["tp_range"]["step"])
SL_PCT_LIST = _float_range_inclusive(CFG["sl_range"]["min"], CFG["sl_range"]["max"], CFG["sl_range"]["step"])

def _parse_utc_dt(dt_in: Any) -> Optional[datetime]:
    if dt_in is None:
        return None

    if isinstance(dt_in, datetime):
        dt = dt_in
    elif isinstance(dt_in, str):
        s = dt_in[:-1] if dt_in.endswith("Z") else dt_in
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            try:
                dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
            except Exception:
                return None
    else:
        try:
            dt = pd.Timestamp(dt_in).to_pydatetime()
        except Exception:
            return None

    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _safe_name(value: str) -> str:
    return value.strip().replace(" ", "_").replace("/", "_").replace(".", "_").replace(":", "_")



def _mean_median(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr))


def _pct_from_ratio(r: Optional[float]) -> Optional[float]:
    return None if r is None else float(r * 100.0)


def _net_expectancy_risk_pct(
    target_first_rate_pct: float,
    target_pct: float,
    sl_first_rate_pct: float,
    sl_pct: float,
    risk_pct: float = RISK_PCT,
    spread_pct: float = 0.0,
) -> float:
    if sl_pct <= 0:
        raise ValueError("sl_pct must be > 0")

    tp_rate = target_first_rate_pct / 100.0
    sl_rate = sl_first_rate_pct / 100.0
    rr_multiple = float(target_pct) / float(sl_pct)
    spread_cost_pct = (float(spread_pct) / float(sl_pct)) * (risk_pct * 100.0) if spread_pct > 0 else 0.0
    return (tp_rate * ((risk_pct * rr_multiple * 100.0) - spread_cost_pct)) - (sl_rate * ((risk_pct * 100.0) + spread_cost_pct))


def _market_tag(instrument: str, pair: str, symbol: str) -> str:
    for value in (pair, instrument, symbol):
        if value:
            return _safe_name(str(value))
    return "market"


def make_output_file(
    instrument: str,
    pair: str,
    symbol: str,
    entry_window_label: Optional[str] = None,
) -> Path:
    market = _market_tag(instrument, pair, symbol)
    window_part = _safe_name(entry_window_label or ENTRY_WINDOW_LABEL)
    return OUTPUT_DIR / f"expectancy_scan_{market}_{window_part}.csv"


def _rng_for_anchor(anchor_idx: int) -> np.random.Generator:
    return np.random.default_rng(RANDOM_ENTRY_SEED + (anchor_idx + 1) * 1_000_003)


def sample_entry_price_nudged(open_px: float, high_px: float, low_px: float, close_px: float, anchor_idx: int) -> float:
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


def get_columns_to_remove() -> list[str]:
    return [
        "source_symbol",
        "pair",
        "net_expectancy_risk_pct",
        "horizon_eligible_count",
        "horizon_eligible_rate_pct",
        "after_loss_trigger_count",
        "after_loss_skip_trades",
        "anchors_total",
        "target_first_count",
        "sl_first_count",
        "censored_count",
        "target_first_then_sl_count",
        "forced_exit_r_positive_count",
    ]
