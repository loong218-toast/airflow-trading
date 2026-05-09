# expectancy_config.py

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

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

LOOKBACK_DAYS = 100
ENTRY_BUCKET_HOURS = 4
USE_ENTRY_BUCKET_HOURS = False
RISK_PCT = 0.005

USE_MA_FILTER = False
MA_TYPE = "ema"  # ema or sma
MA_PERIOD_BARS = 96  # 32 MA 96 = 15m 384 = 1h

# After-loss cooldown / stop logic:
# - trigger_count list: how many consecutive SLs before the rule activates
# - skip_trades list: how many future signals to skip after activation
#   - use None to stop trading for the rest of the sample after the trigger
USE_AFTER_LOSS_FILTER = False

# Example sweep:
#   AFTER_LOSS_TRIGGER_COUNT_LIST = [2, 3, 4]
#   AFTER_LOSS_SKIP_TRADES_LIST = [None, 5, 10]
AFTER_LOSS_TRIGGER_COUNT_LIST: List[int] = [1]
AFTER_LOSS_SKIP_TRADES_LIST: List[Optional[int]] = [1]

RANDOMIZE_ENTRY_PRICE = True
RANDOM_ENTRY_SEED = 11121212
ENTRY_NUDGE_MAX_FRACTION = 0.12
ENTRY_NUDGE_CLIP_TO_CANDLE = True

#SIMULATION_MODES = ["overlapping", "sequential_flip", "sequential_random"]
SIMULATION_MODES = ["sequential_random"]
SEQUENTIAL_SWITCH_ON_LOSS = True                  # If true, prefers opposite side after a loss

TRADES_PER_HOUR = 2

if 24 % ENTRY_BUCKET_HOURS != 0:    
    raise ValueError("ENTRY_BUCKET_HOURS must divide 24 exactly.")

MYT = timezone(timedelta(hours=8), name="MYT")


def _default_utc_window(days_back: int = LOOKBACK_DAYS) -> tuple[str, str]:
    end_dt = datetime.now(timezone.utc).replace(microsecond=0)
    start_dt = end_dt - timedelta(days=days_back)
    return (
        start_dt.isoformat().replace("+00:00", "Z"),
        end_dt.isoformat().replace("+00:00", "Z"),
    )


GRID_START_DATE, GRID_END_DATE = _default_utc_window(LOOKBACK_DAYS)

INSTRUMENT = "BTC"
INSTRUMENT_CONFIG = {
    "BTC": {
        "pair": "BTC",
        "mt5_symbol": "BTCUSD",
        "tp_range": {"min": 0.7, "max": 0.7, "step": 0.1},
        "sl_range": {"min": 1.4, "max": 1.4, "step": 0.1},
        "horizon_hours_list": [24],
    },
    "UK100": {
        "pair": "UK100",
        "mt5_symbol": "UK100",
        "tp_range": {"min": 0.05, "max": 0.22, "step": 0.01},
        "sl_range": {"min": 0.05, "max": 0.22, "step": 0.01},
        "horizon_hours_list": [3],
    },
    "AUDJPY": {
        "pair": "AUDJPY",
        "mt5_symbol": "AUDJPY",
        "tp_range": {"min": 0.15, "max": 0.15, "step": 0.01},
        "sl_range": {"min": 0.20, "max": 0.20, "step": 0.01},
        "horizon_hours_list": [8],
    },
    "USDCHF": {
        "pair": "USDCHF",
        "mt5_symbol": "USDCHF",
        "tp_range": {"min": 0.15, "max": 0.40, "step": 0.05},
        "sl_range": {"min": 0.15, "max": 0.6, "step": 0.05},
        "horizon_hours_list": [72],
    },
    "XAUUSD": {
        "pair": "XAUUSD",
        "mt5_symbol": "XAUUSD",
        "tp_range": {"min": 0.1, "max": 0.8, "step": 0.1},
        "sl_range": {"min": 0.1, "max": 0.8, "step": 0.1},
        "horizon_hours_list": [24],
    },
}

if INSTRUMENT not in INSTRUMENT_CONFIG:
    raise ValueError(f"Unknown INSTRUMENT={INSTRUMENT!r}. Add it to INSTRUMENT_CONFIG.")

CFG = INSTRUMENT_CONFIG[INSTRUMENT]
PAIR = CFG["pair"]
MT5_SYMBOL = CFG["mt5_symbol"]
HORIZON_HOURS_LIST = CFG["horizon_hours_list"]

MT5_CHUNK_DAYS = 90

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


def _bucket_start_hour_from_time_ns_myt(time_ns: int, bucket_hours: int = ENTRY_BUCKET_HOURS) -> int:
    ts_utc = pd.Timestamp(int(time_ns), unit="ns", tz="UTC")
    ts_myt = ts_utc.tz_convert(MYT)
    return int((ts_myt.hour // bucket_hours) * bucket_hours)


def _bucket_label_from_start_hour(start_hour: int, bucket_hours: int = ENTRY_BUCKET_HOURS) -> str:
    return f"{start_hour:02d}:00-{start_hour + bucket_hours:02d}:00 MYT"


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
) -> float:
    if sl_pct <= 0:
        raise ValueError("sl_pct must be > 0")

    tp_rate = target_first_rate_pct / 100.0
    sl_rate = sl_first_rate_pct / 100.0
    rr_multiple = float(target_pct) / float(sl_pct)
    return (tp_rate * (risk_pct * rr_multiple * 100.0)) - (sl_rate * (risk_pct * 100.0))


def _market_tag(instrument: str, pair: str, symbol: str) -> str:
    for value in (pair, instrument, symbol):
        if value:
            return _safe_name(str(value))
    return "market"


def make_output_file(instrument: str, pair: str, symbol: str, bucket_hours: Optional[int], use_bucket_hours: bool) -> Path:
    market = _market_tag(instrument, pair, symbol)
    bucket_part = f"{int(bucket_hours)}h_myt_buckets" if use_bucket_hours else "no_myt_buckets"
    return OUTPUT_DIR / f"expectancy_scan_{market}_{bucket_part}.csv"


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
    ]
