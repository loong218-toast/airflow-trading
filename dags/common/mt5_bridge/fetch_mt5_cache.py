# fetch_mt5_cache.py

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd

DAGS_ROOT = Path(__file__).resolve().parents[2]   # .../dags
REPO_ROOT = DAGS_ROOT.parent                       # .../airflow-trading

# For local runs, default the data root to the repo root.
# In Docker/Airflow, set AIRFLOW_TRADING_ROOT in the environment and it will win.
os.environ.setdefault("AIRFLOW_TRADING_ROOT", str(REPO_ROOT))

if str(DAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(DAGS_ROOT))

from research.expectancy_config import (
    CACHE_FILE,
    MT5_CHUNK_DAYS,
    MT5_SYMBOL,
)

LOOKBACK_DAYS = 365
TIMEFRAME = "M5"


# =========================
# Time helpers
# =========================

def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def utc_days_ago(days: int) -> datetime:
    return utc_now() - timedelta(days=days)


# =========================
# MT5 helpers
# =========================

def init_mt5():
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    return mt5


def shutdown_mt5(mt5) -> None:
    try:
        mt5.shutdown()
    except Exception:
        pass


# =========================
# Data fetching
# =========================

def fetch_mt5_rates_chunk(
    mt5,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    rates = mt5.copy_rates_range(
        symbol,
        mt5.TIMEFRAME_M5,
        start_dt,
        end_dt,
    )

    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rates)


def fetch_mt5_rates_range_chunked(
    mt5,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    chunk_days: int,
) -> pd.DataFrame:
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol_select failed: {mt5.last_error()}")

    chunks: List[pd.DataFrame] = []
    cur = start_dt

    while cur < end_dt:
        chunk_end = min(cur + timedelta(days=chunk_days), end_dt)

        df = fetch_mt5_rates_chunk(mt5, symbol, cur, chunk_end)
        if not df.empty:
            chunks.append(df)

        cur = chunk_end

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    return normalize_mt5_rates(df)


# =========================
# Normalization
# =========================

def normalize_mt5_rates(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        return pd.DataFrame()

    df = df.drop_duplicates(subset=["time"]).sort_values("time")

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_ns"] = df["time"].astype("int64")

    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cols = ["time_ns", "open", "high", "low", "close"]

    if any(c not in df.columns for c in cols):
        return pd.DataFrame()

    return df[cols].dropna().reset_index(drop=True)


# =========================
# Persistence
# =========================

def save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# =========================
# Orchestration
# =========================

def build_mt5_cache() -> pd.DataFrame:
    start_dt = utc_days_ago(LOOKBACK_DAYS)
    end_dt = utc_now()

    mt5 = init_mt5()
    try:
        df = fetch_mt5_rates_range_chunked(
            mt5=mt5,
            symbol=MT5_SYMBOL,
            start_dt=start_dt,
            end_dt=end_dt,
            chunk_days=MT5_CHUNK_DAYS,
        )
    finally:
        shutdown_mt5(mt5)

    if df.empty:
        raise RuntimeError("No MT5 data returned")

    save_cache(df, CACHE_FILE)
    return df


def main() -> None:
    df = build_mt5_cache()
    print(f"saved {len(df):,} rows -> {CACHE_FILE}")


if __name__ == "__main__":
    main()