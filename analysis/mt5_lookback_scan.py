# mt5_lookback_scan.py

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("mt5_lookback_scan")

# --- PATH CONFIGURATION ---
PROJECT_ROOT = Path("/opt/airflow/airflow-trading") if Path("/opt/airflow").exists() else Path("C:/Users/Owner/airflow-trading")
CACHE_DIR = PROJECT_ROOT / "data_lake" / "cache"
METADATA_DIR = PROJECT_ROOT / "dags" / "research" / "bootstrap"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME = mt5.TIMEFRAME_M5
TIMEFRAME_NAME = "m5"
TOTAL_TARGET_BARS = 110_000 
CHUNK_SIZE = 50_000 

INSTRUMENT_CONFIG = {
    "BTC": {"pair": "BTC", "mt5_symbol": "BTCUSD"},
    "UK100": {"pair": "UK100", "mt5_symbol": "UK100"},
    "AUDJPY": {"pair": "AUDJPY", "mt5_symbol": "AUDJPY"},
    "USDCHF": {"pair": "USDCHF", "mt5_symbol": "USDCHF"},
    "XAUUSD": {"pair": "XAUUSD", "mt5_symbol": "XAUUSD"},
}

def _safe_dt_from_unix(ts: int) -> str:
    if ts is None or ts <= 0: return ""
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()

def download_and_cache_data(symbol: str, timeframe: int, total_bars: int) -> dict[str, Any]:
    """Downloads OHLC data in chunks with Force Sync/Retry logic."""
    result = {
        "symbol": symbol, "timeframe": TIMEFRAME_NAME.upper(),
        "bars_available": 0, "oldest_bar_utc": "", "newest_bar_utc": "",
        "lookback_days": 0.0, "lookback_hours": 0.0, "status": "pending"
    }

    # 1. Select Symbol and Force Sync
    if not mt5.symbol_select(symbol, True):
        result["status"] = "symbol_not_found"
        return result

    # 2. Aggressive Retry Loop (MT5 often needs a moment to download history)
    all_rates = []
    max_retries = 3
    for attempt in range(max_retries):
        pos = 0
        all_rates = []
        
        while len(all_rates) < total_bars:
            to_fetch = min(CHUNK_SIZE, total_bars - len(all_rates))
            rates = mt5.copy_rates_from_pos(symbol, timeframe, pos, to_fetch)
            
            if rates is None or len(rates) == 0:
                break
                
            all_rates.append(pd.DataFrame(rates))
            pos += len(rates)
            if len(rates) < to_fetch:
                break
        
        if all_rates:
            break
        else:
            logger.info(f"  -> {symbol} attempt {attempt+1} returned nothing, waiting 2s for sync...")
            time.sleep(2)

    if not all_rates:
        result["status"] = "no_history_returned"
        return result

    # 3. Combine chunks and save
    df = pd.concat(all_rates).drop_duplicates(subset=['time']).sort_values('time')
    df['time_ns'] = df['time'].astype(np.int64) * 1_000_000_000
    cache_df = df[['time_ns', 'open', 'high', 'low', 'close']].copy()
    
    cache_filename = CACHE_DIR / f"{symbol}_{TIMEFRAME_NAME}_cache.csv"
    cache_df.to_csv(cache_filename, index=False)
    
    oldest_ts = int(df['time'].min())
    newest_ts = int(df['time'].max())
    lookback_seconds = newest_ts - oldest_ts

    result.update({
        "bars_available": len(df),
        "oldest_bar_utc": _safe_dt_from_unix(oldest_ts),
        "newest_bar_utc": _safe_dt_from_unix(newest_ts),
        "lookback_days": lookback_seconds / 86_400.0,
        "lookback_hours": lookback_seconds / 3_600.0,
        "status": "ok"
    })
    return result

def main():
    if not mt5.initialize():
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        return

    try:
        rows = []
        for instrument, cfg in INSTRUMENT_CONFIG.items():
            symbol = cfg["mt5_symbol"]
            logger.info(f"Scanning {symbol}...")
            res = download_and_cache_data(symbol, TIMEFRAME, TOTAL_TARGET_BARS)
            res["instrument"], res["pair"] = instrument, cfg["pair"]
            rows.append(res)
            
            if res["status"] == "ok":
                logger.info(f"DONE: {symbol} ({res['bars_available']} bars, {res['lookback_days']:.1f} days)")
            else:
                logger.warning(f"SKIP: {symbol} ({res['status']})")

        df_summary = pd.DataFrame(rows)
        summary_file = METADATA_DIR / "mt5_lookback_scan.csv"
        df_summary.to_csv(summary_file, index=False)
        
        print("\n" + "="*75)
        print(f"{'SYMBOL':<10} | {'BARS':<8} | {'DAYS':<8} | {'STATUS'}")
        print("-" * 75)
        for _, r in df_summary.iterrows():
            print(f"{r['symbol']:<10} | {r['bars_available']:<8} | {r['lookback_days']:<8.1f} | {r['status']}")
        print("="*75)

    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()