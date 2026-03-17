# etl/transform.py
import os
import logging
from typing import Dict, Optional, Tuple
from sqlalchemy import text
import pandas as pd
import time
from etl.db import save_df_to_sql, update_transform_metadata, get_engine
import gc, math
import polars as pl
import re

# Silence the arrow.opaque warning
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_candles_from_db_polars(
    engine,
    pair: str,
    interval_minutes: int = 5,
    market_type: str = "spot",
    start_date: Optional[str] = None,
    grid_end_date: Optional[str] = None,
) -> pl.DataFrame:
    """Loads OHLC data and returns a Polars DataFrame (optimized for RAM)."""
    table = "ohlc_spot_raw" if market_type == "spot" else "ohlc_future_raw"

    # 1. Determine Bounds
    with engine.connect() as conn:
        r = conn.execute(
            text(f"SELECT MIN(time), MAX(time) FROM {table} WHERE pair = :p AND interval_minutes = :i"),
            {"p": pair, "i": interval_minutes}
        ).fetchone()
        
    if not r or r[0] is None:
        return pl.DataFrame()

    db_min_time = pd.to_datetime(start_date, utc=True) if start_date else pd.to_datetime(r[0], utc=True)
    db_max_time = pd.to_datetime(grid_end_date, utc=True) if grid_end_date else pd.to_datetime(r[1], utc=True)

    # 2. Query with ADBC
    q = f"""
        SELECT f.time AT TIME ZONE 'UTC' AS time, f.open, f.high, f.low, f.close, f.volume, f.time_ns,
               COALESCE(h.funding_rate_rel, 0) AS funding_rate
        FROM {table} f
        LEFT JOIN funding_history_raw h ON f.pair = h.pair 
             AND date_trunc('minute', f.time) = date_trunc('minute', h.time)
        WHERE f.pair = $1 AND f.interval_minutes = $2 AND f.time >= $3 AND f.time <= $4
        ORDER BY f.time ASC
    """
    
    url = engine.url
    uri = f"postgresql://{url.username}:{url.password}@{url.host}:{url.port or 5432}/{url.database}"
    

    df_pl = pl.read_database_uri(query=q, uri=uri, engine="adbc", 
                                execute_options={"parameters": [pair, interval_minutes, db_min_time, db_max_time]})

    if df_pl.is_empty():
        return df_pl

    # 3. RAM Optimization & Schema Enforcement
    df_pl = df_pl.with_columns([
        pl.col("time").cast(pl.Datetime("us", "UTC")),
        pl.all().exclude(["time", "time_ns"]).cast(pl.Float32)
    ]).unique(subset=["time"]).sort("time")

    # 4. Create Grid
    grid = pl.datetime_range(
        start=db_min_time, 
        end=db_max_time, 
        interval=f"{interval_minutes}m", 
        time_zone="UTC", 
        eager=True
    ).alias("time").to_frame()

    # 5. Join and Re-derive time_ns
    return grid.join(df_pl, on="time", how="left").with_columns([
        # Derive time_ns from the guaranteed-present 'time' column
        (pl.col("time").cast(pl.Int64) * 1000).alias("time_ns"),
        
        # ONLY fill columns that actually came from the DB query
        pl.col("volume").fill_null(0),
        pl.col("funding_rate").fill_null(0)
        
        # REMOVED: spread and change_24h fill_null calls here
    ])


def build_df_main_from_5m_polars(df_pl: pl.DataFrame, run_config: Dict) -> Tuple[pl.DataFrame, Dict]:
    """
    Polars-native indicator builder compatible with the rest of your code.
    - preserves 'time' as Datetime[us,UTC] and creates 'time_ns' (Int64)
    - returns same column names expected by compute_config_and_save
    - MAs use min_periods == window_size and shift(1) (so MA is NaN until enough history)
    - change_24h is filled with 0 where impossible (keeps existing behaviour)
    - spread default is 0.0 (keeps existing behaviour)
    """
    
    # Defensive sort - Handle Method vs Property for different Polars versions
    if not df_pl["time"].is_sorted():
        df_pl = df_pl.sort("time")

    base_minutes = int(run_config.get("BASE_MINUTES", 5))
    lookback_24h = (24 * 60) // base_minutes  # bars in 24h

    # Ensure time/time_ns
    df = df_pl.with_columns([
        pl.col("time").cast(pl.Datetime("us", "UTC")).alias("time"),
        # keep time_ns if present, else derive from time (Int64 nanoseconds)
        (pl.col("time").cast(pl.Int64) * 1000).alias("time_ns") if "time_ns" not in df_pl.columns else pl.col("time_ns").cast(pl.Int64)
    ])

    # 24h change: keep fill_null(0) to preserve previous behavior
    df = df.with_columns([
        (
            ((pl.col("close") - pl.col("close").shift(lookback_24h)) /
             pl.col("close").shift(lookback_24h)) * 100.0
        ).fill_null(0.0).cast(pl.Float32).alias("change_24h")
    ])

    # ensure spread column exists and default to 0.0 for compatibility
    if "spread" not in df.columns:
        df = df.with_columns([pl.lit(0.0).cast(pl.Float32).alias("spread")])
    else:
        df = df.with_columns([pl.col("spread").cast(pl.Float32)])

    # parse timeframe for MA (supports '1h', '2h', '60m' etc.)
    tf_str = str(run_config.get("ma_timeframe", "1h"))
    m = re.search(r'(\d+)\s*([mh]?)', tf_str)
    if not m:
        tf_hours = 1
    else:
        val = int(m.group(1))
        unit = m.group(2) or "h"
        tf_hours = val if unit == "h" else max(1, (val // 60))

    multiplier = int((tf_hours * 60) // base_minutes) if base_minutes > 0 else 12

    # MA slots
    ma_slots = ["ma_a", "ma_b", "ma_c", "ma_d"]
    ma_periods = run_config.get("ma_periods", []) or []
    ma_exprs = []
    for i, period in enumerate(ma_periods):
        if i >= len(ma_slots):
            break
        window_bars = int(period * multiplier)
        if window_bars <= 0:
            continue
        # require full window -> min_periods = window_bars, then shift(1) to avoid lookahead
        ma_exprs.append(
            pl.col("close")
              .rolling_mean(window_size=window_bars, min_periods=window_bars)
              .shift(1)
              .cast(pl.Float32)
              .alias(ma_slots[i])
        )

    if ma_exprs:
        df = df.with_columns(ma_exprs)

    # price vs ma gap (if ma_a exists)
    if "ma_a" in df.columns:
        df = df.with_columns([
            ((pl.col("close") / pl.col("ma_a") - 1.0).cast(pl.Float32)).alias("price_ma_gap_h")
        ])

    # funding_rate default if absent (keep nulls if we don't have values)
    if "funding_rate" not in df.columns:
        df = df.with_columns([pl.lit(None).cast(pl.Float32).alias("funding_rate")])
    else:
        df = df.with_columns([pl.col("funding_rate").cast(pl.Float32)])

    # volume cast / fill 0 if missing
    if "volume" in df.columns:
        df = df.with_columns([pl.col("volume").cast(pl.Float32)])
    else:
        df = df.with_columns([pl.lit(0.0).cast(pl.Float32).alias("volume")])

    # ensure core numeric columns are Float32 for consistency
    cast_cols = []
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            cast_cols.append(pl.col(c).cast(pl.Float32))
    if cast_cols:
        df = df.with_columns(cast_cols)

    # build small nan_report (keeps your existing pattern)
    nan_report = {}
    for col in df.columns:
        cnt = int(df[col].null_count())
        if cnt > 0:
            nan_report[col] = {"nan_count": cnt}

    return df, nan_report


def build_and_save_df_main_to_sql(engine, pair: str, market_type: str = "spot", run_config: Optional[Dict] = None) -> Dict:
    run_config = run_config or {}
    logger.info(f"🚀 Starting transform for {pair} ({market_type})")
    
    # 1. Load and Process (Polars Native)
    df_processed = load_candles_from_db_polars(engine, pair, market_type=market_type)
    
    if df_processed.is_empty():
        logger.warning(f"⚠️ {pair} returned no data from DB. Skipping.")
        return {"rows": 0, "nan_report": {}}

    logger.info(f"📊 Loaded {len(df_processed)} rows. Calculating indicators...")
    df_processed, nan_report = build_df_main_from_5m_polars(df_processed, run_config=run_config)

    # Metadata capture before chunking
    total_rows = len(df_processed)
    last_ts = df_processed["time"].max()
    last_time_ns = int(last_ts.timestamp() * 1e9)
    
    # Add metadata columns efficiently in Polars
    df_processed = df_processed.with_columns([
        pl.lit(pair).alias("pair"),
        pl.lit(market_type).alias("market_type")
    ])

    # 2. Setup Columns for Saving
    core_cols = [
        'pair', 'market_type', 'time', 'time_ns', 'open', 'high', 'low', 'close', 
        'volume', 'funding_rate', 'spread', 'change_24h'
    ]
    ma_slots = ["ma_a", "ma_b", "ma_c", "ma_d"]
    # Ensure all MA slots exist in the frame (fill with null if missing)
    for slot in ma_slots:
        if slot not in df_processed.columns:
            df_processed = df_processed.with_columns(pl.lit(None).cast(pl.Float32).alias(slot))

    ma_cols = ['pair', 'market_type', 'time', 'time_ns'] + ma_slots + \
               [f for f in ['price_ma_gap_h'] if f in df_processed.columns]

    # 3. Incremental Chunk Saving
    chunk_size = 20000
    total_chunks = math.ceil(total_rows / chunk_size)
    
    logger.info(f"💾 Saving {total_rows} rows to DB across {total_chunks} chunks...")

    

    for i in range(0, total_rows, chunk_size):
        chunk_num = (i // chunk_size) + 1
        percent = (chunk_num / total_chunks) * 100
        
        # Slice the Polars frame (CPU efficient)
        chunk_pl = df_processed.slice(i, chunk_size)
        
        # Log progress
        logger.info(f"📦 Processing Chunk {chunk_num}/{total_chunks} ({percent:.1f}%) for {pair}...")

        # Select and convert to pandas
        df_core_pd = chunk_pl.select(core_cols).to_pandas()
        
        # Double-check: Force int64 one last time before SQL
        df_core_pd['time_ns'] = df_core_pd['time_ns'].astype('int64')

        # Convert and Save Core
        save_df_to_sql(engine, chunk_pl.select(core_cols).to_pandas(), pair=pair, table_name="df_main")
        
        # Convert and Save Indicators
        save_df_to_sql(engine, chunk_pl.select(ma_cols).to_pandas(), pair=pair, table_name="indicator_ma")
        
        # Force cleanup for this chunk
        del chunk_pl
        gc.collect()

    # 4. Finalize Metadata
    update_transform_metadata(engine, pair, market_type, last_time_ns)
    logger.info(f"✅ Success: {pair} processed. Total Rows: {total_rows}. Last TS: {last_ts}")

    # Final Cleanup
    del df_processed
    gc.collect()

    return {"rows": total_rows, "nan_report": nan_report}


# utility that returns True if any key column still has NaNs
def needs_recalc(nan_report: Dict) -> bool:
    for k, v in nan_report.items():
        if v.get("nan_count") and v["nan_count"] > 0:
            return True
    return False