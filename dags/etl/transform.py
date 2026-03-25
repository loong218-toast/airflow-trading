# etl/transform.py
import os
import logging
from typing import Dict, Optional, Tuple
from sqlalchemy import text
from sqlalchemy.engine import Engine
import pandas as pd
import time
from etl.db import save_df_to_sql, update_transform_metadata
import gc, math
import polars as pl
import re

# Silence the arrow.opaque warning
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_MINUTES = 5

RAW_TABLES = {
    "spot": "ohlc_spot_raw",
    "future": "ohlc_future_raw",
    "xstock": "ohlc_xstock_raw",
}

def needs_transform(engine: Engine, pair: str, market_type: str) -> bool:
    latest_raw = get_latest_raw_time_ns(engine, pair, market_type)
    if latest_raw is None:
        return False

    last_done = get_transform_watermark(engine, pair, market_type)
    if last_done is None:
        return True

    return int(latest_raw) > int(last_done)

def get_latest_df_main_time_ns(engine: Engine, pair: str, market_type: str) -> Optional[int]:
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                SELECT MAX(time_ns)
                FROM df_main
                WHERE pair = :p
                  AND market_type = :m
            """),
            {"p": pair, "m": market_type},
        ).fetchone()

    if not row or row[0] is None:
        return None
    return int(row[0])

def discover_pending_transform_items(engine: Engine) -> list[dict]:
    tables = {
        "spot": "ohlc_spot_raw",
        "future": "ohlc_future_raw",
        "xstock": "ohlc_xstock_raw",
    }

    # First, let's peek at what intervals actually exist (Debug logging)
    with engine.begin() as conn:
        for market_type, table_name in tables.items():
            debug_stats = conn.execute(text(f"""
                SELECT interval_minutes, COUNT(*), MIN(time), MAX(time)
                FROM {table_name}
                GROUP BY interval_minutes
            """)).fetchall()
            for row in debug_stats:
                logger.info(f"📊 Table {table_name} contains: Interval={row}, Rows={row}, Range={row} to {row}")

    pending = []

    for market_type, table_name in tables.items():
        with engine.begin() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT pair, MAX(time_ns) AS latest_raw_time_ns
                    FROM {table_name}
                    WHERE interval_minutes = :base_minutes
                    GROUP BY pair
                """),
                {"base_minutes": BASE_MINUTES},
            ).fetchall()

        for pair, latest_raw_time_ns in rows:
            if latest_raw_time_ns is None:
                continue

            # --- FIX 1: Normalize Raw Timestamp Scale for comparison ---
            # If Raw is in seconds (10 digits), scale to nanoseconds (19 digits)
            if latest_raw_time_ns < 10**11:
                normalized_raw = int(latest_raw_time_ns * 1_000_000_000)
            else:
                normalized_raw = int(latest_raw_time_ns)

            eff = get_latest_df_main_time_ns(engine, str(pair), market_type)
            
            # Use normalized_raw for the "if" check
            if eff is None or normalized_raw > int(eff):
                logger.info(f"✅ Pending: {pair} (Raw NS: {normalized_raw} > Main: {eff})")
                pending.append({
                    "pair": str(pair),
                    "market_type": market_type,
                    "latest_raw_time_ns": normalized_raw, # Use the fixed version
                    "last_time_ns": None if eff is None else int(eff),
                })
            else:
                logger.info(f"⏭️ Skipping {pair}: Raw {normalized_raw} is not ahead of Watermark {eff}.")

    pending.sort(key=lambda x: (x["market_type"], x["pair"]))
    return pending

def get_effective_watermark(engine: Engine, pair: str, market_type: str) -> Optional[int]:
    meta = get_transform_watermark(engine, pair, market_type)
    main = get_latest_df_main_time_ns(engine, pair, market_type)

    # Jan 1, 2010 in nanoseconds (approx 1.2e18)
    # This is safe for your 2017 BTC data but blocks 10-digit "seconds" values
    SAFETY_FLOOR_NS = 1_262_304_000_000_000_000 
    
    # Filter out None and anything that looks like "Seconds" (10-digits) 
    # or "Epoch" (near 0)
    values = [v for v in [meta, main] if v is not None and v > SAFETY_FLOOR_NS]
    
    if not values:
        # If no valid high-precision timestamp exists, we return None 
        # to trigger a full rebuild from the earliest raw data.
        return None

    # We take the MIN of valid values to ensure no data is skipped 
    # if one table updated while the other failed.
    return min(values)

def get_transform_watermark(engine: Engine, pair: str, market_type: str) -> Optional[int]:
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                SELECT last_time_ns
                FROM transform_metadata
                WHERE pair = :p AND market_type = :m
            """),
            {"p": pair, "m": market_type},
        ).fetchone()

    if not row or row[0] is None:
        return None
    return int(row[0])

def load_candles_from_db_polars(
    engine,
    pair: str,
    interval_minutes: int = BASE_MINUTES,
    market_type: str = "spot",
    start_after_ns: Optional[int] = None,
) -> pl.DataFrame:
    if interval_minutes != BASE_MINUTES:
        raise ValueError("df_main is 5m-only. Use interval_minutes=5 here.")

    market_type = (market_type or "spot").lower().strip()
    if market_type not in RAW_TABLES:
        raise ValueError(f"Unsupported market_type: {market_type}")

    table = RAW_TABLES[market_type]

    if market_type == "future":
        base_query = f"""
            SELECT
                f.time AT TIME ZONE 'UTC' AS time,
                f.open, f.high, f.low, f.close, f.volume, f.time_ns,
                COALESCE(h.funding_rate_rel, 0) AS funding_rate
            FROM {table} f
            LEFT JOIN funding_history_raw h
                   ON f.pair = h.pair
                  AND date_trunc('minute', f.time) = date_trunc('minute', h.time)
            WHERE f.pair = :pair
              AND f.interval_minutes = :interval_minutes
        """
    elif market_type == "xstock":
        base_query = f"""
            SELECT
                f.time AT TIME ZONE 'UTC' AS time,
                f.open, f.high, f.low, f.close, f.volume, f.time_ns,
                NULL::double precision AS funding_rate
            FROM {table} f
            WHERE f.pair = :pair
              AND f.interval_minutes = :interval_minutes
        """
    else:
        base_query = f"""
            SELECT
                f.time AT TIME ZONE 'UTC' AS time,
                f.open, f.high, f.low, f.close, f.volume, f.time_ns,
                COALESCE(h.funding_rate_rel, 0) AS funding_rate
            FROM {table} f
            LEFT JOIN funding_history_raw h
                   ON f.pair = h.pair
                  AND date_trunc('minute', f.time) = date_trunc('minute', h.time)
            WHERE f.pair = :pair
              AND f.interval_minutes = :interval_minutes
        """

    params = {
        "pair": pair,
        "interval_minutes": int(interval_minutes),
    }

    if start_after_ns is not None:
        query = base_query + " AND f.time_ns > :start_after_ns ORDER BY f.time ASC"
        params["start_after_ns"] = int(start_after_ns)
    else:
        query = base_query + " ORDER BY f.time ASC"

    with engine.connect() as conn:
        df_pd = pd.read_sql_query(text(query), conn, params=params)

    if df_pd.empty:
        return pl.DataFrame()

    df_pl = pl.from_pandas(df_pd)

    return (
        df_pl.with_columns([
            pl.col("time").cast(pl.Datetime("us", "UTC")),
            pl.col("open").cast(pl.Float32),
            pl.col("high").cast(pl.Float32),
            pl.col("low").cast(pl.Float32),
            pl.col("close").cast(pl.Float32),
            pl.col("volume").cast(pl.Float32),
            pl.col("funding_rate").cast(pl.Float32),
            pl.col("time_ns").cast(pl.Int64),
        ])
        .unique(subset=["time"])
        .sort("time")
    )


def build_df_main_from_5m_polars(df_pl: pl.DataFrame, run_config: Dict) -> Tuple[pl.DataFrame, Dict]:
    if not df_pl["time"].is_sorted():
        df_pl = df_pl.sort("time")

    base_minutes = int(run_config.get("BASE_MINUTES", 5))
    lookback_24h = (24 * 60) // base_minutes

    df = df_pl.with_columns([
        pl.col("time").cast(pl.Datetime("us", "UTC")).alias("time"),
        (pl.col("time").cast(pl.Int64) * 1000).alias("time_ns")
        if "time_ns" not in df_pl.columns
        else pl.col("time_ns").cast(pl.Int64)
    ])

    df = df.with_columns([
        (
            ((pl.col("close") - pl.col("close").shift(lookback_24h)) /
            pl.col("close").shift(lookback_24h)) * 100.0
        ).cast(pl.Float32).alias("change_24h") # REMOVE .fill_null(0.0) here to debug
    ])

    if "spread" not in df.columns:
        df = df.with_columns([pl.lit(0.0).cast(pl.Float32).alias("spread")])
    else:
        df = df.with_columns([pl.col("spread").cast(pl.Float32)])

    if "funding_rate" not in df.columns:
        df = df.with_columns([pl.lit(None).cast(pl.Float32).alias("funding_rate")])
    else:
        df = df.with_columns([pl.col("funding_rate").cast(pl.Float32)])

    if "volume" in df.columns:
        df = df.with_columns([pl.col("volume").cast(pl.Float32)])
    else:
        df = df.with_columns([pl.lit(0.0).cast(pl.Float32).alias("volume")])

    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df = df.with_columns([pl.col(c).cast(pl.Float32)])

    nan_report = {}
    for col in df.columns:
        cnt = int(df[col].null_count())
        if cnt > 0:
            nan_report[col] = {"nan_count": cnt}

    return df, nan_report


def build_and_save_df_main_to_sql(engine, pair: str, market_type: str = "spot", run_config: Optional[Dict] = None) -> Dict:
    run_config = run_config or {}
    market_type = (market_type or "spot").lower().strip()
    logger.info(f"🚀 Starting transform for {pair} ({market_type})")

    base_minutes = BASE_MINUTES
    last_time_ns = get_effective_watermark(engine, pair, market_type)

    overlap_bars = int(run_config.get("transform_overlap_bars", 1000))
    overlap_ns = overlap_bars * base_minutes * 60 * 1_000_000_000

    start_after_ns = None
    if last_time_ns is not None:
        start_after_ns = max(0, int(last_time_ns) - overlap_ns)

    df_processed = load_candles_from_db_polars(
        engine,
        pair,
        interval_minutes=base_minutes,
        market_type=market_type,
        start_after_ns=start_after_ns,
    )

    if df_processed.is_empty():
        logger.warning(f"⚠️ {pair} returned no new data from DB. Skipping.")
        return {"rows": 0, "nan_report": {}}

    logger.info(f"📊 Loaded {len(df_processed)} rows. Calculating indicators...")
    df_processed, nan_report = build_df_main_from_5m_polars(df_processed, run_config=run_config)

    # Remove rows already processed; keep only new tail.
    if last_time_ns is not None:
        df_processed = df_processed.filter(pl.col("time_ns") > int(last_time_ns))

    if df_processed.is_empty():
        logger.info(f"✅ {pair}: no new rows after watermark filter.")
        return {"rows": 0, "nan_report": nan_report}

    df_processed = df_processed.with_columns([
        pl.lit(pair).alias("pair"),
        pl.lit(market_type).alias("market_type")
    ])

    core_cols = [
        "pair", "market_type", "time", "time_ns", "open", "high", "low", "close",
        "volume", "funding_rate", "spread", "change_24h"
    ]

    df_main_out = df_processed.select(core_cols)

    chunk_size = 20000
    total_rows = len(df_main_out)
    total_chunks = math.ceil(total_rows / chunk_size)

    logger.info(f"💾 Saving {total_rows} new df_main rows to DB across {total_chunks} chunks...")

    for i in range(0, total_rows, chunk_size):
        chunk_pl = df_main_out.slice(i, chunk_size)
        df_core_pd = chunk_pl.to_pandas()
        df_core_pd["time_ns"] = df_core_pd["time_ns"].astype("int64")

        save_df_to_sql(engine, df_core_pd, pair=pair, table_name="df_main")
        del chunk_pl
        gc.collect()

    last_ts = df_processed["time"].max()
    new_last_time_ns = int(df_processed["time_ns"].max())
    update_transform_metadata(engine, pair, market_type, new_last_time_ns)

    logger.info(f"✅ Success: {pair} processed. New rows: {total_rows}. Last TS: {last_ts}")
    del df_processed
    gc.collect()

    return {"rows": total_rows, "nan_report": nan_report}


# utility that returns True if any key column still has NaNs
def needs_recalc(nan_report: Dict) -> bool:
    for k, v in nan_report.items():
        if v.get("nan_count") and v["nan_count"] > 0:
            return True
    return False