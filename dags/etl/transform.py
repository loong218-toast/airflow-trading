# etl/transform.py
import os
import logging
from typing import Dict, Optional, Tuple
from sqlalchemy import text
from sqlalchemy.engine import Engine
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
    interval_minutes: int = 5,
    market_type: str = "spot",
    start_after_ns: Optional[int] = None,
) -> pl.DataFrame:
    table = "ohlc_spot_raw" if market_type == "spot" else "ohlc_future_raw"

    q = f"""
        SELECT f.time AT TIME ZONE 'UTC' AS time,
               f.open, f.high, f.low, f.close, f.volume, f.time_ns,
               COALESCE(h.funding_rate_rel, 0) AS funding_rate
        FROM {table} f
        LEFT JOIN funding_history_raw h
               ON f.pair = h.pair
              AND date_trunc('minute', f.time) = date_trunc('minute', h.time)
        WHERE f.pair = $1
          AND f.interval_minutes = $2
          AND ($3::bigint IS NULL OR f.time_ns > $3)
        ORDER BY f.time ASC
    """

    url = engine.url
    uri = f"postgresql://{url.username}:{url.password}@{url.host}:{url.port or 5432}/{url.database}"

    df_pl = pl.read_database_uri(
        query=q,
        uri=uri,
        engine="adbc",
        execute_options={"parameters": [pair, interval_minutes, start_after_ns]},
    )

    if df_pl.is_empty():
        return df_pl

    df_pl = (
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

    return df_pl


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
        ).fill_null(0.0).cast(pl.Float32).alias("change_24h")
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
    logger.info(f"🚀 Starting transform for {pair} ({market_type})")

    from etl.db import get_transform_watermark

    last_time_ns = get_transform_watermark(engine, pair, market_type)

    # Keep a small overlap so change_24h can be calculated correctly.
    # For 5m candles, 24h = 288 bars.
    overlap_bars = int(run_config.get("transform_overlap_bars", 300))
    base_minutes = int(run_config.get("BASE_MINUTES", 5))
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
    new_last_time_ns = int(last_ts.timestamp() * 1e9)
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