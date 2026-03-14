import os
import sys
import logging
import polars as pl
from sqlalchemy import text
import pandas as pd
import gc

# 1. Path Fix: Ensure the script can find 'etl' when running inside Docker/Airflow
# We add the 'dags' folder to sys.path so 'from etl.db import ...' works
current_dir = os.path.dirname(os.path.abspath(__file__))
dags_root = os.path.abspath(os.path.join(current_dir, "../")) # Points to /opt/airflow/dags
if dags_root not in sys.path:
    sys.path.insert(0, dags_root)

from etl.db import get_engine, bulk_upsert_candles

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIG ---
# Use the Docker path as primary, fallback to Windows path for local testing
CSV_PATH = "/opt/airflow/analysis/Backtest_results/BTC/@Main/BTCUSD_1m_Binance.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = r"C:\Users\Owner\airflow-trading\analysis\Backtest_results\BTC\@Main\BTCUSD_1m_Binance.csv"

PAIR = "XXBTZUSD"
INTERVAL = 5

def resample_binance_csv_to_5m(path: str) -> pl.DataFrame:
    logger.info(f"📖 Loading 1m CSV: {path}")
    
    # 1. Use Scan (Lazy) but specify schema to save time
    q = (
        pl.scan_csv(path, low_memory=True)
        .select([
            pl.col("Open time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("time"),
            pl.col("Open").cast(pl.Float32).alias("open"),
            pl.col("High").cast(pl.Float32).alias("high"),
            pl.col("Low").cast(pl.Float32).alias("low"),
            pl.col("Close").cast(pl.Float32).alias("close"),
            pl.col("Volume").cast(pl.Float32).alias("volume"),
        ])
        # 2. Resample logic using truncate (Faster than group_by_dynamic)
        .with_columns(
            pl.col("time").dt.truncate("5m")
        )
        .group_by("time")
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum()
        ])
        .sort("time")
    )

    # 3. Collect and then handle metadata (Efficiency)
    df_5m = q.collect()
    
    logger.info("🛠️ Finalizing metadata...")
    df_5m = df_5m.with_columns([
        pl.col("time").dt.replace_time_zone("UTC"),
        pl.lit(PAIR).alias("pair"),
        pl.lit(INTERVAL).alias("interval_minutes"),
        # Faster nanosecond calculation
        (pl.col("time").cast(pl.Int64) * 1000).alias("time_ns") 
    ])

    return df_5m

if __name__ == "__main__":
    try:
        df_processed = resample_binance_csv_to_5m(CSV_PATH)
        pd_df = df_processed.to_pandas()
        pd_df["time"] = pd.to_datetime(pd_df["time"], utc=True)
        pd_df["time_ns"] = pd_df["time"].values.astype("datetime64[ns]").astype("int64")

        db_user = os.getenv("POSTGRES_USER", "airflow")
        db_pass = os.getenv("POSTGRES_PASSWORD", "airflow")
        db_host = os.getenv("POSTGRES_HOST", "postgres")
        db_name = os.getenv("POSTGRES_DB", "airflow")
        uri = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:5432/{db_name}"
        engine = get_engine(uri)

        with engine.connect() as conn:
            # This prevents the script from hanging forever if there's a DB lock
            conn.execute(text("SET statement_timeout = '600s'"))

        # Tweak 2: Print a diagnostic of the first row before starting the loop
        logger.info(f"🔍 Sample Row Time: {pd_df['time'].iloc[0]} | NS: {pd_df['time_ns'].iloc[0]}")
        # --- CHUNKED IMPORT ---
        chunk_size = 100_000
        total_rows = len(pd_df)
        logger.info(f"🚀 Starting chunked import of {total_rows:,} rows...")

        for i in range(0, total_rows, chunk_size):
            chunk = pd_df.iloc[i : i + chunk_size].copy()
            logger.info(f"📦 Processing chunk {i//chunk_size + 1}...")
            
            rows_affected = bulk_upsert_candles(
                engine=engine,
                df=chunk,
                pair=PAIR,
                interval_minutes=INTERVAL,
                market_type='spot'
            )
            
            # Force cleanup
            del chunk
            gc.collect()
            logger.info(f"✅ Chunk saved: {rows_affected:,} rows.")

        logger.info("🏁 ALL DATA IMPORTED SUCCESSFULLY!")

    except Exception as e:
        logger.error(f"❌ CRITICAL FAILURE: {str(e)}")
        import traceback
        traceback.print_exc()