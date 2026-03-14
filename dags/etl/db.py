# etl/db.py
import io
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from contextlib import closing
from datetime import datetime
from sqlalchemy import text
import io
import logging
import gc

logger = logging.getLogger(__name__)

def get_market_type(pair: str) -> str:
    """Returns 'future' for PF_ pairs, otherwise 'spot'."""
    return "future" if pair.startswith("PF_") else "spot"

def get_engine(sqlalchemy_uri: str, application_name: Optional[str] = None) -> Engine:
    """Create a SQLAlchemy engine with conservative pool sizes and pre-ping.

    - small pool_size and limited max_overflow helps on developer machines
    - pool_pre_ping avoids stale TCP connections
    - pool_recycle prevents very-long-lived connections
    - application_name can be supplied to help identify client sessions in pg_stat_activity
    """
    if application_name:
        # append as connection query param if not already present
        if "?" in sqlalchemy_uri:
            uri = f"{sqlalchemy_uri}&application_name={application_name}"
        else:
            uri = f"{sqlalchemy_uri}?application_name={application_name}"
    else:
        uri = sqlalchemy_uri

    engine = create_engine(
        uri,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=1800,
    )
    return engine


def bulk_upsert_candles(engine: Engine, df, pair: str, interval_minutes: int, market_type: str) -> int:
    if df is None or df.empty:
        return 0

    df = df.copy()
    df["pair"] = pair
    df["interval_minutes"] = int(interval_minutes)

    # 1. Define columns based on market type
    if market_type == "future":
        # Ensure the column exists in the DF before we try to COPY it
        if "funding_rate" not in df.columns:
            df["funding_rate"] = 0.0
        cols = ["pair", "interval_minutes", "time", "time_ns", "open", "high", "low", "close", "volume", "funding_rate"]
        target_table = "ohlc_future_raw"
    else:
        cols = ["pair", "interval_minutes", "time", "time_ns", "open", "high", "low", "close", "volume"]
        target_table = "ohlc_spot_raw"

    # Format time for Postgres
    df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S%z")

    # Prepare buffer
    buf = io.StringIO()
    df[cols].to_csv(buf, sep="\t", index=False, header=False, na_rep="\\N")
    buf.seek(0)

    with closing(engine.raw_connection()) as raw_conn:
        with raw_conn.cursor() as cur:
            try:
                # 2. Create Temp Table mirroring the REAL table schema
                cur.execute(f"CREATE TEMP TABLE tmp_ingest (LIKE {target_table} INCLUDING DEFAULTS) ON COMMIT DROP;")
                
                # 3. Copy data into temp
                cur.copy_from(buf, "tmp_ingest", sep="\t", null="\\N", columns=cols)

                # 4. Build upsert logic
                update_cols = [c for c in cols if c not in ["pair", "interval_minutes", "time"]]
                update_stmt = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])

                query = f"""
                    INSERT INTO {target_table} ({', '.join(cols)})
                    SELECT {', '.join(cols)} FROM tmp_ingest
                    ON CONFLICT (pair, interval_minutes, time) DO UPDATE SET {update_stmt};
                """
                cur.execute(query)
                raw_conn.commit() # Everything worked!
            except Exception as e:
                raw_conn.rollback() # Crucial: clears the "InFailedSqlTransaction" state
                logger.error(f"Bulk upsert failed for {pair}: {e}")
                raise e 
            finally:
                buf.close()

    gc.collect()
    return int(len(df))

def _sanitize_column_names_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace characters that are invalid in SQL column names and normalize pandas columns
    E.g. '%K' -> 'pct_K'
    """
    cols = []
    for c in df.columns:
        name = str(c)
        name = name.replace('%', 'pct_')
        name = name.replace(' ', '_')
        name = name.replace('-', '_')
        name = name.replace('.', '_')
        cols.append(name)
    df = df.copy()
    df.columns = cols
    return df

def save_df_to_sql(engine: Engine, df, pair: str, table_name: str = "df_main") -> int:

    # Build column defs for temp table creation
    col_defs = []
    for col in df.columns:
        if col in ("pair", "market_type"):
            col_defs.append(f"{col} TEXT")
        elif col == "time":
            col_defs.append("time TIMESTAMPTZ")
        elif col.endswith("_ns") or col in ("time_ns", "era_int"):
            col_defs.append(f"{col} BIGINT")
        elif col == "is_outlier":
            col_defs.append(f"{col} BOOLEAN")
        elif col.startswith("ma_") or col.startswith("pct_"):
            col_defs.append(f"{col} DOUBLE PRECISION")
        else:
            col_defs.append(f"{col} REAL")

    total_rows = len(df)
    chunk_size = 100_000

    # Use raw_connection in a context manager to guarantee closure
    with closing(engine.raw_connection()) as raw_conn:
        try:
            with raw_conn.cursor() as cur:
                cur.execute(f"CREATE TEMP TABLE tmp_{table_name} ({', '.join(col_defs)}) ON COMMIT DROP;")

                for i in range(0, total_rows, chunk_size):
                    chunk = df.iloc[i: i + chunk_size]
                    buf = io.StringIO()
                    chunk.to_csv(buf, sep="\t", index=False, header=False, na_rep="\\N")
                    buf.seek(0)
                    cur.copy_from(buf, f"tmp_{table_name}", sep="\t", null="\\N")
                    try:
                        buf.close()
                    except Exception:
                        pass
                    del chunk

                cols = list(df.columns)
                cols_sql = ", ".join(cols)
                update_assigns = [f"{c} = EXCLUDED.{c}" for c in cols if c not in ("pair", "time", "market_type")]
                update_sql = ", ".join(update_assigns) if update_assigns else None

                if update_sql:
                    merge_sql = f"""
                        INSERT INTO {table_name} ({cols_sql})
                        SELECT {cols_sql} FROM tmp_{table_name}
                        ON CONFLICT (pair, market_type, time)
                        DO UPDATE SET {update_sql};
                    """
                else:
                    merge_sql = f"INSERT INTO {table_name} ({cols_sql}) SELECT {cols_sql} FROM tmp_{table_name} ON CONFLICT DO NOTHING;"

                cur.execute(merge_sql)
            # commit after cursor closed
            raw_conn.commit()

            logger.info(f"💾 {table_name}: Upserted {total_rows} rows.")
            return total_rows

        except Exception as e:
            try:
                raw_conn.rollback()
            except Exception:
                pass
            logger.error(f"Failed to save {table_name}: {e}")
            raise
        finally:
            gc.collect()

def update_transform_metadata(engine, pair: str, market_type: str, last_time_ns: int):
    """
    Ensures the metadata table exists and updates the last processed timestamp for a pair.
    """
    with engine.begin() as conn:
        # 1. Create table with market_type in the PK
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS transform_metadata (
                pair TEXT NOT NULL,
                market_type TEXT NOT NULL,
                last_time_ns BIGINT NOT NULL,
                last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (pair, market_type)
            );
        """))
        
        # 2. Upsert the latest timestamp
        conn.execute(text("""
            INSERT INTO transform_metadata (pair, market_type, last_time_ns, last_updated_at)
            VALUES (:p, :m, :ts, CURRENT_TIMESTAMP)
            ON CONFLICT (pair, market_type) DO UPDATE 
            SET last_time_ns = EXCLUDED.last_time_ns,
                last_updated_at = CURRENT_TIMESTAMP;
        """), {"p": pair, "m": market_type, "ts": last_time_ns})