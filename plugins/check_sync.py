# plugins/check_sync.py
import sys
import os
sys.path.append('/opt/airflow/dags') 
import pandas as pd
from sqlalchemy import text
from typing import Dict, List, Tuple, Optional, Union
from etl.db import get_engine



def check_pipeline_sync(pair: str = "XXBTZUSD"):
    # 1. Setup Connection
    db_user = os.getenv("POSTGRES_USER")
    db_pass = os.getenv("POSTGRES_PASSWORD")
    db_host = os.getenv("POSTGRES_HOST", "postgres")
    db_name = os.getenv("POSTGRES_DB", "airflow")
    uri = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/{db_name}"
    
    engine = get_engine(uri)

    # 2. Query both tables
    query = text("""
        SELECT 
            (SELECT MAX(time_ns) FROM candle_raw WHERE pair = :p AND interval_minutes = 5) as raw_max,
            (SELECT last_time_ns FROM transform_metadata WHERE pair = :p) as transform_max
    """)

    with engine.connect() as conn:
        res = conn.execute(query, {"p": pair}).fetchone()
        
    raw_ns = res[0]
    trans_ns = res[1]

    # 3. Compare and Report
    print(f"\n--- SYNC CHECK FOR {pair} ---")
    
    if raw_ns is None:
        print("❌ CRITICAL: No raw data found in candle_raw.")
        return

    if trans_ns is None:
        print("⚠️  WARNING: No metadata found. Transform hasn't run yet.")
        return

    diff_ns = raw_ns - trans_ns
    diff_minutes = diff_ns / (10**9 * 60)

    if diff_ns == 0:
        print("✅ SUCCESS: Transform is perfectly in sync with Raw data.")
    elif diff_ns > 0:
        print(f"🐢 LAGGING: Transform is behind by {int(diff_minutes)} minutes.")
        print(f"   Raw: {raw_ns} | Trans: {trans_ns}")
    else:
        print("❓ WEIRD: Transform is ahead of Raw data (Check your clocks/logic).")

if __name__ == "__main__":
    check_pipeline_sync()