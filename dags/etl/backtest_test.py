import polars as pl
from pathlib import Path

target_file = "/opt/airflow/airflow-trading/data_lake/cache/6mo/backtest/era_2024-09/config_00000_combos.tmp.chunk0000.parquet"

if Path(target_file).exists():
    print(f"--- Inspecting: {target_file} ---")
    
    # 1. Peek at Schema only (No data read)
    schema = pl.read_parquet_schema(target_file)
    print("\n[SCHEMA]")
    for col, dtype in schema.items():
        # Highlight the problematic columns
        status = "⚠️ POTENTIAL BAD TYPE" if "Array" in str(dtype) else "✅"
        print(f"{status} {col}: {dtype}")

    # 2. Preview data (Only first 5 rows)
    # Using fetch(5) is the safest way to prevent memory spikes
    df_preview = pl.scan_parquet(target_file).fetch(5)
    
    print("\n[DATA PREVIEW]")
    print(df_preview)
else:
    print(f"File not found: {target_file}")