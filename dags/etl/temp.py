import polars as pl
from sqlalchemy import text

def debug_discovery(engine):
    # This matches your Airflow discovery query logic
    query = text("""
        SELECT pair, interval_minutes, MAX(time_ns) as max_ns, MAX(time) as max_ts
        FROM ohlc_spot_raw
        GROUP BY pair, interval_minutes
        ORDER BY pair, interval_minutes;
    """)
    
    with engine.connect() as conn:
        res = conn.execute(query).fetchall()
        
    print(f"{'PAIR':<15} | {'INT':<4} | {'PRECISION':<10} | {'HUMAN TIME'}")
    print("-" * 60)
    for r in res:
        precision = "Seconds" if len(str(r)) <= 10 else "Nanos"
        print(f"{r:<15} | {r:<4} | {precision:<10} | {r}")

# Run this!
debug_discovery(engine)