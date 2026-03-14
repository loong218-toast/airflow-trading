from airflow.sdk import Variable, dag, task
import json                         
import pendulum
import os
from datetime import timedelta
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

default_args = {
    'owner': 'loong',
    'retries': 3,
    'retry_delay': timedelta(minutes=3),
}

@dag(
    schedule="0 */4 * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1, # Crucial for Kraken rate limits
    default_args=default_args,
    tags=["raw data", "ingest"]
)
def kraken_ingest():

    @task
    def get_asset_list():

        spots = json.loads(Variable.get("kraken_assets_spot", '["XXBTZUSD"]'))
        perps = json.loads(Variable.get("kraken_assets_futures", '["PF_XBTUSD", "PF_XAUTUSD"]'))
        return spots + perps

        return []

    @task
    def validate_inputs(config: dict):
        """
        Validate each config object. If one fails, this specific 
        mapped task instance will fail.
        """
        from etl.kraken_api import validate_asset_exists
        
        pair = config["pair"]
        m_type = config["market_type"]
        
        # This will raise ValueError and stop the DAG if pair is invalid
        validate_asset_exists(pair, m_type)
        return config

    @task
    def check_market_stats(pair: str):
        from etl.kraken_api import fetch_futures_tickers
        
        # Only check futures stats for actual futures pairs (PF_...)
        if not pair.startswith("PF_"):
            return {"pair": pair, "status": "spot_skipped"}

        stats = fetch_futures_tickers(pair)
        if stats.empty:
            raise ValueError(f"Could not fetch stats for {pair}")
        
        row = stats.iloc[0]
        return row.to_dict()

    @task
    def get_ingest_inputs():
        import json

        intervals = json.loads(Variable.get("ingest_timeframes", '[5, 60, 1440]'))
        spots = json.loads(Variable.get("kraken_assets_spot", '["XXBTZUSD"]'))
        perps = json.loads(Variable.get("kraken_assets_futures", '["PF_XBTUSD", "PF_XAUTUSD"]'))
        
        map_inputs = []
        for i in intervals:
            for p in spots:
                map_inputs.append({"pair": p, "interval_minutes": i, "market_type": "spot"})
            for p in perps:
                map_inputs.append({"pair": p, "interval_minutes": i, "market_type": "future"})
        return map_inputs


    # --- NEW TASK: SYNC HISTORICAL FUNDING ---
    @task
    def sync_funding_history(pair: str):
        if not pair.startswith("PF_"): return "skip_spot"
        
        from etl.kraken_api import fetch_historical_funding 
        from etl.db import get_engine
        from sqlalchemy import table, column, MetaData # Import these
        from sqlalchemy.dialects.postgresql import insert
        import os

        DB_USER = os.getenv("POSTGRES_USER")
        DB_PASS = os.getenv("POSTGRES_PASSWORD")
        DB_NAME = "airflow"
        db_host = os.getenv("POSTGRES_HOST", "postgres")
        conn_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{db_host}/{DB_NAME}"
        
        engine = get_engine(conn_uri)
        df = fetch_historical_funding(pair)
        
        if df is not None and not df.empty:
            # 1. Define the table structure for SQLAlchemy
            funding_table = table("funding_history_raw",
                column("pair"),
                column("time"),
                column("time_ns"),
                column("funding_rate_abs"),
                column("funding_rate_rel")
            )
            
            with engine.begin() as conn:
                for _, row in df.iterrows():
                    # 2. Pass the table object directly to insert()
                    stmt = insert(funding_table).values(
                        pair=pair,
                        time=row['time'],
                        time_ns=row['time_ns'],
                        funding_rate_abs=row['funding_rate_abs'],
                        funding_rate_rel=row['funding_rate_rel']
                    ).on_conflict_do_nothing()
                    
                    conn.execute(stmt)
            return f"Synced {len(df)} points for {pair}"
        return f"No data for {pair}"

    @task(pool="heavy_compute_pool", priority_weight=1)
    def ingest_for_interval(pair: str, interval_minutes: int, market_type: str):
        from sqlalchemy import text
        from etl.kraken_api import fetch_ohlc, fetch_futures_ohlc
        from etl.db import get_engine, bulk_upsert_candles
        import os

        DB_USER = os.getenv("POSTGRES_USER")
        DB_PASS = os.getenv("POSTGRES_PASSWORD")
        DB_NAME = "airflow"
        # Using host 'postgres' assumes Docker network
        db_host = os.getenv("POSTGRES_HOST", "postgres")
        conn_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{db_host}/{DB_NAME}"
        target_table = "ohlc_spot_raw" if market_type == "spot" else "ohlc_future_raw"
        
        engine = get_engine(conn_uri)

        # 1. High Water Mark (HWM) logic
        with engine.begin() as conn:
            res = conn.execute(
                text(f"SELECT MAX(time_ns) FROM {target_table} WHERE pair = :p AND interval_minutes = :i"),
                {"p": pair, "i": interval_minutes}
            ).fetchone()
            last_ns = res[0] if res and res[0] is not None else None

        since = (int(last_ns) // 10**9) + 1 if last_ns else None

        # 2. Fetch data (the interval_minutes variable is passed into the fetcher)
        if market_type == "future":
            # fetch_futures_ohlc handles the 5 -> '5m' conversion internally
            df, last = fetch_futures_ohlc(pair, interval_minutes)
        else:
            df, last = fetch_ohlc(pair, interval_minutes, since=since)

        # 3. Data Cleansing
        if not df.empty:
            df["pair"] = pair
        df = df.drop_duplicates(subset=['time_ns'], keep='last')
        if last_ns:
            df = df[df['time_ns'] > int(last_ns)]

        # 4. Save to Database
        rows_saved = 0
        if not df.empty:
            print(f"DEBUG: Columns in DF: {df.columns.tolist()}")
            print(f"DEBUG: Sample data:\n{df[['pair', 'close', 'volume']].head()}")
            bulk_upsert_candles(engine, df, pair, interval_minutes, market_type=market_type)
            rows_saved = len(df)

        return {"pair": pair, "interval": interval_minutes, "rows": rows_saved}

    @task(pool="heavy_compute_pool")
    def finalize(results: list):
        if not results:
            return {"total_rows": 0, "pairs": []}
        total_rows = sum(r.get("rows", 0) for r in results if isinstance(r, dict))
        pairs_processed = list(set(r.get("pair") for r in results if isinstance(r, dict)))
        return {"total_rows": total_rows, "pairs": pairs_processed}

    # --- EXECUTION FLOW ---
    ingest_configs = get_ingest_inputs()
    all_assets = get_asset_list()

    # 1. Validation is the Gatekeeper
    # This prevents ANY further API calls if a symbol is wrong
    validated_configs = validate_inputs.expand(config=ingest_configs)

    # 2. Parallel Task Definitions
    # stats and funding are simple lists, but they should also be validated
    # To be aggressive, we only run these if the pairs are valid
    stats_checks = check_market_stats.expand(pair=all_assets)
    funding_sync = sync_funding_history.expand(pair=all_assets)
    
    # Ingest uses the validated list of dicts
    mapped_ingest = ingest_for_interval.expand_kwargs(validated_configs)

    # 3. Aggregation
    final_info = finalize(mapped_ingest)

    # 4. For the Trigger operator
    trigger_transform = TriggerDagRunOperator(
        task_id="trigger_transform",
        trigger_dag_id="kraken_transform", 
        conf={"pairs": all_assets},
        wait_for_completion=False
    )

    # Define the Flow: Stats and Ingest can run in parallel, 
    # but both must finish before we finalize and trigger transform.
    [stats_checks, mapped_ingest, funding_sync] >> final_info >> trigger_transform

kraken_ingest_dag = kraken_ingest()