from airflow.sdk import Variable, dag, task
import pendulum
import os
import json

@dag(
    schedule=None,  # Triggered by kraken_ingest
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    # Use max_active_tasks to prevent CPU spikes on your laptop
    max_active_tasks=3, 
    tags=["kraken", "transform"]
)
def kraken_transform():

    @task()
    def get_pairs_to_process(**context):
        conf = context.get("dag_run").conf
        if conf and "pairs" in conf:
            pairs = conf["pairs"]
            print(f"Received pairs from trigger: {pairs}")
        else:
            # Pull from both variables
            spot_json = Variable.get("kraken_assets_spot", default='[]')
            futures_json = Variable.get("kraken_assets_futures", default='[]')
            
            spot_pairs = json.loads(spot_json)
            futures_pairs = json.loads(futures_json)
            
            # Combine them
            pairs = spot_pairs + futures_pairs
            
            # Fallback if both are empty
            if not pairs:
                pairs = ["XXBTZUSD"]
                
            print(f"Combined pairs from Variables: {pairs}")
        
        return pairs

    @task(pool="heavy_compute_pool")
    def run_transform(pair: str):
        from etl.db import get_engine
        from etl.transform import build_and_save_df_main_to_sql, needs_recalc
        from sqlalchemy import text
        import logging

        # 1. Connection Setup
        db_user = os.getenv("POSTGRES_USER")
        db_pass = os.getenv("POSTGRES_PASSWORD")
        db_host = os.getenv("POSTGRES_HOST", "postgres")
        db_name = os.getenv("POSTGRES_DB", "airflow")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        uri = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/{db_name}"
        engine = get_engine(uri)

        # 2. Universal Table Logic
        # Detect if it's a future based on the prefix your ingest uses
        is_future = pair.startswith("PF_")
        target_raw = "ohlc_future_raw" if is_future else "ohlc_spot_raw"
        
        print(f"DEBUG: Checking for pair '{pair}' in {target_raw}")
        with engine.connect() as conn:
            exists = conn.execute(
                text(f"SELECT 1 FROM {target_raw} WHERE pair = :p LIMIT 1"), 
                {"p": pair}
            ).fetchone()
        
        if not exists:
            logging.warning(f"⚠️ Skipping {pair}: No raw data found in candle_raw. Check your Variable spelling!")
            return {"pair": pair, "status": "skipped_no_data"}

        # 3. Try/Except block for math/transform errors
        try:
            res = build_and_save_df_main_to_sql(
                engine, 
                pair=pair, 
                market_type="future" if is_future else "spot"
            )
            
            # If the build returned 0 rows, it's effectively a skip
            if res.get("rows", 0) == 0:
                return {"pair": pair, "status": "no_new_rows"}

            recalc_needed = needs_recalc(res.get("nan_report", {}))
            return {"pair": pair, "result": res, "recalc_needed": bool(recalc_needed)}
            
        except Exception as e:
            logging.error(f"❌ Failed to transform {pair}: {str(e)}")
            raise # Re-raise so Airflow marks the task as failed

    pairs_to_process = get_pairs_to_process()
    run_transform.expand(pair=pairs_to_process)

kraken_transform_dag = kraken_transform()