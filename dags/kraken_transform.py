from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any, Dict, List

import pendulum
from airflow.sdk import dag, task

from etl.db import get_engine
from etl.transform import build_and_save_df_main_to_sql, needs_recalc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

default_args = {
    "owner": "loong",
    "retries": 3,
    "retry_delay": timedelta(minutes=3),
}

VALID_MARKET_TYPES = {"spot", "future", "xstock"}

def _db_uri() -> str:
    db_user = os.getenv("POSTGRES_USER")
    db_pass = os.getenv("POSTGRES_PASSWORD")
    db_host = os.getenv("POSTGRES_HOST", "postgres")
    db_name = os.getenv("POSTGRES_DB", "airflow")
    db_port = os.getenv("POSTGRES_PORT", "5432")

    if not db_user or not db_pass:
        raise RuntimeError("POSTGRES_USER / POSTGRES_PASSWORD are required")

    return f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

@dag(
    dag_id="kraken_transform",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=3,
    default_args=default_args,
    tags=["kraken", "transform"],
)
def kraken_transform():

    @task()
    def get_items_to_process() -> List[Dict[str, str]]:
        engine = get_engine(_db_uri())

        from etl.transform import discover_pending_transform_items

        items = discover_pending_transform_items(engine)
        logger.info("Loaded %d pending items from raw tables", len(items))

        return [
            {
                "pair": x["pair"],
                "market_type": x["market_type"],
            }
            for x in items
        ]

    @task(priority_weight=4)
    def run_transform(item: Dict[str, str]) -> Dict[str, Any]:
        pair = str(item["pair"])
        market_type = str(item["market_type"]).lower().strip()

        if market_type not in VALID_MARKET_TYPES:
            raise ValueError(f"Unsupported market_type: {market_type}")

        engine = get_engine(_db_uri())

        try:
            res = build_and_save_df_main_to_sql(
                engine,
                pair=pair,
                market_type=market_type,
            )

            if res.get("rows", 0) == 0:
                return {
                    "pair": pair,
                    "market_type": market_type,
                    "status": res.get("reason", "no_new_rows"),
                }

            recalc_needed = needs_recalc(res.get("nan_report", {}))
            return {
                "pair": pair,
                "market_type": market_type,
                "status": "ok",
                "rows": int(res.get("rows", 0)),
                "recalc_needed": bool(recalc_needed),
            }

        except Exception as e:
            logger.error("Failed to transform %s (%s): %s", pair, market_type, str(e))
            raise

    items = get_items_to_process()
    run_transform.expand(item=items)


kraken_transform_dag = kraken_transform()