from __future__ import annotations

import gc
import json
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pendulum
import polars as pl
from airflow.sdk import Variable, dag, task
from sqlalchemy import text

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

ENV_CANDIDATES = [
    Path(os.getenv("AIRFLOW_ENV_PATH", "/opt/airflow/.env")),
    Path("/opt/airflow/airflow-trading/.env"),
    Path("/opt/airflow/dags/.env"),
]


def _load_runtime_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        logger.warning("python-dotenv not available; using existing container environment.")
        return

    loaded = None
    for path in ENV_CANDIDATES:
        if path.exists():
            load_dotenv(path, override=False)
            loaded = path
            break

    if loaded:
        logger.info("Loaded environment from %s", loaded)
    else:
        logger.warning("No .env file found in %s", [str(p) for p in ENV_CANDIDATES])


_load_runtime_env()


def _db_uri() -> str:
    db_user = os.getenv("POSTGRES_USER")
    db_pass = os.getenv("POSTGRES_PASSWORD")
    db_host = os.getenv("POSTGRES_HOST", "postgres")
    db_name = os.getenv("POSTGRES_DB", "airflow")
    db_port = os.getenv("POSTGRES_PORT", "5432")

    if not db_user or not db_pass:
        raise RuntimeError("POSTGRES_USER / POSTGRES_PASSWORD are required")

    return f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"


def _normalize_item(item: Any, default_market_type: str = "spot") -> Dict[str, str]:
    if isinstance(item, str):
        return {"pair": item, "market_type": default_market_type}

    if not isinstance(item, dict) or "pair" not in item:
        raise ValueError(f"Invalid item: {item}")

    market_type = str(item.get("market_type", default_market_type)).lower().strip()
    if market_type not in VALID_MARKET_TYPES:
        market_type = default_market_type

    return {
        "pair": str(item["pair"]),
        "market_type": market_type,
    }


def _load_items_from_variables() -> List[Dict[str, str]]:
    spot_json = Variable.get("kraken_assets_spot", default="[]")
    futures_json = Variable.get("kraken_assets_futures", default="[]")
    xstock_json = Variable.get("kraken_assets_xstock", default="[]")

    items: List[Dict[str, str]] = []

    for raw, default_market_type in [
        (spot_json, "spot"),
        (futures_json, "future"),
        (xstock_json, "xstock"),
    ]:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for x in parsed:
                    items.append(_normalize_item(x, default_market_type=default_market_type))
        except Exception:
            logger.warning("Failed to parse one of the Variable lists; skipping it.")

    if not items:
        items = [{"pair": "XXBTZUSD", "market_type": "spot"}]

    return items


def _raw_table_for_market_type(market_type: str) -> str:
    market_type = str(market_type).lower().strip()
    if market_type == "spot":
        return "ohlc_spot_raw"
    if market_type == "future":
        return "ohlc_future_raw"
    if market_type == "xstock":
        return "ohlc_xstock_raw"
    raise ValueError(f"Unsupported market_type: {market_type}")


def _row_exists(engine, table_name: str, pair: str) -> bool:
    with engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT 1 FROM {table_name} WHERE pair = :p LIMIT 1"),
            {"p": pair},
        ).fetchone()
    return row is not None


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
    def get_items_to_process(**context) -> List[Dict[str, str]]:
        dag_run = context.get("dag_run")
        conf = getattr(dag_run, "conf", None) if dag_run else None

        if conf and "pairs" in conf:
            raw_pairs = conf["pairs"]
            if isinstance(raw_pairs, list):
                items = [_normalize_item(x, default_market_type="spot") for x in raw_pairs]
                logger.info("Received %d items from trigger conf", len(items))
                return items

        items = _load_items_from_variables()
        logger.info("Loaded %d items from Variables", len(items))
        return items

    @task(pool="heavy_compute_pool", priority_weight=4)
    def run_transform(item: Dict[str, str]) -> Dict[str, Any]:
        pair = str(item["pair"])
        market_type = str(item["market_type"]).lower().strip()

        if market_type not in VALID_MARKET_TYPES:
            raise ValueError(f"Unsupported market_type: {market_type}")

        engine = get_engine(_db_uri())

        target_raw = _raw_table_for_market_type(market_type)
        logger.info("Checking %s in %s", pair, target_raw)

        if not _row_exists(engine, target_raw, pair):
            logger.warning("Skipping %s (%s): no raw data found", pair, market_type)
            return {"pair": pair, "market_type": market_type, "status": "skipped_no_data"}

        try:
            res = build_and_save_df_main_to_sql(
                engine,
                pair=pair,
                market_type=market_type,
            )

            if res.get("rows", 0) == 0:
                return {"pair": pair, "market_type": market_type, "status": "no_new_rows"}

            recalc_needed = needs_recalc(res.get("nan_report", {}))
            out = {
                "pair": pair,
                "market_type": market_type,
                "status": "ok",
                "rows": int(res.get("rows", 0)),
                "recalc_needed": bool(recalc_needed),
            }

            del engine
            gc.collect()
            return out

        except Exception as e:
            logger.error("Failed to transform %s (%s): %s", pair, market_type, str(e))
            raise

    items = get_items_to_process()
    run_transform.expand(item=items)


kraken_transform_dag = kraken_transform()