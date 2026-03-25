from __future__ import annotations

import json
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import pendulum
import polars as pl
import requests
from airflow.sdk import dag, task
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

from etl.db import bulk_upsert_candles, get_engine

logger = logging.getLogger(__name__)

default_args = {
    "owner": "loong",
    "retries": 3,
    "retry_delay": timedelta(minutes=3),
}

DAG_TIMEOUT = timedelta(minutes=25)
TASK_TIMEOUT = timedelta(minutes=20)

ENV_CANDIDATES = [
    Path(os.getenv("AIRFLOW_ENV_PATH", "/opt/airflow/.env")),
    Path("/opt/airflow/airflow-trading/.env"),
    Path("/opt/airflow/dags/.env"),
]

STATE_PATH = Path(
    os.getenv("TG_BUFFER_STATE_PATH", "/opt/airflow/airflow-trading/tg_buffer_state.json")
)

VALID_MARKET_TYPES = {"spot", "future", "xstock"}


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
        logger.warning(
            "No .env file found in %s. Falling back to existing container environment.",
            [str(p) for p in ENV_CANDIDATES],
        )


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


def _get_telegram_token_and_chat_id() -> tuple[str, str]:
    token = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("TG_DATA_CHANNEL_ID")

    if not token:
        raise RuntimeError("TG_BOT_TOKEN is missing.")
    if not chat_id:
        raise RuntimeError("TG_DATA_CHANNEL_ID is missing.")

    return token, str(chat_id)


def _read_cursor() -> int:
    if not STATE_PATH.exists():
        return 0
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return int(payload.get("last_update_id", 0))
    except Exception:
        return 0


def _write_cursor(last_update_id: int) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"last_update_id": int(last_update_id)}, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, STATE_PATH)


def _raw_table_for_market_type(market_type: str) -> str:
    market_type = str(market_type).lower().strip()
    if market_type == "spot":
        return "ohlc_spot_raw"
    if market_type == "future":
        return "ohlc_future_raw"
    if market_type == "xstock":
        return "ohlc_xstock_raw"
    raise ValueError(f"Unsupported market_type: {market_type}")


def _telegram_get_updates(token: str, offset: int, limit: int = 100) -> list[dict[str, Any]]:
    r = requests.get(
        f"https://api.telegram.org/bot{token}/getUpdates",
        params={
            "offset": offset,
            "limit": limit,
            "allowed_updates": json.dumps(["channel_post", "edited_channel_post", "message"]),
        },
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    if not payload.get("ok"):
        raise RuntimeError(payload)
    return payload.get("result", [])


def _telegram_get_file_path(token: str, file_id: str) -> str:
    r = requests.get(
        f"https://api.telegram.org/bot{token}/getFile",
        params={"file_id": file_id},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    if not payload.get("ok"):
        raise RuntimeError(payload)
    return payload["result"]["file_path"]


def _telegram_delete_message(token: str, chat_id: str, message_id: int) -> None:
    r = requests.post(
        f"https://api.telegram.org/bot{token}/deleteMessage",
        data={"chat_id": chat_id, "message_id": message_id},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    if not payload.get("ok"):
        raise RuntimeError(payload)


def _telegram_download_file(token: str, file_path: str, local_path: Path) -> None:
    url = f"https://api.telegram.org/file/bot{token}/{file_path}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    local_path.write_bytes(r.content)


def _read_parquet_file(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)

    required = {"pair", "interval_minutes", "market_type", "time", "time_ns"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")

    df = df.with_columns([
        pl.col("pair").cast(pl.Utf8),
        pl.col("market_type").cast(pl.Utf8),
        pl.col("interval_minutes").cast(pl.Int64),
        pl.col("time").cast(pl.Datetime("ns", "UTC")),
        pl.col("time_ns").cast(pl.Int64),
    ])

    for c in ["open", "high", "low", "close", "volume", "funding_rate", "spread"]:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Float32))

    return df.sort("time_ns")


@dag(
    dag_id="kraken_ingest",
    schedule="5 */4 * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    dagrun_timeout=DAG_TIMEOUT,
    default_args=default_args,
    tags=["raw data", "ingest"],
)
def kraken_ingest():

    @task(execution_timeout=TASK_TIMEOUT, pool="ingest_pool", priority_weight=1)
    def consume_telegram_parquets() -> dict:
        token, chat_id = _get_telegram_token_and_chat_id()

        engine = get_engine(_db_uri())
        inbox = Path("/tmp/telegram_inbox")
        inbox.mkdir(parents=True, exist_ok=True)

        last_update_id = _read_cursor()
        max_seen_update_id = last_update_id
        affected_pairs: set[str] = set()
        processed_messages: list[dict[str, Any]] = []

        max_batches = 20
        batch_count = 0

        while batch_count < max_batches:
            updates = sorted(
                _telegram_get_updates(token, last_update_id + 1, limit=100),
                key=lambda u: int(u["update_id"]),
            )

            if not updates:
                break

            for update in updates:
                update_id = int(update["update_id"])
                max_seen_update_id = max(max_seen_update_id, update_id)
                local_path: Optional[Path] = None

                try:
                    post = (
                        update.get("channel_post")
                        or update.get("edited_channel_post")
                        or update.get("message")
                    )

                    if not post:
                        _write_cursor(update_id)
                        last_update_id = update_id
                        continue

                    chat = post.get("chat", {})
                    actual_id = str(chat.get("id"))

                    if actual_id != chat_id:
                        if chat.get("type") == "private":
                            logger.info("Ignoring private message from %s", actual_id)
                        else:
                            logger.info(
                                "Ignoring update from different chat. expected=%s got=%s",
                                chat_id,
                                actual_id,
                            )
                        _write_cursor(update_id)
                        last_update_id = update_id
                        continue

                    doc = post.get("document")
                    if not doc:
                        logger.info("Update %s is not a document, skipping.", update_id)
                        _write_cursor(update_id)
                        last_update_id = update_id
                        continue

                    file_name = doc.get("file_name") or f"{post.get('message_id', update_id)}.parquet"
                    if "parquet" not in file_name.lower():
                        _write_cursor(update_id)
                        last_update_id = update_id
                        continue

                    file_path = _telegram_get_file_path(token, doc["file_id"])
                    local_path = inbox / file_name
                    _telegram_download_file(token, file_path, local_path)

                    df = _read_parquet_file(local_path)

                    pair = str(df["pair"][0])
                    interval_minutes = int(df["interval_minutes"][0])
                    market_type = str(df["market_type"][0]).lower().strip()

                    if market_type not in VALID_MARKET_TYPES:
                        raise ValueError(f"Unsupported market_type in {file_name}: {market_type}")

                    bulk_upsert_candles(
                        engine=engine,
                        df=df,
                        pair=pair,
                        interval_minutes=interval_minutes,
                        market_type=market_type,
                    )

                    _telegram_delete_message(token, chat_id, int(post["message_id"]))

                    if interval_minutes == 5:
                        affected_pairs.add(pair)

                    processed_messages.append(
                        {
                            "pair": pair,
                            "interval_minutes": interval_minutes,
                            "market_type": market_type,
                            "message_id": int(post["message_id"]),
                            "file_name": file_name,
                            "rows": int(df.height),
                        }
                    )

                    _write_cursor(update_id)
                    last_update_id = update_id

                finally:
                    if local_path is not None:
                        try:
                            local_path.unlink(missing_ok=True)
                        except Exception:
                            pass

            batch_count += 1
            last_update_id = max_seen_update_id

        if batch_count >= max_batches:
            logger.warning(
                "Stopped after max_batches=%s to avoid endless drain; cursor=%s",
                max_batches,
                last_update_id,
            )

        return {
            "pairs": sorted(affected_pairs),
            "processed_count": len(processed_messages),
            "messages": processed_messages,
            "last_update_id": max_seen_update_id,
            "batches": batch_count,
        }

    @task
    def summarize(result: dict) -> dict:
        pairs = result.get("pairs", []) if isinstance(result, dict) else []
        return {
            "pairs": pairs,
            "count": len(pairs),
            "processed_count": result.get("processed_count", 0) if isinstance(result, dict) else 0,
        }

    processed = consume_telegram_parquets()
    summary = summarize(processed)

    trigger_transform = TriggerDagRunOperator(
        task_id="trigger_transform",
        trigger_dag_id="kraken_transform",
        conf={"pairs": processed["pairs"], "source": "telegram_buffer"},
        wait_for_completion=False,
    )

    summary >> trigger_transform


kraken_ingest_dag = kraken_ingest()