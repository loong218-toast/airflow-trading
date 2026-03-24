# kraken_ingest.py
from airflow.sdk import Variable, dag, task
import pendulum
import os
from datetime import timedelta
from pathlib import Path
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator


default_args = {
    "owner": "loong",
    "retries": 3,
    "retry_delay": timedelta(minutes=3),
}

@dag(
    schedule="0 */4 * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    default_args=default_args,
    tags=["raw data", "ingest"],
)
def kraken_ingest():

    @task(pool="heavy_compute_pool", priority_weight=1)
    def consume_telegram_parquets():
        import json
        import requests
        import pandas as pd
        from dotenv import load_dotenv
        from pathlib import Path
        from etl.db import get_engine, bulk_upsert_candles

        env_path = Path(__file__).resolve().parent / ".env"
        load_dotenv()

        token = os.environ["TG_BOT_TOKEN"]
        chat_id = str(os.environ["TG_DATA_CHANNEL_ID"])

        db_user = os.getenv("POSTGRES_USER")
        db_pass = os.getenv("POSTGRES_PASSWORD")
        db_name = "airflow"
        db_host = os.getenv("POSTGRES_HOST", "postgres")
        conn_uri = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/{db_name}"
        engine = get_engine(conn_uri)

        inbox = Path("/tmp/telegram_inbox")
        inbox.mkdir(parents=True, exist_ok=True)

        last_update_id = int(Variable.get("tg_buffer_last_update_id", "0"))
        affected_pairs = set()

        def get_updates(offset: int):
            r = requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={
                    "offset": offset,
                    "limit": 100,
                    "timeout": 0,
                    "allowed_updates": json.dumps(["channel_post", "edited_channel_post"]),
                },
                timeout=30,
            )
            r.raise_for_status()
            payload = r.json()
            if not payload.get("ok"):
                raise RuntimeError(payload)
            return payload["result"]

        def get_file_path(file_id: str) -> str:
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

        def delete_message(message_id: int):
            r = requests.post(
                f"https://api.telegram.org/bot{token}/deleteMessage",
                data={"chat_id": chat_id, "message_id": message_id},
                timeout=30,
            )
            r.raise_for_status()
            payload = r.json()
            if not payload.get("ok"):
                raise RuntimeError(payload)

        updates = get_updates(last_update_id + 1)

        for update in updates:
            update_id = update["update_id"]
            post = update.get("channel_post") or update.get("edited_channel_post")
            if not post:
                continue

            if str(post.get("chat", {}).get("id")) != chat_id:
                continue

            doc = post.get("document")
            if not doc:
                continue

            file_name = doc.get("file_name") or f"{post['message_id']}.parquet"
            if not file_name.lower().endswith(".parquet"):
                continue

            file_path = get_file_path(doc["file_id"])
            download_url = f"https://api.telegram.org/file/bot{token}/{file_path}"
            local_path = inbox / file_name

            r = requests.get(download_url, timeout=120)
            r.raise_for_status()
            local_path.write_bytes(r.content)

            df = pd.read_parquet(local_path)
            required = {"pair", "interval_minutes", "market_type", "time", "time_ns"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns in {file_name}: {sorted(missing)}")

            pair = str(df["pair"].iloc[0])
            interval_minutes = int(df["interval_minutes"].iloc[0])
            market_type = str(df["market_type"].iloc[0])

            # Save raw exactly as received
            bulk_upsert_candles(engine, df, pair, interval_minutes, market_type=market_type)

            # Only 5m data should drive df_main later
            if interval_minutes == 5:
                affected_pairs.add(pair)

            # Delete only after successful DB write
            delete_message(post["message_id"])

            Variable.set("tg_buffer_last_update_id", str(update_id))

            return list(affected_pairs)

    @task(pool="heavy_compute_pool")
    def finalize(pairs: list):
        return {"pairs": pairs, "count": len(pairs)}

    @task
    def trigger_transform(pairs: list):
        return {"pairs": pairs}

    processed_pairs = consume_telegram_parquets()
    final_info = finalize(processed_pairs)

    trigger = TriggerDagRunOperator(
        task_id="trigger_transform",
        trigger_dag_id="kraken_transform",
        conf={"pairs": processed_pairs, "base_minutes": 5},
        wait_for_completion=False,
    )

    final_info >> trigger


kraken_ingest_dag = kraken_ingest()