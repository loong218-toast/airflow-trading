# satellite_ingest.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

import sys
import os

# Ensure 'dags' folder is in the path so we can import 'etl'
root_dir = os.path.dirname(os.path.abspath(__file__))
dags_path = os.path.join(root_dir, "dags")
if dags_path not in sys.path:
    sys.path.insert(0, dags_path)


from etl.kraken_api import fetch_ohlc, fetch_futures_ohlc

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)


def load_config() -> dict:
    path = ROOT / "pair_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_pair(pair: str, market_type: str, interval: int):
    if market_type == "future":
        return fetch_futures_ohlc(pair, interval)
    return fetch_ohlc(pair, interval)


def send_document(token: str, chat_id: str, file_path: Path, caption: str) -> int:
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    with file_path.open("rb") as f:
        r = requests.post(
            url,
            data={
                "chat_id": chat_id,
                "caption": caption,
            },
            files={
                "document": (file_path.name, f),
            },
            timeout=120,
        )
    r.raise_for_status()
    payload = r.json()
    if not payload.get("ok"):
        raise RuntimeError(payload)
    return payload["result"]["message_id"]

def main() -> None:
    token = os.environ["TG_BOT_TOKEN"]
    chat_id = os.environ["TG_DATA_CHANNEL_ID"]

    cfg = load_config()
    delay_seconds = int(cfg.get("delay_seconds", 3))
    global_intervals = cfg.get("interval_minutes", [5, 60, 1440])

    sent_manifest = []

    for item in cfg["pairs"]:
        pair = item["pair"]
        market_type = item.get("market_type", "spot")
        intervals = item.get("interval_minutes", global_intervals)

        for interval in intervals:
            interval = int(interval)

            df, last = fetch_pair(pair, market_type, interval)
            if df is None or df.empty:
                print(f"skip {pair} {interval}: empty")
                time.sleep(delay_seconds)
                continue

            df = df.copy()
            df["pair"] = pair
            df["interval_minutes"] = interval
            df["market_type"] = market_type

            file_path = OUT / f"{pair}_{interval}m.parquet"
            df.to_parquet(file_path, index=False)

            caption = f"{pair} | {market_type} | {interval}m | rows={len(df)}"
            message_id = send_document(token, chat_id, file_path, caption)

            print(f"sent {file_path.name} message_id={message_id}")
            sent_manifest.append(
                {
                    "pair": pair,
                    "interval_minutes": interval,
                    "market_type": market_type,
                    "message_id": message_id,
                    "file": file_path.name,
                }
            )

            time.sleep(delay_seconds)

    (OUT / "sent_manifest.json").write_text(
        json.dumps(sent_manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()