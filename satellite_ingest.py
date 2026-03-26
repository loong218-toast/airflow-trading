# satellite_ingest.py (called from github action, can't read sql)
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

# Ensure 'dags' folder is in the path so we can import 'etl'
root_dir = os.path.dirname(os.path.abspath(__file__))
dags_path = os.path.join(root_dir, "dags")
if dags_path not in sys.path:
    sys.path.insert(0, dags_path)

from etl.kraken_api import fetch_ohlc, fetch_futures_ohlc

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

VALID_MARKET_TYPES = {"spot", "future", "xstock"}


def load_config() -> dict:
    path = ROOT / "pair_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

import pandas as pd

def normalize_to_ns(df):
    if df is None or df.empty or "time_ns" not in df.columns:
        return df
    if int(df["time_ns"].iloc[0]) < 10**11:
        df["time_ns"] = (df["time_ns"] * 1_000_000_000).astype("int64")
    return df

def fetch_pair(pair: str, market_type: str, interval: int):
    market_type = str(market_type).lower().strip()

    if market_type == "future":
        return fetch_futures_ohlc(pair, interval)

    if market_type in {"spot", "xstock"}:
        return fetch_ohlc(pair, interval, market_type=market_type)

    raise ValueError(f"Unsupported market_type: {market_type}")


def atomic_write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def build_snapshot_id(pair: str, time_ns: int) -> str:
    """
    Creates a unique identifier for a specific candle.
    Since time_ns is now 10^18 (nanoseconds), this ensures 
    uniqueness across different pairs at the same timestamp.
    """
    # pair: 'XXBTZUSD', time_ns: 1774466100000000000
    # result: 'XXBTZUSD_1774466100000000000'
    return f"{pair}_{time_ns}"


def send_document(token: str, chat_id: str, file_path: Path, caption: str) -> int:
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    retries = 0
    max_retries = 5

    while retries < max_retries:
        try:
            with file_path.open("rb") as f:
                r = requests.post(
                    url,
                    data={"chat_id": chat_id, "caption": caption},
                    files={"document": (file_path.name, f)},
                    timeout=120,
                )

            if r.status_code == 429:
                wait_time = r.json().get("parameters", {}).get("retry_after", 30)
                print(f"Rate limited. Waiting {wait_time}s to retry {file_path.name}...")
                time.sleep(wait_time)
                retries += 1
                continue

            r.raise_for_status()
            payload = r.json()

            if not payload.get("ok"):
                raise RuntimeError(payload)

            return int(payload["result"]["message_id"])

        except (requests.exceptions.RequestException, RuntimeError) as e:
            retries += 1
            if retries >= max_retries:
                print(f"Failed to send {file_path.name} after {max_retries} attempts.")
                raise

            wait_before_retry = 5 * retries
            print(f"Error occurred: {e}. Retrying in {wait_before_retry}s...")
            time.sleep(wait_before_retry)

    raise RuntimeError(f"Failed to send {file_path.name}")


def main() -> None:
    token = os.environ["TG_BOT_TOKEN"]
    chat_id = os.environ["TG_DATA_CHANNEL_ID"]

    cfg = load_config()
    delay_seconds = int(cfg.get("delay_seconds", 3))
    global_intervals = cfg.get("interval_minutes", [5])

    manifest_path = OUT / "sent_manifest.json"
    sent_manifest = load_manifest(manifest_path)

    sent_keys = set()
    for row in sent_manifest:
        if not isinstance(row, dict):
            continue
        pair = row.get("pair")
        interval_minutes = row.get("interval_minutes")
        market_type = str(row.get("market_type", "spot")).lower().strip()
        snapshot_id = str(row.get("snapshot_id", "")).strip()
        if pair is None or interval_minutes is None or not snapshot_id:
            continue
        sent_keys.add((str(pair), int(interval_minutes), market_type, snapshot_id))

    pairs_list = cfg.get("pairs") or cfg.get("static_pairs")
    
    if pairs_list is None:
        print(f"❌ ERROR: No pairs list found in config (checked 'pairs' and 'static_pairs').")
        return

    for item in pairs_list:
        pair = item["pair"]
        market_type = str(item.get("market_type", "spot")).lower().strip()
        if market_type not in VALID_MARKET_TYPES:
            raise ValueError(f"Unsupported market_type in config: {market_type}")

        intervals = item.get("interval_minutes", global_intervals)

        for interval in intervals:
            interval = int(interval)

            df, last = fetch_pair(pair, market_type, interval)
            df = normalize_to_ns(df)
            if df is None or df.empty:
                print(f"skip {pair} {market_type} {interval}: empty")
                time.sleep(delay_seconds)
                continue

            last_ns = int(last * 1_000_000_000) if last < 10**11 else int(last)
            snapshot_id = build_snapshot_id(pair, last_ns)
            key = (pair, interval, market_type, snapshot_id)

            if key in sent_keys:
                print(f"skip {pair} {market_type} {interval}: already sent snapshot {snapshot_id}")
                time.sleep(delay_seconds)
                continue

            df = df.copy()
            df["pair"] = pair
            df["interval_minutes"] = interval
            df["market_type"] = market_type

            file_path = OUT / f"{pair}_{interval}m.parquet"
            df.to_parquet(file_path, index=False)

            caption = f"{pair} | {market_type} | {interval}m | rows={len(df)} | snap={snapshot_id}"
            message_id = send_document(token, chat_id, file_path, caption)

            print(f"sent {file_path.name} message_id={message_id}")

            sent_manifest.append(
                {
                    "pair": pair,
                    "interval_minutes": interval,
                    "market_type": market_type,
                    "snapshot_id": snapshot_id,
                    "message_id": message_id,
                    "file": file_path.name,
                }
            )
            sent_keys.add(key)
            atomic_write_json(manifest_path, sent_manifest)

            time.sleep(delay_seconds)

    atomic_write_json(manifest_path, sent_manifest)


if __name__ == "__main__":
    main()