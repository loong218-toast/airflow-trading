from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests

# Ensure 'dags' folder is in the path so we can import 'etl'
root_dir = os.path.dirname(os.path.abspath(__file__))
dags_path = os.path.join(root_dir, "dags")
if dags_path not in sys.path:
    sys.path.insert(0, dags_path)

from etl.kraken_api import fetch_ohlc, fetch_futures_ohlc, get_tradeable_futures_symbols

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

STATE_FILE = ROOT / "selected_universe.json"

KRAKEN_SPOT_BASE = "https://api.kraken.com/0/public"
KRAKEN_FUTURES_BASE = "https://futures.kraken.com"

VALID_MARKET_TYPES = {"spot", "future", "xstock"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_market_type(market_type: Any) -> str:
    mt = str(market_type or "").lower().strip()
    if mt == "futures":
        mt = "future"
    return mt


def pair_state_key(pair: str, market_type: str) -> str:
    return f"{normalize_market_type(market_type)}|{str(pair).upper().strip()}"


def load_config() -> dict:
    path = ROOT / "pair_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def load_state() -> dict:
    """
    Persistent selection state committed back to the repo.
    Format:
    {
      "selected": [
        {
          "pair": "XXBTZUSD",
          "market_type": "spot",
          "source": "static|top_n|historical",
          "selected_at": "...",
          "last_seen_at": "...",
          "disabled": false
        }
      ]
    }
    """
    if not STATE_FILE.exists():
        return {}

    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if isinstance(raw, dict):
        items = raw.get("selected", raw.get("items", []))
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    state: dict[str, dict] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        pair = row.get("pair")
        market_type = row.get("market_type")
        if not pair or not market_type:
            continue
        state[pair_state_key(str(pair), str(market_type))] = row

    return state


def save_state(state: dict[str, dict]) -> None:
    payload = {
        "updated_at": utc_now_iso(),
        "selected": [state[k] for k in sorted(state.keys())],
    }
    atomic_write_json(STATE_FILE, payload)


def is_disabled(pair: str, market_type: str, state: dict[str, dict], excluded: set[tuple[str, str]]) -> bool:
    key = pair_state_key(pair, market_type)
    if (str(pair).upper(), normalize_market_type(market_type)) in excluded:
        return True
    row = state.get(key)
    return bool(row and row.get("disabled") is True)


def upsert_state(
    state: dict[str, dict],
    pair: str,
    market_type: str,
    source: str,
    volume: float | None = None,
    current_rank: int | None = None,
    locked: bool = False,
) -> None:
    key = pair_state_key(pair, market_type)
    now = utc_now_iso()
    row = state.get(key)

    if row is None:
        row = {
            "pair": str(pair).upper(),
            "market_type": normalize_market_type(market_type),
            "source": source,
            "selected_at": now,
            "disabled": False,
        }

    row["pair"] = str(pair).upper()
    row["market_type"] = normalize_market_type(market_type)
    row["source"] = source
    row["last_seen_at"] = now

    if "selected_at" not in row:
        row["selected_at"] = now

    if volume is not None:
        row["volume"] = float(volume)
    if current_rank is not None:
        row["current_rank"] = int(current_rank)
    if locked:
        row["locked"] = True

    state[key] = row


def fetch_pair(pair: str, market_type: str, interval: int):
    market_type = normalize_market_type(market_type)

    if market_type == "future":
        return fetch_futures_ohlc(pair, interval)

    if market_type in {"spot", "xstock"}:
        return fetch_ohlc(pair, interval, market_type=market_type)

    raise ValueError(f"Unsupported market_type: {market_type}")


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
                raise RuntimeError(f"Failed to send {file_path.name}: {e}") from e

            wait_before_retry = 5 * retries
            print(f"Error occurred: {e}. Retrying in {wait_before_retry}s...")
            time.sleep(wait_before_retry)

    raise RuntimeError(f"Failed to send {file_path.name}")


def get_asset_pairs(aclass_base: str) -> dict:
    params = {"aclass_base": aclass_base, "info": "info"}
    r = requests.get(f"{KRAKEN_SPOT_BASE}/AssetPairs", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken AssetPairs error: {data['error']}")
    return data.get("result", {})


def fetch_spot_tickers() -> dict:
    r = requests.get(f"{KRAKEN_SPOT_BASE}/Ticker", timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken Ticker error: {data['error']}")
    return data.get("result", {})


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _spot_volume_from_ticker(payload: Any) -> float:
    if not isinstance(payload, dict):
        return 0.0

    # Kraken spot Ticker commonly exposes volume as a list; use the last numeric value found.
    for key in ("v", "volume", "vol", "volume24h", "vol24h"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)):
            nums = [_coerce_float(x) for x in value]
            nums = [n for n in nums if n is not None]
            if nums:
                return float(nums[-1])
        else:
            num = _coerce_float(value)
            if num is not None:
                return float(num)

    return 0.0


def rank_spot_like_pairs(aclass_base: str, market_type: str, top_n: int) -> list[dict]:
    if top_n <= 0:
        return []

    pairs = get_asset_pairs(aclass_base)
    tickers = fetch_spot_tickers()

    rows: list[dict] = []
    for pair_code in pairs.keys():
        ticker_payload = tickers.get(pair_code)
        if ticker_payload is None:
            continue

        vol = _spot_volume_from_ticker(ticker_payload)
        rows.append(
            {
                "pair": pair_code,
                "market_type": market_type,
                "volume": vol,
            }
        )

    if not rows:
        return []

    df = pd.DataFrame(rows)
    df = df.sort_values(["volume", "pair"], ascending=[False, True]).head(top_n)

    out: list[dict] = []
    for rank, row in enumerate(df.itertuples(index=False), start=1):
        out.append(
            {
                "pair": str(row.pair).upper(),
                "market_type": market_type,
                "volume": float(row.volume),
                "current_rank": rank,
                "source": "top_n",
            }
        )
    return out


def fetch_futures_tickers_raw() -> pd.DataFrame:
    url = f"{KRAKEN_FUTURES_BASE}/derivatives/api/v3/tickers"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("result") != "success":
        return pd.DataFrame()
    return pd.DataFrame(data.get("tickers", []))


def rank_futures_pairs(top_n: int) -> list[dict]:
    if top_n <= 0:
        return []

    df = fetch_futures_tickers_raw()
    if df.empty:
        return []

    # Ensure there is a symbol column
    if "symbol" not in df.columns:
        possible_cols = [c for c in df.columns if "symbol" in c.lower()]
        if possible_cols:
            df = df.rename(columns={possible_cols[0]: "symbol"})
        else:
            print("❌ No symbol column found in futures tickers")
            return []

    tradeable = get_tradeable_futures_symbols()
    if tradeable:
        tradeable_set = {str(s).upper() for s in tradeable}
        df = df[df["symbol"].astype(str).str.upper().isin(tradeable_set)]

    if df.empty:
        return []

    # Find volume column
    volume_candidates = ["volume", "volume24h", "vol24h", "vol", "usdVolume24h", "tradeVolume"]
    volume_col = next((c for c in volume_candidates if c in df.columns), None)

    if volume_col is None:
        fallback_cols = [c for c in df.columns if "vol" in c.lower() or "volume" in c.lower()]
        volume_col = fallback_cols[0] if fallback_cols else None

    df["_volume"] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0.0) if volume_col else 0.0

    df = df.sort_values(["_volume", "symbol"], ascending=[False, True]).head(top_n)

    out: list[dict] = []
    for rank, row in enumerate(df.itertuples(index=False), start=1):
        vol = getattr(row, volume_col, 0.0) if volume_col else 0.0
        out.append(
            {
                "pair": str(row.symbol).upper(),
                "market_type": "future",
                "volume": float(vol),
                "current_rank": rank,
                "source": "top_n",
            }
        )
    return out


def build_universe(cfg: dict, state: dict[str, dict]) -> list[dict]:
    """
    Build the persistent universe:
    - always include static pairs unless explicitly disabled/excluded
    - add current top-N spot / futures / xstocks
    - keep previously selected symbols even if they fall out of top-N later
    """
    excluded_raw = cfg.get("excluded_pairs", [])
    excluded: set[tuple[str, str]] = set()
    if isinstance(excluded_raw, list):
        for item in excluded_raw:
            if isinstance(item, dict) and item.get("pair") and item.get("market_type"):
                excluded.add((str(item["pair"]).upper(), normalize_market_type(item["market_type"])))
            elif isinstance(item, str) and "|" in item:
                mt, pair = item.split("|", 1)
                excluded.add((pair.upper(), normalize_market_type(mt)))

    universe: dict[str, dict] = {}

    def add_item(pair: str, market_type: str, source: str, volume: float | None = None, current_rank: int | None = None, locked: bool = False):
        if is_disabled(pair, market_type, state, excluded):
            return
        key = pair_state_key(pair, market_type)
        if key not in universe:
            universe[key] = {
                "pair": str(pair).upper(),
                "market_type": normalize_market_type(market_type),
                "source": source,
                "selected_at": utc_now_iso(),
                "last_seen_at": utc_now_iso(),
                "disabled": False,
            }
        upsert_state(state, pair, market_type, source, volume=volume, current_rank=current_rank, locked=locked)
        universe[key] = state[key]

    # 1) Static pairs from config.
    for item in cfg.get("static_pairs", []):
        if not isinstance(item, dict):
            continue
        pair = item.get("pair")
        market_type = normalize_market_type(item.get("market_type", "spot"))
        if not pair:
            continue
        add_item(pair, market_type, "static", locked=True)

    # 2) Current top-N discovery by volume.
    top_n_spots = int(cfg.get("top_n_spots", 0) or 0)
    top_n_futures = int(cfg.get("top_n_futures", 0) or 0)
    top_n_xstocks = int(cfg.get("top_n_xstocks", 0) or 0)

    for row in rank_spot_like_pairs("currency", "spot", top_n_spots):
        add_item(
            row["pair"],
            "spot",
            "top_n",
            volume=row.get("volume"),
            current_rank=row.get("current_rank"),
        )

    for row in rank_spot_like_pairs("tokenized_asset", "xstock", top_n_xstocks):
        add_item(
            row["pair"],
            "xstock",
            "top_n",
            volume=row.get("volume"),
            current_rank=row.get("current_rank"),
        )

    for row in rank_futures_pairs(top_n_futures):
        add_item(
            row["pair"],
            "future",
            "top_n",
            volume=row.get("volume"),
            current_rank=row.get("current_rank"),
        )

    # 3) Keep previously selected pairs forever unless manually disabled/excluded.
    for key, row in state.items():
        if not isinstance(row, dict):
            continue
        if row.get("disabled") is True:
            continue
        pair = row.get("pair")
        market_type = row.get("market_type")
        if not pair or not market_type:
            continue
        if is_disabled(pair, market_type, state, excluded):
            continue
        state_key = pair_state_key(pair, market_type)
        if state_key not in universe:
            row["last_seen_at"] = utc_now_iso()
            universe[state_key] = row

    # Deterministic order: static first, then current top-N by rank/volume, then historical.
    def sort_key(item: dict) -> tuple:
        source = item.get("source", "")
        locked = 1 if item.get("locked") else 0
        rank = int(item.get("current_rank") or 9999)
        volume = float(item.get("volume") or 0.0)
        market_type = str(item.get("market_type", ""))
        pair = str(item.get("pair", ""))
        return (-locked, source, rank, -volume, market_type, pair)

    ordered = sorted(universe.values(), key=sort_key)

    # Write the updated persistent state.
    save_state(state)
    return ordered


def normalize_to_ns(df):
    if df is None or df.empty or "time_ns" not in df.columns:
        return df
    t = int(df["time_ns"].iloc[0])

    if t < 10**10:              # seconds
        df["time_ns"] = df["time_ns"] * 1_000_000_000
    elif t < 10**13:            # milliseconds
        df["time_ns"] = df["time_ns"] * 1_000_000
    # else: already nanoseconds

    df["time_ns"] = df["time_ns"].astype("int64")
    return df


def main() -> None:
    token = os.environ["TG_BOT_TOKEN"]
    chat_id = os.environ["TG_DATA_CHANNEL_ID"]

    cfg = load_config()
    delay_seconds = int(cfg.get("delay_seconds", 3))
    global_intervals = cfg.get("interval_minutes", [5])

    state = load_state()
    universe = build_universe(cfg, state)

    manifest_path = OUT / "sent_manifest.json"
    sent_manifest = load_manifest(manifest_path)

    sent_keys = set()
    for row in sent_manifest:
        if not isinstance(row, dict):
            continue
        pair = row.get("pair")
        interval_minutes = row.get("interval_minutes")
        market_type = normalize_market_type(row.get("market_type", "spot"))
        snapshot_id = str(row.get("snapshot_id", "")).strip()
        if pair is None or interval_minutes is None or not snapshot_id:
            continue
        sent_keys.add((str(pair), int(interval_minutes), market_type, snapshot_id))

    if not universe:
        print("❌ ERROR: No pairs selected for ingestion.")
        return

    for item in universe:
        pair = str(item["pair"]).upper()
        market_type = normalize_market_type(item.get("market_type", "spot"))
        intervals = item.get("interval_minutes", global_intervals)

        if market_type not in VALID_MARKET_TYPES:
            print(f"skip {pair}: unsupported market_type={market_type}")
            continue

        for interval in intervals:
            interval = int(interval)

            try:
                df, last = fetch_pair(pair, market_type, interval)
                df = normalize_to_ns(df)

                if df is None or df.empty:
                    print(f"skip {pair} {market_type} {interval}: empty")
                    time.sleep(delay_seconds)
                    continue

                last_ns = int(last * 1_000_000_000) if last < 10**11 else int(last)
                snapshot_id = f"{pair}_{last_ns}"
                key = (pair, interval, market_type, snapshot_id)

                if key in sent_keys:
                    print(f"skip {pair} {market_type} {interval}: already sent snapshot {snapshot_id}")
                    time.sleep(delay_seconds)
                    continue

                df = df.copy()
                df["pair"] = pair
                df["interval_minutes"] = interval
                df["market_type"] = market_type

                file_path = OUT / f"{pair}_{market_type}_{interval}m.parquet"
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

            except Exception as e:
                print(f"failed {pair} {market_type} {interval}: {e}")
                time.sleep(delay_seconds)
                continue

    atomic_write_json(manifest_path, sent_manifest)


if __name__ == "__main__":
    main()