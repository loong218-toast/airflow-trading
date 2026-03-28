from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import text
import pendulum
import polars as pl
import requests
from airflow.sdk import dag, task, Variable

from etl.db import get_engine

from etl.feature_helpers import (
    _as_int,
    _as_int_list,
    _as_bool,
    _as_float,
    _as_str,
    _as_threshold_pairs,
)

from etl.feature_helpers import precompute_all_possible_features, generate_filtered_signals

from etl.schema import enforce_schema

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("/opt/airflow/airflow-trading/signal_config.json")
STATE_PATH = Path("/opt/airflow/airflow-trading/last_signal_state.json")

LIVE_MAX_ROWS = 500


def _load_json_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _db_uri() -> str:
    db_user = os.getenv("POSTGRES_USER")
    db_pass = os.getenv("POSTGRES_PASSWORD")
    db_host = os.getenv("POSTGRES_HOST", "postgres")
    db_name = os.getenv("POSTGRES_DB", "airflow")
    if not db_user or not db_pass:
        raise RuntimeError("POSTGRES_USER / POSTGRES_PASSWORD are required")
    return f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/{db_name}"

def _normalize_live_item(item: Any) -> Dict[str, str]:
    if isinstance(item, str):
        return {"pair": item, "market_type": "spot"}

    if not isinstance(item, dict) or "pair" not in item:
        raise ValueError(f"Invalid live item: {item}")

    market_type = str(item.get("market_type", "spot")).lower().strip()
    if market_type not in {"spot", "future", "xstock"}:
        market_type = "spot"

    return {
        "pair": str(item["pair"]),
        "market_type": market_type,
    }

def _estimate_needed_bars(pair_cfg: dict) -> int:
    base_min = _as_int(pair_cfg.get("BASE_MINUTES", 5), 5)
    modifier = _as_int(pair_cfg.get("signal_timeframe_modifier", 3), 3)

    ma_periods = _as_int_list(pair_cfg.get("ma_periods", []))
    entry_lookback = _as_int(pair_cfg.get("entry_lookback_units", 0), 0)
    stoch_k = _as_int_list(pair_cfg.get("stoch_k", []))
    bbw_periods = _as_int_list(pair_cfg.get("bbw_periods", []))

    max_units = 0
    for x in ma_periods:
        max_units = max(max_units, x)

    max_units = max(max_units, entry_lookback)

    for x in stoch_k:
        max_units = max(max_units, x)

    for x in bbw_periods:
        max_units = max(max_units, x)

    if max_units <= 0:
        return 300

    bars = int(max_units * modifier + 20)
    return max(bars, int((120 / max(base_min, 1))))

def _load_recent_df_main(pair: str, market_type: str, limit_rows: int) -> pl.DataFrame:
    # 1. Get engine and strip the +psycopg2 to make it ADBC-compatible
    # This matches your working etl/transform.py logic
    uri_obj = get_engine(_db_uri()).url
    uri = f"postgresql://{uri_obj.username}:{uri_obj.password}@{uri_obj.host}:{uri_obj.port or 5432}/{uri_obj.database}"
    
    # 2. Use positional '$' placeholders for ADBC/Postgres
    query = """
        SELECT
            pair, market_type, time, time_ns, open, high, low, close,
            volume, funding_rate, spread
        FROM df_main
        WHERE pair = $1 AND market_type = $2
        ORDER BY time_ns DESC
        LIMIT $3
    """

    try:
        # 3. Use positional parameters in a LIST (not dict) via execute_options
        df = pl.read_database_uri(
            query=query,
            uri=uri,
            engine="adbc",
            execute_options={"parameters": [pair, market_type, int(limit_rows)]}
        )
    except Exception as e:
        logger.error(f"ADBC Signal Load failed for {pair}: {e}")
        raise RuntimeError(f"Polars ADBC failed: {e}")

    if df.is_empty():
        return pl.DataFrame()

    # 4. Enforce schema
    return df.with_columns([
        pl.col("time").cast(pl.Datetime("ns", "UTC")),
        pl.col("time_ns").cast(pl.Int64),
        pl.col("open").cast(pl.Float32),
        pl.col("high").cast(pl.Float32),
        pl.col("low").cast(pl.Float32),
        pl.col("close").cast(pl.Float32),
        pl.col("volume").cast(pl.Float32),
        pl.col("spread").cast(pl.Float32),
        pl.col("funding_rate").cast(pl.Float32),
    ]).sort("time").with_row_index("idx")

def _build_live_signal_frame(
    df_main: pl.DataFrame,
    pair_params: dict,
    pair: str,
    market_type: str,
) -> pl.DataFrame:
    # Build runtime config for the existing signal function
    live = dict(pair_params)

    live["pair"] = pair
    live["market_type"] = market_type

    # Keep ma_periods as the source of truth
    ma_periods = live.get("ma_periods", []) or []
    if isinstance(ma_periods, (int, float)):
        ma_periods = [int(ma_periods)]

    ma_periods = [int(x) for x in ma_periods]
    live["ma_periods"] = _as_int_list(live.get("ma_periods", []))
    live["ma_int"] = (1 << len(live["ma_periods"])) - 1 if live["ma_periods"] else 0

    if "stoch_col" not in live and _as_bool(live.get("use_stochastic", False), False):
        k = _as_int(live.get("stoch_k", 0), 0)
        d = _as_int(live.get("stoch_d", 0), 0)
        s = _as_int(live.get("stoch_s", 0), 0)
        if k and d and s:
            live["stoch_col"] = f"stoch_k{k}_d{d}_s{s}"

    thr = live.get("stoch_thresholds", [[20, 80]])
    thr = _as_threshold_pairs(thr)
    if thr and "stoch_lower" not in live and "stoch_upper" not in live:
        live["stoch_lower"] = thr[0][0]
        live["stoch_upper"] = thr[0][1]

    # Precompute features on the recent slice
    df_feat, live = precompute_all_possible_features(df_main, live)

    # Keep the exact same downstream signal generator
    return generate_filtered_signals(df_feat, live, df_main=df_feat)


def _pick_latest_signal(df_signals: pl.DataFrame) -> pl.DataFrame:
    if df_signals is None or df_signals.is_empty():
        return pl.DataFrame()

    latest_ns = df_signals["time_ns"].max()
    return df_signals.filter(pl.col("time_ns") == latest_ns).sort(["side", "idx"])


def _signal_key(row: dict) -> str:
    return f"{row.get('pair')}|{row.get('time_ns')}|{row.get('side')}|{row.get('regime_id')}"


def _read_last_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_last_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, STATE_PATH)

def _build_message(rows: List[dict]) -> str:
    lines = ["Trading signal alert"]
    for r in rows:
        side = "BUY" if int(r["side"]) == 1 else "SELL"
        lines.append(
            f"{side} | {r['pair']} | {r.get('market_type')} | "
            f"regime={r.get('regime_id')} | time_ns={r.get('time_ns')} | idx={r.get('idx')}"
        )
    return "\n".join(lines)


@dag(
    dag_id="kraken_signal_ingest",
    schedule="*/30 * * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=4,
    tags=["kraken", "signal", "telegram"],
)
def kraken_signal_ingest():

    @task()
    def load_cfg() -> dict:
        cfg = _load_json_file(CONFIG_PATH)
        live = cfg.get("live_signal", {})
        if not isinstance(live, dict):
            raise ValueError("'live_signal' must be a JSON object")
        return cfg

    @task()
    def resolve_pairs(cfg: dict) -> list[dict]:
        live = cfg.get("live_signal", {})

        if isinstance(live, dict) and live and all(isinstance(v, dict) for v in live.values()):
            items = []
            for pair, params in live.items():
                market_type = str(params.get("market_type", "spot")).lower().strip()
                if market_type not in {"spot", "future", "xstock"}:
                    market_type = "spot"
                items.append({"pair": pair, "market_type": market_type})
            return items

        pair = live.get("pair")
        if isinstance(pair, str):
            return [{"pair": pair, "market_type": str(live.get("market_type", "spot")).lower().strip()}]

        if isinstance(pair, list):
            return [
                {"pair": str(x), "market_type": str(live.get("market_type", "spot")).lower().strip()}
                for x in pair
            ]

        raise ValueError("No pairs found in signal_config.json")

    @task()
    def build_one(item: dict, cfg: dict) -> dict:
        pair = str(item["pair"])
        market_type = str(item["market_type"]).lower().strip()

        live_all = cfg.get("live_signal", {})
        params = live_all.get(pair, {}) if isinstance(live_all, dict) else {}

        needed_rows = _estimate_needed_bars(params)
        limit_rows = min(needed_rows, LIVE_MAX_ROWS)

        df_main = _load_recent_df_main(pair, market_type, limit_rows)

        if not params:
            return {
                "pair": pair,
                "market_type": market_type,
                "has_signal": False,
                "reason": "no per-pair live config",
            }

        if df_main.is_empty():
            return {
                "pair": pair,
                "market_type": market_type,
                "has_signal": False,
                "reason": "no df_main rows",
            }

        df_signals = _build_live_signal_frame(df_main, params, pair, market_type)
        latest = _pick_latest_signal(df_signals)

        if latest.is_empty():
            return {
                "pair": pair,
                "market_type": market_type,
                "has_signal": False,
                "reason": "no latest signal",
            }

        latest = latest.with_columns([
            pl.lit(pair).alias("pair"),
            pl.lit(market_type).alias("market_type"),
        ])

        rows = latest.select(["pair", "market_type", "time_ns", "side", "regime_id", "idx"]).to_dicts()

        return {
            "pair": pair,
            "market_type": market_type,
            "has_signal": True,
            "rows": rows,
        }

    @task()
    def send_telegram(results: list[dict]) -> dict:
        valid = [r for r in results if isinstance(r, dict) and r.get("has_signal")]
        if not valid:
            logger.info("No valid signals found.")
            return {"sent": False, "count": 0}

        token = os.getenv("TG_BOT_TOKEN_SIGNAL")
        chat_id = os.getenv("TG_SIGNAL_CHANNEL_ID")

        all_rows = []
        for r in valid:
            all_rows.extend(r.get("rows", []))

        if not all_rows:
            return {"sent": False, "count": 0}

        state = _read_last_state()

        fresh_rows = []
        for row in all_rows:
            key = _signal_key(row)
            if state.get(key):
                continue
            fresh_rows.append(row)

        if not fresh_rows:
            logger.info("All signals are duplicates.")
            return {"sent": False, "count": 0, "reason": "duplicate signal"}

        message = _build_message(fresh_rows)
        logger.info("Telegram message:\n%s", message)

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message,
                "disable_web_page_preview": True,
            },
            timeout=20,
        )

        logger.info("Telegram response status=%s body=%s", resp.status_code, resp.text)
        resp.raise_for_status()

        for row in fresh_rows:
            state[_signal_key(row)] = True
        _write_last_state(state)

        return {"sent": True, "count": len(fresh_rows), "status_code": resp.status_code}

    cfg = load_cfg()
    items = resolve_pairs(cfg)

    mapped = build_one.partial(cfg=cfg).expand(item=items)
    send_telegram(mapped)

kraken_signal_ingest_dag = kraken_signal_ingest()