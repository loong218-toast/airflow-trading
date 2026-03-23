from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pendulum
import polars as pl
import requests
from airflow.sdk import dag, task, Variable

from etl.db import get_engine
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


def _market_type_for_pair(pair: str) -> str:
    return "future" if pair.startswith("PF_") else "spot"


def _estimate_needed_bars(cfg: dict) -> int:
    live = cfg.get("live_signal", {})

    base_min = int(cfg.get("BASE_MINUTES", 5))
    modifier = int(cfg.get("signal_timeframe_modifier", 3))

    ma_periods = live.get("ma_periods", cfg.get("ma_periods", [])) or []
    if isinstance(ma_periods, (int, float)):
        ma_periods = [ma_periods]

    entry_lookback = int(live.get("entry_lookback_units", cfg.get("entry_lookback_units", 0)) or 0)

    stoch_k = live.get("stoch_k", cfg.get("stoch_k", [])) or []
    if isinstance(stoch_k, (int, float)):
        stoch_k = [stoch_k]

    bbw_periods = live.get("bbw_periods", cfg.get("bbw_periods", [])) or []
    if isinstance(bbw_periods, (int, float)):
        bbw_periods = [bbw_periods]

    max_units = 0
    for x in ma_periods:
        max_units = max(max_units, int(x))
    max_units = max(max_units, entry_lookback)

    for x in stoch_k:
        max_units = max(max_units, int(x))

    for x in bbw_periods:
        max_units = max(max_units, int(x))

    if max_units <= 0:
        return 300

    bars = int(max_units * modifier + 20)
    return max(bars, int((120 / max(base_min, 1))))


def _load_recent_df_main(pair: str, market_type: str, limit_rows: int) -> pl.DataFrame:
    engine = get_engine(_db_uri())

    query = """
        SELECT
            pair, market_type, time, time_ns, open, high, low, close,
            volume, funding_rate, spread, era_int
        FROM df_main
        WHERE pair = :pair AND market_type = :market_type
        ORDER BY time_ns DESC
        LIMIT :limit_rows
    """

    try:
        import pandas as pd
        with engine.connect() as conn:
            df_pd = pd.read_sql_query(
                query,
                conn,
                params={"pair": pair, "market_type": market_type, "limit_rows": int(limit_rows)},
            )
    except Exception as e:
        raise RuntimeError(f"Failed loading df_main for {pair} ({market_type}): {e}") from e

    if df_pd is None or df_pd.empty:
        return pl.DataFrame()

    df = pl.from_pandas(df_pd)

    df = df.with_columns([
        pl.col("time").cast(pl.Datetime("ns", "UTC")).alias("time"),
        pl.col("time_ns").cast(pl.Int64),
        pl.col("open").cast(pl.Float32),
        pl.col("high").cast(pl.Float32),
        pl.col("low").cast(pl.Float32),
        pl.col("close").cast(pl.Float32),
        pl.col("volume").cast(pl.Float32),
        pl.col("spread").cast(pl.Float32),
        pl.col("funding_rate").cast(pl.Float32),
        pl.col("era_int").cast(pl.Int64),
    ])

    # Query returns newest-first, feature code expects ascending order.
    df = df.sort("time")

    # idx is only needed in memory for signal alignment/debugging.
    df = df.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int64))

    return df


def _build_live_signal_frame(df_main: pl.DataFrame, cfg: dict, pair: str, market_type: str) -> pl.DataFrame:
    live = dict(cfg.get("live_signal", {}))

    live["pair"] = pair
    live["market_type"] = market_type

    # Make sure the signal helper sees scalar values where it expects them.
    # Keep the original grid arrays in cfg, but live_signal should hold the chosen regime.
    if "ma_periods" not in live and "ma_periods" in cfg:
        live["ma_periods"] = cfg["ma_periods"]
    if "entry_lookback_units" not in live and "entry_lookback_units" in cfg:
        live["entry_lookback_units"] = cfg["entry_lookback_units"]
    if "stoch_k" not in live and "stoch_k" in cfg:
        live["stoch_k"] = cfg["stoch_k"]
    if "stoch_d" not in live and "stoch_d" in cfg:
        live["stoch_d"] = cfg["stoch_d"]
    if "stoch_s" not in live and "stoch_s" in cfg:
        live["stoch_s"] = cfg["stoch_s"]
    if "bbw_periods" not in live and "bbw_periods" in cfg:
        live["bbw_periods"] = cfg["bbw_periods"]
    if "bbw_std" not in live and "bbw_std" in cfg:
        live["bbw_std"] = cfg["bbw_std"]

    # Compute all precomputed live features on the slice.
    # This is the same idea as your backtest precompute path, but on the recent live window only.
    df_feat, live = precompute_all_possible_features(df_main, live)

    # generate_filtered_signals expects the chosen stochastic column and thresholds
    # in the runtime config.
    if "stoch_col" not in live and live.get("use_stochastic", False):
        k = live.get("stoch_k")
        d = live.get("stoch_d")
        s = live.get("stoch_s")
        if isinstance(k, list):
            k = k[0] if k else None
        if isinstance(d, list):
            d = d[0] if d else None
        if isinstance(s, list):
            s = s[0] if s else None
        if k is not None and d is not None and s is not None:
            live["stoch_col"] = f"stoch_k{int(k)}_d{int(d)}_s{int(s)}"

    if "stoch_lower" not in live:
        thr = live.get("stoch_thresholds", cfg.get("stoch_thresholds", [[20, 80]]))
        if isinstance(thr, list) and thr and isinstance(thr[0], list) and len(thr[0]) >= 2:
            live["stoch_lower"] = thr[0][0]
            live["stoch_upper"] = thr[0][1]

    df_signals = generate_filtered_signals(df_feat, live, df_main=df_feat)
    return df_signals


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
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _build_message(rows: List[dict]) -> str:
    lines = ["Trading signal alert"]
    for r in rows:
        side = "BUY" if int(r["side"]) == 1 else "SELL"
        lines.append(
            f"{side} | {r['pair']} | regime={r.get('regime_id')} | "
            f"time_ns={r.get('time_ns')} | idx={r.get('idx')}"
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
    def resolve_pairs(cfg: dict) -> list[str]:
        live = cfg.get("live_signal", {})

        # per-pair map: {"XXBTZUSD": {...}, "PF_XBTUSD": {...}}
        if isinstance(live, dict) and live and all(isinstance(v, dict) for v in live.values()):
            return list(live.keys())

        # single-pair legacy form
        pair = live.get("pair")
        if isinstance(pair, str):
            return [pair]
        if isinstance(pair, list):
            return [str(x) for x in pair]

        raise ValueError("No pairs found in signal_config.json")

    @task()
    def build_one(pair: str, cfg: dict) -> dict:
        market_type = _market_type_for_pair(pair)
        needed_rows = _estimate_needed_bars(cfg)
        limit_rows = min(needed_rows, LIVE_MAX_ROWS)

        df_main = _load_recent_df_main(pair, market_type, limit_rows)

        live_all = cfg.get("live_signal", {})
        params = live_all.get(pair, {}) if isinstance(live_all, dict) else {}

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

        df_signals = _build_live_signal_frame(df_main, cfg, pair, market_type)
        latest = _pick_latest_signal(df_signals)

        if latest.is_empty():
            return {
                "pair": pair,
                "market_type": market_type,
                "has_signal": False,
                "reason": "no latest signal",
            }

        # Convert to records and keep only meaningful rows.
        rows = latest.select(["pair", "time_ns", "side", "regime_id", "idx"]).to_dicts()

        state = _read_last_state()
        fresh_rows = []
        for row in rows:
            key = _signal_key(row)
            if state.get(key):
                continue
            fresh_rows.append(row)

        if not fresh_rows:
            return {
                "pair": pair,
                "market_type": market_type,
                "has_signal": False,
                "reason": "duplicate signal",
            }

        return {
            "pair": pair,
            "market_type": market_type,
            "has_signal": True,
            "rows": fresh_rows,
            "message": _build_message([{**r, "pair": pair} for r in fresh_rows]),
        }

    @task()
    def send_telegram(results: list[dict]) -> dict:
        valid = [r for r in results if isinstance(r, dict) and r.get("has_signal")]
        if not valid:
            logger.info("No valid signals found.")
            return {"sent": False, "count": 0}

        token = Variable.get("TELEGRAM_BOT_TOKEN")
        chat_id = Variable.get("TELEGRAM_CHAT_ID")

        all_rows = []
        for r in valid:
            all_rows.extend(r.get("rows", []))

        if not all_rows:
            return {"sent": False, "count": 0}

        message = _build_message(all_rows)
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

        state = _read_last_state()
        for row in all_rows:
            state[_signal_key(row)] = True
        _write_last_state(state)

        return {"sent": True, "count": len(all_rows), "status_code": resp.status_code}

    cfg = load_cfg()
    pairs = resolve_pairs(cfg)

    mapped = build_one.partial(cfg=cfg).expand(pair=pairs)
    send_telegram(mapped)


kraken_signal_ingest_dag = kraken_signal_ingest()