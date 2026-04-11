from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pendulum
import polars as pl
import requests
from airflow.exceptions import AirflowSkipException
from airflow.sdk import dag, task

from common.feature_helpers import (
    _as_bool,
    _as_float,
    _as_int,
    _as_int_list,
    _as_threshold_pairs,
    _as_list,
    _as_str,
    generate_filtered_signals,
)
from common.feature_prep import precompute_all_possible_features
from etl.kraken_api import fetch_futures_ohlc, fetch_ohlc

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("/opt/airflow/airflow-trading/signal_config.json")
STATE_PATH = Path("/opt/airflow/airflow-trading/last_signal_state.json")

LIVE_MAX_ROWS = 500
LOCAL_TZ = "Asia/Kuala_Lumpur"
LIVE_INTERVAL_MINUTES = 15
LIVE_BASE_MINUTES = 15
LIVE_SIGNAL_TIMEFRAME_MODIFIER = 1


def _load_json_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _normalize_live_item(item: Any) -> Dict[str, str]:
    if isinstance(item, str):
        return {"pair": item, "market_type": "spot"}

    if not isinstance(item, dict) or "pair" not in item:
        raise ValueError(f"Invalid live item: {item}")

    market_type = str(item.get("market_type", "spot")).lower().strip()
    if market_type not in {"spot", "future", "xstock"}:
        market_type = "spot"

    return {"pair": str(item["pair"]), "market_type": market_type}


def _normalize_positive_int_list(value: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for x in _as_list(value):
        v = _as_int(x, 0)
        if v > 0 and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _fmt_price(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return "n/a"
    s = f"{v:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _format_local_time_ns(time_ns: int) -> str:
    dt = pendulum.from_timestamp(int(time_ns) / 1_000_000_000.0, tz="UTC").in_timezone(LOCAL_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _as_single_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    try:
        return int(value)
    except Exception:
        return None


def _resolve_stoch_bounds(cfg: dict) -> tuple[float, float]:
    lower = cfg.get("stoch_lower", None)
    upper = cfg.get("stoch_upper", None)

    if lower is not None or upper is not None:
        return _as_float(lower, 30.0), _as_float(upper, 70.0)

    ths = cfg.get("stoch_thresholds", None)
    pairs = _as_threshold_pairs(ths)
    if pairs and len(pairs[0]) >= 2:
        return float(pairs[0][0]), float(pairs[0][1])

    return 30.0, 70.0


def _resolve_trade_window_intervals(cfg: dict) -> list[int]:
    raw = cfg.get("trade_window_interval", [])
    return _normalize_positive_int_list(raw)


def _resolve_live_stoch_col(cfg: dict, df: Optional[pl.DataFrame] = None) -> Optional[str]:
    explicit = str(cfg.get("stoch_col", "") or "").strip()
    if explicit:
        if df is not None and explicit not in df.columns:
            raise ValueError(f"stoch_col='{explicit}' not found in dataframe columns")
        return explicit

    k = _as_single_int(cfg.get("stoch_k"))
    d = _as_single_int(cfg.get("stoch_d"))
    s = _as_single_int(cfg.get("stoch_s"))

    if k and d and s:
        col = f"stoch_k{k}_d{d}_s{s}"
        if df is not None and col not in df.columns:
            raise ValueError(f"Resolved stochastic column '{col}' not found in dataframe columns")
        return col

    stoch_cols = cfg.get("stoch_cols", [])
    if isinstance(stoch_cols, list):
        stoch_cols = [str(x).strip() for x in stoch_cols if str(x).strip()]

        if len(stoch_cols) == 1:
            col = stoch_cols[0]
            if df is not None and col not in df.columns:
                raise ValueError(f"Resolved stochastic column '{col}' not found in dataframe columns")
            return col

        if df is not None:
            available = [c for c in stoch_cols if c in df.columns]
            if len(available) == 1:
                return available[0]

    return None


def _estimate_needed_bars(pair_cfg: dict) -> int:
    base_min = _as_int(pair_cfg.get("BASE_MINUTES", LIVE_BASE_MINUTES), LIVE_BASE_MINUTES)
    modifier = _as_int(pair_cfg.get("signal_timeframe_modifier", LIVE_SIGNAL_TIMEFRAME_MODIFIER), LIVE_SIGNAL_TIMEFRAME_MODIFIER)

    ma_periods = _as_int_list(pair_cfg.get("ma_periods", []))
    entry_lookback = _as_int(pair_cfg.get("entry_lookback_units", 0), 0)
    stoch_k = _as_int_list(pair_cfg.get("stoch_k", []))
    stoch_d = _as_int_list(pair_cfg.get("stoch_d", []))
    stoch_s = _as_int_list(pair_cfg.get("stoch_s", []))
    bbw_periods = _as_int_list(pair_cfg.get("bbw_periods", []))

    max_units = 0
    for x in ma_periods:
        max_units = max(max_units, x)

    if isinstance(entry_lookback, int):
        max_units = max(max_units, entry_lookback)

    for x in stoch_k:
        max_units = max(max_units, x)
    for x in stoch_d:
        max_units = max(max_units, x)
    for x in stoch_s:
        max_units = max(max_units, x)
    for x in bbw_periods:
        max_units = max(max_units, x)

    if max_units <= 0:
        return 120

    bars = int(max_units * modifier + 20)
    return max(bars, int((120 / max(base_min, 1))))


def _load_recent_live_ohlc(pair: str, market_type: str, interval_minutes: int = LIVE_INTERVAL_MINUTES, limit_rows: int = 300) -> pl.DataFrame:
    """
    Live OHLC is fetched directly from Kraken to avoid delayed batch tables.
    """
    market_type = (market_type or "spot").lower().strip()

    if market_type == "future":
        df_pd, _ = fetch_futures_ohlc(pair, interval_minutes)
    else:
        df_pd, _ = fetch_ohlc(pair, interval_minutes, market_type=market_type)

    if df_pd is None or df_pd.empty:
        return pl.DataFrame()

    df = pl.from_pandas(df_pd)

    if "vwap" in df.columns:
        df = df.drop("vwap")
    if "count" in df.columns:
        df = df.drop("count")

    df = df.with_columns([
        pl.col("time").cast(pl.Datetime("ns", "UTC")),
        pl.col("time_ns").cast(pl.Int64),
        pl.col("open").cast(pl.Float32),
        pl.col("high").cast(pl.Float32),
        pl.col("low").cast(pl.Float32),
        pl.col("close").cast(pl.Float32),
        pl.col("volume").cast(pl.Float32),
        pl.lit(0.0).cast(pl.Float32).alias("spread"),
        pl.lit(0.0).cast(pl.Float32).alias("funding_rate"),
    ])

    df = df.sort("time").with_row_index("idx")

    if df.height > limit_rows:
        df = df.tail(limit_rows)

    if df.height > 1:
        df = df[:-1]

    return df


def _resolve_prices_for_row(row: dict, cfg: dict) -> tuple[float, float, float]:
    side = int(row["side"])
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])

    if side == 1:
        limit_price = 0.5 * (close + low)
    else:
        limit_price = 0.5 * (close + high)

    sl_pct = _as_float(cfg.get("SL", cfg.get("sl", 0.0)), 0.0)
    tp_pct = _as_float(cfg.get("TP", cfg.get("tp", 0.0)), 0.0)

    if side == 1:
        sl_price = limit_price * (1.0 - sl_pct / 100.0)
        tp_price = limit_price * (1.0 + tp_pct / 100.0)
    else:
        sl_price = limit_price * (1.0 + sl_pct / 100.0)
        tp_price = limit_price * (1.0 - tp_pct / 100.0)

    return limit_price, sl_price, tp_price


def _cooldown_key(row: dict) -> str:
    return f"{row.get('pair', '')}|{row.get('market_type', '')}|{row.get('side', '')}|{row.get('regime_id', '')}"


def _signal_key(row: dict) -> str:
    return f"{row.get('pair', '')}|{row.get('market_type', '')}|{row.get('time_ns', '')}|{row.get('side', '')}|{row.get('regime_id', '')}"


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


def _last_sent_time_ns(state: dict, cooldown_key: str) -> Optional[int]:
    prev = state.get(cooldown_key)
    if isinstance(prev, dict):
        for k in ("time_ns", "last_time_ns", "sent_time_ns"):
            if k in prev:
                try:
                    return int(prev[k])
                except Exception:
                    return None
    if isinstance(prev, (int, float, str)):
        try:
            return int(prev)
        except Exception:
            return None
    return None


def _within_trade_window(signal_time_ns: int, state: dict, cooldown_key: str, trade_window_intervals: list[int]) -> bool:
    if not trade_window_intervals:
        return True

    prev_time_ns = _last_sent_time_ns(state, cooldown_key)
    if prev_time_ns is None:
        return True

    elapsed_hours = (int(signal_time_ns) - int(prev_time_ns)) / 3_600_000_000_000.0
    return any(elapsed_hours >= float(x) for x in trade_window_intervals)


def _build_message(rows: List[dict]) -> str:
    lines = ["Trading signal alert"]
    for r in rows:
        side = "BUY" if int(r["side"]) == 1 else "SELL"
        lines.append(
            f"{side} | {r['pair']} | {r.get('market_type')} | "
            f"time={r.get('time_local')} | "
            f"limit={_fmt_price(r.get('limit_price'))} | "
            f"SL={_fmt_price(r.get('sl_price'))} | "
            f"TP={_fmt_price(r.get('tp_price'))} | "
            f"within_trade_window={bool(r.get('within_trade_window', False))}"
        )
    return "\n".join(lines)


@dag(
    dag_id="kraken_signal_ingest",
    schedule="*/15 * * * *",
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
        if not params:
            raise AirflowSkipException(f"No per-pair live config for {pair}")

        live = dict(params)
        live["pair"] = pair
        live["market_type"] = market_type
        live.setdefault("BASE_MINUTES", LIVE_BASE_MINUTES)
        live.setdefault("signal_timeframe_modifier", LIVE_SIGNAL_TIMEFRAME_MODIFIER)

        logger.info("LIVE build_one start pair=%s market_type=%s", pair, market_type)
        logger.debug("LIVE raw params pair=%s params=%s", pair, params)

        needed_rows = _estimate_needed_bars(live)
        limit_rows = min(needed_rows, LIVE_MAX_ROWS)

        logger.info(
            "LIVE history request pair=%s market_type=%s needed_rows=%s limit_rows=%s",
            pair,
            market_type,
            needed_rows,
            limit_rows,
        )

        # Live OHLC is fetched directly from Kraken.
        # The fetched frame becomes the source for feature generation and signal filtering.
        df_live = _load_recent_live_ohlc(pair, market_type, interval_minutes=LIVE_INTERVAL_MINUTES, limit_rows=limit_rows)

        logger.info(
            "LIVE df_live loaded pair=%s market_type=%s rows=%s cols=%s",
            pair,
            market_type,
            df_live.height,
            df_live.columns,
        )

        if df_live.is_empty():
            raise AirflowSkipException(f"No OHLC data for {pair}")

        df_feat = df_live.clone()

        # precompute_all_possible_features expands the live indicator columns used by
        # generate_filtered_signals. The frame must retain the same live candle structure.
        df_feat, live = precompute_all_possible_features(df_feat, live)

        stoch_lower, stoch_upper = _resolve_stoch_bounds(live)
        stoch_tol = _as_float(live.get("stoch_threshold_tolerance", 10.0), 10.0)

        logger.info(
            "LIVE DEBUG stoch config pair=%s lower=%s upper=%s tol=%s",
            pair,
            stoch_lower,
            stoch_upper,
            stoch_tol,
        )

        logger.info(
            "LIVE DEBUG last candles pair=%s rows=%s data=%s",
            pair,
            df_feat.height,
            df_feat.select([
                "idx",
                "time_ns",
                "close",
                "high",
                "low",
                "stoch_k12_d8_s4",
                "breakout_2u",
            ]).tail(5).to_dicts(),
        )

        logger.info(
            "LIVE features computed pair=%s market_type=%s rows=%s cols=%s",
            pair,
            market_type,
            df_feat.height,
            df_feat.columns,
        )

        # generate_filtered_signals applies the live regime filters on the freshly built frame.
        # An empty result means no signal at the current closed candle.
        df_signals, stats = generate_filtered_signals(
            df_feat,
            live,
            df_context=df_feat,
            return_stats=True,
        )

        logger.info(
            "LIVE signal stats pair=%s market_type=%s stats=%s",
            pair,
            market_type,
            stats,
        )

        if df_signals.is_empty():
            raise AirflowSkipException(f"Market conditions not met for {pair}")

        df_signals = df_signals.with_columns([
            pl.lit(pair).alias("pair"),
            pl.lit(market_type).alias("market_type"),
        ]).sort("time_ns")

        latest_ts = df_signals["time_ns"].max()
        latest_rows = df_signals.filter(pl.col("time_ns") == latest_ts)

        logger.info(
            "LIVE latest signal candidates pair=%s market_type=%s latest_ts=%s count=%s",
            pair,
            market_type,
            latest_ts,
            latest_rows.height,
        )

        if latest_rows.is_empty():
            raise RuntimeError(f"Signal frame produced rows but latest selection failed for {pair}")

        state = _read_last_state()
        trade_window_intervals = _resolve_trade_window_intervals(params)

        rows = []
        for row in latest_rows.select(
            ["pair", "market_type", "time_ns", "side", "regime_id", "idx"]
        ).to_dicts():
            signal_time_ns = int(row["time_ns"])

            price_frame = (
                df_feat
                .filter(pl.col("time_ns") == signal_time_ns)
                .select(["close", "high", "low"])
                .head(1)
            )

            logger.info(
                "LIVE PRICE LOOKUP pair=%s time_ns=%s found=%s",
                pair,
                signal_time_ns,
                price_frame.height,
            )

            if price_frame.is_empty():
                raise RuntimeError(
                    f"Price lookup failed for {pair} at time_ns={signal_time_ns}. "
                    "Signal row exists, but candle row could not be matched."
                )

            row["close"] = float(price_frame["close"][0])
            row["high"] = float(price_frame["high"][0])
            row["low"] = float(price_frame["low"][0])

            limit_price, sl_price, tp_price = _resolve_prices_for_row(row, params)
            cooldown_key = _cooldown_key(row)

            row["combo"] = (
                ("MA" if _as_int_list(params.get("ma_periods", [])) else "RAW")
                + ("+STOCH" if _as_bool(params.get("use_stochastic", False), False) else "")
                + ("+LOOKBACK" if _as_int_list(params.get("entry_lookback_units", [])) else "")
                + ("+BBW" if _as_bool(params.get("use_bbw", False), False) else "")
            )
            row["time_local"] = _format_local_time_ns(signal_time_ns)
            row["limit_price"] = limit_price
            row["sl_price"] = sl_price
            row["tp_price"] = tp_price
            row["within_trade_window"] = _within_trade_window(
                signal_time_ns,
                state,
                cooldown_key,
                trade_window_intervals,
            )

            logger.info(
                "LIVE candidate pair=%s market_type=%s side=%s regime_id=%s time_ns=%s time_local=%s "
                "close=%s limit=%s sl=%s tp=%s within_trade_window=%s",
                row.get("pair"),
                row.get("market_type"),
                row.get("side"),
                row.get("regime_id"),
                row.get("time_ns"),
                row.get("time_local"),
                row.get("close"),
                _fmt_price(row.get("limit_price")),
                _fmt_price(row.get("sl_price")),
                _fmt_price(row.get("tp_price")),
                row.get("within_trade_window"),
            )

            rows.append(row)

        if not rows:
            raise AirflowSkipException(f"No final live rows for {pair}")

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
            raise AirflowSkipException("No live signal to send")

        token = os.getenv("TG_BOT_TOKEN_SIGNAL")
        chat_id = os.getenv("TG_SIGNAL_CHANNEL_ID")
        if not token or not chat_id:
            raise RuntimeError("TG_BOT_TOKEN_SIGNAL / TG_SIGNAL_CHANNEL_ID are required")

        all_rows = []
        for r in valid:
            all_rows.extend(r.get("rows", []))

        if not all_rows:
            raise AirflowSkipException("No rows to send")

        state = _read_last_state()

        fresh_rows = []
        for row in all_rows:
            key = _signal_key(row)
            if state.get(key):
                continue
            fresh_rows.append(row)

        if not fresh_rows:
            raise AirflowSkipException("All live signals are duplicates")

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
            exact_key = _signal_key(row)
            cooldown_key = _cooldown_key(row)
            sent_time_ns = int(row["time_ns"])

            state[exact_key] = {
                "sent": True,
                "time_ns": sent_time_ns,
            }
            state[cooldown_key] = {
                "time_ns": sent_time_ns,
                "pair": row.get("pair"),
                "market_type": row.get("market_type"),
                "side": int(row.get("side", 0)),
                "regime_id": int(row.get("regime_id", 0)),
            }

        _write_last_state(state)

        return {"sent": True, "count": len(fresh_rows), "status_code": resp.status_code}

    cfg = load_cfg()
    items = resolve_pairs(cfg)
    mapped = build_one.partial(cfg=cfg).expand(item=items)
    send_telegram(mapped)


kraken_signal_ingest_dag = kraken_signal_ingest()