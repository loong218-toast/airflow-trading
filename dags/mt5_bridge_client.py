from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pendulum
import requests
import seaborn as sns
from airflow.exceptions import AirflowSkipException
from airflow.sdk import dag, task

logger = logging.getLogger(__name__)

LOCAL_TZ = pendulum.timezone("Asia/Kuala_Lumpur")

MT5_BRIDGE_URL = os.getenv("MT5_BRIDGE_URL", "http://host.docker.internal:8000")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN_REPORT")
TG_CHAT_ID = os.getenv("TG_REPORT_CHANNEL_ID")

REPORT_DIR = Path("/opt/airflow/airflow-trading/data_lake/trend_report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = Path("/opt/airflow/airflow-trading/data_lake/trend_report_state.json")

TIMEFRAME = "M5"
EXPECTED_BAR_MINUTES = 5

LOOKBACK_DAYS = 7
LOOKBACK_BUFFER_DAYS = 3
TRADE_HORIZON_HOURS = 8

REPORT_INSTRUMENT_CONFIG = {
    "BTCUSD": {"mt5_symbol": "BTCUSD", "tp_pct": 0.7, "sl_pct": 1.4},
    "UK100": {"mt5_symbol": "UK100", "tp_pct": 0.07, "sl_pct": 0.18},
    "AUDJPY": {"mt5_symbol": "AUDJPY", "tp_pct": 0.15, "sl_pct": 0.20},
    "USDCHF": {"mt5_symbol": "USDCHF", "tp_pct": 0.25, "sl_pct": 0.60},
}


class MT5BridgeClient:
    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def rates_range(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        url = f"{self.base_url}/rates_range/{symbol}"
        params = {"timeframe": timeframe, "start": start, "end": end}

        logger.info("MT5 fetch start symbol=%s timeframe=%s start=%s end=%s", symbol, timeframe, start, end)
        r = requests.get(url, params=params, timeout=self.timeout)
        logger.info("MT5 fetch response symbol=%s status=%s", symbol, r.status_code)
        r.raise_for_status()

        data = r.json()
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        df = pd.DataFrame(data)
        if df.empty:
            logger.warning("MT5 fetch returned empty data symbol=%s timeframe=%s", symbol, timeframe)
            return df

        if "time_ns" in df.columns:
            df["dt"] = pd.to_datetime(df["time_ns"], unit="ns", utc=True)
        elif "time_msc" in df.columns:
            df["dt"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
        elif "time" in df.columns:
            if np.issubdtype(df["time"].dtype, np.number):
                median = float(pd.Series(df["time"]).dropna().median())
                unit = "ms" if median > 1e12 else "s"
                df["dt"] = pd.to_datetime(df["time"], unit=unit, utc=True)
            else:
                df["dt"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        else:
            raise ValueError("Bridge response must contain time, time_msc, or time_ns")

        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = (
            df.dropna(subset=["dt", "open", "high", "low", "close"])
            .sort_values("dt")
            .drop_duplicates(subset=["dt"], keep="last")
            .reset_index(drop=True)
        )

        if df.empty:
            logger.warning("MT5 fetch became empty after cleaning symbol=%s timeframe=%s", symbol, timeframe)
            return df

        logger.info(
            "MT5 fetch success symbol=%s timeframe=%s rows=%s span=%s -> %s",
            symbol,
            timeframe,
            len(df),
            df["dt"].min(),
            df["dt"].max(),
        )
        return df


def _read_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def _today_key() -> str:
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")


def _period_start(hours_back: int) -> tuple[str, str]:
    end = pd.Timestamp.now(tz="UTC").floor("s")
    start = end - pd.Timedelta(hours=hours_back)
    return start.isoformat(), end.isoformat()


def _add_local_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date_myt"] = out["dt"].dt.tz_convert(LOCAL_TZ).dt.normalize()
    return out


def _day_labels() -> list[pd.Timestamp]:
    today = pd.Timestamp.now(tz="UTC").tz_convert(LOCAL_TZ).normalize()
    return [today - pd.Timedelta(days=i) for i in range(LOOKBACK_DAYS - 1, -1, -1)]


def _day_label(ts: pd.Timestamp) -> str:
    return ts.tz_convert(LOCAL_TZ).strftime("%d %b")


def _empty_side_stats() -> dict[str, Any]:
    return {
        "win_rate": 0.0,
        "expectancy_r": 0.0,
        "wins": 0,
        "losses": 0,
        "resolved": 0,
        "censored": 0,
        "total": 0,
        "resolve_rate_pct": 0.0,
        "censor_rate_pct": 0.0,
    }


def _resolve_side_stats(
    anchor_df: pd.DataFrame,
    full_df: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
) -> dict[str, Any]:
    if anchor_df.empty or full_df.empty or len(full_df) < 2:
        return {"buy": _empty_side_stats(), "sell": _empty_side_stats()}

    full_df = full_df.sort_values("dt").reset_index(drop=True)
    rr = tp_pct / sl_pct if sl_pct > 0 else np.nan
    horizon = pd.Timedelta(hours=TRADE_HORIZON_HOURS)

    out: dict[str, Any] = {}
    anchor_indices = [int(i) for i in anchor_df.index if 0 <= int(i) < len(full_df)]

    for side in ("buy", "sell"):
        wins = 0
        losses = 0
        censored = 0

        for anchor_idx in anchor_indices:
            entry_row = full_df.iloc[anchor_idx]
            entry_time = entry_row["dt"]
            entry = float(entry_row["close"])
            cutoff = entry_time + horizon

            future = full_df.iloc[anchor_idx + 1 :].copy()
            future = future[future["dt"] <= cutoff]

            if future.empty:
                censored += 1
                continue

            if side == "buy":
                tp = entry * (1.0 + tp_pct / 100.0)
                sl = entry * (1.0 - sl_pct / 100.0)
            else:
                tp = entry * (1.0 - tp_pct / 100.0)
                sl = entry * (1.0 + sl_pct / 100.0)

            hit = None
            for _, bar in future.iterrows():
                high = float(bar["high"])
                low = float(bar["low"])

                if side == "buy":
                    tp_hit = high >= tp
                    sl_hit = low <= sl
                else:
                    tp_hit = low <= tp
                    sl_hit = high >= sl

                # Conservative same-bar rule: if both touch in the same candle, count SL first.
                if tp_hit and sl_hit:
                    hit = "sl"
                    break
                if sl_hit:
                    hit = "sl"
                    break
                if tp_hit:
                    hit = "tp"
                    break

            if hit is None:
                censored += 1
            elif hit == "tp":
                wins += 1
            else:
                losses += 1

        resolved = wins + losses
        total = resolved + censored

        if resolved == 0:
            win_rate = 0.0
            expectancy_r = 0.0
        else:
            win_rate = 100.0 * wins / resolved
            expectancy_r = (wins * rr - losses) / resolved

        out[side] = {
            "win_rate": float(win_rate),
            "expectancy_r": float(expectancy_r),
            "wins": int(wins),
            "losses": int(losses),
            "resolved": int(resolved),
            "censored": int(censored),
            "total": int(total),
            "resolve_rate_pct": float(100.0 * resolved / total) if total > 0 else 0.0,
            "censor_rate_pct": float(100.0 * censored / total) if total > 0 else 0.0,
        }

    return out


def _format_count_label(resolved: int, censored: int) -> str:
    return f"R{resolved}/C{censored}"


def _make_plot(symbol_key: str, cfg: dict[str, Any], day_rows: list[dict[str, Any]], out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    rows = []
    for r in day_rows:
        rows.append(
            {
                "day": r["day_label"],
                "side": "BUY",
                "win_rate": r["buy"]["win_rate"],
                "expectancy_r": r["buy"]["expectancy_r"],
                "resolved": r["buy"]["resolved"],
                "censored": r["buy"]["censored"],
            }
        )
        rows.append(
            {
                "day": r["day_label"],
                "side": "SELL",
                "win_rate": r["sell"]["win_rate"],
                "expectancy_r": r["sell"]["expectancy_r"],
                "resolved": r["sell"]["resolved"],
                "censored": r["sell"]["censored"],
            }
        )

    plot_df = pd.DataFrame(rows)
    day_order = [r["day_label"] for r in day_rows]
    plot_df["day"] = pd.Categorical(plot_df["day"], categories=day_order, ordered=True)
    plot_df = plot_df.sort_values(["day", "side"]).reset_index(drop=True)

    palette = {"BUY": "#2ca02c", "SELL": "#d62728"}
    censored_color = "#bdbdbd"

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), constrained_layout=True)

    ax0 = axes[0]
    sns.barplot(
        data=plot_df,
        x="day",
        y="win_rate",
        hue="side",
        hue_order=["BUY", "SELL"],
        order=day_order,
        palette=palette,
        ax=ax0,
        errorbar=None,
    )
    ax0.set_title(f"{symbol_key} | Daily report | M5 | TP {cfg['tp_pct']}% | SL {cfg['sl_pct']}% | Horizon {TRADE_HORIZON_HOURS}h")
    ax0.set_xlabel("")
    ax0.set_ylabel("Win rate % (resolved only)")
    ax0.legend(title="Side", loc="upper right")
    for container in ax0.containers:
        labels = []
        for bar in container:
            h = bar.get_height()
            labels.append(f"{h:.1f}%" if np.isfinite(h) else "")
        ax0.bar_label(container, labels=labels, padding=3, fontsize=9)

    ax1 = axes[1]
    sns.barplot(
        data=plot_df,
        x="day",
        y="expectancy_r",
        hue="side",
        hue_order=["BUY", "SELL"],
        order=day_order,
        palette=palette,
        ax=ax1,
        errorbar=None,
    )
    ax1.axhline(0, linestyle="--", linewidth=1, color="0.35")
    ax1.set_xlabel("")
    ax1.set_ylabel("Expectancy R (resolved only)")
    ax1.legend(title="Side", loc="upper right")
    for container in ax1.containers:
        labels = []
        for bar in container:
            h = bar.get_height()
            labels.append(f"{h:+.2f}R" if np.isfinite(h) else "")
        ax1.bar_label(container, labels=labels, padding=3, fontsize=9)

    ax2 = axes[2]
    x = np.arange(len(day_order))
    width = 0.34

    buy_rows = plot_df[plot_df["side"] == "BUY"].set_index("day").reindex(day_order)
    sell_rows = plot_df[plot_df["side"] == "SELL"].set_index("day").reindex(day_order)

    buy_resolved = buy_rows["resolved"].fillna(0).to_numpy(dtype=float)
    buy_censored = buy_rows["censored"].fillna(0).to_numpy(dtype=float)
    sell_resolved = sell_rows["resolved"].fillna(0).to_numpy(dtype=float)
    sell_censored = sell_rows["censored"].fillna(0).to_numpy(dtype=float)

    buy_x = x - width / 2
    sell_x = x + width / 2

    buy_resolved_bars = ax2.bar(buy_x, buy_resolved, width, label="BUY resolved", color=palette["BUY"], alpha=0.95)
    buy_censored_bars = ax2.bar(
        buy_x,
        buy_censored,
        width,
        bottom=buy_resolved,
        label="BUY censored",
        color=censored_color,
        alpha=0.7,
        hatch="//",
        edgecolor="0.45",
    )

    sell_resolved_bars = ax2.bar(sell_x, sell_resolved, width, label="SELL resolved", color=palette["SELL"], alpha=0.95)
    sell_censored_bars = ax2.bar(
        sell_x,
        sell_censored,
        width,
        bottom=sell_resolved,
        label="SELL censored",
        color=censored_color,
        alpha=0.7,
        hatch="//",
        edgecolor="0.45",
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(day_order)
    ax2.set_ylabel("Trade count")
    ax2.set_xlabel("")
    ax2.legend(ncol=2, loc="upper right")

    buy_totals = [f"R{int(r)}/C{int(c)}" for r, c in zip(buy_resolved, buy_censored)]
    sell_totals = [f"R{int(r)}/C{int(c)}" for r, c in zip(sell_resolved, sell_censored)]

    ax2.bar_label(buy_censored_bars, labels=buy_totals, padding=3, fontsize=9)
    ax2.bar_label(sell_censored_bars, labels=sell_totals, padding=3, fontsize=9)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _delete_telegram_message(token: str, chat_id: str, message_id: int) -> None:
    url = f"https://api.telegram.org/bot{token}/deleteMessage"
    resp = requests.post(
        url,
        json={"chat_id": chat_id, "message_id": message_id},
        timeout=30,
    )
    resp.raise_for_status()


def _send_telegram_photo(file_path: Path, caption: str, report_key: str) -> int:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        raise RuntimeError("TG_BOT_TOKEN_REPORT and TG_REPORT_CHANNEL_ID are required")

    state = _read_state()
    day_key = _today_key()

    prev = state.get(report_key, {}).get(day_key)
    if isinstance(prev, dict) and prev.get("message_id") is not None:
        try:
            _delete_telegram_message(TG_BOT_TOKEN, TG_CHAT_ID, int(prev["message_id"]))
            logger.info("Deleted previous telegram report report_key=%s day=%s", report_key, day_key)
        except Exception as exc:
            logger.warning("Could not delete previous telegram report report_key=%s day=%s err=%s", report_key, day_key, exc)

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendPhoto"
    with file_path.open("rb") as f:
        resp = requests.post(
            url,
            data={
                "chat_id": TG_CHAT_ID,
                "caption": caption[:1024],
            },
            files={"photo": f},
            timeout=120,
        )
    resp.raise_for_status()
    payload = resp.json()

    message_id = int(payload["result"]["message_id"])
    state.setdefault(report_key, {})[day_key] = {
        "message_id": message_id,
        "file_path": str(file_path),
        "caption": caption,
    }
    _write_state(state)

    return message_id


def _needed_lookback_hours() -> int:
    return (LOOKBACK_DAYS + LOOKBACK_BUFFER_DAYS) * 24 + TRADE_HORIZON_HOURS


def _build_caption(symbol_key: str, cfg: dict[str, Any], latest_stats: dict[str, Any]) -> str:
    buy = latest_stats["buy"]
    sell = latest_stats["sell"]
    return (
        f"{symbol_key} daily report | M5 | "
        f"BUY wr {buy['win_rate']:.1f}% exp {buy['expectancy_r']:+.2f}R | "
        f"SELL wr {sell['win_rate']:.1f}% exp {sell['expectancy_r']:+.2f}R"
    )


@dag(
    dag_id="mt5_daily_daybucket_report",
    schedule="5 */4 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz=LOCAL_TZ),
    catchup=False,
    max_active_runs=1,
    tags=["mt5", "report", "telegram"],
)
def mt5_daily_daybucket_report():
    @task
    def check_bridge() -> dict[str, Any]:
        client = MT5BridgeClient(MT5_BRIDGE_URL)
        health = client.health()
        logger.info("MT5 bridge healthy: %s", health)
        return health

    @task
    def build_report(symbol_key: str) -> dict[str, Any]:
        client = MT5BridgeClient(MT5_BRIDGE_URL)
        cfg = REPORT_INSTRUMENT_CONFIG[symbol_key]

        lookback_hours = _needed_lookback_hours()
        start, end = _period_start(lookback_hours)
        df_all = client.rates_range(cfg["mt5_symbol"], TIMEFRAME, start, end)

        if df_all.empty:
            logger.warning("No MT5 data for symbol=%s mt5_symbol=%s", symbol_key, cfg["mt5_symbol"])
            raise AirflowSkipException(f"No MT5 data for {symbol_key}")

        df_all = _add_local_date(df_all)

        logger.info(
            "MT5 fetch summary symbol=%s mt5_symbol=%s rows=%s dt_min=%s dt_max=%s",
            symbol_key,
            cfg["mt5_symbol"],
            len(df_all),
            df_all["dt"].min(),
            df_all["dt"].max(),
        )

        day_rows: list[dict[str, Any]] = []
        labels = _day_labels()

        for day_ts in labels:
            day_label = _day_label(day_ts)
            anchor_df = df_all[df_all["date_myt"] == day_ts].copy()

            if anchor_df.empty:
                logger.info("Empty day bucket symbol=%s day=%s rows=0", symbol_key, day_label)
                stats = {"buy": _empty_side_stats(), "sell": _empty_side_stats()}
            else:
                stats = _resolve_side_stats(
                    anchor_df=anchor_df,
                    full_df=df_all,
                    tp_pct=float(cfg["tp_pct"]),
                    sl_pct=float(cfg["sl_pct"]),
                )

            logger.info(
                "Day stats symbol=%s day=%s anchors=%s buy(resolved=%s censored=%s wr=%s exp=%s) sell(resolved=%s censored=%s wr=%s exp=%s)",
                symbol_key,
                day_label,
                len(anchor_df),
                stats["buy"]["resolved"],
                stats["buy"]["censored"],
                stats["buy"]["win_rate"],
                stats["buy"]["expectancy_r"],
                stats["sell"]["resolved"],
                stats["sell"]["censored"],
                stats["sell"]["win_rate"],
                stats["sell"]["expectancy_r"],
            )

            day_rows.append(
                {
                    "day_label": day_label,
                    "buy": stats["buy"],
                    "sell": stats["sell"],
                }
            )

        out_path = REPORT_DIR / f"{symbol_key}_daily_report.png"
        _make_plot(symbol_key, cfg, day_rows, out_path)

        caption = _build_caption(symbol_key, cfg, day_rows[-1])

        return {
            "symbol_key": symbol_key,
            "report_key": f"{symbol_key}_daily_report",
            "file_path": str(out_path),
            "caption": caption,
        }

    @task
    def send_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [
            r for r in reports
            if isinstance(r, dict) and r.get("file_path") and r.get("report_key")
        ]
        if not valid:
            raise AirflowSkipException("No report images to send")

        sent = 0
        for item in valid:
            file_path = Path(item["file_path"])
            caption = str(item["caption"])
            report_key = str(item["report_key"])

            if file_path.exists():
                _send_telegram_photo(file_path, caption, report_key)
                sent += 1

        if sent == 0:
            raise AirflowSkipException("No report images to send")

        return {"sent": sent}

    check_bridge()
    instruments = list(REPORT_INSTRUMENT_CONFIG.keys())
    reports = build_report.expand(symbol_key=instruments)
    send_reports(reports)


mt5_daily_daybucket_report_dag = mt5_daily_daybucket_report()