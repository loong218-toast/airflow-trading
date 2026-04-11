from __future__ import annotations

import json
import logging
import os
import os.path
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pendulum
import requests
from airflow.sdk import dag, task

logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TG_BOT_TOKEN_SIGNAL", "").strip()
STATE_PATH = Path(
    os.getenv(
        "TG_COMMAND_STATE_PATH",
        "/opt/airflow/airflow-trading/telegram_command_state.json",
    )
)
POLL_LIMIT = int(os.getenv("TG_POLL_LIMIT", "50"))
POLL_TIMEOUT_SEC = int(os.getenv("TG_POLL_TIMEOUT_SEC", "0"))
LOCAL_TZ = os.getenv("LOCAL_TZ", "Asia/Kuala_Lumpur")

_allowed_raw = os.getenv("TG_ALLOWED_CHAT_IDS", "").strip()
if not _allowed_raw:
    _allowed_raw = os.getenv("TG_SIGNAL_CHANNEL_ID", "").strip()

ALLOWED_CHAT_IDS = {
    x.strip()
    for x in _allowed_raw.split(",")
    if x.strip()
}

if not BOT_TOKEN:
    logger.warning("TG_BOT_TOKEN_SIGNAL is not set")


@dataclass
class CommandResult:
    text: str
    parse_mode: Optional[str] = None


def _read_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def _telegram_get_updates(offset: Optional[int] = None) -> list[dict]:
    if not BOT_TOKEN:
        raise RuntimeError("TG_BOT_TOKEN_SIGNAL is not set")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    payload: dict = {
        "limit": POLL_LIMIT,
        "timeout": POLL_TIMEOUT_SEC,
        "allowed_updates": [
            "message",
            "channel_post",
            "edited_message",
            "edited_channel_post",
        ],
    }
    if offset is not None:
        payload["offset"] = int(offset)

    resp = requests.get(url, params=payload, timeout=30)
    logger.info("Telegram getUpdates status=%s body=%s", resp.status_code, resp.text)
    resp.raise_for_status()

    data = resp.json()
    if not data.get("ok", False):
        raise RuntimeError(f"Telegram getUpdates failed: {data}")

    return data.get("result", [])


def _telegram_send_message(chat_id: str, text: str, parse_mode: Optional[str] = None) -> None:
    if not BOT_TOKEN:
        raise RuntimeError("TG_BOT_TOKEN_SIGNAL is not set")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    resp = requests.post(url, json=payload, timeout=20)
    logger.info("Telegram send status=%s body=%s", resp.status_code, resp.text)
    resp.raise_for_status()


def _extract_chat_and_text(update: dict) -> tuple[str, str]:
    for key in ("message", "edited_message", "channel_post", "edited_channel_post"):
        msg = update.get(key)
        if not isinstance(msg, dict):
            continue

        chat = msg.get("chat", {}) or {}
        chat_id = str(chat.get("id", "")).strip()

        text = msg.get("text")
        if text is None:
            continue

        text = str(text).strip()
        if chat_id and text:
            return chat_id, text

    return "", ""


def _is_allowed_chat(chat_id: str) -> bool:
    if not ALLOWED_CHAT_IDS:
        return True
    return chat_id in ALLOWED_CHAT_IDS


def _as_float(text: str) -> float:
    return float(str(text).strip())


def lotsize(balance: float, risk_decimal: float, entry: float, stop: float) -> float:
    diff = abs(entry - stop)
    if diff <= 0:
        raise ValueError("Entry price and stop loss price must be different")
    if balance <= 0:
        raise ValueError("Account balance must be greater than 0")
    if risk_decimal <= 0:
        raise ValueError("Risk decimal must be greater than 0")

    return (balance * risk_decimal) / diff


def _format_num(value: float) -> str:
    s = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _extract_command(text: str) -> tuple[str, list[str]]:
    raw = (text or "").strip()
    if not raw:
        return "", []

    parts = raw.split()
    if not parts:
        return "", []

    cmd = parts[0].split("@", 1)[0].strip().lower()
    args = parts[1:]
    return cmd, args


def handle_lotsize(args: list[str]) -> CommandResult:
    if len(args) != 4:
        return CommandResult(
            text=(
                "Usage: /lotsize <balance> <risk_decimal> <entry_price> <stop_loss_price>\n"
                "Example: /lotsize 5000 0.02 72000 71000"
            )
        )

    balance = _as_float(args[0])
    risk_percent = _as_float(args[1])
    risk_decimal = risk_percent / 100.0
    entry = _as_float(args[2])
    stop = _as_float(args[3])

    size = lotsize(balance, risk_decimal, entry, stop)
    risk_amount = balance * risk_decimal
    stop_distance = abs(entry - stop)

    return CommandResult(
        text=(
            f"Lotsize: {_format_num(size)}\n"
            f"Risk amount: {_format_num(risk_amount)}\n"
            f"Stop distance: {_format_num(stop_distance)}\n"
            f"Formula: (balance * risk_decimal) / abs(entry - stop)"
        )
    )


def handle_help(_: list[str]) -> CommandResult:
    return CommandResult(
        text=(
            "Available commands:\n"
            "/lotsize <balance> <risk_decimal> <entry_price> <stop_loss_price>\n"
            "Example: /lotsize 5000 0.02 72000 71000"
        )
    )


COMMANDS: dict[str, Callable[[list[str]], CommandResult]] = {
    "/lotsize": handle_lotsize,
    "/help": handle_help,
}


def dispatch(text: str) -> CommandResult:
    cmd, args = _extract_command(text)
    if not cmd:
        return CommandResult(text="Send /help for available commands.")

    handler = COMMANDS.get(cmd)
    if handler is None:
        return CommandResult(text="Unknown command. Send /help for available commands.")

    return handler(args)


@dag(
    dag_id="telegram_commands",
    schedule="*/1 * * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["telegram", "formula", "commands"],
)
def telegram_formula_commands():

    @task()
    def poll_and_reply() -> dict:
        state = _read_state()
        offset = state.get("last_update_id")
        updates = _telegram_get_updates(
            offset=offset + 1 if isinstance(offset, int) else None
        )

        if not updates:
            return {"processed": 0, "replied": 0}

        processed = 0
        replied = 0
        last_seen_update_id = int(offset) if isinstance(offset, int) else -1

        for update in updates:
            update_id = update.get("update_id")
            if not isinstance(update_id, int):
                continue
            if update_id <= last_seen_update_id:
                continue

            processed += 1
            last_seen_update_id = update_id

            chat_id, text = _extract_chat_and_text(update)
            if not chat_id or not text:
                continue

            if not _is_allowed_chat(chat_id):
                continue

            result = dispatch(text)
            _telegram_send_message(chat_id, result.text, result.parse_mode)
            replied += 1

        if last_seen_update_id >= 0:
            state["last_update_id"] = last_seen_update_id
            _write_state(state)

        return {"processed": processed, "replied": replied}

    poll_and_reply()


telegram_formula_commands_dag = telegram_formula_commands()