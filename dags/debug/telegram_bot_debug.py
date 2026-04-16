from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

import requests

BOT_TOKEN = os.getenv("TG_BOT_TOKEN_SIGNAL", "").strip()
TEST_CHAT_ID = os.getenv("TG_TEST_CHAT_ID", "").strip()
EXPECTED_CHAT_ID = os.getenv("TG_EXPECTED_CHAT_ID", "").strip()
UPDATE_OFFSET = os.getenv("TG_UPDATE_OFFSET", "").strip()
MAX_UPDATES = int(os.getenv("TG_MAX_UPDATES", "50"))


def require_bot_token() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("TG_BOT_TOKEN_SIGNAL is NOT set")


def api_get(method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    require_bot_token()
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    resp = requests.get(url, params=params or {}, timeout=30)
    print(f"\nGET {method}")
    print("url:", resp.url)
    print("status:", resp.status_code)
    print("body:", resp.text)
    resp.raise_for_status()
    return resp.json()


def api_post(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    require_bot_token()
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    resp = requests.post(url, data=payload, timeout=30)
    print(f"\nPOST {method}")
    print("url:", resp.url)
    print("status:", resp.status_code)
    print("body:", resp.text)
    resp.raise_for_status()
    return resp.json()


def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)


def is_parquet_document(msg: dict[str, Any]) -> bool:
    doc = msg.get("document")
    if not isinstance(doc, dict):
        return False
    file_name = str(doc.get("file_name") or "").lower()
    mime_type = str(doc.get("mime_type") or "").lower()
    return file_name.endswith(".parquet") or "parquet" in mime_type or "parquet" in file_name


def show_update(update: dict[str, Any]) -> None:
    print("\n====================================")
    print("update_id:", update.get("update_id"))

    for key in ("message", "channel_post", "edited_message", "edited_channel_post"):
        msg = update.get(key)
        if not isinstance(msg, dict):
            continue

        chat = msg.get("chat", {}) or {}
        sender_chat = msg.get("sender_chat", {}) or {}
        doc = msg.get("document", {}) or {}

        print("type:", key)
        print("message_id:", msg.get("message_id"))
        print("date:", msg.get("date"))
        print("chat.id:", chat.get("id"))
        print("chat.type:", chat.get("type"))
        print("chat.title:", chat.get("title"))
        print("chat.username:", chat.get("username"))
        print("sender_chat.id:", sender_chat.get("id"))
        print("sender_chat.title:", sender_chat.get("title"))
        print("text:", msg.get("text"))
        print("caption:", msg.get("caption"))
        print("document.file_name:", doc.get("file_name"))
        print("document.mime_type:", doc.get("mime_type"))
        print("document.file_size:", doc.get("file_size"))
        print("document.file_id:", doc.get("file_id"))
        print("looks_like_parquet:", is_parquet_document(msg))

        if is_parquet_document(msg) and doc.get("file_id"):
            try:
                file_info = api_get("getFile", {"file_id": doc["file_id"]})
                print("getFile:", pretty(file_info))
            except Exception as e:
                print("getFile failed:", repr(e))


def main() -> None:
    print("=== TELEGRAM BOT DEBUG ===")

    if not BOT_TOKEN:
        print("ERROR: TG_BOT_TOKEN_SIGNAL is empty")
        sys.exit(1)

    print("\n1) getMe")
    me = api_get("getMe")
    print(pretty(me))

    print("\n2) getWebhookInfo")
    wh = api_get("getWebhookInfo")
    print(pretty(wh))

    if EXPECTED_CHAT_ID:
        print("\n3) getChat")
        try:
            chat_info = api_get("getChat", {"chat_id": EXPECTED_CHAT_ID})
            print(pretty(chat_info))
        except Exception as e:
            print("getChat failed:", repr(e))

        print("\n4) getChatMember (bot membership in that chat/channel)")
        try:
            member = api_get(
                "getChatMember",
                {
                    "chat_id": EXPECTED_CHAT_ID,
                    "user_id": me["result"]["id"],
                },
            )
            print(pretty(member))
        except Exception as e:
            print("getChatMember failed:", repr(e))

    print("\n5) getUpdates")
    params: dict[str, Any] = {
        "limit": min(MAX_UPDATES, 100),
        "timeout": 0,
        "allowed_updates": json.dumps(
            [
                "message",
                "channel_post",
                "edited_message",
                "edited_channel_post",
            ]
        ),
    }

    if UPDATE_OFFSET:
        try:
            params["offset"] = int(UPDATE_OFFSET)
        except ValueError:
            print("WARNING: TG_UPDATE_OFFSET is not a valid int, ignoring it")

    updates = api_get("getUpdates", params)
    results = updates.get("result", []) if isinstance(updates, dict) else []

    print("\nUpdates count:", len(results))
    parquet_hits = 0

    for upd in results:
        if not isinstance(upd, dict):
            continue
        show_update(upd)
        for key in ("message", "channel_post", "edited_message", "edited_channel_post"):
            msg = upd.get(key)
            if isinstance(msg, dict) and is_parquet_document(msg):
                parquet_hits += 1

    print("\nParquet updates found:", parquet_hits)

    if TEST_CHAT_ID:
        print("\n6) sendMessage test")
        api_post(
            "sendMessage",
            {
                "chat_id": TEST_CHAT_ID,
                "text": "Telegram debug OK. Sending works.",
            },
        )
    else:
        print("\nSkipping sendMessage test (TG_TEST_CHAT_ID not set)")


if __name__ == "__main__":
    main()