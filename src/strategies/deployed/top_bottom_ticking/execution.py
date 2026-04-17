from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import requests
from zoneinfo import ZoneInfo

from config import (
    EXECUTION_MODE,
    LOG_DIR,
    MODEL_NAME,
    PICKMYTRADE_ACCOUNT_ID,
    PICKMYTRADE_BASE_URL,
    PICKMYTRADE_FORCE_FLAT_URL,
    PICKMYTRADE_STRATEGY_ID,
    PICKMYTRADE_TOKEN,
    PICKMYTRADE_WEBHOOK_URL,
)

ET = ZoneInfo("America/New_York")
EXEC_LOG = LOG_DIR / "execution_log.jsonl"

DEFAULT_TRADOVATE_WEBHOOK_URL = "https://api.pickmytrade.trade/v2/add-trade-data"


class ExecutionError(Exception):
    pass


def now_et_iso() -> str:
    return datetime.now(timezone.utc).astimezone(ET).isoformat()


def log_event(payload: dict[str, Any]) -> None:
    with EXEC_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _auth_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if PICKMYTRADE_TOKEN:
        headers["Authorization"] = f"Bearer {PICKMYTRADE_TOKEN}"
    return headers


def _resolve_order_url() -> str:
    # Highest priority: exact copied webhook URL from PickMyTrade
    if PICKMYTRADE_WEBHOOK_URL:
        return PICKMYTRADE_WEBHOOK_URL

    base = (PICKMYTRADE_BASE_URL or "").strip()
    if base.lower().startswith("http://") or base.lower().startswith("https://"):
        if "add-trade-data" in base.lower():
            return base
    return DEFAULT_TRADOVATE_WEBHOOK_URL


def _normalize_action(side: str) -> str:
    s = side.strip().upper()
    if s in {"BUY", "LONG"}:
        return "buy"
    if s in {"SELL", "SHORT"}:
        return "sell"
    if s in {"CLOSE"}:
        return "close"
    raise ExecutionError(f"Unsupported side/action: {side}")


def execute_signal(signal: dict[str, Any]) -> dict[str, Any]:
    if EXECUTION_MODE == "paper":
        result = {
            "ok": True,
            "mode": "paper",
            "signal_id": signal["signal_id"],
            "symbol": signal["symbol"],
            "side": signal["side"],
            "qty": signal["qty"],
            "logged_at_et": now_et_iso(),
        }
        log_event({"event": "paper_order", "model_name": MODEL_NAME, **result, "signal": signal})
        return result
    return send_to_pickmytrade(signal)


def send_to_pickmytrade(signal: dict[str, Any]) -> dict[str, Any]:
    if not PICKMYTRADE_ACCOUNT_ID:
        raise ExecutionError("Missing PICKMYTRADE_ACCOUNT_ID in environment")
    if not PICKMYTRADE_TOKEN:
        raise ExecutionError("Missing PICKMYTRADE_TOKEN in environment")

    url = _resolve_order_url()
    action = _normalize_action(signal["side"])

    # Align with PickMyTrade's documented webhook JSON shape
    payload = {
        "symbol": signal["symbol"],
        "date": signal.get("timestamp_et", now_et_iso()),
        "data": action,
        "quantity": int(signal["qty"]),
        "risk_percentage": 0,
        "price": signal.get("entry", 0) or 0,
        "tp": 0,
        "percentage_tp": 0,
        "dollar_tp": 0,
        "sl": 0,
        "dollar_sl": 0,
        "percentage_sl": 0,
        "trail": 0,
        "trail_stop": 0,
        "trail_trigger": 0,
        "trail_freq": 0,
        "update_tp": False,
        "update_sl": False,
        "breakeven": 0,
        "token": PICKMYTRADE_TOKEN,
        "pyramid": False,
        "reverse_order_close": False,
        "account_id": PICKMYTRADE_ACCOUNT_ID,
        "comment": signal["signal_id"],
    }

    print("PICKMYTRADE ORDER URL =", url)
    print("PICKMYTRADE ACCOUNT ID =", PICKMYTRADE_ACCOUNT_ID)
    print("PICKMYTRADE STRATEGY ID =", PICKMYTRADE_STRATEGY_ID)
    print("PICKMYTRADE ACTION =", action)
    print("PICKMYTRADE QTY =", signal["qty"])
    print("PICKMYTRADE PAYLOAD =", payload)

    response = requests.post(url, headers=_auth_headers(), json=payload, timeout=30)
    text = response.text[:4000]

    if response.status_code >= 300:
        log_event({
            "event": "live_order_error",
            "model_name": MODEL_NAME,
            "logged_at_et": now_et_iso(),
            "status_code": response.status_code,
            "response_text": text,
            "request_url": url,
            "request_payload": payload,
            "signal": signal,
        })
        raise ExecutionError(f"PickMyTrade error {response.status_code}: {text}")

    try:
        data = response.json()
    except Exception:
        data = {"raw_response": text}

    result = {
        "ok": True,
        "mode": EXECUTION_MODE,
        "signal_id": signal["signal_id"],
        "symbol": signal["symbol"],
        "side": action,
        "qty": signal["qty"],
        "logged_at_et": now_et_iso(),
        "request_url": url,
        "response": data,
    }
    log_event({"event": "live_order_sent", "model_name": MODEL_NAME, **result, "signal": signal})
    return result


def force_flat_all_positions(*, reason: str, session_key: str) -> dict[str, Any]:
    base_result = {
        "mode": EXECUTION_MODE,
        "reason": reason,
        "session_key": session_key,
        "logged_at_et": now_et_iso(),
    }

    if EXECUTION_MODE == "paper":
        result = {
            "ok": True,
            **base_result,
            "action": "paper_force_flat_logged",
        }
        log_event({"event": "paper_force_flat", "model_name": MODEL_NAME, **result})
        return result

    if not PICKMYTRADE_ACCOUNT_ID:
        raise ExecutionError("Missing PICKMYTRADE_ACCOUNT_ID in environment")

    if not PICKMYTRADE_FORCE_FLAT_URL:
        result = {
            "ok": False,
            **base_result,
            "action": "force_flat_endpoint_not_configured",
        }
        log_event({"event": "force_flat_skipped", "model_name": MODEL_NAME, **result})
        return result

    url = PICKMYTRADE_FORCE_FLAT_URL
    if not url.lower().startswith("http://") and not url.lower().startswith("https://"):
        base = (PICKMYTRADE_BASE_URL or "").strip()
        if base.lower().startswith("http://") or base.lower().startswith("https://"):
            url = f"{base.rstrip('/')}/{url.lstrip('/')}"

    payload = {
        "token": PICKMYTRADE_TOKEN,
        "account_id": PICKMYTRADE_ACCOUNT_ID,
        "reason": reason,
        "comment": f"force_flat:{session_key}",
    }

    response = requests.post(url, headers=_auth_headers(), json=payload, timeout=30)
    text = response.text[:4000]
    if response.status_code >= 300:
        log_event({
            "event": "force_flat_error",
            "model_name": MODEL_NAME,
            **base_result,
            "status_code": response.status_code,
            "response_text": text,
            "url": url,
        })
        raise ExecutionError(f"PickMyTrade force-flat error {response.status_code}: {text}")

    try:
        data = response.json()
    except Exception:
        data = {"raw_response": text}

    result = {
        "ok": True,
        **base_result,
        "action": "force_flat_sent",
        "response": data,
        "url": url,
    }
    log_event({"event": "force_flat_sent", "model_name": MODEL_NAME, **result})
    return result