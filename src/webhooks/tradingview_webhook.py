from __future__ import annotations

# ============================================================
# 🌙 TradingView Webhook Receiver
#
# Supports two payload styles:
# 1) Legacy direct action payloads:
#    {
#      "symbol": "NQ1!",
#      "action": "BUY",
#      "contracts": 1,
#      "strategy": "ICT_FVG",
#      "timeframe": "1",
#      "price": 20345.25,
#      "secret": "..."
#    }
#
# 2) OHLCV bar payloads for Algotec ingestion:
#    {
#      "source": "tradingview",
#      "event": "bar_close",
#      "symbol": "CME_MINI:NQ1!",
#      "exchange": "CME_MINI",
#      "timeframe": "1",
#      "bar_time": "{{time}}",
#      "fire_time": "{{timenow}}",
#      "open": "{{open}}",
#      "high": "{{high}}",
#      "low": "{{low}}",
#      "close": "{{close}}",
#      "volume": "{{volume}}",
#      "secret": "..."
#    }
#
# Routes:
#   POST /webhook/tradingview
#   GET  /webhook/status
#   GET  /webhook/log
#   GET  /webhook/bars/<symbol>/<timeframe>
# ============================================================

import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from flask import Blueprint, jsonify, request

# Keep imports lazy where possible to avoid breaking the server if optional deps are missing.
from src.config import EXCHANGE

webhook_bp = Blueprint("webhook", __name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
BARS_DIR = DATA_DIR / "tradingview_bars"
BARS_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "algotec_secret_change_me")
WEBHOOK_LOG = DATA_DIR / "webhook_log.json"
DEDUP_FILE = DATA_DIR / "tradingview_dedupe.json"

MAX_LOG_ENTRIES = int(os.getenv("TRADINGVIEW_MAX_LOG_ENTRIES", "500"))
MAX_BARS_PER_STREAM = int(os.getenv("TRADINGVIEW_MAX_BARS_PER_STREAM", "2000"))
USE_PICKMYTRADE = os.getenv("USE_PICKMYTRADE", "0").strip().lower() in {"1", "true", "yes", "on"}
PICKMYTRADE_WEBHOOK_URL = os.getenv(
    "PICKMYTRADE_WEBHOOK_URL",
    "https://api.pickmytrade.trade/v2/add-trade-data-latest",
)
PICKMYTRADE_TOKEN = os.getenv("PICKMYTRADE_TOKEN", "")
PICKMYTRADE_ACCOUNT_ID = os.getenv("PICKMYTRADE_ACCOUNT_ID", "")
PICKMYTRADE_PYRAMID = os.getenv("PICKMYTRADE_PYRAMID", "true").strip().lower() in {"1", "true", "yes", "on"}
PICKMYTRADE_REVERSE_ORDER_CLOSE = os.getenv("PICKMYTRADE_REVERSE_ORDER_CLOSE", "false").strip().lower() in {"1", "true", "yes", "on"}
EXECUTE_ON_BAR_ALERT = os.getenv("TRADINGVIEW_EXECUTE_ON_BAR_ALERT", "0").strip().lower() in {"1", "true", "yes", "on"}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "null"):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value in (None, "", "null"):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _normalize_symbol(raw_symbol: str) -> str:
    symbol = str(raw_symbol or "").upper().strip()
    if ":" in symbol:
        symbol = symbol.split(":", 1)[1]
    symbol = symbol.replace("1!", "")
    return symbol


def _bar_stream_key(symbol: str, timeframe: str) -> str:
    tf = str(timeframe or "unknown").replace("/", "_")
    return f"{symbol}__{tf}"


def _bar_stream_path(symbol: str, timeframe: str) -> Path:
    return BARS_DIR / f"{_bar_stream_key(symbol, timeframe)}.json"


def _log_alert(alert: dict[str, Any], status: str, message: str, extra: dict[str, Any] | None = None) -> None:
    entry = {
        "timestamp": _now_iso(),
        "status": status,
        "message": message,
        "alert": alert,
    }
    if extra:
        entry["extra"] = extra

    logs = _read_json_file(WEBHOOK_LOG, [])
    logs.append(entry)
    logs = logs[-MAX_LOG_ENTRIES:]
    _write_json_file(WEBHOOK_LOG, logs)


def _dedupe_seen(key: str) -> bool:
    data = _read_json_file(DEDUP_FILE, {"keys": []})
    keys = deque(data.get("keys", []), maxlen=5000)
    if key in keys:
        return True
    keys.append(key)
    _write_json_file(DEDUP_FILE, {"keys": list(keys), "updated_at": _now_iso()})
    return False


def _extract_payload(request_json: Any) -> dict[str, Any]:
    if isinstance(request_json, dict):
        return request_json
    raise ValueError("Payload must be a JSON object")


def _payload_secret(alert: dict[str, Any]) -> str:
    return str(
        alert.get("secret")
        or request.headers.get("X-Webhook-Secret")
        or request.headers.get("X-Algotec-Token")
        or ""
    )


def _payload_type(alert: dict[str, Any]) -> str:
    has_ohlcv = all(k in alert for k in ("open", "high", "low", "close"))
    if has_ohlcv:
        return "bar"
    if str(alert.get("action", "")).upper() in {"BUY", "SELL", "CLOSE"}:
        return "action"
    return "unknown"


def _store_bar(alert: dict[str, Any]) -> dict[str, Any]:
    symbol = _normalize_symbol(alert.get("symbol", ""))
    timeframe = str(alert.get("timeframe") or alert.get("interval") or "unknown")
    exchange = str(alert.get("exchange") or "")
    bar_time = str(alert.get("bar_time") or alert.get("time") or alert.get("timestamp") or "")

    if not symbol:
        raise ValueError("Missing symbol in bar payload")
    if not bar_time:
        raise ValueError("Missing bar_time/time in bar payload")

    dedupe_key = f"{symbol}|{timeframe}|{bar_time}"
    duplicate = _dedupe_seen(dedupe_key)

    row = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "bar_time": bar_time,
        "fire_time": str(alert.get("fire_time") or ""),
        "open": _safe_float(alert.get("open")),
        "high": _safe_float(alert.get("high")),
        "low": _safe_float(alert.get("low")),
        "close": _safe_float(alert.get("close")),
        "volume": _safe_float(alert.get("volume")),
        "received_at": _now_iso(),
        "strategy": str(alert.get("strategy") or "TradingView"),
        "raw": alert,
    }

    stream_path = _bar_stream_path(symbol, timeframe)
    bars = _read_json_file(stream_path, [])

    if duplicate:
        return {
            "stored": False,
            "duplicate": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "bar_time": bar_time,
            "bars_buffered": len(bars),
            "stream_path": str(stream_path),
        }

    bars.append(row)
    bars = bars[-MAX_BARS_PER_STREAM:]
    _write_json_file(stream_path, bars)

    return {
        "stored": True,
        "duplicate": False,
        "symbol": symbol,
        "timeframe": timeframe,
        "bar_time": bar_time,
        "bars_buffered": len(bars),
        "stream_path": str(stream_path),
        "latest_close": row["close"],
    }


def _execute_tradovate(symbol: str, action: str, contracts: int, stop_ticks: int = 20, target_ticks: int = 40) -> dict[str, Any]:
    from src.exchanges.tradovate import place_bracket_order, is_market_open

    if not is_market_open():
        return {"success": False, "reason": "Market is closed"}

    result = place_bracket_order(
        symbol=symbol,
        action=action,
        contracts=contracts,
        stop_ticks=stop_ticks,
        target_ticks=target_ticks,
    )
    return {"success": True, "result": result}


def _execute_hyperliquid(symbol: str, action: str, usd_amount: float) -> dict[str, Any]:
    from src.exchanges.router import buy, close, sell

    action_upper = action.upper()
    if action_upper == "BUY":
        result = buy(symbol, usd_amount)
    elif action_upper == "SELL":
        result = sell(symbol, usd_amount)
    elif action_upper == "CLOSE":
        result = close(symbol)
    else:
        return {"success": False, "reason": f"Unsupported action: {action}"}
    return {"success": True, "result": str(result)}


def _execute_pickmytrade(symbol: str, action: str, contracts: int, price: float = 0.0, stop_ticks: int = 0, target_ticks: int = 0) -> dict[str, Any]:
    if not PICKMYTRADE_TOKEN:
        return {"success": False, "reason": "Missing PICKMYTRADE_TOKEN"}

    side_map = {"BUY": "buy", "SELL": "sell", "CLOSE": "close"}
    side = side_map.get(action.upper())
    if not side:
        return {"success": False, "reason": f"Unsupported action for PickMyTrade: {action}"}

    payload = {
        "symbol": symbol,
        "data": side,
        "quantity": contracts,
        "price": price,
        "tp": target_ticks,
        "sl": stop_ticks,
        "token": PICKMYTRADE_TOKEN,
        "reverse_order_close": PICKMYTRADE_REVERSE_ORDER_CLOSE,
        "pyramid": PICKMYTRADE_PYRAMID,
        "date": _now_iso(),
    }
    if PICKMYTRADE_ACCOUNT_ID:
        payload["account_id"] = PICKMYTRADE_ACCOUNT_ID

    resp = requests.post(PICKMYTRADE_WEBHOOK_URL, json=payload, timeout=5)
    ok = 200 <= resp.status_code < 300
    return {
        "success": ok,
        "status_code": resp.status_code,
        "response_text": resp.text[:1000],
        "request_payload": payload,
    }


def _risk_check(symbol: str, usd_amt: float, direction: str) -> tuple[bool, str]:
    try:
        from src.agents.risk_agent import risk

        return risk.check_trade(symbol, usd_amt, direction)
    except Exception as exc:
        return False, f"Risk agent unavailable: {exc}"


def _execute_action(alert: dict[str, Any], symbol: str, action: str) -> tuple[dict[str, Any], int]:
    contracts = _safe_int(alert.get("contracts", 1), 1)
    usd_amt = _safe_float(alert.get("usd_amount", 50), 50.0)
    price = _safe_float(alert.get("price", alert.get("close", 0)))
    stop_ticks = _safe_int(alert.get("stop_ticks", 20), 20)
    target_ticks = _safe_int(alert.get("target_ticks", 40), 40)

    if action == "CLOSE":
        if USE_PICKMYTRADE:
            result = _execute_pickmytrade(symbol, action, contracts, price=price)
        elif EXCHANGE == "tradovate":
            from src.exchanges.tradovate import close_position

            result = {"success": True, "result": close_position(symbol)}
        else:
            from src.exchanges.router import close

            result = {"success": True, "result": close(symbol)}

        if result.get("success"):
            _log_alert(alert, "EXECUTED", f"Closed {symbol}")
            return jsonify({"status": "closed", "symbol": symbol, "route": "pickmytrade" if USE_PICKMYTRADE else EXCHANGE}), 200

        reason = result.get("reason", result.get("response_text", "Unknown close error"))
        _log_alert(alert, "FAILED", reason)
        return jsonify({"status": "failed", "reason": reason}), 500

    direction = "buy" if action == "BUY" else "sell"
    allowed, reason = _risk_check(symbol, usd_amt, direction)
    if not allowed:
        _log_alert(alert, "BLOCKED", reason)
        return jsonify({"status": "blocked", "reason": reason}), 200

    if USE_PICKMYTRADE:
        result = _execute_pickmytrade(symbol, action, contracts, price=price, stop_ticks=stop_ticks, target_ticks=target_ticks)
        route = "pickmytrade"
    elif EXCHANGE == "tradovate":
        result = _execute_tradovate(symbol, action, contracts, stop_ticks, target_ticks)
        route = EXCHANGE
    else:
        result = _execute_hyperliquid(symbol, action, usd_amt)
        route = EXCHANGE

    if result.get("success"):
        _log_alert(alert, "EXECUTED", f"{action} {contracts} {symbol} via {route}", extra={"route": route})
        return jsonify({
            "status": "executed",
            "symbol": symbol,
            "action": action,
            "route": route,
            "details": result,
        }), 200

    reason = result.get("reason", result.get("response_text", "Unknown execution error"))
    _log_alert(alert, "FAILED", reason, extra={"route": route})
    return jsonify({"status": "failed", "reason": reason, "route": route}), 500


@webhook_bp.route("/webhook/tradingview", methods=["POST"])
def tradingview_webhook():
    try:
        payload = _extract_payload(request.get_json(force=True, silent=False))
    except Exception as exc:
        return jsonify({"error": f"Invalid JSON payload: {exc}"}), 400

    try:
        incoming_secret = _payload_secret(payload)
        if incoming_secret != WEBHOOK_SECRET:
            _log_alert(payload, "REJECTED", "Invalid secret")
            return jsonify({"error": "Unauthorized"}), 401

        payload_type = _payload_type(payload)
        symbol = _normalize_symbol(payload.get("symbol", ""))

        if payload_type == "bar":
            store_result = _store_bar(payload)
            action = str(payload.get("action", "")).upper()

            if action in {"BUY", "SELL", "CLOSE"} and EXECUTE_ON_BAR_ALERT:
                exec_response, status_code = _execute_action(payload, symbol, action)
                body = exec_response.get_json() or {}
                body["bar_store"] = store_result
                return jsonify(body), status_code

            _log_alert(payload, "RECEIVED", "Bar payload stored", extra=store_result)
            return jsonify({
                "status": "accepted",
                "payload_type": "bar",
                "execute_on_bar_alert": EXECUTE_ON_BAR_ALERT,
                "bar_store": store_result,
            }), 200

        if payload_type == "action":
            action = str(payload.get("action", "")).upper()
            return _execute_action(payload, symbol, action)

        _log_alert(payload, "REJECTED", "Unknown payload shape")
        return jsonify({
            "error": "Unknown payload shape. Send either action payloads or OHLCV bar payloads.",
        }), 400

    except Exception as exc:
        _log_alert(payload if isinstance(payload, dict) else {}, "ERROR", str(exc))
        return jsonify({"error": str(exc)}), 500


@webhook_bp.route("/webhook/status", methods=["GET"])
def webhook_status():
    log_entries = len(_read_json_file(WEBHOOK_LOG, []))
    dedupe_state = _read_json_file(DEDUP_FILE, {"keys": []})
    stream_count = len(list(BARS_DIR.glob("*.json")))
    return jsonify({
        "status": "online",
        "exchange": EXCHANGE,
        "use_pickmytrade": USE_PICKMYTRADE,
        "execute_on_bar_alert": EXECUTE_ON_BAR_ALERT,
        "time": _now_iso(),
        "message": "TradingView webhook receiver is active",
        "log_entries": log_entries,
        "dedupe_keys": len(dedupe_state.get("keys", [])),
        "bar_streams": stream_count,
    })


@webhook_bp.route("/webhook/log", methods=["GET"])
def webhook_log():
    logs = _read_json_file(WEBHOOK_LOG, [])
    return jsonify(logs[-50:])


@webhook_bp.route("/webhook/bars/<symbol>/<timeframe>", methods=["GET"])
def webhook_bars(symbol: str, timeframe: str):
    symbol_norm = _normalize_symbol(symbol)
    path = _bar_stream_path(symbol_norm, timeframe)
    bars = _read_json_file(path, [])
    return jsonify({
        "symbol": symbol_norm,
        "timeframe": timeframe,
        "count": len(bars),
        "rows": bars[-200:],
        "path": str(path),
    })
