from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

# Keep imports lightweight at module import time so the webhook can start
# even if optional execution/risk modules are unavailable.
from src.config import EXCHANGE

webhook_bp = Blueprint("webhook", __name__)

# ── Security ──────────────────────────────────────────────────
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "algotec_secret_change_me")
EXECUTE_ON_BAR_ALERT = os.getenv("TRADINGVIEW_EXECUTE_ON_BAR_ALERT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# ── Paths ─────────────────────────────────────────────────────
# tradingview_webhook.py lives in src/webhooks/, so parents[2] is repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "src" / "data"
BARS_DIR = DATA_DIR / "tradingview_bars"
WEBHOOK_LOG = DATA_DIR / "webhook_log.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
BARS_DIR.mkdir(parents=True, exist_ok=True)
WEBHOOK_LOG.parent.mkdir(parents=True, exist_ok=True)

MAX_LOG_ENTRIES = 500
MAX_BARS_PER_STREAM = 5000

# In-memory dedupe and bar cache per process.
_RECENT_ALERT_IDS: deque[str] = deque(maxlen=2000)


# ── Helpers ───────────────────────────────────────────────────
def _as_json_payload() -> dict[str, Any] | None:
    if request.is_json:
        return request.get_json(silent=True)
    try:
        raw = request.data.decode("utf-8") if request.data else ""
        return json.loads(raw) if raw else None
    except Exception:
        return None



def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "null"):
            return default
        return float(value)
    except Exception:
        return default



def _parse_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, "", "null"):
            return default
        return int(float(value))
    except Exception:
        return default



def _normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    # Keep TradingView root/futures notation intact for bar storage,
    # but legacy execution path historically strips the continuous suffix.
    return s



def _execution_symbol(symbol: str) -> str:
    return _normalize_symbol(symbol).replace("1!", "")



def _stream_key(symbol: str, timeframe: str) -> str:
    safe_symbol = _normalize_symbol(symbol).replace("/", "_")
    safe_tf = str(timeframe or "unknown").replace("/", "_")
    return f"{safe_symbol}__{safe_tf}"



def _stream_path(symbol: str, timeframe: str) -> Path:
    return BARS_DIR / f"{_stream_key(symbol, timeframe)}.json"



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



def _log_alert(alert: dict[str, Any], status: str, message: str, extra: dict[str, Any] | None = None) -> None:
    entry: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "message": message,
        "alert": alert,
    }
    if extra:
        entry["extra"] = extra
    logs = _read_json_file(WEBHOOK_LOG, default=[])
    if not isinstance(logs, list):
        logs = []
    logs.append(entry)
    logs = logs[-MAX_LOG_ENTRIES:]
    _write_json_file(WEBHOOK_LOG, logs)



def _extract_secret(payload: dict[str, Any]) -> str:
    return str(
        payload.get("token")
        or payload.get("secret")
        or request.headers.get("X-Webhook-Secret", "")
        or request.headers.get("X-Algotec-Token", "")
        or ""
    )



def _is_bar_payload(payload: dict[str, Any]) -> bool:
    keys = set(payload.keys())
    needed = {"open", "high", "low", "close", "volume"}
    return needed.issubset(keys) and any(k in keys for k in ("bar_time", "candle_time", "time"))



def _bar_time(payload: dict[str, Any]) -> str:
    return str(payload.get("bar_time") or payload.get("candle_time") or payload.get("time") or "")



def _bar_record(payload: dict[str, Any]) -> dict[str, Any]:
    symbol = _normalize_symbol(payload.get("symbol") or payload.get("ticker") or "")
    timeframe = str(payload.get("timeframe") or payload.get("interval") or "")
    return {
        "symbol": symbol,
        "exchange": str(payload.get("exchange") or ""),
        "timeframe": timeframe,
        "bar_time": _bar_time(payload),
        "fire_time": str(payload.get("fire_time") or payload.get("timenow") or ""),
        "open": _parse_float(payload.get("open")),
        "high": _parse_float(payload.get("high")),
        "low": _parse_float(payload.get("low")),
        "close": _parse_float(payload.get("close")),
        "volume": _parse_float(payload.get("volume")),
        "event": str(payload.get("event") or "candle_close"),
        "source": str(payload.get("source") or "tradingview"),
        "received_at": datetime.now().isoformat(),
    }



def _bar_alert_id(bar: dict[str, Any]) -> str:
    return f"{bar['symbol']}|{bar['timeframe']}|{bar['bar_time']}"



def _store_bar(bar: dict[str, Any]) -> tuple[bool, int, Path]:
    path = _stream_path(bar["symbol"], bar["timeframe"])
    rows = _read_json_file(path, default=[])
    if not isinstance(rows, list):
        rows = []

    alert_id = _bar_alert_id(bar)
    if alert_id in _RECENT_ALERT_IDS:
        return False, len(rows), path

    if rows and rows[-1].get("bar_time") == bar["bar_time"]:
        # Replace the last row if the same bar arrives again.
        rows[-1] = bar
    else:
        rows.append(bar)
        rows = rows[-MAX_BARS_PER_STREAM:]

    _RECENT_ALERT_IDS.append(alert_id)
    _write_json_file(path, rows)
    return True, len(rows), path



def _execute_tradovate(symbol: str, action: str, contracts: int, stop_ticks: int = 20, target_ticks: int = 40) -> dict[str, Any]:
    from src.exchanges.tradovate import is_market_open, place_bracket_order

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
    from src.exchanges.router import buy, sell

    if action.upper() == "BUY":
        result = buy(symbol, usd_amount)
    else:
        result = sell(symbol, usd_amount)
    return {"success": True, "result": str(result)}



def _run_risk_check(symbol: str, usd_amt: float, action: str) -> tuple[bool, str]:
    """
    Risk is optional here. If the risk agent cannot import because of config drift,
    return a clear reason instead of crashing the webhook service.
    """
    try:
        from src.agents.risk_agent import risk
    except Exception as exc:
        return False, f"Risk agent unavailable: {exc}"

    direction = "buy" if action.upper() == "BUY" else "sell"
    try:
        allowed, reason = risk.check_trade(symbol, usd_amt, direction)
        return bool(allowed), str(reason)
    except Exception as exc:
        return False, f"Risk check failed: {exc}"



def _handle_legacy_signal(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    symbol = _execution_symbol(payload.get("symbol", ""))
    action = str(payload.get("action", "")).upper()
    contracts = _parse_int(payload.get("contracts", 1), default=1)
    strategy = str(payload.get("strategy", "TradingView"))
    price = _parse_float(payload.get("price", 0.0), default=0.0)
    usd_amt = _parse_float(payload.get("usd_amount", 50.0), default=50.0)
    stop_ticks = _parse_int(payload.get("stop_ticks", 20), default=20)
    target_ticks = _parse_int(payload.get("target_ticks", 40), default=40)

    if action not in {"BUY", "SELL", "CLOSE"}:
        _log_alert(payload, "REJECTED", f"Unknown action: {action}")
        return {"error": f"Unknown action: {action}"}, 400

    if not symbol:
        _log_alert(payload, "REJECTED", "Missing symbol")
        return {"error": "Missing symbol"}, 400

    print(f"\n📡 TradingView Signal received: {json.dumps(payload)}")
    print(f"  📋 Signal: {strategy} | {symbol} {action} | {contracts} contracts @ ${price:,.2f}")

    if action == "CLOSE":
        try:
            if EXCHANGE == "tradovate":
                from src.exchanges.tradovate import close_position
                close_position(symbol)
            else:
                from src.exchanges.router import close
                close(symbol)
            _log_alert(payload, "EXECUTED", f"Closed {symbol}")
            return {"status": "closed", "symbol": symbol}, 200
        except Exception as exc:
            _log_alert(payload, "FAILED", f"Close failed: {exc}")
            return {"status": "failed", "reason": str(exc)}, 500

    allowed, reason = _run_risk_check(symbol, usd_amt, action)
    if not allowed:
        _log_alert(payload, "BLOCKED", reason)
        return {"status": "blocked", "reason": reason}, 200

    try:
        if EXCHANGE == "tradovate":
            result = _execute_tradovate(symbol, action, contracts, stop_ticks, target_ticks)
        else:
            result = _execute_hyperliquid(symbol, action, usd_amt)
    except Exception as exc:
        _log_alert(payload, "FAILED", f"Execution failed: {exc}")
        return {"status": "failed", "reason": str(exc)}, 500

    if result.get("success"):
        _log_alert(payload, "EXECUTED", f"{action} {contracts} {symbol} via {EXCHANGE}")
        return {
            "status": "executed",
            "symbol": symbol,
            "action": action,
            "exchange": EXCHANGE,
        }, 200

    reason = str(result.get("reason", "Unknown error"))
    _log_alert(payload, "FAILED", reason)
    return {"status": "failed", "reason": reason}, 500


@webhook_bp.route("/webhook/tradingview", methods=["POST"])
def tradingview_webhook():
    payload = _as_json_payload()
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON"}), 400

    incoming_secret = _extract_secret(payload)
    if incoming_secret != WEBHOOK_SECRET:
        _log_alert(payload, "REJECTED", "Invalid secret/token")
        return jsonify({"error": "Unauthorized"}), 401

    # New candle/bar payload path.
    if _is_bar_payload(payload):
        bar = _bar_record(payload)
        if not bar["symbol"] or not bar["timeframe"] or not bar["bar_time"]:
            _log_alert(payload, "REJECTED", "Missing candle fields")
            return jsonify({"error": "Missing candle fields"}), 400

        inserted, buffered_count, path = _store_bar(bar)
        result: dict[str, Any] = {
            "status": "accepted",
            "type": "candle",
            "inserted": inserted,
            "symbol": bar["symbol"],
            "timeframe": bar["timeframe"],
            "bar_time": bar["bar_time"],
            "bars_buffered": buffered_count,
            "storage": str(path),
            "execution_enabled": EXECUTE_ON_BAR_ALERT,
        }

        if not EXECUTE_ON_BAR_ALERT:
            _log_alert(payload, "ACCEPTED", "Candle stored; execution disabled", extra=result)
            return jsonify(result), 200

        # Execution on bar alerts is intentionally left off by default.
        _log_alert(payload, "ACCEPTED", "Candle stored; execution-on-bar not implemented", extra=result)
        result["trade_sent"] = False
        result["message"] = "Candle stored; execution-on-bar not implemented in this file yet."
        return jsonify(result), 200

    # Legacy direct BUY/SELL/CLOSE signal path.
    response, status_code = _handle_legacy_signal(payload)
    return jsonify(response), status_code


@webhook_bp.route("/webhook/status", methods=["GET"])
def webhook_status():
    return jsonify(
        {
            "status": "online",
            "exchange": EXCHANGE,
            "time": datetime.now().isoformat(),
            "message": "TradingView webhook receiver is active",
            "execution_on_bar_alert": EXECUTE_ON_BAR_ALERT,
            "bars_dir": str(BARS_DIR),
        }
    )


@webhook_bp.route("/webhook/log", methods=["GET"])
def webhook_log():
    logs = _read_json_file(WEBHOOK_LOG, default=[])
    if not isinstance(logs, list):
        logs = []
    return jsonify(logs[-50:])


@webhook_bp.route("/webhook/bars/<symbol>/<timeframe>", methods=["GET"])
def webhook_bars(symbol: str, timeframe: str):
    path = _stream_path(symbol, timeframe)
    rows = _read_json_file(path, default=[])
    if not isinstance(rows, list):
        rows = []
    return jsonify(
        {
            "symbol": _normalize_symbol(symbol),
            "timeframe": timeframe,
            "count": len(rows),
            "rows": rows[-200:],
            "storage": str(path),
        }
    )
