# ============================================================
# 🌙 TradingView Webhook Receiver
#
# Receives webhook alerts from TradingView Pine Script
# strategies and routes them through Algotec's risk agent
# before executing on Tradovate.
#
# SETUP IN TRADINGVIEW:
#   1. Open your chart → Add alert
#   2. Condition: your strategy signal
#   3. Notifications → Webhook URL:
#      http://YOUR_IP:8080/webhook/tradingview
#   4. Message (JSON):
#      {
#        "symbol":    "{{ticker}}",
#        "action":    "BUY",
#        "contracts": 1,
#        "strategy":  "ICT_FVG",
#        "timeframe": "{{interval}}",
#        "price":     {{close}},
#        "secret":    "your_webhook_secret"
#      }
#
# NOTE: You need a public IP for TradingView to reach you.
# Use ngrok for easy setup: ngrok http 8080
# Then use the ngrok URL as your webhook URL in TradingView.
# ============================================================

import os
import json
import hashlib
from datetime import datetime
from flask import Blueprint, request, jsonify

# Import risk and execution components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.risk_agent  import risk
from src.config              import EXCHANGE

webhook_bp = Blueprint("webhook", __name__)

# ── Security ──────────────────────────────────────────────────
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "algotec_secret_change_me")

# ── Alert log ─────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent
WEBHOOK_LOG = REPO_ROOT / "src" / "data" / "webhook_log.json"
WEBHOOK_LOG.parent.mkdir(parents=True, exist_ok=True)


def _log_alert(alert: dict, status: str, message: str):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "alert":     alert,
        "status":    status,
        "message":   message,
    }
    logs = []
    if WEBHOOK_LOG.exists():
        try:
            logs = json.loads(WEBHOOK_LOG.read_text())
        except Exception:
            logs = []
    logs.append(entry)
    logs = logs[-500:]   # keep last 500
    WEBHOOK_LOG.write_text(json.dumps(logs, indent=2))


def _execute_tradovate(symbol: str, action: str, contracts: int,
                        stop_ticks: int = 20, target_ticks: int = 40) -> dict:
    """Execute trade on Tradovate with bracket order."""
    from src.exchanges.tradovate import (
        place_bracket_order, is_market_open, get_price
    )

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


def _execute_hyperliquid(symbol: str, action: str, usd_amount: float) -> dict:
    """Execute trade on Hyperliquid."""
    from src.exchanges.router import buy, sell
    if action.upper() == "BUY":
        result = buy(symbol, usd_amount)
    else:
        result = sell(symbol, usd_amount)
    return {"success": True, "result": str(result)}


@webhook_bp.route("/webhook/tradingview", methods=["POST"])
def tradingview_webhook():
    """
    Receive and process a TradingView webhook alert.
    """
    try:
        # Parse body
        if request.is_json:
            alert = request.get_json()
        else:
            try:
                alert = json.loads(request.data.decode("utf-8"))
            except Exception:
                return jsonify({"error": "Invalid JSON"}), 400

        print(f"\n📡 TradingView Alert received: {json.dumps(alert)}")

        # ── Security check ────────────────────────────────────
        incoming_secret = alert.get("secret", "")
        if incoming_secret != WEBHOOK_SECRET:
            print(f"  🚫 Invalid webhook secret")
            _log_alert(alert, "REJECTED", "Invalid secret")
            return jsonify({"error": "Unauthorized"}), 401

        # ── Parse alert ───────────────────────────────────────
        symbol    = str(alert.get("symbol", "")).upper().replace("1!", "")
        action    = str(alert.get("action", "")).upper()
        contracts = int(alert.get("contracts", 1))
        strategy  = str(alert.get("strategy", "TradingView"))
        price     = float(alert.get("price", 0))
        usd_amt   = float(alert.get("usd_amount", 50))
        stop_ticks   = int(alert.get("stop_ticks", 20))
        target_ticks = int(alert.get("target_ticks", 40))

        if action not in ("BUY", "SELL", "CLOSE"):
            _log_alert(alert, "REJECTED", f"Unknown action: {action}")
            return jsonify({"error": f"Unknown action: {action}"}), 400

        print(f"  📋 Signal: {strategy} | {symbol} {action} "
              f"| {contracts} contracts @ ${price:,.2f}")

        # ── CLOSE position ────────────────────────────────────
        if action == "CLOSE":
            if EXCHANGE == "tradovate":
                from src.exchanges.tradovate import close_position
                result = close_position(symbol)
            else:
                from src.exchanges.router import close
                result = close(symbol)
            _log_alert(alert, "EXECUTED", f"Closed {symbol}")
            return jsonify({"status": "closed", "symbol": symbol})

        # ── Risk check ────────────────────────────────────────
        direction = "buy" if action == "BUY" else "sell"
        allowed, reason = risk.check_trade(symbol, usd_amt, direction)

        if not allowed:
            print(f"  🚫 Risk blocked: {reason}")
            _log_alert(alert, "BLOCKED", reason)
            return jsonify({"status": "blocked", "reason": reason})

        # ── Execute ───────────────────────────────────────────
        if EXCHANGE == "tradovate":
            result = _execute_tradovate(
                symbol, action, contracts, stop_ticks, target_ticks
            )
        else:
            result = _execute_hyperliquid(symbol, action, usd_amt)

        if result.get("success"):
            print(f"  ✅ Trade executed via {EXCHANGE}")
            _log_alert(alert, "EXECUTED",
                       f"{action} {contracts} {symbol} via {EXCHANGE}")
            return jsonify({
                "status":   "executed",
                "symbol":   symbol,
                "action":   action,
                "exchange": EXCHANGE,
            })
        else:
            reason = result.get("reason", "Unknown error")
            _log_alert(alert, "FAILED", reason)
            return jsonify({"status": "failed", "reason": reason}), 500

    except Exception as e:
        print(f"  ❌ Webhook error: {e}")
        _log_alert({}, "ERROR", str(e))
        return jsonify({"error": str(e)}), 500


@webhook_bp.route("/webhook/status", methods=["GET"])
def webhook_status():
    """Check webhook receiver is online."""
    return jsonify({
        "status":   "online",
        "exchange": EXCHANGE,
        "time":     datetime.now().isoformat(),
        "message":  "TradingView webhook receiver is active",
    })


@webhook_bp.route("/webhook/log", methods=["GET"])
def webhook_log():
    """Return last 50 webhook events."""
    if not WEBHOOK_LOG.exists():
        return jsonify([])
    try:
        logs = json.loads(WEBHOOK_LOG.read_text())
        return jsonify(logs[-50:])
    except Exception:
        return jsonify([])
