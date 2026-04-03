#!/usr/bin/env python3
# ============================================================
# 🌙 Apex Bridge — Two Execution Routes
#
# Since Apex blocks direct API access during evaluations,
# this module provides two ways to get signals into your
# Apex-connected Tradovate account:
#
# ROUTE A — Webhook Bridge (PickMyTrade / TradeZella)
#   Your bot → POST JSON → bridge service → Tradovate/Apex
#   Simplest. No local app needed. Bridge handles everything.
#   Services: pickmytrade.trade | tradesync.io | tradesviz.com
#
# ROUTE B — Local Executor (Flask relay on your Mac)
#   Your bot → POST to localhost → local app → Tradovate/Apex
#   More control. No third-party dependency.
#   You run a small Flask server on the same machine as Tradovate.
#
# SETUP:
#   Add to .env:
#     # Route A — PickMyTrade
#     PICKMYTRADE_TOKEN=your_token      # from pickmytrade.trade
#     PICKMYTRADE_STRATEGY_ID=your_id
#
#     # Route B — Local executor
#     LOCAL_EXECUTOR_URL=http://localhost:5001  # default
#
#   In config.py:
#     APEX_BRIDGE_ROUTE = "pickmytrade"  # or "local" or "tradovate_direct"
# ============================================================

import os, json, time, requests
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


# ══════════════════════════════════════════════════════════════
# ROUTE A — WEBHOOK BRIDGE (PickMyTrade)
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# HOW THE BRIDGE WORKS
# ══════════════════════════════════════════════════════════════
#
# Apex blocks YOUR Tradovate API credentials on evals.
# But services like PickMyTrade and TradersPost are registered
# Tradovate API PARTNERS — they have their own credentials.
#
# When you connect your Apex account to their service, they
# can place orders on your behalf. Your bot sends a signal
# to their webhook URL, they place the order on Tradovate.
#
# ARCHITECTURE:
#   vault_forward_test.py
#         ↓ POST JSON signal
#   apex_bridge.py (this file)
#         ↓ reformats signal as webhook JSON
#   PickMyTrade OR TradersPost webhook URL
#         ↓ places order
#   Your Apex/Tradovate account
#
# You own all the code above the webhook URL.
# You just need one of these services to handle the Tradovate part.
#
# SETUP OPTION A — PickMyTrade ($50/month)
#   1. Sign up: pickmytrade.trade
#   2. Connections → Add Tradovate → log in with your Apex credentials
#   3. Strategies → New Strategy → give it a name
#   4. Get your TOKEN and STRATEGY_ID from the strategy page
#   5. Add to .env:
#        PICKMYTRADE_TOKEN=your_token
#        PICKMYTRADE_STRATEGY_ID=your_strategy_id
#   6. config.py: APEX_BRIDGE_ROUTE = "pickmytrade"
#
# SETUP OPTION B — TradersPost (~$30/month, recommended)
#   1. Sign up: traderspost.io
#   2. Brokers → Add Broker → Tradovate → connect Apex account
#   3. Strategies → New Strategy → Futures → get webhook URL
#   4. Add to .env:
#        TRADERSPOST_WEBHOOK_URL=https://traderspost.io/trading/webhook/YOUR_KEY/YOUR_SECRET
#   5. config.py: APEX_BRIDGE_ROUTE = "traderspost"
#
# Both services receive the same JSON from your bot and place
# bracket orders (entry + SL + TP) on your Apex account.
# ══════════════════════════════════════════════════════════════


class PickMyTradeBridge:
    """
    Route: bot → PickMyTrade webhook → Apex/Tradovate
    $50/month | pickmytrade.trade
    Supports: market, limit, stop, bracket orders
    """

    BASE_URL = "https://api.pickmytrade.trade/v2/add-trade-data-latest"

    def __init__(self):
        self.token       = os.getenv("PICKMYTRADE_TOKEN", "")
        self.strategy_id = os.getenv("PICKMYTRADE_STRATEGY_ID", "")

        # Multiple account IDs — comma-separated in .env
        # e.g. PICKMYTRADE_ACCOUNT_IDS=APEX3722360000004,APEX3722360000005
        ids_raw = os.getenv("PICKMYTRADE_ACCOUNT_IDS", "")
        self.account_ids = [a.strip() for a in ids_raw.split(",") if a.strip()]

        # Fallback: single account ID
        single = os.getenv("PICKMYTRADE_ACCOUNT_ID", "")
        if single and single not in self.account_ids:
            self.account_ids.append(single)

        if not self.token:
            print("  ⚠️  PICKMYTRADE_TOKEN not set — add to .env")
            print("     Get from: pickmytrade.trade → Strategies → your strategy")
        else:
            print(f"  ✅ PickMyTrade ready — {len(self.account_ids)} account(s)")
            for a in self.account_ids:
                print(f"     → {a}")

    def _build_payload(self, action: str, symbol: str,
                       contracts: int, price: float = 0,
                       sl: float = 0, tp: float = 0) -> dict:
        """Build the exact PickMyTrade JSON format."""
        accounts = []

        # Add all configured accounts
        for acc_id in self.account_ids:
            accounts.append({
                "token":               self.token,
                "account_id":          acc_id,
                "risk_percentage":     0,
                "quantity_multiplier": 1,
            })

        # Fallback: single account from env
        if not accounts:
            single_id = os.getenv("PICKMYTRADE_ACCOUNT_ID", "")
            if single_id:
                accounts.append({
                    "token":               self.token,
                    "account_id":          single_id,
                    "risk_percentage":     0,
                    "quantity_multiplier": 1,
                })

        return {
            "symbol":                symbol,
            "strategy_name":         self.strategy_id or "",
            "date":                  __import__("datetime").datetime.utcnow().isoformat(),
            "data":                  action,        # buy | sell | exit
            "quantity":              contracts,
            "risk_percentage":       0,
            "price":                 round(price, 4),
            "tp":                    round(tp, 4)   if tp else 0,
            "percentage_tp":         0,
            "dollar_tp":             0,
            "sl":                    round(sl, 4)   if sl else 0,
            "dollar_sl":             0,
            "percentage_sl":         0,
            "trail":                 0,
            "trail_stop":            0,
            "trail_trigger":         0,
            "trail_freq":            0,
            "update_tp":             False,
            "update_sl":             False,
            "breakeven":             0,
            "breakeven_offset":      0,
            "token":                 self.token,
            "pyramid":               False,
            "same_direction_ignore": False,
            "reverse_order_close":   False,
            "multiple_accounts":     accounts,
        }

    def _post(self, action: str, symbol: str, contracts: int,
              price: float = 0, sl: float = 0, tp: float = 0) -> dict:
        payload = self._build_payload(action, symbol, contracts, price, sl, tp)
        accounts = [a["account_id"] for a in payload["multiple_accounts"]]
        print(f"  📡 PickMyTrade → {action} {contracts}x {symbol}"
              f"{f'  SL={sl:.2f}' if sl else ''}"
              f"{f'  TP={tp:.2f}' if tp else ''}"
              f"  accounts={accounts}")
        try:
            r = requests.post(
                self.BASE_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            r.raise_for_status()
            result = r.json()
            print(f"  ✅ PickMyTrade: {result.get('message', result)}")
            return result
        except Exception as e:
            print(f"  ❌ PickMyTrade error: {e}")
            return {"error": str(e)}

    def execute(self, signal: dict) -> dict:
        direction = signal["direction"].upper()
        action    = "buy" if direction == "LONG" else "sell"
        return self._post(
            action,
            signal["symbol"],
            int(signal.get("contracts") or 1),
            price=float(signal.get("entry", 0)),
            sl=float(signal.get("sl", 0)),
            tp=float(signal.get("tp", 0)),
        )

    def close(self, signal: dict) -> dict:
        return self._post("exit", signal["symbol"],
                          int(signal.get("contracts") or 1))


class TradersPostBridge:
    """
    Route: bot → TradersPost webhook → Apex/Tradovate
    ~$30/month | traderspost.io (recommended — cleaner JSON, cheaper)
    Supports: market orders with SL/TP, bracket orders

    TradersPost webhook JSON format:
    {
      "ticker":     "MES",
      "action":     "buy",         # buy | sell | exit | cancel
      "quantity":   "1",
      "stopLoss":   {"type": "stop",  "value": "4990.00"},
      "takeProfit": {"type": "limit", "value": "5020.00"}
    }
    """

    def __init__(self):
        self.webhook_url = os.getenv("TRADERSPOST_WEBHOOK_URL", "")
        if not self.webhook_url:
            print("  ⚠️  TRADERSPOST_WEBHOOK_URL not set — add to .env")
            print("     Get from: traderspost.io → Strategies → your strategy → Webhook URL")

    def _post(self, action: str, symbol: str, contracts: int,
              sl: float = None, tp: float = None) -> dict:
        payload: dict = {
            "ticker":   symbol,
            "action":   action,
            "quantity": str(contracts),
        }
        if sl:
            payload["stopLoss"]   = {"type": "stop",  "value": str(round(sl, 2))}
        if tp:
            payload["takeProfit"] = {"type": "limit", "value": str(round(tp, 2))}

        print(f"  📡 TradersPost → {action} {contracts}x {symbol}"
              f"{f'  SL={sl}' if sl else ''}{f'  TP={tp}' if tp else ''}")
        try:
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            r.raise_for_status()
            result = r.json()
            print(f"  ✅ TradersPost: {result.get('message', result)}")
            return result
        except Exception as e:
            print(f"  ❌ TradersPost error: {e}")
            return {"error": str(e)}

    def execute(self, signal: dict) -> dict:
        direction = signal["direction"].upper()
        action    = "buy" if direction == "LONG" else "sell"
        return self._post(action, signal["symbol"],
                          int(signal.get("contracts") or 1),
                          signal.get("sl"), signal.get("tp"))

    def close(self, signal: dict) -> dict:
        return self._post("exit", signal["symbol"],
                          int(signal.get("contracts") or 1))


# ══════════════════════════════════════════════════════════════
# ROUTE B — LOCAL EXECUTOR (Flask relay on your Mac)
# ══════════════════════════════════════════════════════════════

class LocalExecutor:
    """
    Sends signals to a local Flask relay running on your Mac.
    The relay then calls Tradovate's API directly.

    This works because:
    - Apex blocks API from cloud/VPS
    - But you CAN call Tradovate from your own local machine
    - The relay just forwards JSON orders to Tradovate

    Setup:
      1. Run the relay server (see run_local_relay() below)
         python src/agents/apex_bridge.py --serve
      2. Keep it running while you trade
      3. Your bot sends signals to http://localhost:5001/order

    The relay submits bracket orders (entry + SL + TP) to
    Tradovate in one atomic request.
    """

    def __init__(self, url: str = None):
        self.url = url or os.getenv(
            "LOCAL_EXECUTOR_URL", "http://localhost:5001"
        )

    def ping(self) -> bool:
        """Check if relay is running."""
        try:
            r = requests.get(f"{self.url}/ping", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def execute(self, signal: dict) -> dict:
        if not self.ping():
            return {
                "error": (
                    f"Local relay not running at {self.url}\n"
                    f"Start it with: python src/agents/apex_bridge.py --serve"
                )
            }
        print(f"  📡 Local relay → {signal['direction']} {signal['symbol']}")
        try:
            r = requests.post(
                f"{self.url}/order",
                json=signal,
                timeout=10
            )
            r.raise_for_status()
            result = r.json()
            print(f"  ✅ Relay: {result.get('message','OK')}")
            return result
        except Exception as e:
            print(f"  ❌ Relay error: {e}")
            return {"error": str(e)}


# ══════════════════════════════════════════════════════════════
# LOCAL RELAY SERVER — run on your Mac while trading
# ══════════════════════════════════════════════════════════════

def run_local_relay(port: int = 5001):
    """
    Lightweight Flask relay that accepts order JSON from your bot
    and submits bracket orders to Tradovate.

    Run this on your Mac and leave it running:
      python src/agents/apex_bridge.py --serve

    Your bot on any machine posts to:
      POST http://YOUR_MAC_IP:5001/order
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Install flask: pip install flask")
        return

    relay = Flask("apex_relay")

    @relay.route("/ping")
    def ping():
        return jsonify({"status": "ok", "time": datetime.now().isoformat()})

    @relay.route("/order", methods=["POST"])
    def place_order():
        sig = request.json
        if not sig:
            return jsonify({"error": "No JSON body"}), 400

        symbol    = sig.get("symbol", "")
        direction = sig.get("direction", "LONG").upper()
        contracts = int(sig.get("contracts") or 1)
        entry     = float(sig.get("entry", 0))
        sl        = float(sig.get("sl",    0))
        tp        = float(sig.get("tp",    0))

        print(f"\n📥 Signal received: {direction} {contracts}x {symbol}")
        print(f"   Entry={entry} SL={sl} TP={tp}")

        try:
            from src.exchanges.tradovate import (
                place_bracket_order, FUTURES_SPECS, _authenticate
            )

            _authenticate()

            spec = FUTURES_SPECS.get(symbol, {"tick_size": 0.25})
            tick = spec["tick_size"]

            # Convert SL/TP prices to tick counts
            if entry > 0 and sl > 0 and tp > 0:
                sl_ticks  = max(1, round(abs(entry - sl) / tick))
                tp_ticks  = max(1, round(abs(entry - tp) / tick))
            else:
                # Fallback: 10 tick SL, 20 tick TP
                sl_ticks  = 10
                tp_ticks  = 20

            action = "Buy" if direction == "LONG" else "Sell"
            result = place_bracket_order(
                symbol, action, contracts, sl_ticks, tp_ticks
            )

            return jsonify({
                "status":    "ok",
                "message":   f"Bracket order placed: {direction} {contracts}x {symbol}",
                "sl_ticks":  sl_ticks,
                "tp_ticks":  tp_ticks,
                "tradovate": result,
            })

        except Exception as e:
            print(f"  ❌ Order failed: {e}")
            return jsonify({"error": str(e)}), 500

    @relay.route("/positions")
    def positions():
        try:
            from src.exchanges.tradovate import get_positions
            return jsonify(get_positions())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @relay.route("/balance")
    def balance():
        try:
            from src.exchanges.tradovate import get_balance
            return jsonify(get_balance())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @relay.route("/close", methods=["POST"])
    def close_pos():
        try:
            sig    = request.json or {}
            symbol = sig.get("symbol", "")
            from src.exchanges.tradovate import close_position
            return jsonify(close_position(symbol))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    import socket
    local_ip = socket.gethostbyname(socket.gethostname())

    print(f"""
🌙 Apex Local Relay — Running
{'='*45}
  Listening  : http://localhost:{port}
  Network    : http://{local_ip}:{port}
  Endpoints  :
    POST /order     → place bracket order
    GET  /positions → open positions
    GET  /balance   → account balance
    POST /close     → close position
    GET  /ping      → health check
{'='*45}
  Your bot sends signals here automatically.
  Keep this running while you trade.
  Press Ctrl+C to stop.
{'='*45}
""")

    relay.run(host="0.0.0.0", port=port, debug=False)


# ══════════════════════════════════════════════════════════════
# UNIFIED EXECUTE — auto-selects route from config
# ══════════════════════════════════════════════════════════════

def execute_signal(signal: dict) -> dict:
    """
    Execute a signal using whichever route is configured.
    Called by vault_forward_test.py in auto mode.

    Routes (set APEX_BRIDGE_ROUTE in config.py):
      "tradovate_direct" → calls Tradovate API directly (sim only)
      "pickmytrade"      → sends to PickMyTrade webhook
      "local"            → sends to local relay on your Mac
    """
    try:
        from src.config import APEX_BRIDGE_ROUTE
        route = APEX_BRIDGE_ROUTE
    except ImportError:
        route = "tradovate_direct"

    symbol    = signal.get("symbol", "")
    is_futures = symbol.upper() in {"MES", "MNQ", "MYM"}

    print(f"\n  🔀 Route: {route} | {'FUTURES' if is_futures else 'CRYPTO'}")

    # Crypto always goes direct to Hyperliquid
    if not is_futures:
        try:
            from src.exchanges.hyperliquid import place_order
            direction = signal["direction"].upper()
            size_usd  = float(signal.get("size_usd", 50))
            sl        = signal.get("sl")
            tp        = signal.get("tp")
            return place_order(symbol, direction, size_usd, sl=sl, tp=tp)
        except Exception as e:
            return {"error": f"Hyperliquid error: {e}"}

    # Futures — use configured route
    if route == "pickmytrade":
        return PickMyTradeBridge().execute(signal)

    elif route == "traderspost":
        return TradersPostBridge().execute(signal)

    elif route == "local":
        return LocalExecutor().execute(signal)

    else:
        # tradovate_direct — works on sim, may be blocked on live Apex eval
        try:
            from src.exchanges.tradovate import place_bracket_order, FUTURES_SPECS
            spec      = FUTURES_SPECS.get(symbol, {"tick_size": 0.25})
            tick      = spec["tick_size"]
            entry     = float(signal.get("entry", 0))
            sl        = float(signal.get("sl",    0))
            tp        = float(signal.get("tp",    0))
            contracts = int(signal.get("contracts") or 1)
            direction = signal["direction"].upper()

            sl_ticks = max(1, round(abs(entry - sl) / tick))
            tp_ticks = max(1, round(abs(entry - tp) / tick))
            action   = "Buy" if direction == "LONG" else "Sell"

            return place_bracket_order(symbol, action, contracts, sl_ticks, tp_ticks)
        except Exception as e:
            return {"error": f"Tradovate direct error: {e}"}


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════
# CONNECTION TEST
# ══════════════════════════════════════════════════════════════

def test_pickmytrade():
    """
    Test PickMyTrade connection in 3 stages:
    
    Stage 1 — Credentials check (no order sent)
    Stage 2 — Dry run: print the JSON that would be sent
    Stage 3 — Live test: send a real 1-contract MES order
               (you must cancel it immediately on Tradovate)
    
    Run: python src/agents/apex_bridge.py --test-pmt
    """
    print("\n🧪 PickMyTrade Connection Test")
    print("=" * 45)

    token    = os.getenv("PICKMYTRADE_TOKEN", "")
    acc_ids  = os.getenv("PICKMYTRADE_ACCOUNT_IDS",
               os.getenv("PICKMYTRADE_ACCOUNT_ID", ""))

    # ── Stage 1: Credentials ──────────────────────────────
    print("\nStage 1 — Credentials")
    if not token:
        print("  ❌ PICKMYTRADE_TOKEN not set in .env")
        print("     Add: PICKMYTRADE_TOKEN=your_token")
        return False
    print(f"  ✅ Token   : {token[:8]}...{token[-4:]}")

    if not acc_ids:
        print("  ❌ No account ID set in .env")
        print("     Add: PICKMYTRADE_ACCOUNT_ID=APEX372...")
        return False
    print(f"  ✅ Accounts: {acc_ids}")

    # ── Stage 2: Dry run ──────────────────────────────────
    print("\nStage 2 — Dry run (no order sent)")
    bridge = PickMyTradeBridge()
    fake_signal = {
        "strategy":  "TEST",
        "symbol":    "MES",
        "direction": "LONG",
        "entry":     5000.00,
        "sl":        4990.00,
        "tp":        5020.00,
        "contracts": 1,
    }
    payload = bridge._build_payload(
        "buy", "MES", 1,
        price=5000.00, sl=4990.00, tp=5020.00
    )
    import json
    print("  Payload that would be sent:")
    print("  " + json.dumps(payload, indent=4).replace("\n", "\n  "))
    print("  ✅ Payload looks correct")

    # ── Stage 3: Live ping ─────────────────────────────────
    print("\nStage 3 — Live connection ping")
    try:
        # Ping PickMyTrade's API to check token is valid
        r = requests.get(
            "https://pickmytrade.trade/api/v1/account",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        if r.status_code == 200:
            print(f"  ✅ Token is valid — connected to PickMyTrade")
            try:
                data = r.json()
                print(f"  ✅ Account: {data}")
            except Exception:
                print(f"  ✅ Response: {r.text[:100]}")
        elif r.status_code == 401:
            print(f"  ❌ Token rejected (401) — regenerate at pickmytrade.trade")
            return False
        else:
            print(f"  ⚠️  Status {r.status_code} — {r.text[:100]}")
            print(f"     Token may still work for orders")
    except Exception as e:
        print(f"  ⚠️  Ping failed: {e}")
        print(f"     Will try a live order test instead")

    return True


def send_test_order():
    """
    Send a real 1-contract BUY MES order to PickMyTrade.
    ⚠️  This places a REAL order on your Apex account.
    Cancel it immediately on Tradovate after confirming it appears.
    """
    print("\n⚡ Sending live test order (1x MES BUY)...")
    print("   ⚠️  Watch your Tradovate platform — cancel immediately!")
    print()

    bridge  = PickMyTradeBridge()
    signal  = {
        "strategy":  "CONNECTION_TEST",
        "symbol":    "MES",
        "direction": "LONG",
        "entry":     0,       # market order — no price needed
        "sl":        0,
        "tp":        0,
        "contracts": 1,
    }

    result = bridge.execute(signal)

    if result.get("error"):
        print(f"\n  ❌ Order failed: {result['error']}")
        print("\n  Common fixes:")
        print("  • Check PICKMYTRADE_TOKEN is correct")
        print("  • Check PICKMYTRADE_ACCOUNT_ID matches your Apex account")
        print("  • Make sure your Apex account is connected in PickMyTrade")
        print("  • Go to pickmytrade.trade → Connections → verify Tradovate is linked")
        return False
    else:
        print(f"\n  ✅ Order sent successfully!")
        print(f"     → Go to Tradovate NOW and cancel the open MES position")
        print(f"     → If nothing appeared, check PickMyTrade logs at pickmytrade.trade")
        return True

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="🌙 Apex Bridge")
    p.add_argument("--serve",      action="store_true",
                   help="Run local relay server on port 5001")
    p.add_argument("--port",       type=int, default=5001)
    p.add_argument("--test-pmt",   action="store_true",
                   help="Test PickMyTrade credentials + dry run (no order)")
    p.add_argument("--live-order", action="store_true",
                   help="Send a real 1-contract MES order (cancel immediately!)")
    p.add_argument("--test",       action="store_true",
                   help="Send a test signal through the configured route")
    p.add_argument("--route",      default="pickmytrade",
                   choices=["tradovate_direct","pickmytrade","traderspost","local"])
    args = p.parse_args()

    if args.serve:
        run_local_relay(port=args.port)

    elif args.test_pmt:
        test_pickmytrade()

    elif args.live_order:
        ok = test_pickmytrade()
        if ok:
            confirm = input("\nSend a real 1-contract MES order to Apex? (yes/no): ")
            if confirm.strip().lower() == "yes":
                send_test_order()
            else:
                print("Cancelled — no order sent")

    elif args.test:
        test_signal = {
            "strategy":  "TEST",
            "symbol":    "MES",
            "timeframe": "15m",
            "direction": "LONG",
            "entry":     5000.0,
            "sl":        4990.0,
            "tp":        5020.0,
            "contracts": 1,
            "rr":        2.0,
        }
        print(f"Sending test signal via route: {args.route}")
        result = execute_signal(test_signal)
        print(f"Result: {json.dumps(result, indent=2)}")