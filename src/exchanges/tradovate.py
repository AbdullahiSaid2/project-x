# ============================================================
# 🌙 Tradovate Exchange Connector
#
# Connects Algotec agents to Tradovate for futures trading.
# Supports MNQ, MES, MYM (and full-size NQ, ES, YM).
#
# SETUP:
#   1. Open account at tradovate.com (sim first)
#   2. Go to Account → API Access → generate credentials
#   3. Add to .env:
#      TRADOVATE_USERNAME=your_username
#      TRADOVATE_PASSWORD=your_password
#      TRADOVATE_APP_ID=your_app_id
#      TRADOVATE_APP_VERSION=1.0
#      TRADOVATE_CID=your_cid
#      TRADOVATE_SECRET=your_secret
#      TRADOVATE_SIM=true   (set false for live)
# ============================================================

import os
import json
import time
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Endpoints ─────────────────────────────────────────────────
SIM_URL  = "https://demo.tradovateapi.com/v1"
LIVE_URL = "https://live.tradovateapi.com/v1"

def _base_url() -> str:
    sim = os.getenv("TRADOVATE_SIM", "true").lower()
    return SIM_URL if sim == "true" else LIVE_URL

# ── Tick sizes and contract values ───────────────────────────
FUTURES_SPECS = {
    "MNQ": {"tick_size": 0.25, "tick_value": 0.50,  "margin_intraday": 40},
    "MES": {"tick_size": 0.25, "tick_value": 12.50, "margin_intraday": 50},
    "MYM": {"tick_size": 1.0,  "tick_value": 0.50,  "margin_intraday": 50},
    "NQ":  {"tick_size": 0.25, "tick_value": 5.00,  "margin_intraday": 500},
    "ES":  {"tick_size": 0.25, "tick_value": 12.50, "margin_intraday": 500},
    "YM":  {"tick_size": 1.0,  "tick_value": 5.00,  "margin_intraday": 500},
}

# ── Session token (refreshed automatically) ──────────────────
_session = {"token": None, "expires": 0, "account_id": None}


def _authenticate() -> str:
    """Authenticate and return access token."""
    if _session["token"] and time.time() < _session["expires"]:
        return _session["token"]

    username = os.getenv("TRADOVATE_USERNAME")
    password = os.getenv("TRADOVATE_PASSWORD")
    app_id   = os.getenv("TRADOVATE_APP_ID")
    app_ver  = os.getenv("TRADOVATE_APP_VERSION", "1.0")
    cid      = os.getenv("TRADOVATE_CID")
    secret   = os.getenv("TRADOVATE_SECRET")

    if not all([username, password, app_id, cid, secret]):
        raise EnvironmentError(
            "Missing Tradovate credentials in .env\n"
            "Required: TRADOVATE_USERNAME, TRADOVATE_PASSWORD, "
            "TRADOVATE_APP_ID, TRADOVATE_CID, TRADOVATE_SECRET\n"
            "Get these from tradovate.com → Account → API Access"
        )

    payload = {
        "name":       username,
        "password":   password,
        "appId":      app_id,
        "appVersion": app_ver,
        "cid":        int(cid),
        "sec":        secret,
        "deviceId":   "algotec-trading",
    }

    r = requests.post(
        f"{_base_url()}/auth/accesstokenrequest",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()

    if "errorText" in data:
        raise ValueError(f"Tradovate auth failed: {data['errorText']}")

    _session["token"]   = data["accessToken"]
    _session["expires"] = time.time() + 75 * 60   # 75 min (token valid 80 min)

    # Fetch account ID
    accounts = _get("/account/list")
    if accounts:
        _session["account_id"] = accounts[0]["id"]
        env = "SIM" if os.getenv("TRADOVATE_SIM","true")=="true" else "LIVE"
        print(f"  ✅ Tradovate authenticated [{env}] — Account: {accounts[0].get('name','')}")

    return _session["token"]


def _headers() -> dict:
    token = _authenticate()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }


def _get(path: str, params: dict = None) -> any:
    r = requests.get(
        f"{_base_url()}{path}",
        headers=_headers(),
        params=params,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def _post(path: str, body: dict) -> any:
    r = requests.post(
        f"{_base_url()}{path}",
        headers=_headers(),
        json=body,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# ── Contract resolution ───────────────────────────────────────
_contract_cache: dict[str, int] = {}

def _get_contract_id(symbol: str) -> int:
    """Resolve a symbol like MNQ to its current front-month contract ID."""
    if symbol in _contract_cache:
        return _contract_cache[symbol]

    results = _get("/contract/suggest", params={"t": symbol, "l": 5})
    for c in results:
        if c.get("name", "").startswith(symbol):
            _contract_cache[symbol] = c["id"]
            return c["id"]

    raise ValueError(f"Could not find contract for {symbol}")


# ── Market Data ───────────────────────────────────────────────
def get_price(symbol: str) -> float:
    """Get latest price for a futures contract."""
    contract_id = _get_contract_id(symbol)
    data = _get(f"/marketdata/quote", params={"contractId": contract_id})
    # Use mid price of bid/ask
    bid = float(data.get("bid", 0))
    ask = float(data.get("ask", 0))
    if bid and ask:
        return (bid + ask) / 2
    return float(data.get("lastPrice", 0))


def get_balance() -> dict:
    """Get account cash balance and margin info."""
    account_id = _session.get("account_id")
    if not account_id:
        _authenticate()
        account_id = _session.get("account_id")

    data = _get(f"/cashBalance/getcashbalancesnapshot",
                params={"accountId": account_id})
    return {
        "account_value":     float(data.get("totalCashValue", 0)),
        "cash_balance":      float(data.get("cashBalance", 0)),
        "open_pnl":          float(data.get("openTradingProfit", 0)),
        "available_funds":   float(data.get("realizedPnL", 0)),
        "withdrawable":      float(data.get("cashBalance", 0)),
    }


def get_positions() -> list[dict]:
    """Get all open positions."""
    account_id = _session.get("account_id")
    data = _get("/position/list", params={"accountId": account_id})
    positions = []
    for pos in data:
        net_pos = int(pos.get("netPos", 0))
        if net_pos != 0:
            positions.append({
                "symbol":      pos.get("contractId", ""),
                "size":        net_pos,
                "side":        "LONG" if net_pos > 0 else "SHORT",
                "avg_price":   float(pos.get("avgPrice", 0)),
                "open_pnl":    float(pos.get("openPnL", 0)),
            })
    return positions


# ── Order Execution ───────────────────────────────────────────
def _round_to_tick(price: float, symbol: str) -> float:
    """Round a price to the nearest valid tick for a symbol."""
    spec = FUTURES_SPECS.get(symbol, {"tick_size": 0.25})
    tick = spec["tick_size"]
    return round(round(price / tick) * tick, 10)


def market_buy(symbol: str, contracts: int = 1) -> dict:
    """
    Place a market BUY order.
    contracts: number of contracts (minimum 1, no fractional)
    """
    contract_id = _get_contract_id(symbol)
    price       = get_price(symbol)
    account_id  = _session.get("account_id")

    print(f"  🟢 TV BUY  {symbol}: {contracts} contract(s) @ ~${price:,.2f}")

    body = {
        "accountSpec":     os.getenv("TRADOVATE_USERNAME"),
        "accountId":       account_id,
        "action":          "Buy",
        "symbol":          symbol,
        "orderQty":        contracts,
        "orderType":       "Market",
        "isAutomated":     True,
    }
    result = _post("/order/placeorder", body)
    print(f"  ✅ Order placed: {result}")
    return result


def market_sell(symbol: str, contracts: int = 1) -> dict:
    """Place a market SELL (short) order."""
    contract_id = _get_contract_id(symbol)
    price       = get_price(symbol)
    account_id  = _session.get("account_id")

    print(f"  🔴 TV SELL {symbol}: {contracts} contract(s) @ ~${price:,.2f}")

    body = {
        "accountSpec": os.getenv("TRADOVATE_USERNAME"),
        "accountId":   account_id,
        "action":      "Sell",
        "symbol":      symbol,
        "orderQty":    contracts,
        "orderType":   "Market",
        "isAutomated": True,
    }
    result = _post("/order/placeorder", body)
    print(f"  ✅ Order placed: {result}")
    return result


def close_position(symbol: str) -> dict:
    """Close all open contracts for a symbol."""
    positions = get_positions()
    for pos in positions:
        if str(pos.get("symbol")) in str(_get_contract_id(symbol)):
            size = abs(pos["size"])
            if pos["side"] == "LONG":
                return market_sell(symbol, size)
            else:
                return market_buy(symbol, size)
    print(f"  ⚠️  No open position for {symbol}")
    return {}


def place_bracket_order(symbol: str, action: str, contracts: int,
                         stop_ticks: int, target_ticks: int) -> dict:
    """
    Place a bracket order with automatic SL and TP.
    stop_ticks:   number of ticks for stop loss
    target_ticks: number of ticks for take profit
    """
    spec       = FUTURES_SPECS.get(symbol, {"tick_size": 0.25})
    tick       = spec["tick_size"]
    price      = get_price(symbol)
    account_id = _session.get("account_id")

    if action.upper() == "BUY":
        sl_price = _round_to_tick(price - stop_ticks  * tick, symbol)
        tp_price = _round_to_tick(price + target_ticks * tick, symbol)
    else:
        sl_price = _round_to_tick(price + stop_ticks  * tick, symbol)
        tp_price = _round_to_tick(price - target_ticks * tick, symbol)

    print(f"  {'🟢' if action=='BUY' else '🔴'} BRACKET {symbol}: "
          f"{contracts} x {action} @ ${price:,.2f} | SL ${sl_price:,.2f} | TP ${tp_price:,.2f}")

    body = {
        "accountSpec": os.getenv("TRADOVATE_USERNAME"),
        "accountId":   account_id,
        "action":      action.capitalize(),
        "symbol":      symbol,
        "orderQty":    contracts,
        "orderType":   "Market",
        "isAutomated": True,
        "bracket1": {
            "action":    "Sell" if action.upper() == "BUY" else "Buy",
            "orderType": "Stop",
            "stopPrice": sl_price,
            "orderQty":  contracts,
        },
        "bracket2": {
            "action":    "Sell" if action.upper() == "BUY" else "Buy",
            "orderType": "Limit",
            "price":     tp_price,
            "orderQty":  contracts,
        },
    }
    result = _post("/order/placeoso", body)
    print(f"  ✅ Bracket order placed: {result}")
    return result


# ── Position sizing helper ────────────────────────────────────
def usd_to_contracts(symbol: str, usd_amount: float) -> int:
    """
    Convert a USD amount to number of contracts.
    Ensures we never go below 1 contract.
    """
    spec   = FUTURES_SPECS.get(symbol, {"margin_intraday": 100})
    margin = spec["margin_intraday"]
    contracts = max(1, int(usd_amount / margin))
    return contracts


# ── Market hours check ────────────────────────────────────────
def is_market_open() -> bool:
    """
    Check if futures market is currently open.
    CME futures trade Sun 6pm - Fri 5pm EST with 1hr break 4-5pm EST daily.
    """
    import pytz
    est  = pytz.timezone("America/New_York")
    now  = datetime.now(est)
    hour = now.hour
    wday = now.weekday()   # 0=Mon, 6=Sun

    # Closed Saturday
    if wday == 5:
        return False
    # Closed Friday after 5pm
    if wday == 4 and hour >= 17:
        return False
    # Closed Sunday before 6pm
    if wday == 6 and hour < 18:
        return False
    # Daily maintenance 4-5pm EST
    if hour == 16:
        return False

    return True
