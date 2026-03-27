# ============================================================
# 🌙 Coinbase Advanced Trade Connector
# Uses the Coinbase Advanced Trade REST API v3
# Docs: https://docs.cdp.coinbase.com/advanced-trade/docs/welcome
# ============================================================

import os
import uuid
import time
import hmac
import hashlib
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.coinbase.com"


def _get_keys() -> tuple[str, str]:
    key    = os.getenv("COINBASE_API_KEY")
    secret = os.getenv("COINBASE_API_SECRET")
    if not key or not secret:
        raise EnvironmentError(
            "Set COINBASE_API_KEY and COINBASE_API_SECRET in your .env\n"
            "Create keys at: https://www.coinbase.com/settings/api"
        )
    return key, secret


def _sign(method: str, path: str, body: str = "") -> dict:
    """Build signed headers for Coinbase Advanced Trade API."""
    api_key, api_secret = _get_keys()
    timestamp   = str(int(time.time()))
    message     = timestamp + method.upper() + path + body
    signature   = hmac.new(
        api_secret.encode("utf-8"),
        message.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()
    return {
        "CB-ACCESS-KEY":       api_key,
        "CB-ACCESS-SIGN":      signature,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "Content-Type":        "application/json",
    }


def _get(path: str, params: dict = None) -> dict:
    headers = _sign("GET", path)
    r = requests.get(BASE_URL + path, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def _post(path: str, body: dict) -> dict:
    import json
    body_str = json.dumps(body)
    headers  = _sign("POST", path, body_str)
    r = requests.post(BASE_URL + path, headers=headers, data=body_str, timeout=10)
    r.raise_for_status()
    return r.json()


# ── Market Data ─────────────────────────────────────────────

def get_price(symbol: str) -> float:
    """
    Get best bid/ask mid price.
    symbol format: 'BTC-USD', 'ETH-USD', 'SOL-USD'
    """
    path = f"/api/v3/brokerage/best_bid_ask"
    data = _get(path, params={"product_ids": symbol})
    for entry in data.get("pricebooks", []):
        if entry.get("product_id") == symbol:
            bid = float(entry["bids"][0]["price"])
            ask = float(entry["asks"][0]["price"])
            return (bid + ask) / 2
    raise ValueError(f"Symbol {symbol} not found")


def get_candles(symbol: str, granularity: str = "ONE_HOUR",
                limit: int = 200) -> list[dict]:
    """
    Fetch OHLCV candles.
    granularity options:
      ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE,
      THIRTY_MINUTE, ONE_HOUR, TWO_HOUR, SIX_HOUR, ONE_DAY
    """
    end_ts   = int(time.time())
    gran_sec = {
        "ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900,
        "THIRTY_MINUTE": 1800, "ONE_HOUR": 3600, "TWO_HOUR": 7200,
        "SIX_HOUR": 21600, "ONE_DAY": 86400,
    }.get(granularity, 3600)
    start_ts = end_ts - gran_sec * limit

    path = f"/api/v3/brokerage/products/{symbol}/candles"
    data = _get(path, params={
        "start":       str(start_ts),
        "end":         str(end_ts),
        "granularity": granularity,
    })
    return data.get("candles", [])


def get_balance(currency: str = "USD") -> float:
    """Get available balance for a currency (e.g. 'USD', 'BTC')."""
    data = _get("/api/v3/brokerage/accounts")
    for acct in data.get("accounts", []):
        if acct.get("currency") == currency:
            return float(acct["available_balance"]["value"])
    return 0.0


def get_positions() -> list[dict]:
    """Get open orders / positions."""
    data = _get("/api/v3/brokerage/orders/historical/batch", params={"order_status": "OPEN"})
    return data.get("orders", [])


# ── Order Execution ─────────────────────────────────────────

def market_buy(symbol: str, usd_amount: float) -> dict:
    """
    Place a market BUY order.
    symbol: e.g. 'BTC-USD'
    usd_amount: how much USD to spend
    """
    print(f"  🟢 CB BUY  {symbol}: ${usd_amount:.2f}")
    body = {
        "client_order_id": str(uuid.uuid4()),
        "product_id":      symbol,
        "side":            "BUY",
        "order_configuration": {
            "market_market_ioc": {
                "quote_size": str(round(usd_amount, 2)),
            }
        },
    }
    result = _post("/api/v3/brokerage/orders", body)
    print(f"  ✅ Order result: {result.get('success_response', result)}")
    return result


def market_sell(symbol: str, base_amount: float) -> dict:
    """
    Place a market SELL order.
    symbol: e.g. 'BTC-USD'
    base_amount: how many units of the base currency to sell (e.g. 0.001 BTC)
    """
    print(f"  🔴 CB SELL {symbol}: {base_amount} units")
    body = {
        "client_order_id": str(uuid.uuid4()),
        "product_id":      symbol,
        "side":            "SELL",
        "order_configuration": {
            "market_market_ioc": {
                "base_size": str(round(base_amount, 8)),
            }
        },
    }
    result = _post("/api/v3/brokerage/orders", body)
    print(f"  ✅ Order result: {result.get('success_response', result)}")
    return result


def limit_buy(symbol: str, usd_amount: float, price: float,
              post_only: bool = True) -> dict:
    """Place a limit BUY (maker) order."""
    base_size = round(usd_amount / price, 8)
    body = {
        "client_order_id": str(uuid.uuid4()),
        "product_id":      symbol,
        "side":            "BUY",
        "order_configuration": {
            "limit_limit_gtc": {
                "base_size":   str(base_size),
                "limit_price": str(round(price, 2)),
                "post_only":   post_only,
            }
        },
    }
    return _post("/api/v3/brokerage/orders", body)


def cancel_order(order_id: str) -> dict:
    """Cancel an open order by ID."""
    return _post("/api/v3/brokerage/orders/batch_cancel", {"order_ids": [order_id]})
