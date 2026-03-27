# ============================================================
# 🌙 Hyperliquid Exchange Connector
# Docs: https://hyperliquid.gitbook.io/hyperliquid-docs/
# ============================================================

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL    = "https://api.hyperliquid.xyz"
INFO_URL    = f"{BASE_URL}/info"
EXCHANGE_URL = f"{BASE_URL}/exchange"


def _headers() -> dict:
    return {"Content-Type": "application/json"}


# ── Market Data (no auth needed) ────────────────────────────

def get_price(symbol: str) -> float:
    """Get latest mid price for a symbol (e.g. 'BTC')."""
    payload = {"type": "allMids"}
    r = requests.post(INFO_URL, json=payload, headers=_headers(), timeout=10)
    r.raise_for_status()
    mids = r.json()
    key = f"{symbol}/USDC" if "/" not in symbol else symbol
    # Hyperliquid returns a dict like {"BTC/USDC": "67234.5", ...}
    for k, v in mids.items():
        if symbol.upper() in k.upper():
            return float(v)
    raise ValueError(f"Symbol {symbol} not found in Hyperliquid mids")


def get_ohlcv_hl(symbol: str, interval: str = "1h", lookback: int = 200) -> list[dict]:
    """
    Fetch OHLCV candles from Hyperliquid.
    interval options: '1m','5m','15m','1h','4h','1d'
    Returns list of {t, o, h, l, c, v}
    """
    end_ms   = int(time.time() * 1000)
    interval_ms_map = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000,
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
    }
    ms       = interval_ms_map.get(interval, 3_600_000)
    start_ms = end_ms - ms * lookback

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin":        symbol,
            "interval":    interval,
            "startTime":   start_ms,
            "endTime":     end_ms,
        },
    }
    r = requests.post(INFO_URL, json=payload, headers=_headers(), timeout=15)
    r.raise_for_status()
    return r.json()


def get_positions(wallet_address: str) -> list[dict]:
    """Get all open perpetual positions for a wallet."""
    payload = {"type": "clearinghouseState", "user": wallet_address}
    r = requests.post(INFO_URL, json=payload, headers=_headers(), timeout=10)
    r.raise_for_status()
    data = r.json()
    positions = []
    for pos in data.get("assetPositions", []):
        p = pos.get("position", {})
        if float(p.get("szi", 0)) != 0:
            positions.append({
                "symbol":      p.get("coin"),
                "size":        float(p.get("szi", 0)),
                "entry_price": float(p.get("entryPx", 0)),
                "unrealized_pnl": float(p.get("unrealizedPnl", 0)),
                "leverage":    p.get("leverage", {}),
            })
    return positions


def get_balance(wallet_address: str) -> dict:
    """Get account balance summary."""
    payload = {"type": "clearinghouseState", "user": wallet_address}
    r = requests.post(INFO_URL, json=payload, headers=_headers(), timeout=10)
    r.raise_for_status()
    data  = r.json()
    mval  = data.get("marginSummary", {})
    return {
        "account_value":     float(mval.get("accountValue", 0)),
        "total_margin_used": float(mval.get("totalMarginUsed", 0)),
        "total_raw_usd":     float(mval.get("totalRawUsd", 0)),
        "withdrawable":      float(data.get("withdrawable", 0)),
    }


# ── Order Execution (requires private key) ─────────────────
# NOTE: Full on-chain signing requires the hyperliquid-python-sdk.
# Install it with:  pip install hyperliquid-python-sdk
# The functions below show how to use the SDK once installed.

def _get_agent():
    """Return an authenticated Hyperliquid agent (requires SDK)."""
    try:
        from hyperliquid.utils import constants
        from hyperliquid.exchange import Exchange
        from eth_account import Account
    except ImportError:
        raise ImportError(
            "Install hyperliquid-python-sdk: pip install hyperliquid-python-sdk\n"
            "Also: pip install eth-account"
        )

    private_key     = os.getenv("HYPERLIQUID_PRIVATE_KEY")
    wallet_address  = os.getenv("HYPERLIQUID_WALLET_ADDRESS")

    if not private_key or not wallet_address:
        raise EnvironmentError(
            "Set HYPERLIQUID_PRIVATE_KEY and HYPERLIQUID_WALLET_ADDRESS in your .env"
        )

    account  = Account.from_key(private_key)
    exchange = Exchange(account, constants.MAINNET_API_URL)
    return exchange, wallet_address


def market_buy(symbol: str, usd_amount: float, leverage: int = 1) -> dict:
    """
    Open a LONG position on Hyperliquid.
    usd_amount: how many USD to spend (not number of coins).
    """
    exchange, _ = _get_agent()
    price       = get_price(symbol)
    size        = round(usd_amount / price, 6)
    print(f"  🟢 HL BUY  {symbol}: ${usd_amount:.2f} → {size} @ ~${price:.2f}")
    result = exchange.market_open(symbol, True, size, None, 0.01)   # 1% slippage
    print(f"  ✅ Order result: {result}")
    return result


def market_sell(symbol: str, usd_amount: float) -> dict:
    """Open a SHORT position on Hyperliquid."""
    exchange, _ = _get_agent()
    price       = get_price(symbol)
    size        = round(usd_amount / price, 6)
    print(f"  🔴 HL SELL {symbol}: ${usd_amount:.2f} → {size} @ ~${price:.2f}")
    result = exchange.market_open(symbol, False, size, None, 0.01)
    print(f"  ✅ Order result: {result}")
    return result


def close_position(symbol: str) -> dict:
    """Close all open position for a symbol."""
    exchange, wallet = _get_agent()
    positions = get_positions(wallet)
    for pos in positions:
        if pos["symbol"] == symbol:
            size    = abs(pos["size"])
            is_long = pos["size"] > 0
            print(f"  🔒 Closing {symbol} position: {size} ({'long' if is_long else 'short'})")
            result = exchange.market_close(symbol, size, None, 0.01)
            return result
    print(f"  ⚠️  No open position found for {symbol}")
    return {}


def set_leverage(symbol: str, leverage: int):
    """Set cross-margin leverage for a symbol."""
    exchange, _ = _get_agent()
    exchange.update_leverage(leverage, symbol, is_cross=True)
    print(f"  ⚙️  Leverage set: {symbol} × {leverage}")
