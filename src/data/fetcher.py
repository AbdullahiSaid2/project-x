# ============================================================
# 🌙 Data Fetcher — OHLCV + Liquidation data
# ============================================================

import os
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Coinbase symbol map (yfinance format) ────────────────────
COINBASE_YF_MAP = {
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "SOL-USD": "SOL-USD",
}

# ── Hyperliquid symbol map (yfinance format) ─────────────────
HYPERLIQUID_YF_MAP = {
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
    "SOL":  "SOL-USD",
    "ARB":  "ARB11841-USD",
    "AVAX": "AVAX-USD",
}

# ── Timeframe map ────────────────────────────────────────────
YF_INTERVAL_MAP = {
    "15m": "15m",
    "1H":  "1h",
    "4H":  "1h",   # yfinance doesn't have 4H, resample below
    "1D":  "1d",
}


def get_ohlcv(symbol: str, exchange: str = "hyperliquid",
              timeframe: str = "1H", days_back: int = 365) -> pd.DataFrame:
    """
    Fetch OHLCV data for a symbol.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    Index is a DatetimeIndex.
    """
    if exchange == "hyperliquid":
        yf_symbol = HYPERLIQUID_YF_MAP.get(symbol, symbol + "-USD")
    else:
        yf_symbol = COINBASE_YF_MAP.get(symbol, symbol)

    interval  = YF_INTERVAL_MAP.get(timeframe, "1h")
    end_date  = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    # yfinance caps intraday at 60 days
    if interval in ("15m", "1h") and days_back > 59:
        start_date = end_date - timedelta(days=59)

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {yf_symbol} ({interval})")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)   # strip tz for backtesting.py

    # Resample 1h → 4H if needed
    if timeframe == "4H":
        df = df.resample("4H").agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna()

    return df


def get_all_data(symbols: list, exchange: str = "hyperliquid",
                 timeframes: list = None, days_back: int = 59) -> dict:
    """
    Fetch OHLCV for multiple symbols × timeframes.
    Returns: {(symbol, timeframe): DataFrame}
    """
    if timeframes is None:
        timeframes = ["1H", "4H"]

    results = {}
    for sym in symbols:
        for tf in timeframes:
            key = (sym, tf)
            try:
                df = get_ohlcv(sym, exchange=exchange, timeframe=tf, days_back=days_back)
                results[key] = df
                print(f"  ✅ {sym} {tf}: {len(df)} candles")
            except Exception as e:
                print(f"  ❌ {sym} {tf}: {e}")
            time.sleep(0.3)   # be polite to yfinance

    return results


def get_liquidation_data(symbol: str = "BTC") -> dict:
    """Fetch liquidation data from CoinGlass (requires API key)."""
    api_key = os.getenv("COINGLASS_API_KEY")
    if not api_key:
        print("  ⚠️  No COINGLASS_API_KEY — skipping liquidation data")
        return {}

    url = "https://open-api.coinglass.com/public/v2/liquidation_history"
    headers = {"coinglassSecret": api_key}
    params  = {"symbol": symbol, "interval": "1h", "limit": 24}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        return r.json()
    except Exception as e:
        print(f"  ❌ CoinGlass error: {e}")
        return {}
