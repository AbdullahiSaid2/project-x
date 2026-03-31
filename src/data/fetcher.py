# ============================================================
# 🌙 Data Fetcher — OHLCV + Liquidation + Futures data
# ============================================================

import os
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Coinbase symbol map ───────────────────────────────────────
COINBASE_YF_MAP = {
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "SOL-USD": "SOL-USD",
}

# ── Hyperliquid symbol map ────────────────────────────────────
HYPERLIQUID_YF_MAP = {
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
    "SOL":  "SOL-USD",
    "ARB":  "ARB11841-USD",
    "AVAX": "AVAX-USD",
}

# ── Futures symbol map ────────────────────────────────────────
FUTURES_YF_MAP = {
    "MNQ": "NQ=F",
    "MES": "ES=F",
    "MYM": "YM=F",
    "NQ":  "NQ=F",
    "ES":  "ES=F",
    "YM":  "YM=F",
    "GC":  "GC=F",
    "CL":  "CL=F",
}

FUTURES_SYMBOLS = set(FUTURES_YF_MAP.keys())

# ── Timeframe map ─────────────────────────────────────────────
YF_INTERVAL_MAP = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "1H":  "1h",
    "4H":  "1h",
    "1D":  "1d",
    "1W":  "1wk",
}


def _is_futures(symbol: str) -> bool:
    return symbol.upper() in FUTURES_SYMBOLS


def _get_yf_symbol(symbol: str, exchange: str) -> str:
    sym = symbol.upper()
    if _is_futures(sym):
        return FUTURES_YF_MAP[sym]
    if exchange == "hyperliquid":
        return HYPERLIQUID_YF_MAP.get(sym, sym + "-USD")
    elif exchange == "coinbase":
        return COINBASE_YF_MAP.get(sym, sym)
    elif exchange == "tradovate":
        return FUTURES_YF_MAP.get(sym, sym + "=F")
    return sym + "-USD"


def _has_tv_data(symbol: str, timeframe: str) -> bool:
    try:
        from src.data.tradingview_fetcher import _find_csv
        return _find_csv(symbol, timeframe) is not None
    except Exception:
        return False


def get_ohlcv(symbol: str, exchange: str = "hyperliquid",
              timeframe: str = "1H", days_back: int = 365) -> pd.DataFrame:
    """
    Fetch OHLCV data.
    Priority: Databento → TradingView CSV → yfinance
    Returns DataFrame with Open, High, Low, Close, Volume columns.
    """
    # 1. Try Databento first for futures symbols (best quality, full history)
    try:
        from src.data.databento_fetcher import (
            get_databento_ohlcv, FUTURES_DB_MAP
        )
        import os
        if os.getenv("DATABENTO_API_KEY") and symbol.upper() in FUTURES_DB_MAP:
            return get_databento_ohlcv(symbol, timeframe, days_back=days_back)
    except FileNotFoundError:
        pass
    except Exception as e:
        if "credits" not in str(e).lower():
            pass   # silent fallback for non-credit errors

    # 2. Try CCXT for crypto symbols (Binance — reliable, 3+ years)
    try:
        from src.data.ccxt_fetcher import get_crypto_ohlcv, BINANCE_PAIRS
        if symbol.upper() in BINANCE_PAIRS:
            return get_crypto_ohlcv(symbol, timeframe, days_back)
    except ImportError:
        pass   # ccxt not installed
    except Exception:
        pass   # fall through to TradingView

    # 3. Try TradingView CSV
    try:
        from src.data.tradingview_fetcher import get_tv_ohlcv
        df = get_tv_ohlcv(symbol, timeframe)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # 4. Fall back to yfinance
    yf_symbol  = _get_yf_symbol(symbol, exchange)
    interval   = YF_INTERVAL_MAP.get(timeframe, "1h")
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    if interval in ("1m", "5m", "15m", "1h") and days_back > 59:
        start_date = end_date - timedelta(days=59)

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {yf_symbol} ({interval})")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)

    if timeframe == "4H":
        df = df.resample("4h").agg({
            "Open": "first", "High": "max",
            "Low":  "min",   "Close": "last",
            "Volume": "sum",
        }).dropna()

    return df


def get_futures_ohlcv(symbol: str, timeframe: str = "1H",
                       days_back: int = 59) -> pd.DataFrame:
    """Convenience function for futures data."""
    return get_ohlcv(symbol, exchange="tradovate",
                     timeframe=timeframe, days_back=days_back)


def get_all_data(symbols: list, exchange: str = "hyperliquid",
                 timeframes: list = None, days_back: int = 59) -> dict:
    """Fetch OHLCV for multiple symbols x timeframes."""
    if timeframes is None:
        timeframes = ["1H", "4H"]
    results = {}
    for sym in symbols:
        for tf in timeframes:
            try:
                df = get_ohlcv(sym, exchange=exchange,
                                timeframe=tf, days_back=days_back)
                results[(sym, tf)] = df
                src = "TV" if _has_tv_data(sym, tf) else "yf"
                print(f"  ✅ {sym} {tf}: {len(df)} candles [{src}]")
            except Exception as e:
                print(f"  ❌ {sym} {tf}: {e}")
            time.sleep(0.3)
    return results


def get_liquidation_data(symbol: str = "BTC") -> dict:
    """Fetch liquidation data from CoinGlass."""
    api_key = os.getenv("COINGLASS_API_KEY")
    if not api_key:
        return {}
    try:
        r = requests.get(
            "https://open-api.coinglass.com/public/v2/liquidation_history",
            headers={"coinglassSecret": api_key},
            params={"symbol": symbol, "interval": "1h", "limit": 24},
            timeout=10,
        )
        return r.json()
    except Exception as e:
        print(f"  ❌ CoinGlass error: {e}")
        return {}