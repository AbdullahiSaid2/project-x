# ============================================================
# 🌙 Data Fetcher — OHLCV + Liquidation + Futures data
# ============================================================

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import requests
import yfinance as yf
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
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "ARB": "ARB11841-USD",
    "AVAX": "AVAX-USD",
}

# ── Futures symbol map ────────────────────────────────────────
FUTURES_YF_MAP = {
    "MNQ": "NQ=F",
    "MES": "ES=F",
    "MYM": "YM=F",
    "NQ": "NQ=F",
    "ES": "ES=F",
    "YM": "YM=F",
    "GC": "GC=F",
    "CL": "CL=F",
    "MGC": "GC=F",
    "MCL": "CL=F",
}

FUTURES_SYMBOLS = set(FUTURES_YF_MAP.keys())

# ── Timeframe map ─────────────────────────────────────────────
YF_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1H": "1h",
    "4H": "1h",
    "1D": "1d",
    "1W": "1wk",
}


def _is_futures(symbol: str) -> bool:
    return symbol.upper() in FUTURES_SYMBOLS


def _get_yf_symbol(symbol: str, exchange: str) -> str:
    sym = symbol.upper()
    if _is_futures(sym):
        return FUTURES_YF_MAP[sym]
    if exchange == "hyperliquid":
        return HYPERLIQUID_YF_MAP.get(sym, sym + "-USD")
    if exchange == "coinbase":
        return COINBASE_YF_MAP.get(sym, sym)
    if exchange == "tradovate":
        return FUTURES_YF_MAP.get(sym, sym + "=F")
    return sym + "-USD"


def _has_tv_data(symbol: str, timeframe: str) -> bool:
    try:
        from src.data.tradingview_fetcher import _find_csv
        return _find_csv(symbol, timeframe) is not None
    except Exception:
        return False


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    tf = timeframe.strip()

    if tf in {"1m", "5m", "15m", "1H", "1D", "1W"}:
        return df

    if tf == "4H":
        return (
            df.resample("4h")
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna()
        )

    raise ValueError(f"Unsupported timeframe for resample: {timeframe}")


def get_ohlcv(
    symbol: str,
    exchange: str = "hyperliquid",
    timeframe: str = "1H",
    days_back: int = 365,
) -> pd.DataFrame:
    """
    Fetch OHLCV data.

    Priority:
      1) Databento local cache / Databento API for supported futures symbols
      2) CCXT for crypto
      3) TradingView CSV
      4) yfinance fallback
    """
    symbol = symbol.upper().strip()

    # 1. Databento first for futures
    try:
        from src.data.databento_fetcher import FUTURES_DB_MAP, get_databento_ohlcv

        if os.getenv("DATABENTO_API_KEY") and symbol in FUTURES_DB_MAP:
            return get_databento_ohlcv(symbol, timeframe, days_back=days_back, force_refresh=False)
    except FileNotFoundError:
        pass
    except Exception as e:
        if os.getenv("DEBUG_FETCHER") == "1":
            print(f"  ⚠️ Databento fallback for {symbol} {timeframe}: {e}")

    # 2. CCXT for crypto
    try:
        from src.data.ccxt_fetcher import BINANCE_PAIRS, get_crypto_ohlcv

        if symbol in BINANCE_PAIRS:
            return get_crypto_ohlcv(symbol, timeframe, days_back)
    except ImportError:
        pass
    except Exception as e:
        if os.getenv("DEBUG_FETCHER") == "1":
            print(f"  ⚠️ CCXT fallback for {symbol} {timeframe}: {e}")

    # 3. TradingView CSV
    try:
        from src.data.tradingview_fetcher import get_tv_ohlcv

        df = get_tv_ohlcv(symbol, timeframe)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        if os.getenv("DEBUG_FETCHER") == "1":
            print(f"  ⚠️ TradingView fallback for {symbol} {timeframe}: {e}")

    # 4. yfinance fallback
    yf_symbol = _get_yf_symbol(symbol, exchange)
    interval = YF_INTERVAL_MAP.get(timeframe, "1h")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    if interval in ("1m", "5m", "15m", "1h") and days_back > 59:
        start_date = end_date - timedelta(days=59)

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {yf_symbol} ({interval})")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df = _resample_ohlcv(df, timeframe)
    return df


def get_futures_ohlcv(
    symbol: str,
    timeframe: str = "1H",
    days_back: int = 365,
) -> pd.DataFrame:
    return get_ohlcv(
        symbol,
        exchange="tradovate",
        timeframe=timeframe,
        days_back=days_back,
    )


def get_all_data(
    symbols: List[str],
    exchange: str = "hyperliquid",
    timeframes: List[str] | None = None,
    days_back: int = 59,
) -> Dict[tuple, pd.DataFrame]:
    if timeframes is None:
        timeframes = ["1H", "4H"]

    results: Dict[tuple, pd.DataFrame] = {}
    for sym in symbols:
        for tf in timeframes:
            try:
                df = get_ohlcv(
                    sym,
                    exchange=exchange,
                    timeframe=tf,
                    days_back=days_back,
                )
                results[(sym, tf)] = df
                src = "TV" if _has_tv_data(sym, tf) else "fetcher"
                print(f"  ✅ {sym} {tf}: {len(df)} candles [{src}]")
            except Exception as e:
                print(f"  ❌ {sym} {tf}: {e}")
            time.sleep(0.3)

    return results


def get_liquidation_data(symbol: str = "BTC") -> dict:
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