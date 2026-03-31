# ============================================================
# 🌙 CCXT Crypto Data Fetcher
#
# Provides reliable OHLCV data from Binance (and other
# exchanges) via CCXT — the same approach Moon Dev uses.
#
# WHY CCXT OVER YFINANCE:
#   - yfinance: 59 days max on intraday, unreliable crypto
#   - CCXT: 3-5 years of data, direct from exchanges, free
#   - Binance has the deepest crypto liquidity and cleanest data
#
# SUPPORTED EXCHANGES:
#   - binance (default) — best liquidity, most history
#   - bybit             — good alternative
#   - coinbase          — US-focused
#
# USAGE:
#   from src.data.ccxt_fetcher import get_crypto_ohlcv
#   df = get_crypto_ohlcv("BTC", timeframe="1H", days_back=365)
# ============================================================

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ── Cache config ──────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
CACHE_DIR   = REPO_ROOT / "src" / "data" / "ccxt_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_HOURS = 4   # refresh cache every 4 hours

# ── Symbol map — token → Binance pair ────────────────────────
BINANCE_PAIRS = {
    # Major crypto
    "BTC":   "BTC/USDT",
    "ETH":   "ETH/USDT",
    "SOL":   "SOL/USDT",
    "BNB":   "BNB/USDT",
    "XRP":   "XRP/USDT",
    "DOGE":  "DOGE/USDT",
    "ADA":   "ADA/USDT",
    "AVAX":  "AVAX/USDT",
    "LINK":  "LINK/USDT",
    "DOT":   "DOT/USDT",
    # DeFi / L2
    "ARB":   "ARB/USDT",
    "OP":    "OP/USDT",
    "UNI":   "UNI/USDT",
    "AAVE":  "AAVE/USDT",
    "CRV":   "CRV/USDT",
    # Meme / high volatility
    "PEPE":  "PEPE/USDT",
    "WIF":   "WIF/USDT",
    "BONK":  "BONK/USDT",
    # AI tokens
    "FET":   "FET/USDT",
    "RNDR":  "RENDER/USDT",
    "TAO":   "TAO/USDT",
}

# ── Timeframe map — standard → CCXT ──────────────────────────
TF_MAP = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "1H":  "1h",
    "4H":  "4h",
    "1D":  "1d",
    "1W":  "1w",
}

# Max candles per CCXT request (Binance limit)
BINANCE_LIMIT = 1000


def _cache_path(symbol: str, timeframe: str, days: int) -> Path:
    safe = f"{symbol}_{timeframe}_{days}d"
    return CACHE_DIR / f"{safe}.parquet"


def _cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds()
    return age < CACHE_HOURS * 3600


def get_crypto_ohlcv(symbol: str,
                     timeframe: str = "1H",
                     days_back: int = 365,
                     exchange_id: str = "binance") -> pd.DataFrame:
    """
    Fetch OHLCV data for a crypto token via CCXT.

    Args:
        symbol:      Token name e.g. "BTC", "ETH", "SOL"
        timeframe:   "1m", "5m", "15m", "1H", "4H", "1D"
        days_back:   How many days of history to fetch
        exchange_id: "binance" (default), "bybit", "coinbase"

    Returns:
        DataFrame with Open, High, Low, Close, Volume columns
        indexed by UTC datetime.
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError(
            "ccxt not installed. Run: pip install ccxt"
        )

    sym_upper  = symbol.upper()
    pair       = BINANCE_PAIRS.get(sym_upper, f"{sym_upper}/USDT")
    ccxt_tf    = TF_MAP.get(timeframe, "1h")

    # Check cache
    cache = _cache_path(sym_upper, timeframe, days_back)
    if _cache_valid(cache):
        print(f"  📦 {symbol} {timeframe}: loading from ccxt cache")
        return pd.read_parquet(cache)

    print(f"  📡 CCXT/{exchange_id}: fetching {symbol} {timeframe} "
          f"({days_back} days from {exchange_id})...")

    try:
        exchange = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        # Calculate start time
        since_ms = int(
            (datetime.utcnow() - timedelta(days=days_back)).timestamp() * 1000
        )

        all_candles = []
        fetch_since = since_ms

        while True:
            candles = exchange.fetch_ohlcv(
                pair, ccxt_tf,
                since=fetch_since,
                limit=BINANCE_LIMIT
            )

            if not candles:
                break

            all_candles.extend(candles)

            # If we got fewer than limit, we've reached the end
            if len(candles) < BINANCE_LIMIT:
                break

            # Move forward to next batch
            fetch_since = candles[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)

        if not all_candles:
            raise ValueError(f"No data returned for {pair}")

        # Build DataFrame
        df = pd.DataFrame(all_candles,
                          columns=["timestamp", "Open", "High",
                                   "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        df.index.name = None
        df = df.astype(float)

        # Trim to requested days
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        df = df[df.index >= cutoff]
        df = df.sort_index()

        # Remove duplicate timestamps
        df = df[~df.index.duplicated(keep="last")]

        print(f"  ✅ {symbol} {timeframe}: {len(df):,} candles "
              f"({df.index[0].date()} → {df.index[-1].date()})")

        # Cache result
        df.to_parquet(cache)
        return df

    except Exception as e:
        raise ValueError(
            f"CCXT fetch failed for {symbol} ({pair}) on {exchange_id}: {e}"
        )


def get_multiple_crypto(symbols: list,
                        timeframe: str = "1H",
                        days_back: int = 365) -> dict:
    """Fetch OHLCV for multiple symbols. Returns dict of DataFrames."""
    results = {}
    for sym in symbols:
        try:
            results[sym] = get_crypto_ohlcv(sym, timeframe, days_back)
        except Exception as e:
            print(f"  ❌ {sym}: {e}")
    return results


def list_supported_tokens() -> list:
    """Return all tokens with configured Binance pairs."""
    return list(BINANCE_PAIRS.keys())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CCXT Crypto Data Fetcher")
    p.add_argument("--symbol",   default="BTC")
    p.add_argument("--tf",       default="1H")
    p.add_argument("--days",     type=int, default=365)
    p.add_argument("--exchange", default="binance")
    p.add_argument("--list",     action="store_true")
    args = p.parse_args()

    if args.list:
        tokens = list_supported_tokens()
        print(f"Supported tokens ({len(tokens)}):")
        for t in tokens:
            print(f"  {t}: {BINANCE_PAIRS[t]}")
    else:
        df = get_crypto_ohlcv(args.symbol, args.tf,
                              args.days, args.exchange)
        print(f"\nSample (last 5 rows):")
        print(df.tail())
