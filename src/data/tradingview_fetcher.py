# ============================================================
# 🌙 TradingView Data Fetcher
#
# Imports historical OHLCV data from TradingView CSV exports
# for backtesting. TradingView Pro+ has much more history
# than yfinance — years of intraday data vs 59 days.
#
# HOW TO EXPORT FROM TRADINGVIEW:
#   1. Open TradingView → chart your symbol (e.g. MNQ1!)
#   2. Set your timeframe (1H, 15m etc.)
#   3. Click Export → Download CSV
#   4. Save to: src/data/tradingview/MNQ_1H.csv
#
# HOW TO USE IN BACKTEST:
#   from src.data.tradingview_fetcher import get_tv_ohlcv
#   df = get_tv_ohlcv("MNQ", "1H")
#
# FILE NAMING CONVENTION:
#   src/data/tradingview/SYMBOL_TIMEFRAME.csv
#   Examples:
#     MNQ_1H.csv     MNQ_15m.csv    MNQ_4H.csv
#     MES_1H.csv     MYM_1H.csv
#     BTC_1H.csv     ETH_1H.csv     SOL_1H.csv
# ============================================================

import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── paths ─────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
TV_DIR     = REPO_ROOT / "src" / "data" / "tradingview"
TV_DIR.mkdir(parents=True, exist_ok=True)

# ── TradingView column name variations ───────────────────────
# TV exports use slightly different column names depending on version
TV_COLUMN_MAPS = [
    # Format 1 — most common
    {"time": "time", "open": "open", "high": "high",
     "low": "low",  "close": "close", "volume": "volume"},
    # Format 2 — with capitals
    {"time": "Time", "open": "Open", "high": "High",
     "low": "Low",  "close": "Close", "volume": "Volume"},
    # Format 3 — with date
    {"time": "Date", "open": "Open", "high": "High",
     "low": "Low",  "close": "Close", "volume": "Volume"},
    # Format 4 — timestamp
    {"time": "Timestamp", "open": "Open", "high": "High",
     "low": "Low",  "close": "Close", "volume": "Volume"},
]


def _find_csv(symbol: str, timeframe: str) -> Path | None:
    """Find the CSV file for a symbol/timeframe combination."""
    candidates = [
        TV_DIR / f"{symbol}_{timeframe}.csv",
        TV_DIR / f"{symbol.upper()}_{timeframe}.csv",
        TV_DIR / f"{symbol}_{timeframe.upper()}.csv",
        TV_DIR / f"{symbol.lower()}_{timeframe.lower()}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _parse_tv_csv(path: Path) -> pd.DataFrame:
    """
    Parse a TradingView CSV export into a standard OHLCV DataFrame.
    Handles all TradingView export formats automatically.
    """
    # Try reading with different separators
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) >= 5:
                break
        except Exception:
            continue

    # Find matching column map
    cols = [c.lower().strip() for c in df.columns]
    mapped = None

    for col_map in TV_COLUMN_MAPS:
        time_col  = col_map["time"].lower()
        open_col  = col_map["open"].lower()
        close_col = col_map["close"].lower()
        if time_col in cols and open_col in cols and close_col in cols:
            mapped = {v.lower(): v for v in col_map.values()}
            break

    if mapped is None:
        # Try to auto-detect
        actual_cols = list(df.columns)
        print(f"  ⚠️  Could not match columns. Found: {actual_cols}")
        print(f"      Expected: time, open, high, low, close, volume")
        raise ValueError(f"Unrecognised TradingView CSV format in {path.name}")

    # Rename to standard names
    rename_map = {}
    for standard, tv_name in mapped.items():
        # Find actual column name (case insensitive)
        for col in df.columns:
            if col.lower().strip() == tv_name.lower():
                rename_map[col] = standard
                break

    df = df.rename(columns=rename_map)

    # Parse timestamp
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M",
    ]:
        try:
            df["time"] = pd.to_datetime(df["time"], format=fmt)
            break
        except Exception:
            try:
                df["time"] = pd.to_datetime(df["time"])
                break
            except Exception:
                continue

    df = df.set_index("time")
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)   # strip timezone for backtesting.py

    # Keep only OHLCV, rename to standard capitalisation
    ohlcv_map = {
        "open":  "Open",  "high":  "High",
        "low":   "Low",   "close": "Close",
        "volume":"Volume",
    }
    df = df.rename(columns=ohlcv_map)

    # Ensure numeric
    for col in ["Open","High","Low","Close","Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df = df.sort_index()

    return df


def get_tv_ohlcv(symbol: str, timeframe: str = "1H") -> pd.DataFrame:
    """
    Load TradingView CSV data for a symbol and timeframe.

    Args:
        symbol:    e.g. "MNQ", "MES", "MYM", "BTC", "ETH"
        timeframe: e.g. "1H", "15m", "4H", "1D"

    Returns:
        DataFrame with Open, High, Low, Close, Volume columns.

    Raises:
        FileNotFoundError if CSV not found — shows instructions.
    """
    path = _find_csv(symbol, timeframe)

    if path is None:
        expected = TV_DIR / f"{symbol}_{timeframe}.csv"
        raise FileNotFoundError(
            f"\n\n❌ TradingView data not found for {symbol} {timeframe}\n"
            f"\nHow to export from TradingView:\n"
            f"  1. Open TradingView and load your {symbol} chart\n"
            f"  2. Set timeframe to {timeframe}\n"
            f"  3. Click the Export button (↓) on the chart\n"
            f"  4. Click 'Export chart data'\n"
            f"  5. Save the file as: {expected}\n"
            f"\nFor futures use these TradingView symbols:\n"
            f"  MNQ → MNQ1! (front month) or MNQH2026\n"
            f"  MES → MES1!\n"
            f"  MYM → MYM1!\n"
        )

    print(f"  📊 Loading TradingView data: {path.name}")
    df = _parse_tv_csv(path)

    days = (df.index[-1] - df.index[0]).days
    print(f"  ✅ {symbol} {timeframe}: {len(df):,} candles "
          f"({days} days — {df.index[0].date()} to {df.index[-1].date()})")

    return df


def list_available_tv_data() -> list[dict]:
    """List all TradingView CSV files available for backtesting."""
    files = list(TV_DIR.glob("*.csv"))
    if not files:
        print(f"\n  📂 No TradingView data files found in {TV_DIR}")
        print(f"     Export data from TradingView and save to that folder.")
        return []

    available = []
    for f in sorted(files):
        try:
            df   = _parse_tv_csv(f)
            days = (df.index[-1] - df.index[0]).days
            name = f.stem
            parts = name.split("_", 1)
            available.append({
                "file":      f.name,
                "symbol":    parts[0] if len(parts) > 0 else name,
                "timeframe": parts[1] if len(parts) > 1 else "?",
                "candles":   len(df),
                "days":      days,
                "from":      str(df.index[0].date()),
                "to":        str(df.index[-1].date()),
            })
        except Exception as e:
            available.append({"file": f.name, "error": str(e)})

    print(f"\n  📊 Available TradingView datasets ({len(available)} files):")
    print(f"  {'File':<25} {'Candles':>8} {'Days':>6}  {'Period'}")
    print(f"  {'─'*65}")
    for a in available:
        if "error" not in a:
            print(f"  {a['file']:<25} {a['candles']:>8,} {a['days']:>6}  "
                  f"{a['from']} → {a['to']}")
        else:
            print(f"  {a['file']:<25} ❌ {a['error']}")

    return available


def get_ohlcv_with_fallback(symbol: str, timeframe: str = "1H",
                             days_back: int = 59) -> pd.DataFrame:
    """
    Try TradingView data first, fall back to yfinance.
    This is the recommended function to use in all agents.
    """
    try:
        df = get_tv_ohlcv(symbol, timeframe)
        return df
    except FileNotFoundError:
        print(f"  ℹ️  No TradingView data for {symbol} {timeframe} — using yfinance")
        from src.data.fetcher import get_ohlcv
        return get_ohlcv(symbol, timeframe=timeframe, days_back=days_back)
    except Exception as e:
        print(f"  ⚠️  TradingView load failed ({e}) — using yfinance")
        from src.data.fetcher import get_ohlcv
        return get_ohlcv(symbol, timeframe=timeframe, days_back=days_back)


if __name__ == "__main__":
    print("🌙 TradingView Data Manager\n")
    available = list_available_tv_data()
    if not available:
        print(f"\n📁 Save your exports to: {TV_DIR}")
        print(f"   Naming: SYMBOL_TIMEFRAME.csv  e.g. MNQ_1H.csv")
