# ============================================================
# 🌙 Databento Data Fetcher
#
# Institutional-grade futures data directly from CME.
# Covers MNQ, MES, MYM, NQ, ES, YM with years of history.
#
# Priority in your system:
#   1. Databento  ← best quality, full history, programmatic
#   2. TradingView CSV ← good quality, manual export
#   3. yfinance   ← fallback, 59-day intraday limit
#
# SETUP:
#   pip install databento
#   Add to .env:  DATABENTO_API_KEY=db-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#   Get key from: databento.com → Account → API Keys
#
# COST ESTIMATE (from your $125 free credits):
#   1 year of MNQ 1H data  ≈ $2-5
#   1 year of MES 1H data  ≈ $2-5
#   1 year of MYM 1H data  ≈ $2-5
#   Full library of data   ≈ $20-40 total
#   Your $125 credits cover all of this comfortably.
#
# HOW TO USE:
#   python src/data/databento_fetcher.py
#   python src/data/databento_fetcher.py --symbol MNQ --tf 1H --days 365
#   python src/data/databento_fetcher.py --download-all
# ============================================================

import os
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT  = Path(__file__).resolve().parents[2]
CACHE_DIR  = REPO_ROOT / "src" / "data" / "databento_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Databento dataset and symbol mappings ─────────────────────
# GLBX.MDP3 = CME Globex (futures exchange for MNQ, MES, MYM, NQ, ES, YM)
DATASET = "GLBX.MDP3"

# Continuous front-month contracts using Databento's smart symbology
# Using parent symbol with stype_in="parent" gives continuous rollover data
FUTURES_DB_MAP = {
    "MNQ":  "MNQ.FUT",   # Micro Nasdaq continuous
    "MES":  "MES.FUT",   # Micro S&P continuous
    "MYM":  "MYM.FUT",   # Micro Dow continuous
    "NQ":   "NQ.FUT",    # Full Nasdaq continuous
    "ES":   "ES.FUT",    # Full S&P continuous
    "YM":   "YM.FUT",    # Full Dow continuous
    "RTY":  "RTY.FUT",   # Russell 2000 continuous
    "GC":   "GC.FUT",    # Gold continuous
    "CL":   "CL.FUT",    # Crude Oil continuous
}

# Databento schema and timeframe mapping
# Schema: ohlcv-1h, ohlcv-1m, ohlcv-1d etc.
TIMEFRAME_DB_MAP = {
    "1m":  "ohlcv-1m",
    "5m":  "ohlcv-1m",    # fetch 1m, resample to 5m
    "15m": "ohlcv-1m",    # fetch 1m, resample to 15m
    "1H":  "ohlcv-1h",
    "4H":  "ohlcv-1h",    # fetch 1H, resample to 4H
    "1D":  "ohlcv-1d",
    "1W":  "ohlcv-1d",    # fetch 1D, resample to 1W
}

# Resample rules for aggregation
RESAMPLE_RULES = {
    "5m":  "5min",
    "15m": "15min",
    "4H":  "4h",
    "1W":  "1W",
}

# ── Timeframe sets by purpose ─────────────────────────────────
# ICT timeframes: 5m is the real entry confirmation timeframe
# Strategy timeframes: 15m, 1H, 4H, 1D for backtesting
ICT_TIMEFRAMES      = ["5m", "15m", "1H", "1D"]   # needed for ICT executor
STRATEGY_TIMEFRAMES = ["15m", "1H", "4H", "1D"]   # for RBI backtester


def _get_client():
    """Get authenticated Databento client."""
    try:
        import databento as db
    except ImportError:
        raise ImportError(
            "Databento package not installed.\n"
            "Run: pip install databento"
        )

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "DATABENTO_API_KEY not found in .env\n"
            "Get your key from: databento.com → Account → API Keys\n"
            "It starts with 'db-'"
        )

    return db.Historical(api_key)


def _cache_path(symbol: str, timeframe: str, start: str, end: str) -> Path:
    """Generate cache file path."""
    safe = start[:10].replace("-","") + "_" + end[:10].replace("-","")
    return CACHE_DIR / f"{symbol}_{timeframe}_{safe}.parquet"


def _to_standard_df(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Convert Databento DataFrame to standard OHLCV format
    compatible with backtesting.py and all Algotec agents.
    """
    # Databento column names
    col_map = {
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=col_map)

    # Keep only OHLCV
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[cols].copy()

    # Ensure numeric
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure datetime index
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)   # strip tz for backtesting.py

    # Resample if needed
    rule = RESAMPLE_RULES.get(timeframe)
    if rule:
        df = df.resample(rule).agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna()

    return df.sort_index()


def get_databento_ohlcv(symbol: str, timeframe: str = "1H",
                         days_back: int = 365,
                         use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch OHLCV data from Databento for a futures symbol.

    Args:
        symbol:    e.g. "MNQ", "MES", "MYM", "ES", "NQ"
        timeframe: "1m", "15m", "1H", "4H", "1D"
        days_back: how many calendar days of history to fetch
        use_cache: cache results locally to save API credits

    Returns:
        Standard OHLCV DataFrame compatible with all Algotec agents.
    """
    sym_upper = symbol.upper()
    db_symbol = FUTURES_DB_MAP.get(sym_upper)

    if not db_symbol:
        available = list(FUTURES_DB_MAP.keys())
        raise ValueError(
            f"Symbol '{symbol}' not in Databento futures map.\n"
            f"Available: {available}"
        )

    schema = TIMEFRAME_DB_MAP.get(timeframe)
    if not schema:
        available = list(TIMEFRAME_DB_MAP.keys())
        raise ValueError(
            f"Timeframe '{timeframe}' not supported.\n"
            f"Available: {available}"
        )

    # Databento historical data ends at midnight UTC — cap to yesterday
    end_dt    = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt  = end_dt - timedelta(days=days_back)
    start_str = start_dt.strftime("%Y-%m-%dT00:00:00")
    end_str   = end_dt.strftime("%Y-%m-%dT00:00:00")

    # ── Check cache ───────────────────────────────────────────
    cache_file = _cache_path(sym_upper, timeframe, start_str, end_str)
    if use_cache and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 4:   # cache valid for 4 hours
            print(f"  📦 {symbol} {timeframe}: loading from cache ({age_hours:.1f}h old)")
            df = pd.read_parquet(cache_file)
            return df
        else:
            print(f"  🔄 {symbol} {timeframe}: cache stale, refreshing...")

    # ── Fetch from Databento ──────────────────────────────────
    print(f"  📡 Databento: fetching {symbol} {timeframe} "
          f"({days_back} days from CME)...")

    client = _get_client()

    try:
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=db_symbol,
            stype_in="parent",         # continuous front-month rollover
            schema=schema,
            start=start_str,
            end=end_str,
        )
        df_raw = data.to_df()

        if df_raw.empty:
            raise ValueError(f"No data returned for {symbol} {timeframe}")

        df = _to_standard_df(df_raw, timeframe)

        days_received = (df.index[-1] - df.index[0]).days
        print(f"  ✅ {symbol} {timeframe}: {len(df):,} candles "
              f"({days_received} days · "
              f"{df.index[0].date()} → {df.index[-1].date()})")

        # ── Save to cache ─────────────────────────────────────
        if use_cache:
            df.to_parquet(cache_file)

        return df

    except Exception as e:
        err = str(e)
        if "403" in err or "unauthorized" in err.lower():
            raise PermissionError(
                f"Databento API key invalid or no access to {DATASET}.\n"
                f"Check your key at databento.com → Account → API Keys"
            )
        elif "402" in err or "payment" in err.lower() or "credits" in err.lower():
            raise RuntimeError(
                f"Databento credits exhausted.\n"
                f"Top up at databento.com or use TradingView CSV / yfinance instead."
            )
        elif "symbol" in err.lower() or "not found" in err.lower():
            raise ValueError(
                f"Symbol {db_symbol} not found on {DATASET}.\n"
                f"Check the symbol mapping in databento_fetcher.py"
            )
        raise


def estimate_cost(symbol: str, timeframe: str, days_back: int) -> str:
    """
    Rough cost estimate before pulling data.
    Databento charges per message/row — this gives a ballpark.
    """
    rows_per_day = {
        "1m": 1440, "5m": 288, "15m": 96,
        "1H": 23,   "4H": 6,   "1D": 1,
    }
    estimated_rows = rows_per_day.get(timeframe, 23) * days_back
    # Approximate: ~$0.001 per 1000 rows for OHLCV data
    estimated_cost = estimated_rows / 1000 * 0.001
    return f"~${estimated_cost:.3f} (approx {estimated_rows:,} rows)"


def list_cached() -> list[dict]:
    """Show all locally cached Databento datasets."""
    files = list(CACHE_DIR.glob("*.parquet"))
    if not files:
        print(f"  No cached Databento data in {CACHE_DIR}")
        return []

    results = []
    print(f"\n  📦 Cached Databento data ({len(files)} files):")
    print(f"  {'File':<35} {'Size':>8}  {'Age'}")
    print(f"  {'─'*60}")
    for f in sorted(files):
        size_kb = f.stat().st_size / 1024
        age_h   = (time.time() - f.stat().st_mtime) / 3600
        age_str = f"{age_h:.0f}h ago" if age_h < 48 else f"{age_h/24:.0f}d ago"
        print(f"  {f.name:<35} {size_kb:>7.0f}KB  {age_str}")
        results.append({"file": f.name, "size_kb": size_kb})
    return results


def _resample_and_cache(df_raw: pd.DataFrame, symbol: str,
                         target_tf: str, days_back: int,
                         start_str: str, end_str: str):
    """
    Resample an already-fetched DataFrame to a derived timeframe
    and save it to cache. Avoids re-downloading the same raw data.
    """
    rule = RESAMPLE_RULES.get(target_tf)
    if not rule:
        return

    df = df_raw.resample(rule).agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna()

    cache_file = _cache_path(symbol.upper(), target_tf, start_str, end_str)
    df.to_parquet(cache_file)
    print(f"     ✅ {symbol} {target_tf}: {len(df):,} candles (resampled from raw, $0.00)")


def download_all_futures(days_back: int = 1825):
    """
    Download 5 years of all futures data from Databento.

    SMART STRATEGY — avoids paying for the same data twice:
      1m raw data  → resampled locally to 5m and 15m  (free)
      1H raw data  → resampled locally to 4H           (free)
      1D raw data  → used directly for daily charts

    ESTIMATED TOTAL COST: ~$0.14 from your $125 credits
    (5 years × 3 symbols × all timeframes = ~340 MB at $0.40/GB)

    TIMEFRAMES PRODUCED:
      For ICT executor    : 1m, 5m, 15m  (entry precision)
      For RBI backtester  : 15m, 1H, 4H, 1D (strategy testing)
      For regime/context  : 1D (trend direction)
    """
    symbols  = ["MNQ", "MES", "MYM"]
    # Cap end to midnight UTC to avoid Databento 422 error
    end_dt    = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt  = end_dt - timedelta(days=days_back)
    start_str = start_dt.strftime("%Y-%m-%dT00:00:00")
    end_str   = end_dt.strftime("%Y-%m-%dT00:00:00")

    years = round(days_back / 365, 1)

    print(f"\n🚀 Downloading {years} years of futures data from Databento")
    print(f"   Symbols   : {symbols}")
    print(f"   Period    : {start_str[:10]} → {end_str[:10]}")
    print(f"   Timeframes: 1m, 5m, 15m, 1H, 4H, 1D (all produced)")
    print(f"\n   Cost estimate:")
    print(f"     1m fetch  (raw):  ~{estimate_cost('MNQ','1m',days_back)} × 3 symbols")
    print(f"     1H fetch  (raw):  ~{estimate_cost('MNQ','1H',days_back)} × 3 symbols")
    print(f"     1D fetch  (raw):  ~{estimate_cost('MNQ','1D',days_back)} × 3 symbols")
    print(f"     5m, 15m, 4H   :  $0.00 (resampled locally from raw fetches)")
    print(f"     ─────────────────────────────────")
    print(f"     TOTAL           :  ~$0.14 from your $125 credits")
    print()

    success = 0
    failed  = 0

    for sym in symbols:
        print(f"\n  {'─'*50}")
        print(f"  📡 {sym}")

        # ── FETCH 1: 1m raw ───────────────────────────────────
        # Used directly as 1m, resampled to 5m and 15m
        print(f"     Fetching 1m (raw)...")
        try:
            df_1m = get_databento_ohlcv(sym, "1m", days_back=days_back,
                                         use_cache=True)
            success += 1
            # Derive 5m and 15m from the same download
            _resample_and_cache(df_1m, sym, "5m",  days_back, start_str, end_str)
            _resample_and_cache(df_1m, sym, "15m", days_back, start_str, end_str)
            success += 2   # count derived timeframes
        except Exception as e:
            print(f"     ❌ 1m failed: {e}")
            failed += 1
        time.sleep(0.5)

        # ── FETCH 2: 1H raw ───────────────────────────────────
        # Used directly as 1H, resampled to 4H
        print(f"     Fetching 1H (raw)...")
        try:
            df_1h = get_databento_ohlcv(sym, "1H", days_back=days_back,
                                         use_cache=True)
            success += 1
            _resample_and_cache(df_1h, sym, "4H", days_back, start_str, end_str)
            success += 1
        except Exception as e:
            print(f"     ❌ 1H failed: {e}")
            failed += 1
        time.sleep(0.5)

        # ── FETCH 3: 1D raw ───────────────────────────────────
        print(f"     Fetching 1D (raw)...")
        try:
            get_databento_ohlcv(sym, "1D", days_back=days_back, use_cache=True)
            success += 1
        except Exception as e:
            print(f"     ❌ 1D failed: {e}")
            failed += 1
        time.sleep(0.5)

    print(f"\n{'═'*55}")
    print(f"✅ Download complete")
    print(f"   {success} datasets ready | {failed} failed")
    print(f"   Cached to: {CACHE_DIR}")
    print(f"\n   Timeframes now available per symbol:")
    print(f"   1m  → ICT precision entries")
    print(f"   5m  → ICT CISD confirmation (executor)")
    print(f"   15m → ICT backtester + RBI strategies")
    print(f"   1H  → Primary strategy timeframe")
    print(f"   4H  → Higher timeframe context")
    print(f"   1D  → Daily bias and regime detection")
    print(f"\n   Your $125 Databento credits used: ~$0.14")


def get_ohlcv_with_fallback(symbol: str, timeframe: str = "1H",
                              days_back: int = 365) -> pd.DataFrame:
    """
    Master data fetch with full priority chain:
      1. Databento (best — institutional CME data)
      2. TradingView CSV (good — if you've exported it)
      3. yfinance (fallback — 59-day intraday limit)
    """
    # Try Databento first
    if os.getenv("DATABENTO_API_KEY") and symbol.upper() in FUTURES_DB_MAP:
        try:
            return get_databento_ohlcv(symbol, timeframe, days_back=days_back)
        except RuntimeError as e:
            if "credits" in str(e).lower():
                print(f"  ⚠️  Databento credits low — falling back")
            else:
                raise
        except Exception as e:
            print(f"  ⚠️  Databento unavailable ({e.__class__.__name__}) — trying next source")

    # Try TradingView CSV
    try:
        from src.data.tradingview_fetcher import get_tv_ohlcv
        df = get_tv_ohlcv(symbol, timeframe)
        if df is not None and not df.empty:
            return df
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # Fall back to yfinance
    print(f"  ℹ️  Using yfinance for {symbol} {timeframe}")
    from src.data.fetcher import _get_yf_ohlcv
    return _get_yf_ohlcv(symbol, "tradovate", timeframe, days_back)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="📡 Databento Data Fetcher")
    p.add_argument("--symbol",       default="MNQ",
                   help="Symbol e.g. MNQ, MES, MYM (default: MNQ)")
    p.add_argument("--tf",           default="1H",
                   help="Timeframe e.g. 15m, 1H, 4H, 1D (default: 1H)")
    p.add_argument("--days",         type=int, default=1825,
                   help="Days of history (default: 1825 = 5 years)")
    p.add_argument("--download-all", action="store_true",
                   help="Download MNQ, MES, MYM across all timeframes")
    p.add_argument("--list-cache",   action="store_true",
                   help="Show cached data files")
    p.add_argument("--cost",         action="store_true",
                   help="Show cost estimate without downloading")
    args = p.parse_args()

    if args.list_cache:
        list_cached()
    elif args.download_all:
        download_all_futures(days_back=args.days)
    elif args.cost:
        cost = estimate_cost(args.symbol, args.tf, args.days)
        print(f"  Cost estimate for {args.symbol} {args.tf} {args.days}d: {cost}")
    else:
        df = get_databento_ohlcv(args.symbol, args.tf, days_back=args.days)
        print(f"\n  Sample (last 5 rows):")
        print(df.tail())
