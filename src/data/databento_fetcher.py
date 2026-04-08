from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import databento as db
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


DATABENTO_DATASET = "GLBX.MDP3"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "databento_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FUTURES_DB_MAP: Dict[str, str] = {
    "MNQ": "MNQ.n.0",
    "MES": "MES.n.0",
    "MYM": "MYM.n.0",
    "NQ": "NQ.n.0",
    "ES": "ES.n.0",
    "YM": "YM.n.0",
    "MCL": "MCL.n.0",
    "CL": "CL.n.0",
    "MGC": "MGC.n.0",
    "GC": "GC.n.0",
}

SUPPORTED_DIRECT_TIMEFRAMES = {"1m", "1H", "1D"}
SUPPORTED_RESAMPLE_TIMEFRAMES = {"5m", "15m", "4H"}
SUPPORTED_TIMEFRAMES = SUPPORTED_DIRECT_TIMEFRAMES | SUPPORTED_RESAMPLE_TIMEFRAMES

DEFAULT_HISTORICAL_LAG_HOURS = 48


def _normalize_timeframe(timeframe: str) -> str:
    tf = timeframe.strip()
    allowed = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1H": "1H",
        "4H": "4H",
        "1D": "1D",
    }
    if tf not in allowed:
        raise ValueError(
            f"Unsupported Databento timeframe '{timeframe}'. "
            f"Supported: {sorted(allowed.keys())}"
        )
    return allowed[tf]


def _schema_for_timeframe(timeframe: str) -> str:
    tf = _normalize_timeframe(timeframe)
    if tf in {"1m", "5m", "15m"}:
        return "ohlcv-1m"
    if tf in {"1H", "4H"}:
        return "ohlcv-1h"
    if tf == "1D":
        return "ohlcv-1d"
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _resample_rule(timeframe: str) -> Optional[str]:
    tf = _normalize_timeframe(timeframe)
    if tf == "5m":
        return "5min"
    if tf == "15m":
        return "15min"
    if tf == "4H":
        return "4h"
    return None


def _ensure_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    out = df.rename(columns=rename_map)

    if "ts_event" in out.columns:
        out = out.set_index("ts_event")

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing expected OHLCV columns from Databento: {missing}")

    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    out = out[required].astype(float).dropna()
    out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = _resample_rule(timeframe)
    if rule is None:
        return df

    out = (
        df.resample(rule)
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
    return out


def _cache_base_timeframe(timeframe: str) -> str:
    tf = _normalize_timeframe(timeframe)
    if tf in {"1m", "5m", "15m"}:
        return "1m"
    if tf in {"1H", "4H"}:
        return "1H"
    if tf == "1D":
        return "1D"
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _cache_path(symbol: str, base_timeframe: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_{base_timeframe}.parquet"


def _read_cache(symbol: str, timeframe: str, days_back: int) -> Optional[pd.DataFrame]:
    base_tf = _cache_base_timeframe(timeframe)
    path = _cache_path(symbol, base_tf)
    if not path.exists():
        print(f"📭 No local cache found: {path}")
        return None

    df = pd.read_parquet(path)
    if df.empty:
        return None

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    cutoff = datetime.utcnow() - timedelta(days=days_back)
    df = df[df.index >= cutoff]

    if df.empty:
        return None

    print(f"📦 Using local cache: {path}")
    return _resample_ohlcv(df, timeframe)


def _write_cache(symbol: str, base_timeframe: str, df: pd.DataFrame) -> Path:
    path = _cache_path(symbol, base_timeframe)
    df = df.sort_index()
    df.to_parquet(path)
    return path


def _safe_end_time(lag_hours: int = DEFAULT_HISTORICAL_LAG_HOURS) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=lag_hours)


def _extract_retry_end_from_error(msg: str) -> Optional[datetime]:
    m = re.search(r"before\s+([0-9T:\.\-]+Z)", msg)
    if not m:
        return None
    ts = m.group(1)
    try:
        ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _request_range(
    client: db.Historical,
    continuous_symbol: str,
    schema: str,
    start: datetime,
    end: datetime,
):
    return client.timeseries.get_range(
        dataset=DATABENTO_DATASET,
        symbols=[continuous_symbol],
        stype_in="continuous",
        schema=schema,
        start=start.isoformat(),
        end=end.isoformat(),
    )


def get_databento_ohlcv(
    symbol: str,
    timeframe: str = "1H",
    days_back: int = 365,
    force_refresh: bool = False,
) -> pd.DataFrame:
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY is not set")

    root = symbol.upper().strip()
    if root not in FUTURES_DB_MAP:
        raise ValueError(f"Unsupported Databento futures root: {root}")

    tf = _normalize_timeframe(timeframe)
    base_tf = _cache_base_timeframe(tf)

    if not force_refresh:
        cached = _read_cache(root, tf, days_back)
        if cached is not None and not cached.empty:
            return cached

    schema = _schema_for_timeframe(tf)
    continuous_symbol = FUTURES_DB_MAP[root]

    end = _safe_end_time()
    start = end - timedelta(days=int(days_back))

    client = db.Historical(api_key)

    print(
        f"📡 Fetching from Databento API: "
        f"symbol={continuous_symbol} schema={schema} "
        f"start={start.isoformat()} end={end.isoformat()}"
    )

    try:
        data = _request_range(client, continuous_symbol, schema, start, end)
    except Exception as e:
        msg = str(e)
        retry_end = _extract_retry_end_from_error(msg)
        if retry_end is None:
            raise

        retry_end = retry_end - timedelta(minutes=5)
        retry_start = retry_end - timedelta(days=int(days_back))

        print(
            f"↩️ Retrying Databento API with earlier end: "
            f"symbol={continuous_symbol} schema={schema} "
            f"start={retry_start.isoformat()} end={retry_end.isoformat()}"
        )

        data = _request_range(client, continuous_symbol, schema, retry_start, retry_end)

    df = data.to_df()
    if df is None or df.empty:
        raise ValueError(
            f"No Databento data returned for {continuous_symbol} "
            f"({timeframe}, {days_back} days)"
        )

    df = _ensure_ohlcv_columns(df)
    _write_cache(root, base_tf, df)
    print(f"💾 Wrote local cache: {_cache_path(root, base_tf)}")

    df = _resample_ohlcv(df, tf)

    if df.empty:
        raise ValueError(
            f"Databento returned data but final OHLCV is empty for "
            f"{continuous_symbol} ({timeframe})"
        )

    return df


def download_and_cache_symbol(
    symbol: str,
    years_back: int = 5,
    base_timeframes: Optional[list[str]] = None,
    force_refresh: bool = True,
) -> Dict[str, str]:
    if base_timeframes is None:
        base_timeframes = ["1m", "1H", "1D"]

    results: Dict[str, str] = {}
    days_back = int(years_back * 365)

    for tf in base_timeframes:
        df = get_databento_ohlcv(
            symbol=symbol,
            timeframe=tf,
            days_back=days_back,
            force_refresh=force_refresh,
        )
        path = _cache_path(symbol, tf)
        results[tf] = str(path)
        print(f"✅ Cached {symbol} {tf}: {len(df)} rows -> {path}")

    return results


def download_and_cache_symbols(
    symbols: list[str],
    years_back: int = 5,
    base_timeframes: Optional[list[str]] = None,
    force_refresh: bool = True,
) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for symbol in symbols:
        out[symbol.upper()] = download_and_cache_symbol(
            symbol=symbol,
            years_back=years_back,
            base_timeframes=base_timeframes,
            force_refresh=force_refresh,
        )
    return out


if __name__ == "__main__":
    download_and_cache_symbols(["MCL", "MGC"], years_back=5, force_refresh=True)