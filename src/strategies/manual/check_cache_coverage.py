# save as: src/strategies/manual/check_cache_coverage.py
"""
check_cache_coverage.py

What this does
--------------
This script inspects the local Databento parquet cache files and tells you
whether they actually contain the amount of historical data you think they do.

It prints, for each symbol:
- file path
- whether the file exists
- total row count
- timestamp column used
- earliest timestamp
- latest timestamp
- total span in days
- whether the file covers at least N days

Why this is useful
------------------
Your recent "365 day" backtest only loaded a tiny recent slice instead of a full
year. This script helps confirm whether the local cache files themselves are the
problem.

How to run
----------
From repo root:

    cd /Users/Abdullahi/trading-project/trading_system
    source venv/bin/activate
    python src/strategies/manual/check_cache_coverage.py --days 365

Optional:
    python src/strategies/manual/check_cache_coverage.py --days 30
    python src/strategies/manual/check_cache_coverage.py --symbols MNQ MES MYM
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_SYMBOLS = ["MNQ", "MES", "MYM", "MGC", "MCL"]


def _find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "ts_event",
        "timestamp",
        "datetime",
        "date",
        "time",
        "ts",
        "index",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _load_timestamp_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
    ts_col = _find_timestamp_column(df)

    if ts_col is not None:
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        return ts, ts_col

    if isinstance(df.index, pd.DatetimeIndex):
        ts = pd.to_datetime(df.index, errors="coerce", utc=True)
        return pd.Series(ts, index=df.index), "__index__"

    raise ValueError("No timestamp column or DatetimeIndex found.")


def inspect_file(path: Path, required_days: int) -> dict:
    result = {
        "file": str(path),
        "exists": path.exists(),
        "rows": 0,
        "timestamp_source": None,
        "start_utc": None,
        "end_utc": None,
        "span_days": None,
        "covers_required_days": False,
        "error": None,
    }

    if not path.exists():
        return result

    try:
        df = pd.read_parquet(path)
        result["rows"] = len(df)

        if df.empty:
            result["error"] = "Parquet exists but is empty."
            return result

        ts, ts_source = _load_timestamp_series(df)
        ts = ts.dropna()

        if ts.empty:
            result["error"] = "No valid timestamps found."
            return result

        start_ts = ts.min()
        end_ts = ts.max()
        span_days = (end_ts - start_ts).total_seconds() / 86400.0

        result["timestamp_source"] = ts_source
        result["start_utc"] = start_ts.isoformat()
        result["end_utc"] = end_ts.isoformat()
        result["span_days"] = round(span_days, 2)
        result["covers_required_days"] = span_days >= required_days

        return result

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def print_report(symbol: str, info: dict, required_days: int) -> None:
    print("=" * 90)
    print(f"SYMBOL: {symbol}")
    print(f"file:                 {info['file']}")
    print(f"exists:               {info['exists']}")

    if not info["exists"]:
        print("status:               MISSING FILE")
        return

    print(f"rows:                 {info['rows']}")

    if info["error"]:
        print(f"status:               ERROR -> {info['error']}")
        return

    print(f"timestamp source:     {info['timestamp_source']}")
    print(f"start utc:            {info['start_utc']}")
    print(f"end utc:              {info['end_utc']}")
    print(f"span days:            {info['span_days']}")
    print(f"covers {required_days}d:       {info['covers_required_days']}")

    if info["covers_required_days"]:
        print("diagnosis:            OK - file appears large enough for requested lookback")
    else:
        print("diagnosis:            NOT ENOUGH HISTORY - cache is shorter than requested")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Databento parquet cache coverage.")
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Required history span in days to validate against.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=DEFAULT_SYMBOLS,
        help="Symbols to inspect. Default: MNQ MES MYM MGC MCL",
    )
    parser.add_argument(
        "--cache-dir",
        default="src/data/databento_cache",
        help="Directory containing parquet cache files.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    print("\nDatabento cache coverage check")
    print(f"cache dir:            {cache_dir.resolve()}")
    print(f"required days:        {args.days}")
    print(f"symbols:              {args.symbols}\n")

    summary = []

    for symbol in args.symbols:
        path = cache_dir / f"{symbol}_1m.parquet"
        info = inspect_file(path, args.days)
        print_report(symbol, info, args.days)
        summary.append((symbol, info))

    print("\n" + "=" * 90)
    print("SUMMARY")
    bad = []

    for symbol, info in summary:
        if (not info["exists"]) or info["error"] or (not info["covers_required_days"]):
            bad.append(symbol)

    if not bad:
        print("All checked files appear to cover the requested lookback.")
    else:
        print(f"Files needing attention: {', '.join(bad)}")

    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())