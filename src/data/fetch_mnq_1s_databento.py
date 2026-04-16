from __future__ import annotations

"""
Fetch real Databento 1-second MNQ bars and also save valid 5-second and 30-second
resamples to your local cache directory.

Outputs:
- src/data/databento_cache/MNQ_1s.parquet
- src/data/databento_cache/MNQ_5s.parquet
- src/data/databento_cache/MNQ_30s.parquet

Requirements:
- pip install databento pandas pyarrow
- DATABENTO_API_KEY in your environment or .env

Example:
    export DATABENTO_API_KEY="your_key"
    python src/data/fetch_mnq_1s_databento.py --start 2025-01-01 --end 2025-03-01
"""

import argparse
import os
from pathlib import Path

import pandas as pd

try:
    import databento as db
except ImportError as e:
    raise SystemExit(
        "databento is not installed. Run: pip install databento pandas pyarrow"
    ) from e


DEFAULT_DATASET = "GLBX.MDP3"
DEFAULT_SCHEMA = "ohlcv-1s"
DEFAULT_SYMBOL = "MNQ.v.0"
DEFAULT_OUTDIR = Path("src/data/databento_cache")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start date/time, e.g. 2025-01-01")
    p.add_argument("--end", required=True, help="End date/time, e.g. 2025-03-01")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Databento symbol, default MNQ.v.0")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Databento dataset, default GLBX.MDP3")
    p.add_argument("--schema", default=DEFAULT_SCHEMA, help="Databento schema, default ohlcv-1s")
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory")
    p.add_argument(
        "--basename",
        default="MNQ",
        help="Base filename prefix, e.g. MNQ -> MNQ_1s.parquet",
    )
    return p.parse_args()


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Databento returned no rows.")

    # Find timestamp column
    ts_col = None
    for candidate in ("ts_event", "ts_recv", "timestamp"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise ValueError(f"Could not find timestamp column. Columns were: {list(df.columns)}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index(ts_col).sort_index()

    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "open":
            rename[c] = "Open"
        elif cl == "high":
            rename[c] = "High"
        elif cl == "low":
            rename[c] = "Low"
        elif cl == "close":
            rename[c] = "Close"
        elif cl == "volume":
            rename[c] = "Volume"

    df = df.rename(columns=rename)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns after rename: {missing}")

    return df[required].dropna().copy()


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    return df.resample(rule, label="right", closed="right").agg(agg).dropna()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise SystemExit("Missing DATABENTO_API_KEY in your environment.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_1s = outdir / f"{args.basename}_1s.parquet"
    out_5s = outdir / f"{args.basename}_5s.parquet"
    out_30s = outdir / f"{args.basename}_30s.parquet"

    print(f"Fetching Databento {args.schema} for {args.symbol}")
    print(f"Start: {args.start}")
    print(f"End:   {args.end}")

    client = db.Historical(api_key)

    data = client.timeseries.get_range(
        dataset=args.dataset,
        schema=args.schema,
        symbols=[args.symbol],
        stype_in="continuous",
        start=args.start,
        end=args.end,
    )

    raw = data.to_df().reset_index()
    df_1s = _standardize_ohlcv(raw)

    df_5s = _resample(df_1s, "5s")
    df_30s = _resample(df_1s, "30s")

    df_1s.to_parquet(out_1s)
    df_5s.to_parquet(out_5s)
    df_30s.to_parquet(out_30s)

    print(f"Saved {out_1s} rows={len(df_1s)}")
    print(f"Saved {out_5s} rows={len(df_5s)}")
    print(f"Saved {out_30s} rows={len(df_30s)}")


if __name__ == "__main__":
    main()
