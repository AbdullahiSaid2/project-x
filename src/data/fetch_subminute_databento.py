from __future__ import annotations

"""
Fetch real Databento 1-second OHLCV for one or more futures continuous symbols,
and save valid 5-second and 30-second resamples to local parquet cache files.

Outputs for each contract basename, e.g. MES:
- src/data/databento_cache/MES_1s.parquet
- src/data/databento_cache/MES_5s.parquet
- src/data/databento_cache/MES_30s.parquet

Examples:
  export DATABENTO_API_KEY="your_key"

  # Single symbol
  python src/data/fetch_subminute_databento.py --start 2025-01-01 --end 2026-04-14 --symbol MES.v.0 --basename MES

  # Multiple symbols in one run
  python src/data/fetch_subminute_databento.py --start 2025-01-01 --end 2026-04-14 \
      --contracts MES,MYM,MGC,MCL,MNQ

Notes:
- Continuous futures symbols like ES.v.0 / MNQ.v.0 require stype_in="continuous".
- 5s and 30s are resampled from true 1s data, so they are valid.
"""

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import databento as db
except ImportError as e:
    raise SystemExit(
        "databento is not installed. Run: pip install databento pandas pyarrow"
    ) from e


DEFAULT_DATASET = "GLBX.MDP3"
DEFAULT_SCHEMA = "ohlcv-1s"
DEFAULT_OUTDIR = Path("src/data/databento_cache")

# Contract basename -> Databento continuous symbol
CONTRACT_MAP = {
    "MES": "MES.v.0",
    "MYM": "MYM.v.0",
    "MGC": "MGC.v.0",
    "MCL": "MCL.v.0",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start date/time, e.g. 2025-01-01")
    p.add_argument("--end", required=True, help="End date/time, e.g. 2026-04-14")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Databento dataset")
    p.add_argument("--schema", default=DEFAULT_SCHEMA, help="Databento schema")
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory")
    p.add_argument(
        "--symbol",
        default=None,
        help="Single Databento symbol, e.g. MNQ.v.0. Use with --basename.",
    )
    p.add_argument(
        "--basename",
        default=None,
        help="Single output prefix, e.g. MNQ. Use with --symbol.",
    )
    p.add_argument(
        "--contracts",
        default=None,
        help="Comma-separated basenames from: MNQ,MES,MYM,MGC,MCL",
    )
    return p.parse_args()


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Databento returned no rows.")

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


def _single_targets(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.symbol or args.basename:
        if not (args.symbol and args.basename):
            raise SystemExit("If using --symbol, also provide --basename.")
        return [(args.basename.upper().strip(), args.symbol.strip())]

    if args.contracts:
        basenames = [c.strip().upper() for c in args.contracts.split(",") if c.strip()]
    else:
        basenames = ["MNQ"]

    unknown = [b for b in basenames if b not in CONTRACT_MAP]
    if unknown:
        raise SystemExit(f"Unknown contracts: {unknown}. Allowed: {sorted(CONTRACT_MAP)}")

    return [(b, CONTRACT_MAP[b]) for b in basenames]


def fetch_one(
    client: db.Historical,
    *,
    dataset: str,
    schema: str,
    symbol: str,
    basename: str,
    start: str,
    end: str,
    outdir: Path,
) -> None:
    out_1s = outdir / f"{basename}_1s.parquet"
    out_5s = outdir / f"{basename}_5s.parquet"
    out_30s = outdir / f"{basename}_30s.parquet"

    print(f"\nFetching Databento {schema} for {symbol} -> {basename}")
    print(f"Start: {start}")
    print(f"End:   {end}")

    data = client.timeseries.get_range(
        dataset=dataset,
        schema=schema,
        symbols=[symbol],
        stype_in="continuous",
        start=start,
        end=end,
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


def main() -> None:
    args = parse_args()

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise SystemExit("Missing DATABENTO_API_KEY in your environment.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    targets = _single_targets(args)
    client = db.Historical(api_key)

    for basename, symbol in targets:
        fetch_one(
            client,
            dataset=args.dataset,
            schema=args.schema,
            symbol=symbol,
            basename=basename,
            start=args.start,
            end=args.end,
            outdir=outdir,
        )


if __name__ == "__main__":
    main()
