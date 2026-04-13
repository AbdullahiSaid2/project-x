# src/data/fetch_nq_30s_databento.py

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    import databento as db
except ImportError as exc:
    raise SystemExit(
        "databento is not installed. Run: pip install databento python-dotenv"
    ) from exc


API_KEY_ENV = "DATABENTO_API_KEY"
OUTPUT_PATH = Path("src/data/databento_cache/NQ_30s.parquet")

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1s"

# Use Databento continuous symbology for the rolling front-month NQ contract
SYMBOL = "NQ.v.0"
STYPE_IN = "continuous"

START = "2025-04-08"
END = "2026-04-09"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        low = str(col).lower()
        if low == "open":
            rename_map[col] = "Open"
        elif low == "high":
            rename_map[col] = "High"
        elif low == "low":
            rename_map[col] = "Low"
        elif low == "close":
            rename_map[col] = "Close"
        elif low == "volume":
            rename_map[col] = "Volume"

    df = df.rename(columns=rename_map)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    return df[needed].copy()


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        possible_ts_cols = [
            c for c in df.columns
            if str(c).lower() in {"ts_event", "timestamp", "time", "datetime"}
        ]
        if not possible_ts_cols:
            raise ValueError("No DatetimeIndex and no obvious timestamp column found.")
        ts_col = possible_ts_cols[0]
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.set_index(ts_col)

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    df.index = idx
    return df.sort_index()


def resample_to_30s(df_1s: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_1s.resample("30s").size().index)
    out["Open"] = df_1s["Open"].resample("30s").first()
    out["High"] = df_1s["High"].resample("30s").max()
    out["Low"] = df_1s["Low"].resample("30s").min()
    out["Close"] = df_1s["Close"].resample("30s").last()
    out["Volume"] = df_1s["Volume"].resample("30s").sum()
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out


def main() -> None:
    # Load variables from .env into the current Python process
    load_dotenv()

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise SystemExit(
            f"Missing {API_KEY_ENV}. Put it in .env or export it in your shell."
        )

    client = db.Historical(api_key)

    print("Fetching 1-second OHLCV from Databento...")
    print(f"Dataset: {DATASET}")
    print(f"Schema: {SCHEMA}")
    print(f"Symbol: {SYMBOL} (stype_in={STYPE_IN})")
    print(f"Start: {START} | End: {END}")

    data = client.timeseries.get_range(
        dataset=DATASET,
        schema=SCHEMA,
        symbols=SYMBOL,
        stype_in=STYPE_IN,
        start=START,
        end=END,
    )

    df = data.to_df()
    print(f"Raw rows fetched: {len(df)}")

    if df.empty:
        raise SystemExit(
            "Databento returned 0 rows. Double-check symbol, symbology type, "
            "dataset access, and date range."
        )

    print("Raw columns:", list(df.columns))

    df = ensure_datetime_index(df)
    df = normalize_columns(df)

    print("Resampling to 30-second bars...")
    out = resample_to_30s(df)

    if out.empty:
        raise SystemExit("Resample produced 0 rows. Inspect the raw 1-second data.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH)

    print(f"Saved: {OUTPUT_PATH}")
    print(out.head())
    print(out.tail())
    print(f"Rows: {len(out)}")
    print(f"Start: {out.index.min()} | End: {out.index.max()}")


if __name__ == "__main__":
    main()