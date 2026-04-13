from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from backtesting import Backtest

from src.strategies.manual.ict_multi_setup_v463 import ICT_MULTI_SETUP_V463


CACHE_30S_PATH = Path("src/data/databento_cache/NQ_30s.parquet")
CACHE_1M_PATH = Path("src/data/databento_cache/NQ_1m.parquet")
PREFERRED_BAR_SIZE = os.getenv("NQ_BAR_SIZE", "1m").strip().lower()


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts_event" in df.columns:
            df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
            df = df.set_index("ts_event")
        else:
            raise ValueError("Parquet must have a DatetimeIndex or ts_event column")

    df = df.sort_index().copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    df.index = df.index.tz_localize(None)

    rename_map = {c: c.capitalize() for c in ["open", "high", "low", "close", "volume"] if c in df.columns}
    df = df.rename(columns=rename_map)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    return df[needed].dropna().copy()


def load_nq_data() -> pd.DataFrame:
    print("1) Loading NQ data...")

    chosen_path = None
    if PREFERRED_BAR_SIZE == "30s" and CACHE_30S_PATH.exists():
        chosen_path = CACHE_30S_PATH
    elif CACHE_1M_PATH.exists():
        chosen_path = CACHE_1M_PATH
    elif CACHE_30S_PATH.exists():
        chosen_path = CACHE_30S_PATH

    if chosen_path is None:
        raise FileNotFoundError(f"Missing cache files: {CACHE_1M_PATH} and {CACHE_30S_PATH}")

    print(f"📦 Using local cache: {chosen_path.resolve()}")
    if PREFERRED_BAR_SIZE == "30s" and chosen_path == CACHE_1M_PATH:
        print("⚠️ NQ_30s.parquet not found. Falling back to 1-minute bars.")
    if PREFERRED_BAR_SIZE == "1m" and chosen_path == CACHE_1M_PATH:
        print("✅ Production mode: using 1-minute bars (best validated path to the 100k target).")

    df = pd.read_parquet(chosen_path)
    df = _normalize_ohlcv(df)
    print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()}")
    return df


def main() -> None:
    df = load_nq_data()
    print("2) Creating backtest...")
    bt = Backtest(df, ICT_MULTI_SETUP_V463, cash=1_000_000, commission=0.0, exclusive_orders=False, trade_on_close=False)
    print("3) Running backtest...")
    stats = bt.run()
    print("4) Done.")
    print(stats)

    trades = stats.get("_trades")
    if trades is not None:
        print(f"\nTrade count: {len(trades)}")
        print(trades.tail(30).to_string())

    print("\n5) Debug counters...")
    for k, v in dict(ICT_MULTI_SETUP_V463.DEBUG_COUNTERS).items():
        print(f"{k}: {v}")

    from src.strategies.manual.reporting_v463 import load_strategy_meta

    meta = load_strategy_meta(ICT_MULTI_SETUP_V463)
    if not meta.empty:
        print("\n6) Trade metadata log...")
        print(meta.tail(40).to_string(index=False))


if __name__ == "__main__":
    main()
