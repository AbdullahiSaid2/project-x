from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from backtesting import Backtest

from src.strategies.manual.ict_multi_setup_v458 import ICT_MULTI_SETUP_V458


def load_nq_data() -> pd.DataFrame:
    cache_path = Path("src/data/databento_cache/NQ_1m.parquet")
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_path}")

    print("1) Loading NQ data...")
    print(f"📦 Using local cache: {cache_path.resolve()}")

    df = pd.read_parquet(cache_path).copy()

    rename_map = {}
    cols = {c.lower(): c for c in df.columns}
    if "open" not in cols and "o" in cols:
        rename_map[cols["o"]] = "Open"
    if "high" not in cols and "h" in cols:
        rename_map[cols["h"]] = "High"
    if "low" not in cols and "l" in cols:
        rename_map[cols["l"]] = "Low"
    if "close" not in cols and "c" in cols:
        rename_map[cols["c"]] = "Close"
    if "volume" not in cols and "v" in cols:
        rename_map[cols["v"]] = "Volume"
    if rename_map:
        df = df.rename(columns=rename_map)

    canonical = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    for low_name, proper in canonical.items():
        for c in list(df.columns):
            if c.lower() == low_name:
                df = df.rename(columns={c: proper})

    needed = ["Open", "High", "Low", "Close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    if "Volume" not in df.columns:
        df["Volume"] = 0

    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = None
        for candidate in ["ts_event", "timestamp", "datetime", "date", "time"]:
            if candidate in df.columns:
                ts_col = candidate
                break
        if ts_col is None:
            raise ValueError("Could not find a datetime index or timestamp column.")
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.set_index(ts_col)

    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df.index = idx.tz_localize(None)

    df = df.sort_index()
    print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()}")
    return df


def main() -> None:
    df = load_nq_data()

    print("2) Creating backtest...")
    bt = Backtest(
        df,
        ICT_MULTI_SETUP_V458,
        cash=1_000_000,
        commission=0.0,
        exclusive_orders=False,
        trade_on_close=False,
    )

    print("3) Running backtest...")
    stats = bt.run()

    print("4) Done.")
    print(stats)

    trades = stats.get("_trades", pd.DataFrame())
    print(f"\nTrade count: {len(trades)}")
    if not trades.empty:
        print(trades.tail(30).to_string())

    print("\n5) Debug counters...")
    for k, v in dict(ICT_MULTI_SETUP_V458.DEBUG_COUNTERS).items():
        print(f"{k}: {v}")

    print("\n6) Trade metadata log...")
    meta = pd.DataFrame(ICT_MULTI_SETUP_V458.TRADE_METADATA_LOG)
    if not meta.empty:
        print(meta.tail(40).to_string(index=False))
    else:
        print("No metadata rows logged.")


if __name__ == "__main__":
    main()
