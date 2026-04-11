from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backtesting import Backtest
from src.data.fetcher import get_ohlcv
from src.strategies.manual.ict_multi_setup_v43 import ICT_MULTI_SETUP_V43


symbol = "NQ"
timeframe = "1m"
days_back = 365
tail_rows = 180000


def to_et(ts):
    if pd.isna(ts):
        return ts
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("America/New_York")


def summarize_group(df: pd.DataFrame, group_cols):
    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            trades=("return_pct", "size"),
            win_rate_pct=("return_pct", lambda s: (s > 0).mean() * 100),
            avg_return_pct=("return_pct", "mean"),
            median_return_pct=("return_pct", "median"),
            total_return_pct=("return_pct", "sum"),
            best_return_pct=("return_pct", "max"),
            worst_return_pct=("return_pct", "min"),
        )
        .reset_index()
        .sort_values(["total_return_pct", "avg_return_pct", "trades"], ascending=[False, False, False])
    )
    return out


print("1) Loading data...")
df = get_ohlcv(symbol, exchange="tradovate", timeframe=timeframe, days_back=days_back)
df = df.tail(tail_rows)
print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()}")

print("2) Running backtest...")
bt = Backtest(
    df,
    ICT_MULTI_SETUP_V43,
    cash=1_000_000,
    commission=0.0,
    exclusive_orders=True,
)
stats = bt.run()

print("\n3) Headline stats")
print(stats)

print("\n4) Metadata log")
meta = pd.DataFrame(ICT_MULTI_SETUP_V43.last_trade_log)

if meta.empty:
    print("No trade metadata found.")
    raise SystemExit(0)

meta["entry_time"] = pd.to_datetime(meta["entry_time"], errors="coerce")
meta["exit_time"] = pd.to_datetime(meta["exit_time"], errors="coerce")
meta["entry_time_et"] = meta["entry_time"].apply(to_et)
meta["exit_time_et"] = meta["exit_time"].apply(to_et)

print(meta.tail(20))

print("\n5) By setup_type")
by_setup = summarize_group(meta, ["setup_type"])
print(by_setup.to_string(index=False))

print("\n6) By bridge_type")
by_bridge = summarize_group(meta, ["bridge_type"])
print(by_bridge.to_string(index=False))

print("\n7) By setup_type + bridge_type")
by_combo = summarize_group(meta, ["setup_type", "bridge_type"])
print(by_combo.to_string(index=False))

print("\n8) Direction split")
by_side = summarize_group(meta, ["side"])
print(by_side.to_string(index=False))

print("\n9) Setup + side")
by_setup_side = summarize_group(meta, ["setup_type", "side"])
print(by_setup_side.to_string(index=False))