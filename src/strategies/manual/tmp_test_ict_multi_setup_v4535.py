from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backtesting import Backtest
from src.data.fetcher import get_ohlcv
from src.strategies.manual.ict_multi_setup_v4535 import ICT_MULTI_SETUP_V4535


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


print("1) Loading NQ data...")
df = get_ohlcv(symbol, exchange="tradovate", timeframe=timeframe, days_back=days_back)
df = df.tail(tail_rows)
print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()}")

print("2) Creating backtest...")
bt = Backtest(
    df,
    ICT_MULTI_SETUP_V4535,
    cash=1_000_000,
    commission=0.0,
    exclusive_orders=True,
)

print("3) Running backtest...")
stats = bt.run()

print("4) Done.")
print(stats)

trades = stats.get("_trades")
if trades is not None and not trades.empty:
    trades = trades.copy()
    trades["EntryTime_ET"] = pd.to_datetime(trades["EntryTime"], errors="coerce").apply(to_et)
    trades["ExitTime_ET"] = pd.to_datetime(trades["ExitTime"], errors="coerce").apply(to_et)
    print(f"\nTrade count: {len(trades)}")
    print(trades.tail(30))
else:
    print("\nTrade count: 0")
    print(trades)

print("\n5) Debug counters...")
debug = getattr(ICT_MULTI_SETUP_V4535, "last_debug_counts", {})
if debug:
    for k, v in debug.items():
        print(f"{k}: {v}")
else:
    print("debug counts unavailable from strategy class")

print("\n6) Trade metadata log...")
trade_log = getattr(ICT_MULTI_SETUP_V4535, "last_trade_log", [])
if trade_log:
    meta = pd.DataFrame(trade_log)
    if "entry_time" in meta.columns:
        meta["entry_time_et"] = pd.to_datetime(meta["entry_time"], errors="coerce").apply(to_et)
    if "exit_time" in meta.columns:
        meta["exit_time_et"] = pd.to_datetime(meta["exit_time"], errors="coerce").apply(to_et)
    print(meta.tail(30).to_string(index=False))
else:
    print("No trade metadata logged.")