from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backtesting import Backtest
from src.data.fetcher import get_ohlcv
from src.strategies.manual.ict_multi_setup_v4 import ICT_MULTI_SETUP_V4

symbol = "MES"
timeframe = "3m"
days_back = 365
tail_rows = 60000


def to_et(ts):
    if pd.isna(ts):
        return ts
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("America/New_York")


print("1) Loading MES data...")
df = get_ohlcv(symbol, exchange="tradovate", timeframe=timeframe, days_back=days_back)
df = df.tail(tail_rows)
print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()}")

print("2) Creating backtest...")
bt = Backtest(
    df,
    ICT_MULTI_SETUP_V4,
    cash=1_000_000,
    commission=0.0,
    exclusive_orders=True,
)

print("3) Running backtest...")
stats = bt.run()

print("4) Done.")
print(stats)

trades = stats.get("_trades")
if trades is not None:
    print(f"\nTrade count: {len(trades)}")
    if not trades.empty:
        trades = trades.copy()
        if "EntryTime" in trades.columns:
            trades["EntryTime_ET"] = trades["EntryTime"].apply(to_et)
        if "ExitTime" in trades.columns:
            trades["ExitTime_ET"] = trades["ExitTime"].apply(to_et)
        print(trades.tail(30))

print("\n5) Debug counters...")
for k, v in ICT_MULTI_SETUP_V4.last_debug_counts.items():
    print(f"{k}: {v}")

print("\n6) Trade metadata log...")
if ICT_MULTI_SETUP_V4.last_trade_log:
    meta_df = pd.DataFrame(ICT_MULTI_SETUP_V4.last_trade_log)
    print(meta_df.tail(30))
else:
    print("No trade metadata logged.")