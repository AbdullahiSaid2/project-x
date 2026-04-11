from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backtesting import Backtest
from src.data.fetcher import get_ohlcv
from src.strategies.manual.pdh_pdl_sweep_reclaim import PDHSweepRejectPDLSweepReclaim

symbol = "MES"
timeframe = "15m"
days_back = 120

print("1) Loading data...")
df = get_ohlcv(symbol, exchange="tradovate", timeframe=timeframe, days_back=days_back)
df = df.tail(5000)
print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()}")

print("2) Creating backtest...")
bt = Backtest(
    df,
    PDHSweepRejectPDLSweepReclaim,
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
    print(trades.tail(20))