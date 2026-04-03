# ============================================================
# 🏛️  VAULT: VWAP Volume
# Symbol:    TAO | Timeframe: 4H
# Vaulted:   2026-04-02
# Market:    CRYPTO (Hyperliquid)
#
# BACKTEST RESULTS:
#   Return    : +20.5%
#   Sharpe    : 1.36
#   Drawdown  : -7.4%
#   Trades    : 106
#   Win Rate  : 29.2%
#
# IDEA: Long when price crosses above VWAP and volume is above 20-period average, exit w
# ============================================================

import sys, warnings
import numpy as np
import pandas as pd
import ta as ta_lib
from pathlib import Path
from datetime import datetime
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

warnings.filterwarnings("ignore", category=UserWarning, module="backtesting")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="backtesting")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

class VaultStrategy(Strategy):
    """Long when price crosses above VWAP and volume above 20-period avg. Exit when price crosses below VWAP."""
    vol_period=20; size=0.1

    def init(self):
        self.vwap   = self.I(lambda h,l,c,v: (pd.Series(h+l+c)/3 * pd.Series(v)).cumsum() / pd.Series(v).cumsum(), self.data.High, self.data.Low, self.data.Close, self.data.Volume)
        self.vol_ma = self.I(lambda v: pd.Series(v).rolling(self.vol_period).mean().values, self.data.Volume)

    def next(self):
        import numpy as np
        close=float(self.data.Close[-1]); prev=float(self.data.Close[-2])
        vwap=float(self.vwap[-1]); vol=float(self.data.Volume[-1]); vol_ma=float(self.vol_ma[-1])
        if np.isnan(vwap) or np.isnan(vol_ma): return
        if self.position.is_long:
            if close < vwap: self.position.close()
            return
        crossed_above = prev < float(self.vwap[-2]) and close > vwap
        if not self.position and crossed_above and vol > vol_ma:
            self.buy(size=self.size)


def run(symbol: str = "TAO", timeframe: str = "4H",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 VWAP Volume — {symbol} {timeframe} ({days_back} days)")
    df = get_ohlcv(symbol, exchange=EXCHANGE,
                   timeframe=timeframe, days_back=days_back)
    df = pd.DataFrame({
        "Open":   df["Open"].astype(float).values,
        "High":   df["High"].astype(float).values,
        "Low":    df["Low"].astype(float).values,
        "Close":  df["Close"].astype(float).values,
        "Volume": df["Volume"].astype(float).values,
    }, index=df.index)

    bt    = Backtest(df, VaultStrategy,
                     cash=BACKTEST_INITIAL_CASH,
                     commission=BACKTEST_COMMISSION,
                     exclusive_orders=True)
    stats = bt.run()
    result = {
        "strategy":     "VWAP Volume",
        "symbol":       symbol, "timeframe": timeframe,
        "return_pct":   round(float(stats["Return [%]"]), 2),
        "max_drawdown": round(float(stats["Max. Drawdown [%]"]), 2),
        "sharpe":       round(float(stats.get("Sharpe Ratio", 0) or 0), 3),
        "num_trades":   int(stats["# Trades"]),
        "win_rate":     round(float(stats.get("Win Rate [%]", 0) or 0), 2),
        "date":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"  Return   : {result['return_pct']:>+8.1f}%")
    print(f"  Sharpe   : {result['sharpe']:>8.2f}")
    print(f"  Drawdown : {result['max_drawdown']:>8.1f}%")
    print(f"  Trades   : {result['num_trades']:>8}")
    print(f"  Win Rate : {result['win_rate']:>8.1f}%")
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="TAO")
    p.add_argument("--tf",     default="4H")
    p.add_argument("--days",   type=int, default=1825)
    args = p.parse_args()
    run(args.symbol, args.tf, args.days)
