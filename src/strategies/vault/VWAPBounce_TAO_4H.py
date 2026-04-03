# ============================================================
# 🏛️  VAULT: VWAP Bounce
# Symbol:    TAO | Timeframe: 4H
# Vaulted:   2026-04-02
# Market:    CRYPTO (Hyperliquid)
#
# BACKTEST RESULTS:
#   Return    : +21.0%
#   Sharpe    : 1.34
#   Drawdown  : -7.7%
#   Trades    : 97
#   Win Rate  : 39.2%
#
# IDEA: Long when price bounces off VWAP (price touches VWAP then closes above it) and M
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
    """Long when price touches VWAP then closes above it with MACD positive. Exit when price closes below VWAP."""
    size=0.1

    def init(self):
        self.vwap = self.I(lambda h,l,c,v: (pd.Series(h+l+c)/3 * pd.Series(v)).cumsum() / pd.Series(v).cumsum(), self.data.High, self.data.Low, self.data.Close, self.data.Volume)
        self.macd = self.I(lambda c: ta_lib.trend.macd_diff(pd.Series(c)).values, self.data.Close)

    def next(self):
        import numpy as np
        close=float(self.data.Close[-1]); low=float(self.data.Low[-1])
        vwap=float(self.vwap[-1]); macd=float(self.macd[-1])
        if np.isnan(vwap) or np.isnan(macd): return
        if self.position.is_long:
            if close < vwap: self.position.close()
            return
        touched = low <= vwap
        if not self.position and touched and close > vwap and macd > 0:
            self.buy(size=self.size)


def run(symbol: str = "TAO", timeframe: str = "4H",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 VWAP Bounce — {symbol} {timeframe} ({days_back} days)")
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
        "strategy":     "VWAP Bounce",
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
