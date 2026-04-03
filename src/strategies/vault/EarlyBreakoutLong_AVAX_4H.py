# ============================================================
# 🏛️  VAULT: EarlyBreakoutLong
# Symbol:    AVAX | Timeframe: 4H
# Vaulted:   2026-04-02
# Market:    CRYPTO (Hyperliquid)
#
# BACKTEST RESULTS:
#   Return    : +9.8%
#   Sharpe    : 1.02
#   Drawdown  : -1.9%
#   Trades    : 94
#   Win Rate  : 84.0%
#
# IDEA: Long when price moves more than 1.5 ATR above the open within the first 3 bars, 
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
    """Long when price moves >1.5 ATR above open within first 3 bars."""
    atr_mult=1.5; size=0.1; _sl=0.0; _tp=0.0

    def init(self):
        self.atr = self.I(lambda h,l,c: ta_lib.volatility.average_true_range(pd.Series(h),pd.Series(l),pd.Series(c),window=14).values, self.data.High, self.data.Low, self.data.Close)

    def next(self):
        import numpy as np
        open_=float(self.data.Open[-1]); close=float(self.data.Close[-1])
        low=float(self.data.Low[-1]); atr=float(self.atr[-1])
        if np.isnan(atr): return
        if self.position.is_long:
            if low <= self._sl or close >= self._tp: self.position.close()
            return
        if not self.position and close > open_ + atr*self.atr_mult:
            self._sl=open_; self._tp=close+atr*2.0; self.buy(size=self.size)


def run(symbol: str = "AVAX", timeframe: str = "4H",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 EarlyBreakoutLong — {symbol} {timeframe} ({days_back} days)")
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
        "strategy":     "EarlyBreakoutLong",
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
    p.add_argument("--symbol", default="AVAX")
    p.add_argument("--tf",     default="4H")
    p.add_argument("--days",   type=int, default=1825)
    args = p.parse_args()
    run(args.symbol, args.tf, args.days)
