# ============================================================
# 🏛️  VAULT: TrendStrengthEntry
# Symbol:    TAO | Timeframe: 4H
# Vaulted:   2026-04-02
# Market:    CRYPTO (Hyperliquid)
#
# BACKTEST RESULTS:
#   Return    : +8.0%
#   Sharpe    : 1.04
#   Drawdown  : -3.7%
#   Trades    : 36
#   Win Rate  : 30.6%
#
# IDEA: Long when ADX crosses above 25 indicating a strong trend and price is above the 
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
    """Long when ADX > 25 (strong trend) and price above 50 EMA."""
    adx_period=14; ema_period=50; adx_threshold=25; size=0.1; _sl=0.0

    def init(self):
        self.adx = self.I(lambda h,l,c: ta_lib.trend.ADXIndicator(pd.Series(h),pd.Series(l),pd.Series(c),window=self.adx_period).adx().values, self.data.High, self.data.Low, self.data.Close)
        self.ema = self.I(lambda c: ta_lib.trend.ema_indicator(pd.Series(c),window=self.ema_period).values, self.data.Close)
        self.atr = self.I(lambda h,l,c: ta_lib.volatility.average_true_range(pd.Series(h),pd.Series(l),pd.Series(c),window=14).values, self.data.High, self.data.Low, self.data.Close)

    def next(self):
        import numpy as np
        adx=float(self.adx[-1]); ema=float(self.ema[-1]); atr=float(self.atr[-1]); close=float(self.data.Close[-1])
        if any(np.isnan(x) for x in [adx,ema,atr]): return
        if self.position.is_long:
            if close < ema or adx < 20: self.position.close()
            return
        if not self.position and adx > self.adx_threshold and close > ema:
            self._sl = close - atr * 1.5; self.buy(size=self.size)


def run(symbol: str = "TAO", timeframe: str = "4H",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 TrendStrengthEntry — {symbol} {timeframe} ({days_back} days)")
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
        "strategy":     "TrendStrengthEntry",
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
