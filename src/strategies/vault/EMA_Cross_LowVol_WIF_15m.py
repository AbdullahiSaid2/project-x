# ============================================================
# 🏛️  VAULT: EMA_Cross_LowVol
# Symbol:    WIF | Timeframe: 15m
# Vaulted:   2026-04-02
# Market:    CRYPTO (Hyperliquid)
#
# BACKTEST RESULTS:
#   Return    : +21.2%
#   Sharpe    : 1.27
#   Drawdown  : -4.8%
#   Trades    : 291
#   Win Rate  : 20.6%
#
# IDEA: Long when 20 EMA crosses above 50 EMA and ATR is below the 20-bar ATR average in
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
    """Long when 9 EMA crosses above 21 EMA during low volatility (ATR below avg)."""
    fast=9; slow=21; atr_p=14; size=0.1; _sl=0.0

    def init(self):
        self.ema_f = self.I(lambda c: ta_lib.trend.ema_indicator(pd.Series(c), window=self.fast).values, self.data.Close)
        self.ema_s = self.I(lambda c: ta_lib.trend.ema_indicator(pd.Series(c), window=self.slow).values, self.data.Close)
        self.atr   = self.I(lambda h,l,c: ta_lib.volatility.average_true_range(pd.Series(h),pd.Series(l),pd.Series(c),window=self.atr_p).values, self.data.High, self.data.Low, self.data.Close)
        self.atr_ma= self.I(lambda h,l,c: pd.Series(ta_lib.volatility.average_true_range(pd.Series(h),pd.Series(l),pd.Series(c),window=self.atr_p).values).rolling(20).mean().values, self.data.High, self.data.Low, self.data.Close)

    def next(self):
        import numpy as np
        ef=float(self.ema_f[-1]); es=float(self.ema_s[-1])
        pf=float(self.ema_f[-2]); ps=float(self.ema_s[-2])
        atr=float(self.atr[-1]); atr_ma=float(self.atr_ma[-1])
        close=float(self.data.Close[-1])
        if any(np.isnan(x) for x in [ef,es,pf,ps,atr,atr_ma]): return
        if self.position.is_long:
            if ef < es: self.position.close()
            return
        crossed = pf < ps and ef > es
        if not self.position and crossed and atr < atr_ma:
            self._sl = close - atr * 1.5
            self.buy(size=self.size)


def run(symbol: str = "WIF", timeframe: str = "15m",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 EMA_Cross_LowVol — {symbol} {timeframe} ({days_back} days)")
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
        "strategy":     "EMA_Cross_LowVol",
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
    p.add_argument("--symbol", default="WIF")
    p.add_argument("--tf",     default="15m")
    p.add_argument("--days",   type=int, default=1825)
    args = p.parse_args()
    run(args.symbol, args.tf, args.days)
