# ============================================================
# 🏛️  VAULT: BollingerOversoldReversal
# Symbol:    TAO | Timeframe: 4H
# Vaulted:   2026-04-02
# Market:    CRYPTO (Hyperliquid)
#
# BACKTEST RESULTS:
#   Return    : +21.8%
#   Sharpe    : 1.41
#   Drawdown  : -5.2%
#   Trades    : 187
#   Win Rate  : 45.5%
#
# IDEA: Long when %b (Bollinger Band) is below 0 and close is below the lower Bollinger 
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
    """Long when %b below 0 (price below lower BB). Exit when %b above 1."""
    period=20; std=2.0; size=0.1

    def init(self):
        self.pctb = self.I(lambda c: ta_lib.volatility.BollingerBands(pd.Series(c), window=self.period, window_dev=self.std).bollinger_pband().values, self.data.Close)
        self.atr  = self.I(lambda h,l,c: ta_lib.volatility.average_true_range(pd.Series(h),pd.Series(l),pd.Series(c),window=14).values, self.data.High, self.data.Low, self.data.Close)

    def next(self):
        import numpy as np
        pb=float(self.pctb[-1]); atr=float(self.atr[-1])
        if np.isnan(pb) or np.isnan(atr): return
        if self.position.is_long:
            if pb > 1.0: self.position.close()
            return
        if not self.position and pb < 0:
            self._sl = float(self.data.Close[-1]) - atr*1.5
            self.buy(size=self.size)


def run(symbol: str = "TAO", timeframe: str = "4H",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 BollingerOversoldReversal — {symbol} {timeframe} ({days_back} days)")
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
        "strategy":     "BollingerOversoldReversal",
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
