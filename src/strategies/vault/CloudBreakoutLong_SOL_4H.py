# ============================================================
# 🏛️  VAULT: CloudBreakoutLong
# Symbol:    SOL | Timeframe: 4H
# Vaulted:   2026-03-31
#
# BACKTEST RESULTS:
#   Return    : +60.2%
#   Sharpe    : 1.12
#   Drawdown  : -7.8%
#   Trades    : 233
#   Win Rate  : 25.3%
#
# IDEA: Long when price breaks above the Ichimoku cloud and tenkan-sen is above kijun-se
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
    """
    Cloud Breakout Long — Long when price breaks above Ichimoku cloud
    and tenkan-sen is above kijun-sen.
    SOL 4H: Sharpe 1.12, +60.2%, 233 trades.
    """
    size = 0.1

    def init(self):
        self.tenkan = self.I(
            lambda h, l: ((pd.Series(h).rolling(9).max() +
                           pd.Series(l).rolling(9).min()) / 2).values,
            self.data.High, self.data.Low, name="Tenkan"
        )
        self.kijun = self.I(
            lambda h, l: ((pd.Series(h).rolling(26).max() +
                           pd.Series(l).rolling(26).min()) / 2).values,
            self.data.High, self.data.Low, name="Kijun"
        )
        self.senkou_a = self.I(
            lambda h, l: (((pd.Series(h).rolling(9).max() +
                            pd.Series(l).rolling(9).min()) / 2 +
                           (pd.Series(h).rolling(26).max() +
                            pd.Series(l).rolling(26).min()) / 2) / 2).shift(26).values,
            self.data.High, self.data.Low, name="SenkouA"
        )
        self.senkou_b = self.I(
            lambda h, l: ((pd.Series(h).rolling(52).max() +
                           pd.Series(l).rolling(52).min()) / 2).shift(26).values,
            self.data.High, self.data.Low, name="SenkouB"
        )

    def next(self):
        import numpy as np
        close    = float(self.data.Close[-1])
        tenkan   = float(self.tenkan[-1])
        kijun    = float(self.kijun[-1])
        sa       = float(self.senkou_a[-1])
        sb       = float(self.senkou_b[-1])
        if any(np.isnan(x) for x in [tenkan, kijun, sa, sb]): return

        cloud_top = max(sa, sb)
        if self.position.is_long:
            if close < cloud_top:
                self.position.close()
            return
        if self.position.is_short:
            self.position.close()
            return
        if not self.position:
            if close > cloud_top and tenkan > kijun:
                self.buy(size=self.size)


def run(symbol: str = "SOL", timeframe: str = "4H",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 CloudBreakoutLong — {symbol} {timeframe} ({days_back} days)")
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
        "strategy":     "CloudBreakoutLong",
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
    p.add_argument("--symbol", default="SOL")
    p.add_argument("--tf",     default="4H")
    p.add_argument("--days",   type=int, default=1825)
    args = p.parse_args()
    run(args.symbol, args.tf, args.days)
