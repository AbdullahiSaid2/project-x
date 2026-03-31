# ============================================================
# 🌙 VAULT STRATEGY: Z-Score Reversion
# Symbol: MES | Timeframe: 4H
#
# DISCOVERY STATS (Run 3 — Claude — 2026-03-29):
#   Return   : +44.2%
#   Sharpe   : 1.66
#   Drawdown : -0.2%
#   Trades   : 46
#   Win Rate : 89.1%
#
# WHY IT WAS KEPT:
#   89.1% win rate — highest of any strategy discovered so far.
#   -0.2% max drawdown means it almost never has a losing period.
#   46 trades over 5 years = ~1 per month. Low frequency but
#   extremely precise when it fires.
#
# STRATEGY LOGIC:
#   Z-score measures how many standard deviations price is from
#   its 20-bar mean. When z-score rises above +2.0 (statistically
#   extreme), price is overextended and likely to revert.
#   Short at z-score > 2.0, exit when z-score returns to 0.
#   Reverse for longs at z-score < -2.0.
#
# ORIGINAL IDEA:
#   Short when the 20-bar z-score of price rises above positive
#   2.0, exit when z-score returns to zero.
#
# VAULTED: 29/03/2026
# DO NOT DELETE — verified winning strategy
# ============================================================

import sys
import warnings
import numpy as np
import pandas as pd
import ta as ta_lib
import ta
from pathlib import Path
from datetime import datetime
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

warnings.filterwarnings("ignore", category=UserWarning,    module="backtesting")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="backtesting")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class VaultStrategy(Strategy):
    """
    Z-Score Mean Reversion — trades statistical extremes.

    Short when 20-bar z-score rises above +2.0 (overextended up).
    Long when 20-bar z-score falls below -2.0 (overextended down).
    Exit when z-score returns to 0 (mean).
    """

    zscore_period  = 20    # lookback for mean and std calculation
    zscore_entry   = 2.0   # z-score threshold to trigger trade
    zscore_exit    = 0.0   # z-score level to exit (mean reversion complete)
    atr_period     = 14
    atr_sl_mult    = 2.0   # wider stop — z-score can overshoot before reverting
    size           = 0.1

    def init(self):
        close = pd.Series(self.data.Close)

        # Z-score = (price - rolling_mean) / rolling_std
        self.zscore = self.I(
            lambda c: (
                (pd.Series(c) - pd.Series(c).rolling(self.zscore_period).mean()) /
                pd.Series(c).rolling(self.zscore_period).std()
            ).values,
            close,
            name="ZScore"
        )

        self.atr = self.I(
            lambda h, l, c: ta_lib.volatility.average_true_range(
                pd.Series(h), pd.Series(l), pd.Series(c),
                window=self.atr_period
            ).values,
            self.data.High, self.data.Low, close,
            name="ATR"
        )

    def next(self):
        z   = self.zscore[-1]
        atr = self.atr[-1]

        if np.isnan(z) or np.isnan(atr) or atr == 0:
            return

        price = self.data.Close[-1]

        # Exit conditions — z-score returned to mean
        if self.position.is_short and z <= self.zscore_exit:
            self.position.close()
            return
        if self.position.is_long and z >= self.zscore_exit:
            self.position.close()
            return

        if not self.position:
            if z > self.zscore_entry:
                # Overextended up — short
                sl = price + (atr * self.atr_sl_mult)
                self.sell(size=self.size, sl=sl)

            elif z < -self.zscore_entry:
                # Overextended down — long
                sl = price - (atr * self.atr_sl_mult)
                self.buy(size=self.size, sl=sl)


def run(symbol: str = "MES", timeframe: str = "4H",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 Z-Score Reversion — {symbol} {timeframe} ({days_back} days)")

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
        "strategy":     "Z-Score Reversion",
        "symbol":       symbol,
        "timeframe":    timeframe,
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
    p.add_argument("--symbol", default="MES")
    p.add_argument("--tf",     default="4H")
    p.add_argument("--days",   type=int, default=1825)
    args = p.parse_args()
    run(args.symbol, args.tf, args.days)
