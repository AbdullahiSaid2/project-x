# ============================================================
# 🏛️  VAULT: SAR Flip
# Symbol:    TAO | Timeframe: 4H
# Vaulted:   2026-03-31
#
# BACKTEST RESULTS:
#   Return    : +22.1%
#   Sharpe    : 1.44
#   Drawdown  : -4.2%
#   Trades    : 184
#   Win Rate  : 40.8%
#
# IDEA: Long when Parabolic SAR flips from above price to below price, stop at SAR level
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
    SAR Flip — Long when Parabolic SAR flips from above to below price.
    Stop at SAR level. TAO 4H: Sharpe 1.44, +22.1%, 184 trades.
    """
    atr_period = 14
    size       = 0.1

    def init(self):
        self.sar = self.I(
            lambda h, l, c: ta_lib.trend.PSARIndicator(
                pd.Series(h), pd.Series(l), pd.Series(c),
                step=0.02, max_step=0.2
            ).psar().values,
            self.data.High, self.data.Low, self.data.Close,
            name="SAR"
        )
        self._sl = 0.0

    def next(self):
        close = float(self.data.Close[-1])
        sar   = float(self.sar[-1])
        prev_sar = float(self.sar[-2]) if len(self.sar) > 1 else sar

        if self.position.is_long:
            if close < self._sl:
                self.position.close()
            return
        if self.position.is_short:
            self.position.close()
            return

        # SAR flipped from above price to below price → Long
        if not self.position:
            was_above = prev_sar > float(self.data.Close[-2])
            is_below  = sar < close
            if was_above and is_below:
                self._sl = sar
                self.buy(size=self.size)


def run(symbol: str = "TAO", timeframe: str = "4H",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 SAR Flip — {symbol} {timeframe} ({days_back} days)")
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
        "strategy":     "SAR Flip",
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
