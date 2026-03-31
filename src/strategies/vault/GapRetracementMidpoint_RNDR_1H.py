# ============================================================
# 🏛️  VAULT: GapRetracementMidpoint
# Symbol:    RNDR | Timeframe: 1H
# Vaulted:   2026-03-31
#
# BACKTEST RESULTS:
#   Return    : +0.6%
#   Sharpe    : 1.02
#   Drawdown  : -0.3%
#   Trades    : 22
#   Win Rate  : 54.5%
#
# IDEA: Long when price retraces into the midpoint of the lowest gap in the past 20 bars
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
    Gap Retracement Midpoint — Long when price retraces into the midpoint
    of the lowest gap in the past 20 bars (FVG/imbalance fill).
    RNDR 1H: Sharpe 1.02, +0.6%, 22 trades.
    """
    lookback = 20
    atr_mult = 1.5
    size     = 0.1

    def init(self):
        self.atr = self.I(
            lambda h, l, c: ta_lib.volatility.average_true_range(
                pd.Series(h), pd.Series(l), pd.Series(c), window=14).values,
            self.data.High, self.data.Low, self.data.Close,
            name="ATR"
        )
        self._sl = 0.0
        self._tp = 0.0

    def next(self):
        import numpy as np
        close = float(self.data.Close[-1])
        atr   = float(self.atr[-1])
        if np.isnan(atr) or atr == 0: return
        n = min(self.lookback, len(self.data.Close)-2)

        if self.position.is_long:
            if close < self._sl or close > self._tp:
                self.position.close()
            return

        # Find lowest gap (where candle low > previous candle high = bullish gap)
        lowest_gap_mid = None
        for i in range(1, n+1):
            prev_high = float(self.data.High[-i-1])
            curr_low  = float(self.data.Low[-i])
            if curr_low > prev_high:
                gap_mid = (curr_low + prev_high) / 2
                if lowest_gap_mid is None or gap_mid < lowest_gap_mid:
                    lowest_gap_mid = gap_mid

        if not self.position and lowest_gap_mid is not None:
            if abs(close - lowest_gap_mid) < atr * 0.3:
                self._sl = close - atr * 1.0
                self._tp = close + atr * self.atr_mult
                if self._sl < close < self._tp:
                    self.buy(size=self.size)


def run(symbol: str = "RNDR", timeframe: str = "1H",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 GapRetracementMidpoint — {symbol} {timeframe} ({days_back} days)")
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
        "strategy":     "GapRetracementMidpoint",
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
    p.add_argument("--symbol", default="RNDR")
    p.add_argument("--tf",     default="1H")
    p.add_argument("--days",   type=int, default=1825)
    args = p.parse_args()
    run(args.symbol, args.tf, args.days)
