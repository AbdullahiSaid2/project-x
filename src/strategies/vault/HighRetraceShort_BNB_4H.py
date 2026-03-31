# ============================================================
# 🏛️  VAULT: HighRetraceShort
# Symbol:    BNB | Timeframe: 4H
# Vaulted:   2026-03-31
#
# BACKTEST RESULTS:
#   Return    : +8.7%
#   Sharpe    : 1.03
#   Drawdown  : -1.3%
#   Trades    : 48
#   Win Rate  : 50.0%
#
# IDEA: Short when price makes a new 20-bar high then within the next 3 bars makes a sin
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
    High Retrace Short — Short when price makes new 20-bar high then
    within 3 bars makes a single bar move down.
    BNB 4H: Sharpe 1.03, +8.7%, 48 trades.
    """
    lookback = 20
    atr_mult = 2.0
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
        self._triggered_bar = -99
        self._trigger_high  = 0.0

    def next(self):
        import numpy as np
        close = float(self.data.Close[-1])
        high  = float(self.data.High[-1])
        atr   = float(self.atr[-1])
        bar   = len(self.data.Close)
        if np.isnan(atr) or atr == 0: return

        if self.position.is_short:
            if high >= self._sl or close <= self._tp:
                self.position.close()
            return

        prior_highs = [float(self.data.High[-i]) for i in range(2, self.lookback+2)]
        new_high = high > max(prior_highs)
        if new_high:
            self._triggered_bar = bar
            self._trigger_high  = high

        if not self.position and self._triggered_bar > 0:
            bars_since = bar - self._triggered_bar
            if 1 <= bars_since <= 3:
                prev_close = float(self.data.Close[-2])
                if close < prev_close:
                    self._sl = self._trigger_high + atr
                    self._tp = close - self.atr_mult * atr
                    if self._tp < close < self._sl:
                        self.sell(size=self.size)


def run(symbol: str = "BNB", timeframe: str = "4H",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 HighRetraceShort — {symbol} {timeframe} ({days_back} days)")
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
        "strategy":     "HighRetraceShort",
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
    p.add_argument("--symbol", default="BNB")
    p.add_argument("--tf",     default="4H")
    p.add_argument("--days",   type=int, default=1825)
    args = p.parse_args()
    run(args.symbol, args.tf, args.days)
