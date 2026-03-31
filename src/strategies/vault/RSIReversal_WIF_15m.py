# ============================================================
# 🏛️  VAULT: RSI Reversal
# Symbol:    WIF | Timeframe: 15m
# Vaulted:   2026-03-31
#
# BACKTEST RESULTS:
#   Return    : +3.5%
#   Sharpe    : 1.15
#   Drawdown  : -1.4%
#   Trades    : 47
#   Win Rate  : 55.3%
#
# IDEA: Short when RSI rises above 70 and price is below the 200 EMA, exit when RSI fall
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
    RSI Reversal — Short when RSI > 70 and price below 200 EMA.
    Exit when RSI falls below 50. WIF 15m: Sharpe 1.15, +3.5%, 47 trades.
    """
    rsi_period = 14
    ema_period = 200
    size       = 0.1

    def init(self):
        self.rsi = self.I(
            lambda c: ta_lib.momentum.rsi(pd.Series(c), window=self.rsi_period).values,
            self.data.Close, name="RSI"
        )
        self.ema = self.I(
            lambda c: ta_lib.trend.ema_indicator(pd.Series(c), window=self.ema_period).values,
            self.data.Close, name="EMA200"
        )

    def next(self):
        rsi   = float(self.rsi[-1])
        ema   = float(self.ema[-1])
        close = float(self.data.Close[-1])
        import numpy as np
        if np.isnan(rsi) or np.isnan(ema): return

        if self.position.is_short:
            if rsi < 50:
                self.position.close()
            return

        if not self.position:
            if rsi > 70 and close < ema:
                self.sell(size=self.size)


def run(symbol: str = "WIF", timeframe: str = "15m",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 RSI Reversal — {symbol} {timeframe} ({days_back} days)")
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
        "strategy":     "RSI Reversal",
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
