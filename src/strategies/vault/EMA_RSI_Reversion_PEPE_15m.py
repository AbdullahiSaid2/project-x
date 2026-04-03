# ============================================================
# 🏛️  VAULT: EMA_RSI_Reversion
# Symbol:    PEPE | Timeframe: 15m
# Vaulted:   2026-04-02
# Market:    CRYPTO (Hyperliquid)
#
# BACKTEST RESULTS:
#   Return    : +25.4%
#   Sharpe    : 1.44
#   Drawdown  : -5.6%
#   Trades    : 899
#   Win Rate  : 44.2%
#
# IDEA: Long when price is more than 3 percent below the 20 EMA and RSI is below 40, tar
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
    """Long when price >3% below 20 EMA and RSI <40. Exit when RSI >60."""
    ema_period = 20; rsi_period = 14; size = 0.1

    def init(self):
        self.ema = self.I(lambda c: ta_lib.trend.ema_indicator(pd.Series(c), window=self.ema_period).values, self.data.Close)
        self.rsi = self.I(lambda c: ta_lib.momentum.rsi(pd.Series(c), window=self.rsi_period).values, self.data.Close)

    def next(self):
        import numpy as np
        close=float(self.data.Close[-1]); ema=float(self.ema[-1]); rsi=float(self.rsi[-1])
        if np.isnan(ema) or np.isnan(rsi): return
        if self.position.is_long:
            if rsi > 60: self.position.close()
            return
        if not self.position and close < ema * 0.97 and rsi < 40:
            self.buy(size=self.size)


def run(symbol: str = "PEPE", timeframe: str = "15m",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 EMA_RSI_Reversion — {symbol} {timeframe} ({days_back} days)")
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
        "strategy":     "EMA_RSI_Reversion",
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
    p.add_argument("--symbol", default="PEPE")
    p.add_argument("--tf",     default="15m")
    p.add_argument("--days",   type=int, default=1825)
    args = p.parse_args()
    run(args.symbol, args.tf, args.days)
