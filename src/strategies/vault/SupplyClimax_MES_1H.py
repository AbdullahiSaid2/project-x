# ============================================================
# 🌙 VAULT STRATEGY: Supply Climax
# Symbol: MES | Timeframe: 1H
#
# DISCOVERY STATS (Run 3 — Claude — 2026-03-29):
#   Return   : +120.9%
#   Sharpe   : 1.17
#   Drawdown : -0.5%
#   Trades   : 78
#   Win Rate : 21.8%
#
# WHY IT WAS KEPT:
#   Asymmetric payoff profile — 21.8% win rate but -0.5% drawdown
#   means winners are dramatically larger than losers.
#   78 trades over 5 years is reasonable.
#   Works on MES 1H only.
#
# STRATEGY LOGIC:
#   A supply climax is a wide-range up bar that closes near its
#   LOW with very high volume. This signals institutional supply
#   entering — smart money selling into retail buying euphoria.
#   The wide range and volume confirm urgency; the poor close
#   (near low) confirms that sellers overwhelmed buyers.
#   Short after this pattern. Exit on reversal signal.
#
# ORIGINAL IDEA:
#   Short when a wide spread up bar closes on its low with very
#   high volume indicating supply entering and price likely to
#   reverse down.
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
    Supply Climax Short — VSA wide-spread bar with poor close.

    Short when:
    - Bar range > 1.5x ATR (wide range = climactic move)
    - Volume > 2x 20-bar average (institutional participation)
    - Close in lower 25% of bar range (sellers won)
    - Bar is bullish (close > open — retail buying into supply)

    Exit when RSI falls below 40 (momentum exhausted) or
    price reaches 2x ATR profit target.
    """

    # Entry parameters
    atr_period        = 14
    range_atr_mult    = 1.5    # bar range must be > this * ATR
    volume_lookback   = 20
    volume_mult       = 2.0    # volume must be > this * average
    close_pct         = 0.25   # close must be in bottom X% of bar range

    # Exit parameters
    rsi_period        = 14
    rsi_exit          = 40     # exit when RSI drops below this
    atr_tp_mult       = 2.0    # take profit distance in ATR
    atr_sl_mult       = 1.0    # stop loss distance in ATR

    size              = 0.1

    def init(self):
        close  = pd.Series(self.data.Close)
        high   = pd.Series(self.data.High)
        low    = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        self.atr = self.I(
            lambda h, l, c: ta_lib.volatility.average_true_range(
                pd.Series(h), pd.Series(l), pd.Series(c),
                window=self.atr_period
            ).values,
            self.data.High, self.data.Low, close,
            name="ATR"
        )

        self.avg_vol = self.I(
            lambda v: pd.Series(v).rolling(self.volume_lookback).mean().values,
            self.data.Volume,
            name="AvgVol"
        )

        self.rsi = self.I(
            lambda c: ta_lib.momentum.rsi(
                pd.Series(c), window=self.rsi_period
            ).values,
            close,
            name="RSI"
        )

    def next(self):
        bar_high   = self.data.High[-1]
        bar_low    = self.data.Low[-1]
        bar_close  = self.data.Close[-1]
        bar_open   = self.data.Open[-1]
        volume     = self.data.Volume[-1]
        atr        = self.atr[-1]
        avg_vol    = self.avg_vol[-1]
        rsi        = self.rsi[-1]

        if np.isnan(atr) or atr == 0 or np.isnan(avg_vol) or avg_vol == 0:
            return

        bar_range = bar_high - bar_low
        if bar_range == 0:
            return

        close_position = (bar_close - bar_low) / bar_range

        # Supply Climax conditions
        wide_range    = bar_range > (self.range_atr_mult * atr)
        high_volume   = volume > (self.volume_mult * avg_vol)
        poor_close    = close_position <= self.close_pct
        bullish_bar   = bar_close > bar_open   # retail buying = up bar

        supply_climax = wide_range and high_volume and poor_close and bullish_bar

        # Exit: RSI exhausted
        rsi_exit = not np.isnan(rsi) and rsi <= self.rsi_exit

        if self.position.is_short:
            if rsi_exit:
                self.position.close()

        elif not self.position:
            if supply_climax:
                sl = bar_close + (atr * self.atr_sl_mult)
                tp = bar_close - (atr * self.atr_tp_mult)
                self.sell(size=self.size, sl=sl, tp=tp)


def run(symbol: str = "MES", timeframe: str = "1H",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 Supply Climax — {symbol} {timeframe} ({days_back} days)")

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
        "strategy":     "Supply Climax",
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
    p.add_argument("--tf",     default="1H")
    p.add_argument("--days",   type=int, default=1825)
    args = p.parse_args()
    run(args.symbol, args.tf, args.days)
