# ============================================================
# 🌙 VAULT STRATEGY: VolumeSpikeFade
# Symbol: MES | Timeframe: 15m
#
# DISCOVERY STATS (Run 2 — 2026-03-28):
#   Return   : +178.3%
#   Sharpe   : 1.62
#   Drawdown : -7.8%
#   Trades   : 501
#   Win Rate : 61.7%
#
# WHY IT WAS KEPT:
#   501 trades over 5 years = ~2 per week — most statistically
#   reliable result in run 2. Sharpe 1.62 is solid. -7.8% drawdown
#   is well within Apex prop firm limits.
#
#   Note: Only confirmed on MES 15m. MNQ 15m blew up (-100%).
#   Forward test on MES 15m only before live trading.
#
# STRATEGY LOGIC:
#   When an up bar (close > open) has volume 3x the 20-bar
#   average AND closes in the lower 25% of its range,
#   it signals supply entering — smart money selling into
#   retail buying. Go short. The opposite applies for longs.
#
#   This is a classic volume spread analysis (VSA) pattern:
#   high volume + poor close = distribution/stopping volume.
#
# ORIGINAL IDEA:
#   Long when volume is 3x the 20-bar average on a down bar
#   and price closes in the upper 25 percent of bar range.
#   Short when volume is 3x the 20-bar average on an up bar
#   and price closes in the lower 25 percent of bar range.
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
    Volume Spike Fade — VSA distribution/accumulation signal.

    Short when: up bar + volume 3x average + close in lower 25% of range
    Long when:  down bar + volume 3x average + close in upper 75% of range

    Parameters tuned for MES 15m.
    """

    # Tunable parameters
    volume_lookback  = 20    # bars for average volume calculation
    volume_multiple  = 3     # how many times average volume must be exceeded
    close_threshold  = 0.25  # close must be in bottom/top X% of bar range
    size             = 0.1   # fixed position size — never change this

    def init(self):
        # 20-bar average volume
        self.avg_vol = self.I(
            lambda v: pd.Series(v).rolling(self.volume_lookback).mean().values,
            self.data.Volume,
            name="AvgVol"
        )

    def next(self):
        bar_range = self.data.High[-1] - self.data.Low[-1]
        if bar_range == 0:
            return

        close      = self.data.Close[-1]
        open_      = self.data.Open[-1]
        low        = self.data.Low[-1]
        high       = self.data.High[-1]
        volume     = self.data.Volume[-1]
        avg_volume = self.avg_vol[-1]

        if avg_volume == 0 or np.isnan(avg_volume):
            return

        # Volume spike condition
        volume_spike = volume >= (self.volume_multiple * avg_volume)

        # Close position within bar range (0 = at low, 1 = at high)
        close_position = (close - low) / bar_range

        # Bar direction
        is_up_bar   = close > open_
        is_down_bar = close < open_

        # Entry conditions
        # Short: up bar + volume spike + closes near low (supply entering)
        short_signal = (
            is_up_bar and
            volume_spike and
            close_position <= self.close_threshold
        )

        # Long: down bar + volume spike + closes near high (demand entering)
        long_signal = (
            is_down_bar and
            volume_spike and
            close_position >= (1 - self.close_threshold)
        )

        # ATR-based stop and target
        atr = float(ta_lib.volatility.average_true_range(
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
            pd.Series(self.data.Close),
            window=14
        ).iloc[-1])

        if self.position.is_long:
            # Exit long: up bar volume spike (distribution) or ATR target hit
            if short_signal:
                self.position.close()

        elif self.position.is_short:
            # Exit short: down bar volume spike (accumulation) or ATR target hit
            if long_signal:
                self.position.close()

        else:
            if short_signal:
                sl = close + (atr * 1.5)
                tp = close - (atr * 3.0)
                self.sell(size=self.size, sl=sl, tp=tp)

            elif long_signal:
                sl = close - (atr * 1.5)
                tp = close + (atr * 3.0)
                self.buy(size=self.size, sl=sl, tp=tp)


def run(symbol: str = "MES", timeframe: str = "15m",
        days_back: int = 1825) -> dict:
    """Run VolumeSpikeFade on fresh data to verify it still works."""
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 VolumeSpikeFade — {symbol} {timeframe} ({days_back} days)")

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
        "strategy":     "VolumeSpikeFade",
        "symbol":       symbol,
        "timeframe":    timeframe,
        "return_pct":   round(float(stats["Return [%]"]), 2),
        "buy_hold_pct": round(float(stats["Buy & Hold Return [%]"]), 2),
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
    p = argparse.ArgumentParser(description="📊 VolumeSpikeFade")
    p.add_argument("--symbol",  default="MES")
    p.add_argument("--tf",      default="15m")
    p.add_argument("--days",    type=int, default=1825)
    args = p.parse_args()
    run(args.symbol, args.tf, args.days)
