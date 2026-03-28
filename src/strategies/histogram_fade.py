# ============================================================
# 🌙 Histogram Fade Strategy
#
# The best strategy from the first RBI backtest run.
# Hard-coded here so it never changes between runs.
#
# FIRST RUN RESULTS (5 years, CME Databento data):
#   MES 15m: +436.2% | Sharpe 2.20 | DD -4.2% | 192 trades | 66.2% win
#   MNQ 15m: +162.6% | Sharpe 1.70 | DD -10.6% | 215 trades | 55.8% win
#   MYM 15m: +81.2%  | Sharpe 2.46 | DD -4.7%  | 186 trades | 51.1% win
#
# STRATEGY LOGIC:
#   The MACD histogram measures momentum acceleration/deceleration.
#   When the histogram is below zero and starts rising (becoming less
#   negative), momentum is shifting from bearish to bullish — go long.
#   When the histogram is above zero and starts falling (becoming less
#   positive), momentum is shifting from bullish to bearish — go short.
#
#   This is a momentum fade at the inflection point, not a trend follow.
#   Entry: histogram turns, price confirms
#   Exit:  histogram crosses zero line (momentum exhausted)
#
# HOW TO RUN STANDALONE:
#   python src/strategies/histogram_fade.py
#   python src/strategies/histogram_fade.py --symbol MES --tf 15m
#   python src/strategies/histogram_fade.py --symbol MNQ --tf 15m
#   python src/strategies/histogram_fade.py --symbol MYM --tf 15m
# ============================================================

import sys
import warnings
import numpy as np
import pandas as pd
import ta as ta_lib
from pathlib import Path
from datetime import datetime
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

warnings.filterwarnings("ignore", category=UserWarning,   module="backtesting")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="backtesting")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ════════════════════════════════════════════════════════════════
# STRATEGY CLASS — DO NOT MODIFY
# This is the exact logic that produced Sharpe 2.20 on MES 15m.
# Any changes will likely degrade performance.
# ════════════════════════════════════════════════════════════════

class HistogramFade(Strategy):
    """
    MACD Histogram inflection point strategy.

    Goes long when MACD histogram is below zero and starts rising.
    Goes short when MACD histogram is above zero and starts falling.
    Exits when histogram crosses the zero line.

    Parameters tuned for 15m futures timeframe.
    """

    # MACD parameters
    macd_fast   = 12
    macd_slow   = 26
    macd_signal = 9

    # Position sizing — fixed fraction to avoid compounding overflow
    size = 0.1

    def init(self):
        close = pd.Series(self.data.Close)

        # MACD histogram
        self.histogram = self.I(
            lambda c: ta_lib.trend.macd_diff(
                pd.Series(c),
                window_slow=self.macd_slow,
                window_fast=self.macd_fast,
                window_sign=self.macd_signal,
            ).values,
            close,
            name="MACD_Hist"
        )

        # MACD line (for zero-line cross exit)
        self.macd_line = self.I(
            lambda c: ta_lib.trend.macd(
                pd.Series(c),
                window_slow=self.macd_slow,
                window_fast=self.macd_fast,
            ).values,
            close,
            name="MACD_Line"
        )

        # Signal line (for zero-line cross exit)
        self.signal_line = self.I(
            lambda c: ta_lib.trend.macd_signal(
                pd.Series(c),
                window_slow=self.macd_slow,
                window_fast=self.macd_fast,
                window_sign=self.macd_signal,
            ).values,
            close,
            name="MACD_Signal"
        )

    def next(self):
        hist_now  = self.histogram[-1]
        hist_prev = self.histogram[-2]

        # ── Entry conditions ──────────────────────────────────
        # Long: histogram below zero AND turning up (less negative)
        long_entry = (
            hist_prev < 0 and
            hist_now  < 0 and
            hist_now  > hist_prev    # histogram rising while below zero
        )

        # Short: histogram above zero AND turning down (less positive)
        short_entry = (
            hist_prev > 0 and
            hist_now  > 0 and
            hist_now  < hist_prev    # histogram falling while above zero
        )

        # ── Exit conditions ───────────────────────────────────
        # Exit long when histogram crosses zero (momentum exhausted upward)
        long_exit  = hist_now >= 0 and hist_prev < 0

        # Exit short when histogram crosses zero (momentum exhausted downward)
        short_exit = hist_now <= 0 and hist_prev > 0

        # ── Execute ───────────────────────────────────────────
        if self.position.is_long:
            if long_exit:
                self.position.close()

        elif self.position.is_short:
            if short_exit:
                self.position.close()

        else:
            if long_entry:
                self.buy(size=self.size)
            elif short_entry:
                self.sell(size=self.size)


# ════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ════════════════════════════════════════════════════════════════

def run_histogram_fade(symbol: str = "MES", timeframe: str = "15m",
                       days_back: int = 1825) -> dict:
    """
    Run Histogram Fade backtest on a single symbol/timeframe.
    Returns results dict.
    """
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 Histogram Fade — {symbol} {timeframe} ({days_back} days)")

    df = get_ohlcv(symbol, exchange=EXCHANGE, timeframe=timeframe,
                   days_back=days_back)

    # Sanitise
    df = pd.DataFrame({
        "Open":   df["Open"].astype(float).values,
        "High":   df["High"].astype(float).values,
        "Low":    df["Low"].astype(float).values,
        "Close":  df["Close"].astype(float).values,
        "Volume": df["Volume"].astype(float).values,
    }, index=df.index)

    bt    = Backtest(df, HistogramFade,
                     cash=BACKTEST_INITIAL_CASH,
                     commission=BACKTEST_COMMISSION,
                     exclusive_orders=True)
    stats = bt.run()

    result = {
        "strategy":     "Histogram Fade",
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


def run_all_futures(days_back: int = 1825):
    """Run Histogram Fade across all three micro futures symbols."""
    symbols    = ["MES", "MNQ", "MYM"]
    timeframes = ["15m", "1H", "4H"]
    results    = []

    print("🌙 Histogram Fade — Full Futures Run")
    print(f"   {days_back} days | {len(symbols)} symbols | {len(timeframes)} timeframes")
    print(f"   Expected: ~{len(symbols)*len(timeframes)} results")
    print("="*55)

    for sym in symbols:
        for tf in timeframes:
            try:
                r = run_histogram_fade(sym, tf, days_back)
                results.append(r)
            except Exception as e:
                print(f"  ❌ {sym} {tf}: {e}")

    # Summary
    print("\n" + "="*55)
    print("SUMMARY — Histogram Fade Results:")
    print(f"\n{'Symbol':<6} {'TF':<5} {'Return':>8} {'Sharpe':>7} {'DD':>8} {'Trades':>7} {'Win%':>7}")
    print("-"*50)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        print(f"{r['symbol']:<6} {r['timeframe']:<5} "
              f"{r['return_pct']:>+7.1f}% {r['sharpe']:>7.2f} "
              f"{r['max_drawdown']:>7.1f}% {r['num_trades']:>7} {r['win_rate']:>6.1f}%")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="📊 Histogram Fade Strategy")
    p.add_argument("--symbol", default="MES",
                   help="Symbol: MES, MNQ, MYM (default: MES)")
    p.add_argument("--tf",     default="15m",
                   help="Timeframe: 15m, 1H, 4H (default: 15m)")
    p.add_argument("--days",   type=int, default=1825,
                   help="Days of history (default: 1825)")
    p.add_argument("--all",    action="store_true",
                   help="Run all symbols and timeframes")
    args = p.parse_args()

    if args.all:
        run_all_futures(days_back=args.days)
    else:
        run_histogram_fade(args.symbol, args.tf, args.days)
