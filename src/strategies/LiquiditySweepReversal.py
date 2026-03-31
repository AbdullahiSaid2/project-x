# ============================================================
# 🌙 Liquidity Sweep Reversal (Hypomeno / ICT)
# Symbol: MNQ | Timeframe: 5m
# Kill zones: Asian, London, NY AM, NY PM + Macros
#
# LOGIC:
#   Long:  new 20-bar low swept, bullish CISD+ (close above
#          open of first down candle in prior sequence)
#   Short: new 20-bar high swept, bearish CISD- (close below
#          open of first up candle in prior sequence)
#   Exits managed manually to avoid backtesting.py order issues
# ============================================================

import sys, warnings
import numpy as np
import pandas as pd
import ta as ta_lib
import ta
from pathlib import Path
from datetime import datetime
from backtesting import Backtest, Strategy

warnings.filterwarnings("ignore", category=UserWarning,    module="backtesting")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="backtesting")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class VaultStrategy(Strategy):
    sweep_lookback = 20
    seq_lookback   = 5
    atr_period     = 14
    atr_sl_mult    = 1.0
    atr_tp_mult    = 2.5
    size           = 0.1

    def init(self):
        self.atr = self.I(
            lambda h, l, c: ta_lib.volatility.average_true_range(
                pd.Series(h), pd.Series(l), pd.Series(c),
                window=self.atr_period
            ).values,
            self.data.High, self.data.Low, self.data.Close,
            name="ATR"
        )
        # Manual SL/TP tracking — avoids exclusive_orders cancellation bug
        self._sl = 0.0
        self._tp = 0.0

    def _in_kill_zone(self):
        try:
            import pytz
            ts  = self.data.index[-1]
            ts  = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            est = ts.astimezone(pytz.timezone("America/New_York"))
            hm  = est.hour * 60 + est.minute
            return any([
                hm >= 1200,           # Asian KZ 20:00-00:00
                120  <= hm < 300,     # London KZ 02:00-05:00
                510  <= hm < 660,     # NY AM 08:30-11:00
                810  <= hm < 960,     # NY PM 13:30-16:00
                590  <= hm < 610,     # Macro 09:50-10:10 (4H open)
                650  <= hm < 670,     # Macro 10:50-11:10
                830  <= hm < 850,     # Macro 13:50-14:10
                170  <= hm < 190,     # London Macro 02:50-03:10
                230  <= hm < 250,     # London Macro 03:50-04:10
            ])
        except Exception:
            return True

    def next(self):
        close = float(self.data.Close[-1])
        high  = float(self.data.High[-1])
        low   = float(self.data.Low[-1])

        # ── Manage open position manually ─────────────────────
        if self.position.is_long:
            if low  <= self._sl:
                self.position.close()
            elif high >= self._tp:
                self.position.close()
            return

        if self.position.is_short:
            if high >= self._sl:
                self.position.close()
            elif low  <= self._tp:
                self.position.close()
            return

        # ── Entry checks ──────────────────────────────────────
        if len(self.data.Close) < self.sweep_lookback + self.seq_lookback + 3:
            return
        if not self._in_kill_zone():
            return

        atr = float(self.atr[-1])
        if np.isnan(atr) or atr == 0:
            return

        # Sweep levels from prior bars
        ph = np.array([float(self.data.High[-i]) for i in range(2, self.sweep_lookback + 2)])
        pl = np.array([float(self.data.Low[-i])  for i in range(2, self.sweep_lookback + 2)])
        sweep_high = ph.max()
        sweep_low  = pl.min()

        bearish_sweep = high > sweep_high and close < sweep_high
        bullish_sweep = low  < sweep_low  and close > sweep_low

        # CISD — find first candle open in directional sequence
        n = min(self.seq_lookback, len(self.data.Close) - 2)
        first_up_open = first_down_open = None
        in_up = in_dn = False

        for j in range(1, n + 1):
            if float(self.data.Close[-j]) > float(self.data.Close[-j-1]):
                first_up_open = float(self.data.Open[-j])
                in_up = True
            elif in_up:
                break

        for j in range(1, n + 1):
            if float(self.data.Close[-j]) < float(self.data.Close[-j-1]):
                first_down_open = float(self.data.Open[-j])
                in_dn = True
            elif in_dn:
                break

        cisd_long  = first_down_open is not None and close > first_down_open
        cisd_short = first_up_open   is not None and close < first_up_open

        # ── Execute — simple market order, manual SL/TP ───────
        if bullish_sweep and cisd_long:
            sl   = sweep_low - (atr * self.atr_sl_mult)
            risk = close - sl
            if risk <= 0:
                return
            self._sl = sl
            self._tp = close + risk * self.atr_tp_mult
            self.buy(size=self.size)

        elif bearish_sweep and cisd_short:
            sl   = sweep_high + (atr * self.atr_sl_mult)
            risk = sl - close
            if risk <= 0:
                return
            self._sl = sl
            self._tp = close - risk * self.atr_tp_mult
            self.sell(size=self.size)


def run(symbol="MNQ", timeframe="5m", days_back=1825):
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\n📊 Liquidity Sweep Reversal — {symbol} {timeframe} ({days_back} days)")
    df = get_ohlcv(symbol, exchange=EXCHANGE, timeframe=timeframe, days_back=days_back)
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
        "strategy":     "LiquiditySweepReversal",
        "symbol":       symbol, "timeframe": timeframe,
        "return_pct":   round(float(stats["Return [%]"]), 2),
        "max_drawdown": round(float(stats["Max. Drawdown [%]"]), 2),
        "sharpe":       round(float(stats.get("Sharpe Ratio", 0) or 0), 3),
        "num_trades":   int(stats["# Trades"]),
        "win_rate":     round(float(stats.get("Win Rate [%]", 0) or 0), 2),
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
    p.add_argument("--symbol", default="MNQ")
    p.add_argument("--tf",     default="5m")
    p.add_argument("--days",   type=int, default=1825)
    p.add_argument("--all",    action="store_true")
    args = p.parse_args()
    if args.all:
        for s in ["MES", "MNQ", "MYM"]:
            run(s, args.tf, args.days)
    else:
        run(args.symbol, args.tf, args.days)