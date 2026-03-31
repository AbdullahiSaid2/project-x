"""
Runs the strategy logic INSIDE backtesting.py to see exactly
why signals aren't converting to trades.
"""
import sys, warnings, numpy as np, pandas as pd, pytz
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtesting import Backtest, Strategy
from src.data.fetcher import get_ohlcv
from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION
import ta as ta_lib

EST = pytz.timezone("America/New_York")

print("Loading MNQ 5m (90 days)...")
df = get_ohlcv("MNQ", exchange=EXCHANGE, timeframe="5m", days_back=90)
df = pd.DataFrame({
    "Open":   df["Open"].astype(float).values,
    "High":   df["High"].astype(float).values,
    "Low":    df["Low"].astype(float).values,
    "Close":  df["Close"].astype(float).values,
    "Volume": df["Volume"].astype(float).values,
}, index=df.index)
print(f"Loaded {len(df)} bars\n")

class DiagStrategy(Strategy):
    sweep_lookback = 20
    seq_lookback   = 5
    atr_period     = 14
    size           = 0.1

    counts = {
        'bars_seen': 0, 'in_kz': 0,
        'bearish_sweep': 0, 'bullish_sweep': 0,
        'cisd_short': 0, 'cisd_long': 0,
        'short_signal': 0, 'long_signal': 0,
        'tp_fail_short': 0, 'tp_fail_long': 0,
        'orders_placed': 0,
    }
    signal_log = []

    def init(self):
        self.atr = self.I(
            lambda h, l, c: ta_lib.volatility.average_true_range(
                pd.Series(h), pd.Series(l), pd.Series(c), window=14
            ).values,
            self.data.High, self.data.Low, self.data.Close,
            name="ATR"
        )

    def _in_kz(self):
        try:
            ts = self.data.index[-1]
            if hasattr(ts, 'tzinfo') and ts.tzinfo is None:
                ts = pd.Timestamp(ts).tz_localize("UTC")
            else:
                ts = pd.Timestamp(ts)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
            est = ts.astimezone(EST)
            hm  = est.hour * 60 + est.minute
            return any([
                hm >= 1200,            # Asian
                120 <= hm < 300,       # London
                510 <= hm < 660,       # NY AM
                810 <= hm < 960,       # NY PM
                590 <= hm < 610,       # Macro 09:50
                650 <= hm < 670,       # Macro 10:50
            ])
        except Exception as e:
            return True   # don't block on error

    def next(self):
        DiagStrategy.counts['bars_seen'] += 1

        if len(self.data.Close) < self.sweep_lookback + self.seq_lookback + 3:
            return

        in_kz = self._in_kz()
        if in_kz:
            DiagStrategy.counts['in_kz'] += 1
        else:
            return

        close = float(self.data.Close[-1])
        high  = float(self.data.High[-1])
        low   = float(self.data.Low[-1])
        atr   = float(self.atr[-1])

        if np.isnan(atr) or atr == 0:
            return

        # Sweep levels
        ph = np.array([float(self.data.High[-i]) for i in range(2, self.sweep_lookback + 2)])
        pl = np.array([float(self.data.Low[-i])  for i in range(2, self.sweep_lookback + 2)])
        sweep_high = ph.max()
        sweep_low  = pl.min()

        bearish_sweep = high > sweep_high and close < sweep_high
        bullish_sweep = low  < sweep_low  and close > sweep_low

        if bearish_sweep: DiagStrategy.counts['bearish_sweep'] += 1
        if bullish_sweep: DiagStrategy.counts['bullish_sweep'] += 1

        # CISD
        n = min(self.seq_lookback, len(self.data.Close) - 2)
        first_up_open = first_down_open = None

        in_up = in_dn = False
        for j in range(1, n + 1):
            c_j   = float(self.data.Close[-j])
            c_j1  = float(self.data.Close[-j-1])
            o_j   = float(self.data.Open[-j])
            if c_j > c_j1:
                first_up_open = o_j; in_up = True
            elif in_up:
                break

        in_dn = False
        for j in range(1, n + 1):
            c_j   = float(self.data.Close[-j])
            c_j1  = float(self.data.Close[-j-1])
            o_j   = float(self.data.Open[-j])
            if c_j < c_j1:
                first_down_open = o_j; in_dn = True
            elif in_dn:
                break

        cisd_short = first_up_open   is not None and close < first_up_open
        cisd_long  = first_down_open is not None and close > first_down_open

        if cisd_short: DiagStrategy.counts['cisd_short'] += 1
        if cisd_long:  DiagStrategy.counts['cisd_long']  += 1

        short_sig = bearish_sweep and cisd_short
        long_sig  = bullish_sweep and cisd_long

        if short_sig: DiagStrategy.counts['short_signal'] += 1
        if long_sig:  DiagStrategy.counts['long_signal']  += 1

        # Execute
        if not self.position:
            if short_sig:
                sl   = sweep_high + atr
                risk = sl - close
                if risk <= 0:
                    DiagStrategy.counts['tp_fail_short'] += 1
                    return
                tp = close - risk * 2.5
                DiagStrategy.signal_log.append({
                    'dir':'SHORT','close':round(close,2),
                    'sl':round(sl,2),'tp':round(tp,2),
                    'valid': tp < close < sl
                })
                if tp < close < sl:
                    DiagStrategy.counts['orders_placed'] += 1
                    self.sell(size=self.size, sl=sl, tp=tp)
                else:
                    DiagStrategy.counts['tp_fail_short'] += 1

            elif long_sig:
                sl   = sweep_low - atr
                risk = close - sl
                if risk <= 0:
                    DiagStrategy.counts['tp_fail_long'] += 1
                    return
                tp = close + risk * 2.5
                DiagStrategy.signal_log.append({
                    'dir':'LONG','close':round(close,2),
                    'sl':round(sl,2),'tp':round(tp,2),
                    'valid': sl < close < tp
                })
                if sl < close < tp:
                    DiagStrategy.counts['orders_placed'] += 1
                    self.buy(size=self.size, sl=sl, tp=tp)
                else:
                    DiagStrategy.counts['tp_fail_long'] += 1

bt = Backtest(df, DiagStrategy,
              cash=BACKTEST_INITIAL_CASH,
              commission=BACKTEST_COMMISSION,
              exclusive_orders=True)
stats = bt.run()

print("=== DIAGNOSTIC RESULTS ===")
for k, v in DiagStrategy.counts.items():
    print(f"  {k:<20}: {v}")

print(f"\n  Actual trades placed: {stats['# Trades']}")

print(f"\n  Signal log (first 10):")
for s in DiagStrategy.signal_log[:10]:
    valid = "✅" if s['valid'] else "❌"
    print(f"    {valid} {s['dir']:<6} close={s['close']} sl={s['sl']} tp={s['tp']}")
