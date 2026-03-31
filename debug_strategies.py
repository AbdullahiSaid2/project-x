"""
Debug script — checks each condition independently on real data
to find exactly why zero trades are firing.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import pytz
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.fetcher import get_ohlcv
from src.config import EXCHANGE

print("Loading MNQ 5m data...")
df = get_ohlcv("MNQ", exchange=EXCHANGE, timeframe="5m", days_back=90)
df = pd.DataFrame({
    "Open":   df["Open"].astype(float).values,
    "High":   df["High"].astype(float).values,
    "Low":    df["Low"].astype(float).values,
    "Close":  df["Close"].astype(float).values,
    "Volume": df["Volume"].astype(float).values,
}, index=df.index)

print(f"Loaded {len(df)} candles")
print(f"Date range: {df.index[0]} → {df.index[-1]}\n")

EST = pytz.timezone("America/New_York")

# ── Kill zone check ───────────────────────────────────────────
def in_kill_zone(ts):
    try:
        if ts.tzinfo is None:
            ts = pd.Timestamp(ts).tz_localize("UTC")
        est = ts.astimezone(EST)
        hm  = est.hour * 60 + est.minute
        asian_kz  = hm >= 1200
        london_kz = 120  <= hm < 300
        ny_am_kz  = 510  <= hm < 660
        ny_pm_kz  = 810  <= hm < 960
        macro_0950 = 590 <= hm < 610
        macro_1050 = 650 <= hm < 670
        macro_1350 = 830 <= hm < 850
        macro_1550 = 950 <= hm < 970
        macro_0250 = 170 <= hm < 190
        macro_0350 = 230 <= hm < 250
        return any([asian_kz, london_kz, ny_am_kz, ny_pm_kz,
                    macro_0950, macro_1050, macro_1350, macro_1550,
                    macro_0250, macro_0350])
    except:
        return False

kz_bars = sum(1 for ts in df.index if in_kill_zone(ts))
print(f"Kill zone check:")
print(f"  Total bars      : {len(df)}")
print(f"  In kill zone    : {kz_bars} ({kz_bars/len(df)*100:.1f}%)")
print(f"  Outside KZ      : {len(df)-kz_bars}")

# Check a sample of timestamps
print(f"\n  Sample EST times in kill zone:")
count = 0
for ts in df.index:
    if in_kill_zone(ts) and count < 5:
        est = pd.Timestamp(ts).tz_localize("UTC").astimezone(EST) if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts).astimezone(EST)
        print(f"    {est.strftime('%Y-%m-%d %H:%M EST')}")
        count += 1

# ── Sweep detection ───────────────────────────────────────────
print(f"\nSweep detection (20-bar lookback):")
sweep_highs = 0
sweep_lows  = 0

for i in range(25, len(df)):
    prior_highs = df['High'].iloc[i-21:i-1].values
    prior_lows  = df['Low'].iloc[i-21:i-1].values
    curr_high   = df['High'].iloc[i]
    curr_low    = df['Low'].iloc[i]
    curr_close  = df['Close'].iloc[i]

    if curr_high > prior_highs.max() and curr_close < prior_highs.max():
        sweep_highs += 1
    if curr_low < prior_lows.min() and curr_close > prior_lows.min():
        sweep_lows  += 1

print(f"  Bearish sweep (wick above, close below 20-bar high): {sweep_highs}")
print(f"  Bullish sweep (wick below, close above 20-bar low) : {sweep_lows}")

# ── CISD detection ────────────────────────────────────────────
print(f"\nCISD detection (seq_lookback=4):")
cisd_plus  = 0
cisd_minus = 0

for i in range(10, len(df)):
    closes = df['Close'].iloc[i-5:i+1].values
    opens  = df['Open'].iloc[i-5:i+1].values

    # Find down sequence ending before current bar
    count_down = 0
    for j in range(1, 5):
        if closes[-j-1] < closes[-j-2]:
            count_down = j
        else:
            break

    # Find up sequence ending before current bar
    count_up = 0
    for j in range(1, 5):
        if closes[-j-1] > closes[-j-2]:
            count_up = j
        else:
            break

    if count_down >= 1:
        first_down_open = opens[-count_down-1]
        if closes[-1] > first_down_open:
            cisd_plus += 1

    if count_up >= 1:
        first_up_open = opens[-count_up-1]
        if closes[-1] < first_up_open:
            cisd_minus += 1

print(f"  CISD+ (close above first down candle open): {cisd_plus}")
print(f"  CISD- (close below first up candle open)  : {cisd_minus}")

# ── Combined conditions ───────────────────────────────────────
print(f"\nCombined — sweep + CISD (no MSS filter):")
long_signals  = 0
short_signals = 0

for i in range(25, len(df)):
    if not in_kill_zone(df.index[i]):
        continue

    prior_highs = df['High'].iloc[i-21:i-1].values
    prior_lows  = df['Low'].iloc[i-21:i-1].values
    closes = df['Close'].iloc[i-6:i+1].values
    opens  = df['Open'].iloc[i-6:i+1].values
    curr_high  = df['High'].iloc[i]
    curr_low   = df['Low'].iloc[i]
    curr_close = df['Close'].iloc[i]

    sweep_high = prior_highs.max()
    sweep_low  = prior_lows.min()

    # Bearish sweep rejection
    bearish_sweep = curr_high > sweep_high and curr_close < sweep_high

    # Bullish sweep rejection
    bullish_sweep = curr_low < sweep_low and curr_close > sweep_low

    # CISD-
    count_up = 0
    for j in range(1, 5):
        if len(closes) > j+1 and closes[-j-1] > closes[-j-2]:
            count_up = j
        else:
            break
    cisd_short = count_up >= 1 and curr_close < opens[-count_up-1]

    # CISD+
    count_down = 0
    for j in range(1, 5):
        if len(closes) > j+1 and closes[-j-1] < closes[-j-2]:
            count_down = j
        else:
            break
    cisd_long = count_down >= 1 and curr_close > opens[-count_down-1]

    if bearish_sweep and cisd_short:
        short_signals += 1
    if bullish_sweep and cisd_long:
        long_signals += 1

print(f"  Long signals  (bullish sweep + CISD+): {long_signals}")
print(f"  Short signals (bearish sweep + CISD-): {short_signals}")
print(f"\n  → If these are > 0 but trades = 0, the MSS filter is too strict")
print(f"  → If these are 0, the sweep+CISD conditions need adjustment")
