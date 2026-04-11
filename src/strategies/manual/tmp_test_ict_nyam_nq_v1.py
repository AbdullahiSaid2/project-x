from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backtesting import Backtest
from src.data.fetcher import get_ohlcv
from src.strategies.manual.ict_nyam_nq_model_v1 import (
    ICT_NYAM_NQ_Model_V1,
    build_model_frame,
    load_external_market,
)

symbol = "NQ"
timeframe = "3m"
days_back = 90
tail_rows = 15000

print("1) Loading NQ data...")
df = get_ohlcv(symbol, exchange="tradovate", timeframe=timeframe, days_back=days_back)
df = df.tail(tail_rows)
print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()}")

print("2) Creating backtest...")
bt = Backtest(
    df,
    ICT_NYAM_NQ_Model_V1,
    cash=1_000_000,
    commission=0.0,
    exclusive_orders=True,
)

print("3) Running backtest...")
stats = bt.run()

print("4) Done.")
print(stats)

trades = stats.get("_trades")
if trades is not None:
    print(f"\nTrade count: {len(trades)}")
    print(trades.tail(20))

print("\n5) Building diagnostics frame...")

start = df.index.min()
end = df.index.max()

es_30m = load_external_market("ES", start, end)
ym_30m = load_external_market("YM", start, end)

m = build_model_frame(df, es_30m=es_30m, ym_30m=ym_30m)

print(f"Model rows: {len(m)}")

diag_cols = [
    "bull_daily_bias",
    "bear_daily_bias",
    "bull_4h_bias",
    "bear_4h_bias",
    "above_4h_eq",
    "below_4h_eq",
    "bullish_smt",
    "bearish_smt",
    "bull_c2",
    "bear_c2",
    "bull_c3",
    "bear_c3",
    "bull_c2_or_c3",
    "bear_c2_or_c3",
    "swept_session_low",
    "swept_session_high",
    "reclaimed_session_low",
    "rejected_session_high",
    "swept_prev_day_low",
    "swept_prev_day_high",
    "reclaimed_prev_day_low",
    "rejected_prev_day_high",
    "bull_disp_30m",
    "bear_disp_30m",
    "bull_close_strong_30m",
    "bear_close_strong_30m",
    "bull_disp_3m",
    "bear_disp_3m",
    "bull_close_strong_3m",
    "bear_close_strong_3m",
    "is_nyam_valid",
    "is_0930_to_1030",
    "bull_fvg_4h",
    "bear_fvg_4h",
    "bull_disp_4h",
    "bear_disp_4h",
]

for col in diag_cols:
    if col in m.columns:
        count = int(m[col].fillna(False).sum())
        print(f"{col}: {count}")

# Match CURRENT strategy logic exactly
use_smt_filter = False
require_prev_day_context = True

no_clear_bull_dol = m["prev_day_high"].isna() & m["session_high"].isna()
no_clear_bear_dol = m["prev_day_low"].isna() & m["session_low"].isna()

bull_eq_invalid = ~m["above_4h_eq"].fillna(False)
bear_eq_invalid = ~m["below_4h_eq"].fillna(False)

bull_structure_invalid = ~(
    m["bull_fvg_4h"].fillna(False)
    | m["bull_disp_4h"].fillna(False)
    | m["bull_profile_4h"].fillna(False)
)
bear_structure_invalid = ~(
    m["bear_fvg_4h"].fillna(False)
    | m["bear_disp_4h"].fillna(False)
    | m["bear_profile_4h"].fillna(False)
)

weak_bull_delivery = m["weak_bull_delivery_3m"].fillna(False)
weak_bear_delivery = m["weak_bear_delivery_3m"].fillna(False)

wrong_time = ~m["is_nyam_valid"].fillna(False)
late_time = ~m["is_0930_to_1030"].fillna(False)

bull_smt_invalid = False
bear_smt_invalid = False
if use_smt_filter:
    bull_smt_invalid = ~m["bullish_smt"].fillna(False)
    bear_smt_invalid = ~m["bearish_smt"].fillna(False)

bull_narrative = (
    m["bull_4h_bias"].fillna(False)
    & m["above_4h_eq"].fillna(False)
    & (
        m["bull_profile_4h"].fillna(False)
        | m["bull_disp_4h"].fillna(False)
        | m["bull_fvg_4h"].fillna(False)
    )
)

bear_narrative = (
    m["bear_4h_bias"].fillna(False)
    & m["below_4h_eq"].fillna(False)
    & (
        m["bear_profile_4h"].fillna(False)
        | m["bear_disp_4h"].fillna(False)
        | m["bear_fvg_4h"].fillna(False)
    )
)

bull_context = (
    (m["swept_session_low"].fillna(False) & m["reclaimed_session_low"].fillna(False))
    |
    (m["swept_prev_day_low"].fillna(False) & m["reclaimed_prev_day_low"].fillna(False))
)

bear_context = (
    (m["swept_session_high"].fillna(False) & m["rejected_session_high"].fillna(False))
    |
    (m["swept_prev_day_high"].fillna(False) & m["rejected_prev_day_high"].fillna(False))
)

bull_prevday_invalid = False
bear_prevday_invalid = False
if require_prev_day_context:
    bull_prevday_invalid = ~(
        m["swept_prev_day_low"].fillna(False) | m["swept_session_low"].fillna(False)
    )
    bear_prevday_invalid = ~(
        m["swept_prev_day_high"].fillna(False) | m["swept_session_high"].fillna(False)
    )

bull_bridge = (
    m["bull_c2_or_c3"].fillna(False)
    & m["bull_disp_30m"].fillna(False)
    & m["bull_close_strong_30m"].fillna(False)
)

bear_bridge = (
    m["bear_c2_or_c3"].fillna(False)
    & m["bear_disp_30m"].fillna(False)
    & m["bear_close_strong_30m"].fillna(False)
)

bull_execution = (
    m["is_0930_to_1030"].fillna(False)
    & m["bull_disp_3m"].fillna(False)
    & m["bull_close_strong_3m"].fillna(False)
)

bear_execution = (
    m["is_0930_to_1030"].fillna(False)
    & m["bear_disp_3m"].fillna(False)
    & m["bear_close_strong_3m"].fillna(False)
)

bull_invalid = (
    no_clear_bull_dol
    | bull_eq_invalid
    | bull_structure_invalid
    | weak_bull_delivery
    | wrong_time
    | late_time
    | bull_smt_invalid
    | bull_prevday_invalid
)

bear_invalid = (
    no_clear_bear_dol
    | bear_eq_invalid
    | bear_structure_invalid
    | weak_bear_delivery
    | wrong_time
    | late_time
    | bear_smt_invalid
    | bear_prevday_invalid
)

long_setup = bull_narrative & bull_context & bull_bridge & bull_execution & ~bull_invalid
short_setup = bear_narrative & bear_context & bear_bridge & bear_execution & ~bear_invalid

print(f"\nBull narrative bars: {int(bull_narrative.sum())}")
print(f"Bear narrative bars: {int(bear_narrative.sum())}")
print(f"Bull context bars: {int(bull_context.sum())}")
print(f"Bear context bars: {int(bear_context.sum())}")
print(f"Bull bridge bars: {int(bull_bridge.sum())}")
print(f"Bear bridge bars: {int(bear_bridge.sum())}")
print(f"Bull execution bars: {int(bull_execution.sum())}")
print(f"Bear execution bars: {int(bear_execution.sum())}")
print(f"Bull invalid bars: {int(bull_invalid.sum())}")
print(f"Bear invalid bars: {int(bear_invalid.sum())}")

print(f"\nLong setup bars after invalidation stack: {int(long_setup.sum())}")
print(f"Short setup bars after invalidation stack: {int(short_setup.sum())}")