from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from backtesting import Strategy


# ============================================================
# ICT_MULTI_SETUP_V45.2
#
# V45.2 goals:
# 1. Keep V45 pullback + partial + runner structure.
# 2. Prune weaker flow.
# 3. Rank setups for prop-mode trading.
# 4. Restrict CISD to stronger contexts.
# 5. Tighten NYPM and keep NYAM improvements.
# ============================================================


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def ema_cross_up(fast: pd.Series, slow: pd.Series) -> pd.Series:
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def ema_cross_down(fast: pd.Series, slow: pd.Series) -> pd.Series:
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


def detect_ttrades_c2_c3(bars_30m: pd.DataFrame) -> pd.DataFrame:
    out = bars_30m.copy()
    prev_high = out["High"].shift(1)
    prev_low = out["Low"].shift(1)

    out["bull_c2"] = (
        (out["Low"] < prev_low)
        & (out["Close"] > prev_low)
        & (out["Close"] < prev_high)
    )
    out["bear_c2"] = (
        (out["High"] > prev_high)
        & (out["Close"] < prev_high)
        & (out["Close"] > prev_low)
    )

    bull_c2_failed = ((out["Low"] < prev_low) & ~out["bull_c2"]).astype(bool)
    bear_c2_failed = ((out["High"] > prev_high) & ~out["bear_c2"]).astype(bool)

    prev2_high = out["High"].shift(2)
    prev2_low = out["Low"].shift(2)

    out["bull_c3"] = (
        bull_c2_failed.shift(1, fill_value=False)
        & (out["Close"] > prev2_low)
        & (out["Close"] < prev2_high)
        & (out["Close"] > out["Open"])
    )
    out["bear_c3"] = (
        bear_c2_failed.shift(1, fill_value=False)
        & (out["Close"] < prev2_high)
        & (out["Close"] > prev2_low)
        & (out["Close"] < out["Open"])
    )

    out["bull_c2_or_c3"] = out["bull_c2"] | out["bull_c3"]
    out["bear_c2_or_c3"] = out["bear_c2"] | out["bear_c3"]
    return out


def build_4h_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bars_4h = df.resample("4h").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()

    bars_4h["ema20_4h"] = ema(bars_4h["Close"], 20)
    bars_4h["ema50_4h"] = ema(bars_4h["Close"], 50)
    bars_4h["eq_4h"] = (bars_4h["High"].rolling(8).max() + bars_4h["Low"].rolling(8).min()) / 2.0

    bars_4h["atr14_4h"] = atr(bars_4h, 14)
    body_4h = (bars_4h["Close"] - bars_4h["Open"]).abs()
    rng_4h = (bars_4h["High"] - bars_4h["Low"]).replace(0, np.nan)

    bars_4h["bull_disp_4h"] = (bars_4h["Close"] > bars_4h["Open"]) & (body_4h > bars_4h["atr14_4h"] * 0.8)
    bars_4h["bear_disp_4h"] = (bars_4h["Close"] < bars_4h["Open"]) & (body_4h > bars_4h["atr14_4h"] * 0.8)
    bars_4h["bull_fvg_4h"] = bars_4h["Low"] > bars_4h["High"].shift(2)
    bars_4h["bear_fvg_4h"] = bars_4h["High"] < bars_4h["Low"].shift(2)
    bars_4h["bull_profile_4h"] = (
        (bars_4h["Close"] > bars_4h["Open"])
        & (((bars_4h["Close"] - bars_4h["Low"]) / rng_4h) >= 0.60)
    )
    bars_4h["bear_profile_4h"] = (
        (bars_4h["Close"] < bars_4h["Open"])
        & (((bars_4h["High"] - bars_4h["Close"]) / rng_4h) >= 0.60)
    )

    aligned = bars_4h[[
        "ema20_4h", "ema50_4h", "eq_4h",
        "bull_disp_4h", "bear_disp_4h",
        "bull_fvg_4h", "bear_fvg_4h",
        "bull_profile_4h", "bear_profile_4h",
    ]].reindex(out.index, method="ffill")

    for col in aligned.columns:
        out[col] = aligned[col]

    out["bull_4h_bias"] = out["ema20_4h"] > out["ema50_4h"]
    out["bear_4h_bias"] = out["ema20_4h"] < out["ema50_4h"]
    out["above_4h_eq"] = out["Close"] > out["eq_4h"]
    out["below_4h_eq"] = out["Close"] < out["eq_4h"]
    return out


def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx_utc = pd.DatetimeIndex(out.index).tz_localize("UTC")
    idx_et = idx_utc.tz_convert("America/New_York")

    out["et_date"] = idx_et.date
    out["et_hour"] = idx_et.hour
    out["et_minute"] = idx_et.minute

    out["is_asia"] = (out["et_hour"] >= 20) | (out["et_hour"] <= 0)
    out["is_asia_entry_window"] = (out["et_hour"] >= 20) | (out["et_hour"] <= 0)

    out["is_london"] = ((out["et_hour"] >= 2) & (out["et_hour"] <= 5))
    out["is_london_entry_window"] = ((out["et_hour"] >= 2) & (out["et_hour"] <= 4))

    out["is_nyam"] = (
        ((out["et_hour"] == 9) & (out["et_minute"] >= 30))
        | (out["et_hour"] == 10)
        | ((out["et_hour"] == 11) & (out["et_minute"] == 0))
    )
    out["is_nyam_entry_window"] = (
        ((out["et_hour"] == 9) & (out["et_minute"] >= 36))
        | (out["et_hour"] == 10)
        | ((out["et_hour"] == 11) & (out["et_minute"] <= 0))
    )

    out["is_nypm"] = (
        ((out["et_hour"] == 13) & (out["et_minute"] >= 30))
        | (out["et_hour"] == 14)
        | ((out["et_hour"] == 15) & (out["et_minute"] == 0))
    )
    out["is_nypm_entry_window"] = (
        ((out["et_hour"] == 13) & (out["et_minute"] >= 36))
        | (out["et_hour"] == 14)
        | ((out["et_hour"] == 15) & (out["et_minute"] <= 0))
    )

    prior_us = out[(out["et_hour"] < 9) | ((out["et_hour"] == 9) & (out["et_minute"] < 30))].copy()
    prior_us_levels = prior_us.groupby("et_date").agg(prior_us_high=("High", "max"), prior_us_low=("Low", "min"))
    out = out.merge(prior_us_levels, left_on="et_date", right_index=True, how="left")

    asia = out[out["is_asia"]].copy()
    asia_levels = asia.groupby("et_date").agg(asia_high=("High", "max"), asia_low=("Low", "min"))
    out = out.merge(asia_levels, left_on="et_date", right_index=True, how="left")

    london = out[out["is_london"]].copy()
    london_levels = london.groupby("et_date").agg(london_high=("High", "max"), london_low=("Low", "min"))
    out = out.merge(london_levels, left_on="et_date", right_index=True, how="left")

    nyam = out[out["is_nyam"]].copy()
    nyam_levels = nyam.groupby("et_date").agg(nyam_high=("High", "max"), nyam_low=("Low", "min"))
    out = out.merge(nyam_levels, left_on="et_date", right_index=True, how="left")

    day_agg = out.groupby("et_date").agg(day_high=("High", "max"), day_low=("Low", "min"))
    day_agg["prev_day_high"] = day_agg["day_high"].shift(1)
    day_agg["prev_day_low"] = day_agg["day_low"].shift(1)
    out = out.merge(day_agg[["prev_day_high", "prev_day_low"]], left_on="et_date", right_index=True, how="left")

    out["swept_prior_us_low"] = out["Low"] < out["prior_us_low"]
    out["swept_prior_us_high"] = out["High"] > out["prior_us_high"]
    out["reclaimed_prior_us_low"] = out["Close"] > out["prior_us_low"]
    out["rejected_prior_us_high"] = out["Close"] < out["prior_us_high"]

    out["swept_asia_low"] = out["Low"] < out["asia_low"]
    out["swept_asia_high"] = out["High"] > out["asia_high"]
    out["reclaimed_asia_low"] = out["Close"] > out["asia_low"]
    out["rejected_asia_high"] = out["Close"] < out["asia_high"]

    out["swept_london_low"] = out["Low"] < out["london_low"]
    out["swept_london_high"] = out["High"] > out["london_high"]
    out["reclaimed_london_low"] = out["Close"] > out["london_low"]
    out["rejected_london_high"] = out["Close"] < out["london_high"]

    out["swept_prev_day_low"] = out["Low"] < out["prev_day_low"]
    out["swept_prev_day_high"] = out["High"] > out["prev_day_high"]
    out["reclaimed_prev_day_low"] = out["Close"] > out["prev_day_low"]
    out["rejected_prev_day_high"] = out["Close"] < out["prev_day_high"]

    out["swept_nyam_low"] = out["Low"] < out["nyam_low"]
    out["swept_nyam_high"] = out["High"] > out["nyam_high"]
    out["reclaimed_nyam_low"] = out["Close"] > out["nyam_low"]
    out["rejected_nyam_high"] = out["Close"] < out["nyam_high"]
    return out


def build_30m_bridge(df_1m: pd.DataFrame) -> pd.DataFrame:
    out = df_1m.copy()
    bars_30m = df_1m.resample("30min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()

    bars_30m["atr14_30m"] = atr(bars_30m, 14)
    body = (bars_30m["Close"] - bars_30m["Open"]).abs()
    rng = (bars_30m["High"] - bars_30m["Low"]).replace(0, np.nan)

    bars_30m["bull_close_strong_30m"] = (
        (bars_30m["Close"] > bars_30m["Open"]) & (((bars_30m["Close"] - bars_30m["Low"]) / rng) >= 0.65)
    )
    bars_30m["bear_close_strong_30m"] = (
        (bars_30m["Close"] < bars_30m["Open"]) & (((bars_30m["High"] - bars_30m["Close"]) / rng) >= 0.65)
    )
    bars_30m["bull_disp_30m"] = (bars_30m["Close"] > bars_30m["Open"]) & (body > bars_30m["atr14_30m"] * 0.80)
    bars_30m["bear_disp_30m"] = (bars_30m["Close"] < bars_30m["Open"]) & (body > bars_30m["atr14_30m"] * 0.80)

    bars_30m = detect_ttrades_c2_c3(bars_30m)

    fast = ema(bars_30m["Close"], 5)
    slow = ema(bars_30m["Close"], 13)
    bars_30m["bull_cisd_30m"] = ema_cross_up(fast, slow) & (bars_30m["Close"] > bars_30m["Open"])
    bars_30m["bear_cisd_30m"] = ema_cross_down(fast, slow) & (bars_30m["Close"] < bars_30m["Open"])

    bars_30m["swing_high_30m"] = bars_30m["High"].rolling(5).max().shift(1)
    bars_30m["swing_low_30m"] = bars_30m["Low"].rolling(5).min().shift(1)
    bars_30m["bull_mss_30m"] = bars_30m["Close"] > bars_30m["swing_high_30m"]
    bars_30m["bear_mss_30m"] = bars_30m["Close"] < bars_30m["swing_low_30m"]

    bars_30m["bull_ifvg_30m"] = (
        (bars_30m["Low"] > bars_30m["High"].shift(2))
        | ((bars_30m["Close"] > bars_30m["Open"]) & (bars_30m["Low"] > bars_30m["Low"].shift(1)))
    )
    bars_30m["bear_ifvg_30m"] = (
        (bars_30m["High"] < bars_30m["Low"].shift(2))
        | ((bars_30m["Close"] < bars_30m["Open"]) & (bars_30m["High"] < bars_30m["High"].shift(1)))
    )

    bars_30m["bull_bridge_low_30m"] = np.nan
    bars_30m["bull_bridge_high_30m"] = np.nan
    bars_30m["bear_bridge_low_30m"] = np.nan
    bars_30m["bear_bridge_high_30m"] = np.nan
    bars_30m["bridge_type_30m"] = ""

    for i in range(2, len(bars_30m)):
        row = bars_30m.iloc[i]
        if bool(row.get("bull_ifvg_30m", False)):
            lo = max(float(bars_30m.iloc[i - 2]["High"]), float(min(row["Open"], row["Close"])))
            hi = float(row["Low"]) if row["Low"] > bars_30m.iloc[i - 2]["High"] else float(max(row["Open"], row["Close"]))
            lo, hi = min(lo, hi), max(lo, hi)
            bars_30m.iloc[i, bars_30m.columns.get_loc("bull_bridge_low_30m")] = lo
            bars_30m.iloc[i, bars_30m.columns.get_loc("bull_bridge_high_30m")] = hi
            bars_30m.iloc[i, bars_30m.columns.get_loc("bridge_type_30m")] = "iFVG"
        elif bool(row.get("bull_cisd_30m", False)):
            lo = float(min(row["Open"], row["Close"]))
            hi = float(max(row["Open"], row["Close"]))
            bars_30m.iloc[i, bars_30m.columns.get_loc("bull_bridge_low_30m")] = lo
            bars_30m.iloc[i, bars_30m.columns.get_loc("bull_bridge_high_30m")] = hi
            bars_30m.iloc[i, bars_30m.columns.get_loc("bridge_type_30m")] = "CISD"
        elif bool(row.get("bull_mss_30m", False)):
            lo = float(row["Low"])
            hi = float(min(row["Open"], row["Close"]))
            bars_30m.iloc[i, bars_30m.columns.get_loc("bull_bridge_low_30m")] = min(lo, hi)
            bars_30m.iloc[i, bars_30m.columns.get_loc("bull_bridge_high_30m")] = max(lo, hi)
            bars_30m.iloc[i, bars_30m.columns.get_loc("bridge_type_30m")] = "MSS"
        elif bool(row.get("bull_c2_or_c3", False)):
            lo = float(row["Low"])
            hi = float(max(row["Open"], row["Close"]))
            bars_30m.iloc[i, bars_30m.columns.get_loc("bull_bridge_low_30m")] = lo
            bars_30m.iloc[i, bars_30m.columns.get_loc("bull_bridge_high_30m")] = hi
            bars_30m.iloc[i, bars_30m.columns.get_loc("bridge_type_30m")] = "C2C3"

        if bool(row.get("bear_ifvg_30m", False)):
            lo = float(row["High"]) if row["High"] < bars_30m.iloc[i - 2]["Low"] else float(min(row["Open"], row["Close"]))
            hi = min(float(bars_30m.iloc[i - 2]["Low"]), float(max(row["Open"], row["Close"])))
            lo, hi = min(lo, hi), max(lo, hi)
            bars_30m.iloc[i, bars_30m.columns.get_loc("bear_bridge_low_30m")] = lo
            bars_30m.iloc[i, bars_30m.columns.get_loc("bear_bridge_high_30m")] = hi
            bars_30m.iloc[i, bars_30m.columns.get_loc("bridge_type_30m")] = "iFVG"
        elif bool(row.get("bear_cisd_30m", False)):
            lo = float(min(row["Open"], row["Close"]))
            hi = float(max(row["Open"], row["Close"]))
            bars_30m.iloc[i, bars_30m.columns.get_loc("bear_bridge_low_30m")] = lo
            bars_30m.iloc[i, bars_30m.columns.get_loc("bear_bridge_high_30m")] = hi
            bars_30m.iloc[i, bars_30m.columns.get_loc("bridge_type_30m")] = "CISD"
        elif bool(row.get("bear_mss_30m", False)):
            lo = float(min(row["Open"], row["Close"]))
            hi = float(row["High"])
            bars_30m.iloc[i, bars_30m.columns.get_loc("bear_bridge_low_30m")] = min(lo, hi)
            bars_30m.iloc[i, bars_30m.columns.get_loc("bear_bridge_high_30m")] = max(lo, hi)
            bars_30m.iloc[i, bars_30m.columns.get_loc("bridge_type_30m")] = "MSS"
        elif bool(row.get("bear_c2_or_c3", False)):
            lo = float(min(row["Open"], row["Close"]))
            hi = float(row["High"])
            bars_30m.iloc[i, bars_30m.columns.get_loc("bear_bridge_low_30m")] = min(lo, hi)
            bars_30m.iloc[i, bars_30m.columns.get_loc("bear_bridge_high_30m")] = max(lo, hi)
            bars_30m.iloc[i, bars_30m.columns.get_loc("bridge_type_30m")] = "C2C3"

    aligned = bars_30m[[
        "bull_close_strong_30m", "bear_close_strong_30m",
        "bull_disp_30m", "bear_disp_30m",
        "bull_c2_or_c3", "bear_c2_or_c3",
        "bull_cisd_30m", "bear_cisd_30m",
        "bull_mss_30m", "bear_mss_30m",
        "bull_ifvg_30m", "bear_ifvg_30m",
        "bull_bridge_low_30m", "bull_bridge_high_30m",
        "bear_bridge_low_30m", "bear_bridge_high_30m",
        "bridge_type_30m",
    ]].reindex(out.index, method="ffill")

    for col in aligned.columns:
        out[col] = aligned[col]
    return out


def build_3m_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    out = df_1m.copy()
    bars_3m = df_1m.resample("3min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()

    bars_3m["atr14_3m"] = atr(bars_3m, 14)
    body = (bars_3m["Close"] - bars_3m["Open"]).abs()
    rng = (bars_3m["High"] - bars_3m["Low"]).replace(0, np.nan)
    bars_3m["bull_disp_3m"] = (bars_3m["Close"] > bars_3m["Open"]) & (body > bars_3m["atr14_3m"] * 0.60)
    bars_3m["bear_disp_3m"] = (bars_3m["Close"] < bars_3m["Open"]) & (body > bars_3m["atr14_3m"] * 0.60)
    bars_3m["bull_close_strong_3m"] = ((bars_3m["Close"] - bars_3m["Low"]) / rng) >= 0.65
    bars_3m["bear_close_strong_3m"] = ((bars_3m["High"] - bars_3m["Close"]) / rng) >= 0.65
    bars_3m["bull_overextended_3m"] = bars_3m["Close"] > (bars_3m["Close"].rolling(20).mean() + bars_3m["atr14_3m"] * 2.2)
    bars_3m["bear_overextended_3m"] = bars_3m["Close"] < (bars_3m["Close"].rolling(20).mean() - bars_3m["atr14_3m"] * 2.2)

    aligned = bars_3m[[
        "atr14_3m", "bull_disp_3m", "bear_disp_3m",
        "bull_close_strong_3m", "bear_close_strong_3m",
        "bull_overextended_3m", "bear_overextended_3m",
    ]].reindex(out.index, method="ffill")
    for col in aligned.columns:
        out[col] = aligned[col]
    return out


def build_1m_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    out = df_1m.copy()
    fast = ema(out["Close"], 5)
    slow = ema(out["Close"], 13)
    out["bull_cisd_1m"] = ema_cross_up(fast, slow) & (out["Close"] > out["Open"])
    out["bear_cisd_1m"] = ema_cross_down(fast, slow) & (out["Close"] < out["Open"])
    out["swing_high_1m"] = out["High"].rolling(5).max().shift(1)
    out["swing_low_1m"] = out["Low"].rolling(5).min().shift(1)
    out["bull_mss_1m"] = out["Close"] > out["swing_high_1m"]
    out["bear_mss_1m"] = out["Close"] < out["swing_low_1m"]
    out["bull_ifvg_1m"] = (
        (out["Low"] > out["High"].shift(2))
        | ((out["Close"] > out["Open"]) & (out["Low"] > out["Low"].shift(1)))
    )
    out["bear_ifvg_1m"] = (
        (out["High"] < out["Low"].shift(2))
        | ((out["Close"] < out["Open"]) & (out["High"] < out["High"].shift(1)))
    )
    return out


def build_model_frame(df_1m: pd.DataFrame) -> pd.DataFrame:
    out = df_1m.copy()
    out = build_session_features(out)
    out = build_4h_context(out)
    out = build_30m_bridge(out)
    out = build_3m_features(out)
    out = build_1m_features(out)
    return out


@dataclass
class PendingSignal:
    direction: str = ""
    entry_trigger: float = np.nan
    stop_price: float = np.nan
    target_price: float = np.nan
    expiry_bar: int = -1
    setup_type: str = ""
    bridge_type: str = ""
    entry_variant: str = ""
    pullback_low: float = np.nan
    pullback_high: float = np.nan
    partial_target: float = np.nan
    runner_target: float = np.nan
    setup_tier: str = ""


@dataclass
class SetupState:
    direction: str = ""
    setup_type: str = ""
    narrative_bar: int = -1
    context_bar: int = -1
    bridge_bar: int = -1
    active_until_bar: int = -1
    bridge_type: str = ""
    bridge_low: float = np.nan
    bridge_high: float = np.nan


class ICT_MULTI_SETUP_V452(Strategy):
    risk_multiple = 2.0
    min_target_points = 50.0
    min_stop_points = 25.0
    stop_buffer_atr = 0.15
    fixed_size = 0.10

    require_pullback_entry = True
    pullback_entry_tolerance_points = 6.0
    pending_expiry_bars = 18

    enable_partial = True
    partial_rr = 1.0
    partial_close_fraction = 0.50
    enable_runner_to_liquidity = True
    breakeven_only_after_confirmation = True
    be_confirm_rr = 1.25
    min_bars_before_be = 3

    prop_mode = True
    prop_daily_loss_limit = -1200.0
    prop_daily_max_trades = 3
    prop_max_consecutive_losses = 2
    prop_reduce_size_after_drawdown = True
    prop_drawdown_reduce_threshold = -800.0
    prop_reduced_size_multiplier = 0.5
    payout_defense_mode = False
    payout_defense_daily_profit_lock = 800.0

    narrative_expiry_bars = 240
    context_expiry_bars = 180
    bridge_expiry_bars = 90

    enable_asia_continuation = True
    enable_asia_reversal = False
    enable_london_continuation = True
    enable_london_reversal = False
    enable_nyam_continuation = True
    enable_nyam_reversal = False
    enable_nypm_continuation = True
    enable_nypm_reversal = False

    allow_cisd_london = True
    allow_cisd_asia = False
    allow_cisd_nyam = False
    allow_cisd_nypm = False
    allow_ifvg_asia = True
    allow_ifvg_london = True
    allow_ifvg_nyam = True
    allow_ifvg_nypm = True
    prop_trade_only_ranked_setups = True
    allowed_prop_setup_tiers = {"A", "B"}

    last_debug_counts = {}
    last_trade_log = []
    last_event_log = []

    def init(self):
        self.m = build_model_frame(self.data.df.copy())
        self.pending = PendingSignal()
        self.state = SetupState()
        self.active_risk = np.nan
        self.planned_stop = np.nan
        self.planned_target = np.nan
        self.partial_taken = False
        self.be_moved = False
        self.entry_bar_idx = -1
        self.entry_day = None
        self.entry_side = ""
        self.open_trade_meta = None
        self.prev_closed_count = 0

        self.realized_pnl_today = 0.0
        self.current_et_day = None
        self.daily_trade_count = 0
        self.consecutive_losses = 0

        self.debug_counts = {
            "arm_long_narrative": 0,
            "arm_short_narrative": 0,
            "confirm_long_context": 0,
            "confirm_short_context": 0,
            "confirm_long_bridge": 0,
            "confirm_short_bridge": 0,
            "arm_pending_long": 0,
            "arm_pending_short": 0,
            "trigger_long_entry": 0,
            "trigger_short_entry": 0,
            "pullback_long_touched": 0,
            "pullback_short_touched": 0,
            "partial_taken": 0,
            "be_after_confirmation": 0,
            "prop_block_daily_loss": 0,
            "prop_block_daily_trade_cap": 0,
            "prop_block_consecutive_losses": 0,
            "blocked_by_ranking": 0,
            "blocked_cisd_session": 0,
            "blocked_ifvg_session": 0,
            "reject_overextended_long": 0,
            "reject_overextended_short": 0,
            "reject_1m_refine_long": 0,
            "reject_1m_refine_short": 0,
            "expire_pending": 0,
            "expire_state": 0,
            "clear_state_invalid_long": 0,
            "clear_state_invalid_short": 0,
        }
        self.__class__.last_debug_counts = dict(self.debug_counts)
        self.__class__.last_trade_log = []
        self.__class__.last_event_log = []

    def _sync_debug(self):
        self.__class__.last_debug_counts = dict(self.debug_counts)

    def _sync_trade_log(self):
        self.__class__.last_trade_log = list(self.__class__.last_trade_log)

    def _sync_event_log(self):
        self.__class__.last_event_log = list(self.__class__.last_event_log)

    def _event_timestamp_et(self) -> str:
        try:
            row = self.m.iloc[self._i()]
            if "et_ts" in row and pd.notna(row["et_ts"]):
                return str(row["et_ts"])
        except Exception:
            pass
        try:
            idx = self.data.index[-1]
            return str(pd.Timestamp(idx))
        except Exception:
            return ""

    def _emit_event(self, event_type: str, **payload):
        item = {
            "event_type": event_type,
            "timestamp_et": self._event_timestamp_et(),
        }
        item.update(payload)
        self.__class__.last_event_log.append(item)
        self._sync_event_log()

    def _i(self) -> int:
        return len(self.data) - 1

    def _clear_pending(self):
        self.pending = PendingSignal()

    def _clear_state(self):
        self.state = SetupState()

    def _latest_trade(self):
        try:
            if self.trades:
                return self.trades[-1]
        except Exception:
            pass
        return None

    def _update_day_reset(self, row: pd.Series):
        if self.current_et_day != row["et_date"]:
            self.current_et_day = row["et_date"]
            self.realized_pnl_today = 0.0
            self.daily_trade_count = 0
            self.consecutive_losses = 0

    def _log_newly_closed_trades(self):
        try:
            closed = list(self.closed_trades)
        except Exception:
            return

        if len(closed) <= self.prev_closed_count:
            return

        new_items = closed[self.prev_closed_count:]
        for t in new_items:
            pnl = float(getattr(t, "pl", 0.0))
            self.realized_pnl_today += pnl
            if pnl < 0:
                self.consecutive_losses += 1
            elif pnl > 0:
                self.consecutive_losses = 0

            meta = self.open_trade_meta or {}
            planned_entry = meta.get("planned_entry_price", np.nan)
            planned_stop = meta.get("planned_stop_price", np.nan)
            planned_target = meta.get("planned_target_price", np.nan)

            stop_points = np.nan
            tp_points = np.nan
            planned_rr = np.nan
            if pd.notna(planned_entry) and pd.notna(planned_stop):
                stop_points = abs(float(planned_entry) - float(planned_stop))
            if pd.notna(planned_entry) and pd.notna(planned_target):
                tp_points = abs(float(planned_target) - float(planned_entry))
            if pd.notna(stop_points) and stop_points > 0 and pd.notna(tp_points):
                planned_rr = float(tp_points) / float(stop_points)

            event_side = "LONG" if float(t.size) > 0 else "SHORT"
            self.__class__.last_trade_log.append({
                "side": event_side,
                "setup_type": meta.get("setup_type", ""),
                "bridge_type": meta.get("bridge_type", ""),
                "entry_variant": meta.get("entry_variant", ""),
                "setup_tier": meta.get("setup_tier", ""),
                "planned_entry_price": planned_entry,
                "planned_stop_price": planned_stop,
                "planned_target_price": planned_target,
                "partial_target_price": meta.get("partial_target_price", np.nan),
                "runner_target_price": meta.get("runner_target_price", np.nan),
                "stop_points": stop_points,
                "tp_points": tp_points,
                "planned_rr": planned_rr,
                "entry_price": float(t.entry_price),
                "exit_price": float(t.exit_price),
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "return_pct": float(getattr(t, "pl_pct", np.nan)) if hasattr(t, "pl_pct") else np.nan,
                "pnl": pnl,
            })
            self.__class__.last_event_log.append({
                "event_type": "trade_closed",
                "timestamp_et": str(t.exit_time),
                "side": event_side,
                "setup_type": meta.get("setup_type", ""),
                "bridge_type": meta.get("bridge_type", ""),
                "setup_tier": meta.get("setup_tier", ""),
                "planned_entry_price": planned_entry,
                "planned_stop_price": planned_stop,
                "planned_target_price": planned_target,
                "partial_target_price": meta.get("partial_target_price", np.nan),
                "runner_target_price": meta.get("runner_target_price", np.nan),
                "entry_price": float(t.entry_price),
                "exit_price": float(t.exit_price),
                "pnl": pnl,
            })

        if new_items and not self.position:
            self.open_trade_meta = None

        self.prev_closed_count = len(closed)
        self._sync_trade_log()
        self._sync_event_log()

    def _prop_can_trade(self) -> bool:
        if not self.prop_mode:
            return True
        if self.realized_pnl_today <= self.prop_daily_loss_limit:
            self.debug_counts["prop_block_daily_loss"] += 1
            self._sync_debug()
            return False
        if self.daily_trade_count >= self.prop_daily_max_trades:
            self.debug_counts["prop_block_daily_trade_cap"] += 1
            self._sync_debug()
            return False
        if self.consecutive_losses >= self.prop_max_consecutive_losses:
            self.debug_counts["prop_block_consecutive_losses"] += 1
            self._sync_debug()
            return False
        if self.payout_defense_mode and self.realized_pnl_today >= self.payout_defense_daily_profit_lock:
            self.debug_counts["prop_block_daily_trade_cap"] += 1
            self._sync_debug()
            return False
        return True

    def _effective_size(self) -> float:
        size = self.fixed_size
        if self.prop_mode and self.prop_reduce_size_after_drawdown and self.realized_pnl_today <= self.prop_drawdown_reduce_threshold:
            size *= self.prop_reduced_size_multiplier
        return size

    def _setup_tier(self, setup_type: str, bridge_type: str, side: str) -> str:
        if bridge_type == "CISD":
            return "C"
        if setup_type == "NYPM_CONTINUATION" and bridge_type in {"C2C3", "MSS"} and side == "SHORT":
            return "A"
        if setup_type == "NYAM_CONTINUATION" and bridge_type in {"MSS", "C2C3"} and side == "SHORT":
            return "A"
        if setup_type == "LONDON_CONTINUATION" and bridge_type in {"MSS", "C2C3", "iFVG"}:
            return "A"
        if setup_type == "ASIA_CONTINUATION" and bridge_type in {"C2C3", "iFVG"}:
            return "B"
        if setup_type == "NYPM_CONTINUATION" and bridge_type == "iFVG" and side == "SHORT":
            return "B"
        if setup_type == "NYAM_CONTINUATION" and bridge_type == "iFVG" and side == "SHORT":
            return "B"
        return "C"

    def _bridge_allowed_for_session(self, setup_type: str, bridge_type: str) -> bool:
        if bridge_type == "CISD":
            if setup_type.startswith("LONDON_"):
                return self.allow_cisd_london
            if setup_type.startswith("ASIA_"):
                return self.allow_cisd_asia
            if setup_type.startswith("NYAM_"):
                return self.allow_cisd_nyam
            if setup_type.startswith("NYPM_"):
                return self.allow_cisd_nypm
        if bridge_type == "iFVG":
            if setup_type.startswith("ASIA_"):
                return self.allow_ifvg_asia
            if setup_type.startswith("LONDON_"):
                return self.allow_ifvg_london
            if setup_type.startswith("NYAM_"):
                return self.allow_ifvg_nyam
            if setup_type.startswith("NYPM_"):
                return self.allow_ifvg_nypm
        return True

    def _bull_narrative_ok(self, row: pd.Series) -> bool:
        return bool(row["bull_4h_bias"] and row["above_4h_eq"] and (row["bull_profile_4h"] or row["bull_disp_4h"] or row["bull_fvg_4h"]))

    def _bear_narrative_ok(self, row: pd.Series) -> bool:
        return bool(row["bear_4h_bias"] and row["below_4h_eq"] and (row["bear_profile_4h"] or row["bear_disp_4h"] or row["bear_fvg_4h"]))

    def _bull_context_continuation_asia(self, row: pd.Series) -> bool:
        return bool(row["is_asia"] and row["bull_4h_bias"] and row["above_4h_eq"] and (row["bull_disp_30m"] or row["bull_close_strong_30m"] or row["bull_cisd_30m"]))

    def _bear_context_continuation_asia(self, row: pd.Series) -> bool:
        return bool(row["is_asia"] and row["bear_4h_bias"] and row["below_4h_eq"] and (row["bear_disp_30m"] or row["bear_close_strong_30m"] or row["bear_cisd_30m"]))

    def _bull_context_continuation_london(self, row: pd.Series) -> bool:
        return bool(row["is_london"] and row["bull_4h_bias"] and row["above_4h_eq"] and (row["bull_disp_30m"] or row["bull_close_strong_30m"] or row["bull_cisd_30m"]))

    def _bear_context_continuation_london(self, row: pd.Series) -> bool:
        return bool(row["is_london"] and row["bear_4h_bias"] and row["below_4h_eq"] and (row["bear_disp_30m"] or row["bear_close_strong_30m"] or row["bear_cisd_30m"]))

    def _bull_context_continuation_nyam(self, row: pd.Series) -> bool:
        return bool(
            row["is_nyam"]
            and row["bull_4h_bias"]
            and row["above_4h_eq"]
            and (row["swept_prior_us_low"] or row["reclaimed_prior_us_low"] or row["swept_prev_day_low"])
            and (row["bull_disp_30m"] or row["bull_close_strong_30m"] or row["bull_cisd_30m"])
        )

    def _bear_context_continuation_nyam(self, row: pd.Series) -> bool:
        return bool(
            row["is_nyam"]
            and row["bear_4h_bias"]
            and row["below_4h_eq"]
            and (row["swept_prior_us_high"] or row["rejected_prior_us_high"] or row["swept_prev_day_high"])
            and (row["bear_disp_30m"] or row["bear_close_strong_30m"] or row["bear_cisd_30m"])
        )

    def _bull_context_continuation_nypm(self, row: pd.Series) -> bool:
        return bool(
            row["is_nypm"]
            and row["bull_4h_bias"]
            and row["above_4h_eq"]
            and (row["reclaimed_nyam_low"] or row["reclaimed_prior_us_low"] or row["swept_nyam_low"])
            and (row["bull_disp_30m"] or row["bull_close_strong_30m"] or row["bull_cisd_30m"])
        )

    def _bear_context_continuation_nypm(self, row: pd.Series) -> bool:
        return bool(
            row["is_nypm"]
            and row["bear_4h_bias"]
            and row["below_4h_eq"]
            and (row["rejected_nyam_high"] or row["rejected_prior_us_high"] or row["swept_nyam_high"])
            and (row["bear_disp_30m"] or row["bear_close_strong_30m"] or row["bear_cisd_30m"])
        )

    def _long_entry_window_ok(self, row: pd.Series) -> bool:
        return bool(row["is_asia_entry_window"] or row["is_london_entry_window"] or row["is_nyam_entry_window"] or row["is_nypm_entry_window"])

    def _short_entry_window_ok(self, row: pd.Series) -> bool:
        return self._long_entry_window_ok(row)

    def _bull_invalid(self, row: pd.Series) -> bool:
        return bool((not row["above_4h_eq"]) or (not row["bull_4h_bias"]))

    def _bear_invalid(self, row: pd.Series) -> bool:
        return bool((not row["below_4h_eq"]) or (not row["bear_4h_bias"]))

    def _bull_execution_ok(self, row: pd.Series) -> bool:
        return bool(self._long_entry_window_ok(row) and row["bull_disp_3m"] and row["bull_close_strong_3m"] and not row["bull_overextended_3m"])

    def _bear_execution_ok(self, row: pd.Series) -> bool:
        return bool(self._short_entry_window_ok(row) and row["bear_disp_3m"] and row["bear_close_strong_3m"] and not row["bear_overextended_3m"])

    def _1m_refine_long_ok(self, row: pd.Series) -> bool:
        return bool(row["bull_cisd_1m"] or row["bull_mss_1m"] or row["bull_ifvg_1m"])

    def _1m_refine_short_ok(self, row: pd.Series) -> bool:
        return bool(row["bear_cisd_1m"] or row["bear_mss_1m"] or row["bear_ifvg_1m"])

    def _hybrid_target_above(self, row: pd.Series, entry: float, risk: float) -> float:
        rr_target = entry + (risk * self.risk_multiple)
        min_target = entry + self.min_target_points
        candidates = []
        for level_name in ("prev_day_high", "prior_us_high", "asia_high", "london_high", "nyam_high", "day_high"):
            level = row.get(level_name, np.nan)
            if pd.notna(level) and float(level) > entry:
                candidates.append(float(level))
        final_target = max(rr_target, min_target)
        if candidates:
            final_target = max(final_target, max(candidates))
        return final_target

    def _hybrid_target_below(self, row: pd.Series, entry: float, risk: float) -> float:
        rr_target = entry - (risk * self.risk_multiple)
        min_target = entry - self.min_target_points
        candidates = []
        for level_name in ("prev_day_low", "prior_us_low", "asia_low", "london_low", "nyam_low", "day_low"):
            level = row.get(level_name, np.nan)
            if pd.notna(level) and float(level) < entry:
                candidates.append(float(level))
        final_target = min(rr_target, min_target)
        if candidates:
            final_target = min(final_target, min(candidates))
        return final_target

    def _expire_state_if_needed(self, i: int):
        if self.state.direction == "":
            return
        if self.state.narrative_bar >= 0 and i - self.state.narrative_bar > self.narrative_expiry_bars:
            self.debug_counts["expire_state"] += 1
            self._sync_debug()
            self._clear_state()
            return
        if self.state.context_bar >= 0 and i - self.state.context_bar > self.context_expiry_bars:
            self.debug_counts["expire_state"] += 1
            self._sync_debug()
            self._clear_state()
            return
        if self.state.bridge_bar >= 0 and i > self.state.active_until_bar:
            self.debug_counts["expire_state"] += 1
            self._sync_debug()
            self._clear_state()
            return

    def _arm_long_pullback(self, row: pd.Series, high_now: float, low_now: float, atr3: float, i: int):
        if not self._bridge_allowed_for_session(self.state.setup_type, self.state.bridge_type):
            if self.state.bridge_type == "CISD":
                self.debug_counts["blocked_cisd_session"] += 1
            elif self.state.bridge_type == "iFVG":
                self.debug_counts["blocked_ifvg_session"] += 1
            self._sync_debug()
            self._clear_state()
            return

        tier = self._setup_tier(self.state.setup_type, self.state.bridge_type, "LONG")
        if self.prop_trade_only_ranked_setups and self.prop_mode and tier not in self.allowed_prop_setup_tiers:
            self.debug_counts["blocked_by_ranking"] += 1
            self._sync_debug()
            self._clear_state()
            return

        pullback_low = float(row.get("bull_bridge_low_30m", np.nan))
        pullback_high = float(row.get("bull_bridge_high_30m", np.nan))
        if not (np.isfinite(pullback_low) and np.isfinite(pullback_high)):
            return

        structural_stop = min(low_now, pullback_low) - (atr3 * self.stop_buffer_atr if atr3 > 0 else 0.0)
        entry_trigger = pullback_high + 0.25
        stop = min(structural_stop, entry_trigger - self.min_stop_points)
        if stop >= entry_trigger:
            return

        risk = entry_trigger - stop
        runner_target = self._hybrid_target_above(row, entry_trigger, risk)
        partial_target = entry_trigger + (risk * self.partial_rr)

        self.pending = PendingSignal(
            direction="long",
            entry_trigger=entry_trigger,
            stop_price=stop,
            target_price=runner_target,
            expiry_bar=i + self.pending_expiry_bars,
            setup_type=self.state.setup_type,
            bridge_type=self.state.bridge_type,
            entry_variant="PULLBACK_1M",
            pullback_low=pullback_low,
            pullback_high=pullback_high,
            partial_target=partial_target,
            runner_target=runner_target,
            setup_tier=tier,
        )
        self.debug_counts["arm_pending_long"] += 1
        self._sync_debug()

    def _arm_short_pullback(self, row: pd.Series, high_now: float, low_now: float, atr3: float, i: int):
        if not self._bridge_allowed_for_session(self.state.setup_type, self.state.bridge_type):
            if self.state.bridge_type == "CISD":
                self.debug_counts["blocked_cisd_session"] += 1
            elif self.state.bridge_type == "iFVG":
                self.debug_counts["blocked_ifvg_session"] += 1
            self._sync_debug()
            self._clear_state()
            return

        tier = self._setup_tier(self.state.setup_type, self.state.bridge_type, "SHORT")
        if self.prop_trade_only_ranked_setups and self.prop_mode and tier not in self.allowed_prop_setup_tiers:
            self.debug_counts["blocked_by_ranking"] += 1
            self._sync_debug()
            self._clear_state()
            return

        pullback_low = float(row.get("bear_bridge_low_30m", np.nan))
        pullback_high = float(row.get("bear_bridge_high_30m", np.nan))
        if not (np.isfinite(pullback_low) and np.isfinite(pullback_high)):
            return

        structural_stop = max(high_now, pullback_high) + (atr3 * self.stop_buffer_atr if atr3 > 0 else 0.0)
        entry_trigger = pullback_low - 0.25
        stop = max(structural_stop, entry_trigger + self.min_stop_points)
        if stop <= entry_trigger:
            return

        risk = stop - entry_trigger
        runner_target = self._hybrid_target_below(row, entry_trigger, risk)
        partial_target = entry_trigger - (risk * self.partial_rr)

        self.pending = PendingSignal(
            direction="short",
            entry_trigger=entry_trigger,
            stop_price=stop,
            target_price=runner_target,
            expiry_bar=i + self.pending_expiry_bars,
            setup_type=self.state.setup_type,
            bridge_type=self.state.bridge_type,
            entry_variant="PULLBACK_1M",
            pullback_low=pullback_low,
            pullback_high=pullback_high,
            partial_target=partial_target,
            runner_target=runner_target,
            setup_tier=tier,
        )
        self.debug_counts["arm_pending_short"] += 1
        self._sync_debug()

    def _pending_long_ready(self, row: pd.Series, high_now: float, low_now: float) -> bool:
        if not self._1m_refine_long_ok(row):
            self.debug_counts["reject_1m_refine_long"] += 1
            self._sync_debug()
            return False
        zone_low = self.pending.pullback_low - self.pullback_entry_tolerance_points
        zone_high = self.pending.pullback_high + self.pullback_entry_tolerance_points
        touched = (low_now <= zone_high) and (high_now >= zone_low)
        if touched:
            self.debug_counts["pullback_long_touched"] += 1
            self._sync_debug()
        return touched and high_now >= self.pending.entry_trigger

    def _pending_short_ready(self, row: pd.Series, high_now: float, low_now: float) -> bool:
        if not self._1m_refine_short_ok(row):
            self.debug_counts["reject_1m_refine_short"] += 1
            self._sync_debug()
            return False
        zone_low = self.pending.pullback_low - self.pullback_entry_tolerance_points
        zone_high = self.pending.pullback_high + self.pullback_entry_tolerance_points
        touched = (low_now <= zone_high) and (high_now >= zone_low)
        if touched:
            self.debug_counts["pullback_short_touched"] += 1
            self._sync_debug()
        return touched and low_now <= self.pending.entry_trigger

    def _record_open_trade_meta(self, entry: float):
        self.open_trade_meta = {
            "setup_type": self.pending.setup_type,
            "bridge_type": self.pending.bridge_type,
            "entry_variant": self.pending.entry_variant,
            "setup_tier": self.pending.setup_tier,
            "planned_entry_price": entry,
            "planned_stop_price": self.pending.stop_price,
            "planned_target_price": self.pending.target_price,
            "partial_target_price": self.pending.partial_target,
            "runner_target_price": self.pending.runner_target,
        }

    def _manage_trade(self):
        trade = self._latest_trade()
        if trade is None or not self.position or not np.isfinite(self.active_risk):
            return

        price = float(self.data.Close[-1])
        i = self._i()

        if self.position.is_long:
            r_now = (price - float(trade.entry_price)) / self.active_risk
            if self.enable_partial and not self.partial_taken and r_now >= self.partial_rr:
                try:
                    self.position.close(portion=self.partial_close_fraction)
                    self.partial_taken = True
                    self._emit_event(
                        "partial_taken",
                        side="LONG",
                        portion=float(self.partial_close_fraction),
                        rr_now=float(r_now),
                        price=float(price),
                        entry_price=float(trade.entry_price),
                        stop_price=float(trade.sl) if trade.sl is not None else np.nan,
                    )
                    self.debug_counts["partial_taken"] += 1
                    self._sync_debug()
                except Exception:
                    pass
            if self.breakeven_only_after_confirmation and (not self.be_moved) and i - self.entry_bar_idx >= self.min_bars_before_be and r_now >= self.be_confirm_rr:
                try:
                    if trade.sl is not None:
                        trade.sl = max(float(trade.sl), float(trade.entry_price))
                    self.be_moved = True
                    self._emit_event(
                        "stop_moved_to_be",
                        side="LONG",
                        new_stop=float(trade.sl) if trade.sl is not None else float(trade.entry_price),
                        entry_price=float(trade.entry_price),
                        rr_now=float(r_now),
                    )
                    self.debug_counts["be_after_confirmation"] += 1
                    self._sync_debug()
                except Exception:
                    pass

        if self.position.is_short:
            r_now = (float(trade.entry_price) - price) / self.active_risk
            if self.enable_partial and not self.partial_taken and r_now >= self.partial_rr:
                try:
                    self.position.close(portion=self.partial_close_fraction)
                    self.partial_taken = True
                    self._emit_event(
                        "partial_taken",
                        side="SHORT",
                        portion=float(self.partial_close_fraction),
                        rr_now=float(r_now),
                        price=float(price),
                        entry_price=float(trade.entry_price),
                        stop_price=float(trade.sl) if trade.sl is not None else np.nan,
                    )
                    self.debug_counts["partial_taken"] += 1
                    self._sync_debug()
                except Exception:
                    pass
            if self.breakeven_only_after_confirmation and (not self.be_moved) and i - self.entry_bar_idx >= self.min_bars_before_be and r_now >= self.be_confirm_rr:
                try:
                    if trade.sl is not None:
                        trade.sl = min(float(trade.sl), float(trade.entry_price))
                    self.be_moved = True
                    self._emit_event(
                        "stop_moved_to_be",
                        side="SHORT",
                        new_stop=float(trade.sl) if trade.sl is not None else float(trade.entry_price),
                        entry_price=float(trade.entry_price),
                        rr_now=float(r_now),
                    )
                    self.debug_counts["be_after_confirmation"] += 1
                    self._sync_debug()
                except Exception:
                    pass

    def next(self):
        self._log_newly_closed_trades()
        i = self._i()
        if i < 200:
            return

        row = self.m.iloc[i]
        self._update_day_reset(row)

        close_now = float(self.data.Close[-1])
        high_now = float(self.data.High[-1])
        low_now = float(self.data.Low[-1])
        atr3 = float(row["atr14_3m"]) if pd.notna(row["atr14_3m"]) else 0.0

        self._manage_trade()

        if self.pending.expiry_bar >= 0 and i > self.pending.expiry_bar:
            self.debug_counts["expire_pending"] += 1
            self._sync_debug()
            self._clear_pending()

        if self.pending.direction == "long" and not self.position:
            if self._pending_long_ready(row, high_now, low_now) and self._prop_can_trade():
                entry = max(close_now, self.pending.entry_trigger)
                risk = entry - float(self.pending.stop_price)
                if risk > 0:
                    self.active_risk = risk
                    self.buy(size=self._effective_size(), sl=float(self.pending.stop_price), tp=float(self.pending.runner_target))
                    self.daily_trade_count += 1
                    self.partial_taken = False
                    self.be_moved = False
                    self.entry_bar_idx = i
                    self.entry_day = row["et_date"]
                    self.entry_side = "LONG"
                    self._record_open_trade_meta(entry)
                    self._emit_event(
                        "entry_opened",
                        side="LONG",
                        setup_type=self.pending.setup_type,
                        bridge_type=self.pending.bridge_type,
                        entry_variant=self.pending.entry_variant,
                        setup_tier=self.pending.setup_tier,
                        planned_entry_price=entry,
                        planned_stop_price=float(self.pending.stop_price),
                        planned_target_price=float(self.pending.target_price),
                        partial_target_price=float(self.pending.partial_target),
                        runner_target_price=float(self.pending.runner_target),
                        qty=float(self._effective_size()),
                    )
                    self.debug_counts["trigger_long_entry"] += 1
                    self._sync_debug()
                self._clear_pending()
                self._clear_state()
                return

        if self.pending.direction == "short" and not self.position:
            if self._pending_short_ready(row, high_now, low_now) and self._prop_can_trade():
                entry = min(close_now, self.pending.entry_trigger)
                risk = float(self.pending.stop_price) - entry
                if risk > 0:
                    self.active_risk = risk
                    self.sell(size=self._effective_size(), sl=float(self.pending.stop_price), tp=float(self.pending.runner_target))
                    self.daily_trade_count += 1
                    self.partial_taken = False
                    self.be_moved = False
                    self.entry_bar_idx = i
                    self.entry_day = row["et_date"]
                    self.entry_side = "SHORT"
                    self._record_open_trade_meta(entry)
                    self._emit_event(
                        "entry_opened",
                        side="SHORT",
                        setup_type=self.pending.setup_type,
                        bridge_type=self.pending.bridge_type,
                        entry_variant=self.pending.entry_variant,
                        setup_tier=self.pending.setup_tier,
                        planned_entry_price=entry,
                        planned_stop_price=float(self.pending.stop_price),
                        planned_target_price=float(self.pending.target_price),
                        partial_target_price=float(self.pending.partial_target),
                        runner_target_price=float(self.pending.runner_target),
                        qty=float(self._effective_size()),
                    )
                    self.debug_counts["trigger_short_entry"] += 1
                    self._sync_debug()
                self._clear_pending()
                self._clear_state()
                return

        if self.position:
            return

        self._expire_state_if_needed(i)

        if self.state.direction == "":
            if self._bull_narrative_ok(row):
                self.state.direction = "long"
                self.state.narrative_bar = i
                self.state.setup_type = "GLOBAL"
                self.debug_counts["arm_long_narrative"] += 1
                self._sync_debug()
                return
            if self._bear_narrative_ok(row):
                self.state.direction = "short"
                self.state.narrative_bar = i
                self.state.setup_type = "GLOBAL"
                self.debug_counts["arm_short_narrative"] += 1
                self._sync_debug()
                return

        if self.state.direction == "long":
            if self._bull_invalid(row):
                self.debug_counts["clear_state_invalid_long"] += 1
                self._sync_debug()
                self._clear_state()
                return

            if self.state.context_bar < 0:
                hit = False
                if self.enable_asia_continuation and self._bull_context_continuation_asia(row):
                    self.state.context_bar = i
                    self.state.setup_type = "ASIA_CONTINUATION"
                    hit = True
                elif self.enable_london_continuation and self._bull_context_continuation_london(row):
                    self.state.context_bar = i
                    self.state.setup_type = "LONDON_CONTINUATION"
                    hit = True
                elif self.enable_nypm_continuation and self._bull_context_continuation_nypm(row):
                    self.state.context_bar = i
                    self.state.setup_type = "NYPM_CONTINUATION"
                    hit = True
                if hit:
                    self.debug_counts["confirm_long_context"] += 1
                    self._sync_debug()
                return

            if self.state.bridge_bar < 0:
                bridge_type = str(row.get("bridge_type_30m", ""))
                bull_cisd_ok = bool(row["bull_cisd_30m"] and row["bull_disp_30m"] and row["bull_close_strong_30m"] and self.state.setup_type.startswith("LONDON_"))
                bull_ifvg_ok = bool(row["bull_ifvg_30m"] and row["bull_disp_30m"] and row["bull_close_strong_30m"])
                bull_ok = bool(bull_cisd_ok or row["bull_mss_30m"] or bull_ifvg_ok or row["bull_c2_or_c3"])
                if bull_ok and bridge_type:
                    self.state.bridge_bar = i
                    self.state.bridge_type = bridge_type
                    self.state.bridge_low = float(row.get("bull_bridge_low_30m", np.nan))
                    self.state.bridge_high = float(row.get("bull_bridge_high_30m", np.nan))
                    self.state.active_until_bar = i + self.bridge_expiry_bars
                    self.debug_counts["confirm_long_bridge"] += 1
                    self._sync_debug()
                return

            if self._bull_execution_ok(row):
                self._arm_long_pullback(row, high_now, low_now, atr3, i)
                return

        if self.state.direction == "short":
            if self._bear_invalid(row):
                self.debug_counts["clear_state_invalid_short"] += 1
                self._sync_debug()
                self._clear_state()
                return

            if self.state.context_bar < 0:
                hit = False
                if self.enable_asia_continuation and self._bear_context_continuation_asia(row):
                    self.state.context_bar = i
                    self.state.setup_type = "ASIA_CONTINUATION"
                    hit = True
                elif self.enable_london_continuation and self._bear_context_continuation_london(row):
                    self.state.context_bar = i
                    self.state.setup_type = "LONDON_CONTINUATION"
                    hit = True
                elif self.enable_nyam_continuation and self._bear_context_continuation_nyam(row):
                    self.state.context_bar = i
                    self.state.setup_type = "NYAM_CONTINUATION"
                    hit = True
                elif self.enable_nypm_continuation and self._bear_context_continuation_nypm(row):
                    self.state.context_bar = i
                    self.state.setup_type = "NYPM_CONTINUATION"
                    hit = True
                if hit:
                    self.debug_counts["confirm_short_context"] += 1
                    self._sync_debug()
                return

            if self.state.bridge_bar < 0:
                bridge_type = str(row.get("bridge_type_30m", ""))
                bear_cisd_ok = bool(row["bear_cisd_30m"] and row["bear_disp_30m"] and row["bear_close_strong_30m"] and self.state.setup_type.startswith("LONDON_"))
                bear_ifvg_ok = bool(row["bear_ifvg_30m"] and row["bear_disp_30m"] and row["bear_close_strong_30m"])
                bear_ok = bool(bear_cisd_ok or row["bear_mss_30m"] or bear_ifvg_ok or row["bear_c2_or_c3"])
                if bear_ok and bridge_type:
                    self.state.bridge_bar = i
                    self.state.bridge_type = bridge_type
                    self.state.bridge_low = float(row.get("bear_bridge_low_30m", np.nan))
                    self.state.bridge_high = float(row.get("bear_bridge_high_30m", np.nan))
                    self.state.active_until_bar = i + self.bridge_expiry_bars
                    self.debug_counts["confirm_short_bridge"] += 1
                    self._sync_debug()
                return

            if self._bear_execution_ok(row):
                self._arm_short_pullback(row, high_now, low_now, atr3, i)
                return
