from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from backtesting import Strategy


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


def ema_cross_up(fast: pd.Series, slow: pd.Series) -> pd.Series:
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def ema_cross_down(fast: pd.Series, slow: pd.Series) -> pd.Series:
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


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
        & (((bars_4h["Close"] - bars_4h["Low"]) / rng_4h) >= 0.6)
    )
    bars_4h["bear_profile_4h"] = (
        (bars_4h["Close"] < bars_4h["Open"])
        & (((bars_4h["High"] - bars_4h["Close"]) / rng_4h) >= 0.6)
    )

    aligned = bars_4h[
        [
            "ema20_4h",
            "ema50_4h",
            "eq_4h",
            "bull_disp_4h",
            "bear_disp_4h",
            "bull_fvg_4h",
            "bear_fvg_4h",
            "bull_profile_4h",
            "bear_profile_4h",
        ]
    ].reindex(out.index, method="ffill")

    for col in aligned.columns:
        out[col] = aligned[col]

    out["bull_4h_bias"] = out["ema20_4h"] > out["ema50_4h"]
    out["bear_4h_bias"] = out["ema20_4h"] < out["ema50_4h"]
    out["above_4h_eq"] = out["Close"] > out["eq_4h"]
    out["below_4h_eq"] = out["Close"] < out["eq_4h"]

    return out


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


def build_30m_features(df_3m: pd.DataFrame) -> pd.DataFrame:
    out = df_3m.copy()

    bars_30m = df_3m.resample("30min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()

    bars_30m["atr14_30m"] = atr(bars_30m, 14)
    body = (bars_30m["Close"] - bars_30m["Open"]).abs()
    rng = (bars_30m["High"] - bars_30m["Low"]).replace(0, np.nan)

    bars_30m["bull_close_strong_30m"] = (
        (bars_30m["Close"] > bars_30m["Open"])
        & (((bars_30m["Close"] - bars_30m["Low"]) / rng) >= 0.65)
    )
    bars_30m["bear_close_strong_30m"] = (
        (bars_30m["Close"] < bars_30m["Open"])
        & (((bars_30m["High"] - bars_30m["Close"]) / rng) >= 0.65)
    )
    bars_30m["bull_disp_30m"] = (bars_30m["Close"] > bars_30m["Open"]) & (body > bars_30m["atr14_30m"] * 0.8)
    bars_30m["bear_disp_30m"] = (bars_30m["Close"] < bars_30m["Open"]) & (body > bars_30m["atr14_30m"] * 0.8)

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

    aligned = bars_30m[
        [
            "bull_close_strong_30m",
            "bear_close_strong_30m",
            "bull_disp_30m",
            "bear_disp_30m",
            "bull_c2_or_c3",
            "bear_c2_or_c3",
            "bull_cisd_30m",
            "bear_cisd_30m",
            "bull_mss_30m",
            "bear_mss_30m",
            "bull_ifvg_30m",
            "bear_ifvg_30m",
        ]
    ].reindex(out.index, method="ffill")

    for col in aligned.columns:
        out[col] = aligned[col]

    return out


def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    idx_utc = pd.DatetimeIndex(out.index).tz_localize("UTC")
    idx_et = idx_utc.tz_convert("America/New_York")
    out["et_date"] = idx_et.date
    out["et_hour"] = idx_et.hour
    out["et_minute"] = idx_et.minute

    out["is_nyam"] = (
        ((out["et_hour"] == 9) & (out["et_minute"] >= 30))
        | (out["et_hour"] == 10)
        | ((out["et_hour"] == 11) & (out["et_minute"] == 0))
    )
    out["is_nyam_entry_window"] = (
        ((out["et_hour"] == 9) & (out["et_minute"] >= 42))
        | (out["et_hour"] == 10)
        | ((out["et_hour"] == 11) & (out["et_minute"] <= 6))
    )

    out["is_nypm"] = (
        ((out["et_hour"] == 13) & (out["et_minute"] >= 30))
        | (out["et_hour"] == 14)
        | ((out["et_hour"] == 15) & (out["et_minute"] <= 0))
    )
    out["is_nypm_entry_window"] = (
        ((out["et_hour"] == 13) & (out["et_minute"] >= 36))
        | (out["et_hour"] == 14)
        | ((out["et_hour"] == 15) & (out["et_minute"] <= 0))
    )

    prior_session = out[(out["et_hour"] < 9) | ((out["et_hour"] == 9) & (out["et_minute"] < 30))].copy()
    session_levels = prior_session.groupby("et_date").agg(
        session_high=("High", "max"),
        session_low=("Low", "min"),
    )
    out = out.merge(session_levels, left_on="et_date", right_index=True, how="left")

    am_session = out[
        (((out["et_hour"] == 9) & (out["et_minute"] >= 30)) | (out["et_hour"] == 10) | (out["et_hour"] == 11))
        & ~((out["et_hour"] == 11) & (out["et_minute"] > 30))
    ].copy()
    am_levels = am_session.groupby("et_date").agg(
        am_high=("High", "max"),
        am_low=("Low", "min"),
    )
    out = out.merge(am_levels, left_on="et_date", right_index=True, how="left")

    day_agg = out.groupby("et_date").agg(day_high=("High", "max"), day_low=("Low", "min"))
    day_agg["prev_day_high"] = day_agg["day_high"].shift(1)
    day_agg["prev_day_low"] = day_agg["day_low"].shift(1)
    out = out.merge(day_agg[["prev_day_high", "prev_day_low"]], left_on="et_date", right_index=True, how="left")

    out["swept_session_low"] = out["Low"] < out["session_low"]
    out["swept_session_high"] = out["High"] > out["session_high"]
    out["reclaimed_session_low"] = out["Close"] > out["session_low"]
    out["rejected_session_high"] = out["Close"] < out["session_high"]

    out["swept_prev_day_low"] = out["Low"] < out["prev_day_low"]
    out["swept_prev_day_high"] = out["High"] > out["prev_day_high"]
    out["reclaimed_prev_day_low"] = out["Close"] > out["prev_day_low"]
    out["rejected_prev_day_high"] = out["Close"] < out["prev_day_high"]

    out["swept_am_low"] = out["Low"] < out["am_low"]
    out["swept_am_high"] = out["High"] > out["am_high"]
    out["reclaimed_am_low"] = out["Close"] > out["am_low"]
    out["rejected_am_high"] = out["Close"] < out["am_high"]

    return out


def build_3m_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr14_3m"] = atr(out, 14)

    body = (out["Close"] - out["Open"]).abs()
    rng = (out["High"] - out["Low"]).replace(0, np.nan)

    out["bull_disp_3m"] = (out["Close"] > out["Open"]) & (body > out["atr14_3m"] * 0.60)
    out["bear_disp_3m"] = (out["Close"] < out["Open"]) & (body > out["atr14_3m"] * 0.60)
    out["bull_close_strong_3m"] = ((out["Close"] - out["Low"]) / rng) >= 0.65
    out["bear_close_strong_3m"] = ((out["High"] - out["Close"]) / rng) >= 0.65

    out["bull_reentry_3m"] = (
        out["bull_close_strong_3m"]
        & (out["Low"] > out["Low"].shift(1))
        & (out["Close"] > out["Close"].rolling(5).mean())
        & (out["Close"] > out["High"].shift(1))
    )
    out["bear_reentry_3m"] = (
        out["bear_close_strong_3m"]
        & (out["High"] < out["High"].shift(1))
        & (out["Close"] < out["Close"].rolling(5).mean())
        & (out["Close"] < out["Low"].shift(1))
    )

    out["bull_overextended_3m"] = out["Close"] > (out["Close"].rolling(20).mean() + out["atr14_3m"] * 2.2)
    out["bear_overextended_3m"] = out["Close"] < (out["Close"].rolling(20).mean() - out["atr14_3m"] * 2.2)

    return out


def build_model_frame(df_3m: pd.DataFrame) -> pd.DataFrame:
    out = df_3m.copy()
    out = build_session_features(out)
    out = build_4h_context(out)
    out = build_30m_features(out)
    out = build_3m_features(out)
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


@dataclass
class SetupState:
    direction: str = ""
    setup_type: str = ""
    narrative_bar: int = -1
    context_bar: int = -1
    bridge_bar: int = -1
    active_until_bar: int = -1
    bridge_type: str = ""


class ICT_MULTI_SETUP_V41(Strategy):
    risk_multiple = 2.0
    min_target_points = 50.0
    stop_buffer_atr = 0.15
    fixed_size = 0.1

    narrative_expiry_bars = 180
    context_expiry_bars = 120
    bridge_expiry_bars = 54
    pending_expiry_bars = 8

    enable_nyam = True
    enable_nypm = True
    enable_reversal = True
    enable_continuation = True
    allow_reentry = False

    max_one_trade_per_day_per_direction = True

    last_debug_counts = {}
    last_trade_log = []

    def init(self):
        self.m = build_model_frame(self.data.df.copy())

        self.pending = PendingSignal()
        self.state = SetupState()
        self.active_risk = np.nan
        self.last_trade_day_direction = {}

        self.protection_activation_bar = -1
        self.planned_stop = np.nan
        self.planned_target = np.nan
        self.open_trade_meta = None

        self.prev_closed_count = 0

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
            "trigger_reentry_long": 0,
            "trigger_reentry_short": 0,
            "reject_overextended_long": 0,
            "reject_overextended_short": 0,
            "reject_late_session_long": 0,
            "reject_late_session_short": 0,
            "reject_tiny_risk_long": 0,
            "reject_tiny_risk_short": 0,
            "expire_pending": 0,
            "expire_state": 0,
            "clear_state_invalid_long": 0,
            "clear_state_invalid_short": 0,
            "protective_orders_armed": 0,
        }
        self.__class__.last_debug_counts = dict(self.debug_counts)
        self.__class__.last_trade_log = []

    def _sync_debug(self):
        self.__class__.last_debug_counts = dict(self.debug_counts)

    def _sync_trade_log(self):
        self.__class__.last_trade_log = list(self.__class__.last_trade_log)

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

    def _log_newly_closed_trades(self):
        try:
            closed = list(self.closed_trades)
        except Exception:
            return

        if len(closed) <= self.prev_closed_count:
            return

        new_items = closed[self.prev_closed_count :]
        for t in new_items:
            meta = self.open_trade_meta or {}
            try:
                self.__class__.last_trade_log.append(
                    {
                        "side": "LONG" if float(t.size) > 0 else "SHORT",
                        "setup_type": meta.get("setup_type", ""),
                        "bridge_type": meta.get("bridge_type", ""),
                        "entry_variant": meta.get("entry_variant", ""),
                        "entry_price": float(t.entry_price),
                        "exit_price": float(t.exit_price),
                        "entry_time": str(t.entry_time),
                        "exit_time": str(t.exit_time),
                        "return_pct": float(getattr(t, "pl_pct", np.nan)) if hasattr(t, "pl_pct") else np.nan,
                    }
                )
            except Exception:
                pass

        if new_items:
            self.open_trade_meta = None

        self.prev_closed_count = len(closed)
        self._sync_trade_log()

    def _can_trade_today(self, direction: str, row: pd.Series) -> bool:
        if not self.max_one_trade_per_day_per_direction:
            return True
        key = (row["et_date"], direction)
        return key not in self.last_trade_day_direction

    def _mark_trade_today(self, direction: str, row: pd.Series):
        self.last_trade_day_direction[(row["et_date"], direction)] = True

    def _apply_delayed_protection(self, i: int):
        trade = self._latest_trade()
        if trade is None or not self.position:
            return

        if self.protection_activation_bar < 0:
            return

        if i >= self.protection_activation_bar:
            try:
                if trade.sl is None and np.isfinite(self.planned_stop):
                    trade.sl = float(self.planned_stop)
                if trade.tp is None and np.isfinite(self.planned_target):
                    trade.tp = float(self.planned_target)
                self.debug_counts["protective_orders_armed"] += 1
                self._sync_debug()
            except Exception:
                pass
            self.protection_activation_bar = -1

    def _manage_trade(self):
        trade = self._latest_trade()
        if trade is None or not self.position or not np.isfinite(self.active_risk):
            return

        price = float(self.data.Close[-1])

        if self.position.is_long:
            r_now = (price - float(trade.entry_price)) / self.active_risk
            if r_now >= 1.0 and trade.sl is not None:
                try:
                    trade.sl = max(float(trade.sl), float(trade.entry_price))
                except Exception:
                    pass

        if self.position.is_short:
            r_now = (float(trade.entry_price) - price) / self.active_risk
            if r_now >= 1.0 and trade.sl is not None:
                try:
                    trade.sl = min(float(trade.sl), float(trade.entry_price))
                except Exception:
                    pass

    def _hybrid_target_above(self, i: int, entry: float, risk: float) -> float:
        row = self.m.iloc[i]
        rr_target = entry + (risk * self.risk_multiple)
        points_target = entry + self.min_target_points

        candidates = []
        for level_name in ("prev_day_high", "session_high", "am_high"):
            level = row.get(level_name, np.nan)
            if pd.notna(level) and float(level) > entry:
                candidates.append(float(level))

        liquidity_target = max(candidates) if candidates else np.nan

        final_target = max(rr_target, points_target)
        if pd.notna(liquidity_target):
            final_target = max(final_target, float(liquidity_target))
        return final_target

    def _hybrid_target_below(self, i: int, entry: float, risk: float) -> float:
        row = self.m.iloc[i]
        rr_target = entry - (risk * self.risk_multiple)
        points_target = entry - self.min_target_points

        candidates = []
        for level_name in ("prev_day_low", "session_low", "am_low"):
            level = row.get(level_name, np.nan)
            if pd.notna(level) and float(level) < entry:
                candidates.append(float(level))

        liquidity_target = min(candidates) if candidates else np.nan

        final_target = min(rr_target, points_target)
        if pd.notna(liquidity_target):
            final_target = min(final_target, float(liquidity_target))
        return final_target

    def _bull_narrative_ok(self, row: pd.Series) -> bool:
        return bool(
            row["bull_4h_bias"]
            and row["above_4h_eq"]
            and (row["bull_profile_4h"] or row["bull_disp_4h"] or row["bull_fvg_4h"])
        )

    def _bear_narrative_ok(self, row: pd.Series) -> bool:
        return bool(
            row["bear_4h_bias"]
            and row["below_4h_eq"]
            and (row["bear_profile_4h"] or row["bear_disp_4h"] or row["bear_fvg_4h"])
        )

    def _bull_context_reversal_nyam(self, row: pd.Series) -> bool:
        return bool(
            row["is_nyam"]
            and (
                (row["swept_session_low"] and row["reclaimed_session_low"])
                or (row["swept_prev_day_low"] and row["reclaimed_prev_day_low"])
            )
        )

    def _bear_context_reversal_nyam(self, row: pd.Series) -> bool:
        return bool(
            row["is_nyam"]
            and (
                (row["swept_session_high"] and row["rejected_session_high"])
                or (row["swept_prev_day_high"] and row["rejected_prev_day_high"])
            )
        )

    def _bull_context_continuation_nyam(self, row: pd.Series) -> bool:
        return bool(
            row["is_nyam"]
            and row["bull_4h_bias"]
            and row["above_4h_eq"]
            and (row["bull_disp_30m"] or row["bull_close_strong_30m"] or row["bull_cisd_30m"])
        )

    def _bear_context_continuation_nyam(self, row: pd.Series) -> bool:
        return bool(
            row["is_nyam"]
            and row["bear_4h_bias"]
            and row["below_4h_eq"]
            and (row["bear_disp_30m"] or row["bear_close_strong_30m"] or row["bear_cisd_30m"])
        )

    def _bull_context_reversal_nypm(self, row: pd.Series) -> bool:
        return bool(
            row["is_nypm"]
            and (
                (row["swept_am_low"] and row["reclaimed_am_low"])
                or (row["swept_prev_day_low"] and row["reclaimed_prev_day_low"])
            )
        )

    def _bear_context_reversal_nypm(self, row: pd.Series) -> bool:
        return bool(
            row["is_nypm"]
            and (
                (row["swept_am_high"] and row["rejected_am_high"])
                or (row["swept_prev_day_high"] and row["rejected_prev_day_high"])
            )
        )

    def _bull_context_continuation_nypm(self, row: pd.Series) -> bool:
        return bool(
            row["is_nypm"]
            and row["bull_4h_bias"]
            and row["above_4h_eq"]
            and (row["bull_disp_30m"] or row["bull_close_strong_30m"] or row["bull_cisd_30m"])
        )

    def _bear_context_continuation_nypm(self, row: pd.Series) -> bool:
        return bool(
            row["is_nypm"]
            and row["bear_4h_bias"]
            and row["below_4h_eq"]
            and (row["bear_disp_30m"] or row["bear_close_strong_30m"] or row["bear_cisd_30m"])
        )

    def _bull_bridge_type(self, row: pd.Series) -> str:
        if bool(row["bull_cisd_30m"]):
            return "CISD"
        if bool(row["bull_mss_30m"]):
            return "MSS"
        if bool(row["bull_ifvg_30m"]):
            return "iFVG"
        if bool(row["bull_c2_or_c3"]):
            return "C2C3"
        if bool(row["bull_disp_30m"]):
            return "DISP"
        if bool(row["bull_close_strong_30m"]):
            return "STRONG_CLOSE"
        return ""

    def _bear_bridge_type(self, row: pd.Series) -> str:
        if bool(row["bear_cisd_30m"]):
            return "CISD"
        if bool(row["bear_mss_30m"]):
            return "MSS"
        if bool(row["bear_ifvg_30m"]):
            return "iFVG"
        if bool(row["bear_c2_or_c3"]):
            return "C2C3"
        if bool(row["bear_disp_30m"]):
            return "DISP"
        if bool(row["bear_close_strong_30m"]):
            return "STRONG_CLOSE"
        return ""

    def _bull_bridge_ok(self, row: pd.Series) -> bool:
        return self._bull_bridge_type(row) != ""

    def _bear_bridge_ok(self, row: pd.Series) -> bool:
        return self._bear_bridge_type(row) != ""

    def _long_entry_window_ok(self, row: pd.Series) -> bool:
        return bool(row["is_nyam_entry_window"] or row["is_nypm_entry_window"])

    def _short_entry_window_ok(self, row: pd.Series) -> bool:
        return bool(row["is_nyam_entry_window"] or row["is_nypm_entry_window"])

    def _bull_execution_ok(self, row: pd.Series) -> tuple[bool, str]:
        if not self._long_entry_window_ok(row):
            return False, ""
        if bool(row["bull_overextended_3m"]):
            return False, ""
        if bool(row["bull_disp_3m"] and row["bull_close_strong_3m"]):
            return True, "DISP_STRONG"
        if self.allow_reentry and bool(row["bull_reentry_3m"]):
            return True, "REENTRY"
        return False, ""

    def _bear_execution_ok(self, row: pd.Series) -> tuple[bool, str]:
        if not self._short_entry_window_ok(row):
            return False, ""
        if bool(row["bear_overextended_3m"]):
            return False, ""
        if bool(row["bear_disp_3m"] and row["bear_close_strong_3m"]):
            return True, "DISP_STRONG"
        if self.allow_reentry and bool(row["bear_reentry_3m"]):
            return True, "REENTRY"
        return False, ""

    def _bull_invalid(self, row: pd.Series) -> bool:
        return bool((not row["above_4h_eq"]) or (not row["bull_4h_bias"]))

    def _bear_invalid(self, row: pd.Series) -> bool:
        return bool((not row["below_4h_eq"]) or (not row["bear_4h_bias"]))

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

    def next(self):
        self._log_newly_closed_trades()
        i = self._i()
        self._apply_delayed_protection(i)
        self._manage_trade()

        if i < 200:
            return

        row = self.m.iloc[i]
        close_now = float(self.data.Close[-1])
        high_now = float(self.data.High[-1])
        low_now = float(self.data.Low[-1])
        atr3 = float(row["atr14_3m"]) if pd.notna(row["atr14_3m"]) else 0.0

        if self.pending.expiry_bar >= 0 and i > self.pending.expiry_bar:
            self.debug_counts["expire_pending"] += 1
            self._sync_debug()
            self._clear_pending()

        if self.pending.direction == "long" and not self.position:
            if not self._long_entry_window_ok(row):
                self.debug_counts["reject_late_session_long"] += 1
                self._sync_debug()
                self._clear_pending()
                self._clear_state()
                return

            if close_now >= self.pending.entry_trigger:
                entry = close_now
                stop = float(self.pending.stop_price)
                target = float(self.pending.target_price)
                risk = entry - stop

                if risk <= 0:
                    self._clear_pending()
                    self._clear_state()
                    return

                if bool(row["bull_overextended_3m"]):
                    self.debug_counts["reject_overextended_long"] += 1
                    self._sync_debug()
                    self._clear_pending()
                    self._clear_state()
                    return

                if risk < max(atr3 * 0.35, 1.0):
                    self.debug_counts["reject_tiny_risk_long"] += 1
                    self._sync_debug()
                    self._clear_pending()
                    self._clear_state()
                    return

                if self._can_trade_today("long", row):
                    self.active_risk = risk
                    self.buy(size=self.fixed_size)
                    self.protection_activation_bar = i + 1
                    self.planned_stop = stop
                    self.planned_target = target
                    self.open_trade_meta = {
                        "setup_type": self.pending.setup_type,
                        "bridge_type": self.pending.bridge_type,
                        "entry_variant": self.pending.entry_variant,
                    }
                    self._mark_trade_today("long", row)
                    if self.pending.entry_variant == "REENTRY":
                        self.debug_counts["trigger_reentry_long"] += 1
                    else:
                        self.debug_counts["trigger_long_entry"] += 1
                    self._sync_debug()

                self._clear_pending()
                self._clear_state()
                return

        if self.pending.direction == "short" and not self.position:
            if not self._short_entry_window_ok(row):
                self.debug_counts["reject_late_session_short"] += 1
                self._sync_debug()
                self._clear_pending()
                self._clear_state()
                return

            if close_now <= self.pending.entry_trigger:
                entry = close_now
                stop = float(self.pending.stop_price)
                target = float(self.pending.target_price)
                risk = stop - entry

                if risk <= 0:
                    self._clear_pending()
                    self._clear_state()
                    return

                if bool(row["bear_overextended_3m"]):
                    self.debug_counts["reject_overextended_short"] += 1
                    self._sync_debug()
                    self._clear_pending()
                    self._clear_state()
                    return

                if risk < max(atr3 * 0.35, 1.0):
                    self.debug_counts["reject_tiny_risk_short"] += 1
                    self._sync_debug()
                    self._clear_pending()
                    self._clear_state()
                    return

                if self._can_trade_today("short", row):
                    self.active_risk = risk
                    self.sell(size=self.fixed_size)
                    self.protection_activation_bar = i + 1
                    self.planned_stop = stop
                    self.planned_target = target
                    self.open_trade_meta = {
                        "setup_type": self.pending.setup_type,
                        "bridge_type": self.pending.bridge_type,
                        "entry_variant": self.pending.entry_variant,
                    }
                    self._mark_trade_today("short", row)
                    if self.pending.entry_variant == "REENTRY":
                        self.debug_counts["trigger_reentry_short"] += 1
                    else:
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
                self.state.setup_type = "US_SESSION"
                self.debug_counts["arm_long_narrative"] += 1
                self._sync_debug()
                return

            if self._bear_narrative_ok(row):
                self.state.direction = "short"
                self.state.narrative_bar = i
                self.state.setup_type = "US_SESSION"
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
                context_hit = False

                if self.enable_nyam and self.enable_reversal and self._bull_context_reversal_nyam(row):
                    self.state.context_bar = i
                    self.state.setup_type = "NYAM_REVERSAL"
                    context_hit = True
                elif self.enable_nyam and self.enable_continuation and self._bull_context_continuation_nyam(row):
                    self.state.context_bar = i
                    self.state.setup_type = "NYAM_CONTINUATION"
                    context_hit = True
                elif self.enable_nypm and self.enable_reversal and self._bull_context_reversal_nypm(row):
                    self.state.context_bar = i
                    self.state.setup_type = "NYPM_REVERSAL"
                    context_hit = True
                elif self.enable_nypm and self.enable_continuation and self._bull_context_continuation_nypm(row):
                    self.state.context_bar = i
                    self.state.setup_type = "NYPM_CONTINUATION"
                    context_hit = True

                if context_hit:
                    self.debug_counts["confirm_long_context"] += 1
                    self._sync_debug()
                return

            if self.state.bridge_bar < 0:
                if self._bull_bridge_ok(row):
                    self.state.bridge_bar = i
                    self.state.bridge_type = self._bull_bridge_type(row)
                    self.state.active_until_bar = i + self.bridge_expiry_bars
                    self.debug_counts["confirm_long_bridge"] += 1
                    self._sync_debug()
                return

            ok, entry_variant = self._bull_execution_ok(row)
            if ok:
                entry_trigger = high_now
                stop = low_now - (atr3 * self.stop_buffer_atr if atr3 > 0 else 0.0)
                if stop < entry_trigger:
                    risk = entry_trigger - stop
                    target = self._hybrid_target_above(i, entry_trigger, risk)
                    self.pending.direction = "long"
                    self.pending.entry_trigger = entry_trigger
                    self.pending.stop_price = stop
                    self.pending.target_price = target
                    self.pending.expiry_bar = i + self.pending_expiry_bars
                    self.pending.setup_type = self.state.setup_type
                    self.pending.bridge_type = self.state.bridge_type
                    self.pending.entry_variant = entry_variant
                    self.debug_counts["arm_pending_long"] += 1
                    self._sync_debug()
                return

        if self.state.direction == "short":
            if self._bear_invalid(row):
                self.debug_counts["clear_state_invalid_short"] += 1
                self._sync_debug()
                self._clear_state()
                return

            if self.state.context_bar < 0:
                context_hit = False

                if self.enable_nyam and self.enable_reversal and self._bear_context_reversal_nyam(row):
                    self.state.context_bar = i
                    self.state.setup_type = "NYAM_REVERSAL"
                    context_hit = True
                elif self.enable_nyam and self.enable_continuation and self._bear_context_continuation_nyam(row):
                    self.state.context_bar = i
                    self.state.setup_type = "NYAM_CONTINUATION"
                    context_hit = True
                elif self.enable_nypm and self.enable_reversal and self._bear_context_reversal_nypm(row):
                    self.state.context_bar = i
                    self.state.setup_type = "NYPM_REVERSAL"
                    context_hit = True
                elif self.enable_nypm and self.enable_continuation and self._bear_context_continuation_nypm(row):
                    self.state.context_bar = i
                    self.state.setup_type = "NYPM_CONTINUATION"
                    context_hit = True

                if context_hit:
                    self.debug_counts["confirm_short_context"] += 1
                    self._sync_debug()
                return

            if self.state.bridge_bar < 0:
                if self._bear_bridge_ok(row):
                    self.state.bridge_bar = i
                    self.state.bridge_type = self._bear_bridge_type(row)
                    self.state.active_until_bar = i + self.bridge_expiry_bars
                    self.debug_counts["confirm_short_bridge"] += 1
                    self._sync_debug()
                return

            ok, entry_variant = self._bear_execution_ok(row)
            if ok:
                entry_trigger = low_now
                stop = high_now + (atr3 * self.stop_buffer_atr if atr3 > 0 else 0.0)
                if stop > entry_trigger:
                    risk = stop - entry_trigger
                    target = self._hybrid_target_below(i, entry_trigger, risk)
                    self.pending.direction = "short"
                    self.pending.entry_trigger = entry_trigger
                    self.pending.stop_price = stop
                    self.pending.target_price = target
                    self.pending.expiry_bar = i + self.pending_expiry_bars
                    self.pending.setup_type = self.state.setup_type
                    self.pending.bridge_type = self.state.bridge_type
                    self.pending.entry_variant = entry_variant
                    self.debug_counts["arm_pending_short"] += 1
                    self._sync_debug()
                return