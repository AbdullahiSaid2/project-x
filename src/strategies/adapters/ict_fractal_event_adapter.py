
from __future__ import annotations

"""
Exact-ish ICT Fractal / V473 adapter for the shared event engine.

Ports the actual V45.2/V473 state machine from:
  src/strategies/manual/ict_multi_setup_v452.py
  src/strategies/manual/v473_shared.py

This adapter reuses the real V45.2 feature builder and implements the V473 next()
state flow, then returns OrderPlan to the new event engine.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from src.backtesting.event_engine.models import OrderPlan, SymbolSpec, PropProfile
from src.strategies.manual.ict_multi_setup_v452 import (
    build_model_frame,
    PendingSignal,
    SetupState,
    ICT_MULTI_SETUP_V452,
)


@dataclass
class SymbolState:
    pending: PendingSignal
    setup: SetupState


class ICTFractalAdapter:
    name = "ict_fractal_v473_exact"

    def __init__(self):
        self.states: Dict[str, SymbolState] = {}

        # Core parameters from ICT_MULTI_SETUP_V452 / v473 shared config style.
        self.risk_multiple = float(getattr(ICT_MULTI_SETUP_V452, "risk_multiple", 2.0))
        self.min_target_points = float(getattr(ICT_MULTI_SETUP_V452, "min_target_points", 50.0))
        self.min_stop_points = float(getattr(ICT_MULTI_SETUP_V452, "min_stop_points", 25.0))
        self.stop_buffer_atr = float(getattr(ICT_MULTI_SETUP_V452, "stop_buffer_atr", 0.15))
        self.partial_rr = float(getattr(ICT_MULTI_SETUP_V452, "partial_rr", 1.0))
        self.pullback_entry_tolerance_points = float(getattr(ICT_MULTI_SETUP_V452, "pullback_entry_tolerance_points", 6.0))
        self.pending_expiry_bars = int(getattr(ICT_MULTI_SETUP_V452, "pending_expiry_bars", 18))

        self.narrative_expiry_bars = int(getattr(ICT_MULTI_SETUP_V452, "narrative_expiry_bars", 240))
        self.context_expiry_bars = int(getattr(ICT_MULTI_SETUP_V452, "context_expiry_bars", 180))
        self.bridge_expiry_bars = int(getattr(ICT_MULTI_SETUP_V452, "bridge_expiry_bars", 90))

        self.enable_asia_continuation = bool(getattr(ICT_MULTI_SETUP_V452, "enable_asia_continuation", True))
        self.enable_london_continuation = bool(getattr(ICT_MULTI_SETUP_V452, "enable_london_continuation", True))
        self.enable_nyam_continuation = bool(getattr(ICT_MULTI_SETUP_V452, "enable_nyam_continuation", True))
        self.enable_nypm_continuation = bool(getattr(ICT_MULTI_SETUP_V452, "enable_nypm_continuation", True))

        self.allow_cisd_london = bool(getattr(ICT_MULTI_SETUP_V452, "allow_cisd_london", True))
        self.allow_cisd_asia = bool(getattr(ICT_MULTI_SETUP_V452, "allow_cisd_asia", False))
        self.allow_cisd_nyam = bool(getattr(ICT_MULTI_SETUP_V452, "allow_cisd_nyam", False))
        self.allow_cisd_nypm = bool(getattr(ICT_MULTI_SETUP_V452, "allow_cisd_nypm", False))
        self.allow_ifvg_asia = bool(getattr(ICT_MULTI_SETUP_V452, "allow_ifvg_asia", True))
        self.allow_ifvg_london = bool(getattr(ICT_MULTI_SETUP_V452, "allow_ifvg_london", True))
        self.allow_ifvg_nyam = bool(getattr(ICT_MULTI_SETUP_V452, "allow_ifvg_nyam", True))
        self.allow_ifvg_nypm = bool(getattr(ICT_MULTI_SETUP_V452, "allow_ifvg_nypm", True))

        self.prop_trade_only_ranked_setups = bool(getattr(ICT_MULTI_SETUP_V452, "prop_trade_only_ranked_setups", True))
        self.allowed_prop_setup_tiers = set(getattr(ICT_MULTI_SETUP_V452, "allowed_prop_setup_tiers", {"A", "B"}))

    def _state(self, symbol: str) -> SymbolState:
        if symbol not in self.states:
            self.states[symbol] = SymbolState(pending=PendingSignal(), setup=SetupState())
        return self.states[symbol]

    def build_features(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        return build_model_frame(df.copy())

    def _bull_narrative_ok(self, row: pd.Series) -> bool:
        return bool(row["bull_4h_bias"] and row["above_4h_eq"] and (row["bull_profile_4h"] or row["bull_disp_4h"] or row["bull_fvg_4h"]))

    def _bear_narrative_ok(self, row: pd.Series) -> bool:
        return bool(row["bear_4h_bias"] and row["below_4h_eq"] and (row["bear_profile_4h"] or row["bear_disp_4h"] or row["bear_fvg_4h"]))

    def _bull_context_continuation_asia(self, row): return bool(row["is_asia"] and row["bull_4h_bias"] and row["above_4h_eq"] and (row["bull_disp_30m"] or row["bull_close_strong_30m"] or row["bull_cisd_30m"]))
    def _bear_context_continuation_asia(self, row): return bool(row["is_asia"] and row["bear_4h_bias"] and row["below_4h_eq"] and (row["bear_disp_30m"] or row["bear_close_strong_30m"] or row["bear_cisd_30m"]))
    def _bull_context_continuation_london(self, row): return bool(row["is_london"] and row["bull_4h_bias"] and row["above_4h_eq"] and (row["bull_disp_30m"] or row["bull_close_strong_30m"] or row["bull_cisd_30m"]))
    def _bear_context_continuation_london(self, row): return bool(row["is_london"] and row["bear_4h_bias"] and row["below_4h_eq"] and (row["bear_disp_30m"] or row["bear_close_strong_30m"] or row["bear_cisd_30m"]))

    def _bull_context_continuation_nyam(self, row):
        return bool(row["is_nyam"] and row["bull_4h_bias"] and row["above_4h_eq"] and (row["swept_prior_us_low"] or row["reclaimed_prior_us_low"] or row["swept_prev_day_low"]) and (row["bull_disp_30m"] or row["bull_close_strong_30m"] or row["bull_cisd_30m"]))

    def _bear_context_continuation_nyam(self, row):
        return bool(row["is_nyam"] and row["bear_4h_bias"] and row["below_4h_eq"] and (row["swept_prior_us_high"] or row["rejected_prior_us_high"] or row["swept_prev_day_high"]) and (row["bear_disp_30m"] or row["bear_close_strong_30m"] or row["bear_cisd_30m"]))

    def _bull_context_continuation_nypm(self, row):
        return bool(row["is_nypm"] and row["bull_4h_bias"] and row["above_4h_eq"] and (row["reclaimed_nyam_low"] or row["reclaimed_prior_us_low"] or row["swept_nyam_low"]) and (row["bull_disp_30m"] or row["bull_close_strong_30m"] or row["bull_cisd_30m"]))

    def _bear_context_continuation_nypm(self, row):
        return bool(row["is_nypm"] and row["bear_4h_bias"] and row["below_4h_eq"] and (row["rejected_nyam_high"] or row["rejected_prior_us_high"] or row["swept_nyam_high"]) and (row["bear_disp_30m"] or row["bear_close_strong_30m"] or row["bear_cisd_30m"]))

    def _long_entry_window_ok(self, row): return bool(row["is_asia_entry_window"] or row["is_london_entry_window"] or row["is_nyam_entry_window"] or row["is_nypm_entry_window"])
    def _short_entry_window_ok(self, row): return self._long_entry_window_ok(row)
    def _bull_invalid(self, row): return bool((not row["above_4h_eq"]) or (not row["bull_4h_bias"]))
    def _bear_invalid(self, row): return bool((not row["below_4h_eq"]) or (not row["bear_4h_bias"]))
    def _bull_execution_ok(self, row): return bool(self._long_entry_window_ok(row) and row["bull_disp_3m"] and row["bull_close_strong_3m"] and not row["bull_overextended_3m"])
    def _bear_execution_ok(self, row): return bool(self._short_entry_window_ok(row) and row["bear_disp_3m"] and row["bear_close_strong_3m"] and not row["bear_overextended_3m"])
    def _1m_refine_long_ok(self, row): return bool(row["bull_cisd_1m"] or row["bull_mss_1m"] or row["bull_ifvg_1m"])
    def _1m_refine_short_ok(self, row): return bool(row["bear_cisd_1m"] or row["bear_mss_1m"] or row["bear_ifvg_1m"])

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
            if setup_type.startswith("LONDON_"): return self.allow_cisd_london
            if setup_type.startswith("ASIA_"): return self.allow_cisd_asia
            if setup_type.startswith("NYAM_"): return self.allow_cisd_nyam
            if setup_type.startswith("NYPM_"): return self.allow_cisd_nypm
        if bridge_type == "iFVG":
            if setup_type.startswith("ASIA_"): return self.allow_ifvg_asia
            if setup_type.startswith("LONDON_"): return self.allow_ifvg_london
            if setup_type.startswith("NYAM_"): return self.allow_ifvg_nyam
            if setup_type.startswith("NYPM_"): return self.allow_ifvg_nypm
        return True

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

    def _expire_state_if_needed(self, s: SymbolState, i: int):
        state = s.setup
        if state.direction == "":
            return
        if state.narrative_bar >= 0 and i - state.narrative_bar > self.narrative_expiry_bars:
            s.setup = SetupState(); return
        if state.context_bar >= 0 and i - state.context_bar > self.context_expiry_bars:
            s.setup = SetupState(); return
        if state.bridge_bar >= 0 and i > state.active_until_bar:
            s.setup = SetupState(); return

    def _arm_long_pullback(self, s: SymbolState, row: pd.Series, high_now: float, low_now: float, atr3: float, i: int):
        state = s.setup
        if not self._bridge_allowed_for_session(state.setup_type, state.bridge_type):
            s.setup = SetupState(); return

        tier = self._setup_tier(state.setup_type, state.bridge_type, "LONG")
        if self.prop_trade_only_ranked_setups and tier not in self.allowed_prop_setup_tiers:
            s.setup = SetupState(); return

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

        s.pending = PendingSignal(
            direction="long",
            entry_trigger=entry_trigger,
            stop_price=stop,
            target_price=runner_target,
            expiry_bar=i + self.pending_expiry_bars,
            setup_type=state.setup_type,
            bridge_type=state.bridge_type,
            entry_variant="PULLBACK_1M",
            pullback_low=pullback_low,
            pullback_high=pullback_high,
            partial_target=partial_target,
            runner_target=runner_target,
            setup_tier=tier,
        )

    def _arm_short_pullback(self, s: SymbolState, row: pd.Series, high_now: float, low_now: float, atr3: float, i: int):
        state = s.setup
        if not self._bridge_allowed_for_session(state.setup_type, state.bridge_type):
            s.setup = SetupState(); return

        tier = self._setup_tier(state.setup_type, state.bridge_type, "SHORT")
        if self.prop_trade_only_ranked_setups and tier not in self.allowed_prop_setup_tiers:
            s.setup = SetupState(); return

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

        s.pending = PendingSignal(
            direction="short",
            entry_trigger=entry_trigger,
            stop_price=stop,
            target_price=runner_target,
            expiry_bar=i + self.pending_expiry_bars,
            setup_type=state.setup_type,
            bridge_type=state.bridge_type,
            entry_variant="PULLBACK_1M",
            pullback_low=pullback_low,
            pullback_high=pullback_high,
            partial_target=partial_target,
            runner_target=runner_target,
            setup_tier=tier,
        )

    def _pending_long_ready(self, pending: PendingSignal, row: pd.Series, high_now: float, low_now: float) -> bool:
        if not self._1m_refine_long_ok(row):
            return False
        zone_low = pending.pullback_low - self.pullback_entry_tolerance_points
        zone_high = pending.pullback_high + self.pullback_entry_tolerance_points
        touched = (low_now <= zone_high) and (high_now >= zone_low)
        return bool(touched and high_now >= pending.entry_trigger)

    def _pending_short_ready(self, pending: PendingSignal, row: pd.Series, high_now: float, low_now: float) -> bool:
        if not self._1m_refine_short_ok(row):
            return False
        zone_low = pending.pullback_low - self.pullback_entry_tolerance_points
        zone_high = pending.pullback_high + self.pullback_entry_tolerance_points
        touched = (low_now <= zone_high) and (high_now >= zone_low)
        return bool(touched and low_now <= pending.entry_trigger)

    def signal_for_row(self, symbol: str, row: pd.Series, history: pd.DataFrame, spec: SymbolSpec, profile: PropProfile, args):
        i = len(history) - 1
        if i < 200:
            return None

        s = self._state(symbol)
        pending = s.pending
        state = s.setup

        close_now = float(row["Close"])
        high_now = float(row["High"])
        low_now = float(row["Low"])
        atr3 = float(row["atr14_3m"]) if pd.notna(row.get("atr14_3m")) else 0.0

        # Pending expiry.
        if pending.expiry_bar >= 0 and i > pending.expiry_bar:
            s.pending = PendingSignal()
            pending = s.pending

        # Trigger pending long.
        if pending.direction == "long":
            if self._pending_long_ready(pending, row, high_now, low_now):
                entry = max(close_now, float(pending.entry_trigger))
                risk = entry - float(pending.stop_price)
                if risk > 0:
                    order = OrderPlan(
                        symbol=symbol,
                        side="LONG",
                        entry_price=entry,
                        stop_price=float(pending.stop_price),
                        target_price=float(pending.runner_target),
                        trade_type=f"{pending.setup_type}_{pending.bridge_type}_{pending.setup_tier}_LONG",
                        reason=pending.entry_variant,
                        strategy_name=self.name,
                        setup_score=0.0,
                    )
                    s.pending = PendingSignal()
                    s.setup = SetupState()
                    return order
                s.pending = PendingSignal()
                s.setup = SetupState()
            return None

        # Trigger pending short.
        if pending.direction == "short":
            if self._pending_short_ready(pending, row, high_now, low_now):
                entry = min(close_now, float(pending.entry_trigger))
                risk = float(pending.stop_price) - entry
                if risk > 0:
                    order = OrderPlan(
                        symbol=symbol,
                        side="SHORT",
                        entry_price=entry,
                        stop_price=float(pending.stop_price),
                        target_price=float(pending.runner_target),
                        trade_type=f"{pending.setup_type}_{pending.bridge_type}_{pending.setup_tier}_SHORT",
                        reason=pending.entry_variant,
                        strategy_name=self.name,
                        setup_score=0.0,
                    )
                    s.pending = PendingSignal()
                    s.setup = SetupState()
                    return order
                s.pending = PendingSignal()
                s.setup = SetupState()
            return None

        self._expire_state_if_needed(s, i)
        state = s.setup

        # Narrative.
        if state.direction == "":
            if self._bull_narrative_ok(row):
                state.direction = "long"; state.narrative_bar = i; state.setup_type = "GLOBAL"
                return None
            if self._bear_narrative_ok(row):
                state.direction = "short"; state.narrative_bar = i; state.setup_type = "GLOBAL"
                return None

        # Long state.
        if state.direction == "long":
            if self._bull_invalid(row):
                s.setup = SetupState(); return None

            if state.context_bar < 0:
                hit = False
                if self.enable_asia_continuation and self._bull_context_continuation_asia(row):
                    state.context_bar = i; state.setup_type = "ASIA_CONTINUATION"; hit = True
                elif self.enable_london_continuation and self._bull_context_continuation_london(row):
                    state.context_bar = i; state.setup_type = "LONDON_CONTINUATION"; hit = True
                elif self.enable_nypm_continuation and self._bull_context_continuation_nypm(row):
                    state.context_bar = i; state.setup_type = "NYPM_CONTINUATION"; hit = True
                return None

            if state.bridge_bar < 0:
                bridge_type = str(row.get("bridge_type_30m", ""))
                bull_cisd_ok = bool(row["bull_cisd_30m"] and row["bull_disp_30m"] and row["bull_close_strong_30m"] and state.setup_type.startswith("LONDON_"))
                bull_ifvg_ok = bool(row["bull_ifvg_30m"] and row["bull_disp_30m"] and row["bull_close_strong_30m"])
                bull_ok = bool(bull_cisd_ok or row["bull_mss_30m"] or bull_ifvg_ok or row["bull_c2_or_c3"])
                if bull_ok and bridge_type:
                    state.bridge_bar = i
                    state.bridge_type = bridge_type
                    state.bridge_low = float(row.get("bull_bridge_low_30m", np.nan))
                    state.bridge_high = float(row.get("bull_bridge_high_30m", np.nan))
                    state.active_until_bar = i + self.bridge_expiry_bars
                return None

            if self._bull_execution_ok(row):
                self._arm_long_pullback(s, row, high_now, low_now, atr3, i)
                return None

        # Short state.
        if state.direction == "short":
            if self._bear_invalid(row):
                s.setup = SetupState(); return None

            if state.context_bar < 0:
                if self.enable_asia_continuation and self._bear_context_continuation_asia(row):
                    state.context_bar = i; state.setup_type = "ASIA_CONTINUATION"
                elif self.enable_london_continuation and self._bear_context_continuation_london(row):
                    state.context_bar = i; state.setup_type = "LONDON_CONTINUATION"
                elif self.enable_nyam_continuation and self._bear_context_continuation_nyam(row):
                    state.context_bar = i; state.setup_type = "NYAM_CONTINUATION"
                elif self.enable_nypm_continuation and self._bear_context_continuation_nypm(row):
                    state.context_bar = i; state.setup_type = "NYPM_CONTINUATION"
                return None

            if state.bridge_bar < 0:
                bridge_type = str(row.get("bridge_type_30m", ""))
                bear_cisd_ok = bool(row["bear_cisd_30m"] and row["bear_disp_30m"] and row["bear_close_strong_30m"] and state.setup_type.startswith("LONDON_"))
                bear_ifvg_ok = bool(row["bear_ifvg_30m"] and row["bear_disp_30m"] and row["bear_close_strong_30m"])
                bear_ok = bool(bear_cisd_ok or row["bear_mss_30m"] or bear_ifvg_ok or row["bear_c2_or_c3"])
                if bear_ok and bridge_type:
                    state.bridge_bar = i
                    state.bridge_type = bridge_type
                    state.bridge_low = float(row.get("bear_bridge_low_30m", np.nan))
                    state.bridge_high = float(row.get("bear_bridge_high_30m", np.nan))
                    state.active_until_bar = i + self.bridge_expiry_bars
                return None

            if self._bear_execution_ok(row):
                self._arm_short_pullback(s, row, high_now, low_now, atr3, i)
                return None

        return None
