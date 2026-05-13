
from __future__ import annotations

"""
Exact-ish Top/Bottom Ticking adapter for the shared event engine.

This adapter ports the real signal state-machine from:
  src/strategies/manual/ict_top_bottom_ticking.py

It does NOT use Backtesting.py execution. It only reuses the model's feature builder
and signal rules, then returns an OrderPlan to the new event engine.

Execution, sizing, SL/TP, commissions, prop rules, news blackout and payouts are
handled by the shared event engine / lifecycle simulator.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from src.backtesting.event_engine.models import OrderPlan, SymbolSpec, PropProfile

from src.strategies.manual.ict_top_bottom_ticking import (
    build_model_frame,
    PendingSetup,
    ICT_TOP_BOTTOM_TICKING,
)


@dataclass
class SymbolState:
    pending: PendingSetup


class TopBottomTickingAdapter:
    name = "top_bottom_ticking_exact"

    def __init__(self):
        self.states: Dict[str, SymbolState] = {}

        # Pull exact default params from the existing model class.
        self.min_warmup_bars = int(getattr(ICT_TOP_BOTTOM_TICKING, "min_warmup_bars", 150))
        self.setup_expiry_bars = int(getattr(ICT_TOP_BOTTOM_TICKING, "setup_expiry_bars", 18))
        self.require_cos_confirmation = bool(getattr(ICT_TOP_BOTTOM_TICKING, "require_cos_confirmation", True))
        self.require_internal_sweep_filter = bool(getattr(ICT_TOP_BOTTOM_TICKING, "require_internal_sweep_filter", False))

        self.tick_size = float(getattr(ICT_TOP_BOTTOM_TICKING, "tick_size", 0.25))
        self.min_sweep_points = float(getattr(ICT_TOP_BOTTOM_TICKING, "min_sweep_points", 0.25))
        self.retest_tolerance_points = float(getattr(ICT_TOP_BOTTOM_TICKING, "retest_tolerance_points", 0.25))
        self.stop_buffer_points = float(getattr(ICT_TOP_BOTTOM_TICKING, "stop_buffer_points", 0.25))
        self.max_zone_width_points = float(getattr(ICT_TOP_BOTTOM_TICKING, "max_zone_width_points", np.inf))

        self.target1_r = float(getattr(ICT_TOP_BOTTOM_TICKING, "target1_r", 1.0))
        self.target2_r = float(getattr(ICT_TOP_BOTTOM_TICKING, "target2_r", 2.25))
        self.target3_r = float(getattr(ICT_TOP_BOTTOM_TICKING, "target3_r", 4.25))

        self.min_stop_points = float(getattr(ICT_TOP_BOTTOM_TICKING, "min_stop_points", 6.0))
        self.max_stop_points = float(getattr(ICT_TOP_BOTTOM_TICKING, "max_stop_points", 30.0))

    def _state(self, symbol: str) -> SymbolState:
        if symbol not in self.states:
            self.states[symbol] = SymbolState(pending=PendingSetup())
        return self.states[symbol]

    def build_features(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        return build_model_frame(df.copy(), tick_size=self.tick_size)

    def _short_sweep_distance(self, row: pd.Series) -> float:
        ext = row.get("external_buyside", np.nan)
        if pd.isna(ext):
            return 0.0
        return max(0.0, float(row["High"]) - float(ext))

    def _long_sweep_distance(self, row: pd.Series) -> float:
        ext = row.get("external_sellside", np.nan)
        if pd.isna(ext):
            return 0.0
        return max(0.0, float(ext) - float(row["Low"]))

    def _short_zone(self, row: pd.Series) -> tuple[float, float, float, float]:
        zone_high = float(row["High"])
        zone_low = min(float(row["Open"]), float(row["Close"]))
        entry_ce = (zone_high + zone_low) / 2.0
        zone_width = zone_high - zone_low
        return zone_high, zone_low, entry_ce, zone_width

    def _long_zone(self, row: pd.Series) -> tuple[float, float, float, float]:
        zone_low = float(row["Low"])
        zone_high = max(float(row["Open"]), float(row["Close"]))
        entry_ce = (zone_high + zone_low) / 2.0
        zone_width = zone_high - zone_low
        return zone_low, zone_high, entry_ce, zone_width

    def _arm_short_from_sweep(self, state: SymbolState, row: pd.Series, i: int):
        if self.require_internal_sweep_filter and not bool(row.get("internal_sweep_short", False)):
            return

        sweep_distance = self._short_sweep_distance(row)
        if sweep_distance < self.min_sweep_points:
            return

        zone_high, zone_low, entry_ce, zone_width = self._short_zone(row)
        if zone_width > self.max_zone_width_points:
            return

        stop = zone_high + self.stop_buffer_points
        risk = stop - entry_ce
        if risk < self.min_stop_points or risk > self.max_stop_points:
            return

        state.pending = PendingSetup(
            direction="short",
            created_bar=i,
            expiry_bar=i + self.setup_expiry_bars,
            entry_ce=entry_ce,
            zone_high=zone_high,
            zone_low=zone_low,
            stop_price=stop,
            target1=entry_ce - (risk * self.target1_r),
            target2=entry_ce - (risk * self.target2_r),
            target3=entry_ce - (risk * self.target3_r),
            external_level=float(row.get("external_buyside", np.nan)),
            setup_type="TYPE2_SHORT_TOP_TICK",
            entry_variant="CE_LIMIT" if not self.require_cos_confirmation else "CE_PLUS_COS",
            internal_sweep=bool(row.get("internal_sweep_short", False)),
        )

    def _arm_long_from_sweep(self, state: SymbolState, row: pd.Series, i: int):
        if self.require_internal_sweep_filter and not bool(row.get("internal_sweep_long", False)):
            return

        sweep_distance = self._long_sweep_distance(row)
        if sweep_distance < self.min_sweep_points:
            return

        zone_low, zone_high, entry_ce, zone_width = self._long_zone(row)
        if zone_width > self.max_zone_width_points:
            return

        stop = zone_low - self.stop_buffer_points
        risk = entry_ce - stop
        if risk < self.min_stop_points or risk > self.max_stop_points:
            return

        state.pending = PendingSetup(
            direction="long",
            created_bar=i,
            expiry_bar=i + self.setup_expiry_bars,
            entry_ce=entry_ce,
            zone_high=zone_high,
            zone_low=zone_low,
            stop_price=stop,
            target1=entry_ce + (risk * self.target1_r),
            target2=entry_ce + (risk * self.target2_r),
            target3=entry_ce + (risk * self.target3_r),
            external_level=float(row.get("external_sellside", np.nan)),
            setup_type="TYPE2_LONG_BOTTOM_TICK",
            entry_variant="CE_LIMIT" if not self.require_cos_confirmation else "CE_PLUS_COS",
            internal_sweep=bool(row.get("internal_sweep_long", False)),
        )

    def _pending_short_ready(self, pending: PendingSetup, row: pd.Series) -> bool:
        touched = float(row["High"]) >= (float(pending.entry_ce) - self.retest_tolerance_points)
        if not touched:
            return False

        if self.require_cos_confirmation:
            return bool(row.get("cos_short", False)) and float(row["Close"]) < float(pending.entry_ce)

        return float(row["Close"]) <= float(pending.entry_ce)

    def _pending_long_ready(self, pending: PendingSetup, row: pd.Series) -> bool:
        touched = float(row["Low"]) <= (float(pending.entry_ce) + self.retest_tolerance_points)
        if not touched:
            return False

        if self.require_cos_confirmation:
            return bool(row.get("cos_long", False)) and float(row["Close"]) > float(pending.entry_ce)

        return float(row["Close"]) >= float(pending.entry_ce)

    def signal_for_row(self, symbol: str, row: pd.Series, history: pd.DataFrame, spec: SymbolSpec, profile: PropProfile, args):
        i = len(history) - 1
        if i < self.min_warmup_bars:
            return None

        state = self._state(symbol)
        pending = state.pending

        # Pending expiry.
        if pending.expiry_bar >= 0 and i > pending.expiry_bar:
            state.pending = PendingSetup()
            pending = state.pending

        # Existing pending short entry.
        if pending.direction == "short":
            if self._pending_short_ready(pending, row):
                entry = min(float(row["Close"]), float(pending.entry_ce))
                stop = float(pending.stop_price)
                risk = stop - entry
                if risk > 0 and np.isfinite(risk):
                    target = entry - (risk * self.target3_r)
                    state.pending = PendingSetup()
                    return OrderPlan(
                        symbol=symbol,
                        side="SHORT",
                        entry_price=entry,
                        stop_price=stop,
                        target_price=target,
                        trade_type=pending.setup_type,
                        reason=pending.entry_variant,
                        strategy_name=self.name,
                        setup_score=0.0,
                    )
            return None

        # Existing pending long entry.
        if pending.direction == "long":
            if self._pending_long_ready(pending, row):
                entry = max(float(row["Close"]), float(pending.entry_ce))
                stop = float(pending.stop_price)
                risk = entry - stop
                if risk > 0 and np.isfinite(risk):
                    target = entry + (risk * self.target3_r)
                    state.pending = PendingSetup()
                    return OrderPlan(
                        symbol=symbol,
                        side="LONG",
                        entry_price=entry,
                        stop_price=stop,
                        target_price=target,
                        trade_type=pending.setup_type,
                        reason=pending.entry_variant,
                        strategy_name=self.name,
                        setup_score=0.0,
                    )
            return None

        # No pending: look for new top/bottom tick sweep.
        if bool(row.get("sweep_short", False)):
            self._arm_short_from_sweep(state, row, i)
            return None

        if bool(row.get("sweep_long", False)):
            self._arm_long_from_sweep(state, row, i)
            return None

        return None
