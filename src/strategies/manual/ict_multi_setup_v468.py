from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.strategies.manual.ict_multi_setup_v452 import PendingSignal
from src.strategies.manual.ict_multi_setup_v467 import ICT_MULTI_SETUP_V467


class ICT_MULTI_SETUP_V468(ICT_MULTI_SETUP_V467):
    """
    V468
    - Built from V467.
    - Adds a hard projected-profit gate so every accepted trade must target at least
      $500 of projected gross profit at the intended contract size.
    - Keeps the V467 aggressive 5/10 MNQ structure, but rejects low-dollar setups
      before they ever arm.

    Rule added
    - projected_target_dollars >= 500 for every pending trade
    - projected_target_dollars = abs(target - entry) * 2 * contracts
      because MNQ is $2 per point per contract
    """

    name = "ICT_MULTI_SETUP_V468"

    MIN_PROJECTED_PROFIT_DOLLARS = 500.0

    def _default_meta(self) -> Dict[str, Any]:
        base = super()._default_meta()
        base["version"] = self.name
        base["v468_profile"] = "aggressive_with_min_500_projected_profit_gate"
        base["min_projected_profit_dollars"] = self.MIN_PROJECTED_PROFIT_DOLLARS
        return base

    def _planned_dynamic_contracts(
        self,
        *,
        tier: str,
        setup_type: str,
        bridge_type: str,
        stop_points: float | None,
        planned_rr: float | None,
    ) -> int:
        tier = str(tier or "").upper()
        setup_type = str(setup_type or "").upper()
        bridge_upper = str(bridge_type or "").upper()

        if stop_points is None or planned_rr is None:
            return self.BASE_REPORT_CONTRACTS

        if bridge_upper not in {b.upper() for b in self.STRONG_BRIDGES}:
            return self.BASE_REPORT_CONTRACTS
        if setup_type not in self.STRONG_SETUP_TYPES:
            return self.BASE_REPORT_CONTRACTS
        if stop_points > self.MAX_STOP_POINTS_FOR_SIZEUP:
            return self.BASE_REPORT_CONTRACTS
        if planned_rr < self.MIN_FAVORABLE_PLANNED_RR_FOR_SIZEUP:
            return self.BASE_REPORT_CONTRACTS

        if tier == "A":
            return self.MAX_REPORT_CONTRACTS

        if tier == "B":
            if setup_type in {"LONDON_CONTINUATION", "NYAM_CONTINUATION", "NYPM_CONTINUATION"}:
                if bridge_upper in {"IFVG", "C2C3", "MSS", "CISD"} and planned_rr >= 3.75 and stop_points <= 34.0:
                    return self.MAX_REPORT_CONTRACTS
            if setup_type == "ASIA_CONTINUATION":
                if bridge_upper in {"IFVG", "C2C3"} and planned_rr >= 4.5 and stop_points <= 30.0:
                    return self.MAX_REPORT_CONTRACTS

        return self.BASE_REPORT_CONTRACTS

    def _pending_dynamic_contracts(self) -> int:
        stop_points = getattr(self.pending, "stop_points", None)
        planned_rr = getattr(self.pending, "planned_rr", None)
        try:
            stop_points = float(stop_points) if stop_points is not None else None
        except Exception:
            stop_points = None
        try:
            planned_rr = float(planned_rr) if planned_rr is not None else None
        except Exception:
            planned_rr = None

        return self._planned_dynamic_contracts(
            tier=str(getattr(self.pending, "setup_tier", "") or ""),
            setup_type=str(getattr(self.pending, "setup_type", "") or ""),
            bridge_type=str(getattr(self.pending, "bridge_type", "") or ""),
            stop_points=stop_points,
            planned_rr=planned_rr,
        )

    def _projected_profit_dollars(self, entry_trigger: float, target_price: float, contracts: int) -> float:
        try:
            points = abs(float(target_price) - float(entry_trigger))
            return points * 2.0 * float(contracts)
        except Exception:
            return 0.0

    def _passes_min_profit_gate(
        self,
        *,
        entry_trigger: float,
        target_price: float,
        contracts: int,
        debug_key: str,
    ) -> bool:
        projected_profit = self._projected_profit_dollars(entry_trigger, target_price, contracts)
        if projected_profit + 1e-9 < self.MIN_PROJECTED_PROFIT_DOLLARS:
            type(self).DEBUG_COUNTERS[debug_key] += 1
            type(self).DEBUG_COUNTERS["reject_min_projected_profit_500"] += 1
            return False
        return True

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
        stop_points = abs(entry_trigger - stop)
        tp_points = abs(runner_target - entry_trigger)
        planned_rr = (tp_points / stop_points) if stop_points > 0 else None
        contracts = self._planned_dynamic_contracts(
            tier=tier,
            setup_type=self.state.setup_type,
            bridge_type=self.state.bridge_type,
            stop_points=stop_points,
            planned_rr=planned_rr,
        )

        if not self._passes_min_profit_gate(
            entry_trigger=entry_trigger,
            target_price=runner_target,
            contracts=contracts,
            debug_key="reject_min_projected_profit_500_long",
        ):
            self._sync_debug()
            self._clear_state()
            return

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
        self.pending.stop_points = stop_points
        self.pending.tp_points = tp_points
        self.pending.planned_rr = planned_rr
        self.pending.projected_profit_dollars = self._projected_profit_dollars(entry_trigger, runner_target, contracts)
        self.pending.projected_profit_contracts = contracts
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
        stop_points = abs(stop - entry_trigger)
        tp_points = abs(entry_trigger - runner_target)
        planned_rr = (tp_points / stop_points) if stop_points > 0 else None
        contracts = self._planned_dynamic_contracts(
            tier=tier,
            setup_type=self.state.setup_type,
            bridge_type=self.state.bridge_type,
            stop_points=stop_points,
            planned_rr=planned_rr,
        )

        if not self._passes_min_profit_gate(
            entry_trigger=entry_trigger,
            target_price=runner_target,
            contracts=contracts,
            debug_key="reject_min_projected_profit_500_short",
        ):
            self._sync_debug()
            self._clear_state()
            return

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
        self.pending.stop_points = stop_points
        self.pending.tp_points = tp_points
        self.pending.planned_rr = planned_rr
        self.pending.projected_profit_dollars = self._projected_profit_dollars(entry_trigger, runner_target, contracts)
        self.pending.projected_profit_contracts = contracts
        self.debug_counts["arm_pending_short"] += 1
        self._sync_debug()

    def _record_open_trade_meta(self, entry: float):
        super()._record_open_trade_meta(entry)
        if not isinstance(self.open_trade_meta, dict):
            self.open_trade_meta = {}

        planned_contracts = self._pending_dynamic_contracts()
        self.open_trade_meta["version"] = self.name
        self.open_trade_meta["v468_profile"] = "aggressive_with_min_500_projected_profit_gate"
        self.open_trade_meta["min_projected_profit_dollars"] = self.MIN_PROJECTED_PROFIT_DOLLARS
        self.open_trade_meta["projected_profit_dollars"] = self._projected_profit_dollars(
            entry,
            getattr(self.pending, "target_price", entry),
            planned_contracts,
        )
        self.open_trade_meta["projected_profit_contracts"] = int(planned_contracts)
