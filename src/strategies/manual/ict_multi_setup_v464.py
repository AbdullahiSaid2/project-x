from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict

from src.strategies.manual.ict_multi_setup_v463 import ICT_MULTI_SETUP_V463


class ICT_MULTI_SETUP_V464(ICT_MULTI_SETUP_V463):
    """
    V464
    - Real follow-up to V463 with actual trade selection changes, not just renamed outputs.
    - Keeps the stable V460/V462/V463 core intact.
    - Prunes the weakest buckets that diluted avg trade dollars.
    - Keeps the user's requested 5 MNQ / 10 MNQ regime only.
    - Makes 10 MNQ qualification materially stricter so size-up is concentrated on the best setups.

    Main changes versus V463
    1) Weak bucket pruning
       - NYPM_CONTINUATION + C2C3 is demoted out of ranked execution.
       - CISD is fully demoted out of ranked execution.
       - LONDON_CONTINUATION + iFVG is demoted from A to B so it can still trade,
         but it no longer gets treated as a primary strong bucket.

    2) Strong-only 10 MNQ sizing
       - 10 MNQ only on a narrower set of A-tier continuation structures in the most liquid sessions.
       - Harder RR and stop filters than V463.

    3) Better runner bias on quality setups
       - Slightly more ambitious trade management to improve dollar capture on winners.
    """

    name = "ICT_MULTI_SETUP_V464"

    # Slightly stronger runner profile than V460/V463.
    risk_multiple = 3.5
    partial_rr = 1.0
    partial_close_fraction = 0.20
    be_confirm_rr = 2.0
    min_bars_before_be = 6
    pending_expiry_bars = 20

    PREFERRED_PRODUCTION_BAR_SIZE = "1m"
    OPTIONAL_FAST_BAR_SIZE = "30s"

    BASE_REPORT_CONTRACTS = 5
    MAX_REPORT_CONTRACTS = 10

    # Preserve validated fractional sizing path inside backtesting.py.
    BASE_ENGINE_FRACTION = 0.10
    MAX_ENGINE_FRACTION = 0.20

    ENABLE_DYNAMIC_REPORT_SIZING = False
    ENABLE_REAL_DYNAMIC_ENGINE_SIZING = True

    # More selective size-up regime than V463.
    MIN_FAVORABLE_PLANNED_RR_FOR_SIZEUP = 4.5
    MAX_STOP_POINTS_FOR_SIZEUP = 32.0
    STRONG_SETUP_TYPES = {"LONDON_CONTINUATION", "NYAM_CONTINUATION", "NYPM_CONTINUATION"}
    STRONG_BRIDGES = {"IFVG", "iFVG", "MSS", "C2C3"}

    # Buckets intentionally pruned or deprioritized based on V462/V463 analytics.
    BLOCKED_BUCKETS = {
        ("NYPM_CONTINUATION", "C2C3"),
        ("ASIA_CONTINUATION", "CISD"),
        ("LONDON_CONTINUATION", "CISD"),
    }
    DEMOTED_TO_B_BUCKETS = {
        ("LONDON_CONTINUATION", "iFVG"),
        ("LONDON_CONTINUATION", "IFVG"),
    }

    def _default_meta(self) -> Dict[str, Any]:
        base = super()._default_meta()
        base["version"] = self.name
        base["v464_profile"] = "strong_only_pruned"
        base["preferred_production_bar_size"] = self.PREFERRED_PRODUCTION_BAR_SIZE
        base["optional_fast_bar_size"] = self.OPTIONAL_FAST_BAR_SIZE
        base["base_report_contracts"] = self.BASE_REPORT_CONTRACTS
        base["max_report_contracts"] = self.MAX_REPORT_CONTRACTS
        base["base_engine_fraction"] = self.BASE_ENGINE_FRACTION
        base["max_engine_fraction"] = self.MAX_ENGINE_FRACTION
        base["min_favorable_planned_rr_for_sizeup"] = self.MIN_FAVORABLE_PLANNED_RR_FOR_SIZEUP
        base["max_stop_points_for_sizeup"] = self.MAX_STOP_POINTS_FOR_SIZEUP
        return base

    def _setup_tier(self, setup_type: str, bridge_type: str, side: str) -> str:
        setup_upper = str(setup_type or "").upper()
        bridge_raw = str(bridge_type or "")
        bridge_upper = bridge_raw.upper()

        if (setup_upper, bridge_upper) in {(a, b.upper()) for a, b in self.BLOCKED_BUCKETS}:
            return "C"

        if (setup_upper, bridge_raw) in self.DEMOTED_TO_B_BUCKETS or (setup_upper, bridge_upper) in {
            (a, b.upper()) for a, b in self.DEMOTED_TO_B_BUCKETS
        }:
            return "B"

        tier = super()._setup_tier(setup_type, bridge_type, side)

        # NYPM + C2C3 produced too many low-value outcomes; keep it out of ranked prop flow.
        if setup_upper == "NYPM_CONTINUATION" and bridge_upper == "C2C3":
            return "C"

        return tier

    def _pending_dynamic_contracts(self) -> int:
        tier = str(getattr(self.pending, "setup_tier", "") or "").upper()
        setup_type = str(getattr(self.pending, "setup_type", "") or "").upper()
        bridge_type = str(getattr(self.pending, "bridge_type", "") or "")
        bridge_upper = bridge_type.upper()

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

        if tier != "A":
            return self.BASE_REPORT_CONTRACTS
        if setup_type not in self.STRONG_SETUP_TYPES:
            return self.BASE_REPORT_CONTRACTS
        if bridge_upper not in {b.upper() for b in self.STRONG_BRIDGES}:
            return self.BASE_REPORT_CONTRACTS
        if stop_points is None or stop_points > self.MAX_STOP_POINTS_FOR_SIZEUP:
            return self.BASE_REPORT_CONTRACTS
        if planned_rr is None or planned_rr < self.MIN_FAVORABLE_PLANNED_RR_FOR_SIZEUP:
            return self.BASE_REPORT_CONTRACTS

        # Additional bucket-specific tightening.
        if setup_type == "LONDON_CONTINUATION" and bridge_upper in {"IFVG", "IFVG", "IFVG"}:
            # London iFVG stays tradable but only sizes up if the RR is exceptional.
            if planned_rr < 6.0:
                return self.BASE_REPORT_CONTRACTS

        if setup_type == "NYPM_CONTINUATION" and bridge_upper == "C2C3":
            return self.BASE_REPORT_CONTRACTS

        return self.MAX_REPORT_CONTRACTS

    def _effective_size(self) -> float:
        if not self.ENABLE_REAL_DYNAMIC_ENGINE_SIZING:
            return super()._effective_size()

        contracts = self._pending_dynamic_contracts()
        size = self.BASE_ENGINE_FRACTION if contracts <= self.BASE_REPORT_CONTRACTS else self.MAX_ENGINE_FRACTION

        if self.prop_mode and self.prop_reduce_size_after_drawdown and self.realized_pnl_today <= self.prop_drawdown_reduce_threshold:
            size *= self.prop_reduced_size_multiplier

        return float(size)

    def _record_open_trade_meta(self, entry: float):
        super()._record_open_trade_meta(entry)
        if not isinstance(self.open_trade_meta, dict):
            self.open_trade_meta = {}

        planned_contracts = self._pending_dynamic_contracts()
        executed_fraction = self._effective_size()
        executed_contracts = self.BASE_REPORT_CONTRACTS if executed_fraction < self.MAX_ENGINE_FRACTION else self.MAX_REPORT_CONTRACTS

        self.open_trade_meta["version"] = self.name
        self.open_trade_meta["v464_profile"] = "strong_only_pruned"
        self.open_trade_meta["report_contracts"] = int(planned_contracts)
        self.open_trade_meta["executed_size_units"] = float(executed_fraction)
        self.open_trade_meta["executed_size_mode"] = "fractional"
        self.open_trade_meta["executed_contracts_est"] = int(executed_contracts)
        self.open_trade_meta["real_dynamic_engine_sizing"] = True
