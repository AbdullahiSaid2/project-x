from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict

from src.strategies.manual.ict_multi_setup_v464 import ICT_MULTI_SETUP_V464


class ICT_MULTI_SETUP_V465(ICT_MULTI_SETUP_V464):
    """
    V465
    - Real follow-up to V464.
    - Keeps the actual V464 pruning, 5/10 MNQ regime, and no-CISD ranked execution.
    - Softens runner management slightly so quality trades have more room to expand.
    - Tightens weak Asia iFVG sizing while preserving the cleanest continuation buckets.

    Design intent
    - Keep 5 MNQ on standard setups.
    - Keep 10 MNQ only on strongest setups in the more liquid sessions.
    - Avoid the over-tight winner choking that likely reduced total dollar capture in V464.
    - Leave the good V464 bucket pruning intact instead of reverting to the noisier V462 mix.
    """

    name = "ICT_MULTI_SETUP_V465"

    # Slightly less restrictive than V464 so good runners breathe more.
    risk_multiple = 3.25
    partial_rr = 0.90
    partial_close_fraction = 0.20
    be_confirm_rr = 1.90
    min_bars_before_be = 5
    pending_expiry_bars = 22

    # Keep real 5/10 MNQ only.
    MIN_FAVORABLE_PLANNED_RR_FOR_SIZEUP = 4.5
    MAX_STOP_POINTS_FOR_SIZEUP = 35.0
    STRONG_SETUP_TYPES = {"LONDON_CONTINUATION", "NYPM_CONTINUATION", "NYAM_CONTINUATION"}
    STRONG_BRIDGES = {"IFVG", "iFVG", "MSS", "C2C3"}

    # Maintain V464 pruning and add a little more discipline around weaker Asia iFVGs.
    BLOCKED_BUCKETS = {
        ("NYPM_CONTINUATION", "C2C3"),
        ("ASIA_CONTINUATION", "CISD"),
        ("LONDON_CONTINUATION", "CISD"),
    }
    DEMOTED_TO_B_BUCKETS = {
        ("LONDON_CONTINUATION", "iFVG"),
        ("LONDON_CONTINUATION", "IFVG"),
        ("NYAM_CONTINUATION", "iFVG"),
        ("NYAM_CONTINUATION", "IFVG"),
    }

    ASIA_IFVG_MAX_STOP_FOR_NORMAL_USE = 30.0
    ASIA_IFVG_MIN_RR_FOR_NORMAL_USE = 5.5
    LONDON_IFVG_MIN_RR_FOR_SIZEUP = 6.0

    def _default_meta(self) -> Dict[str, Any]:
        base = super()._default_meta()
        base["version"] = self.name
        base["v465_profile"] = "pruned_with_breathing_room"
        base["asia_ifvg_max_stop_for_normal_use"] = self.ASIA_IFVG_MAX_STOP_FOR_NORMAL_USE
        base["asia_ifvg_min_rr_for_normal_use"] = self.ASIA_IFVG_MIN_RR_FOR_NORMAL_USE
        base["london_ifvg_min_rr_for_sizeup"] = self.LONDON_IFVG_MIN_RR_FOR_SIZEUP
        return base

    def _setup_tier(self, setup_type: str, bridge_type: str, side: str) -> str:
        tier = super()._setup_tier(setup_type, bridge_type, side)
        setup_upper = str(setup_type or "").upper()
        bridge_upper = str(bridge_type or "").upper()

        # Keep NYAM tradable but not strong enough for aggressive size-up.
        if setup_upper == "NYAM_CONTINUATION" and bridge_upper == "IFVG":
            return "B"

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

        # Asia iFVG stays 5 MNQ only and only when the structure is reasonably compact/clean.
        if setup_type == "ASIA_CONTINUATION" and bridge_upper == "IFVG":
            return self.BASE_REPORT_CONTRACTS

        # Only the strongest, liquid-session A-tier setups get 10 MNQ.
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

        # London iFVG is allowed to trade, but size-up only when the RR is exceptional.
        if setup_type == "LONDON_CONTINUATION" and bridge_upper == "IFVG" and planned_rr < self.LONDON_IFVG_MIN_RR_FOR_SIZEUP:
            return self.BASE_REPORT_CONTRACTS

        # NYAM remains 5 MNQ only in V465.
        if setup_type == "NYAM_CONTINUATION":
            return self.BASE_REPORT_CONTRACTS

        return self.MAX_REPORT_CONTRACTS

    def _record_open_trade_meta(self, entry: float):
        super()._record_open_trade_meta(entry)
        if not isinstance(self.open_trade_meta, dict):
            self.open_trade_meta = {}
        self.open_trade_meta["version"] = self.name
        self.open_trade_meta["v465_profile"] = "pruned_with_breathing_room"
