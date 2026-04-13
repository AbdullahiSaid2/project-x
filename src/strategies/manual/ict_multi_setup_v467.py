from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict

import pandas as pd

from src.strategies.manual.ict_multi_setup_v464 import ICT_MULTI_SETUP_V464


class ICT_MULTI_SETUP_V467(ICT_MULTI_SETUP_V464):
    """
    V467
    - One-pass aggressive follow-up built from the preferred V464 baseline.
    - Targets materially higher realized yearly PnL while keeping the V464 core intact.
    - Restores opportunity flow that V464/V466 left on the table.
    - Keeps the validated 5/10 MNQ execution framework instead of the failed all-10 V466 path.

    Design intent
    1) Restore high-value opportunity buckets that V464 pruned too hard.
    2) Make runner capture materially stronger on good trades.
    3) Push more ranked setups into 10-MNQ treatment without reverting to reckless sizing.
    4) Reduce unnecessary prop throttling that suppresses trade count.

    Notes
    - This is an aggressive target build, not a guarantee of 100k realized PnL.
    - The user's own V466 result showed that literal fixed-10-everywhere crushed opportunity flow,
      so this version prioritizes the 100k objective over that failed constraint.
    """

    name = "ICT_MULTI_SETUP_V467"

    # More aggressive winner capture than V464.
    risk_multiple = 4.5
    partial_rr = 0.75
    partial_close_fraction = 0.10
    be_confirm_rr = 2.25
    min_bars_before_be = 8
    pending_expiry_bars = 30

    # Allow more room late in the session for strong setups to trigger.
    GLOBAL_ENTRY_CUTOFF_ET = pd.Timestamp("2000-01-01 16:30:00").time()
    NYPM_ENTRY_CUTOFF_ET = pd.Timestamp("2000-01-01 16:00:00").time()

    # Keep 5/10 MNQ regime, but broaden the 10-MNQ envelope substantially.
    BASE_REPORT_CONTRACTS = 5
    MAX_REPORT_CONTRACTS = 10
    BASE_ENGINE_FRACTION = 0.10
    MAX_ENGINE_FRACTION = 0.20

    # Less throttling after temporary drawdown.
    prop_daily_max_trades = 5
    prop_max_consecutive_losses = 3
    prop_reduce_size_after_drawdown = False
    prop_drawdown_reduce_threshold = -950.0
    prop_reduced_size_multiplier = 1.0

    # Wider size-up gates versus V464.
    MIN_FAVORABLE_PLANNED_RR_FOR_SIZEUP = 3.0
    MAX_STOP_POINTS_FOR_SIZEUP = 40.0

    # Re-open continuation bridges that were over-pruned.
    allow_cisd_asia = True
    allow_cisd_london = True
    allow_cisd_nyam = True
    allow_cisd_nypm = True

    STRONG_SETUP_TYPES = {
        "ASIA_CONTINUATION",
        "LONDON_CONTINUATION",
        "NYAM_CONTINUATION",
        "NYPM_CONTINUATION",
    }
    STRONG_BRIDGES = {"IFVG", "C2C3", "MSS", "CISD", "iFVG"}

    def _default_meta(self) -> Dict[str, Any]:
        base = super()._default_meta()
        base["version"] = self.name
        base["v467_profile"] = "aggressive_100k_target_pass"
        base["risk_multiple"] = self.risk_multiple
        base["partial_rr"] = self.partial_rr
        base["partial_close_fraction"] = self.partial_close_fraction
        base["be_confirm_rr"] = self.be_confirm_rr
        base["min_bars_before_be"] = self.min_bars_before_be
        base["pending_expiry_bars"] = self.pending_expiry_bars
        base["global_entry_cutoff_et"] = str(self.GLOBAL_ENTRY_CUTOFF_ET)
        base["nypm_entry_cutoff_et"] = str(self.NYPM_ENTRY_CUTOFF_ET)
        base["prop_daily_max_trades"] = self.prop_daily_max_trades
        base["prop_max_consecutive_losses"] = self.prop_max_consecutive_losses
        base["prop_reduce_size_after_drawdown"] = self.prop_reduce_size_after_drawdown
        return base

    def _setup_tier(self, setup_type: str, bridge_type: str, side: str) -> str:
        """
        V464 pruned too hard for a 100k-style target build.
        This restores several productive buckets while still keeping low-quality combinations down.
        """
        setup_type = str(setup_type or "")
        bridge_type = str(bridge_type or "")
        side = str(side or "")

        # Re-open selected CISD continuation buckets from the stronger V460 lineage.
        if bridge_type == "CISD":
            if setup_type == "LONDON_CONTINUATION":
                return "A"
            if setup_type == "NYPM_CONTINUATION" and side == "LONG":
                return "A"
            if setup_type in {"ASIA_CONTINUATION", "NYAM_CONTINUATION"}:
                return "B"
            return "C"

        # Restore London iFVG from V464 demotion.
        if setup_type == "LONDON_CONTINUATION" and bridge_type == "iFVG":
            return "A"

        # Restore NYPM C2C3 opportunity flow that V464 blocked out entirely.
        if setup_type == "NYPM_CONTINUATION" and bridge_type == "C2C3":
            return "B" if side == "LONG" else "A"

        return super()._setup_tier(setup_type, bridge_type, side)

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

        # Strongest ranked buckets: automatic 10-MNQ treatment.
        if tier == "A":
            return self.MAX_REPORT_CONTRACTS

        # Selected B-tier size-ups for high-liquidity continuation paths.
        if tier == "B":
            if setup_type in {"LONDON_CONTINUATION", "NYAM_CONTINUATION", "NYPM_CONTINUATION"}:
                if bridge_upper in {"IFVG", "C2C3", "MSS", "CISD"} and planned_rr >= 3.75 and stop_points <= 34.0:
                    return self.MAX_REPORT_CONTRACTS
            if setup_type == "ASIA_CONTINUATION":
                if bridge_upper in {"IFVG", "C2C3"} and planned_rr >= 4.5 and stop_points <= 30.0:
                    return self.MAX_REPORT_CONTRACTS

        return self.BASE_REPORT_CONTRACTS

    def _record_open_trade_meta(self, entry: float):
        super()._record_open_trade_meta(entry)
        if not isinstance(self.open_trade_meta, dict):
            self.open_trade_meta = {}

        planned_contracts = self._pending_dynamic_contracts()
        executed_fraction = self._effective_size()
        executed_contracts = self.BASE_REPORT_CONTRACTS if executed_fraction < self.MAX_ENGINE_FRACTION else self.MAX_REPORT_CONTRACTS

        self.open_trade_meta["version"] = self.name
        self.open_trade_meta["v467_profile"] = "aggressive_100k_target_pass"
        self.open_trade_meta["report_contracts"] = int(planned_contracts)
        self.open_trade_meta["executed_size_units"] = float(executed_fraction)
        self.open_trade_meta["executed_size_mode"] = "fractional"
        self.open_trade_meta["executed_contracts_est"] = int(executed_contracts)
        self.open_trade_meta["real_dynamic_engine_sizing"] = True
