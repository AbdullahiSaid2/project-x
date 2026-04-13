from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict

from src.strategies.manual.ict_multi_setup_v460 import ICT_MULTI_SETUP_V460


class ICT_MULTI_SETUP_V462(ICT_MULTI_SETUP_V460):
    """
    V462
    - Keeps the healthy V460 signal engine intact.
    - Makes the former 5-to-10 MNQ dynamic sizing real inside the strategy.
    - Uses *fractional* backtesting.py sizing so trade frequency stays aligned with the
      validated V460 behaviour instead of collapsing.
    - Keeps metadata/reporting fields needed for clean deduped exports.

    Why fractional sizing:
    The upstream strategy uses backtesting.py fractional size semantics (`fixed_size=0.10`).
    Replacing that with absolute unit counts choked opportunity flow in the prior V462 build.
    This version preserves the original order semantics while still making the 5-to-10 MNQ
    sizing path real in-engine.

    Approximate mapping at a $1,000,000 test account:
    - 5 MNQ mode  -> 0.10 fraction of equity (validated V460 baseline path)
    - 10 MNQ mode -> 0.20 fraction of equity for stronger setups
    """

    name = "ICT_MULTI_SETUP_V462"

    PREFERRED_PRODUCTION_BAR_SIZE = "1m"
    OPTIONAL_FAST_BAR_SIZE = "30s"

    BASE_REPORT_CONTRACTS = 5
    MAX_REPORT_CONTRACTS = 10

    # Keep backtesting.py fractional sizing semantics.
    BASE_ENGINE_FRACTION = 0.10
    MAX_ENGINE_FRACTION = 0.20

    ENABLE_DYNAMIC_REPORT_SIZING = False
    ENABLE_REAL_DYNAMIC_ENGINE_SIZING = True

    MIN_FAVORABLE_PLANNED_RR_FOR_SIZEUP = 4.0
    MAX_STOP_POINTS_FOR_SIZEUP = 35.0
    PREFER_A_TIER_SIZEUP = True
    STRONG_SETUP_TYPES = {"LONDON_CONTINUATION", "NYAM_CONTINUATION", "NYPM_CONTINUATION"}
    STRONG_BRIDGES = {"IFVG", "C2C3", "MSS", "CISD", "iFVG"}

    def _default_meta(self) -> Dict[str, Any]:
        base = super()._default_meta()
        base.setdefault("entry_variant", "PULLBACK_1M")
        base["preferred_production_bar_size"] = self.PREFERRED_PRODUCTION_BAR_SIZE
        base["optional_fast_bar_size"] = self.OPTIONAL_FAST_BAR_SIZE
        base["base_report_contracts"] = self.BASE_REPORT_CONTRACTS
        base["max_report_contracts"] = self.MAX_REPORT_CONTRACTS
        base["base_engine_fraction"] = self.BASE_ENGINE_FRACTION
        base["max_engine_fraction"] = self.MAX_ENGINE_FRACTION
        base["min_favorable_planned_rr_for_sizeup"] = self.MIN_FAVORABLE_PLANNED_RR_FOR_SIZEUP
        base["max_stop_points_for_sizeup"] = self.MAX_STOP_POINTS_FOR_SIZEUP
        return base

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

        if (
            tier == "A"
            and setup_type in self.STRONG_SETUP_TYPES
            and bridge_upper in {b.upper() for b in self.STRONG_BRIDGES}
            and stop_points is not None
            and stop_points <= self.MAX_STOP_POINTS_FOR_SIZEUP
            and planned_rr is not None
            and planned_rr >= self.MIN_FAVORABLE_PLANNED_RR_FOR_SIZEUP
        ):
            return self.MAX_REPORT_CONTRACTS
        return self.BASE_REPORT_CONTRACTS

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
        # Approximate the live contract intent from the chosen fraction.
        executed_contracts = self.BASE_REPORT_CONTRACTS if executed_fraction < self.MAX_ENGINE_FRACTION else self.MAX_REPORT_CONTRACTS

        self.open_trade_meta["report_contracts"] = int(planned_contracts)
        self.open_trade_meta["executed_size_units"] = float(executed_fraction)
        self.open_trade_meta["executed_size_mode"] = "fractional"
        self.open_trade_meta["executed_contracts_est"] = int(executed_contracts)
        self.open_trade_meta["real_dynamic_engine_sizing"] = True
