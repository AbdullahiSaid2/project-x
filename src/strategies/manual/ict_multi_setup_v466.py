from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict

from src.strategies.manual.ict_multi_setup_v464 import ICT_MULTI_SETUP_V464


class ICT_MULTI_SETUP_V466(ICT_MULTI_SETUP_V464):
    """
    V466
    - Built directly from V464, which is currently the preferred baseline.
    - Preserves the V464 setup pruning and bucket quality controls.
    - Forces 10 MNQ on every executed trade.
    - Keeps Apex 50k reporting assumptions.

    Purpose
    - Establish a clean fixed-size 10-MNQ benchmark before making further edge changes.
    - This is a sizing build, not a claim that the strategy now reaches 100k realized PnL.
    """

    name = "ICT_MULTI_SETUP_V466"

    # Keep V464 trade-selection logic, but normalize sizing to a constant 10 MNQ.
    BASE_REPORT_CONTRACTS = 10
    MAX_REPORT_CONTRACTS = 10

    # Preserve the validated fractional sizing path in backtesting.py,
    # but force the same larger size on every trade.
    BASE_ENGINE_FRACTION = 0.20
    MAX_ENGINE_FRACTION = 0.20

    ENABLE_DYNAMIC_REPORT_SIZING = False
    ENABLE_REAL_DYNAMIC_ENGINE_SIZING = True

    def _default_meta(self) -> Dict[str, Any]:
        base = super()._default_meta()
        base["version"] = self.name
        base["v466_profile"] = "v464_fixed_10_mnq_all_trades"
        base["base_report_contracts"] = self.BASE_REPORT_CONTRACTS
        base["max_report_contracts"] = self.MAX_REPORT_CONTRACTS
        base["base_engine_fraction"] = self.BASE_ENGINE_FRACTION
        base["max_engine_fraction"] = self.MAX_ENGINE_FRACTION
        return base

    def _pending_dynamic_contracts(self) -> int:
        return self.MAX_REPORT_CONTRACTS

    def _effective_size(self) -> float:
        size = float(self.MAX_ENGINE_FRACTION)

        if self.prop_mode and self.prop_reduce_size_after_drawdown and self.realized_pnl_today <= self.prop_drawdown_reduce_threshold:
            size *= self.prop_reduced_size_multiplier

        return float(size)

    def _record_open_trade_meta(self, entry: float):
        super()._record_open_trade_meta(entry)
        if not isinstance(self.open_trade_meta, dict):
            self.open_trade_meta = {}

        self.open_trade_meta["version"] = self.name
        self.open_trade_meta["v466_profile"] = "v464_fixed_10_mnq_all_trades"
        self.open_trade_meta["report_contracts"] = int(self.MAX_REPORT_CONTRACTS)
        self.open_trade_meta["executed_size_units"] = float(self._effective_size())
        self.open_trade_meta["executed_size_mode"] = "fractional"
        self.open_trade_meta["executed_contracts_est"] = int(self.MAX_REPORT_CONTRACTS)
        self.open_trade_meta["real_dynamic_engine_sizing"] = True
