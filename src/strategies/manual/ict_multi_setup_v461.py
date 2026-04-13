from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict

from src.strategies.manual.ict_multi_setup_v460 import ICT_MULTI_SETUP_V460


class ICT_MULTI_SETUP_V461(ICT_MULTI_SETUP_V460):
    """
    V461
    - Repairs duplicate metadata row handling at the reporting layer.
    - Adds optional 30-second data-loader support in the companion runner.
    - Adds dynamic 5-to-10 MNQ reporting columns for tighter, stronger setups.

    Important:
    The real strategy logic still runs on whatever bar data you load.
    To truly trade/test 30-second entries, provide an NQ_30s.parquet cache.
    """

    name = "ICT_MULTI_SETUP_V461"

    ENTRY_TRIGGER_SECONDS = 30
    BASE_REPORT_CONTRACTS = 5
    MAX_REPORT_CONTRACTS = 10
    ENABLE_30S_TRIGGER_MODE = True
    ENABLE_DYNAMIC_REPORT_SIZING = True

    def _default_meta(self) -> Dict[str, Any]:
        base = super()._default_meta()
        base.setdefault("entry_variant", "PULLBACK_1M")
        base["trigger_timeframe_seconds"] = self.ENTRY_TRIGGER_SECONDS
        base["base_report_contracts"] = self.BASE_REPORT_CONTRACTS
        base["max_report_contracts"] = self.MAX_REPORT_CONTRACTS
        return base
