from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd

from src.strategies.manual.ict_multi_setup_v458 import ICT_MULTI_SETUP_V458


class ICT_MULTI_SETUP_V460(ICT_MULTI_SETUP_V458):
    """
    V460
    - Keeps the V458 export/metadata/session fixes.
    - Pushes for better yearly PnL by changing trade management, not fake counting.
    - Makes runners matter more and reduces the number of BE / tiny-win outcomes.
    - Opens up selected CISD continuation flows that were previously tier-C blocked.

    Important:
    This is still a real strategy variant, not a synthetic scaling report. The goal is
    better points per trade and slightly higher opportunity count while keeping Apex-safe
    session handling intact.
    """

    name = "ICT_MULTI_SETUP_V460"

    # More ambitious trade management than V458.
    risk_multiple = 3.0
    partial_rr = 0.75
    partial_close_fraction = 0.25
    be_confirm_rr = 1.75
    min_bars_before_be = 5
    pending_expiry_bars = 24

    # Keep time protections, but allow a little more trade development before cutoff.
    GLOBAL_ENTRY_CUTOFF_ET = pd.Timestamp("2000-01-01 16:20:00").time()
    NYPM_ENTRY_CUTOFF_ET = pd.Timestamp("2000-01-01 15:50:00").time()

    # Open up more continuation bridges.
    allow_cisd_asia = True
    allow_cisd_london = True
    allow_cisd_nyam = True
    allow_cisd_nypm = True

    TRADE_METADATA_LOG: List[Dict[str, Any]] = []
    DEBUG_COUNTERS: Dict[str, int] = defaultdict(int)

    def init(self) -> None:
        super().init()
        type(self).TRADE_METADATA_LOG = []
        type(self).DEBUG_COUNTERS = defaultdict(int)

    def _setup_tier(self, setup_type: str, bridge_type: str, side: str) -> str:
        """
        Promote selected CISD continuation cases into ranked A/B buckets so they can trade.
        Keep the more conservative behavior for everything else.
        """
        setup_type = str(setup_type or "")
        bridge_type = str(bridge_type or "")
        side = str(side or "")

        if bridge_type == "CISD":
            if setup_type == "LONDON_CONTINUATION":
                return "A"
            if setup_type == "ASIA_CONTINUATION":
                return "B"
            if setup_type == "NYAM_CONTINUATION":
                return "B"
            if setup_type == "NYPM_CONTINUATION" and side == "LONG":
                return "B"
            return "C"

        return super()._setup_tier(setup_type, bridge_type, side)

    def _default_meta(self) -> Dict[str, Any]:
        base = super()._default_meta()
        # Make sure exported metadata reflects the active V460 management defaults whenever
        # the upstream strategy does not fill these explicitly.
        if base.get("entry_variant") in (None, ""):
            base["entry_variant"] = "PULLBACK_1M"
        return base
