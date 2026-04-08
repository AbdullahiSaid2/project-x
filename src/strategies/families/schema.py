from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


DEFAULT_SETUP_PARAMS = {
    "lookback": 20,
    "retest_required": True,
    "volume_confirmation": False,
    "large_bar_confirmation": False,
    "max_retest_bars": 3,
    "retest_tolerance_pct": 0.001,
    "volume_multiplier": 1.05,
    "large_bar_atr_mult": 0.2,
    "min_breakout_range_mult": 1.0,
    "close_confirmation": True,
    "rejection_confirmation": False,
    "move_to_be_at_r": 0.0,
    "trail_atr_after_r": 0.0,
    "failure_exit_on_level_reclaim": False,
}

DEFAULT_RISK_PARAMS = {
    "sl_atr_mult": 0.75,
    "tp_r_multiple": 2.0,
}

DEFAULT_DIRECTION = "both"
DEFAULT_TIMEFRAME = "15m"
DEFAULT_FAMILY = "breakout"
DEFAULT_NAME = "Breakout"


def _deep_merge(base: Dict[str, Any], incoming: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base)
    if not incoming:
        return out
    for k, v in incoming.items():
        out[k] = v
    return out


@dataclass
class StrategySchema:
    name: str = DEFAULT_NAME
    family: str = DEFAULT_FAMILY
    description: str = ""
    normalized_idea: str = ""
    source_idea: str = ""
    direction: str = DEFAULT_DIRECTION
    timeframe_hint: str = DEFAULT_TIMEFRAME
    setup_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SETUP_PARAMS))
    risk_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_RISK_PARAMS))
    metadata: Dict[str, Any] = field(default_factory=dict)
    locked_symbol: Optional[str] = None
    locked_timeframe: Optional[str] = None

    def __post_init__(self):
        self.family = (self.family or DEFAULT_FAMILY).strip()
        self.name = (self.name or DEFAULT_NAME).strip()
        self.description = self.description or self.source_idea or ""
        self.normalized_idea = self.normalized_idea or self.description or self.source_idea or ""
        self.direction = (self.direction or DEFAULT_DIRECTION).strip()
        self.timeframe_hint = (self.timeframe_hint or DEFAULT_TIMEFRAME).strip()

        self.setup_params = _deep_merge(DEFAULT_SETUP_PARAMS, self.setup_params)
        self.risk_params = _deep_merge(DEFAULT_RISK_PARAMS, self.risk_params)

        if self.locked_symbol:
            self.locked_symbol = str(self.locked_symbol).upper()
        if self.locked_timeframe:
            self.locked_timeframe = str(self.locked_timeframe)

        # soft normalization
        if self.direction not in {"long_only", "short_only", "both"}:
            self.direction = DEFAULT_DIRECTION

        if self.timeframe_hint not in {"5m", "15m", "1H", "1h", "4H", "4h", "1D", "1d"}:
            self.timeframe_hint = DEFAULT_TIMEFRAME

        # normalize case
        tf_map = {
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }
        self.timeframe_hint = tf_map.get(self.timeframe_hint, self.timeframe_hint)
        if self.locked_timeframe:
            self.locked_timeframe = tf_map.get(self.locked_timeframe, self.locked_timeframe)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "normalized_idea": self.normalized_idea,
            "source_idea": self.source_idea,
            "direction": self.direction,
            "timeframe_hint": self.timeframe_hint,
            "setup_params": dict(self.setup_params),
            "risk_params": dict(self.risk_params),
            "metadata": dict(self.metadata),
            "locked_symbol": self.locked_symbol,
            "locked_timeframe": self.locked_timeframe,
        }


def schema_from_dict(raw: Dict[str, Any], source_idea: str = "") -> StrategySchema:
    if raw is None:
        raw = {}

    # allow legacy field names safely
    name = raw.get("name") or raw.get("strategy_name") or DEFAULT_NAME
    family = raw.get("family") or DEFAULT_FAMILY
    description = raw.get("description") or raw.get("idea") or source_idea or ""
    normalized_idea = raw.get("normalized_idea") or description
    direction = raw.get("direction") or DEFAULT_DIRECTION
    timeframe_hint = raw.get("timeframe_hint") or raw.get("timeframe") or DEFAULT_TIMEFRAME

    setup_params = _deep_merge(DEFAULT_SETUP_PARAMS, raw.get("setup_params"))
    risk_params = _deep_merge(DEFAULT_RISK_PARAMS, raw.get("risk_params"))

    # common aliases
    if "rr" in raw and "tp_r_multiple" not in risk_params:
        risk_params["tp_r_multiple"] = raw["rr"]

    if "lookback" in raw and "lookback" not in setup_params:
        setup_params["lookback"] = raw["lookback"]

    metadata = raw.get("metadata") or {}
    locked_symbol = raw.get("locked_symbol")
    locked_timeframe = raw.get("locked_timeframe")

    return StrategySchema(
        name=name,
        family=family,
        description=description,
        normalized_idea=normalized_idea,
        source_idea=source_idea or raw.get("source_idea", ""),
        direction=direction,
        timeframe_hint=timeframe_hint,
        setup_params=setup_params,
        risk_params=risk_params,
        metadata=metadata,
        locked_symbol=locked_symbol,
        locked_timeframe=locked_timeframe,
    )