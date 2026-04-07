from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict


@dataclass
class StrategySchema:
    family: str
    name: str
    description: str
    direction: str = "long_only"
    timeframe_hint: str = "15m"
    trade_frequency_target: str = "medium"
    indicator_params: Dict[str, Any] = field(default_factory=dict)
    setup_params: Dict[str, Any] = field(default_factory=dict)
    risk_params: Dict[str, Any] = field(default_factory=dict)
    source_idea: str = ""
    normalized_idea: str = ""
    family_confidence: float = 0.7
    codifiability_score: float = 7.0
    ambiguity_score: float = 3.0

    def to_dict(self):
        return asdict(self)


def _to_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def schema_from_dict(raw, source_idea=""):
    family = str(raw.get("family", "mean_reversion")).strip()

    indicator_params = {
        "rsi_window": _to_int(raw.get("indicator_params", {}).get("rsi_window", 14), 14),
        "atr_window": _to_int(raw.get("indicator_params", {}).get("atr_window", 14), 14),
        "ema_fast": _to_int(raw.get("indicator_params", {}).get("ema_fast", 20), 20),
        "ema_slow": _to_int(raw.get("indicator_params", {}).get("ema_slow", 50), 50),
        "bb_window": _to_int(raw.get("indicator_params", {}).get("bb_window", 20), 20),
        "bb_std": _to_float(raw.get("indicator_params", {}).get("bb_std", 2.0), 2.0),
    }

    setup_defaults = {
        "lookback": 20,
        "retest_required": False,
        "volume_confirmation": False,
        "large_bar_confirmation": False,
        "rejection_confirmation": False,
        "close_confirmation": False,
        "max_retest_bars": 3,
        "price_tolerance_pct": 0.002,
        "break_buffer_pct": 0.0002,
        "retest_tolerance_pct": 0.001,
        "volume_multiplier": 1.0,
        "large_bar_atr_mult": 0.0,
        "use_session_filter": True,
        "use_volatility_filter": True,
        "use_trend_filter": True,
        "strict_mode": True,
    }
    setup_defaults.update(raw.get("setup_params", {}) or {})

    risk_defaults = {
        "sl_atr_mult": 1.0,
        "tp_r_multiple": 1.5,
        "fixed_size": 0.1,
    }
    risk_defaults.update(raw.get("risk_params", {}) or {})

    setup_params = {
        "lookback": _to_int(setup_defaults.get("lookback", 20), 20),
        "retest_required": bool(setup_defaults.get("retest_required", False)),
        "volume_confirmation": bool(setup_defaults.get("volume_confirmation", False)),
        "large_bar_confirmation": bool(setup_defaults.get("large_bar_confirmation", False)),
        "rejection_confirmation": bool(setup_defaults.get("rejection_confirmation", False)),
        "close_confirmation": bool(setup_defaults.get("close_confirmation", False)),
        "max_retest_bars": _to_int(setup_defaults.get("max_retest_bars", 3), 3),
        "price_tolerance_pct": _to_float(setup_defaults.get("price_tolerance_pct", 0.002), 0.002),
        "break_buffer_pct": _to_float(setup_defaults.get("break_buffer_pct", 0.0002), 0.0002),
        "retest_tolerance_pct": _to_float(setup_defaults.get("retest_tolerance_pct", 0.001), 0.001),
        "volume_multiplier": _to_float(setup_defaults.get("volume_multiplier", 1.0), 1.0),
        "large_bar_atr_mult": _to_float(setup_defaults.get("large_bar_atr_mult", 0.0), 0.0),
        "use_session_filter": bool(setup_defaults.get("use_session_filter", True)),
        "use_volatility_filter": bool(setup_defaults.get("use_volatility_filter", True)),
        "use_trend_filter": bool(setup_defaults.get("use_trend_filter", True)),
        "strict_mode": bool(setup_defaults.get("strict_mode", True)),
    }

    risk_params = {
        "sl_atr_mult": _to_float(risk_defaults.get("sl_atr_mult", 1.0), 1.0),
        "tp_r_multiple": _to_float(risk_defaults.get("tp_r_multiple", 1.5), 1.5),
        "fixed_size": min(max(_to_float(risk_defaults.get("fixed_size", 0.1), 0.1), 0.01), 0.2),
    }

    return StrategySchema(
        family=family,
        name=raw.get("name", family.replace("_", " ").title()),
        description=raw.get("description", source_idea),
        direction=raw.get("direction", "long_only"),
        timeframe_hint=raw.get("timeframe_hint", "15m"),
        trade_frequency_target=raw.get("trade_frequency_target", "medium"),
        indicator_params=indicator_params,
        setup_params=setup_params,
        risk_params=risk_params,
        source_idea=source_idea,
        normalized_idea=raw.get("normalized_idea", source_idea),
        family_confidence=float(raw.get("family_confidence", 0.7)),
        codifiability_score=float(raw.get("codifiability_score", 7.0)),
        ambiguity_score=float(raw.get("ambiguity_score", 3.0)),
    )