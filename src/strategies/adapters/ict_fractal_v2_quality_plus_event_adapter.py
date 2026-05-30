from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


STRATEGY_NAME = "ict_fractal_v2_quality_plus"

DIAG_DIR = Path("src/backtesting/event_engine/outputs")
DIAG_DIR.mkdir(parents=True, exist_ok=True)
DIAG_PATH = DIAG_DIR / "ict_fractal_v2_quality_plus_filter_diagnostics.csv"


BLOCKED_KEYWORDS = (
    "ASIA_CONTINUATION_iFVG_B_LONG",
    "LONDON_CONTINUATION_MSS_A_LONG",
    "ASIA_CONTINUATION_C2C3_B_LONG",
)


def _get_attr_or_key(obj: Any, name: str, default=None):
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def _set_attr_if_possible(obj: Any, name: str, value: Any):
    if isinstance(obj, dict):
        obj[name] = value
        return obj

    try:
        setattr(obj, name, value)
    except Exception:
        pass

    return obj


def _extract_trade_type(order: Any) -> str:
    parts = []

    for field in (
        "trade_type",
        "reason",
        "setup",
        "signal_name",
        "model_signal",
        "strategy_name",
    ):
        value = _get_attr_or_key(order, field, None)
        if value:
            parts.append(str(value))

    return "|".join(parts)


class ICTFractalV2QualityPlusAdapter:
    """
    Light refinement layer over ICTFractalV2Adapter.

    It keeps the profitable quality-style flow but blocks only the specific
    weak setup families found in analysis.
    """

    name = STRATEGY_NAME
    strategy_name = STRATEGY_NAME

    def __init__(self):
        from src.strategies.adapters.ict_fractal_v2_event_adapter import (
            ICTFractalV2Adapter,
        )

        self.base = ICTFractalV2Adapter()
        self.diag = {
            "base_orders_seen": 0,
            "accepted": 0,
            "blocked_family": 0,
        }

    def write_diag(self):
        pd.DataFrame([self.diag]).to_csv(DIAG_PATH, index=False)

    def build_features(self, symbol, df):
        if hasattr(self.base, "build_features"):
            return self.base.build_features(symbol, df)
        return df.copy()

    def build_features_with_args(self, symbol, df, args=None):
        if hasattr(self.base, "build_features_with_args"):
            return self.base.build_features_with_args(symbol, df, args)
        return self.build_features(symbol, df)

    def signal_for_row(self, symbol, row, history, spec, profile, args):
        try:
            order = self.base.signal_for_row(
                symbol=symbol,
                row=row,
                history=history,
                spec=spec,
                profile=profile,
                args=args,
            )
        except TypeError:
            order = self.base.signal_for_row(symbol, row, history, spec, profile, args)

        if order is None:
            return None

        self.diag["base_orders_seen"] += 1
        trade_type = _extract_trade_type(order)

        for keyword in BLOCKED_KEYWORDS:
            if keyword in trade_type:
                self.diag["blocked_family"] += 1
                self.write_diag()
                return None

        _set_attr_if_possible(order, "strategy_name", self.name)

        reason = _get_attr_or_key(order, "reason", "")
        if reason:
            _set_attr_if_possible(order, "reason", f"{reason}|quality_plus")
        else:
            _set_attr_if_possible(order, "reason", "quality_plus")

        self.diag["accepted"] += 1
        self.write_diag()

        return order
