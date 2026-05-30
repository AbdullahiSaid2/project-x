from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# =========================================================
# ICT FRACTAL V2 QUALITY — BLOCK MYM LONG VERSION
# =========================================================
#
# Strategy name remains:
#
#   --strategy ict_fractal_v2_quality
#
# Why:
#   Latest BE + partial-only run showed:
#
#       MYM LONG: -$1,243.12
#
#   while these areas were strong:
#
#       MNQ LONG:  +$4,483.45
#       MYM SHORT: +$2,543.53
#       MNQ SHORT: +$1,535.14
#       MES SHORT: +$227.75
#
# This adapter keeps the current profitable quality flow but adds:
#
#   BLOCK MYM LONG
#
# It also continues blocking the weak setup family already found:
#
#   ASIA_CONTINUATION_iFVG_B_LONG
#
# =========================================================


STRATEGY_NAME = "ict_fractal_v2_quality"

DIAG_DIR = Path("src/backtesting/event_engine/outputs")
DIAG_DIR.mkdir(parents=True, exist_ok=True)
DIAG_PATH = DIAG_DIR / "ict_fractal_v2_quality_filter_diagnostics.csv"


# Known weak setup family from previous diagnostics.
BLOCKED_SETUP_KEYWORDS = (
    "ASIA_CONTINUATION_iFVG_B_LONG",
)

# New symbol/side filter from BE + partial-only result.
BLOCKED_SYMBOL_SIDE = {
    ("MYM", "LONG"),
}


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


def _extract_side(order: Any) -> str:
    value = str(_get_attr_or_key(order, "side", "") or "").upper()

    if value in {"BUY", "BULL", "BULLISH", "LONG"}:
        return "LONG"

    if value in {"SELL", "BEAR", "BEARISH", "SHORT"}:
        return "SHORT"

    return value


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


class ICTFractalV2QualityAdapter:
    """
    Safe quality wrapper around ICTFractalV2Adapter.

    This version preserves the profitable quality behaviour and adds:
      - block known weak setup family
      - block MYM LONG trades
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
            "blocked_setup_keyword": 0,
            "blocked_symbol_side": 0,
            "long_seen": 0,
            "short_seen": 0,
            "mym_long_blocked": 0,
            "mnq_accepted": 0,
            "mym_accepted": 0,
            "mes_accepted": 0,
            "sample_blocked": "",
            "sample_accepted": "",
        }

        self.sample_blocked = []
        self.sample_accepted = []

    def _sample(self, bucket: str, text: str):
        if bucket == "blocked":
            if len(self.sample_blocked) < 25 and text not in self.sample_blocked:
                self.sample_blocked.append(text[:500])
        else:
            if len(self.sample_accepted) < 25 and text not in self.sample_accepted:
                self.sample_accepted.append(text[:500])

    def write_diag(self):
        self.diag["sample_blocked"] = " || ".join(self.sample_blocked[:25])
        self.diag["sample_accepted"] = " || ".join(self.sample_accepted[:25])
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

        symbol = str(symbol).upper()
        side = _extract_side(order)
        trade_type = _extract_trade_type(order)

        if side == "LONG":
            self.diag["long_seen"] += 1
        elif side == "SHORT":
            self.diag["short_seen"] += 1

        # 1. Block weak setup family.
        for keyword in BLOCKED_SETUP_KEYWORDS:
            if keyword in trade_type:
                self.diag["blocked_setup_keyword"] += 1
                self._sample("blocked", f"setup_keyword={keyword}|symbol={symbol}|side={side}|{trade_type}")
                self.write_diag()
                return None

        # 2. Block MYM LONG from latest analysis.
        if (symbol, side) in BLOCKED_SYMBOL_SIDE:
            self.diag["blocked_symbol_side"] += 1
            if symbol == "MYM" and side == "LONG":
                self.diag["mym_long_blocked"] += 1
            self._sample("blocked", f"symbol_side={symbol}_{side}|{trade_type}")
            self.write_diag()
            return None

        _set_attr_if_possible(order, "strategy_name", self.name)

        reason = _get_attr_or_key(order, "reason", "")
        suffix = "v2_quality_block_mym_long"

        if reason:
            _set_attr_if_possible(order, "reason", f"{reason}|{suffix}")
        else:
            _set_attr_if_possible(order, "reason", suffix)

        self.diag["accepted"] += 1

        if symbol == "MNQ":
            self.diag["mnq_accepted"] += 1
        elif symbol == "MYM":
            self.diag["mym_accepted"] += 1
        elif symbol == "MES":
            self.diag["mes_accepted"] += 1

        self._sample("accepted", f"symbol={symbol}|side={side}|{trade_type}")
        self.write_diag()

        return order
