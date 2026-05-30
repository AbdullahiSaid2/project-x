from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# =========================================================
# ICT FRACTAL V2 EVAL ADAPTER
# =========================================================
#
# Strategy name:
#
#   --strategy ict_fractal_v2_eval
#
# Purpose:
#
#   Pass evaluations faster.
#
# Important:
#
#   This is NOT the PA payout strategy.
#
#   PA/payout strategy should remain:
#
#       --strategy ict_fractal_v2_quality
#
# Why separate eval and PA?
#
#   Eval goal:
#       pass quickly
#
#   PA goal:
#       survive + payout consistently
#
# Eval mode is more permissive:
#
#   - allows more setup families
#   - does NOT block MYM LONG
#   - does NOT block ASIA_CONTINUATION_iFVG_B_LONG
#   - only blocks clearly bad / unusable signals
#   - writes diagnostics
#
# Then the lifecycle simulator can use:
#
#   Eval trade log from ict_fractal_v2_eval
#   PA trade log from ict_fractal_v2_quality
#
# =========================================================


STRATEGY_NAME = "ict_fractal_v2_eval"

DIAG_DIR = Path("src/backtesting/event_engine/outputs")
DIAG_DIR.mkdir(parents=True, exist_ok=True)
DIAG_PATH = DIAG_DIR / "ict_fractal_v2_eval_filter_diagnostics.csv"


# Eval should be broader than PA.
# Keep this list intentionally small.
BLOCKED_SETUP_KEYWORDS = (
    # Keep truly dangerous/excessively weak future exclusions here.
    # For now, do not block the families we blocked in PA,
    # because eval needs more frequency.
)

# Optional minimum score. The base ICTFractalV2Adapter already scores setups.
# We keep this permissive.
MIN_EVAL_SCORE = 4.0

# Keep MGC disabled by default because previous accepted edge was poor/none.
BLOCKED_SYMBOLS = {
    "MGC",
}

# Allow core equity futures.
ALLOWED_SYMBOLS = {
    "MNQ",
    "MYM",
    "MES",
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


def _extract_score(order: Any) -> float:
    for field in ("setup_score", "trend_score", "score", "signal_score"):
        value = _get_attr_or_key(order, field, None)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            pass

    text = _extract_trade_type(order)
    marker = "v2_score="
    if marker in text:
        try:
            raw = text.split(marker, 1)[1].split("|", 1)[0]
            return float(raw)
        except Exception:
            return MIN_EVAL_SCORE

    return MIN_EVAL_SCORE


class ICTFractalV2EvalAdapter:
    """
    Fast evaluation wrapper over ICTFractalV2Adapter.

    This adapter accepts more trades than the PA quality adapter.
    It should be used only for eval backtests / eval-mode lifecycle tests.
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
            "blocked_symbol": 0,
            "blocked_score": 0,
            "blocked_setup_keyword": 0,
            "long_seen": 0,
            "short_seen": 0,
            "mnq_accepted": 0,
            "mym_accepted": 0,
            "mes_accepted": 0,
            "sample_accepted": "",
            "sample_blocked": "",
        }

        self.sample_accepted = []
        self.sample_blocked = []

    def _sample(self, bucket: str, text: str):
        if bucket == "accepted":
            if len(self.sample_accepted) < 25 and text not in self.sample_accepted:
                self.sample_accepted.append(text[:500])
        else:
            if len(self.sample_blocked) < 25 and text not in self.sample_blocked:
                self.sample_blocked.append(text[:500])

    def write_diag(self):
        self.diag["sample_accepted"] = " || ".join(self.sample_accepted[:25])
        self.diag["sample_blocked"] = " || ".join(self.sample_blocked[:25])
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
        score = _extract_score(order)

        if side == "LONG":
            self.diag["long_seen"] += 1
        elif side == "SHORT":
            self.diag["short_seen"] += 1

        if symbol in BLOCKED_SYMBOLS or (ALLOWED_SYMBOLS and symbol not in ALLOWED_SYMBOLS):
            self.diag["blocked_symbol"] += 1
            self._sample("blocked", f"symbol={symbol}|side={side}|score={score}|{trade_type}")
            self.write_diag()
            return None

        if score < MIN_EVAL_SCORE:
            self.diag["blocked_score"] += 1
            self._sample("blocked", f"score_low|symbol={symbol}|side={side}|score={score}|{trade_type}")
            self.write_diag()
            return None

        for keyword in BLOCKED_SETUP_KEYWORDS:
            if keyword and keyword in trade_type:
                self.diag["blocked_setup_keyword"] += 1
                self._sample("blocked", f"keyword={keyword}|symbol={symbol}|side={side}|score={score}|{trade_type}")
                self.write_diag()
                return None

        _set_attr_if_possible(order, "strategy_name", self.name)

        reason = _get_attr_or_key(order, "reason", "")
        suffix = f"v2_eval|score={score}"

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

        self._sample("accepted", f"symbol={symbol}|side={side}|score={score}|{trade_type}")
        self.write_diag()

        return order
