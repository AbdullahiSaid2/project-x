from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# =========================================================
# ICT FRACTAL V2 PRO ADAPTER
# =========================================================
#
# Strategy name:
#
#   --strategy ict_fractal_v2_pro
#
# This adapter is a professional refinement layer on top of the existing
# ICTFractalV2Adapter.
#
# It does NOT overwrite:
#
#   --strategy ict_fractal
#   --strategy ict_fractal_v2
#   --strategy ict_fractal_v2_quality
#
# Improvements added:
#
#   1. Setup-family allow/block rules.
#   2. Stronger selection for the proven families.
#   3. Symbol prioritisation.
#   4. Session-family control.
#   5. Daily cluster control.
#   6. Score thresholds by side/session/family.
#   7. Diagnostics output.
#
# NOTE:
#   True variable risk-per-setup requires event engine sizing support.
#   This adapter cannot change risk_per_trade unless the engine supports
#   per-order risk fields. So this version filters/scales by acceptance,
#   not by contract size yet.
#
# =========================================================


STRATEGY_NAME = "ict_fractal_v2_pro"

DIAG_DIR = Path("src/backtesting/event_engine/outputs")
DIAG_DIR.mkdir(parents=True, exist_ok=True)
DIAG_PATH = DIAG_DIR / "ict_fractal_v2_pro_filter_diagnostics.csv"


# =========================================================
# CONFIG — setup family controls
# =========================================================

# Known weak family from previous diagnostics.
BLOCKED_KEYWORDS = (
    "ASIA_CONTINUATION_iFVG_B_LONG",
)

# Very strong families from latest results / diagnostics.
ELITE_KEYWORDS = (
    "LONDON_CONTINUATION_C2C3_A_LONG",
    "LONDON_CONTINUATION_iFVG_A_LONG",
    "NYAM_CONTINUATION_C2C3_A_SHORT",
)

# Good but more selective.
GOOD_KEYWORDS = (
    "ASIA_CONTINUATION_iFVG_B_SHORT",
    "ASIA_CONTINUATION_C2C3_B_SHORT",
    "LONDON_CONTINUATION_iFVG_A_SHORT",
    "LONDON_CONTINUATION_C2C3_A_SHORT",
)

# Unknown setup families can pass only if their score is high.
ALLOW_UNKNOWN_FAMILIES = True

# Score gates.
ELITE_MIN_SCORE = 7.00
GOOD_MIN_SCORE = 6.00
UNKNOWN_MIN_SCORE = 8.00

# Longs are profitable but need quality.
LONG_MIN_SCORE = 7.00

# Shorts are also profitable, especially specific families.
SHORT_MIN_SCORE = 6.00

# Require good RR unless setup is elite.
REQUIRE_GOOD_RR_FOR_NON_ELITE = False

# Penalize wide risk unless setup is elite.
BLOCK_WIDE_RISK_NON_ELITE = False


# =========================================================
# CONFIG — symbol controls
# =========================================================

# Current evidence:
# MNQ strongest, MYM decent, MES small but OK, MGC not useful in last run.
ALLOWED_SYMBOLS = {
    "MNQ",
    "MYM",
    "MES",
}

BLOCKED_SYMBOLS = {
    "MGC",
}

# Daily caps by symbol.
MAX_SIGNALS_PER_SYMBOL_PER_DAY = {
    "MNQ": 4,
    "MYM": 3,
    "MES": 2,
}

# Total daily cap across all symbols.
MAX_TOTAL_SIGNALS_PER_DAY = 7

# Prevent same-symbol rapid-fire clustering.
MIN_MINUTES_BETWEEN_SYMBOL_SIGNALS = {
    "MNQ": 20,
    "MYM": 20,
    "MES": 20,
}


# =========================================================
# CONFIG — session controls
# =========================================================

# These are ET.
LONDON_START = time(2, 0)
LONDON_END = time(5, 30)

NYAM_START = time(8, 30)
NYAM_END = time(11, 30)

ASIA_START_1 = time(18, 0)
ASIA_END_1 = time(23, 59)

ASIA_START_2 = time(0, 0)
ASIA_END_2 = time(1, 30)

ENABLE_SESSION_FILTER = True

# Allow Asia only for shorts unless the family is elite.
ALLOW_ASIA_LONGS = False
ALLOW_ASIA_SHORTS = True

# NYAM shorts were strong; allow.
ALLOW_NYAM_LONGS = True
ALLOW_NYAM_SHORTS = True

ALLOW_LONDON_LONGS = True
ALLOW_LONDON_SHORTS = True


# =========================================================
# HELPERS
# =========================================================

def _to_et_timestamp(ts):
    ts = pd.Timestamp(ts)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    return ts.tz_convert("America/New_York")


def _date_et(ts):
    return _to_et_timestamp(ts).date()


def _time_et(ts):
    return _to_et_timestamp(ts).time()


def _in_window(t, start, end):
    return start <= t <= end


def _session_name(ts) -> str:
    t = _time_et(ts)

    if _in_window(t, LONDON_START, LONDON_END):
        return "london"

    if _in_window(t, NYAM_START, NYAM_END):
        return "nyam"

    if _in_window(t, ASIA_START_1, ASIA_END_1) or _in_window(t, ASIA_START_2, ASIA_END_2):
        return "asia"

    return "other"


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


def _extract_score(order: Any) -> Optional[float]:
    # First try explicit fields.
    for field in ("setup_score", "trend_score", "score", "signal_score"):
        value = _get_attr_or_key(order, field, None)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            pass

    # Then parse v2_score= from trade type/reason.
    text = _extract_trade_type(order)
    marker = "v2_score="
    if marker in text:
        try:
            raw = text.split(marker, 1)[1].split("|", 1)[0]
            return float(raw)
        except Exception:
            return None

    return None


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(k in text for k in keywords)


def _matched_family(text: str) -> str:
    for k in ELITE_KEYWORDS:
        if k in text:
            return k
    for k in GOOD_KEYWORDS:
        if k in text:
            return k
    for k in BLOCKED_KEYWORDS:
        if k in text:
            return k
    return "UNKNOWN"


def _minutes_between(a, b) -> float:
    if a is None or b is None:
        return 999999.0

    aa = pd.Timestamp(a)
    bb = pd.Timestamp(b)

    if aa.tzinfo is None:
        aa = aa.tz_localize("UTC")
    if bb.tzinfo is None:
        bb = bb.tz_localize("UTC")

    return abs((aa - bb).total_seconds()) / 60.0


# =========================================================
# DIAGNOSTICS
# =========================================================

def _empty_diag() -> dict:
    return {
        "base_orders_seen": 0,
        "accepted": 0,
        "blocked_symbol": 0,
        "blocked_session": 0,
        "blocked_family": 0,
        "blocked_score": 0,
        "blocked_rr_quality": 0,
        "blocked_cluster": 0,
        "blocked_daily_cap": 0,
        "long_seen": 0,
        "short_seen": 0,
        "long_accepted": 0,
        "short_accepted": 0,
        "mnq_accepted": 0,
        "mym_accepted": 0,
        "mes_accepted": 0,
        "elite_seen": 0,
        "elite_accepted": 0,
        "good_seen": 0,
        "good_accepted": 0,
        "unknown_seen": 0,
        "unknown_accepted": 0,
        "sample_accepted": "",
        "sample_blocked": "",
    }


# =========================================================
# FILTER
# =========================================================

@dataclass
class Decision:
    allowed: bool
    reason: str
    family: str
    score: Optional[float]
    session: str


class ICTFractalV2ProFilter:
    def __init__(self):
        self.diag = _empty_diag()
        self.symbol_day_counts: dict[tuple[str, object], int] = {}
        self.total_day_counts: dict[object, int] = {}
        self.last_symbol_signal_ts: dict[str, object] = {}
        self.sample_accepted: list[str] = []
        self.sample_blocked: list[str] = []

    def write_diag(self):
        self.diag["sample_accepted"] = " || ".join(self.sample_accepted[:25])
        self.diag["sample_blocked"] = " || ".join(self.sample_blocked[:25])
        pd.DataFrame([self.diag]).to_csv(DIAG_PATH, index=False)

    def _sample(self, bucket: str, text: str):
        if bucket == "accepted":
            if len(self.sample_accepted) < 25 and text not in self.sample_accepted:
                self.sample_accepted.append(text[:400])
        else:
            if len(self.sample_blocked) < 25 and text not in self.sample_blocked:
                self.sample_blocked.append(text[:400])

    def decide(self, symbol: str, row, order: Any) -> Decision:
        self.diag["base_orders_seen"] += 1

        symbol = str(symbol).upper()
        ts = getattr(row, "name", None)
        trade_type = _extract_trade_type(order)
        side = _extract_side(order)
        score = _extract_score(order)
        family = _matched_family(trade_type)
        session = _session_name(ts) if ts is not None else "unknown"

        if side == "LONG":
            self.diag["long_seen"] += 1
        elif side == "SHORT":
            self.diag["short_seen"] += 1

        if family in ELITE_KEYWORDS:
            self.diag["elite_seen"] += 1
        elif family in GOOD_KEYWORDS:
            self.diag["good_seen"] += 1
        else:
            self.diag["unknown_seen"] += 1

        if symbol in BLOCKED_SYMBOLS or (ALLOWED_SYMBOLS and symbol not in ALLOWED_SYMBOLS):
            self.diag["blocked_symbol"] += 1
            self._sample("blocked", f"symbol={symbol}|family={family}|side={side}|score={score}|{trade_type}")
            return Decision(False, "blocked_symbol", family, score, session)

        if score is None:
            score = 0.0

        # Known bad family.
        if _contains_any(trade_type, BLOCKED_KEYWORDS):
            self.diag["blocked_family"] += 1
            self._sample("blocked", f"blocked_keyword|symbol={symbol}|family={family}|side={side}|score={score}|{trade_type}")
            return Decision(False, "blocked_family", family, score, session)

        is_elite = _contains_any(trade_type, ELITE_KEYWORDS)
        is_good = _contains_any(trade_type, GOOD_KEYWORDS)

        # Session controls.
        if ENABLE_SESSION_FILTER:
            if session == "other":
                self.diag["blocked_session"] += 1
                self._sample("blocked", f"other_session|symbol={symbol}|family={family}|side={side}|score={score}|{trade_type}")
                return Decision(False, "blocked_session", family, score, session)

            if session == "asia":
                if side == "LONG" and not (ALLOW_ASIA_LONGS or is_elite):
                    self.diag["blocked_session"] += 1
                    self._sample("blocked", f"asia_long_blocked|symbol={symbol}|family={family}|score={score}|{trade_type}")
                    return Decision(False, "blocked_session", family, score, session)

                if side == "SHORT" and not ALLOW_ASIA_SHORTS:
                    self.diag["blocked_session"] += 1
                    self._sample("blocked", f"asia_short_blocked|symbol={symbol}|family={family}|score={score}|{trade_type}")
                    return Decision(False, "blocked_session", family, score, session)

            if session == "nyam":
                if side == "LONG" and not ALLOW_NYAM_LONGS:
                    self.diag["blocked_session"] += 1
                    return Decision(False, "blocked_session", family, score, session)
                if side == "SHORT" and not ALLOW_NYAM_SHORTS:
                    self.diag["blocked_session"] += 1
                    return Decision(False, "blocked_session", family, score, session)

            if session == "london":
                if side == "LONG" and not ALLOW_LONDON_LONGS:
                    self.diag["blocked_session"] += 1
                    return Decision(False, "blocked_session", family, score, session)
                if side == "SHORT" and not ALLOW_LONDON_SHORTS:
                    self.diag["blocked_session"] += 1
                    return Decision(False, "blocked_session", family, score, session)

        # Score controls.
        if is_elite:
            min_score = ELITE_MIN_SCORE
        elif is_good:
            min_score = GOOD_MIN_SCORE
        else:
            if not ALLOW_UNKNOWN_FAMILIES:
                self.diag["blocked_family"] += 1
                self._sample("blocked", f"unknown_family|symbol={symbol}|side={side}|score={score}|{trade_type}")
                return Decision(False, "blocked_unknown_family", family, score, session)
            min_score = UNKNOWN_MIN_SCORE

        if side == "LONG":
            min_score = max(min_score, LONG_MIN_SCORE)
        elif side == "SHORT":
            min_score = max(min_score, SHORT_MIN_SCORE)

        if score < min_score:
            self.diag["blocked_score"] += 1
            self._sample("blocked", f"score_low|min={min_score}|symbol={symbol}|family={family}|side={side}|score={score}|{trade_type}")
            return Decision(False, "blocked_score", family, score, session)

        # Optional qualitative tags.
        if REQUIRE_GOOD_RR_FOR_NON_ELITE and not is_elite and "good_rr" not in trade_type:
            self.diag["blocked_rr_quality"] += 1
            self._sample("blocked", f"no_good_rr|symbol={symbol}|family={family}|side={side}|score={score}|{trade_type}")
            return Decision(False, "blocked_rr_quality", family, score, session)

        if BLOCK_WIDE_RISK_NON_ELITE and not is_elite and "wide_risk_vs_atr_penalty" in trade_type:
            self.diag["blocked_rr_quality"] += 1
            self._sample("blocked", f"wide_risk|symbol={symbol}|family={family}|side={side}|score={score}|{trade_type}")
            return Decision(False, "blocked_wide_risk", family, score, session)

        # Daily caps and cluster controls.
        if ts is not None:
            day = _date_et(ts)
            sym_key = (symbol, day)

            sym_count = self.symbol_day_counts.get(sym_key, 0)
            total_count = self.total_day_counts.get(day, 0)

            max_sym = MAX_SIGNALS_PER_SYMBOL_PER_DAY.get(symbol, 2)

            if sym_count >= max_sym:
                self.diag["blocked_daily_cap"] += 1
                self._sample("blocked", f"symbol_daily_cap|symbol={symbol}|family={family}|side={side}|score={score}|{trade_type}")
                return Decision(False, "blocked_symbol_daily_cap", family, score, session)

            if total_count >= MAX_TOTAL_SIGNALS_PER_DAY:
                self.diag["blocked_daily_cap"] += 1
                return Decision(False, "blocked_total_daily_cap", family, score, session)

            last_ts = self.last_symbol_signal_ts.get(symbol)
            min_gap = MIN_MINUTES_BETWEEN_SYMBOL_SIGNALS.get(symbol, 20)

            if _minutes_between(ts, last_ts) < min_gap:
                self.diag["blocked_cluster"] += 1
                self._sample("blocked", f"cluster|symbol={symbol}|family={family}|side={side}|score={score}|{trade_type}")
                return Decision(False, "blocked_cluster", family, score, session)

            self.symbol_day_counts[sym_key] = sym_count + 1
            self.total_day_counts[day] = total_count + 1
            self.last_symbol_signal_ts[symbol] = ts

        self.diag["accepted"] += 1

        if side == "LONG":
            self.diag["long_accepted"] += 1
        elif side == "SHORT":
            self.diag["short_accepted"] += 1

        if symbol == "MNQ":
            self.diag["mnq_accepted"] += 1
        elif symbol == "MYM":
            self.diag["mym_accepted"] += 1
        elif symbol == "MES":
            self.diag["mes_accepted"] += 1

        if is_elite:
            self.diag["elite_accepted"] += 1
        elif is_good:
            self.diag["good_accepted"] += 1
        else:
            self.diag["unknown_accepted"] += 1

        self._sample("accepted", f"symbol={symbol}|family={family}|side={side}|score={score}|session={session}|{trade_type}")

        return Decision(True, "accepted", family, score, session)


# =========================================================
# ADAPTER
# =========================================================

class ICTFractalV2ProAdapter:
    name = STRATEGY_NAME
    strategy_name = STRATEGY_NAME

    def __init__(self):
        from src.strategies.adapters.ict_fractal_v2_event_adapter import (
            ICTFractalV2Adapter,
        )

        self.base = ICTFractalV2Adapter()
        self.filter = ICTFractalV2ProFilter()

    def build_features(self, symbol, df):
        if hasattr(self.base, "build_features"):
            return self.base.build_features(symbol, df)
        return df.copy()

    def build_features_with_args(self, symbol, df, args=None):
        if hasattr(self.base, "build_features_with_args"):
            return self.base.build_features_with_args(symbol, df, args)
        if hasattr(self.base, "build_features"):
            return self.base.build_features(symbol, df)
        return df.copy()

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

        decision = self.filter.decide(symbol, row, order)
        self.filter.write_diag()

        if not decision.allowed:
            return None

        _set_attr_if_possible(order, "strategy_name", self.name)

        reason = _get_attr_or_key(order, "reason", "")
        suffix = (
            f"v2_pro|family={decision.family}|score={decision.score}|session={decision.session}"
        )

        if reason:
            _set_attr_if_possible(order, "reason", f"{reason}|{suffix}")
        else:
            _set_attr_if_possible(order, "reason", suffix)

        return order
