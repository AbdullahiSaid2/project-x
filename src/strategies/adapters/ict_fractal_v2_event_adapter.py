from __future__ import annotations

"""
ICT Fractal V2 event adapter.

This is a safe wrapper around the existing exact ICT Fractal adapter.
It does NOT overwrite or rewrite v473_shared.py.

Old strategy remains:
    --strategy ict_fractal

Improved wrapper strategy:
    --strategy ict_fractal_v2

Improvements added here:
    1. Setup scoring
    2. Regime filtering
    3. Session-quality filtering
    4. Overextension protection
    5. Better OrderPlan setup_score values

The existing ICT Fractal adapter still controls:
    - original v473 state machine
    - narrative/context/bridge logic
    - entry trigger
    - stop placement
    - hybrid target selection

The event engine still controls:
    - fills
    - commissions
    - prop rules
    - news blackout
    - daily locks
    - account drawdown locks
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.strategies.adapters.ict_fractal_event_adapter import ICTFractalAdapter


@dataclass
class V2Decision:
    allowed: bool
    score: float
    reason: str


class ICTFractalV2Adapter:
    name = "ict_fractal_v2"

    def __init__(self):
        self.base = ICTFractalAdapter()

        # Minimum score gate. This is intentionally moderate so it improves
        # quality without killing the model. Can be controlled with the normal
        # --min-trend-score argument.
        self.default_min_score = 4.0

        # Session policy.
        self.allow_asia_if_elite = True
        self.prefer_london = True
        self.prefer_nyam = True
        self.allow_nypm = True

    def build_features(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        return self.base.build_features(symbol, df)

    def build_features_with_args(self, symbol: str, df: pd.DataFrame, args=None) -> pd.DataFrame:
        return self.build_features(symbol, df)

    # ---------------------------------------------------------
    # scoring helpers
    # ---------------------------------------------------------

    @staticmethod
    def _b(row, name: str) -> bool:
        try:
            return bool(row.get(name, False))
        except Exception:
            return False

    @staticmethod
    def _f(row, name: str, default: float = 0.0) -> float:
        try:
            val = row.get(name, default)
            if pd.isna(val):
                return default
            return float(val)
        except Exception:
            return default

    @staticmethod
    def _trade_side(order) -> str:
        return str(order.side).upper()

    @staticmethod
    def _trade_type(order) -> str:
        return str(getattr(order, "trade_type", ""))

    def _session_name(self, row) -> str:
        if self._b(row, "is_london") or self._b(row, "is_london_entry_window"):
            return "LONDON"
        if self._b(row, "is_nyam") or self._b(row, "is_nyam_entry_window"):
            return "NYAM"
        if self._b(row, "is_nypm") or self._b(row, "is_nypm_entry_window"):
            return "NYPM"
        if self._b(row, "is_asia") or self._b(row, "is_asia_entry_window"):
            return "ASIA"
        return "OTHER"

    def _setup_tier_from_trade_type(self, trade_type: str) -> str:
        # Existing exact adapter emits things like:
        #   LONDON_CONTINUATION_MSS_A_LONG
        parts = trade_type.split("_")
        for p in parts:
            if p in {"A", "B", "C"}:
                return p
        return ""

    def _score_order(self, symbol: str, order, row: pd.Series, history: pd.DataFrame) -> V2Decision:
        side = self._trade_side(order)
        trade_type = self._trade_type(order)
        tier = self._setup_tier_from_trade_type(trade_type)
        session = self._session_name(row)

        score = 0.0
        reasons = []

        # 1. Existing strategy tier.
        if tier == "A":
            score += 2.0
            reasons.append("tier_A")
        elif tier == "B":
            score += 1.25
            reasons.append("tier_B")
        elif tier == "C":
            score -= 1.0
            reasons.append("tier_C_penalty")

        # 2. Session quality.
        if session == "NYAM":
            score += 1.5
            reasons.append("nyam")
        elif session == "LONDON":
            score += 1.25
            reasons.append("london")
        elif session == "NYPM":
            score += 0.75
            reasons.append("nypm")
        elif session == "ASIA":
            score -= 0.25
            reasons.append("asia_penalty")
        else:
            score -= 1.0
            reasons.append("outside_core_session")

        # 3. HTF narrative alignment.
        if side == "LONG":
            if self._b(row, "bull_4h_bias"):
                score += 1.0
                reasons.append("bull_4h_bias")
            if self._b(row, "above_4h_eq"):
                score += 0.75
                reasons.append("above_4h_eq")
            if self._b(row, "bull_profile_4h") or self._b(row, "bull_disp_4h") or self._b(row, "bull_fvg_4h"):
                score += 0.75
                reasons.append("bull_4h_delivery")
        else:
            if self._b(row, "bear_4h_bias"):
                score += 1.0
                reasons.append("bear_4h_bias")
            if self._b(row, "below_4h_eq"):
                score += 0.75
                reasons.append("below_4h_eq")
            if self._b(row, "bear_profile_4h") or self._b(row, "bear_disp_4h") or self._b(row, "bear_fvg_4h"):
                score += 0.75
                reasons.append("bear_4h_delivery")

        # 4. 30m bridge / displacement quality.
        if side == "LONG":
            if self._b(row, "bull_disp_30m"):
                score += 1.0
                reasons.append("bull_disp_30m")
            if self._b(row, "bull_close_strong_30m"):
                score += 0.75
                reasons.append("bull_close_strong_30m")
            if self._b(row, "bull_mss_30m") or self._b(row, "bull_ifvg_30m") or self._b(row, "bull_c2_or_c3"):
                score += 0.75
                reasons.append("bull_bridge")
        else:
            if self._b(row, "bear_disp_30m"):
                score += 1.0
                reasons.append("bear_disp_30m")
            if self._b(row, "bear_close_strong_30m"):
                score += 0.75
                reasons.append("bear_close_strong_30m")
            if self._b(row, "bear_mss_30m") or self._b(row, "bear_ifvg_30m") or self._b(row, "bear_c2_or_c3"):
                score += 0.75
                reasons.append("bear_bridge")

        # 5. 3m execution quality and overextension protection.
        if side == "LONG":
            if self._b(row, "bull_disp_3m"):
                score += 0.75
                reasons.append("bull_disp_3m")
            if self._b(row, "bull_close_strong_3m"):
                score += 0.75
                reasons.append("bull_close_strong_3m")
            if self._b(row, "bull_overextended_3m"):
                score -= 2.0
                reasons.append("bull_overextended_block")
        else:
            if self._b(row, "bear_disp_3m"):
                score += 0.75
                reasons.append("bear_disp_3m")
            if self._b(row, "bear_close_strong_3m"):
                score += 0.75
                reasons.append("bear_close_strong_3m")
            if self._b(row, "bear_overextended_3m"):
                score -= 2.0
                reasons.append("bear_overextended_block")

        # 6. Liquidity context, especially useful in NY sessions.
        if side == "LONG":
            if self._b(row, "swept_prior_us_low") or self._b(row, "reclaimed_prior_us_low") or self._b(row, "swept_prev_day_low"):
                score += 0.75
                reasons.append("sellside_liquidity_context")
        else:
            if self._b(row, "swept_prior_us_high") or self._b(row, "rejected_prior_us_high") or self._b(row, "swept_prev_day_high"):
                score += 0.75
                reasons.append("buyside_liquidity_context")

        # 7. Risk/target sanity.
        entry = float(order.entry_price)
        stop = float(order.stop_price)
        target = float(order.target_price)
        risk_points = abs(entry - stop)
        target_points = abs(target - entry)
        rr = target_points / risk_points if risk_points > 0 else 0.0

        if risk_points <= 0:
            return V2Decision(False, score, "bad_risk")

        if rr < 1.5:
            score -= 1.0
            reasons.append("low_rr_penalty")
        elif rr >= 3.0:
            score += 0.5
            reasons.append("good_rr")

        # 8. Avoid extreme late/chasing entries after very large 3m ATR ranges.
        atr3 = self._f(row, "atr14_3m", 0.0)
        if atr3 > 0 and risk_points > atr3 * 5.0:
            score -= 1.0
            reasons.append("wide_risk_vs_atr_penalty")

        return V2Decision(True, score, "+".join(reasons))

    # ---------------------------------------------------------
    # event-engine interface
    # ---------------------------------------------------------

    def signal_for_row(self, symbol: str, row: pd.Series, history: pd.DataFrame, spec, profile, args):
        order = self.base.signal_for_row(symbol, row, history, spec, profile, args)
        if order is None:
            return None

        decision = self._score_order(symbol, order, row, history)

        min_score = float(getattr(args, "min_trend_score", self.default_min_score) or self.default_min_score)

        # Asia trades need to be elite because they are generally less useful for payout consistency.
        session = self._session_name(row)
        if session == "ASIA":
            min_score += 1.0

        if not decision.allowed or decision.score < min_score:
            return None

        # Set setup_score so the event engine / logs can later analyse quality.
        try:
            order.setup_score = float(decision.score)
        except Exception:
            pass

        # Make strategy name clear in logs.
        try:
            order.strategy_name = self.name
        except Exception:
            pass

        # Append score marker to reason for CSV analysis.
        try:
            order.reason = f"{order.reason}|v2_score={decision.score:.2f}|{decision.reason}"
        except Exception:
            pass

        return order
