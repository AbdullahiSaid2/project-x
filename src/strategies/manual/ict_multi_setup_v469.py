"""
V469 strategy policy layer.

This file is designed to be dropped into the user's existing trading system and
used as the gating/configuration layer for the ICT multi-setup engine.

Core V469 rules:
- A-tier only
- London + NYPM continuation only
- Fixed 10 MNQ contracts
- Minimum planned RR 5.0
- Preserve runner logic
- Remove hard per-trade dollar floor filter

Because the full project is not available in this environment, this module is
kept intentionally dependency-light and focuses on the filtering/sizing policy.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional, Tuple


ALLOWED_SETUP_TYPES = {"LONDON_CONTINUATION", "NYPM_CONTINUATION"}
ALLOWED_SETUP_TIERS = {"A"}
MIN_PLANNED_RR = 5.0
FIXED_MNQ_CONTRACTS = 10
ENTRY_VARIANT = "PULLBACK_1M"


@dataclass(frozen=True)
class V469Policy:
    allowed_setup_types: frozenset[str] = frozenset(ALLOWED_SETUP_TYPES)
    allowed_setup_tiers: frozenset[str] = frozenset(ALLOWED_SETUP_TIERS)
    min_planned_rr: float = MIN_PLANNED_RR
    fixed_mnq_contracts: int = FIXED_MNQ_CONTRACTS
    required_entry_variant: str = ENTRY_VARIANT
    preserve_runner: bool = True
    allow_partial: bool = True
    hard_min_trade_profit_dollars: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_POLICY = V469Policy()


@dataclass
class GateDecision:
    allowed: bool
    reason: str
    contracts: int = 0


class ICTMultiSetupV469Mixin:
    """
    Mixin/policy adapter intended to be combined with the user's existing
    ICT multi-setup engine.

    Expected candidate keys:
    - setup_type
    - setup_tier
    - bridge_type
    - planned_rr
    - entry_variant

    Optional keys:
    - side
    - planned_entry_price
    - planned_stop_price
    - planned_target_price
    """

    policy: V469Policy = DEFAULT_POLICY

    def v469_gate_candidate(self, candidate: Dict[str, Any]) -> GateDecision:
        setup_type = str(candidate.get("setup_type", "")).upper()
        setup_tier = str(candidate.get("setup_tier", "")).upper()
        entry_variant = str(candidate.get("entry_variant", ""))

        planned_rr_raw = candidate.get("planned_rr", 0.0)
        try:
            planned_rr = float(planned_rr_raw)
        except (TypeError, ValueError):
            return GateDecision(False, f"bad planned_rr={planned_rr_raw!r}")

        if setup_type not in self.policy.allowed_setup_types:
            return GateDecision(False, f"filtered setup_type={setup_type}")

        if setup_tier not in self.policy.allowed_setup_tiers:
            return GateDecision(False, f"filtered setup_tier={setup_tier}")

        if planned_rr < self.policy.min_planned_rr:
            return GateDecision(False, f"planned_rr {planned_rr:.3f} < {self.policy.min_planned_rr:.3f}")

        if entry_variant and entry_variant != self.policy.required_entry_variant:
            return GateDecision(False, f"filtered entry_variant={entry_variant}")

        return GateDecision(True, "accepted", contracts=self.policy.fixed_mnq_contracts)

    def v469_apply_position_sizing(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(candidate)
        updated["report_contracts"] = self.policy.fixed_mnq_contracts
        updated["fixed_contracts"] = self.policy.fixed_mnq_contracts
        updated["dynamic_contracts"] = self.policy.fixed_mnq_contracts
        return updated

    def v469_apply_exit_preferences(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preserve the partial + runner model.
        This deliberately does not impose a hard dollar target floor.
        """
        updated = dict(candidate)
        updated["preserve_runner"] = self.policy.preserve_runner
        updated["allow_partial"] = self.policy.allow_partial
        updated["hard_min_trade_profit_dollars"] = self.policy.hard_min_trade_profit_dollars
        return updated

    def v469_prepare_candidate(self, candidate: Dict[str, Any]) -> Tuple[GateDecision, Dict[str, Any]]:
        decision = self.v469_gate_candidate(candidate)
        prepared = dict(candidate)
        if decision.allowed:
            prepared = self.v469_apply_position_sizing(prepared)
            prepared = self.v469_apply_exit_preferences(prepared)
        return decision, prepared


def filter_candidates_v469(candidates: Iterable[Dict[str, Any]], policy: V469Policy = DEFAULT_POLICY) -> list[Dict[str, Any]]:
    mixin = ICTMultiSetupV469Mixin()
    mixin.policy = policy
    accepted: list[Dict[str, Any]] = []
    for candidate in candidates:
        decision, prepared = mixin.v469_prepare_candidate(candidate)
        if decision.allowed:
            prepared["v469_gate_reason"] = decision.reason
            accepted.append(prepared)
    return accepted


# ---------------------------------------------------------------------------
# Adapter helper for direct integration into an existing strategy class.
# ---------------------------------------------------------------------------

def apply_v469_policy_to_strategy(strategy: Any, policy: V469Policy = DEFAULT_POLICY) -> Any:
    """
    Mutates an existing strategy object with V469-style knobs where present.
    Safe no-op for missing attributes.
    """
    setattr(strategy, "fixed_mnq_contracts", policy.fixed_mnq_contracts)
    setattr(strategy, "report_contracts", policy.fixed_mnq_contracts)
    setattr(strategy, "allowed_setup_types", set(policy.allowed_setup_types))
    setattr(strategy, "allowed_setup_tiers", set(policy.allowed_setup_tiers))
    setattr(strategy, "min_planned_rr", policy.min_planned_rr)
    setattr(strategy, "required_entry_variant", policy.required_entry_variant)
    setattr(strategy, "preserve_runner", policy.preserve_runner)
    setattr(strategy, "allow_partial", policy.allow_partial)
    setattr(strategy, "hard_min_trade_profit_dollars", policy.hard_min_trade_profit_dollars)
    return strategy
