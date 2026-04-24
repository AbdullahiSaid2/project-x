"""
HOW TO USE / WHERE THIS FILE GOES
=================================

Place this file at:
    trading_system/src/strategies/manual/prop_firm_profiles.py

Purpose:
- Central registry for prop-firm rule profiles.
- The strategy/model should NOT contain prop-firm-specific rules.
- Add new firms here and toggle them from the runner via --prop-profile.

Common usage:
- Add a new profile inside PROP_FIRM_PROFILES
- Keep "none" for raw model testing with no prop rules
- Use names like:
    apex_50k_eval
    apex_50k_eval_6loss
    topstep_50k_template
    ftmo_100k_template

Used by:
- src/strategies/manual/prop_guard.py
- src/strategies/manual/top_bottom_ticking_shared.py
- deployed launchers that pass --prop-profile

Example:
    from src.strategies.manual.prop_firm_profiles import get_prop_profile
    profile = get_prop_profile("apex_50k_eval")
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict

@dataclass(frozen=True)
class PropFirmProfile:
    name: str
    starting_balance: float
    daily_loss_limit: Optional[float] = None
    trailing_drawdown: Optional[float] = None
    max_consecutive_losses_per_day: Optional[int] = None
    max_trades_per_day: Optional[int] = None
    force_flat_time_et: Optional[str] = None
    allow_overnight: bool = False
    consistency_rule_pct: Optional[float] = None
    min_profitable_days: Optional[int] = None
    notes: str = ""

    def to_runtime_dict(self) -> dict:
        return asdict(self)

PROP_FIRM_PROFILES: Dict[str, PropFirmProfile] = {
    "none": PropFirmProfile(name="none", starting_balance=50_000.0, notes="No prop-firm restrictions applied."),
    "apex_50k_eval": PropFirmProfile(name="apex_50k_eval", starting_balance=50_000.0, daily_loss_limit=-1_000.0, force_flat_time_et="16:59", allow_overnight=False, notes="Apex-style daily loss only."),
    "apex_50k_eval_6loss": PropFirmProfile(name="apex_50k_eval_6loss", starting_balance=50_000.0, daily_loss_limit=-1_000.0, max_consecutive_losses_per_day=6, force_flat_time_et="16:59", allow_overnight=False, notes="Apex-style daily loss plus 6 same-day consecutive losses breaker."),
    "topstep_50k_template": PropFirmProfile(name="topstep_50k_template", starting_balance=50_000.0, allow_overnight=False, notes="Template profile. Fill exact Topstep rules before production use."),

    # FTMO 50k Standard / 2-Step style profile.
    # FTMO 2-Step rules: 5% maximum daily loss and 10% maximum loss.
    # For a 50k account this is: daily loss = $2,500, maximum loss = $5,000.
    # Note: this generic profile stores the 10% max-loss value in `trailing_drawdown`
    # because the current PropFirmProfile schema/guard already uses that field for
    # account-level drawdown protection. FTMO's 2-Step max loss is static, not
    # Apex-style trailing. If/when prop_guard.py gets a dedicated static max-loss
    # field, map this value to that field instead.
    "ftmo_50k_standard": PropFirmProfile(
        name="ftmo_50k_standard",
        starting_balance=50_000.0,
        daily_loss_limit=-2_500.0,
        trailing_drawdown=5_000.0,
        force_flat_time_et=None,
        allow_overnight=True,
        notes=(
            "FTMO 50k Standard / 2-Step style profile: 5% max daily loss ($2,500) "
            "and 10% max loss ($5,000). Standard FTMO Account news/weekend restrictions "
            "are handled outside this generic profile by cfd_restrictions.py / runner flags."
        ),
    ),

    "ftmo_50k_challenge_2step": PropFirmProfile(
        name="ftmo_50k_challenge_2step",
        starting_balance=50_000.0,
        daily_loss_limit=-2_500.0,
        trailing_drawdown=5_000.0,
        force_flat_time_et=None,
        allow_overnight=True,
        min_profitable_days=4,
        notes=(
            "FTMO 50k Challenge 2-Step style profile: 5% max daily loss ($2,500), "
            "10% max loss ($5,000), and 4 minimum trading days. Profit target is not "
            "enforced by prop_guard.py; use reports to check target progress."
        ),
    ),

    "ftmo_100k_template": PropFirmProfile(name="ftmo_100k_template", starting_balance=100_000.0, daily_loss_limit=-5_000.0, trailing_drawdown=10_000.0, allow_overnight=True, notes="Template FTMO 100k 2-Step style profile: 5% daily loss and 10% max loss. Confirm account/product-specific rules before production use."),
}

def get_prop_profile(name: str) -> PropFirmProfile:
    try:
        return PROP_FIRM_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROP_FIRM_PROFILES))
        raise KeyError(f"Unknown prop profile '{name}'. Available: {available}") from exc

def list_prop_profiles() -> list[str]:
    return sorted(PROP_FIRM_PROFILES)
