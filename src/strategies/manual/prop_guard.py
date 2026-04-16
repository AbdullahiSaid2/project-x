"""
HOW TO USE / WHERE THIS FILE GOES
=================================

Place this file at:
    trading_system/src/strategies/manual/prop_guard.py

Purpose:
- Generic prop-firm guard engine.
- Reads a selected profile from prop_firm_profiles.py
- Decides whether a new trade is allowed BEFORE entry.
- Updates account state AFTER a trade closes.

Design intent:
- Strategy/model logic stays unchanged.
- Prop-firm rules remain separate and swappable.

Used by:
- src/strategies/manual/top_bottom_ticking_shared.py
- Future forward/live wrappers
- Future deployed apps

Typical flow:
    profile = get_prop_profile("apex_50k_eval")
    guard = PropFirmGuard(profile)

    decision = guard.can_open_trade(trade_day)
    if decision.allowed:
        # allow entry
        ...
    else:
        # block entry and log decision.reason
        ...

    # when trade closes
    guard.on_trade_closed(pnl_dollars, trade_day)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from prop_firm_profiles import PropFirmProfile

@dataclass
class GuardDecision:
    allowed: bool
    reason: Optional[str] = None

class PropFirmGuard:
    def __init__(self, profile: PropFirmProfile):
        self.profile = profile
        self.balance = float(profile.starting_balance)
        self.high_watermark = float(profile.starting_balance)
        self.current_day = None
        self.day_realized = 0.0
        self.trades_today = 0
        self.consecutive_losses_today = 0
        self.block_counts = {"daily_loss": 0, "consecutive_losses": 0, "max_trades": 0, "trailing_drawdown": 0}

    def on_new_day(self, trade_day) -> None:
        if self.current_day != trade_day:
            self.current_day = trade_day
            self.day_realized = 0.0
            self.trades_today = 0
            self.consecutive_losses_today = 0

    def can_open_trade(self, trade_day) -> GuardDecision:
        self.on_new_day(trade_day)
        p = self.profile
        if p.daily_loss_limit is not None and self.day_realized <= p.daily_loss_limit:
            self.block_counts["daily_loss"] += 1
            return GuardDecision(False, "daily_loss")
        if p.max_consecutive_losses_per_day is not None and self.consecutive_losses_today >= p.max_consecutive_losses_per_day:
            self.block_counts["consecutive_losses"] += 1
            return GuardDecision(False, "consecutive_losses")
        if p.max_trades_per_day is not None and self.trades_today >= p.max_trades_per_day:
            self.block_counts["max_trades"] += 1
            return GuardDecision(False, "max_trades")
        if p.trailing_drawdown is not None and self.balance <= self.high_watermark - p.trailing_drawdown:
            self.block_counts["trailing_drawdown"] += 1
            return GuardDecision(False, "trailing_drawdown")
        return GuardDecision(True, None)

    def on_trade_closed(self, pnl_dollars: float, trade_day) -> None:
        self.on_new_day(trade_day)
        pnl_dollars = float(pnl_dollars)
        self.balance += pnl_dollars
        self.day_realized += pnl_dollars
        self.trades_today += 1
        self.high_watermark = max(self.high_watermark, self.balance)
        self.consecutive_losses_today = self.consecutive_losses_today + 1 if pnl_dollars < 0 else 0
