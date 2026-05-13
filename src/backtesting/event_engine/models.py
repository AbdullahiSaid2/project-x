from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    dollars_per_point: float
    min_tick: float = 0.25


@dataclass(frozen=True)
class PropProfile:
    name: str
    mode: str
    account_size: float = 50_000.0
    risk_per_trade: float = 250.0
    daily_loss_limit: float = 1_000.0
    daily_soft_loss_stop: Optional[float] = None
    daily_profit_target: Optional[float] = None
    max_drawdown: float = 2_000.0
    drawdown_type: str = "eod"
    drawdown_stop_level: Optional[float] = None
    flatten_time_et: str = "16:50"
    reopen_time_et: str = "18:00"
    news_blackout_enabled: bool = False
    news_minutes_before: int = 5
    news_minutes_after: int = 5
    flatten_before_news: bool = True
    max_trades_per_day: Optional[int] = None
    pause_after_consecutive_losses: Optional[int] = None


@dataclass
class OrderPlan:
    symbol: str
    side: str
    entry_price: float
    stop_price: float
    target_price: float
    trade_type: str
    reason: str = ""
    strategy_name: str = ""
    setup_score: float = 0.0


@dataclass
class Position:
    symbol: str
    side: str
    size: int
    entry_price: float
    stop_price: float
    target_price: float
    entry_time: object
    entry_bar_index: int
    trade_type: str
    strategy_name: str
    planned_risk_dollars: float
    planned_target_dollars: float
