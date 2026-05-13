from __future__ import annotations

from .models import SymbolSpec, PropProfile


SYMBOL_SPECS = {
    "MNQ": SymbolSpec("MNQ", dollars_per_point=2.0, min_tick=0.25),
    "MES": SymbolSpec("MES", dollars_per_point=5.0, min_tick=0.25),
    "MYM": SymbolSpec("MYM", dollars_per_point=0.5, min_tick=1.0),
    "MGC": SymbolSpec("MGC", dollars_per_point=10.0, min_tick=0.1),
    "MCL": SymbolSpec("MCL", dollars_per_point=100.0, min_tick=0.01),
}


PROP_PROFILES = {
    "apex_50k_eval": PropProfile(
        name="apex_50k_eval",
        mode="eval",
        account_size=50_000,
        risk_per_trade=250,
        daily_loss_limit=1_000,
        daily_soft_loss_stop=None,
        daily_profit_target=None,
        max_drawdown=2_000,
        drawdown_type="eod",
        drawdown_stop_level=None,
        news_blackout_enabled=False,
    ),
    "apex_50k_pa": PropProfile(
        name="apex_50k_pa",
        mode="pa",
        account_size=50_000,
        risk_per_trade=150,
        daily_loss_limit=1_000,
        daily_soft_loss_stop=350,
        daily_profit_target=500,
        max_drawdown=2_000,
        drawdown_type="eod",
        drawdown_stop_level=50_100,
        news_blackout_enabled=True,
        news_minutes_before=5,
        news_minutes_after=5,
        flatten_before_news=True,
        max_trades_per_day=5,
        pause_after_consecutive_losses=1,
    ),
}
