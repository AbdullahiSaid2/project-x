from __future__ import annotations

"""
Shared event-driven backtest engine v2.

Adds the payout-management CLI flags required by payout_optimisation_runner.py:

  --risk-per-trade
  --daily-profit-target
  --daily-soft-loss-stop
  --max-trades-per-day
  --pause-after-consecutive-losses

Also adds:
  - realised + unrealised daily equity profit lock
  - realised + unrealised daily soft-loss lock
  - daily summary output
"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fetcher import get_ohlcv
from src.backtesting.event_engine.defaults import SYMBOL_SPECS, PROP_PROFILES
from src.backtesting.event_engine.models import Position, PropProfile
from src.backtesting.event_engine.time_rules import (
    to_et,
    is_allowed_futures_time,
    should_force_flat,
    session_date,
    load_news_events,
    news_blackout_status,
)

OUT_DIR = Path("src/backtesting/event_engine/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_adapter(name: str):
    if name == "top_bottom_ticking":
        from src.strategies.adapters.top_bottom_ticking_event_adapter import (
            TopBottomTickingAdapter,
        )
        return TopBottomTickingAdapter()

    if name == "top_bottom_ticking_v2":
        from src.strategies.adapters.top_bottom_ticking_v2_event_adapter import (
            TopBottomTickingV2Adapter,
        )
        return TopBottomTickingV2Adapter()

    if name == "ict_fractal":
        from src.strategies.adapters.ict_fractal_event_adapter import (
            ICTFractalAdapter,
        )
        return ICTFractalAdapter()

    if name == "ict_fractal_v2":
        from src.strategies.adapters.ict_fractal_v2_event_adapter import (
            ICTFractalV2Adapter,
        )
        return ICTFractalV2Adapter()

    if name == "ict_fractal_v2_quality":
        from src.strategies.adapters.ict_fractal_v2_quality_event_adapter import (
            ICTFractalV2QualityAdapter,
        )
        return ICTFractalV2QualityAdapter()

    if name == "ict_fractal_v2_pro":
        from src.strategies.adapters.ict_fractal_v2_pro_event_adapter import (
            ICTFractalV2ProAdapter,
        )
        return ICTFractalV2ProAdapter()

    if name == "ict_fractal_v2_quality_plus":
        from src.strategies.adapters.ict_fractal_v2_quality_plus_event_adapter import (
            ICTFractalV2QualityPlusAdapter,
        )
        return ICTFractalV2QualityPlusAdapter()

    if name == "ict_fractal_v2_eval":
        from src.strategies.adapters.ict_fractal_v2_eval_event_adapter import (
            ICTFractalV2EvalAdapter,
        )
        return ICTFractalV2EvalAdapter()

    raise ValueError(f"Unknown strategy adapter: {name}")


def apply_profile_overrides(profile: PropProfile, args) -> PropProfile:
    updates = {}
    if args.risk_per_trade is not None:
        updates["risk_per_trade"] = float(args.risk_per_trade)
    if args.daily_profit_target is not None:
        updates["daily_profit_target"] = float(args.daily_profit_target)
    if args.daily_soft_loss_stop is not None:
        updates["daily_soft_loss_stop"] = float(args.daily_soft_loss_stop)
    if args.max_trades_per_day is not None:
        updates["max_trades_per_day"] = int(args.max_trades_per_day)
    if args.pause_after_consecutive_losses is not None:
        updates["pause_after_consecutive_losses"] = int(args.pause_after_consecutive_losses)
    return replace(profile, **updates) if updates else profile


def calculate_contracts(profile: PropProfile, spec, entry: float, stop: float) -> tuple[int, float, float]:
    risk_points = abs(entry - stop)
    if risk_points <= 0:
        return 0, 0.0, 0.0
    risk_per_contract = risk_points * spec.dollars_per_point
    if risk_per_contract <= 0:
        return 0, risk_per_contract, 0.0
    contracts = int(profile.risk_per_trade // risk_per_contract)
    if contracts < 1:
        return 0, risk_per_contract, 0.0
    return contracts, risk_per_contract, risk_per_contract * contracts


def position_unrealized(pos: Position, spec, mark_price: float) -> float:
    if pos.side == "LONG":
        points = mark_price - pos.entry_price
    else:
        points = pos.entry_price - mark_price
    return points * spec.dollars_per_point * pos.size


def total_open_unrealized(positions: Dict[str, Position], data: dict, ts, management_states: dict | None = None) -> float:
    total = 0.0
    management_states = management_states or {}

    for sym, pos in positions.items():
        if ts not in data[sym].index:
            continue
        mark = float(data[sym].loc[ts]["Close"])
        total += position_unrealized_managed(
            pos,
            SYMBOL_SPECS[sym],
            mark,
            management_states.get(sym, {}),
        )

    return total


def _empty_trade_management_state(pos: Position) -> dict:
    """
    Stores trade-management state outside the Position dataclass so we do not
    need to modify src/backtesting/event_engine/models.py.
    """
    return {
        "initial_stop_price": float(pos.stop_price),
        "breakeven_active": False,
        "partial_taken": False,
        "runner_mode": False,
        "remaining_fraction": 1.0,
        "partial_realized_pnl": 0.0,
        "partial_commissions_dollars": 0.0,
        "breakeven_triggered_count": 0,
        "partial_triggered_count": 0,
        "runner_trail_updates": 0,
    }


def _risk_points_from_state(pos: Position, state: dict) -> float:
    initial_stop = float(state.get("initial_stop_price", pos.stop_price))
    return abs(float(pos.entry_price) - initial_stop)


def _favourable_r_for_bar(pos: Position, state: dict, row) -> float:
    risk_points = _risk_points_from_state(pos, state)
    if risk_points <= 0:
        return 0.0

    if pos.side == "LONG":
        favourable_points = float(row["High"]) - float(pos.entry_price)
    else:
        favourable_points = float(pos.entry_price) - float(row["Low"])

    return favourable_points / risk_points


def _partial_exit_price(pos: Position, state: dict, trigger_r: float) -> float:
    risk_points = _risk_points_from_state(pos, state)
    if pos.side == "LONG":
        return float(pos.entry_price) + (risk_points * trigger_r)
    return float(pos.entry_price) - (risk_points * trigger_r)


def _partial_pnl_at_price(
    pos: Position,
    spec,
    exit_price: float,
    exit_fraction: float,
    commission_per_contract_side: float,
) -> tuple[float, float, float]:
    """
    Returns gross, commissions, net for a partial exit.

    Uses fractional contract accounting for backtest smoothness. The final
    trade log still reports the original contract count and the remaining
    fraction so you can audit the result.
    """
    if pos.side == "LONG":
        points = float(exit_price) - float(pos.entry_price)
    else:
        points = float(pos.entry_price) - float(exit_price)

    effective_contracts = float(pos.size) * float(exit_fraction)
    gross = points * float(spec.dollars_per_point) * effective_contracts

    # Partial exit pays one side commission because entry commission is charged
    # on final close in the existing engine's round-trip model. This is not
    # perfect exchange accounting, but it avoids double-charging the entry side.
    commissions = float(commission_per_contract_side) * effective_contracts
    net = gross - commissions

    return gross, commissions, net


def _update_runner_trailing_stop(pos: Position, state: dict, history, lookback: int):
    if history is None or len(history) < 2:
        return

    recent = history.tail(max(1, int(lookback)))

    if pos.side == "LONG":
        if "Low" in recent.columns:
            new_stop = float(recent["Low"].min())
        else:
            new_stop = float(recent["low"].min())

        if new_stop > float(pos.stop_price) and new_stop < float(pos.entry_price) or (
            state.get("breakeven_active") and new_stop > float(pos.stop_price)
        ):
            pos.stop_price = new_stop
            state["runner_trail_updates"] = int(state.get("runner_trail_updates", 0)) + 1

    else:
        if "High" in recent.columns:
            new_stop = float(recent["High"].max())
        else:
            new_stop = float(recent["high"].max())

        if new_stop < float(pos.stop_price) and new_stop > float(pos.entry_price) or (
            state.get("breakeven_active") and new_stop < float(pos.stop_price)
        ):
            pos.stop_price = new_stop
            state["runner_trail_updates"] = int(state.get("runner_trail_updates", 0)) + 1


def manage_position_before_exit(
    pos: Position,
    state: dict,
    spec,
    row,
    history,
    args,
    commission_per_contract_side: float,
):
    """
    Applies trade management before stop/target checks:
      - move stop to breakeven after +R trigger
      - take partial after +R trigger
      - trail runner after partial

    This is intentionally engine-level so all strategies can use it.
    """
    current_r = _favourable_r_for_bar(pos, state, row)

    if args.enable_breakeven and not state.get("breakeven_active", False):
        if current_r >= float(args.breakeven_trigger_r):
            pos.stop_price = float(pos.entry_price)
            state["breakeven_active"] = True
            state["breakeven_triggered_count"] = int(state.get("breakeven_triggered_count", 0)) + 1

    if args.enable_partials and not state.get("partial_taken", False):
        if current_r >= float(args.partial_trigger_r):
            close_fraction = max(0.0, min(1.0, float(args.partial_close_pct)))
            close_fraction = min(close_fraction, float(state.get("remaining_fraction", 1.0)))

            if close_fraction > 0:
                partial_price = _partial_exit_price(pos, state, float(args.partial_trigger_r))
                _, partial_commissions, partial_net = _partial_pnl_at_price(
                    pos=pos,
                    spec=spec,
                    exit_price=partial_price,
                    exit_fraction=close_fraction,
                    commission_per_contract_side=commission_per_contract_side,
                )

                state["remaining_fraction"] = max(
                    0.0,
                    float(state.get("remaining_fraction", 1.0)) - close_fraction,
                )
                state["partial_realized_pnl"] = float(state.get("partial_realized_pnl", 0.0)) + partial_net
                state["partial_commissions_dollars"] = float(
                    state.get("partial_commissions_dollars", 0.0)
                ) + partial_commissions
                state["partial_taken"] = True
                state["runner_mode"] = True
                state["partial_triggered_count"] = int(state.get("partial_triggered_count", 0)) + 1

                if args.enable_breakeven and not state.get("breakeven_active", False):
                    pos.stop_price = float(pos.entry_price)
                    state["breakeven_active"] = True
                    state["breakeven_triggered_count"] = int(state.get("breakeven_triggered_count", 0)) + 1

    if args.enable_runner_trail and state.get("runner_mode", False):
        _update_runner_trailing_stop(
            pos=pos,
            state=state,
            history=history,
            lookback=int(args.runner_trail_lookback),
        )


def position_unrealized_managed(pos: Position, spec, mark_price: float, state: dict | None = None) -> float:
    remaining_fraction = 1.0
    partial_realized = 0.0

    if state is not None:
        remaining_fraction = float(state.get("remaining_fraction", 1.0))
        partial_realized = float(state.get("partial_realized_pnl", 0.0))

    if pos.side == "LONG":
        points = mark_price - pos.entry_price
    else:
        points = pos.entry_price - mark_price

    open_unreal = points * spec.dollars_per_point * pos.size * remaining_fraction
    return open_unreal + partial_realized


def close_position(
    pos: Position,
    spec,
    ts,
    row,
    exit_price: float,
    exit_reason: str,
    commission_per_contract_side: float,
    same_bar: bool,
    state: dict | None = None,
):
    state = state or {}

    remaining_fraction = float(state.get("remaining_fraction", 1.0))
    partial_realized = float(state.get("partial_realized_pnl", 0.0))
    partial_commissions = float(state.get("partial_commissions_dollars", 0.0))

    if pos.side == "LONG":
        points = exit_price - pos.entry_price
    else:
        points = pos.entry_price - exit_price

    effective_contracts = float(pos.size) * remaining_fraction

    gross_remaining = points * spec.dollars_per_point * effective_contracts

    # Remaining exit pays full round-trip commission scaled by remaining size,
    # while partial exits separately paid one-sided commission when taken.
    remaining_commissions = commission_per_contract_side * 2.0 * effective_contracts

    gross = gross_remaining + partial_realized + partial_commissions
    commissions = remaining_commissions + partial_commissions
    net = gross_remaining - remaining_commissions + partial_realized

    tags = []
    if state.get("partial_taken", False):
        tags.append("partial_taken")
    if state.get("breakeven_active", False):
        tags.append("breakeven_active")
    if state.get("runner_mode", False):
        tags.append("runner_mode")
    if tags:
        exit_reason = exit_reason + "|" + "|".join(tags)

    return {
        "strategy_name": pos.strategy_name,
        "symbol": pos.symbol,
        "side": pos.side,
        "size": pos.size,
        "entry_time_et": to_et(pos.entry_time),
        "exit_time_et": to_et(ts),
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "realized_points": points,
        "dollars_per_point": spec.dollars_per_point,
        "gross_pnl_dollars": gross,
        "commissions_dollars": commissions,
        "net_pnl_dollars": net,
        "trade_type": pos.trade_type,
        "exit_reason": exit_reason,
        "planned_risk_dollars": pos.planned_risk_dollars,
        "planned_target_dollars": pos.planned_target_dollars,
        "same_bar_exit": same_bar,
        "breakeven_active": bool(state.get("breakeven_active", False)),
        "partial_taken": bool(state.get("partial_taken", False)),
        "runner_mode": bool(state.get("runner_mode", False)),
        "remaining_fraction": remaining_fraction,
        "partial_realized_pnl": partial_realized,
        "partial_commissions_dollars": partial_commissions,
        "runner_trail_updates": int(state.get("runner_trail_updates", 0)),
    }


def flatten_all(
    positions: Dict[str, Position],
    data: dict,
    ts,
    reason: str,
    commission_per_contract_side: float,
    management_states: dict | None = None,
):
    closed = []
    management_states = management_states or {}

    for sym in list(positions.keys()):
        if ts not in data[sym].index:
            continue

        row = data[sym].loc[ts]
        state = management_states.get(sym, {})

        trade = close_position(
            positions[sym],
            SYMBOL_SPECS[sym],
            ts,
            row,
            float(row["Close"]),
            reason,
            commission_per_contract_side,
            False,
            state,
        )
        closed.append(trade)
        del positions[sym]
        management_states.pop(sym, None)

    return closed


def maybe_exit_position(
    pos: Position,
    spec,
    ts,
    bar_index: int,
    row,
    commission_per_contract_side: float,
    state: dict | None = None,
):
    # No same-bar exits to avoid candle-order ambiguity.
    if bar_index <= pos.entry_bar_index:
        return None

    high = float(row["High"])
    low = float(row["Low"])

    state = state or {}

    if pos.side == "LONG":
        # Conservative: if both target and stop are touched, stop wins.
        if low <= pos.stop_price:
            return close_position(
                pos,
                spec,
                ts,
                row,
                pos.stop_price,
                "stop_loss",
                commission_per_contract_side,
                False,
                state,
            )
        if high >= pos.target_price:
            return close_position(
                pos,
                spec,
                ts,
                row,
                pos.target_price,
                "take_profit",
                commission_per_contract_side,
                False,
                state,
            )

    if pos.side == "SHORT":
        if high >= pos.stop_price:
            return close_position(
                pos,
                spec,
                ts,
                row,
                pos.stop_price,
                "stop_loss",
                commission_per_contract_side,
                False,
                state,
            )
        if low <= pos.target_price:
            return close_position(
                pos,
                spec,
                ts,
                row,
                pos.target_price,
                "take_profit",
                commission_per_contract_side,
                False,
                state,
            )

    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True, choices=["top_bottom_ticking", "top_bottom_ticking_v2", "ict_fractal", "ict_fractal_v2", "ict_fractal_v2_quality", "ict_fractal_v2_pro", "ict_fractal_v2_quality_plus", "ict_fractal_v2_eval"])
    p.add_argument("--symbols", nargs="+", default=["MNQ", "MES", "MYM", "MGC"])
    p.add_argument("--prop-profile", default="apex_50k_pa", choices=list(PROP_PROFILES.keys()))
    p.add_argument("--days-back", type=int, default=365)
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--tail-rows", type=int, default=180_000)
    p.add_argument("--no-tail", action="store_true")
    p.add_argument("--commission-per-contract-side", type=float, default=2.0)
    p.add_argument("--min-trend-score", type=float, default=3)
    p.add_argument("--target-r", type=float, default=4.0)
    p.add_argument("--min-planned-target-dollars", type=float, default=0.0)
    p.add_argument("--news-events", default="")
    p.add_argument("--output-prefix", default="")

    # Required by payout_optimisation_runner.py
    p.add_argument("--risk-per-trade", type=float, default=None)
    p.add_argument("--daily-profit-target", type=float, default=None)
    p.add_argument("--daily-soft-loss-stop", type=float, default=None)
    p.add_argument("--max-trades-per-day", type=int, default=None)
    p.add_argument("--pause-after-consecutive-losses", type=int, default=None)

    # Optional behaviour flags
    p.add_argument("--disable-unrealized-daily-lock", action="store_true")
    p.add_argument("--disable-account-drawdown-lock", action="store_true")
    # Trade management controls
    p.add_argument("--enable-breakeven", action="store_true")
    p.add_argument("--breakeven-trigger-r", type=float, default=1.0)
    p.add_argument("--enable-partials", action="store_true")
    p.add_argument("--partial-trigger-r", type=float, default=2.0)
    p.add_argument("--partial-close-pct", type=float, default=0.50)
    p.add_argument("--enable-runner-trail", action="store_true")
    p.add_argument("--runner-trail-lookback", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    adapter = load_adapter(args.strategy)
    profile = apply_profile_overrides(PROP_PROFILES[args.prop_profile], args)
    unrealized_daily_lock_enabled = not args.disable_unrealized_daily_lock
    account_drawdown_lock_enabled = not args.disable_account_drawdown_lock

    news_events = load_news_events(args.news_events) if profile.news_blackout_enabled else pd.DataFrame()

    symbols = [s.upper() for s in args.symbols]
    data = {}
    bar_index_maps = {}

    print("Loading data and building features...")
    for sym in symbols:
        if sym not in SYMBOL_SPECS:
            raise ValueError(f"Unknown symbol spec: {sym}. Add it to defaults.py")
        print(f"\n=== loading {sym} ===")
        df = get_ohlcv(sym, exchange="tradovate", timeframe=args.timeframe, days_back=args.days_back)
        if not args.no_tail:
            df = df.tail(args.tail_rows)
        if hasattr(adapter, "build_features_with_args"):
            df = adapter.build_features_with_args(sym, df, args)
        else:
            df = adapter.build_features(sym, df)
        data[sym] = df
        bar_index_maps[sym] = {ts: i for i, ts in enumerate(df.index)}
        print(f"{sym}: rows={len(df)} start={df.index.min()} end={df.index.max()}")

    all_timestamps = sorted(set().union(*[set(df.index) for df in data.values()]))
    print(f"\nRunning {adapter.name} event engine v2 over {len(all_timestamps)} timestamps...")

    positions: Dict[str, Position] = {}
    management_states: Dict[str, dict] = {}
    trades = []
    rejected = []
    daily_rows = []

    balance = profile.account_size
    peak_balance = balance
    eod_peak_balance = balance
    drawdown_floor = profile.account_size - profile.max_drawdown

    current_session = None
    daily_net_pnl = 0.0
    daily_trades = 0
    consecutive_losses = 0
    day_locked = False
    day_lock_reason = ""

    breakeven_activation_count = 0
    partial_exit_count = 0
    runner_trail_update_count = 0

    def record_trade_and_update(trade):
        nonlocal balance, daily_net_pnl, peak_balance, consecutive_losses
        trades.append(trade)
        pnl = float(trade["net_pnl_dollars"])
        balance += pnl
        daily_net_pnl += pnl
        peak_balance = max(peak_balance, balance)
        consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0

    def close_session_if_needed(next_session):
        nonlocal eod_peak_balance, drawdown_floor, current_session, daily_net_pnl, daily_trades, consecutive_losses, day_locked, day_lock_reason
        if current_session is not None:
            eod_peak_balance = max(eod_peak_balance, balance)
            if profile.drawdown_type == "eod":
                new_floor = eod_peak_balance - profile.max_drawdown
                if profile.drawdown_stop_level is not None:
                    new_floor = min(new_floor, profile.drawdown_stop_level)
                drawdown_floor = max(drawdown_floor, new_floor)
            daily_rows.append({
                "session_date": current_session,
                "balance": balance,
                "daily_net_pnl": daily_net_pnl,
                "daily_trades": daily_trades,
                "eod_peak_balance": eod_peak_balance,
                "drawdown_floor": drawdown_floor,
                "day_locked": day_locked,
                "day_lock_reason": day_lock_reason,
            })

        current_session = next_session
        daily_net_pnl = 0.0
        daily_trades = 0
        consecutive_losses = 0
        day_locked = False
        day_lock_reason = ""

    for ts in all_timestamps:
        et = to_et(ts)
        sess = session_date(et, reopen_time=profile.reopen_time_et)
        if sess != current_session:
            close_session_if_needed(sess)

        in_news, _ = news_blackout_status(
            et, news_events, profile.news_minutes_before, profile.news_minutes_after
        ) if profile.news_blackout_enabled else (False, "")

        force_flat_now = should_force_flat(et, flatten_time=profile.flatten_time_et) or not is_allowed_futures_time(
            et,
            flatten_time=profile.flatten_time_et,
            reopen_time=profile.reopen_time_et,
        )
        if in_news and profile.flatten_before_news:
            force_flat_now = True

        if force_flat_now and positions:
            reason = "force_flat_news" if in_news else "force_flat_session"
            for trade in flatten_all(positions, data, ts, reason, args.commission_per_contract_side, management_states):
                record_trade_and_update(trade)

        # Normal exits.
        for sym in list(positions.keys()):
            if ts not in data[sym].index:
                continue
            row = data[sym].loc[ts]
            state = management_states.setdefault(sym, _empty_trade_management_state(positions[sym]))
            history = data[sym].iloc[: bar_index_maps[sym][ts] + 1]

            if bar_index_maps[sym][ts] > positions[sym].entry_bar_index:
                before_be = bool(state.get("breakeven_active", False))
                before_partial = bool(state.get("partial_taken", False))
                before_trails = int(state.get("runner_trail_updates", 0))

                manage_position_before_exit(
                    positions[sym],
                    state,
                    SYMBOL_SPECS[sym],
                    row,
                    history,
                    args,
                    args.commission_per_contract_side,
                )

                if (not before_be) and state.get("breakeven_active", False):
                    breakeven_activation_count += 1
                if (not before_partial) and state.get("partial_taken", False):
                    partial_exit_count += 1
                runner_trail_update_count += max(
                    0,
                    int(state.get("runner_trail_updates", 0)) - before_trails,
                )

            trade = maybe_exit_position(
                positions[sym],
                SYMBOL_SPECS[sym],
                ts,
                bar_index_maps[sym][ts],
                row,
                args.commission_per_contract_side,
                state,
            )
            if trade:
                record_trade_and_update(trade)
                del positions[sym]
                management_states.pop(sym, None)

        # Realised + unrealised daily locks.
        open_unreal = total_open_unrealized(positions, data, ts, management_states) if positions else 0.0
        daily_equity_pnl = daily_net_pnl + open_unreal

        if unrealized_daily_lock_enabled and positions:
            if profile.daily_profit_target is not None and daily_equity_pnl >= profile.daily_profit_target:
                for trade in flatten_all(positions, data, ts, "daily_equity_profit_lock", args.commission_per_contract_side, management_states):
                    record_trade_and_update(trade)
                day_locked = True
                day_lock_reason = "daily_equity_profit_lock"

            elif profile.daily_soft_loss_stop is not None and daily_equity_pnl <= -abs(profile.daily_soft_loss_stop):
                for trade in flatten_all(positions, data, ts, "daily_equity_soft_loss_lock", args.commission_per_contract_side, management_states):
                    record_trade_and_update(trade)
                day_locked = True
                day_lock_reason = "daily_equity_soft_loss_lock"

        # Realised-only locks.
        if profile.daily_loss_limit is not None and daily_net_pnl <= -abs(profile.daily_loss_limit):
            day_locked = True
            day_lock_reason = "daily_loss_limit"
        if profile.daily_soft_loss_stop is not None and daily_net_pnl <= -abs(profile.daily_soft_loss_stop):
            day_locked = True
            day_lock_reason = "daily_soft_loss_stop"
        if profile.daily_profit_target is not None and daily_net_pnl >= profile.daily_profit_target:
            day_locked = True
            day_lock_reason = "daily_profit_target"
        if profile.max_trades_per_day is not None and daily_trades >= profile.max_trades_per_day:
            day_locked = True
            day_lock_reason = "max_trades_per_day"
        if profile.pause_after_consecutive_losses is not None and consecutive_losses >= profile.pause_after_consecutive_losses:
            day_locked = True
            day_lock_reason = "consecutive_losses"

        if account_drawdown_lock_enabled and balance <= drawdown_floor:
            day_locked = True
            day_lock_reason = "max_drawdown_breach"

        # Entries.
        if force_flat_now or day_locked or in_news:
            continue

        for sym in symbols:
            if sym in positions:
                continue
            if ts not in data[sym].index:
                continue

            df = data[sym]
            idx = bar_index_maps[sym][ts]
            row = df.iloc[idx]
            history = df.iloc[: idx + 1]
            spec = SYMBOL_SPECS[sym]

            order = adapter.signal_for_row(sym, row, history, spec, profile, args)
            if order is None:
                continue

            contracts, risk_per_contract, planned_risk = calculate_contracts(profile, spec, order.entry_price, order.stop_price)
            if contracts < 1:
                rejected.append({
                    "timestamp_et": et,
                    "symbol": sym,
                    "reject_reason": "risk_too_large_for_one_contract",
                    "trade_type": order.trade_type,
                    "entry_price": order.entry_price,
                    "stop_price": order.stop_price,
                    "risk_per_contract": risk_per_contract,
                })
                continue

            planned_target = abs(order.target_price - order.entry_price) * spec.dollars_per_point * contracts
            if planned_target < args.min_planned_target_dollars:
                rejected.append({
                    "timestamp_et": et,
                    "symbol": sym,
                    "reject_reason": "planned_target_too_small",
                    "trade_type": order.trade_type,
                    "planned_target_dollars": planned_target,
                })
                continue

            positions[sym] = Position(
                symbol=sym,
                side=order.side,
                size=contracts,
                entry_price=order.entry_price,
                stop_price=order.stop_price,
                target_price=order.target_price,
                entry_time=ts,
                entry_bar_index=idx,
                trade_type=order.trade_type,
                strategy_name=adapter.name,
                planned_risk_dollars=planned_risk,
                planned_target_dollars=planned_target,
            )
            management_states[sym] = _empty_trade_management_state(positions[sym])
            daily_trades += 1

    # End-of-test close.
    for sym in list(positions.keys()):
        df = data[sym]
        ts = df.index[-1]
        row = df.iloc[-1]
        state = management_states.get(sym, {})
        trade = close_position(
            positions[sym],
            SYMBOL_SPECS[sym],
            ts,
            row,
            float(row["Close"]),
            "end_of_test",
            args.commission_per_contract_side,
            False,
            state,
        )
        record_trade_and_update(trade)
        del positions[sym]
        management_states.pop(sym, None)

    close_session_if_needed(None)

    trades_df = pd.DataFrame(trades)
    rejected_df = pd.DataFrame(rejected)
    daily_df = pd.DataFrame(daily_rows)

    prefix = args.output_prefix or args.strategy
    out_trade = OUT_DIR / f"{prefix}_event_trade_log.csv"
    out_reject = OUT_DIR / f"{prefix}_event_rejected_signals.csv"
    out_daily = OUT_DIR / f"{prefix}_event_daily_summary.csv"

    trades_df.to_csv(out_trade, index=False)
    rejected_df.to_csv(out_reject, index=False)
    daily_df.to_csv(out_daily, index=False)

    gross = trades_df["gross_pnl_dollars"].sum() if not trades_df.empty else 0.0
    net = trades_df["net_pnl_dollars"].sum() if not trades_df.empty else 0.0
    comm = trades_df["commissions_dollars"].sum() if not trades_df.empty else 0.0

    print("\n================ EVENT ENGINE V2 FINAL REPORT ================")
    print(f"Strategy:                  {adapter.name}")
    print(f"Profile:                   {profile.name}")
    print(f"Risk per trade:             ${profile.risk_per_trade:,.2f}")
    print(f"Daily profit target:        {profile.daily_profit_target}")
    print(f"Daily soft loss stop:       {profile.daily_soft_loss_stop}")
    print(f"Unrealized daily lock:      {unrealized_daily_lock_enabled}")
    print(f"Account drawdown lock:      {account_drawdown_lock_enabled}")
    print(f"Breakeven enabled:          {args.enable_breakeven} @ {args.breakeven_trigger_r}R")
    print(f"Partials enabled:           {args.enable_partials} @ {args.partial_trigger_r}R close {args.partial_close_pct:.2%}")
    print(f"Runner trail enabled:       {args.enable_runner_trail} lookback={args.runner_trail_lookback}")
    print(f"Breakeven activations:      {breakeven_activation_count}")
    print(f"Partial exits:              {partial_exit_count}")
    print(f"Runner trail updates:       {runner_trail_update_count}")
    print(f"Trades:                    {len(trades_df)}")
    print(f"Gross PnL:                 ${gross:,.2f}")
    print(f"Commissions:               ${comm:,.2f}")
    print(f"Net PnL:                   ${net:,.2f}")
    print(f"Final balance:             ${profile.account_size + net:,.2f}")
    print(f"Rejected signals:          {len(rejected_df)}")

    if not trades_df.empty:
        by_symbol = (
            trades_df.groupby("symbol")
            .agg(
                trades=("net_pnl_dollars", "size"),
                net_pnl_dollars=("net_pnl_dollars", "sum"),
                avg_trade=("net_pnl_dollars", "mean"),
                median_trade=("net_pnl_dollars", "median"),
                win_rate_pct=("net_pnl_dollars", lambda s: (s > 0).mean() * 100),
                worst_trade=("net_pnl_dollars", "min"),
                best_trade=("net_pnl_dollars", "max"),
            )
            .reset_index()
            .sort_values("net_pnl_dollars", ascending=False)
        )
        print("\nBy symbol:")
        print(by_symbol.to_string(index=False))

    if not daily_df.empty:
        print("\nDaily lock counts:")
        print(daily_df["day_lock_reason"].fillna("").replace("", "none").value_counts().to_string())

    print("\nWrote files:")
    print(f"  {out_trade}")
    print(f"  {out_reject}")
    print(f"  {out_daily}")


if __name__ == "__main__":
    main()
