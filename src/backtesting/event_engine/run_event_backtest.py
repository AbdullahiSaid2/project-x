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
        from src.strategies.adapters.top_bottom_ticking_event_adapter import TopBottomTickingAdapter
        return TopBottomTickingAdapter()
    if name == "ict_fractal":
        from src.strategies.adapters.ict_fractal_event_adapter import ICTFractalAdapter
        return ICTFractalAdapter()
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


def total_open_unrealized(positions: Dict[str, Position], data: dict, ts) -> float:
    total = 0.0
    for sym, pos in positions.items():
        if ts not in data[sym].index:
            continue
        mark = float(data[sym].loc[ts]["Close"])
        total += position_unrealized(pos, SYMBOL_SPECS[sym], mark)
    return total


def close_position(pos: Position, spec, ts, row, exit_price: float, exit_reason: str, commission_per_contract_side: float, same_bar: bool):
    if pos.side == "LONG":
        points = exit_price - pos.entry_price
    else:
        points = pos.entry_price - exit_price

    gross = points * spec.dollars_per_point * pos.size
    commissions = commission_per_contract_side * 2.0 * pos.size
    net = gross - commissions

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
    }


def flatten_all(positions: Dict[str, Position], data: dict, ts, reason: str, commission_per_contract_side: float):
    closed = []
    for sym in list(positions.keys()):
        if ts not in data[sym].index:
            continue
        row = data[sym].loc[ts]
        trade = close_position(
            positions[sym],
            SYMBOL_SPECS[sym],
            ts,
            row,
            float(row["Close"]),
            reason,
            commission_per_contract_side,
            False,
        )
        closed.append(trade)
        del positions[sym]
    return closed


def maybe_exit_position(pos: Position, spec, ts, bar_index: int, row, commission_per_contract_side: float):
    # No same-bar exits to avoid candle-order ambiguity.
    if bar_index <= pos.entry_bar_index:
        return None

    high = float(row["High"])
    low = float(row["Low"])

    if pos.side == "LONG":
        # Conservative: if both target and stop are touched, stop wins.
        if low <= pos.stop_price:
            return close_position(pos, spec, ts, row, pos.stop_price, "stop_loss", commission_per_contract_side, False)
        if high >= pos.target_price:
            return close_position(pos, spec, ts, row, pos.target_price, "take_profit", commission_per_contract_side, False)

    if pos.side == "SHORT":
        if high >= pos.stop_price:
            return close_position(pos, spec, ts, row, pos.stop_price, "stop_loss", commission_per_contract_side, False)
        if low <= pos.target_price:
            return close_position(pos, spec, ts, row, pos.target_price, "take_profit", commission_per_contract_side, False)

    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True, choices=["top_bottom_ticking", "ict_fractal"])
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
        df = adapter.build_features(sym, df)
        data[sym] = df
        bar_index_maps[sym] = {ts: i for i, ts in enumerate(df.index)}
        print(f"{sym}: rows={len(df)} start={df.index.min()} end={df.index.max()}")

    all_timestamps = sorted(set().union(*[set(df.index) for df in data.values()]))
    print(f"\nRunning {adapter.name} event engine v2 over {len(all_timestamps)} timestamps...")

    positions: Dict[str, Position] = {}
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
            for trade in flatten_all(positions, data, ts, reason, args.commission_per_contract_side):
                record_trade_and_update(trade)

        # Normal exits.
        for sym in list(positions.keys()):
            if ts not in data[sym].index:
                continue
            row = data[sym].loc[ts]
            trade = maybe_exit_position(
                positions[sym],
                SYMBOL_SPECS[sym],
                ts,
                bar_index_maps[sym][ts],
                row,
                args.commission_per_contract_side,
            )
            if trade:
                record_trade_and_update(trade)
                del positions[sym]

        # Realised + unrealised daily locks.
        open_unreal = total_open_unrealized(positions, data, ts) if positions else 0.0
        daily_equity_pnl = daily_net_pnl + open_unreal

        if unrealized_daily_lock_enabled and positions:
            if profile.daily_profit_target is not None and daily_equity_pnl >= profile.daily_profit_target:
                for trade in flatten_all(positions, data, ts, "daily_equity_profit_lock", args.commission_per_contract_side):
                    record_trade_and_update(trade)
                day_locked = True
                day_lock_reason = "daily_equity_profit_lock"

            elif profile.daily_soft_loss_stop is not None and daily_equity_pnl <= -abs(profile.daily_soft_loss_stop):
                for trade in flatten_all(positions, data, ts, "daily_equity_soft_loss_lock", args.commission_per_contract_side):
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
            daily_trades += 1

    # End-of-test close.
    for sym in list(positions.keys()):
        df = data[sym]
        ts = df.index[-1]
        row = df.iloc[-1]
        trade = close_position(
            positions[sym],
            SYMBOL_SPECS[sym],
            ts,
            row,
            float(row["Close"]),
            "end_of_test",
            args.commission_per_contract_side,
            False,
        )
        record_trade_and_update(trade)
        del positions[sym]

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
