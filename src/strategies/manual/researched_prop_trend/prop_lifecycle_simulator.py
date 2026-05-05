
from __future__ import annotations

"""
Prop-firm lifecycle simulator for Algotec.

This simulates the realistic journey:

  Eval attempt
    -> pass target?
      -> PA account starts
         -> PA trades until max loss/EOD drawdown is hit, or data ends
         -> if PA blown, return to Eval
    -> if eval fails max loss/EOD drawdown, start next Eval

Daily Loss Limit behaviour:
  - DLL / daily loss is NOT account failure.
  - It is recorded as a daily pause/lockout event.
  - The event engine should already stop new entries once DLL is reached.
  - This lifecycle simulator records DLL hit days but does not fail the account for DLL.

Max Loss / EOD Drawdown behaviour:
  - This is the account failure condition.
  - Eval failure if hit before profit target.
  - PA blown if hit after funded account starts.

Input:
  event_trade_log.csv from prop_event_engine_backtest.py

Recommended:
  Run PA/event engine with news blackout first, then run this lifecycle simulator.

Example:

PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.prop_lifecycle_simulator \
  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv \
  --eval-profile apex_50k_eod_eval \
  --pa-profile apex_50k_pa
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, List
import argparse

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent

OUT_CYCLES = ROOT / "prop_lifecycle_cycles.csv"
OUT_EVENTS = ROOT / "prop_lifecycle_events.csv"
OUT_DAILY = ROOT / "prop_lifecycle_daily.csv"
OUT_SUMMARY = ROOT / "prop_lifecycle_summary.txt"


@dataclass(frozen=True)
class Profile:
    name: str
    mode: str
    account_size: float
    profit_target: Optional[float]
    max_drawdown: float
    drawdown_type: str
    daily_loss_limit: Optional[float]


def load_profiles(path: Path = ROOT / "prop_profiles.yaml") -> dict[str, Profile]:
    raw = yaml.safe_load(path.read_text()) or {}
    out: dict[str, Profile] = {}
    for name, cfg in raw.items():
        out[name] = Profile(
            name=name,
            mode=str(cfg.get("mode", "eval")),
            account_size=float(cfg.get("account_size", 50000)),
            profit_target=float(cfg["profit_target"]) if cfg.get("profit_target") is not None else None,
            max_drawdown=float(cfg.get("max_drawdown", 2000)),
            drawdown_type=str(cfg.get("drawdown_type", "eod")),
            daily_loss_limit=float(cfg["daily_loss_limit"]) if cfg.get("daily_loss_limit") is not None else None,
        )
    return out


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df

    if "net_pnl_dollars" not in df.columns:
        if "gross_pnl_dollars" not in df.columns:
            raise ValueError("Trade log needs net_pnl_dollars or gross_pnl_dollars")
        df["net_pnl_dollars"] = df["gross_pnl_dollars"]

    if "exit_time_et" in df.columns:
        df["exit_dt_et"] = pd.to_datetime(df["exit_time_et"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    elif "exit_time" in df.columns:
        df["exit_dt_et"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    else:
        raise ValueError("Trade log needs exit_time_et or exit_time")

    if "entry_time_et" in df.columns:
        df["entry_dt_et"] = pd.to_datetime(df["entry_time_et"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    elif "entry_time" in df.columns:
        df["entry_dt_et"] = pd.to_datetime(df["entry_time"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    else:
        df["entry_dt_et"] = df["exit_dt_et"]

    df = df.dropna(subset=["exit_dt_et"]).sort_values(["exit_dt_et", "symbol" if "symbol" in df.columns else "net_pnl_dollars"]).reset_index(drop=True)
    df["exit_date_et"] = df["exit_dt_et"].dt.date
    df["exit_month_et"] = df["exit_dt_et"].dt.to_period("M").astype(str)
    return df


def simulate_phase(
    trades: pd.DataFrame,
    profile: Profile,
    start_index: int,
    cycle_id: int,
    phase: str,
    max_calendar_days: Optional[int] = None,
    min_trading_days: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int]:
    """
    Simulates one phase:
      - eval: ends at PASS, FAIL, TIME_EXPIRED, or DATA_END
      - pa: ends at PA_BLOWN or DATA_END

    Returns:
      result, events, daily_records, next_index
    """
    if start_index >= len(trades):
        return {
            "cycle_id": cycle_id,
            "phase": phase,
            "status": "NO_TRADES",
            "start_index": start_index,
            "end_index": start_index,
            "net_pnl": 0.0,
            "trades": 0,
        }, [], [], start_index + 1

    start_balance = profile.account_size
    balance = start_balance
    peak_trade_balance = balance
    eod_peak_balance = balance
    floor = start_balance - profile.max_drawdown

    start_time = trades.iloc[start_index]["exit_dt_et"]
    start_date = trades.iloc[start_index]["exit_date_et"]
    current_day = None
    day_pnl = 0.0
    trading_days = set()
    dll_hit_days = set()

    max_balance = balance
    min_balance = balance
    worst_intraday_dd = 0.0
    worst_eod_dd = 0.0

    events: list[dict[str, Any]] = []
    daily_records: list[dict[str, Any]] = []
    status = "OPEN"
    reason = ""
    end_index = start_index - 1

    trades_count = 0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    best_trade = float("-inf")
    worst_trade = float("inf")

    def close_day(day):
        nonlocal eod_peak_balance, floor, worst_eod_dd, day_pnl, balance
        if day is None:
            return

        eod_peak_balance = max(eod_peak_balance, balance)
        if profile.drawdown_type.lower() == "eod":
            floor = max(floor, eod_peak_balance - profile.max_drawdown)

        eod_dd = balance - eod_peak_balance
        worst_eod_dd = min(worst_eod_dd, eod_dd)

        daily_records.append({
            "cycle_id": cycle_id,
            "phase": phase,
            "date": str(day),
            "day_pnl": day_pnl,
            "balance": balance,
            "eod_peak_balance": eod_peak_balance,
            "drawdown_floor": floor,
            "eod_drawdown": eod_dd,
            "dll_hit": day in dll_hit_days,
        })

    for i in range(start_index, len(trades)):
        row = trades.iloc[i]
        t = row["exit_dt_et"]
        day = row["exit_date_et"]

        if max_calendar_days is not None:
            if (pd.Timestamp(day) - pd.Timestamp(start_date)).days >= max_calendar_days:
                status = "TIME_EXPIRED"
                reason = "max_calendar_days"
                end_index = max(i - 1, start_index)
                break

        if current_day is None:
            current_day = day

        if day != current_day:
            close_day(current_day)
            current_day = day
            day_pnl = 0.0

        pnl = float(row["net_pnl_dollars"])
        balance += pnl
        day_pnl += pnl
        trading_days.add(day)
        trades_count += 1
        end_index = i

        best_trade = max(best_trade, pnl)
        worst_trade = min(worst_trade, pnl)
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        elif pnl < 0:
            losses += 1
            gross_loss += pnl

        max_balance = max(max_balance, balance)
        min_balance = min(min_balance, balance)
        peak_trade_balance = max(peak_trade_balance, balance)
        worst_intraday_dd = min(worst_intraday_dd, balance - max_balance)

        if profile.drawdown_type.lower() in ("intraday", "trailing", "intraday_trailing"):
            floor = max(floor, peak_trade_balance - profile.max_drawdown)

        if profile.daily_loss_limit is not None and day_pnl <= -abs(profile.daily_loss_limit):
            if day not in dll_hit_days:
                dll_hit_days.add(day)
                events.append({
                    "cycle_id": cycle_id,
                    "phase": phase,
                    "event_type": "DLL_PAUSE",
                    "time_et": t,
                    "trade_index": i,
                    "day_pnl": day_pnl,
                    "balance": balance,
                    "floor": floor,
                    "symbol": row.get("symbol", ""),
                    "trade_type": row.get("trade_type", ""),
                })

        # Max Loss / Drawdown breach = failure/blown.
        if balance <= floor:
            if phase == "eval":
                status = "EVAL_FAILED"
                reason = f"max_loss_breach_{profile.drawdown_type}"
                events.append({
                    "cycle_id": cycle_id,
                    "phase": phase,
                    "event_type": "EVAL_FAILED_MAX_LOSS",
                    "time_et": t,
                    "trade_index": i,
                    "day_pnl": day_pnl,
                    "balance": balance,
                    "floor": floor,
                    "symbol": row.get("symbol", ""),
                    "trade_type": row.get("trade_type", ""),
                })
            else:
                status = "PA_BLOWN"
                reason = f"max_loss_breach_{profile.drawdown_type}"
                events.append({
                    "cycle_id": cycle_id,
                    "phase": phase,
                    "event_type": "PA_BLOWN_MAX_LOSS",
                    "time_et": t,
                    "trade_index": i,
                    "day_pnl": day_pnl,
                    "balance": balance,
                    "floor": floor,
                    "symbol": row.get("symbol", ""),
                    "trade_type": row.get("trade_type", ""),
                })
            break

        # Eval pass target.
        if phase == "eval" and profile.profit_target is not None:
            if balance >= start_balance + profile.profit_target and len(trading_days) >= min_trading_days:
                status = "EVAL_PASSED"
                reason = "profit_target_hit"
                events.append({
                    "cycle_id": cycle_id,
                    "phase": phase,
                    "event_type": "EVAL_PASSED",
                    "time_et": t,
                    "trade_index": i,
                    "day_pnl": day_pnl,
                    "balance": balance,
                    "floor": floor,
                    "symbol": row.get("symbol", ""),
                    "trade_type": row.get("trade_type", ""),
                })
                break

    if current_day is not None:
        close_day(current_day)

    if status == "OPEN":
        status = "DATA_END"
        reason = "no_more_trades"
        end_index = len(trades) - 1

    end_time = trades.iloc[end_index]["exit_dt_et"] if 0 <= end_index < len(trades) else pd.NaT
    next_index = end_index + 1

    result = {
        "cycle_id": cycle_id,
        "phase": phase,
        "profile": profile.name,
        "status": status,
        "reason": reason,
        "start_index": start_index,
        "end_index": end_index,
        "start_time_et": start_time,
        "end_time_et": end_time,
        "calendar_days": (pd.Timestamp(end_time).date() - pd.Timestamp(start_date).date()).days if pd.notna(end_time) else 0,
        "trading_days": len(trading_days),
        "trades": trades_count,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": (wins / trades_count * 100.0) if trades_count else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": (gross_profit / abs(gross_loss)) if gross_loss < 0 else float("inf"),
        "net_pnl": balance - start_balance,
        "start_balance": start_balance,
        "final_balance": balance,
        "max_balance": max_balance,
        "min_balance": min_balance,
        "best_trade": best_trade if trades_count else 0.0,
        "worst_trade": worst_trade if trades_count else 0.0,
        "worst_intraday_drawdown": worst_intraday_dd,
        "worst_eod_drawdown": worst_eod_dd,
        "dll_hit_days": len(dll_hit_days),
        "dll_hit_day_list": ",".join(str(x) for x in sorted(dll_hit_days)),
        "next_index": next_index,
    }
    return result, events, daily_records, next_index


def run_lifecycle(
    trades: pd.DataFrame,
    eval_profile: Profile,
    pa_profile: Profile,
    max_cycles: int,
    eval_max_calendar_days: Optional[int],
    eval_min_trading_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = 0
    cycle_id = 1
    cycle_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []

    while idx < len(trades) and cycle_id <= max_cycles:
        eval_result, eval_events, eval_daily, next_idx = simulate_phase(
            trades,
            eval_profile,
            start_index=idx,
            cycle_id=cycle_id,
            phase="eval",
            max_calendar_days=eval_max_calendar_days,
            min_trading_days=eval_min_trading_days,
        )
        cycle_rows.append(eval_result)
        events.extend(eval_events)
        daily.extend(eval_daily)

        if eval_result["status"] == "EVAL_PASSED":
            pa_result, pa_events, pa_daily, pa_next_idx = simulate_phase(
                trades,
                pa_profile,
                start_index=next_idx,
                cycle_id=cycle_id,
                phase="pa",
                max_calendar_days=None,
                min_trading_days=0,
            )
            cycle_rows.append(pa_result)
            events.extend(pa_events)
            daily.extend(pa_daily)
            idx = pa_next_idx
        else:
            idx = next_idx

        if eval_result["status"] == "DATA_END":
            break

        cycle_id += 1

    return pd.DataFrame(cycle_rows), pd.DataFrame(events), pd.DataFrame(daily)


def write_summary(cycles: pd.DataFrame, events: pd.DataFrame):
    lines = []
    lines.append("================ PROP LIFECYCLE SUMMARY ================")
    if cycles.empty:
        lines.append("No cycles.")
    else:
        evals = cycles[cycles["phase"] == "eval"]
        pas = cycles[cycles["phase"] == "pa"]

        eval_passed = evals[evals["status"] == "EVAL_PASSED"]
        eval_failed = evals[evals["status"] == "EVAL_FAILED"]
        pa_blown = pas[pas["status"] == "PA_BLOWN"]
        pa_ended = pas[pas["status"] == "DATA_END"]

        lines.append(f"Eval attempts: {len(evals)}")
        lines.append(f"Eval passed: {len(eval_passed)}")
        lines.append(f"Eval failed max loss: {len(eval_failed)}")
        lines.append(f"PA accounts started: {len(pas)}")
        lines.append(f"PA blown: {len(pa_blown)}")
        lines.append(f"PA active/data-ended: {len(pa_ended)}")
        lines.append("")

        if not eval_passed.empty:
            lines.append(f"Eval pass rate: {len(eval_passed) / len(evals) * 100:.2f}%")
            lines.append(f"Avg eval days to pass: {eval_passed['calendar_days'].mean():.2f}")
            lines.append(f"Median eval days to pass: {eval_passed['calendar_days'].median():.2f}")
            lines.append(f"Avg eval trades to pass: {eval_passed['trades'].mean():.2f}")
            lines.append(f"Avg eval PnL at pass: ${eval_passed['net_pnl'].mean():,.2f}")
            lines.append("")

        if not pas.empty:
            lines.append(f"Total PA net PnL before PA stops/data end: ${pas['net_pnl'].sum():,.2f}")
            lines.append(f"Avg PA net PnL: ${pas['net_pnl'].mean():,.2f}")
            lines.append(f"Median PA net PnL: ${pas['net_pnl'].median():,.2f}")
            lines.append(f"Best PA net PnL: ${pas['net_pnl'].max():,.2f}")
            lines.append(f"Worst PA net PnL: ${pas['net_pnl'].min():,.2f}")
            lines.append(f"Avg PA days survived: {pas['calendar_days'].mean():.2f}")
            lines.append(f"Median PA days survived: {pas['calendar_days'].median():.2f}")
            lines.append(f"Avg PA trades: {pas['trades'].mean():.2f}")
            lines.append("")

        if not events.empty:
            lines.append("Lifecycle events:")
            lines.append(events["event_type"].value_counts().to_string())

    text = "\n".join(lines)
    OUT_SUMMARY.write_text(text + "\n", encoding="utf-8")
    print(text)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trade-log", default=str(ROOT / "event_trade_log.csv"))
    p.add_argument("--eval-profile", default="apex_50k_eod_eval")
    p.add_argument("--pa-profile", default="apex_50k_pa")
    p.add_argument("--max-cycles", type=int, default=9999)
    p.add_argument("--eval-max-calendar-days", type=int, default=None)
    p.add_argument("--eval-min-trading-days", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    profiles = load_profiles()
    eval_profile = profiles[args.eval_profile]
    pa_profile = profiles[args.pa_profile]
    trades = load_trades(Path(args.trade_log))

    cycles, events, daily = run_lifecycle(
        trades,
        eval_profile,
        pa_profile,
        max_cycles=args.max_cycles,
        eval_max_calendar_days=args.eval_max_calendar_days,
        eval_min_trading_days=args.eval_min_trading_days,
    )

    cycles.to_csv(OUT_CYCLES, index=False)
    events.to_csv(OUT_EVENTS, index=False)
    daily.to_csv(OUT_DAILY, index=False)
    write_summary(cycles, events)

    print("\nWrote files:")
    print(f"  {OUT_CYCLES}")
    print(f"  {OUT_EVENTS}")
    print(f"  {OUT_DAILY}")
    print(f"  {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
