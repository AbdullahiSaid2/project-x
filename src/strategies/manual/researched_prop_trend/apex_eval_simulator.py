from __future__ import annotations

"""
Apex-style prop-firm evaluation simulator.

This reads event_trade_log.csv from the event-driven backtest and answers:

- Did the strategy pass the 50K eval?
- Did it fail max drawdown before hitting the profit target? DLL pauses the day and is not treated as failure.
- How many calendar/trading days to pass?
- What was the worst drawdown before pass/fail?
- What happens if we split the 5-year trade log into rolling/sequential eval attempts?

Important:
This simulator uses realised net PnL from the event engine.

Default profile:
  account_size = 50,000
  profit_target = 3,000
  max_drawdown = 2,000
  drawdown_type = eod

For EOD drawdown:
  The trailing drawdown is updated from end-of-day balance peaks.
  The fail threshold is peak_eod_balance - max_drawdown.
  This approximates an EOD trailing drawdown eval.

For static drawdown:
  fail threshold is account_size - max_drawdown.

For intraday trailing:
  The trailing drawdown updates from every trade-level balance peak.

Run:

PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.apex_eval_simulator \
  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv \
  --prop-profile apex_50k_eod_eval

Rolling/sequential attempts:

PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.apex_eval_simulator \
  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv \
  --prop-profile apex_50k_eod_eval \
  --mode sequential \
  --max-calendar-days 30
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import yaml
except Exception as exc:
    raise RuntimeError("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

ROOT = Path(__file__).resolve().parent

OUT_SINGLE = ROOT / "apex_eval_single_result.csv"
OUT_ATTEMPTS = ROOT / "apex_eval_attempts.csv"
OUT_DAILY = ROOT / "apex_eval_daily_curve.csv"
OUT_SUMMARY = ROOT / "apex_eval_summary.txt"


@dataclass(frozen=True)
class EvalProfile:
    name: str
    account_size: float
    profit_target: float
    max_drawdown: float
    drawdown_type: str
    daily_loss_limit: Optional[float]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_profile(name: str, path: Path = ROOT / "prop_profiles.yaml") -> EvalProfile:
    raw = load_yaml(path)
    if name not in raw:
        raise ValueError(f"Unknown profile {name}. Available: {list(raw)}")
    cfg = raw[name]
    return EvalProfile(
        name=name,
        account_size=float(cfg.get("account_size", 50000)),
        profit_target=float(cfg.get("profit_target", 3000)) if cfg.get("profit_target") is not None else 0.0,
        max_drawdown=float(cfg.get("max_drawdown", 2000)),
        drawdown_type=str(cfg.get("drawdown_type", "eod")),
        daily_loss_limit=float(cfg["daily_loss_limit"]) if cfg.get("daily_loss_limit") is not None else None,
    )


def _to_et_naive(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)


def prepare_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df

    if "net_pnl_dollars" not in df.columns:
        if "gross_pnl_dollars" not in df.columns:
            raise ValueError("Trade log must contain net_pnl_dollars or gross_pnl_dollars")
        df["net_pnl_dollars"] = df["gross_pnl_dollars"]

    if "exit_time_et" in df.columns:
        # Handles ISO strings with offset.
        df["exit_dt_et"] = pd.to_datetime(df["exit_time_et"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    elif "exit_time" in df.columns:
        df["exit_dt_et"] = _to_et_naive(df["exit_time"])
    else:
        raise ValueError("Trade log must have exit_time_et or exit_time")

    if "entry_time_et" in df.columns:
        df["entry_dt_et"] = pd.to_datetime(df["entry_time_et"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    elif "entry_time" in df.columns:
        df["entry_dt_et"] = _to_et_naive(df["entry_time"])
    else:
        df["entry_dt_et"] = df["exit_dt_et"]

    df = df.dropna(subset=["exit_dt_et"]).sort_values(["exit_dt_et", "symbol" if "symbol" in df.columns else "net_pnl_dollars"]).reset_index(drop=True)
    df["exit_date_et"] = df["exit_dt_et"].dt.date
    df["exit_month_et"] = df["exit_dt_et"].dt.to_period("M").astype(str)
    return df


def simulate_attempt(
    trades: pd.DataFrame,
    profile: EvalProfile,
    attempt_id: int = 1,
    max_calendar_days: Optional[int] = None,
    min_trading_days: int = 0,
    start_index: int = 0,
) -> Dict[str, Any]:
    """
    Simulates one eval attempt from trades[start_index:].

    The attempt ends when:
      - profit target hit
      - drawdown breached
      - max_calendar_days elapsed
      - no trades left
    """
    if trades.empty or start_index >= len(trades):
        return {
            "attempt_id": attempt_id,
            "status": "NO_TRADES",
            "start_index": start_index,
            "end_index": start_index,
            "trades": 0,
            "net_pnl": 0.0,
        }

    balance = profile.account_size
    start_balance = profile.account_size
    trade_peak_balance = balance
    eod_peak_balance = balance
    static_floor = profile.account_size - profile.max_drawdown
    trailing_floor = profile.account_size - profile.max_drawdown

    start_time = trades.iloc[start_index]["exit_dt_et"]
    start_date = start_time.date()
    end_index = start_index

    daily_records: List[Dict[str, Any]] = []
    current_day = None
    day_pnl = 0.0
    trading_days = set()
    dll_hit_days = set()

    max_equity = balance
    min_equity = balance
    worst_drawdown = 0.0
    max_eod_drawdown = 0.0
    status = "OPEN"
    fail_reason = ""
    pass_time = pd.NaT
    fail_time = pd.NaT

    trade_count = 0
    winning_trades = 0
    losing_trades = 0
    best_trade = float("-inf")
    worst_trade = float("inf")
    gross_profit = 0.0
    gross_loss = 0.0

    def close_day(day, day_pnl_value, balance_value, floor_value):
        nonlocal eod_peak_balance, max_eod_drawdown
        if day is None:
            return
        eod_peak_balance = max(eod_peak_balance, balance_value)
        eod_dd = balance_value - eod_peak_balance
        max_eod_drawdown = min(max_eod_drawdown, eod_dd)
        daily_records.append({
            "attempt_id": attempt_id,
            "date": day,
            "day_pnl": day_pnl_value,
            "balance": balance_value,
            "eod_peak_balance": eod_peak_balance,
            "trailing_floor": floor_value,
            "eod_drawdown": eod_dd,
        })

    for i in range(start_index, len(trades)):
        row = trades.iloc[i]
        t = row["exit_dt_et"]
        day = row["exit_date_et"]

        if max_calendar_days is not None:
            if (pd.Timestamp(day) - pd.Timestamp(start_date)).days >= max_calendar_days:
                status = "TIME_EXPIRED"
                end_index = i
                break

        if current_day is None:
            current_day = day

        if day != current_day:
            close_day(current_day, day_pnl, balance, trailing_floor)
            current_day = day
            day_pnl = 0.0

            # For EOD trailing, update floor after prior day close.
            if profile.drawdown_type.lower() == "eod":
                trailing_floor = max(trailing_floor, eod_peak_balance - profile.max_drawdown)

        pnl = float(row["net_pnl_dollars"])
        balance += pnl
        day_pnl += pnl
        trading_days.add(day)
        trade_count += 1
        end_index = i

        best_trade = max(best_trade, pnl)
        worst_trade = min(worst_trade, pnl)
        if pnl > 0:
            winning_trades += 1
            gross_profit += pnl
        elif pnl < 0:
            losing_trades += 1
            gross_loss += pnl

        trade_peak_balance = max(trade_peak_balance, balance)
        max_equity = max(max_equity, balance)
        min_equity = min(min_equity, balance)
        worst_drawdown = min(worst_drawdown, balance - max_equity)

        if profile.drawdown_type.lower() in ("intraday", "intraday_trailing", "trailing"):
            trailing_floor = max(trailing_floor, trade_peak_balance - profile.max_drawdown)
        elif profile.drawdown_type.lower() in ("static", "fixed"):
            trailing_floor = static_floor
        else:
            # EOD trailing floor updates at day boundary, but start with account - drawdown.
            trailing_floor = max(trailing_floor, profile.account_size - profile.max_drawdown)

        if balance <= trailing_floor:
            status = "FAILED"
            fail_reason = f"drawdown_breach_{profile.drawdown_type}"
            fail_time = t
            break

        # DLL / Daily Loss Limit:
        # Apex EOD eval DLL does NOT fail the account. It liquidates/pauses the
        # account for the remainder of that trading day, then trading resumes at
        # the next 18:00 ET session reset. The event engine should already stop
        # entering after DLL. In this simulator we record the breach but do not
        # mark the evaluation as failed.
        if profile.daily_loss_limit is not None and day_pnl <= -abs(profile.daily_loss_limit):
            dll_hit_days.add(day)

        if balance >= start_balance + profile.profit_target and len(trading_days) >= min_trading_days:
            status = "PASSED"
            pass_time = t
            break

    if status == "OPEN":
        status = "INCOMPLETE"

    close_day(current_day, day_pnl, balance, trailing_floor)

    net_pnl = balance - start_balance
    days_elapsed = (pd.Timestamp(trades.iloc[end_index]["exit_dt_et"]).date() - pd.Timestamp(start_date).date()).days if end_index >= start_index else 0

    return {
        "attempt_id": attempt_id,
        "status": status,
        "fail_reason": fail_reason,
        "start_index": start_index,
        "end_index": end_index,
        "start_time_et": start_time,
        "end_time_et": trades.iloc[end_index]["exit_dt_et"] if end_index < len(trades) else pd.NaT,
        "pass_time_et": pass_time,
        "fail_time_et": fail_time,
        "calendar_days": days_elapsed,
        "trading_days": len(trading_days),
        "trades": trade_count,
        "wins": winning_trades,
        "losses": losing_trades,
        "win_rate_pct": (winning_trades / trade_count * 100) if trade_count else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": (gross_profit / abs(gross_loss)) if gross_loss < 0 else float("inf"),
        "net_pnl": net_pnl,
        "final_balance": balance,
        "max_equity": max_equity,
        "min_equity": min_equity,
        "worst_trade": worst_trade if trade_count else 0.0,
        "best_trade": best_trade if trade_count else 0.0,
        "worst_intraday_drawdown": worst_drawdown,
        "worst_eod_drawdown": max_eod_drawdown,
        "trailing_floor": trailing_floor,
        "dll_hit_days": len(dll_hit_days),
        "dll_hit_day_list": ",".join(str(x) for x in sorted(dll_hit_days)),
        "_daily_records": daily_records,
    }


def run_sequential_attempts(
    trades: pd.DataFrame,
    profile: EvalProfile,
    max_calendar_days: Optional[int],
    min_trading_days: int,
    max_attempts: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts = []
    all_daily = []
    start_index = 0
    attempt_id = 1

    while start_index < len(trades) and attempt_id <= max_attempts:
        result = simulate_attempt(
            trades,
            profile,
            attempt_id=attempt_id,
            max_calendar_days=max_calendar_days,
            min_trading_days=min_trading_days,
            start_index=start_index,
        )
        daily = result.pop("_daily_records", [])
        attempts.append(result)
        all_daily.extend(daily)

        end_index = int(result.get("end_index", start_index))
        if end_index <= start_index:
            start_index += 1
        else:
            start_index = end_index + 1

        attempt_id += 1

    return pd.DataFrame(attempts), pd.DataFrame(all_daily)


def write_summary(attempts: pd.DataFrame, profile: EvalProfile, mode: str):
    lines = []
    lines.append("================ APEX EVAL SIM SUMMARY ================")
    lines.append(f"Profile: {profile.name}")
    lines.append(f"Account size: ${profile.account_size:,.2f}")
    lines.append(f"Profit target: ${profile.profit_target:,.2f}")
    lines.append(f"Max drawdown: ${profile.max_drawdown:,.2f}")
    lines.append(f"Drawdown type: {profile.drawdown_type}")
    lines.append(f"Mode: {mode}")
    lines.append("")

    if attempts.empty:
        lines.append("No attempts.")
    else:
        passed = attempts[attempts["status"] == "PASSED"]
        failed = attempts[attempts["status"] == "FAILED"]
        incomplete = attempts[~attempts["status"].isin(["PASSED", "FAILED"])]

        lines.append(f"Attempts: {len(attempts)}")
        lines.append(f"Passed: {len(passed)}")
        lines.append(f"Failed: {len(failed)}")
        lines.append(f"Incomplete/other: {len(incomplete)}")
        lines.append(f"Pass rate: {(len(passed) / len(attempts) * 100):.2f}%")
        lines.append("")

        if not passed.empty:
            lines.append(f"Avg days to pass: {passed['calendar_days'].mean():.2f}")
            lines.append(f"Median days to pass: {passed['calendar_days'].median():.2f}")
            lines.append(f"Avg trades to pass: {passed['trades'].mean():.2f}")
            lines.append(f"Median trades to pass: {passed['trades'].median():.2f}")
            lines.append(f"Avg PnL on pass: ${passed['net_pnl'].mean():,.2f}")
            lines.append(f"Worst intraday DD before pass: ${passed['worst_intraday_drawdown'].min():,.2f}")
            lines.append(f"Worst EOD DD before pass: ${passed['worst_eod_drawdown'].min():,.2f}")
            lines.append("")

        if "dll_hit_days" in attempts.columns:
            lines.append(f"Attempts with DLL pause days: {(attempts['dll_hit_days'] > 0).sum()}")
            lines.append(f"Total DLL pause days: {int(attempts['dll_hit_days'].sum())}")
            lines.append("")

        if not failed.empty:
            lines.append("Fail reasons:")
            lines.append(failed["fail_reason"].value_counts().to_string())
            lines.append(f"Avg days to fail: {failed['calendar_days'].mean():.2f}")
            lines.append(f"Avg PnL on fail: ${failed['net_pnl'].mean():,.2f}")

    text = "\n".join(lines)
    OUT_SUMMARY.write_text(text + "\n", encoding="utf-8")
    print(text)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trade-log", default=str(ROOT / "event_trade_log.csv"))
    p.add_argument("--prop-profile", default="apex_50k_eod_eval")
    p.add_argument("--mode", choices=["single", "sequential"], default="single")
    p.add_argument("--max-calendar-days", type=int, default=None)
    p.add_argument("--min-trading-days", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=9999)
    return p.parse_args()


def main():
    args = parse_args()
    profile = load_profile(args.prop_profile)
    trades = prepare_trades(Path(args.trade_log))

    if args.mode == "single":
        result = simulate_attempt(
            trades,
            profile,
            attempt_id=1,
            max_calendar_days=args.max_calendar_days,
            min_trading_days=args.min_trading_days,
            start_index=0,
        )
        daily = pd.DataFrame(result.pop("_daily_records", []))
        attempts = pd.DataFrame([result])
    else:
        attempts, daily = run_sequential_attempts(
            trades,
            profile,
            max_calendar_days=args.max_calendar_days,
            min_trading_days=args.min_trading_days,
            max_attempts=args.max_attempts,
        )

    attempts.to_csv(OUT_ATTEMPTS if args.mode == "sequential" else OUT_SINGLE, index=False)
    daily.to_csv(OUT_DAILY, index=False)
    write_summary(attempts, profile, args.mode)

    print("\nWrote files:")
    print(f"  {OUT_ATTEMPTS if args.mode == 'sequential' else OUT_SINGLE}")
    print(f"  {OUT_DAILY}")
    print(f"  {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
