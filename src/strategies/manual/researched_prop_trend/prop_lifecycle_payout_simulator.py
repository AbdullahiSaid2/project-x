
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import argparse
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent

OUT_SUMMARY = ROOT / "prop_lifecycle_payout_summary.txt"
OUT_CYCLES = ROOT / "prop_lifecycle_payout_cycles.csv"
OUT_EVENTS = ROOT / "prop_lifecycle_payout_events.csv"
OUT_DAILY = ROOT / "prop_lifecycle_payout_daily.csv"
OUT_PAYOUTS = ROOT / "prop_lifecycle_payouts.csv"


@dataclass(frozen=True)
class LifecycleProfile:
    name: str
    prop_firm: str
    account_size: float

    eval_profit_target: float
    eval_max_drawdown: float
    eval_daily_loss_limit: Optional[float]
    eval_max_calendar_days: Optional[int]
    eval_min_trading_days: int

    pa_max_drawdown: float
    pa_daily_loss_limit: Optional[float]
    pa_drawdown_stop_level: Optional[float]
    pa_safety_net_balance: float
    pa_min_balance_to_request: float

    pa_daily_profit_target: Optional[float]
    pa_daily_soft_loss_stop: Optional[float]
    pa_max_trades_per_day: Optional[int]
    pa_pause_after_consecutive_losses: Optional[int]
    pa_post_payout_buffer: float

    min_qualifying_days: int
    min_qualifying_day_profit: float
    min_payout_amount: float
    max_payouts_per_pa: int
    payout_caps: list[float]
    consistency_rule_pct: float
    close_pa_after_max_payouts: bool


def _optional_float(cfg, key):
    return float(cfg[key]) if cfg.get(key) is not None else None


def _optional_int(cfg, key):
    return int(cfg[key]) if cfg.get(key) is not None else None


def load_lifecycle_profiles(path: Path = ROOT / "prop_firm_lifecycle_profiles.yaml") -> dict[str, LifecycleProfile]:
    raw = yaml.safe_load(path.read_text()) or {}
    out = {}
    for name, cfg in raw.items():
        out[name] = LifecycleProfile(
            name=name,
            prop_firm=str(cfg.get("prop_firm", "apex")),
            account_size=float(cfg.get("account_size", 50000)),

            eval_profit_target=float(cfg.get("eval_profit_target", 3000)),
            eval_max_drawdown=float(cfg.get("eval_max_drawdown", 2000)),
            eval_daily_loss_limit=_optional_float(cfg, "eval_daily_loss_limit"),
            eval_max_calendar_days=_optional_int(cfg, "eval_max_calendar_days"),
            eval_min_trading_days=int(cfg.get("eval_min_trading_days", 0)),

            pa_max_drawdown=float(cfg.get("pa_max_drawdown", 2000)),
            pa_daily_loss_limit=_optional_float(cfg, "pa_daily_loss_limit"),
            pa_drawdown_stop_level=_optional_float(cfg, "pa_drawdown_stop_level"),
            pa_safety_net_balance=float(cfg.get("pa_safety_net_balance", 52100)),
            pa_min_balance_to_request=float(cfg.get("pa_min_balance_to_request", 52600)),

            pa_daily_profit_target=_optional_float(cfg, "pa_daily_profit_target"),
            pa_daily_soft_loss_stop=_optional_float(cfg, "pa_daily_soft_loss_stop"),
            pa_max_trades_per_day=_optional_int(cfg, "pa_max_trades_per_day"),
            pa_pause_after_consecutive_losses=_optional_int(cfg, "pa_pause_after_consecutive_losses"),
            pa_post_payout_buffer=float(cfg.get("pa_post_payout_buffer", 500)),

            min_qualifying_days=int(cfg.get("min_qualifying_days", 5)),
            min_qualifying_day_profit=float(cfg.get("min_qualifying_day_profit", 250)),
            min_payout_amount=float(cfg.get("min_payout_amount", 500)),
            max_payouts_per_pa=int(cfg.get("max_payouts_per_pa", 6)),
            payout_caps=[float(x) for x in cfg.get("payout_caps", [])],
            consistency_rule_pct=float(cfg.get("consistency_rule_pct", 50)),
            close_pa_after_max_payouts=bool(cfg.get("close_pa_after_max_payouts", True)),
        )
    return out


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "net_pnl_dollars" not in df.columns:
        df["net_pnl_dollars"] = df["gross_pnl_dollars"]
    if "exit_time_et" in df.columns:
        df["exit_dt_et"] = pd.to_datetime(df["exit_time_et"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    elif "exit_time" in df.columns:
        df["exit_dt_et"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    else:
        raise ValueError("Trade log needs exit_time_et or exit_time")
    df = df.dropna(subset=["exit_dt_et"]).sort_values(["exit_dt_et", "symbol" if "symbol" in df.columns else "net_pnl_dollars"]).reset_index(drop=True)
    df["exit_date_et"] = df["exit_dt_et"].dt.date
    df["exit_month_et"] = df["exit_dt_et"].dt.to_period("M").astype(str)
    return df


def update_eod_floor(floor: float, eod_peak: float, max_drawdown: float, stop_level: Optional[float]) -> float:
    proposed = eod_peak - max_drawdown
    if stop_level is not None:
        proposed = min(proposed, stop_level)
    return max(floor, proposed)


def qualifying_days(daily_pnls: dict[Any, float], min_profit: float) -> int:
    return sum(1 for v in daily_pnls.values() if v >= min_profit)


def consistency_status(daily_pnls: dict[Any, float], pct_limit: float):
    pos = [v for v in daily_pnls.values() if v > 0]
    total = sum(pos)
    high = max(pos) if pos else 0.0
    pct = (high / total * 100.0) if total > 0 else 0.0
    return total > 0 and pct < pct_limit, high, total, pct


def payout_cap(profile: LifecycleProfile, already_paid_count: int) -> float:
    if not profile.payout_caps:
        return float("inf")
    return profile.payout_caps[min(already_paid_count, len(profile.payout_caps) - 1)]


def maybe_payout(profile, cycle_id, t, balance, payout_count, daily_pnls, events, payouts):
    if payout_count >= profile.max_payouts_per_pa:
        return balance, payout_count, True, daily_pnls

    qdays = qualifying_days(daily_pnls, profile.min_qualifying_day_profit)
    ok_cons, high_day, cycle_profit, cons_pct = consistency_status(daily_pnls, profile.consistency_rule_pct)
    eligible_above_safety = balance - profile.pa_safety_net_balance - profile.pa_post_payout_buffer
    amount = min(eligible_above_safety, payout_cap(profile, payout_count))

    eligible = (
        qdays >= profile.min_qualifying_days
        and balance >= profile.pa_min_balance_to_request
        and ok_cons
        and amount >= profile.min_payout_amount
    )
    if not eligible:
        return balance, payout_count, False, daily_pnls

    new_count = payout_count + 1
    new_balance = balance - amount
    event = {
        "cycle_id": cycle_id, "phase": "pa", "event_type": "PAYOUT_APPROVED",
        "time_et": t, "payout_number": new_count, "payout_amount": amount,
        "balance_before_payout": balance, "balance_after_payout": new_balance,
        "qualifying_days": qdays, "cycle_profit": cycle_profit,
        "max_day_profit": high_day, "consistency_pct": cons_pct,
        "safety_net_balance": profile.pa_safety_net_balance,
        "post_payout_buffer": profile.pa_post_payout_buffer,
    }
    events.append(event); payouts.append(event.copy())
    completed = new_count >= profile.max_payouts_per_pa and profile.close_pa_after_max_payouts
    if completed:
        events.append({"cycle_id": cycle_id, "phase": "pa", "event_type": "PA_COMPLETED_MAX_PAYOUTS", "time_et": t, "balance": new_balance})
    return new_balance, new_count, completed, {}


def simulate_eval(trades, profile, start_idx, cycle_id):
    start_bal = profile.account_size
    balance = start_bal
    max_bal = start_bal
    eod_peak = start_bal
    floor = start_bal - profile.eval_max_drawdown
    events, daily = [], []
    start_time = trades.iloc[start_idx]["exit_dt_et"]
    start_date = trades.iloc[start_idx]["exit_date_et"]
    cur_day, day_pnl, locked = None, 0.0, False
    trading_days, dll_days = set(), set()
    trades_count = wins = losses = 0
    gp = gl = 0.0
    worst_dd = worst_eod_dd = 0.0
    status, reason = "OPEN", ""
    end_idx = start_idx - 1

    def close_day(day):
        nonlocal eod_peak, floor, worst_eod_dd
        if day is None: return
        eod_peak = max(eod_peak, balance)
        floor = update_eod_floor(floor, eod_peak, profile.eval_max_drawdown, None)
        eod_dd = balance - eod_peak
        worst_eod_dd = min(worst_eod_dd, eod_dd)
        daily.append({"cycle_id": cycle_id, "phase": "eval", "date": str(day), "day_pnl": day_pnl, "balance": balance, "drawdown_floor": floor, "eod_drawdown": eod_dd, "dll_hit": day in dll_days})

    for i in range(start_idx, len(trades)):
        row = trades.iloc[i]; t = row["exit_dt_et"]; day = row["exit_date_et"]
        if profile.eval_max_calendar_days is not None and (pd.Timestamp(day) - pd.Timestamp(start_date)).days >= profile.eval_max_calendar_days:
            status, reason, end_idx = "TIME_EXPIRED", "eval_max_calendar_days", max(i-1, start_idx); break
        if cur_day is None: cur_day = day
        if day != cur_day:
            close_day(cur_day); cur_day, day_pnl, locked = day, 0.0, False
        if locked: continue
        pnl = float(row["net_pnl_dollars"])
        balance += pnl; day_pnl += pnl; trading_days.add(day)
        trades_count += 1; end_idx = i
        if pnl > 0: wins += 1; gp += pnl
        elif pnl < 0: losses += 1; gl += pnl
        max_bal = max(max_bal, balance); worst_dd = min(worst_dd, balance - max_bal)
        if profile.eval_daily_loss_limit is not None and day_pnl <= -abs(profile.eval_daily_loss_limit):
            dll_days.add(day); locked = True
            events.append({"cycle_id": cycle_id, "phase": "eval", "event_type": "DLL_PAUSE", "time_et": t, "trade_index": i, "day_pnl": day_pnl, "balance": balance})
        if balance <= floor:
            status, reason = "EVAL_FAILED", "eval_max_loss_breach"
            events.append({"cycle_id": cycle_id, "phase": "eval", "event_type": "EVAL_FAILED_MAX_LOSS", "time_et": t, "trade_index": i, "balance": balance, "floor": floor})
            break
        if balance >= start_bal + profile.eval_profit_target and len(trading_days) >= profile.eval_min_trading_days:
            status, reason = "EVAL_PASSED", "profit_target_hit"
            events.append({"cycle_id": cycle_id, "phase": "eval", "event_type": "EVAL_PASSED", "time_et": t, "trade_index": i, "balance": balance, "floor": floor})
            break
    if cur_day is not None: close_day(cur_day)
    if status == "OPEN": status, reason, end_idx = "DATA_END", "no_more_trades", len(trades)-1
    end_time = trades.iloc[end_idx]["exit_dt_et"] if 0 <= end_idx < len(trades) else pd.NaT
    result = {"cycle_id": cycle_id, "phase": "eval", "status": status, "reason": reason, "start_index": start_idx, "end_index": end_idx, "next_index": end_idx+1, "start_time_et": start_time, "end_time_et": end_time, "calendar_days": (pd.Timestamp(end_time).date()-pd.Timestamp(start_date).date()).days if pd.notna(end_time) else 0, "trading_days": len(trading_days), "trades": trades_count, "wins": wins, "losses": losses, "win_rate_pct": wins/trades_count*100 if trades_count else 0, "gross_profit": gp, "gross_loss": gl, "profit_factor": gp/abs(gl) if gl < 0 else float("inf"), "net_pnl": balance-start_bal, "start_balance": start_bal, "final_balance": balance, "worst_intraday_drawdown": worst_dd, "worst_eod_drawdown": worst_eod_dd, "dll_hit_days": len(dll_days), "payouts_approved": 0, "payout_amount_total": 0.0, "net_plus_payouts": balance-start_bal}
    return result, events, daily, end_idx+1


def simulate_pa(trades, profile, start_idx, cycle_id):
    start_bal = profile.account_size
    balance = start_bal
    max_bal = start_bal
    eod_peak = start_bal
    floor = start_bal - profile.pa_max_drawdown
    events, daily, payouts = [], [], []
    start_time = trades.iloc[start_idx]["exit_dt_et"]
    start_date = trades.iloc[start_idx]["exit_date_et"]
    cur_day, day_pnl, day_trades, locked = None, 0.0, 0, False
    trading_days, dll_days, profit_lock_days, soft_loss_days = set(), set(), set(), set()
    daily_pnls_since_payout = {}
    payout_count, total_payouts = 0, 0.0
    trades_count = wins = losses = consecutive_losses = 0
    gp = gl = 0.0
    worst_dd = worst_eod_dd = 0.0
    status, reason = "OPEN", ""
    end_idx = start_idx - 1

    def close_day(day, t):
        nonlocal eod_peak, floor, worst_eod_dd, balance, payout_count, total_payouts, daily_pnls_since_payout, status, reason
        if day is None: return
        eod_peak = max(eod_peak, balance)
        floor = update_eod_floor(floor, eod_peak, profile.pa_max_drawdown, profile.pa_drawdown_stop_level)
        eod_dd = balance - eod_peak
        worst_eod_dd = min(worst_eod_dd, eod_dd)
        daily_pnls_since_payout[day] = daily_pnls_since_payout.get(day, 0.0) + day_pnl
        before = balance
        balance, payout_count, completed, daily_pnls_since_payout = maybe_payout(profile, cycle_id, t, balance, payout_count, daily_pnls_since_payout, events, payouts)
        paid = before - balance
        total_payouts += paid
        daily.append({"cycle_id": cycle_id, "phase": "pa", "date": str(day), "day_pnl": day_pnl, "balance": balance, "drawdown_floor": floor, "eod_drawdown": eod_dd, "dll_hit": day in dll_days, "profit_lock_hit": day in profit_lock_days, "soft_loss_hit": day in soft_loss_days, "payout_count": payout_count, "payout_amount_eod": paid})
        if completed:
            status, reason = "PA_COMPLETED_MAX_PAYOUTS", "max_payouts_completed"

    for i in range(start_idx, len(trades)):
        if status != "OPEN": break
        row = trades.iloc[i]; t = row["exit_dt_et"]; day = row["exit_date_et"]
        if cur_day is None: cur_day = day
        if day != cur_day:
            close_day(cur_day, t)
            if status != "OPEN": end_idx = max(i-1, start_idx); break
            cur_day, day_pnl, day_trades, locked, consecutive_losses = day, 0.0, 0, False, 0
        if locked: continue
        if profile.pa_max_trades_per_day is not None and day_trades >= profile.pa_max_trades_per_day:
            locked = True
            events.append({"cycle_id": cycle_id, "phase": "pa", "event_type": "DAILY_MAX_TRADES_LOCK", "time_et": t, "trade_index": i, "day_pnl": day_pnl, "balance": balance})
            continue
        pnl = float(row["net_pnl_dollars"])
        balance += pnl; day_pnl += pnl; day_trades += 1; trading_days.add(day)
        trades_count += 1; end_idx = i
        if pnl > 0: wins += 1; gp += pnl; consecutive_losses = 0
        elif pnl < 0: losses += 1; gl += pnl; consecutive_losses += 1
        max_bal = max(max_bal, balance); worst_dd = min(worst_dd, balance - max_bal)
        if balance <= floor:
            status, reason = "PA_BLOWN", "pa_max_loss_breach"
            events.append({"cycle_id": cycle_id, "phase": "pa", "event_type": "PA_BLOWN_MAX_LOSS", "time_et": t, "trade_index": i, "day_pnl": day_pnl, "balance": balance, "floor": floor, "payout_count": payout_count, "payout_amount_total": total_payouts})
            break
        if profile.pa_daily_loss_limit is not None and day_pnl <= -abs(profile.pa_daily_loss_limit):
            dll_days.add(day); locked = True
            events.append({"cycle_id": cycle_id, "phase": "pa", "event_type": "DLL_PAUSE", "time_et": t, "trade_index": i, "day_pnl": day_pnl, "balance": balance})
            continue
        if profile.pa_daily_soft_loss_stop is not None and day_pnl <= -abs(profile.pa_daily_soft_loss_stop):
            soft_loss_days.add(day); locked = True
            events.append({"cycle_id": cycle_id, "phase": "pa", "event_type": "DAILY_SOFT_LOSS_LOCK", "time_et": t, "trade_index": i, "day_pnl": day_pnl, "balance": balance})
            continue
        if profile.pa_daily_profit_target is not None and day_pnl >= profile.pa_daily_profit_target:
            profit_lock_days.add(day); locked = True
            events.append({"cycle_id": cycle_id, "phase": "pa", "event_type": "DAILY_PROFIT_TARGET_LOCK", "time_et": t, "trade_index": i, "day_pnl": day_pnl, "balance": balance, "target": profile.pa_daily_profit_target})
            continue
        if profile.pa_pause_after_consecutive_losses is not None and consecutive_losses >= profile.pa_pause_after_consecutive_losses:
            locked = True
            events.append({"cycle_id": cycle_id, "phase": "pa", "event_type": "CONSECUTIVE_LOSS_LOCK", "time_et": t, "trade_index": i, "day_pnl": day_pnl, "balance": balance, "consecutive_losses": consecutive_losses})
            continue
    if cur_day is not None and status == "OPEN":
        close_day(cur_day, trades.iloc[end_idx]["exit_dt_et"] if end_idx >= start_idx else start_time)
    if status == "OPEN":
        status, reason, end_idx = "DATA_END", "no_more_trades", len(trades)-1
    end_time = trades.iloc[end_idx]["exit_dt_et"] if 0 <= end_idx < len(trades) else pd.NaT
    result = {"cycle_id": cycle_id, "phase": "pa", "status": status, "reason": reason, "start_index": start_idx, "end_index": end_idx, "next_index": end_idx+1, "start_time_et": start_time, "end_time_et": end_time, "calendar_days": (pd.Timestamp(end_time).date()-pd.Timestamp(start_date).date()).days if pd.notna(end_time) else 0, "trading_days": len(trading_days), "trades": trades_count, "wins": wins, "losses": losses, "win_rate_pct": wins/trades_count*100 if trades_count else 0, "gross_profit": gp, "gross_loss": gl, "profit_factor": gp/abs(gl) if gl < 0 else float("inf"), "net_pnl": balance-start_bal, "start_balance": start_bal, "final_balance": balance, "worst_intraday_drawdown": worst_dd, "worst_eod_drawdown": worst_eod_dd, "dll_hit_days": len(dll_days), "profit_lock_days": len(profit_lock_days), "soft_loss_days": len(soft_loss_days), "payouts_approved": payout_count, "payout_amount_total": total_payouts, "net_plus_payouts": (balance-start_bal)+total_payouts}
    return result, events, daily, payouts, end_idx+1


def run_lifecycle(trades, profile, max_cycles):
    idx, cycle_id = 0, 1
    cycles, events, daily, payouts = [], [], [], []
    while idx < len(trades) and cycle_id <= max_cycles:
        er, ee, ed, next_idx = simulate_eval(trades, profile, idx, cycle_id)
        cycles.append(er); events.extend(ee); daily.extend(ed)
        if er["status"] == "EVAL_PASSED" and next_idx < len(trades):
            pr, pe, pdaily, pp, pa_next = simulate_pa(trades, profile, next_idx, cycle_id)
            cycles.append(pr); events.extend(pe); daily.extend(pdaily); payouts.extend(pp)
            idx = pa_next
        else:
            idx = next_idx
        if er["status"] == "DATA_END": break
        cycle_id += 1
    return pd.DataFrame(cycles), pd.DataFrame(events), pd.DataFrame(daily), pd.DataFrame(payouts)


def write_summary(cycles, events, payouts, profile):
    lines = [
        "================ PROP LIFECYCLE PAYOUT SUMMARY ================",
        f"Profile: {profile.name}",
        f"Prop firm: {profile.prop_firm}",
        f"PA daily profit target: {profile.pa_daily_profit_target}",
        f"PA soft loss stop: {profile.pa_daily_soft_loss_stop}",
        f"PA max trades/day: {profile.pa_max_trades_per_day}",
        f"PA post-payout buffer: {profile.pa_post_payout_buffer}",
        "",
    ]
    if cycles.empty:
        lines.append("No cycles.")
    else:
        evals = cycles[cycles["phase"] == "eval"]
        pas = cycles[cycles["phase"] == "pa"]
        ep = evals[evals["status"] == "EVAL_PASSED"]
        ef = evals[evals["status"] == "EVAL_FAILED"]
        pb = pas[pas["status"] == "PA_BLOWN"]
        pc = pas[pas["status"] == "PA_COMPLETED_MAX_PAYOUTS"]
        pe = pas[pas["status"] == "DATA_END"]
        lines += [
            f"Eval attempts: {len(evals)}",
            f"Eval passed: {len(ep)}",
            f"Eval failed max loss: {len(ef)}",
            f"Eval pass rate: {(len(ep)/len(evals)*100) if len(evals) else 0:.2f}%",
            "",
            f"PA accounts started: {len(pas)}",
            f"PA blown: {len(pb)}",
            f"PA completed max payouts: {len(pc)}",
            f"PA active/data-ended: {len(pe)}",
            "",
        ]
        if len(pas):
            lines += [
                f"Total PA retained net PnL: ${pas['net_pnl'].sum():,.2f}",
                f"Total approved payouts: ${pas['payout_amount_total'].sum():,.2f}",
                f"Total PA net + payouts: ${pas['net_plus_payouts'].sum():,.2f}",
                f"Avg PA net + payouts: ${pas['net_plus_payouts'].mean():,.2f}",
                f"Median PA net + payouts: ${pas['net_plus_payouts'].median():,.2f}",
                f"Best PA net + payouts: ${pas['net_plus_payouts'].max():,.2f}",
                f"Worst PA net + payouts: ${pas['net_plus_payouts'].min():,.2f}",
                f"Avg PA days survived: {pas['calendar_days'].mean():.2f}",
                f"Median PA days survived: {pas['calendar_days'].median():.2f}",
                f"Avg PA trades: {pas['trades'].mean():.2f}",
                f"Total PA profit-lock days: {int(pas.get('profit_lock_days', pd.Series(dtype=float)).sum())}",
                f"Total PA soft-loss days: {int(pas.get('soft_loss_days', pd.Series(dtype=float)).sum())}",
                f"Total PA DLL days: {int(pas.get('dll_hit_days', pd.Series(dtype=float)).sum())}",
                "",
            ]
        if not payouts.empty:
            lines += [
                f"Approved payout count: {len(payouts)}",
                f"Approved payout total: ${payouts['payout_amount'].sum():,.2f}",
                f"Avg payout: ${payouts['payout_amount'].mean():,.2f}",
                f"Median payout: ${payouts['payout_amount'].median():,.2f}",
                "",
            ]
        if not events.empty:
            lines += ["Lifecycle events:", events["event_type"].value_counts().to_string()]
    text = "\n".join(lines)
    OUT_SUMMARY.write_text(text + "\n")
    print(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trade-log", default=str(ROOT / "event_trade_log.csv"))
    p.add_argument("--lifecycle-profile", default="apex_50k_eod_lifecycle")
    p.add_argument("--profiles-yaml", default=str(ROOT / "prop_firm_lifecycle_profiles.yaml"))
    p.add_argument("--max-cycles", type=int, default=9999)
    args = p.parse_args()

    profiles = load_lifecycle_profiles(Path(args.profiles_yaml))
    profile = profiles[args.lifecycle_profile]
    trades = load_trades(Path(args.trade_log))
    cycles, events, daily, payouts = run_lifecycle(trades, profile, args.max_cycles)
    cycles.to_csv(OUT_CYCLES, index=False)
    events.to_csv(OUT_EVENTS, index=False)
    daily.to_csv(OUT_DAILY, index=False)
    payouts.to_csv(OUT_PAYOUTS, index=False)
    write_summary(cycles, events, payouts, profile)
    print("\nWrote files:")
    print(f"  {OUT_SUMMARY}")
    print(f"  {OUT_CYCLES}")
    print(f"  {OUT_EVENTS}")
    print(f"  {OUT_DAILY}")
    print(f"  {OUT_PAYOUTS}")


if __name__ == "__main__":
    main()
