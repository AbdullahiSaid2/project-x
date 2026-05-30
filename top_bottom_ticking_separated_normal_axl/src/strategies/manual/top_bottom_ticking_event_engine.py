"""
Event-driven portfolio replay + Eval/PA simulator for top_bottom_ticking.

Place at:
    trading_system/src/strategies/manual/top_bottom_ticking_event_engine.py

Main use:
    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_event_engine \
      --input src/strategies/manual/top_bottom_ticking_trade_log_apex_50k_eval_365d_notail_axlfilter.csv \
      --preset apex_50k_eval_pa \
      --out-dir src/strategies/manual/event_reports \
      --mode replay

Why this exists:
- The normal Backtesting.py runner executes each symbol/variant separately.
- This module replays all accepted trade rows in chronological order as one combined account.
- It enforces portfolio-level daily loss, max loss, max open trades, max daily open risk,
  evaluation pass/fail, and PA pass/fail/survival metrics.

Important limitation:
- This is a trade-log event replay engine, not a tick-level fill simulator.
- It can enforce entry-time and closed-PnL portfolio rules accurately from the trade log.
- It cannot know exact intra-bar drawdown unless the trade log contains MAE/tick-level path data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PropLifecyclePreset:
    name: str = "apex_50k_eval_pa"
    starting_balance: float = 50_000.0
    eval_profit_target: float = 3_000.0
    eval_max_loss: float = 2_500.0
    eval_daily_loss: float = 1_250.0
    eval_min_trading_days: int = 7
    pa_starting_balance: float = 50_000.0
    pa_max_loss: float = 2_500.0
    pa_daily_loss: float = 1_250.0
    pa_min_survival_days: int = 20
    max_trades_per_day: int = 8
    max_open_trades: int = 1
    max_daily_open_risk: float = 1_000.0
    block_after_daily_loss: bool = True
    reset_daily_loss_each_session: bool = True
    one_trade_per_bar: bool = True
    allow_pa_after_eval_pass: bool = True


PRESETS: Dict[str, PropLifecyclePreset] = {
    "apex_50k_eval_pa": PropLifecyclePreset(),
    "apex_50k_conservative": PropLifecyclePreset(
        name="apex_50k_conservative",
        eval_daily_loss=900.0,
        pa_daily_loss=900.0,
        max_trades_per_day=4,
        max_open_trades=1,
        max_daily_open_risk=600.0,
    ),
    "paper_50k_loose": PropLifecyclePreset(
        name="paper_50k_loose",
        eval_daily_loss=2_500.0,
        pa_daily_loss=2_500.0,
        max_trades_per_day=99,
        max_open_trades=99,
        max_daily_open_risk=99_999.0,
    ),
}


@dataclass
class AccountState:
    phase: str = "EVAL"  # EVAL, PA, FAILED, COMPLETE
    balance: float = 50_000.0
    eval_start_balance: float = 50_000.0
    pa_start_balance: float = 50_000.0
    day: Optional[str] = None
    day_realized: float = 0.0
    day_trade_count: int = 0
    day_open_risk: float = 0.0
    trading_days: int = 0
    pa_trading_days: int = 0
    eval_passed: bool = False
    pa_failed: bool = False
    failed_reason: str = ""
    eval_pass_time: str = ""
    fail_time: str = ""
    peak_balance: float = 50_000.0
    trough_balance: float = 50_000.0
    total_accepted_entries: int = 0
    total_rejected_entries: int = 0
    total_exit_events: int = 0
    total_pnl: float = 0.0


@dataclass
class ReplayResult:
    summary: Dict[str, object]
    accepted_entries: pd.DataFrame
    rejected_entries: pd.DataFrame
    exit_events: pd.DataFrame
    daily_summary: pd.DataFrame
    event_audit: pd.DataFrame


def _coerce_time(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def _session_day(ts: pd.Timestamp) -> str:
    """CME-style session date in New York. 18:00 ET belongs to next session day."""
    if pd.isna(ts):
        return "NaT"
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    et = t.tz_convert("America/New_York")
    d = et.date()
    if et.hour >= 18:
        d = (et + pd.Timedelta(days=1)).date()
    return str(d)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return default


def _make_trade_key(row: pd.Series) -> str:
    parts = [
        str(row.get("symbol", "")),
        str(row.get("variant", "")),
        str(row.get("setup_type", "")),
        str(row.get("entry_time", "")),
        f"{_safe_float(row.get('entry_price', np.nan), np.nan):.8f}",
        f"{_safe_float(row.get('planned_stop_price', np.nan), np.nan):.8f}",
        f"{_safe_float(row.get('planned_entry_price', np.nan), np.nan):.8f}",
    ]
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_trade_log(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df

    if "entry_time" not in df.columns or "exit_time" not in df.columns:
        raise ValueError("Trade log must contain entry_time and exit_time columns.")
    if "gross_pnl_dollars_dynamic" not in df.columns:
        raise ValueError("Trade log must contain gross_pnl_dollars_dynamic. Run top_bottom_ticking_shared first.")

    df = df.copy()
    df["entry_time"] = _coerce_time(df["entry_time"])
    df["exit_time"] = _coerce_time(df["exit_time"])
    df = df.dropna(subset=["entry_time", "exit_time"]).sort_values(["entry_time", "exit_time"]).reset_index(drop=True)
    df["trade_key"] = df.apply(_make_trade_key, axis=1)
    df["entry_session_day"] = df["entry_time"].apply(_session_day)
    df["exit_session_day"] = df["exit_time"].apply(_session_day)

    # Planned risk is used only for portfolio open-risk checks.
    contracts = pd.to_numeric(df.get("report_contracts", 1), errors="coerce").fillna(1.0)
    dpp = pd.to_numeric(df.get("dollars_per_point", 1), errors="coerce").fillna(1.0)
    entry = pd.to_numeric(df.get("planned_entry_price", df.get("entry_price")), errors="coerce")
    stop = pd.to_numeric(df.get("planned_stop_price", np.nan), errors="coerce")
    risk_points = (entry - stop).abs()
    df["planned_risk_dollars"] = (risk_points * contracts * dpp).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _build_events(trades: pd.DataFrame) -> pd.DataFrame:
    entries = []
    for key, g in trades.groupby("trade_key", sort=False):
        first = g.sort_values("entry_time").iloc[0]
        entries.append({
            "time": first["entry_time"],
            "event_type": "ENTRY",
            "trade_key": key,
            "symbol": first.get("symbol", ""),
            "variant": first.get("variant", ""),
            "side": first.get("side", ""),
            "session_day": first.get("entry_session_day", _session_day(first["entry_time"])),
            "planned_risk_dollars": _safe_float(first.get("planned_risk_dollars", 0.0)),
            "pnl": 0.0,
            "source_row": int(g.index.min()),
        })

    exits = []
    for idx, row in trades.iterrows():
        exits.append({
            "time": row["exit_time"],
            "event_type": "EXIT",
            "trade_key": row["trade_key"],
            "symbol": row.get("symbol", ""),
            "variant": row.get("variant", ""),
            "side": row.get("side", ""),
            "session_day": row.get("exit_session_day", _session_day(row["exit_time"])),
            "planned_risk_dollars": 0.0,
            "pnl": _safe_float(row.get("gross_pnl_dollars_dynamic", 0.0)),
            "source_row": int(idx),
        })

    events = pd.DataFrame(entries + exits)
    if events.empty:
        return events
    order = {"EXIT": 0, "ENTRY": 1}  # Realize exits before same-timestamp new entries.
    events["event_order"] = events["event_type"].map(order).fillna(9)
    return events.sort_values(["time", "event_order", "source_row"]).reset_index(drop=True)


def _new_day_if_needed(state: AccountState, day: str, audit_rows: List[dict]):
    if state.day == day:
        return
    if state.day is not None:
        audit_rows.append({"time": "", "event_type": "DAY_CLOSE", "session_day": state.day, "balance": state.balance, "day_realized": state.day_realized})
    state.day = day
    state.day_realized = 0.0
    state.day_trade_count = 0
    state.day_open_risk = 0.0


def _reject(reason: str, row: pd.Series, state: AccountState, rejected: List[dict], audit: List[dict]):
    state.total_rejected_entries += 1
    out = row.to_dict()
    out.update({"reject_reason": reason, "phase": state.phase, "balance": state.balance, "day_realized": state.day_realized})
    rejected.append(out)
    audit.append({
        "time": row["time"],
        "event_type": "REJECT_ENTRY",
        "trade_key": row["trade_key"],
        "reason": reason,
        "phase": state.phase,
        "balance": state.balance,
        "day_realized": state.day_realized,
    })


def _allowed_entry(row: pd.Series, state: AccountState, preset: PropLifecyclePreset, open_trades: Dict[str, dict], used_bar_keys: set) -> Tuple[bool, str]:
    if state.phase in {"FAILED", "COMPLETE"}:
        return False, "account_not_active"
    if preset.block_after_daily_loss:
        daily_limit = preset.eval_daily_loss if state.phase == "EVAL" else preset.pa_daily_loss
        if state.day_realized <= -abs(daily_limit):
            return False, "daily_loss_lockout"
    if state.day_trade_count >= int(preset.max_trades_per_day):
        return False, "max_trades_per_day"
    if len(open_trades) >= int(preset.max_open_trades):
        return False, "max_open_trades"
    planned_risk = _safe_float(row.get("planned_risk_dollars", 0.0))
    if state.day_open_risk + planned_risk > float(preset.max_daily_open_risk):
        return False, "max_daily_open_risk"
    if preset.one_trade_per_bar:
        bar_key = (str(row.get("time")), str(row.get("symbol", "")))
        if bar_key in used_bar_keys:
            return False, "one_trade_per_bar"
    return True, "allowed"


def _check_failure_or_pass(state: AccountState, preset: PropLifecyclePreset, now) -> None:
    state.peak_balance = max(state.peak_balance, state.balance)
    state.trough_balance = min(state.trough_balance, state.balance)
    if state.phase == "EVAL":
        if state.balance <= state.eval_start_balance - abs(preset.eval_max_loss):
            state.phase = "FAILED"
            state.failed_reason = "eval_max_loss"
            state.fail_time = str(now)
            return
        if state.day_realized <= -abs(preset.eval_daily_loss):
            # Failure, not just lockout, because many prop accounts hard-fail on daily loss breach.
            state.phase = "FAILED"
            state.failed_reason = "eval_daily_loss"
            state.fail_time = str(now)
            return
        if (state.balance >= state.eval_start_balance + preset.eval_profit_target) and (state.trading_days >= preset.eval_min_trading_days):
            state.eval_passed = True
            state.eval_pass_time = str(now)
            if preset.allow_pa_after_eval_pass:
                state.phase = "PA"
                state.balance = preset.pa_starting_balance
                state.pa_start_balance = preset.pa_starting_balance
                state.peak_balance = state.balance
                state.trough_balance = state.balance
                state.day_realized = 0.0
                state.day_trade_count = 0
                state.day_open_risk = 0.0
            else:
                state.phase = "COMPLETE"
            return
    elif state.phase == "PA":
        if state.balance <= state.pa_start_balance - abs(preset.pa_max_loss):
            state.phase = "FAILED"
            state.pa_failed = True
            state.failed_reason = "pa_max_loss"
            state.fail_time = str(now)
            return
        if state.day_realized <= -abs(preset.pa_daily_loss):
            state.phase = "FAILED"
            state.pa_failed = True
            state.failed_reason = "pa_daily_loss"
            state.fail_time = str(now)
            return


def replay_trade_log(trades: pd.DataFrame, preset: PropLifecyclePreset) -> ReplayResult:
    events = _build_events(trades)
    state = AccountState(balance=preset.starting_balance, eval_start_balance=preset.starting_balance, pa_start_balance=preset.pa_starting_balance, peak_balance=preset.starting_balance, trough_balance=preset.starting_balance)
    open_trades: Dict[str, dict] = {}
    accepted_keys = set()
    used_bar_keys = set()
    accepted: List[dict] = []
    rejected: List[dict] = []
    exits: List[dict] = []
    audit: List[dict] = []
    daily: Dict[str, dict] = {}
    active_days_eval = set()
    active_days_pa = set()

    for _, row in events.iterrows():
        day = str(row.get("session_day", _session_day(row["time"])))
        _new_day_if_needed(state, day, audit)
        if state.phase in {"FAILED", "COMPLETE"}:
            # Still audit skipped future events.
            audit.append({"time": row["time"], "event_type": "SKIP_AFTER_DONE", "trade_key": row["trade_key"], "phase": state.phase, "balance": state.balance})
            continue

        if row["event_type"] == "ENTRY":
            allowed, reason = _allowed_entry(row, state, preset, open_trades, used_bar_keys)
            if not allowed:
                _reject(reason, row, state, rejected, audit)
                continue

            key = str(row["trade_key"])
            accepted_keys.add(key)
            open_trades[key] = row.to_dict()
            state.total_accepted_entries += 1
            state.day_trade_count += 1
            state.day_open_risk += _safe_float(row.get("planned_risk_dollars", 0.0))
            if state.phase == "EVAL":
                active_days_eval.add(day)
                state.trading_days = len(active_days_eval)
            elif state.phase == "PA":
                active_days_pa.add(day)
                state.pa_trading_days = len(active_days_pa)
            bar_key = (str(row.get("time")), str(row.get("symbol", "")))
            used_bar_keys.add(bar_key)
            accepted.append({**row.to_dict(), "phase": state.phase, "balance_before": state.balance})
            audit.append({"time": row["time"], "event_type": "ACCEPT_ENTRY", "trade_key": key, "phase": state.phase, "balance": state.balance, "day_trade_count": state.day_trade_count})

        elif row["event_type"] == "EXIT":
            key = str(row["trade_key"])
            if key not in accepted_keys:
                audit.append({"time": row["time"], "event_type": "SKIP_EXIT_REJECTED_TRADE", "trade_key": key, "phase": state.phase, "balance": state.balance})
                continue

            pnl = _safe_float(row.get("pnl", 0.0))
            state.balance += pnl
            state.day_realized += pnl
            state.total_pnl += pnl
            state.total_exit_events += 1
            exits.append({**row.to_dict(), "phase": state.phase, "balance_after": state.balance, "day_realized_after": state.day_realized})
            d = daily.setdefault(day, {"session_day": day, "gross_pnl": 0.0, "exit_events": 0, "entries": 0, "phase_last": state.phase})
            d["gross_pnl"] += pnl
            d["exit_events"] += 1
            d["phase_last"] = state.phase
            audit.append({"time": row["time"], "event_type": "EXIT_APPLIED", "trade_key": key, "phase": state.phase, "pnl": pnl, "balance": state.balance, "day_realized": state.day_realized})
            # If this is the final exit row for a trade, remove it from open risk.
            # We identify final by checking whether there are future EXIT events with same key.
            future_exists = bool(((events.index > _) & (events["trade_key"] == key) & (events["event_type"] == "EXIT")).any())
            if not future_exists and key in open_trades:
                state.day_open_risk = max(0.0, state.day_open_risk - _safe_float(open_trades[key].get("planned_risk_dollars", 0.0)))
                open_trades.pop(key, None)
            _check_failure_or_pass(state, preset, row["time"])

    for day, d in daily.items():
        # entries are counted from accepted entries by day below after converting to DF.
        pass

    accepted_df = pd.DataFrame(accepted)
    rejected_df = pd.DataFrame(rejected)
    exits_df = pd.DataFrame(exits)
    audit_df = pd.DataFrame(audit)
    daily_df = pd.DataFrame(daily.values()) if daily else pd.DataFrame(columns=["session_day", "gross_pnl", "exit_events", "entries", "phase_last"])
    if not accepted_df.empty:
        entries_by_day = accepted_df.groupby("session_day").size().rename("entries").reset_index()
        daily_df = daily_df.drop(columns=["entries"], errors="ignore").merge(entries_by_day, on="session_day", how="outer")
        daily_df["entries"] = daily_df["entries"].fillna(0).astype(int)
    if not daily_df.empty:
        daily_df = daily_df.sort_values("session_day").reset_index(drop=True)
        daily_df["cum_pnl"] = daily_df["gross_pnl"].fillna(0.0).cumsum()

    summary = {
        "preset": asdict(preset),
        "final_phase": state.phase,
        "eval_passed": bool(state.eval_passed),
        "pa_failed": bool(state.pa_failed),
        "failed_reason": state.failed_reason,
        "eval_pass_time": state.eval_pass_time,
        "fail_time": state.fail_time,
        "final_balance": round(float(state.balance), 2),
        "total_pnl_applied": round(float(state.total_pnl), 2),
        "peak_balance": round(float(state.peak_balance), 2),
        "trough_balance": round(float(state.trough_balance), 2),
        "eval_trading_days": int(state.trading_days),
        "pa_trading_days": int(state.pa_trading_days),
        "accepted_entries": int(state.total_accepted_entries),
        "rejected_entries": int(state.total_rejected_entries),
        "exit_events_applied": int(state.total_exit_events),
        "raw_rows_in_input": int(len(trades)),
    }
    return ReplayResult(summary, accepted_df, rejected_df, exits_df, daily_df, audit_df)


def monte_carlo_by_day(trades: pd.DataFrame, preset: PropLifecyclePreset, iterations: int, seed: int = 42, days_per_path: Optional[int] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    day_groups = {day: g.copy() for day, g in trades.groupby("entry_session_day", sort=True)}
    days = np.array(list(day_groups.keys()))
    if len(days) == 0:
        return pd.DataFrame()
    if days_per_path is None:
        days_per_path = len(days)
    rows = []
    for n in range(iterations):
        sampled_days = rng.choice(days, size=days_per_path, replace=True)
        parts = []
        offset = pd.Timedelta(days=0)
        base_time = pd.Timestamp("2025-01-01", tz="UTC")
        for j, day in enumerate(sampled_days):
            g = day_groups[day].copy()
            # Preserve within-day sequence but remap dates so session ordering is clean.
            min_entry = g["entry_time"].min()
            delta = (base_time + pd.Timedelta(days=j)) - min_entry
            g["entry_time"] = g["entry_time"] + delta
            g["exit_time"] = g["exit_time"] + delta
            g["entry_session_day"] = g["entry_time"].apply(_session_day)
            g["exit_session_day"] = g["exit_time"].apply(_session_day)
            parts.append(g)
        path_trades = pd.concat(parts, ignore_index=True).sort_values(["entry_time", "exit_time"]).reset_index(drop=True)
        result = replay_trade_log(path_trades, preset)
        s = dict(result.summary)
        rows.append({
            "iteration": n + 1,
            "eval_passed": s["eval_passed"],
            "pa_failed": s["pa_failed"],
            "final_phase": s["final_phase"],
            "failed_reason": s["failed_reason"],
            "final_balance": s["final_balance"],
            "total_pnl_applied": s["total_pnl_applied"],
            "accepted_entries": s["accepted_entries"],
            "rejected_entries": s["rejected_entries"],
            "eval_trading_days": s["eval_trading_days"],
            "pa_trading_days": s["pa_trading_days"],
        })
    return pd.DataFrame(rows)


def _write_outputs(result: ReplayResult, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}_lifecycle_summary.json").write_text(json.dumps(result.summary, indent=2, default=str), encoding="utf-8")
    result.accepted_entries.to_csv(out_dir / f"{prefix}_accepted_entries.csv", index=False)
    result.rejected_entries.to_csv(out_dir / f"{prefix}_rejected_entries.csv", index=False)
    result.exit_events.to_csv(out_dir / f"{prefix}_exit_events_applied.csv", index=False)
    result.daily_summary.to_csv(out_dir / f"{prefix}_daily_summary.csv", index=False)
    result.event_audit.to_csv(out_dir / f"{prefix}_event_audit.csv", index=False)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Event-driven portfolio replay + Eval/PA simulator for top_bottom_ticking trade logs.")
    parser.add_argument("--input", required=True, help="Path to top_bottom_ticking trade_log CSV.")
    parser.add_argument("--out-dir", default="src/strategies/manual/event_reports")
    parser.add_argument("--preset", default="apex_50k_eval_pa", choices=list(PRESETS))
    parser.add_argument("--mode", default="replay", choices=["replay", "monte_carlo", "both"])
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days-per-path", type=int, default=None)
    parser.add_argument("--max-open-trades", type=int, default=None)
    parser.add_argument("--max-trades-per-day", type=int, default=None)
    parser.add_argument("--max-daily-open-risk", type=float, default=None)
    parser.add_argument("--daily-loss", type=float, default=None, help="Override both eval and PA daily loss.")
    parser.add_argument("--one-trade-per-bar", action="store_true", default=None)
    parser.add_argument("--allow-multiple-per-bar", action="store_true", default=False)
    args = parser.parse_args(argv)

    preset = PRESETS[args.preset]
    updates = asdict(preset)
    if args.max_open_trades is not None:
        updates["max_open_trades"] = args.max_open_trades
    if args.max_trades_per_day is not None:
        updates["max_trades_per_day"] = args.max_trades_per_day
    if args.max_daily_open_risk is not None:
        updates["max_daily_open_risk"] = args.max_daily_open_risk
    if args.daily_loss is not None:
        updates["eval_daily_loss"] = abs(args.daily_loss)
        updates["pa_daily_loss"] = abs(args.daily_loss)
    if args.allow_multiple_per_bar:
        updates["one_trade_per_bar"] = False
    elif args.one_trade_per_bar is not None:
        updates["one_trade_per_bar"] = True
    preset = PropLifecyclePreset(**updates)

    trades = load_trade_log(args.input)
    out_dir = Path(args.out_dir)
    prefix = Path(args.input).stem

    if args.mode in {"replay", "both"}:
        result = replay_trade_log(trades, preset)
        _write_outputs(result, out_dir, prefix)
        print(json.dumps(result.summary, indent=2, default=str))
        print(f"Saved replay reports -> {out_dir.resolve()}")

    if args.mode in {"monte_carlo", "both"}:
        mc = monte_carlo_by_day(trades, preset, iterations=args.iterations, seed=args.seed, days_per_path=args.days_per_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        mc_path = out_dir / f"{prefix}_monte_carlo.csv"
        mc.to_csv(mc_path, index=False)
        if not mc.empty:
            summary = {
                "iterations": int(len(mc)),
                "eval_pass_rate_pct": round(float(mc["eval_passed"].mean() * 100), 2),
                "pa_fail_rate_after_eval_pct": round(float(mc.loc[mc["eval_passed"], "pa_failed"].mean() * 100), 2) if mc["eval_passed"].any() else None,
                "median_final_balance": round(float(mc["final_balance"].median()), 2),
                "p10_final_balance": round(float(mc["final_balance"].quantile(0.10)), 2),
                "p90_final_balance": round(float(mc["final_balance"].quantile(0.90)), 2),
                "top_failure_reasons": mc["failed_reason"].value_counts().head(10).to_dict(),
            }
            (out_dir / f"{prefix}_monte_carlo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
        print(f"Saved Monte Carlo reports -> {mc_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
