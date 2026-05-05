
from __future__ import annotations

"""
Performance Account simulator.

This reads event_trade_log.csv and simulates what happens AFTER eval pass.

Unlike eval mode:
  - no profit target
  - same daily loss/drawdown rules
  - reports survival, payout-style monthly PnL, drawdown breaches, daily loss breaches
  - can start after a provided timestamp/index

Recommended flow:
  1. Run event engine with apex_50k_pa profile and news_events.csv.
  2. Run this PA simulator on event_trade_log.csv.

Example:

PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.performance_account_simulator \
  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv \
  --prop-profile apex_50k_pa
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import Any, Dict, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent

OUT_PA_SUMMARY = ROOT / "performance_account_summary.txt"
OUT_PA_DAILY = ROOT / "performance_account_daily.csv"
OUT_PA_MONTHLY = ROOT / "performance_account_monthly.csv"
OUT_PA_BREACHES = ROOT / "performance_account_breaches.csv"


@dataclass(frozen=True)
class Profile:
    name: str
    account_size: float
    max_drawdown: float
    drawdown_type: str
    daily_loss_limit: Optional[float]


def load_profile(name: str) -> Profile:
    raw = yaml.safe_load((ROOT / "prop_profiles.yaml").read_text()) or {}
    cfg = raw[name]
    return Profile(
        name=name,
        account_size=float(cfg.get("account_size", 50000)),
        max_drawdown=float(cfg.get("max_drawdown", 2000)),
        drawdown_type=str(cfg.get("drawdown_type", "eod")),
        daily_loss_limit=float(cfg["daily_loss_limit"]) if cfg.get("daily_loss_limit") is not None else None,
    )


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "net_pnl_dollars" not in df.columns:
        df["net_pnl_dollars"] = df["gross_pnl_dollars"]
    if "exit_time_et" in df.columns:
        df["exit_dt_et"] = pd.to_datetime(df["exit_time_et"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    else:
        df["exit_dt_et"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    df = df.dropna(subset=["exit_dt_et"]).sort_values("exit_dt_et").reset_index(drop=True)
    df["exit_date_et"] = df["exit_dt_et"].dt.date
    df["exit_month_et"] = df["exit_dt_et"].dt.to_period("M").astype(str)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-log", default=str(ROOT / "event_trade_log.csv"))
    parser.add_argument("--prop-profile", default="apex_50k_pa")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--start-after", default=None, help="Optional ET datetime, e.g. 2025-01-01 00:00:00")
    args = parser.parse_args()

    profile = load_profile(args.prop_profile)
    df = load_trades(Path(args.trade_log))

    if args.start_after:
        start_dt = pd.to_datetime(args.start_after)
        df = df[df["exit_dt_et"] >= start_dt].copy()
    elif args.start_index:
        df = df.iloc[args.start_index:].copy()

    balance = profile.account_size
    peak = balance
    eod_peak = balance
    floor = balance - profile.max_drawdown
    daily = []
    breaches = []
    current_day = None
    day_pnl = 0.0
    current_month = None

    for _, row in df.iterrows():
        day = row["exit_date_et"]
        if current_day is None:
            current_day = day
        if day != current_day:
            eod_peak = max(eod_peak, balance)
            if profile.drawdown_type.lower() == "eod":
                floor = max(floor, eod_peak - profile.max_drawdown)
            daily.append({
                "date": current_day,
                "day_pnl": day_pnl,
                "balance": balance,
                "eod_peak": eod_peak,
                "drawdown_floor": floor,
                "eod_drawdown": balance - eod_peak,
            })
            current_day = day
            day_pnl = 0.0

        pnl = float(row["net_pnl_dollars"])
        balance += pnl
        day_pnl += pnl
        peak = max(peak, balance)

        if profile.drawdown_type.lower() in ("intraday", "trailing"):
            floor = max(floor, peak - profile.max_drawdown)

        if profile.daily_loss_limit is not None and day_pnl <= -abs(profile.daily_loss_limit):
            breaches.append({
                "time_et": row["exit_dt_et"],
                "type": "daily_loss_breach",
                "day_pnl": day_pnl,
                "balance": balance,
            })

        if balance <= floor:
            breaches.append({
                "time_et": row["exit_dt_et"],
                "type": f"drawdown_breach_{profile.drawdown_type}",
                "day_pnl": day_pnl,
                "balance": balance,
                "floor": floor,
            })

    if current_day is not None:
        eod_peak = max(eod_peak, balance)
        daily.append({
            "date": current_day,
            "day_pnl": day_pnl,
            "balance": balance,
            "eod_peak": eod_peak,
            "drawdown_floor": floor,
            "eod_drawdown": balance - eod_peak,
        })

    daily_df = pd.DataFrame(daily)
    breaches_df = pd.DataFrame(breaches)
    monthly = (
        daily_df.assign(month=pd.to_datetime(daily_df["date"]).dt.to_period("M").astype(str))
        .groupby("month")
        .agg(
            trading_days=("day_pnl", "size"),
            net_pnl=("day_pnl", "sum"),
            best_day=("day_pnl", "max"),
            worst_day=("day_pnl", "min"),
            ending_balance=("balance", "last"),
            worst_eod_drawdown=("eod_drawdown", "min"),
        )
        .reset_index()
    ) if not daily_df.empty else pd.DataFrame()

    net = balance - profile.account_size
    lines = []
    lines.append("================ PERFORMANCE ACCOUNT SUMMARY ================")
    lines.append(f"Profile: {profile.name}")
    lines.append(f"Trades analysed: {len(df)}")
    lines.append(f"Start balance: ${profile.account_size:,.2f}")
    lines.append(f"Final balance: ${balance:,.2f}")
    lines.append(f"Net PnL: ${net:,.2f}")
    lines.append(f"Daily loss breaches: {len(breaches_df[breaches_df['type']=='daily_loss_breach']) if not breaches_df.empty else 0}")
    lines.append(f"Drawdown breaches: {len(breaches_df[breaches_df['type'].astype(str).str.contains('drawdown')]) if not breaches_df.empty else 0}")
    if not daily_df.empty:
        lines.append(f"Worst day: ${daily_df['day_pnl'].min():,.2f}")
        lines.append(f"Best day: ${daily_df['day_pnl'].max():,.2f}")
        lines.append(f"Worst EOD drawdown: ${daily_df['eod_drawdown'].min():,.2f}")
    text = "\n".join(lines)
    print(text)

    OUT_PA_SUMMARY.write_text(text + "\n")
    daily_df.to_csv(OUT_PA_DAILY, index=False)
    monthly.to_csv(OUT_PA_MONTHLY, index=False)
    breaches_df.to_csv(OUT_PA_BREACHES, index=False)

    print("\nWrote files:")
    print(f"  {OUT_PA_SUMMARY}")
    print(f"  {OUT_PA_DAILY}")
    print(f"  {OUT_PA_MONTHLY}")
    print(f"  {OUT_PA_BREACHES}")


if __name__ == "__main__":
    main()
