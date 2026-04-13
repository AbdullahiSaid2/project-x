from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class V471Policy:
    source_trade_csv_name: str = "v470_trade_log.csv"
    output_trade_csv_name: str = "v471_trade_log.csv"
    output_monthly_csv_name: str = "v471_monthly_pnl_summary.csv"
    output_apex_monthly_csv_name: str = "v471_apex_50k_monthly_summary.csv"
    output_apex_daily_csv_name: str = "v471_apex_50k_daily_summary.csv"
    output_daily_calendar_csv_name: str = "v471_daily_pnl_calendar_et.csv"
    output_daily_apex_csv_name: str = "v471_daily_pnl_apex_session.csv"

    allowed_setup_tiers: tuple[str, ...] = ("A",)
    allowed_setup_types: tuple[str, ...] = ("LONDON_CONTINUATION", "NYPM_CONTINUATION")
    min_planned_rr: float = 5.0
    min_target_dollars_10_mnq: float = 500.0
    min_realized_winner_dollars: float = 500.0
    fixed_contracts: int = 10

    apex_start_balance: float = 50_000.0
    apex_daily_loss_limit: float = -1_000.0
    apex_max_drawdown_limit: float = -2_000.0


POLICY = V471Policy()

ROOT = Path(__file__).resolve().parents[3]

SOURCE_TRADE_CSV = ROOT / POLICY.source_trade_csv_name
OUT_TRADE_CSV = ROOT / POLICY.output_trade_csv_name
OUT_MONTHLY_CSV = ROOT / POLICY.output_monthly_csv_name
OUT_APEX_MONTHLY_CSV = ROOT / POLICY.output_apex_monthly_csv_name
OUT_APEX_DAILY_CSV = ROOT / POLICY.output_apex_daily_csv_name
OUT_DAILY_CALENDAR_CSV = ROOT / POLICY.output_daily_calendar_csv_name
OUT_DAILY_APEX_CSV = ROOT / POLICY.output_daily_apex_csv_name


REQUIRED_COLUMNS: tuple[str, ...] = (
    "setup_type",
    "setup_tier",
    "planned_rr",
    "planned_target_dollars_10_mnq",
    "gross_pnl_dollars_dynamic",
    "gross_points",
    "entry_time_et",
    "exit_time_et",
    "entry_apex_session_date",
    "exit_apex_session_date",
    "calendar_exit_date_et",
    "entry_month_et",
    "exit_month_et",
)


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in source CSV: {missing}")


def _load_source_trades() -> pd.DataFrame:
    if not SOURCE_TRADE_CSV.exists():
        raise FileNotFoundError(
            f"Missing source trade CSV: {SOURCE_TRADE_CSV}\n"
            f"Expected it in the trading_system repo root."
        )

    df = pd.read_csv(SOURCE_TRADE_CSV)
    _require_columns(df, REQUIRED_COLUMNS)
    return _coerce_numeric(
        df,
        [
            "planned_rr",
            "planned_target_dollars_10_mnq",
            "gross_pnl_dollars_dynamic",
            "gross_points",
        ],
    )


def build_v471_trade_log() -> pd.DataFrame:
    df = _load_source_trades()

    df = df[
        df["setup_tier"].astype(str).isin(POLICY.allowed_setup_tiers)
        & df["setup_type"].astype(str).isin(POLICY.allowed_setup_types)
    ].copy()

    df = df[df["planned_rr"] >= POLICY.min_planned_rr].copy()
    df = df[df["planned_target_dollars_10_mnq"] >= POLICY.min_target_dollars_10_mnq].copy()

    # V471 execution filter:
    # - keep all losers and flats
    # - keep winners only if realized >= $500
    pnl = pd.to_numeric(df["gross_pnl_dollars_dynamic"], errors="coerce")
    df = df[(pnl <= 0) | (pnl >= POLICY.min_realized_winner_dollars)].copy()

    if "report_contracts" not in df.columns:
        df["report_contracts"] = POLICY.fixed_contracts
    else:
        df["report_contracts"] = POLICY.fixed_contracts

    if "exit_time_et" in df.columns and "entry_time_et" in df.columns:
        df = df.sort_values(["exit_time_et", "entry_time_et"], kind="stable").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def save_v471_trade_log(df: pd.DataFrame) -> Path:
    df.to_csv(OUT_TRADE_CSV, index=False)
    return OUT_TRADE_CSV


def load_v471_trades() -> pd.DataFrame:
    if not OUT_TRADE_CSV.exists():
        raise FileNotFoundError(f"Missing {OUT_TRADE_CSV}")
    return pd.read_csv(OUT_TRADE_CSV)


def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["gross_pnl_dollars_dynamic"] = pd.to_numeric(work["gross_pnl_dollars_dynamic"], errors="coerce")
    work["gross_points"] = pd.to_numeric(work["gross_points"], errors="coerce")

    out = (
        work.groupby("exit_month_et", dropna=False)
        .agg(
            trades=("gross_pnl_dollars_dynamic", "count"),
            gross_pnl_dollars_dynamic=("gross_pnl_dollars_dynamic", "sum"),
            gross_points=("gross_points", "sum"),
            avg_trade_dollars_dynamic=("gross_pnl_dollars_dynamic", "mean"),
            avg_points_per_trade=("gross_points", "mean"),
        )
        .reset_index()
        .sort_values("exit_month_et")
    )

    out["win_rate_pct"] = (
        work.assign(win=work["gross_pnl_dollars_dynamic"] > 0)
        .groupby("exit_month_et")["win"]
        .mean()
        .mul(100)
        .values
    )
    out["cumulative_pnl_dollars_dynamic"] = out["gross_pnl_dollars_dynamic"].cumsum()
    return out


def build_apex_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["gross_pnl_dollars"] = pd.to_numeric(work["gross_pnl_dollars_dynamic"], errors="coerce")
    work["gross_points"] = pd.to_numeric(work["gross_points"], errors="coerce")

    out = (
        work.groupby("exit_apex_session_date", dropna=False)
        .agg(
            trades=("gross_pnl_dollars", "count"),
            gross_pnl_dollars=("gross_pnl_dollars", "sum"),
            gross_points=("gross_points", "sum"),
            avg_trade_dollars=("gross_pnl_dollars", "mean"),
        )
        .reset_index()
        .sort_values("exit_apex_session_date")
    )
    out["win_rate_pct"] = (
        work.assign(win=work["gross_pnl_dollars"] > 0)
        .groupby("exit_apex_session_date")["win"]
        .mean()
        .mul(100)
        .values
    )
    out["cumulative_pnl_dollars"] = out["gross_pnl_dollars"].cumsum()
    return out


def build_calendar_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["gross_pnl_dollars"] = pd.to_numeric(work["gross_pnl_dollars_dynamic"], errors="coerce")
    work["gross_points"] = pd.to_numeric(work["gross_points"], errors="coerce")

    out = (
        work.groupby("calendar_exit_date_et", dropna=False)
        .agg(
            trades=("gross_pnl_dollars", "count"),
            gross_pnl_dollars=("gross_pnl_dollars", "sum"),
            gross_points=("gross_points", "sum"),
            avg_trade_dollars=("gross_pnl_dollars", "mean"),
        )
        .reset_index()
        .sort_values("calendar_exit_date_et")
    )
    out["win_rate_pct"] = (
        work.assign(win=work["gross_pnl_dollars"] > 0)
        .groupby("calendar_exit_date_et")["win"]
        .mean()
        .mul(100)
        .values
    )
    out["cumulative_pnl_dollars"] = out["gross_pnl_dollars"].cumsum()
    return out


def build_apex_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["gross_pnl_dollars"] = pd.to_numeric(work["gross_pnl_dollars_dynamic"], errors="coerce")
    work["gross_points"] = pd.to_numeric(work["gross_points"], errors="coerce")

    monthly = (
        work.groupby("exit_month_et", dropna=False)
        .agg(
            trades=("gross_pnl_dollars", "count"),
            gross_pnl_dollars=("gross_pnl_dollars", "sum"),
            gross_points=("gross_points", "sum"),
            avg_trade_dollars=("gross_pnl_dollars", "mean"),
            avg_points_per_trade=("gross_points", "mean"),
        )
        .reset_index()
        .sort_values("exit_month_et")
    )

    monthly["win_rate_pct"] = (
        work.assign(win=work["gross_pnl_dollars"] > 0)
        .groupby("exit_month_et")["win"]
        .mean()
        .mul(100)
        .values
    )

    daily = build_apex_daily_summary(df)
    daily["month"] = daily["exit_apex_session_date"].astype(str).str.slice(0, 7)

    day_rollup = (
        daily.groupby("month", dropna=False)
        .agg(
            trading_days=("exit_apex_session_date", "count"),
            green_days=("gross_pnl_dollars", lambda s: int((s > 0).sum())),
            red_days=("gross_pnl_dollars", lambda s: int((s < 0).sum())),
            worst_day_dollars=("gross_pnl_dollars", "min"),
            best_day_dollars=("gross_pnl_dollars", "max"),
            avg_day_dollars=("gross_pnl_dollars", "mean"),
            soft_loss_cap_breach_days=("gross_pnl_dollars", lambda s: int((s <= POLICY.apex_daily_loss_limit).sum())),
        )
        .reset_index()
        .rename(columns={"month": "exit_month_et"})
    )

    out = monthly.merge(day_rollup, on="exit_month_et", how="left")
    out["cumulative_pnl_dollars"] = out["gross_pnl_dollars"].cumsum()
    out["account_balance_est"] = POLICY.apex_start_balance + out["cumulative_pnl_dollars"]
    out["month_green"] = out["gross_pnl_dollars"] > 0
    out["strong_month_flag"] = out["gross_pnl_dollars"] >= 2_000.0
    out["equity_peak_est"] = out["account_balance_est"].cummax()
    out["drawdown_from_peak_dollars"] = out["account_balance_est"] - out["equity_peak_est"]
    return out
