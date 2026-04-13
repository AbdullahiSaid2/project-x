
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

POINT_VALUE_PER_MNQ = 2.0
SOURCE_TRADE_CSV = ROOT / "v452_trade_log.csv"

OUT_TRADE_CSV = ROOT / "v460_trade_log.csv"
OUT_MONTHLY_CSV = ROOT / "v460_monthly_pnl_summary.csv"
OUT_APEX_MONTHLY_CSV = ROOT / "v460_apex_50k_monthly_summary.csv"
OUT_DAILY_CALENDAR_CSV = ROOT / "v460_daily_pnl_calendar_et.csv"
OUT_DAILY_APEX_CSV = ROOT / "v460_daily_pnl_apex_session.csv"


@dataclass(frozen=True)
class V460Policy:
    name: str = "V460"
    apex_start_balance: float = 50_000.0
    apex_daily_soft_loss_cap: float = -1_000.0
    apex_max_drawdown_limit: float = -2_000.0
    min_planned_rr: float = 3.0
    min_planned_target_dollars_10_mnq: float = 300.0
    use_short_bias_discount_for_longs: bool = True


POLICY = V460Policy()


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in source CSV: {missing}")


def load_source_trades() -> pd.DataFrame:
    if not SOURCE_TRADE_CSV.exists():
        raise FileNotFoundError(
            f"Missing source trade log: {SOURCE_TRADE_CSV}\n"
            "Place v452_trade_log.csv in the repo root trading_system directory."
        )
    df = pd.read_csv(SOURCE_TRADE_CSV)
    _require_columns(
        df,
        [
            "side",
            "setup_type",
            "bridge_type",
            "setup_tier",
            "planned_rr",
            "tp_points",
            "realized_points",
            "entry_time_et_naive",
            "exit_time_et_naive",
            "entry_month_et",
            "exit_month_et",
        ],
    )
    return df


def planned_target_dollars_10_mnq(df: pd.DataFrame) -> pd.Series:
    return df["tp_points"].astype(float) * POINT_VALUE_PER_MNQ * 10.0


def _base_contracts(setup_tier: str, setup_type: str) -> int:
    mapping = {
        ("A", "LONDON_CONTINUATION"): 10,
        ("A", "NYAM_CONTINUATION"): 8,
        ("A", "NYPM_CONTINUATION"): 7,
        ("A", "ASIA_CONTINUATION"): 5,
        ("B", "LONDON_CONTINUATION"): 5,
        ("B", "NYAM_CONTINUATION"): 4,
        ("B", "NYPM_CONTINUATION"): 4,
        ("B", "ASIA_CONTINUATION"): 3,
    }
    return mapping.get((str(setup_tier), str(setup_type)), 3)


def _directionally_weighted_contracts(setup_tier: str, setup_type: str, side: str) -> int:
    contracts = _base_contracts(setup_tier, setup_type)
    if POLICY.use_short_bias_discount_for_longs and str(side).upper() == "LONG":
        contracts = max(1, math.ceil(contracts * 0.75))
    return contracts


def apply_v460_policy(source_df: pd.DataFrame) -> pd.DataFrame:
    df = source_df.copy()

    df["planned_target_dollars_10_mnq"] = planned_target_dollars_10_mnq(df)
    df = df[
        (df["planned_rr"].astype(float) >= POLICY.min_planned_rr)
        & (df["planned_target_dollars_10_mnq"] >= POLICY.min_planned_target_dollars_10_mnq)
    ].copy()

    df["report_contracts"] = df.apply(
        lambda row: _directionally_weighted_contracts(
            row["setup_tier"], row["setup_type"], row["side"]
        ),
        axis=1,
    )

    df["entry_time_et"] = pd.to_datetime(df["entry_time_et_naive"], errors="coerce")
    df["exit_time_et"] = pd.to_datetime(df["exit_time_et_naive"], errors="coerce")
    df["calendar_exit_date_et"] = df["exit_time_et"].dt.date.astype(str)

    # Apex session date = prior ET calendar date for exits before 17:00 ET, else same date.
    exit_dt = df["exit_time_et"]
    session_date = exit_dt.dt.normalize()
    session_date = session_date.where(exit_dt.dt.hour >= 17, session_date - pd.Timedelta(days=1))
    df["exit_apex_session_date"] = session_date.dt.date.astype(str)

    df["gross_pnl_dollars_dynamic"] = (
        df["realized_points"].astype(float) * POINT_VALUE_PER_MNQ * df["report_contracts"].astype(float)
    )
    df["gross_pnl_dollars_5_mnq"] = df["realized_points"].astype(float) * POINT_VALUE_PER_MNQ * 5.0
    df["gross_pnl_dollars_10_mnq"] = df["realized_points"].astype(float) * POINT_VALUE_PER_MNQ * 10.0

    return df.sort_values(["exit_time_et", "entry_time_et"]).reset_index(drop=True)


def load_v460_trades() -> pd.DataFrame:
    if not OUT_TRADE_CSV.exists():
        raise FileNotFoundError(f"Missing {OUT_TRADE_CSV}")
    return pd.read_csv(OUT_TRADE_CSV, parse_dates=["entry_time_et", "exit_time_et"], low_memory=False)


def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby("exit_month_et", dropna=False)
        .agg(
            trades=("gross_pnl_dollars_dynamic", "size"),
            gross_pnl_dollars_dynamic=("gross_pnl_dollars_dynamic", "sum"),
            gross_points=("realized_points", "sum"),
            avg_trade_dollars_dynamic=("gross_pnl_dollars_dynamic", "mean"),
            avg_points_per_trade=("realized_points", "mean"),
            win_rate_pct=("gross_pnl_dollars_dynamic", lambda s: (s > 0).mean() * 100.0),
        )
        .reset_index()
        .sort_values("exit_month_et")
    )
    monthly["cumulative_pnl_dollars_dynamic"] = monthly["gross_pnl_dollars_dynamic"].cumsum()
    return monthly


def build_apex_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    day_summary = (
        df.groupby("exit_apex_session_date", dropna=False)
        .agg(gross_pnl_dollars=("gross_pnl_dollars_dynamic", "sum"))
        .reset_index()
        .sort_values("exit_apex_session_date")
    )
    day_summary["month"] = day_summary["exit_apex_session_date"].astype(str).str.slice(0, 7)

    monthly = (
        df.groupby("exit_month_et", dropna=False)
        .agg(
            trades=("gross_pnl_dollars_dynamic", "size"),
            gross_pnl_dollars=("gross_pnl_dollars_dynamic", "sum"),
            gross_points=("realized_points", "sum"),
            avg_trade_dollars=("gross_pnl_dollars_dynamic", "mean"),
            avg_points_per_trade=("realized_points", "mean"),
            win_rate_pct=("gross_pnl_dollars_dynamic", lambda s: (s > 0).mean() * 100.0),
        )
        .reset_index()
        .sort_values("exit_month_et")
    )

    month_days = []
    running_balance = POLICY.apex_start_balance
    equity_peak = POLICY.apex_start_balance

    for _, row in monthly.iterrows():
        month_key = str(row["exit_month_et"])
        d = day_summary[day_summary["month"] == month_key]["gross_pnl_dollars"]
        trading_days = int(d.size)
        green_days = int((d > 0).sum())
        red_days = int((d < 0).sum())
        worst_day = float(d.min()) if trading_days else 0.0
        best_day = float(d.max()) if trading_days else 0.0
        avg_day = float(d.mean()) if trading_days else 0.0
        soft_breach_days = int((d <= POLICY.apex_daily_soft_loss_cap).sum()) if trading_days else 0

        running_balance += float(row["gross_pnl_dollars"])
        equity_peak = max(equity_peak, running_balance)

        month_days.append(
            {
                "trading_days": trading_days,
                "green_days": green_days,
                "red_days": red_days,
                "worst_day_dollars": worst_day,
                "best_day_dollars": best_day,
                "avg_day_dollars": avg_day,
                "soft_loss_cap_breach_days": soft_breach_days,
                "cumulative_pnl_dollars": running_balance - POLICY.apex_start_balance,
                "account_balance_est": running_balance,
                "month_green": float(row["gross_pnl_dollars"]) > 0,
                "strong_month_flag": float(row["gross_pnl_dollars"]) >= 2500.0,
                "equity_peak_est": equity_peak,
                "drawdown_from_peak_dollars": running_balance - equity_peak,
            }
        )

    extra = pd.DataFrame(month_days)
    return pd.concat([monthly.reset_index(drop=True), extra], axis=1)


def build_daily_calendar_summary(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("calendar_exit_date_et", dropna=False)
        .agg(
            trades=("gross_pnl_dollars_dynamic", "size"),
            gross_pnl_dollars=("gross_pnl_dollars_dynamic", "sum"),
            gross_points=("realized_points", "sum"),
            avg_trade_dollars=("gross_pnl_dollars_dynamic", "mean"),
            win_rate_pct=("gross_pnl_dollars_dynamic", lambda s: (s > 0).mean() * 100.0),
        )
        .reset_index()
        .sort_values("calendar_exit_date_et")
    )
    daily["cumulative_pnl_dollars"] = daily["gross_pnl_dollars"].cumsum()
    return daily


def build_daily_apex_summary(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("exit_apex_session_date", dropna=False)
        .agg(
            trades=("gross_pnl_dollars_dynamic", "size"),
            gross_pnl_dollars=("gross_pnl_dollars_dynamic", "sum"),
            gross_points=("realized_points", "sum"),
            avg_trade_dollars=("gross_pnl_dollars_dynamic", "mean"),
            win_rate_pct=("gross_pnl_dollars_dynamic", lambda s: (s > 0).mean() * 100.0),
        )
        .reset_index()
        .sort_values("exit_apex_session_date")
    )
    daily["cumulative_pnl_dollars"] = daily["gross_pnl_dollars"].cumsum()
    return daily
