from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Type

import pandas as pd

ROOT = Path(__file__).resolve().parents[0]
PROJECT_ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) >= 4 else ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting import Backtest
from src.data.fetcher import get_ohlcv
from ict_multi_setup_v452 import ICT_MULTI_SETUP_V452


@dataclass(frozen=True)
class InstrumentConfig:
    symbol: str
    exchange: str
    timeframe: str
    days_back: int
    tail_rows: int
    contracts: int
    dollars_per_point: float
    min_target_points: float
    min_stop_points: float
    partial_rr: float
    risk_multiple: float
    pullback_entry_tolerance_points: float


# First-pass instrument profiles.
# These preserve the original NQ behavior and add pragmatic starter profiles for the new markets.
INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "NQ": InstrumentConfig("NQ", "tradovate", "1m", 365, 180_000, 5, 10.0, 50.0, 25.0, 1.0, 2.0, 6.0),
    "MES": InstrumentConfig("MES", "tradovate", "1m", 365, 180_000, 5, 25.0, 20.0, 10.0, 1.0, 2.0, 3.0),
    "MYM": InstrumentConfig("MYM", "tradovate", "1m", 365, 180_000, 5, 2.5, 150.0, 75.0, 1.0, 2.0, 20.0),
    "MGC": InstrumentConfig("MGC", "tradovate", "1m", 365, 180_000, 5, 50.0, 8.0, 4.0, 1.0, 2.0, 0.8),
    "MCL": InstrumentConfig("MCL", "tradovate", "1m", 365, 180_000, 5, 500.0, 0.8, 0.4, 1.0, 2.0, 0.08),
}

OUT_TRADE_CSV = ROOT / "v472_trade_log.csv"
OUT_MONTHLY_CSV = ROOT / "v472_monthly_pnl_summary.csv"
OUT_APEX_MONTHLY_CSV = ROOT / "v472_apex_50k_monthly_summary.csv"
OUT_APEX_DAILY_CSV = ROOT / "v472_apex_50k_daily_summary.csv"
OUT_CALENDAR_DAILY_CSV = ROOT / "v472_daily_pnl_calendar_et.csv"
OUT_APEX_SESSION_DAILY_CSV = ROOT / "v472_daily_pnl_apex_session.csv"
OUT_MONTHLY_XLSX = ROOT / "v472_monthly_pnl_export.xlsx"
OUT_APEX_XLSX = ROOT / "v472_apex_50k_monthly_payout_export.xlsx"

APEX_START_BALANCE = 50_000
DAILY_SOFT_LOSS_CAP = 1_000


def to_et(ts):
    if pd.isna(ts):
        return ts
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("America/New_York")


def _make_strategy_class(cfg: InstrumentConfig):
    attrs = {
        "min_target_points": cfg.min_target_points,
        "min_stop_points": cfg.min_stop_points,
        "partial_rr": cfg.partial_rr,
        "risk_multiple": cfg.risk_multiple,
        "pullback_entry_tolerance_points": cfg.pullback_entry_tolerance_points,
        "last_trade_log": [],
        "last_debug_counts": {},
    }
    return type(f"ICT_MULTI_SETUP_V472_{cfg.symbol}", (ICT_MULTI_SETUP_V452,), attrs)


def realized_points(row):
    if row.get("side") == "LONG":
        return float(row["exit_price"]) - float(row["entry_price"])
    return float(row["entry_price"]) - float(row["exit_price"])


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _prepare_meta(meta: pd.DataFrame, cfg: InstrumentConfig) -> pd.DataFrame:
    if meta.empty:
        return meta
    meta = meta.copy()
    meta["symbol"] = cfg.symbol
    meta["exchange"] = cfg.exchange
    meta["timeframe"] = cfg.timeframe
    meta["report_contracts"] = cfg.contracts
    meta["dollars_per_point"] = cfg.dollars_per_point
    meta["entry_time"] = pd.to_datetime(meta.get("entry_time"), errors="coerce")
    meta["exit_time"] = pd.to_datetime(meta.get("exit_time"), errors="coerce")
    meta["entry_time_et"] = meta["entry_time"].apply(to_et)
    meta["exit_time_et"] = meta["exit_time"].apply(to_et)
    meta["entry_time_et_naive"] = pd.to_datetime(meta["entry_time_et"], errors="coerce").dt.tz_localize(None)
    meta["exit_time_et_naive"] = pd.to_datetime(meta["exit_time_et"], errors="coerce").dt.tz_localize(None)
    meta["entry_month_et"] = meta["entry_time_et_naive"].dt.to_period("M").astype(str)
    meta["exit_month_et"] = meta["exit_time_et_naive"].dt.to_period("M").astype(str)
    meta["calendar_exit_date_et"] = meta["exit_time_et_naive"].dt.date
    # Apex session day approximated as prior ET date for exits before 17:00, matching prior project conventions loosely.
    et_exit = meta["exit_time_et_naive"]
    meta["exit_apex_session_date"] = (et_exit - pd.to_timedelta((et_exit.dt.hour < 17).astype(int), unit="D")).dt.date
    meta["realized_points"] = meta.apply(realized_points, axis=1)
    meta["gross_pnl_dollars_dynamic"] = meta["realized_points"] * cfg.dollars_per_point
    target_points = (pd.to_numeric(meta.get("planned_target_price"), errors="coerce") - pd.to_numeric(meta.get("planned_entry_price"), errors="coerce")).abs()
    meta["planned_target_dollars_dynamic"] = target_points * cfg.dollars_per_point
    return meta


def run_symbol(cfg: InstrumentConfig):
    StrategyCls = _make_strategy_class(cfg)
    print(f"\n=== {cfg.symbol} ===")
    print(f"Loading {cfg.symbol} data...")
    df = get_ohlcv(cfg.symbol, exchange=cfg.exchange, timeframe=cfg.timeframe, days_back=cfg.days_back)
    df = df.tail(cfg.tail_rows)
    print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()}")
    bt = Backtest(df, StrategyCls, cash=1_000_000, commission=0.0, exclusive_orders=True)
    stats = bt.run()
    meta = pd.DataFrame(getattr(StrategyCls, "last_trade_log", []))
    meta = _prepare_meta(meta, cfg)
    return stats, meta


def run_all_symbols(symbols: List[str] | None = None) -> pd.DataFrame:
    selected = symbols or list(INSTRUMENTS.keys())
    all_meta = []
    for sym in selected:
        cfg = INSTRUMENTS[sym]
        _, meta = run_symbol(cfg)
        if not meta.empty:
            all_meta.append(meta)
    if not all_meta:
        return pd.DataFrame()
    combined = pd.concat(all_meta, ignore_index=True)
    combined = combined.sort_values(["exit_time_et_naive", "symbol", "setup_type", "bridge_type"], na_position="last")
    return combined


def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["exit_month_et", "symbol"], dropna=False)
        .agg(
            trades=("gross_pnl_dollars_dynamic", "size"),
            gross_pnl_dollars_dynamic=("gross_pnl_dollars_dynamic", "sum"),
            gross_points=("realized_points", "sum"),
            avg_trade_dollars_dynamic=("gross_pnl_dollars_dynamic", "mean"),
            avg_points_per_trade=("realized_points", "mean"),
            win_rate_pct=("gross_pnl_dollars_dynamic", lambda s: (s > 0).mean() * 100),
        )
        .reset_index()
        .sort_values(["exit_month_et", "symbol"])
    )


def build_apex_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    daily = build_daily_summary(df, by_apex_session=True)
    monthly = (
        df.groupby(["exit_month_et", "symbol"], dropna=False)
        .agg(
            trades=("gross_pnl_dollars_dynamic", "size"),
            gross_pnl_dollars=("gross_pnl_dollars_dynamic", "sum"),
            gross_points=("realized_points", "sum"),
            avg_trade_dollars=("gross_pnl_dollars_dynamic", "mean"),
            avg_points_per_trade=("realized_points", "mean"),
            win_rate_pct=("gross_pnl_dollars_dynamic", lambda s: (s > 0).mean() * 100),
        )
        .reset_index()
        .sort_values(["exit_month_et", "symbol"])
    )
    month_days = (
        daily.groupby(["exit_month_et", "symbol"], dropna=False)
        .agg(
            trading_days=("session_date", "size"),
            green_days=("gross_pnl_dollars", lambda s: (s > 0).sum()),
            red_days=("gross_pnl_dollars", lambda s: (s < 0).sum()),
            worst_day_dollars=("gross_pnl_dollars", "min"),
            best_day_dollars=("gross_pnl_dollars", "max"),
            avg_day_dollars=("gross_pnl_dollars", "mean"),
            soft_loss_cap_breach_days=("gross_pnl_dollars", lambda s: (s <= -DAILY_SOFT_LOSS_CAP).sum()),
        )
        .reset_index()
    )
    monthly = monthly.merge(month_days, on=["exit_month_et", "symbol"], how="left")
    monthly["cumulative_pnl_dollars"] = monthly.groupby("symbol")["gross_pnl_dollars"].cumsum()
    monthly["account_balance_est"] = APEX_START_BALANCE + monthly["cumulative_pnl_dollars"]
    monthly["month_green"] = monthly["gross_pnl_dollars"] > 0
    monthly["strong_month_flag"] = monthly["gross_pnl_dollars"] >= 2_000
    monthly["equity_peak_est"] = monthly.groupby("symbol")["account_balance_est"].cummax()
    monthly["drawdown_from_peak_dollars"] = monthly["account_balance_est"] - monthly["equity_peak_est"]
    return monthly


def build_daily_summary(df: pd.DataFrame, by_apex_session: bool = False) -> pd.DataFrame:
    date_col = "exit_apex_session_date" if by_apex_session else "calendar_exit_date_et"
    out_name = "session_date" if by_apex_session else "calendar_date"
    daily = (
        df.groupby([date_col, "symbol"], dropna=False)
        .agg(
            trades=("gross_pnl_dollars_dynamic", "size"),
            gross_pnl_dollars=("gross_pnl_dollars_dynamic", "sum"),
            gross_points=("realized_points", "sum"),
            avg_trade_dollars=("gross_pnl_dollars_dynamic", "mean"),
            win_rate_pct=("gross_pnl_dollars_dynamic", lambda s: (s > 0).mean() * 100),
        )
        .reset_index()
        .rename(columns={date_col: out_name})
        .sort_values([out_name, "symbol"])
    )
    date_series = pd.to_datetime(daily[out_name], errors="coerce")
    daily["exit_month_et"] = date_series.dt.to_period("M").astype(str)
    daily["cumulative_pnl_dollars"] = daily.groupby("symbol")["gross_pnl_dollars"].cumsum()
    return daily
