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
}

APEX_ALLOWED_START_HOUR_ET = 18
APEX_CUTOFF_HOUR_ET = 17
APEX_CUTOFF_MINUTE_ET = 0

FORCE_FLAT_HOUR_ET = 16
FORCE_FLAT_MINUTE_ET = 50

OUT_TRADE_CSV = ROOT / "v473_trade_log.csv"
OUT_MONTHLY_CSV = ROOT / "v473_monthly_pnl_summary.csv"
OUT_APEX_MONTHLY_CSV = ROOT / "v473_apex_50k_monthly_summary.csv"
OUT_APEX_DAILY_CSV = ROOT / "v473_apex_50k_daily_summary.csv"
OUT_CALENDAR_DAILY_CSV = ROOT / "v473_daily_pnl_calendar_et.csv"
OUT_APEX_SESSION_DAILY_CSV = ROOT / "v473_daily_pnl_apex_session.csv"
OUT_MONTHLY_XLSX = ROOT / "v473_monthly_pnl_export.xlsx"
OUT_APEX_XLSX = ROOT / "v473_apex_50k_monthly_payout_export.xlsx"

APEX_START_BALANCE = 50_000
DAILY_SOFT_LOSS_CAP = 1_000


def to_et(ts):
    if pd.isna(ts):
        return ts
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("America/New_York")




class V473ForcedFlatMixin:
    def _after_force_flat_cutoff(self, row: pd.Series) -> bool:
        hour = int(row.get("et_hour", -1))
        minute = int(row.get("et_minute", -1))
        return (hour > FORCE_FLAT_HOUR_ET) or (hour == FORCE_FLAT_HOUR_ET and minute >= FORCE_FLAT_MINUTE_ET)

    def _force_flat_if_needed(self, row: pd.Series):
        if not self.position:
            return
        if not self._after_force_flat_cutoff(row):
            return
        try:
            self.position.close()
            self.debug_counts["forced_flat_1630"] = self.debug_counts.get("forced_flat_1630", 0) + 1
            self._sync_debug()
        except Exception:
            pass

    def next(self):
        self._log_newly_closed_trades()
        i = self._i()
        if i < 200:
            return

        row = self.m.iloc[i]
        self._update_day_reset(row)

        close_now = float(self.data.Close[-1])
        high_now = float(self.data.High[-1])
        low_now = float(self.data.Low[-1])
        atr3 = float(row["atr14_3m"]) if pd.notna(row["atr14_3m"]) else 0.0

        self._manage_trade()
        self._force_flat_if_needed(row)

        if self.position:
            return

        if self.pending.expiry_bar >= 0 and i > self.pending.expiry_bar:
            self.debug_counts["expire_pending"] += 1
            self._sync_debug()
            self._clear_pending()

        if self._after_force_flat_cutoff(row):
            if self.pending.direction:
                self.debug_counts["clear_pending_1630"] = self.debug_counts.get("clear_pending_1630", 0) + 1
                self._sync_debug()
                self._clear_pending()
            if self.state.direction:
                self.debug_counts["clear_state_1630"] = self.debug_counts.get("clear_state_1630", 0) + 1
                self._sync_debug()
                self._clear_state()
            return

        if self.pending.direction == "long":
            if self._pending_long_ready(row, high_now, low_now) and self._prop_can_trade():
                entry = max(close_now, self.pending.entry_trigger)
                risk = entry - float(self.pending.stop_price)
                if risk > 0:
                    self.active_risk = risk
                    self.buy(size=self._effective_size(), sl=float(self.pending.stop_price), tp=float(self.pending.runner_target))
                    self.daily_trade_count += 1
                    self.partial_taken = False
                    self.be_moved = False
                    self.entry_bar_idx = i
                    self.entry_day = row["et_date"]
                    self.entry_side = "LONG"
                    self._record_open_trade_meta(entry)
                    self.debug_counts["trigger_long_entry"] += 1
                    self._sync_debug()
                self._clear_pending()
                self._clear_state()
                return

        if self.pending.direction == "short":
            if self._pending_short_ready(row, high_now, low_now) and self._prop_can_trade():
                entry = min(close_now, self.pending.entry_trigger)
                risk = float(self.pending.stop_price) - entry
                if risk > 0:
                    self.active_risk = risk
                    self.sell(size=self._effective_size(), sl=float(self.pending.stop_price), tp=float(self.pending.runner_target))
                    self.daily_trade_count += 1
                    self.partial_taken = False
                    self.be_moved = False
                    self.entry_bar_idx = i
                    self.entry_day = row["et_date"]
                    self.entry_side = "SHORT"
                    self._record_open_trade_meta(entry)
                    self.debug_counts["trigger_short_entry"] += 1
                    self._sync_debug()
                self._clear_pending()
                self._clear_state()
                return

        self._expire_state_if_needed(i)

        if self.state.direction == "":
            if self._bull_narrative_ok(row):
                self.state.direction = "long"
                self.state.narrative_bar = i
                self.state.setup_type = "GLOBAL"
                self.debug_counts["arm_long_narrative"] += 1
                self._sync_debug()
                return
            if self._bear_narrative_ok(row):
                self.state.direction = "short"
                self.state.narrative_bar = i
                self.state.setup_type = "GLOBAL"
                self.debug_counts["arm_short_narrative"] += 1
                self._sync_debug()
                return

        if self.state.direction == "long":
            if self._bull_invalid(row):
                self.debug_counts["clear_state_invalid_long"] += 1
                self._sync_debug()
                self._clear_state()
                return

            if self.state.context_bar < 0:
                hit = False
                if self.enable_asia_continuation and self._bull_context_continuation_asia(row):
                    self.state.context_bar = i
                    self.state.setup_type = "ASIA_CONTINUATION"
                    hit = True
                elif self.enable_london_continuation and self._bull_context_continuation_london(row):
                    self.state.context_bar = i
                    self.state.setup_type = "LONDON_CONTINUATION"
                    hit = True
                elif self.enable_nypm_continuation and self._bull_context_continuation_nypm(row):
                    self.state.context_bar = i
                    self.state.setup_type = "NYPM_CONTINUATION"
                    hit = True
                if hit:
                    self.debug_counts["confirm_long_context"] += 1
                    self._sync_debug()
                return

            if self.state.bridge_bar < 0:
                bridge_type = str(row.get("bridge_type_30m", ""))
                bull_cisd_ok = bool(row["bull_cisd_30m"] and row["bull_disp_30m"] and row["bull_close_strong_30m"] and self.state.setup_type.startswith("LONDON_"))
                bull_ifvg_ok = bool(row["bull_ifvg_30m"] and row["bull_disp_30m"] and row["bull_close_strong_30m"])
                bull_ok = bool(bull_cisd_ok or row["bull_mss_30m"] or bull_ifvg_ok or row["bull_c2_or_c3"])
                if bull_ok and bridge_type:
                    self.state.bridge_bar = i
                    self.state.bridge_type = bridge_type
                    self.state.bridge_low = float(row.get("bull_bridge_low_30m", float("nan")))
                    self.state.bridge_high = float(row.get("bull_bridge_high_30m", float("nan")))
                    self.state.active_until_bar = i + self.bridge_expiry_bars
                    self.debug_counts["confirm_long_bridge"] += 1
                    self._sync_debug()
                return

            if self._bull_execution_ok(row):
                self._arm_long_pullback(row, high_now, low_now, atr3, i)
                return

        if self.state.direction == "short":
            if self._bear_invalid(row):
                self.debug_counts["clear_state_invalid_short"] += 1
                self._sync_debug()
                self._clear_state()
                return

            if self.state.context_bar < 0:
                hit = False
                if self.enable_asia_continuation and self._bear_context_continuation_asia(row):
                    self.state.context_bar = i
                    self.state.setup_type = "ASIA_CONTINUATION"
                    hit = True
                elif self.enable_london_continuation and self._bear_context_continuation_london(row):
                    self.state.context_bar = i
                    self.state.setup_type = "LONDON_CONTINUATION"
                    hit = True
                elif self.enable_nyam_continuation and self._bear_context_continuation_nyam(row):
                    self.state.context_bar = i
                    self.state.setup_type = "NYAM_CONTINUATION"
                    hit = True
                elif self.enable_nypm_continuation and self._bear_context_continuation_nypm(row):
                    self.state.context_bar = i
                    self.state.setup_type = "NYPM_CONTINUATION"
                    hit = True
                if hit:
                    self.debug_counts["confirm_short_context"] += 1
                    self._sync_debug()
                return

            if self.state.bridge_bar < 0:
                bridge_type = str(row.get("bridge_type_30m", ""))
                bear_cisd_ok = bool(row["bear_cisd_30m"] and row["bear_disp_30m"] and row["bear_close_strong_30m"] and self.state.setup_type.startswith("LONDON_"))
                bear_ifvg_ok = bool(row["bear_ifvg_30m"] and row["bear_disp_30m"] and row["bear_close_strong_30m"])
                bear_ok = bool(bear_cisd_ok or row["bear_mss_30m"] or bear_ifvg_ok or row["bear_c2_or_c3"])
                if bear_ok and bridge_type:
                    self.state.bridge_bar = i
                    self.state.bridge_type = bridge_type
                    self.state.bridge_low = float(row.get("bear_bridge_low_30m", float("nan")))
                    self.state.bridge_high = float(row.get("bear_bridge_high_30m", float("nan")))
                    self.state.active_until_bar = i + self.bridge_expiry_bars
                    self.debug_counts["confirm_short_bridge"] += 1
                    self._sync_debug()
                return

            if self._bear_execution_ok(row):
                self._arm_short_pullback(row, high_now, low_now, atr3, i)
                return


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
    return type(f"ICT_MULTI_SETUP_V473_{cfg.symbol}", (V473ForcedFlatMixin, ICT_MULTI_SETUP_V452,), attrs)


def realized_points(row):
    if row.get("side") == "LONG":
        return float(row["exit_price"]) - float(row["entry_price"])
    return float(row["entry_price"]) - float(row["exit_price"])


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _is_apex_allowed_timestamp(ts) -> bool:
    if pd.isna(ts):
        return False
    t = pd.Timestamp(ts)
    cutoff_minutes = (APEX_CUTOFF_HOUR_ET * 60) + APEX_CUTOFF_MINUTE_ET
    ts_minutes = (t.hour * 60) + t.minute
    return ts_minutes <= cutoff_minutes


def _apply_apex_time_filter(meta: pd.DataFrame) -> pd.DataFrame:
    if meta.empty:
        return meta
    out = meta.copy()
    out["entry_in_apex_hours"] = out["entry_time_et_naive"].apply(_is_apex_allowed_timestamp)
    out["exit_in_apex_hours"] = out["exit_time_et_naive"].apply(_is_apex_allowed_timestamp)
    out["apex_hours_ok"] = out["entry_in_apex_hours"] & out["exit_in_apex_hours"]
    out["forced_flat_before_close_ok"] = out["exit_time_et_naive"].dt.hour < APEX_CUTOFF_HOUR_ET
    return out[out["apex_hours_ok"]].copy()


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
    meta = _apply_apex_time_filter(meta)
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
