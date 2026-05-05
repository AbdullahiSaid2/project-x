"""
Corrected top/bottom ticking management-mode backtest engine.

Purpose
-------
This file is a standalone research/backtest runner for the ICT top/bottom ticking
strategy family. It is intentionally separate from the live execution harness.

The main correction is PnL accounting:

    OLD / WRONG FOR PARTIAL EXITS:
        realized_points * dollars_per_point * configured_contracts

    CORRECT:
        realized_points * dollars_per_point * actual_closed_contracts

Backtesting.py stores each partial exit as a closed trade row. Therefore each row
must use the number of contracts closed on that row, not the original configured
position size. This engine logs `closed_size_contracts` from Backtesting.py and
uses it for all dollar PnL, win rate, profit factor, R-multiple and drawdown
calculations.

Install path:
    trading_system/src/strategies/manual/top_bottom_ticking_management_mode_backtest.py

Examples:
    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_corrected_backtest_engine \
      --prop-profile apex_50k_eval --days-back 365 --no-tail

    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_corrected_backtest_engine \
      --prop-profile apex_50k_eval --days-back 1825 --no-tail

Outputs:
    corrected trade logs, position logs, daily/monthly/yearly/variant/symbol
    summaries, portfolio summaries, portfolio compare, debug counts, and an
    accounting audit file showing old overstated PnL vs corrected PnL.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent
# When run with PYTHONPATH=. this is usually already importable. The extra paths
# make direct execution from src/strategies/manual more forgiving.
PROJECT_ROOT = ROOT.parents[2] if len(ROOT.parents) >= 3 else ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fetcher import get_ohlcv
from prop_firm_profiles import get_prop_profile, list_prop_profiles
from prop_guard import PropFirmGuard

mod = importlib.import_module("ict_top_bottom_ticking_management_modes")

BASE_TYPE2 = None
for name in ("ICTTopBottomTickingType2Baseline", "ICT_TOP_BOTTOM_TICKING_TYPE2", "ICT_TOP_BOTTOM_TICKING"):
    if hasattr(mod, name):
        BASE_TYPE2 = getattr(mod, name)
        break
if BASE_TYPE2 is None:
    raise ImportError("Could not find baseline class in ict_top_bottom_ticking.py")

BASE_TYPE1 = None
for name in ("ICTTopBottomTickingType1Sniper", "ICT_TOP_BOTTOM_TICKING_TYPE1"):
    if hasattr(mod, name):
        BASE_TYPE1 = getattr(mod, name)
        break
if BASE_TYPE1 is None:
    class ICTTopBottomTickingType1Sniper(BASE_TYPE2):
        require_internal_sweep_filter = False
        require_cos_confirmation = False
        setup_expiry_bars = 14
        limit_touch_tolerance_ticks = 2
        min_stop_points = 5.0
        max_stop_points = 34.0
    BASE_TYPE1 = ICTTopBottomTickingType1Sniper

REPORT_ACCOUNT_CASH = 50_000.0
ENGINE_CASH = 1_000_000.0
DEFAULT_DAYS_BACK = 365
DEFAULT_TAIL_ROWS = 120_000
EPS = 1e-12


@dataclass(frozen=True)
class InstrumentConfig:
    symbol: str
    exchange: str
    timeframe: str
    days_back: int
    tail_rows: Optional[int]
    contracts: int
    dollars_per_point: float


@dataclass(frozen=True)
class SymbolSpec:
    tick_size: float
    min_stop_baseline: float
    max_stop_baseline: float
    min_stop_sniper: float
    max_stop_sniper: float
    expiry_baseline: int
    expiry_sniper: int
    touch_tol_baseline_ticks: int
    touch_tol_sniper_ticks: int
    require_cos_baseline: bool
    require_cos_sniper: bool
    require_internal_baseline: bool
    require_internal_sniper: bool


INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "MNQ": InstrumentConfig("MNQ", "tradovate", "5m", DEFAULT_DAYS_BACK, DEFAULT_TAIL_ROWS, 5, 2.0),
    "MES": InstrumentConfig("MES", "tradovate", "5m", DEFAULT_DAYS_BACK, DEFAULT_TAIL_ROWS, 5, 5.0),
    "MYM": InstrumentConfig("MYM", "tradovate", "5m", DEFAULT_DAYS_BACK, DEFAULT_TAIL_ROWS, 5, 0.5),
    "MGC": InstrumentConfig("MGC", "tradovate", "5m", DEFAULT_DAYS_BACK, DEFAULT_TAIL_ROWS, 5, 10.0),
    "MCL": InstrumentConfig("MCL", "tradovate", "5m", DEFAULT_DAYS_BACK, DEFAULT_TAIL_ROWS, 5, 100.0),
}

SYMBOL_SPECS: Dict[str, SymbolSpec] = {
    "MNQ": SymbolSpec(0.25, 6.0, 30.0, 5.0, 34.0, 18, 14, 1, 2, True, False, False, False),
    "MES": SymbolSpec(0.25, 3.0, 15.0, 2.5, 16.0, 18, 14, 1, 2, True, False, False, False),
    "MYM": SymbolSpec(1.0, 20.0, 120.0, 15.0, 140.0, 18, 14, 1, 2, True, False, False, False),
    "MGC": SymbolSpec(0.1, 0.8, 6.0, 0.6, 7.0, 18, 14, 1, 2, True, False, False, False),
    "MCL": SymbolSpec(0.01, 0.05, 0.60, 0.04, 0.80, 18, 14, 1, 2, True, False, False, False),
}

VARIANTS: Dict[str, type] = {
    "type2_baseline": BASE_TYPE2,
    "type1_sniper": BASE_TYPE1,
}

MANAGEMENT_MODES = ("partial_contracts", "full_contract_trail", "no_partial")



def _as_float(value, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _is_finite(*values: float) -> bool:
    return all(math.isfinite(_as_float(v)) for v in values)


def to_et(ts):
    if pd.isna(ts):
        return pd.NaT
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("America/New_York")


def realized_points_from_prices(side: str, entry_price: float, exit_price: float) -> float:
    side = str(side).upper()
    if side == "LONG":
        return float(exit_price) - float(entry_price)
    return float(entry_price) - float(exit_price)


def _trade_day_et(exit_time) -> object:
    exit_et = to_et(pd.Timestamp(str(exit_time)))
    if pd.isna(exit_et):
        return None
    return exit_et.tz_localize(None).date()


def _build_guarded_strategy_class(
    cfg: InstrumentConfig,
    variant_name: str,
    base_cls: type,
    prop_profile_name: str,
    management_mode: str,
    target1_r: float,
    target2_r: float,
    target3_r: float,
    no_partial_target_r: float,
) -> type:
    """Build a strategy wrapper with corrected guard PnL and safe bracket orders."""

    spec = SYMBOL_SPECS[cfg.symbol]
    profile = get_prop_profile(prop_profile_name)
    is_sniper = variant_name == "type1_sniper"
    use_guard = prop_profile_name != "none"

    class CorrectedStrategy(base_cls):
        fixed_size = cfg.contracts
        trade_management_mode = management_mode
        # Target R values are assigned after class creation.
        # Python class bodies cannot safely reference outer variables with the same names.
        tick_size = spec.tick_size
        setup_expiry_bars = spec.expiry_sniper if is_sniper else spec.expiry_baseline
        limit_touch_tolerance_ticks = spec.touch_tol_sniper_ticks if is_sniper else spec.touch_tol_baseline_ticks
        require_cos_confirmation = spec.require_cos_sniper if is_sniper else spec.require_cos_baseline
        require_internal_sweep_filter = spec.require_internal_sniper if is_sniper else spec.require_internal_baseline
        min_stop_points = spec.min_stop_sniper if is_sniper else spec.min_stop_baseline
        max_stop_points = spec.max_stop_sniper if is_sniper else spec.max_stop_baseline
        last_trade_log: List[dict] = []
        last_debug_counts: dict = {}

        def init(self):
            # Must exist before base init in case base init or future subclasses call _sync_debug().
            self.prop_guard = PropFirmGuard(profile) if use_guard else None
            self._guard_seen_closed = 0
            super().init()
            self.debug_counts.setdefault("blocked_prop_daily_loss", 0)
            self.debug_counts.setdefault("blocked_prop_consecutive_losses", 0)
            self.debug_counts.setdefault("blocked_prop_max_trades", 0)
            self.debug_counts.setdefault("blocked_prop_trailing_drawdown", 0)
            self.debug_counts.setdefault("reject_invalid_bracket", 0)
            self.debug_counts.setdefault("reject_order_error", 0)
            self.debug_counts.setdefault("guard_realized_pnl_dollars", 0.0)
            self.__class__.last_trade_log = []
            self.__class__.last_debug_counts = {}
            self._sync_debug()

        def _sync_debug(self):
            if not hasattr(self, "debug_counts"):
                return
            if getattr(self, "prop_guard", None) is not None:
                self.debug_counts["prop_balance"] = float(self.prop_guard.balance)
                self.debug_counts["prop_day_realized"] = float(self.prop_guard.day_realized)
                self.debug_counts["prop_consecutive_losses_today"] = int(self.prop_guard.consecutive_losses_today)
            else:
                self.debug_counts["prop_balance"] = np.nan
                self.debug_counts["prop_day_realized"] = 0.0
                self.debug_counts["prop_consecutive_losses_today"] = 0
            self.__class__.last_debug_counts = dict(self.debug_counts)

        def _log_newly_closed_trades(self):
            """Log closed rows with actual closed size.

            Backtesting.py creates one closed trade per partial exit, so t.size is the
            number of contracts closed on that row. That is the source of truth for
            correct futures-dollar PnL.
            """
            try:
                closed = list(self.closed_trades)
            except Exception:
                return
            if len(closed) <= self.prev_closed_count:
                return

            for t in closed[self.prev_closed_count:]:
                meta = self.open_trade_meta or {}
                signed_size = _as_float(getattr(t, "size", np.nan))
                side = "LONG" if signed_size > 0 else "SHORT"
                self.__class__.last_trade_log.append(
                    {
                        "side": side,
                        "setup_type": meta.get("setup_type", ""),
                        "entry_variant": meta.get("entry_variant", ""),
                        "entry_price": float(t.entry_price),
                        "exit_price": float(t.exit_price),
                        "entry_time": str(t.entry_time),
                        "exit_time": str(t.exit_time),
                        "closed_size_contracts": abs(signed_size) if math.isfinite(signed_size) else np.nan,
                        "signed_size_contracts": signed_size,
                        "engine_pnl_points_contracts": float(getattr(t, "pl", np.nan)),
                        "return_pct": float(getattr(t, "pl_pct", np.nan)) if hasattr(t, "pl_pct") else np.nan,
                        "signal_status": "CLOSED",
                        **meta,
                    }
                )

            if len(closed) > self.prev_closed_count and not self.position:
                self.open_trade_meta = None
            self.prev_closed_count = len(closed)

        def _update_guard_from_closed_trades(self):
            if getattr(self, "prop_guard", None) is None:
                return
            try:
                closed = list(self.closed_trades)
            except Exception:
                return
            if len(closed) <= self._guard_seen_closed:
                return

            for t in closed[self._guard_seen_closed:]:
                # Correct guard PnL: Backtesting.py t.pl already includes actual closed size.
                pnl_dollars = _as_float(getattr(t, "pl", np.nan), 0.0) * cfg.dollars_per_point
                trade_day = _trade_day_et(getattr(t, "exit_time", None))
                self.prop_guard.on_trade_closed(pnl_dollars, trade_day)
                self.debug_counts["guard_realized_pnl_dollars"] = (
                    float(self.debug_counts.get("guard_realized_pnl_dollars", 0.0)) + float(pnl_dollars)
                )
            self._guard_seen_closed = len(closed)
            self._sync_debug()

        def _guard_allows_entry(self, row: pd.Series) -> bool:
            if getattr(self, "prop_guard", None) is None:
                return True
            trade_day = row.get("session_date", row.get("et_date"))
            decision = self.prop_guard.can_open_trade(trade_day)
            if decision.allowed:
                return True
            key = f"blocked_prop_{decision.reason}"
            self.debug_counts[key] = self.debug_counts.get(key, 0) + 1
            self._sync_debug()
            self._clear_pending()
            return False

        def _safe_enter_short(self, row: pd.Series):
            entry = min(float(row["Close"]), float(self.pending.entry_ce))
            stop = float(self.pending.stop_price)
            risk = stop - entry
            if not _is_finite(entry, stop, risk) or risk <= 0:
                self.debug_counts["reject_invalid_bracket"] += 1
                self._clear_pending()
                self._sync_debug()
                return

            target1 = entry - (risk * float(self.target1_r))
            target2 = entry - (risk * float(self.target2_r))
            target3 = entry - (risk * float(self._active_target3_r() if hasattr(self, "_active_target3_r") else self.target3_r))
            if not _is_finite(target1, target2, target3) or not (target3 < entry < stop):
                self.debug_counts["reject_invalid_bracket"] += 1
                self._clear_pending()
                self._sync_debug()
                return

            self.pending.entry_ce = entry
            self.pending.target1 = target1
            self.pending.target2 = target2
            self.pending.target3 = target3
            self.active_risk = risk
            try:
                self.sell(size=self.fixed_size, sl=stop, tp=target3)
            except Exception:
                self.debug_counts["reject_order_error"] += 1
                self._clear_pending()
                self._sync_debug()
                return
            self.partial1_taken = False
            self.partial2_taken = False
            self.be_moved = False
            self._record_open_trade_meta(entry, row)
            self.debug_counts["entry_short"] += 1
            self._sync_debug()
            self._clear_pending()

        def _safe_enter_long(self, row: pd.Series):
            entry = max(float(row["Close"]), float(self.pending.entry_ce))
            stop = float(self.pending.stop_price)
            risk = entry - stop
            if not _is_finite(entry, stop, risk) or risk <= 0:
                self.debug_counts["reject_invalid_bracket"] += 1
                self._clear_pending()
                self._sync_debug()
                return

            target1 = entry + (risk * float(self.target1_r))
            target2 = entry + (risk * float(self.target2_r))
            target3 = entry + (risk * float(self._active_target3_r() if hasattr(self, "_active_target3_r") else self.target3_r))
            if not _is_finite(target1, target2, target3) or not (stop < entry < target3):
                self.debug_counts["reject_invalid_bracket"] += 1
                self._clear_pending()
                self._sync_debug()
                return

            self.pending.entry_ce = entry
            self.pending.target1 = target1
            self.pending.target2 = target2
            self.pending.target3 = target3
            self.active_risk = risk
            try:
                self.buy(size=self.fixed_size, sl=stop, tp=target3)
            except Exception:
                self.debug_counts["reject_order_error"] += 1
                self._clear_pending()
                self._sync_debug()
                return
            self.partial1_taken = False
            self.partial2_taken = False
            self.be_moved = False
            self._record_open_trade_meta(entry, row)
            self.debug_counts["entry_long"] += 1
            self._sync_debug()
            self._clear_pending()

        def _enter_short(self, row: pd.Series, i: int):
            self._update_guard_from_closed_trades()
            if not self._guard_allows_entry(row):
                return
            return self._safe_enter_short(row)

        def _enter_long(self, row: pd.Series, i: int):
            self._update_guard_from_closed_trades()
            if not self._guard_allows_entry(row):
                return
            return self._safe_enter_long(row)

        def next(self):
            self._update_guard_from_closed_trades()
            return super().next()

    # Assign target R values after class creation to avoid class-body scope shadowing.
    CorrectedStrategy.target1_r = float(target1_r)
    CorrectedStrategy.target2_r = float(target2_r)
    CorrectedStrategy.target3_r = float(target3_r)
    CorrectedStrategy.no_partial_target_r = float(no_partial_target_r)

    CorrectedStrategy.__name__ = f"{base_cls.__name__}_{cfg.symbol}_{variant_name}_{prop_profile_name}_{management_mode}_corrected"
    return CorrectedStrategy


def _load_data(cfg: InstrumentConfig) -> pd.DataFrame:
    df = get_ohlcv(cfg.symbol, exchange=cfg.exchange, timeframe=cfg.timeframe, days_back=cfg.days_back)
    if cfg.tail_rows is not None:
        df = df.tail(cfg.tail_rows)
    return df


def _position_key_columns(df: pd.DataFrame) -> List[str]:
    wanted = [
        "symbol",
        "variant",
        "prop_profile",
        "side",
        "entry_time",
        "entry_price",
        "planned_stop_price",
        "setup_type",
        "entry_variant",
    ]
    return [c for c in wanted if c in df.columns]


def prepare_trade_log(meta: pd.DataFrame, cfg: InstrumentConfig, variant_name: str, prop_profile_name: str, management_mode: str) -> pd.DataFrame:
    if meta.empty:
        return meta

    out = meta.copy()
    out["variant"] = variant_name
    out["symbol"] = cfg.symbol
    out["exchange"] = cfg.exchange
    out["timeframe"] = cfg.timeframe
    out["prop_profile"] = prop_profile_name
    out["management_mode"] = out.get("trade_management_mode", management_mode)
    out["configured_contracts"] = cfg.contracts
    out["dollars_per_point"] = cfg.dollars_per_point
    out["report_account_cash"] = REPORT_ACCOUNT_CASH

    # Normalize original / older column names.
    if "pnl" in out.columns and "engine_pnl_points_contracts" not in out.columns:
        out["engine_pnl_points_contracts"] = pd.to_numeric(out["pnl"], errors="coerce")
    if "engine_pnl_points_contracts" not in out.columns:
        out["engine_pnl_points_contracts"] = np.nan

    out["entry_time"] = pd.to_datetime(out.get("entry_time"), errors="coerce")
    out["exit_time"] = pd.to_datetime(out.get("exit_time"), errors="coerce")
    out["entry_time_et"] = out["entry_time"].apply(to_et)
    out["exit_time_et"] = out["exit_time"].apply(to_et)
    out["entry_time_et_naive"] = pd.to_datetime(out["entry_time_et"], errors="coerce").dt.tz_localize(None)
    out["exit_time_et_naive"] = pd.to_datetime(out["exit_time_et"], errors="coerce").dt.tz_localize(None)
    out["calendar_exit_date_et"] = out["exit_time_et_naive"].dt.date
    out["exit_month_et"] = out["exit_time_et_naive"].dt.to_period("M").astype(str)
    out["exit_year_et"] = out["exit_time_et_naive"].dt.year

    out["entry_price"] = pd.to_numeric(out["entry_price"], errors="coerce")
    out["exit_price"] = pd.to_numeric(out["exit_price"], errors="coerce")
    out["planned_stop_price"] = pd.to_numeric(out.get("planned_stop_price"), errors="coerce")
    out["side"] = out["side"].astype(str).str.upper()
    out["realized_points"] = [
        realized_points_from_prices(side, entry, exit_)
        for side, entry, exit_ in zip(out["side"], out["entry_price"], out["exit_price"])
    ]

    # Source of truth for actual closed size:
    # 1) logged Backtesting.py t.size if available,
    # 2) infer from t.pl / realized_points for older logs.
    if "closed_size_contracts" in out.columns:
        out["closed_size_contracts"] = pd.to_numeric(out["closed_size_contracts"], errors="coerce").abs()
    else:
        out["closed_size_contracts"] = np.nan

    inferred_size = np.where(
        out["realized_points"].abs() > EPS,
        (pd.to_numeric(out["engine_pnl_points_contracts"], errors="coerce") / out["realized_points"]).abs(),
        np.nan,
    )
    out["inferred_closed_size_contracts"] = inferred_size
    out["closed_size_contracts"] = out["closed_size_contracts"].fillna(out["inferred_closed_size_contracts"])

    # Correct futures-dollar PnL. This is the canonical field.
    out["realized_pnl_dollars"] = out["realized_points"] * out["dollars_per_point"] * out["closed_size_contracts"]

    # Equivalent check using Backtesting.py t.pl. Difference should be approximately 0.
    out["engine_dollarized_pnl_check"] = pd.to_numeric(out["engine_pnl_points_contracts"], errors="coerce") * out["dollars_per_point"]
    out["pnl_math_diff_dollars"] = out["realized_pnl_dollars"] - out["engine_dollarized_pnl_check"]

    # Legacy metric retained only as an audit column. Do NOT use as final PnL.
    out["legacy_full_size_pnl_dollars"] = out["realized_points"] * out["dollars_per_point"] * out["configured_contracts"]
    out["legacy_overstatement_dollars"] = out["legacy_full_size_pnl_dollars"] - out["realized_pnl_dollars"]

    stop_points = np.where(
        out["side"] == "LONG",
        out["entry_price"] - out["planned_stop_price"],
        out["planned_stop_price"] - out["entry_price"],
    )
    out["initial_stop_points"] = stop_points
    out["risk_dollars_for_closed_piece"] = out["initial_stop_points"] * out["dollars_per_point"] * out["closed_size_contracts"]
    out["r_multiple_exit"] = np.where(
        out["risk_dollars_for_closed_piece"].abs() > EPS,
        out["realized_pnl_dollars"] / out["risk_dollars_for_closed_piece"],
        np.nan,
    )
    out["realized_return_pct_on_50k"] = (out["realized_pnl_dollars"] / REPORT_ACCOUNT_CASH) * 100.0
    out["is_win_exit"] = out["realized_pnl_dollars"] > 0
    out["is_loss_exit"] = out["realized_pnl_dollars"] < 0

    key_cols = _position_key_columns(out)
    out["position_key"] = out.groupby(key_cols, dropna=False).ngroup().astype(int) + 1 if key_cols else np.arange(len(out)) + 1

    # Sort for deterministic downstream equity curves.
    out = out.sort_values(["exit_time_et_naive", "symbol", "variant", "position_key"], na_position="last").reset_index(drop=True)
    return out


def build_position_log(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["position_key"]
    for key, grp in trades.groupby(group_cols, dropna=False):
        grp = grp.sort_values("exit_time_et_naive")
        first = grp.iloc[0]
        last = grp.iloc[-1]
        position_contracts = float(grp["closed_size_contracts"].sum())
        risk_points = float(first.get("initial_stop_points", np.nan))
        dpp = float(first.get("dollars_per_point", np.nan))
        initial_risk = risk_points * dpp * position_contracts if np.isfinite(risk_points * dpp * position_contracts) else np.nan
        realized_pnl = float(grp["realized_pnl_dollars"].sum())
        rows.append(
            {
                "position_key": int(key if not isinstance(key, tuple) else key[0]),
                "symbol": first.get("symbol"),
                "variant": first.get("variant"),
                "prop_profile": first.get("prop_profile"),
                "side": first.get("side"),
                "setup_type": first.get("setup_type"),
                "entry_variant": first.get("entry_variant"),
                "entry_time": first.get("entry_time"),
                "entry_time_et": first.get("entry_time_et"),
                "final_exit_time": last.get("exit_time"),
                "final_exit_time_et": last.get("exit_time_et"),
                "calendar_exit_date_et": last.get("calendar_exit_date_et"),
                "exit_month_et": last.get("exit_month_et"),
                "exit_year_et": last.get("exit_year_et"),
                "entry_price": first.get("entry_price"),
                "planned_stop_price": first.get("planned_stop_price"),
                "initial_stop_points": risk_points,
                "closed_exits": int(len(grp)),
                "position_contracts_closed": position_contracts,
                "realized_pnl_dollars": realized_pnl,
                "initial_risk_dollars": initial_risk,
                "r_multiple_position": realized_pnl / initial_risk if np.isfinite(initial_risk) and abs(initial_risk) > EPS else np.nan,
                "is_win_position": realized_pnl > 0,
                "is_loss_position": realized_pnl < 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["final_exit_time_et", "symbol", "variant"], na_position="last").reset_index(drop=True)


def _profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    if losses <= EPS:
        return np.nan
    return float(wins / losses)


def _max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = pnl.fillna(0).cumsum()
    peak = equity.cummax().clip(lower=0)
    dd = equity - peak
    return float(dd.min()) if not dd.empty else 0.0


def _max_consecutive_losses(pnl: Iterable[float]) -> int:
    best = 0
    cur = 0
    for value in pnl:
        if value < 0:
            cur += 1
            best = max(best, cur)
        elif value > 0:
            cur = 0
    return best


def _summary_from_pnl(df: pd.DataFrame, label: str, unit_name: str, pnl_col: str = "realized_pnl_dollars") -> dict:
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0) if not df.empty else pd.Series(dtype=float)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    breakeven = int((pnl == 0).sum())
    total = float(pnl.sum()) if not pnl.empty else 0.0
    gross_profit = float(pnl[pnl > 0].sum()) if not pnl.empty else 0.0
    gross_loss = float(pnl[pnl < 0].sum()) if not pnl.empty else 0.0
    return {
        "scope": label,
        "unit": unit_name,
        "count": int(len(df)),
        "realized_pnl_dollars": total,
        "gross_profit_dollars": gross_profit,
        "gross_loss_dollars": gross_loss,
        "net_return_pct_on_50k": (total / REPORT_ACCOUNT_CASH) * 100.0,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate_pct": (wins / len(df) * 100.0) if len(df) else 0.0,
        "loss_rate_pct": (losses / len(df) * 100.0) if len(df) else 0.0,
        "profit_factor": _profit_factor(pnl),
        "expectancy_dollars": float(pnl.mean()) if len(pnl) else 0.0,
        "median_pnl_dollars": float(pnl.median()) if len(pnl) else 0.0,
        "avg_win_dollars": float(pnl[pnl > 0].mean()) if wins else 0.0,
        "avg_loss_dollars": float(pnl[pnl < 0].mean()) if losses else 0.0,
        "payoff_ratio": (float(pnl[pnl > 0].mean()) / abs(float(pnl[pnl < 0].mean()))) if wins and losses else np.nan,
        "best_unit_dollars": float(pnl.max()) if len(pnl) else np.nan,
        "worst_unit_dollars": float(pnl.min()) if len(pnl) else np.nan,
        "max_drawdown_dollars": _max_drawdown(pnl),
        "max_drawdown_pct_on_50k": (_max_drawdown(pnl) / REPORT_ACCOUNT_CASH) * 100.0,
        "max_consecutive_losses": _max_consecutive_losses(pnl.tolist()),
    }


def build_portfolio_summary(trades: pd.DataFrame, positions: pd.DataFrame, label: str) -> pd.DataFrame:
    trade_row = _summary_from_pnl(trades, label, "closed_exit_rows")
    pos_row = _summary_from_pnl(positions, label, "parent_positions")
    if not trades.empty:
        ordered = trades.sort_values("exit_time_et_naive")
        daily = ordered.groupby("calendar_exit_date_et", dropna=False)["realized_pnl_dollars"].sum()
        monthly = ordered.groupby("exit_month_et", dropna=False)["realized_pnl_dollars"].sum()
        yearly = ordered.groupby("exit_year_et", dropna=False)["realized_pnl_dollars"].sum()
        for row in (trade_row, pos_row):
            row.update(
                {
                    "best_day_dollars": float(daily.max()) if not daily.empty else np.nan,
                    "worst_day_dollars": float(daily.min()) if not daily.empty else np.nan,
                    "best_month_dollars": float(monthly.max()) if not monthly.empty else np.nan,
                    "worst_month_dollars": float(monthly.min()) if not monthly.empty else np.nan,
                    "best_year_dollars": float(yearly.max()) if not yearly.empty else np.nan,
                    "worst_year_dollars": float(yearly.min()) if not yearly.empty else np.nan,
                }
            )
    return pd.DataFrame([trade_row, pos_row])


def _group_summary(df: pd.DataFrame, group_cols: List[str], unit_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(_summary_from_pnl(grp, "group", unit_name))
        row.pop("scope", None)
        row.pop("unit", None)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_daily_summary(trades: pd.DataFrame) -> pd.DataFrame:
    return _group_summary(trades, ["calendar_exit_date_et"], "closed_exit_rows")


def build_monthly_summary(trades: pd.DataFrame) -> pd.DataFrame:
    return _group_summary(trades, ["exit_month_et"], "closed_exit_rows")


def build_yearly_summary(trades: pd.DataFrame) -> pd.DataFrame:
    return _group_summary(trades, ["exit_year_et"], "closed_exit_rows")


def build_symbol_summary(positions: pd.DataFrame) -> pd.DataFrame:
    return _group_summary(positions, ["symbol"], "parent_positions")


def build_variant_summary(positions: pd.DataFrame) -> pd.DataFrame:
    return _group_summary(positions, ["variant"], "parent_positions")


def build_symbol_variant_summary(positions: pd.DataFrame) -> pd.DataFrame:
    return _group_summary(positions, ["symbol", "variant"], "parent_positions")


def build_audit_summary(trades: pd.DataFrame, label: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([
            {
                "scope": label,
                "rows": 0,
                "correct_realized_pnl_dollars": 0.0,
                "legacy_full_size_pnl_dollars": 0.0,
                "legacy_overstatement_dollars": 0.0,
                "engine_dollarized_pnl_check": 0.0,
                "max_abs_math_diff_dollars": 0.0,
                "rows_missing_closed_size": 0,
            }
        ])
    return pd.DataFrame([
        {
            "scope": label,
            "rows": int(len(trades)),
            "correct_realized_pnl_dollars": float(trades["realized_pnl_dollars"].sum()),
            "legacy_full_size_pnl_dollars": float(trades["legacy_full_size_pnl_dollars"].sum()),
            "legacy_overstatement_dollars": float(trades["legacy_overstatement_dollars"].sum()),
            "engine_dollarized_pnl_check": float(trades["engine_dollarized_pnl_check"].sum()),
            "max_abs_math_diff_dollars": float(trades["pnl_math_diff_dollars"].abs().max()),
            "rows_missing_closed_size": int(trades["closed_size_contracts"].isna().sum()),
            "partial_exit_rows_less_than_configured": int((trades["closed_size_contracts"] < trades["configured_contracts"]).sum()),
            "full_exit_rows_equal_configured": int(np.isclose(trades["closed_size_contracts"], trades["configured_contracts"]).sum()),
        }
    ])


def run_symbol_variant(
    cfg: InstrumentConfig,
    variant_name: str,
    base_cls: type,
    prop_profile_name: str,
    management_mode: str,
    target1_r: float,
    target2_r: float,
    target3_r: float,
    no_partial_target_r: float,
):
    StrategyCls = _build_guarded_strategy_class(
        cfg, variant_name, base_cls, prop_profile_name, management_mode, target1_r, target2_r, target3_r, no_partial_target_r
    )
    print(f"\n=== {cfg.symbol} | {variant_name} | {prop_profile_name} | {management_mode} ===")
    df = _load_data(cfg)
    tail_label = "none" if cfg.tail_rows is None else str(cfg.tail_rows)
    print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()} | days_back={cfg.days_back} | tail_rows={tail_label}")
    if len(df) == 0:
        return None, pd.DataFrame(), pd.DataFrame()
    bt = Backtest(df, StrategyCls, cash=ENGINE_CASH, commission=0.0, exclusive_orders=True, trade_on_close=False)
    stats = bt.run()
    meta = pd.DataFrame(getattr(StrategyCls, "last_trade_log", []))
    meta = prepare_trade_log(meta, cfg, variant_name, prop_profile_name, management_mode)
    debug = pd.DataFrame([getattr(StrategyCls, "last_debug_counts", {})])
    if not debug.empty:
        debug.insert(0, "variant", variant_name)
        debug.insert(1, "symbol", cfg.symbol)
        debug.insert(2, "prop_profile", prop_profile_name)
        debug.insert(3, "management_mode", management_mode)
        debug.insert(4, "days_back", cfg.days_back)
        debug.insert(5, "tail_rows", tail_label)
    return stats, meta, debug


def run_all_symbols(
    symbols: List[str],
    variants: List[str],
    prop_profile_name: str,
    instrument_overrides: Dict[str, InstrumentConfig],
    management_mode: str,
    target1_r: float,
    target2_r: float,
    target3_r: float,
    no_partial_target_r: float,
):
    metas: List[pd.DataFrame] = []
    debugs: List[pd.DataFrame] = []
    stats_by_variant_symbol: Dict[Tuple[str, str], object] = {}
    for sym in symbols:
        if sym not in instrument_overrides:
            raise KeyError(f"Unknown symbol '{sym}'. Available: {', '.join(instrument_overrides)}")
        cfg = instrument_overrides[sym]
        for variant in variants:
            if variant not in VARIANTS:
                raise KeyError(f"Unknown variant '{variant}'. Available: {', '.join(VARIANTS)}")
            stats, meta, debug = run_symbol_variant(
                cfg, variant, VARIANTS[variant], prop_profile_name, management_mode, target1_r, target2_r, target3_r, no_partial_target_r
            )
            stats_by_variant_symbol[(variant, sym)] = stats
            if stats is not None:
                print(f"{variant} | {sym} engine closed rows={stats.get('# Trades', np.nan)} corrected_pnl=${meta['realized_pnl_dollars'].sum() if not meta.empty else 0.0:.2f}")
            if not meta.empty:
                metas.append(meta)
            if not debug.empty:
                debugs.append(debug)
    combined = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
    debug_df = pd.concat(debugs, ignore_index=True) if debugs else pd.DataFrame()
    return stats_by_variant_symbol, combined, debug_df


def _suffix_for_args(prop_profile: str, days_back: int, tail_rows: Optional[int], management_mode: str) -> str:
    tail_tag = "notail" if tail_rows is None else f"tail{tail_rows}"
    return f"_{prop_profile}_{management_mode}_{days_back}d_{tail_tag}_corrected"


def _write_df(path: Path, df: pd.DataFrame, outputs: dict, key: str):
    df.to_csv(path, index=False)
    outputs[key] = path


def write_report_set(trades: pd.DataFrame, prefix: str, suffix: str) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    positions = build_position_log(trades)
    base = f"top_bottom_ticking_{prefix}{suffix}"
    _write_df(ROOT / f"{base}_trade_log.csv", trades, outputs, "trades")
    _write_df(ROOT / f"{base}_position_log.csv", positions, outputs, "positions")
    _write_df(ROOT / f"{base}_daily_summary.csv", build_daily_summary(trades), outputs, "daily")
    _write_df(ROOT / f"{base}_monthly_summary.csv", build_monthly_summary(trades), outputs, "monthly")
    _write_df(ROOT / f"{base}_yearly_summary.csv", build_yearly_summary(trades), outputs, "yearly")
    _write_df(ROOT / f"{base}_symbol_summary.csv", build_symbol_summary(positions), outputs, "symbol")
    _write_df(ROOT / f"{base}_variant_summary.csv", build_variant_summary(positions), outputs, "variant")
    _write_df(ROOT / f"{base}_symbol_variant_summary.csv", build_symbol_variant_summary(positions), outputs, "symbol_variant")
    _write_df(ROOT / f"{base}_portfolio_summary.csv", build_portfolio_summary(trades, positions, prefix), outputs, "portfolio")
    _write_df(ROOT / f"{base}_accounting_audit.csv", build_audit_summary(trades, prefix), outputs, "audit")
    return outputs


def _parse_csv_arg(value: str, allowed: Iterable[str], label: str) -> List[str]:
    allowed_set = set(allowed)
    if value == "all":
        return list(allowed)
    items = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in items if x not in allowed_set]
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}. Available: {sorted(allowed_set)}")
    return items


def _apply_contract_overrides(instruments: Dict[str, InstrumentConfig], override_text: str) -> Dict[str, InstrumentConfig]:
    if not override_text:
        return instruments
    updated = dict(instruments)
    for item in [x.strip() for x in override_text.split(",") if x.strip()]:
        if ":" not in item:
            raise ValueError(f"Bad contract override '{item}'. Expected SYMBOL:CONTRACTS, e.g. MNQ:3")
        sym, qty = item.split(":", 1)
        sym = sym.strip().upper()
        if sym not in updated:
            raise ValueError(f"Unknown contract override symbol '{sym}'. Available: {sorted(updated)}")
        contracts = int(qty)
        if contracts <= 0:
            raise ValueError(f"Contracts must be positive for {sym}")
        base = updated[sym]
        updated[sym] = InstrumentConfig(base.symbol, base.exchange, base.timeframe, base.days_back, base.tail_rows, contracts, base.dollars_per_point)
    return updated


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Corrected ICT top/bottom ticking backtest engine with proper partial-exit PnL accounting.")
    parser.add_argument("--prop-profile", default="apex_50k_eval", choices=list_prop_profiles(), help="Filtered/live-style prop profile.")
    parser.add_argument("--symbols", default="all", help="Comma-separated symbols or 'all'.")
    parser.add_argument("--variants", default="all", help="Comma-separated variants or 'all'.")
    parser.add_argument("--days-back", type=int, default=365)
    parser.add_argument("--tail-rows", type=int, default=120_000)
    parser.add_argument("--no-tail", action="store_true")
    parser.add_argument("--management-mode", default="partial_contracts", choices=["all", *MANAGEMENT_MODES], help="Trade management mode to test.")
    parser.add_argument("--target1-r", type=float, default=1.0)
    parser.add_argument("--target2-r", type=float, default=2.5)
    parser.add_argument("--target3-r", type=float, default=4.0)
    parser.add_argument("--no-partial-target-r", type=float, default=2.0)
    parser.add_argument("--contract-overrides", default="", help="Optional per-symbol contract overrides, e.g. MNQ:3,MGC:2. Default keeps current per-symbol config.")
    parser.add_argument("--raw-only", action="store_true", help="Only run raw/no-prop profile.")
    parser.add_argument("--filtered-only", action="store_true", help="Only run the selected prop profile.")
    parser.add_argument("--list-profiles", action="store_true")
    args = parser.parse_args(argv)

    if args.list_profiles:
        for name in list_prop_profiles():
            print(name)
        return 0
    if args.raw_only and args.filtered_only:
        raise SystemExit("Use either --raw-only or --filtered-only, not both.")

    symbols = _parse_csv_arg(args.symbols, INSTRUMENTS.keys(), "symbols")
    variants = _parse_csv_arg(args.variants, VARIANTS.keys(), "variants")
    tail_rows = None if args.no_tail else args.tail_rows

    instrument_overrides = {
        sym: InstrumentConfig(
            base.symbol,
            base.exchange,
            base.timeframe,
            args.days_back,
            tail_rows,
            base.contracts,
            base.dollars_per_point,
        )
        for sym, base in INSTRUMENTS.items()
    }
    instrument_overrides = _apply_contract_overrides(instrument_overrides, args.contract_overrides)

    modes = list(MANAGEMENT_MODES) if args.management_mode == "all" else [args.management_mode]
    all_compare_rows = []

    for management_mode in modes:
        print(f"\n######## MANAGEMENT MODE: {management_mode} ########")
        suffix = _suffix_for_args(args.prop_profile, args.days_back, tail_rows, management_mode)
        report_sets = []
        debug_frames = []

        if not args.filtered_only:
            _, raw_trades, raw_debug = run_all_symbols(
                symbols, variants, "none", instrument_overrides, management_mode,
                args.target1_r, args.target2_r, args.target3_r, args.no_partial_target_r,
            )
            if not raw_trades.empty:
                outputs = write_report_set(raw_trades, "raw", suffix)
                report_sets.append(("raw", raw_trades, outputs))
                print(f"\nRaw corrected realized PnL [{management_mode}]: ${raw_trades['realized_pnl_dollars'].sum():.2f}")
            else:
                print("\nNo raw trades produced.")
            if not raw_debug.empty:
                raw_debug.insert(0, "scope", "raw")
                debug_frames.append(raw_debug)

        if not args.raw_only:
            _, filtered_trades, filtered_debug = run_all_symbols(
                symbols, variants, args.prop_profile, instrument_overrides, management_mode,
                args.target1_r, args.target2_r, args.target3_r, args.no_partial_target_r,
            )
            if not filtered_trades.empty:
                outputs = write_report_set(filtered_trades, "filtered", suffix)
                report_sets.append(("filtered", filtered_trades, outputs))
                print(f"\nFiltered corrected realized PnL [{management_mode}]: ${filtered_trades['realized_pnl_dollars'].sum():.2f}")
            else:
                print("\nNo filtered trades produced.")
            if not filtered_debug.empty:
                filtered_debug.insert(0, "scope", "filtered")
                debug_frames.append(filtered_debug)

        if debug_frames:
            debug_df = pd.concat(debug_frames, ignore_index=True)
            debug_path = ROOT / f"top_bottom_ticking_debug_counts{suffix}.csv"
            debug_df.to_csv(debug_path, index=False)
            print(f"Saved debug -> {debug_path}")

        if report_sets:
            for label, trades, _ in report_sets:
                positions = build_position_log(trades)
                row = build_portfolio_summary(trades, positions, label)
                row.insert(0, "management_mode", management_mode)
                all_compare_rows.append(row)

        if len(report_sets) >= 2:
            portfolio_rows = []
            audit_rows = []
            for label, trades, _ in report_sets:
                positions = build_position_log(trades)
                p = build_portfolio_summary(trades, positions, label)
                p.insert(0, "management_mode", management_mode)
                portfolio_rows.append(p)
                a = build_audit_summary(trades, label)
                a.insert(0, "management_mode", management_mode)
                audit_rows.append(a)
            compare = pd.concat(portfolio_rows, ignore_index=True)
            audit = pd.concat(audit_rows, ignore_index=True)
            compare_path = ROOT / f"top_bottom_ticking_portfolio_compare{suffix}.csv"
            audit_path = ROOT / f"top_bottom_ticking_accounting_audit_compare{suffix}.csv"
            compare.to_csv(compare_path, index=False)
            audit.to_csv(audit_path, index=False)
            print(f"Saved portfolio compare -> {compare_path}")
            print(f"Saved accounting audit compare -> {audit_path}")

        print(f"\nFinished corrected backtest run for {management_mode}.")
        for label, _, outputs in report_sets:
            print(f"{label} outputs:")
            for key, path in outputs.items():
                print(f"  {key}: {path}")

    if len(modes) > 1 and all_compare_rows:
        mode_compare = pd.concat(all_compare_rows, ignore_index=True)
        tail_tag = "notail" if tail_rows is None else f"tail{tail_rows}"
        mode_compare_path = ROOT / f"top_bottom_ticking_management_mode_compare_{args.prop_profile}_{args.days_back}d_{tail_tag}_corrected.csv"
        mode_compare.to_csv(mode_compare_path, index=False)
        print(f"\nSaved management mode compare -> {mode_compare_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
