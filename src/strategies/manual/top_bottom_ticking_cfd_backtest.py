"""
Algotec CFD backtester for ict_top_bottom_ticking.

Place at:
    trading_system/src/strategies/manual/top_bottom_ticking_cfd_backtest.py

Required companion files:
    trading_system/src/strategies/manual/cfd_instruments.py
    trading_system/src/strategies/manual/cfd_restrictions.py
    trading_system/src/exchanges/mt5_cfd_connector.py
    trading_system/src/strategies/manual/ict_top_bottom_ticking.py
    trading_system/src/strategies/manual/prop_firm_profiles.py
    trading_system/src/strategies/manual/prop_guard.py

Purpose:
- Keep CFD backtesting separate from futures backtesting.
- Use CFD symbols and CFD lot sizing.
- Pull CFD candles from MT5/cache.
- Reuse the existing prop profile / prop guard system so prop firms are swappable.
- Produce raw and prop-filtered CFD trade logs/summaries.

Examples:
    # List prop profiles from your existing prop_firm_profiles.py
    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_cfd_backtest --list-profiles

    # Cached CFD backtest with no prop filter
    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_cfd_backtest \
        --prop-profile none --days-back 365 --no-tail --source cache

    # Prop-filtered CFD backtest, pulling MT5 candles and exporting reports
    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_cfd_backtest \
        --prop-profile ftmo_50k_standard --days-back 365 --no-tail --source mt5 --refresh-data \
        --symbol-specs-json src/data/mt5_cfd_cache/cfd_symbol_specs.json
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2] if len(ROOT.parents) >= 3 else ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfd_instruments import CFD_INSTRUMENTS, CFDInstrumentConfig, with_runtime_overrides
from cfd_restrictions import RestrictedNewsCalendar, is_late_friday_cutoff
from prop_firm_profiles import get_prop_profile, list_prop_profiles
from prop_guard import PropFirmGuard

try:
    from src.exchanges.mt5_cfd_connector import (
        get_historical_candles,
        load_symbol_specs_json,
        initialize as mt5_initialize,
        shutdown as mt5_shutdown,
    )
except Exception:
    EXCHANGES = PROJECT_ROOT / "src" / "exchanges"
    if str(EXCHANGES) not in sys.path:
        sys.path.insert(0, str(EXCHANGES))
    try:
        from mt5_cfd_connector import (  # type: ignore
            get_historical_candles,
            load_symbol_specs_json,
            initialize as mt5_initialize,
            shutdown as mt5_shutdown,
        )
    except Exception:
        get_historical_candles = None
        load_symbol_specs_json = None
        mt5_initialize = None
        mt5_shutdown = None

mod = importlib.import_module("ict_top_bottom_ticking")

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

VARIANTS: Dict[str, type] = {"type2_baseline": BASE_TYPE2, "type1_sniper": BASE_TYPE1}
ENGINE_CASH = 1_000_000.0
DEFAULT_ACCOUNT_CASH = 50_000.0


# CFD-specific strategy tuning. This intentionally keeps the ICT model close to
# the futures version, but uses CFD symbol keys and CFD tick sizes/stop ranges.
# Retune after real CFD data is available because CFD feeds are broker-specific.
SYMBOL_SPECS = {
    "US100": dict(tick_size=0.1, min_stop_baseline=6.0, max_stop_baseline=35.0, min_stop_sniper=5.0, max_stop_sniper=40.0, expiry_baseline=18, expiry_sniper=14, touch_tol_baseline_ticks=2, touch_tol_sniper_ticks=3),
    "US500": dict(tick_size=0.1, min_stop_baseline=3.0, max_stop_baseline=18.0, min_stop_sniper=2.5, max_stop_sniper=20.0, expiry_baseline=18, expiry_sniper=14, touch_tol_baseline_ticks=2, touch_tol_sniper_ticks=3),
    "US30": dict(tick_size=1.0, min_stop_baseline=20.0, max_stop_baseline=140.0, min_stop_sniper=15.0, max_stop_sniper=160.0, expiry_baseline=18, expiry_sniper=14, touch_tol_baseline_ticks=1, touch_tol_sniper_ticks=2),
    "XAUUSD": dict(tick_size=0.01, min_stop_baseline=0.8, max_stop_baseline=8.0, min_stop_sniper=0.6, max_stop_sniper=9.0, expiry_baseline=18, expiry_sniper=14, touch_tol_baseline_ticks=10, touch_tol_sniper_ticks=15),
    "USOIL": dict(tick_size=0.01, min_stop_baseline=0.05, max_stop_baseline=0.80, min_stop_sniper=0.04, max_stop_sniper=1.00, expiry_baseline=18, expiry_sniper=14, touch_tol_baseline_ticks=2, touch_tol_sniper_ticks=3),
}


def _get_attr(obj, names: tuple[str, ...], default=None):
    if obj is None:
        return default
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    if isinstance(obj, dict):
        for n in names:
            if n in obj and obj[n] is not None:
                return obj[n]
    return default


def profile_account_cash(profile, default: float = DEFAULT_ACCOUNT_CASH) -> float:
    return float(_get_attr(profile, ("account_size", "initial_balance", "starting_balance", "starting_cash", "cash", "balance"), default))


def profile_timezone(profile, default: str = "Europe/Prague") -> str:
    return str(_get_attr(profile, ("timezone", "tz", "daily_reset_timezone"), default))


def profile_name(profile, fallback: str) -> str:
    return str(_get_attr(profile, ("name", "profile_name"), fallback))


def to_tz(ts, tz: str):
    if pd.isna(ts):
        return ts
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(tz)


def to_et(ts):
    return to_tz(ts, "America/New_York")


def realized_points(row: pd.Series) -> float:
    side = str(row.get("side", "")).upper()
    if side == "LONG":
        return float(row["exit_price"]) - float(row["entry_price"])
    return float(row["entry_price"]) - float(row["exit_price"])


def _tail_label(tail_rows: Optional[int]) -> str:
    return "notail" if tail_rows is None else f"tail{int(tail_rows)}"


def _safe_profile_label(prop_profile_name: str) -> str:
    return str(prop_profile_name).replace("/", "_").replace(" ", "_")


def _run_suffix(prop_profile_name: str, days_back: int, tail_rows: Optional[int]) -> str:
    return f"{_safe_profile_label(prop_profile_name)}_{int(days_back)}d_{_tail_label(tail_rows)}"


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Cached parquet must have DatetimeIndex: {path}")
    return df


def load_cfd_data(cfg: CFDInstrumentConfig, source: str, cache_dir: Path, refresh_data: bool) -> pd.DataFrame:
    safe_symbol = cfg.mt5_symbol.replace(".", "_").replace("/", "_")
    cache_path = cache_dir / f"{safe_symbol}_{cfg.timeframe}_{cfg.days_back}d.parquet"

    if source == "cache":
        print(f"📦 Using CFD cache: {cache_path}")
        return _safe_read_parquet(cache_path)

    if source == "mt5":
        if get_historical_candles is None:
            raise RuntimeError("MT5 connector not available. Copy mt5_cfd_connector.py into src/exchanges first.")
        return get_historical_candles(
            cfg.mt5_symbol,
            timeframe=cfg.timeframe,
            days_back=cfg.days_back,
            cache_dir=cache_dir,
            refresh=refresh_data,
        )

    raise ValueError("source must be 'cache' or 'mt5'")


def lots_to_internal_units(lots: float, lot_unit: float) -> int:
    if lot_unit <= 0:
        raise ValueError("lot_unit must be > 0")
    return max(0, int(round(float(lots) / float(lot_unit))))


def internal_units_to_lots(units: float, lot_unit: float) -> float:
    return abs(float(units)) * float(lot_unit)


def floor_to_lot_step(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    return float(np.floor(float(value) / float(step)) * float(step))


def calculate_lots_for_risk(
    *,
    risk_points: float,
    usd_per_point_per_lot: float,
    risk_usd: float,
    min_lot: float,
    lot_step: float,
    max_lot: float,
) -> float:
    risk_points = float(risk_points)
    usd_per_point_per_lot = float(usd_per_point_per_lot)
    risk_usd = float(risk_usd)
    if risk_points <= 0 or usd_per_point_per_lot <= 0 or risk_usd <= 0:
        return 0.0
    raw_lots = risk_usd / (risk_points * usd_per_point_per_lot)
    lots = min(floor_to_lot_step(raw_lots, lot_step), float(max_lot))
    if lots < min_lot:
        return 0.0
    return round(lots, 8)


def _prepare_meta(
    meta: pd.DataFrame,
    cfg: CFDInstrumentConfig,
    variant_name: str,
    prop_profile_name: str,
    account_cash: float,
    report_timezone: str,
) -> pd.DataFrame:
    if meta.empty:
        return meta
    out = meta.copy()
    out["variant"] = variant_name
    out["symbol"] = cfg.key
    out["mt5_symbol"] = cfg.mt5_symbol
    out["futures_equivalent"] = cfg.futures_equivalent
    out["prop_profile"] = prop_profile_name
    out["lot_unit"] = cfg.lot_unit
    out["usd_per_point_per_lot"] = cfg.usd_per_point_per_lot
    out["report_account_cash"] = float(account_cash)
    out["entry_time"] = pd.to_datetime(out.get("entry_time"), errors="coerce", utc=True)
    out["exit_time"] = pd.to_datetime(out.get("exit_time"), errors="coerce", utc=True)
    out["entry_time_et"] = out["entry_time"].apply(to_et)
    out["exit_time_et"] = out["exit_time"].apply(to_et)
    out["exit_time_report_tz"] = out["exit_time"].apply(lambda x: to_tz(x, report_timezone))
    report_naive = pd.to_datetime(out["exit_time_report_tz"], errors="coerce").dt.tz_localize(None)
    out["calendar_exit_date_report_tz"] = report_naive.dt.date
    out["calendar_exit_month_report_tz"] = report_naive.dt.to_period("M").astype(str)
    out["calendar_exit_year_report_tz"] = report_naive.dt.year
    out["realized_points"] = out.apply(realized_points, axis=1)

    # Backtesting.py raw pnl is price points * internal units.
    # Each internal unit = cfg.lot_unit lots. Convert that to USD.
    raw_pnl = pd.to_numeric(out.get("pnl"), errors="coerce")
    realized_pts = pd.to_numeric(out["realized_points"], errors="coerce")
    implied_units = np.where(np.abs(realized_pts) > 1e-12, raw_pnl / realized_pts, np.nan)
    out["closed_internal_units"] = np.abs(implied_units)
    out["closed_lots"] = out["closed_internal_units"] * cfg.lot_unit
    out["realized_pnl_usd"] = raw_pnl * cfg.usd_per_point_per_lot * cfg.lot_unit
    out["gross_return_pct_on_account"] = (out["realized_pnl_usd"] / float(account_cash)) * 100.0
    return out


def _decision_allowed(decision) -> bool:
    if isinstance(decision, bool):
        return bool(decision)
    return bool(getattr(decision, "allowed", False))


def _decision_reason(decision) -> str:
    if isinstance(decision, bool):
        return "allowed" if decision else "blocked"
    return str(getattr(decision, "reason", "blocked"))


def _build_cfd_strategy_class(
    cfg: CFDInstrumentConfig,
    variant_name: str,
    base_cls: type,
    prop_profile_name: str,
    profile,
    account_cash: float,
    risk_pct: float,
    news_calendar: Optional[RestrictedNewsCalendar],
    block_news: bool,
    block_weekend: bool,
    weekend_timezone: str,
    weekend_cutoff_hour: int,
) -> type:
    spec = SYMBOL_SPECS[cfg.key]
    is_sniper = variant_name == "type1_sniper"

    class CFDStrategy(base_cls):
        # Backtesting.py internal size is integer units where 1 unit = cfg.lot_unit lot.
        fixed_size = lots_to_internal_units(cfg.min_lot, cfg.lot_unit)
        symbol_name = cfg.key
        tick_size = spec["tick_size"]
        setup_expiry_bars = spec["expiry_sniper"] if is_sniper else spec["expiry_baseline"]
        limit_touch_tolerance_ticks = spec["touch_tol_sniper_ticks"] if is_sniper else spec["touch_tol_baseline_ticks"]
        require_cos_confirmation = False if is_sniper else True
        require_internal_sweep_filter = False
        min_stop_points = spec["min_stop_sniper"] if is_sniper else spec["min_stop_baseline"]
        max_stop_points = spec["max_stop_sniper"] if is_sniper else spec["max_stop_baseline"]
        last_trade_log: List[dict] = []
        last_debug_counts: dict = {}

        def init(self):
            self.prop_guard = PropFirmGuard(profile) if profile is not None else None
            self._guard_seen_closed = 0
            self.__class__.last_trade_log = []
            self.__class__.last_debug_counts = {}
            super().init()
            self.debug_counts.setdefault("blocked_prop_daily_loss", 0)
            self.debug_counts.setdefault("blocked_prop_consecutive_losses", 0)
            self.debug_counts.setdefault("blocked_prop_max_trades", 0)
            self.debug_counts.setdefault("blocked_prop_trailing_drawdown", 0)
            self.debug_counts.setdefault("blocked_prop_max_loss", 0)
            self.debug_counts.setdefault("blocked_prop_news", 0)
            self.debug_counts.setdefault("blocked_prop_weekend", 0)
            self.debug_counts.setdefault("blocked_cfd_size_zero", 0)
            self.debug_counts.setdefault("dynamic_size_updates", 0)
            self._sync_debug()

        def _sync_debug(self):
            guard = getattr(self, "prop_guard", None)
            if guard is not None:
                self.debug_counts["prop_balance"] = float(getattr(guard, "balance", 0.0))
                self.debug_counts["prop_day_realized"] = float(getattr(guard, "day_realized", 0.0))
                self.debug_counts["prop_consecutive_losses_today"] = int(getattr(guard, "consecutive_losses_today", 0))
            self.__class__.last_debug_counts = dict(getattr(self, "debug_counts", {}))

        def _now_utc_from_row(self, row: pd.Series):
            try:
                ts = row.name
                t = pd.Timestamp(ts)
                return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            except Exception:
                return pd.Timestamp.utcnow()

        def _risk_points_for_pending(self, row: pd.Series, direction: str) -> float:
            if getattr(self, "pending", None) is None:
                return 0.0
            if direction == "short":
                entry = min(float(row["Close"]), float(self.pending.entry_ce))
                stop = float(self.pending.stop_price)
                return stop - entry
            entry = max(float(row["Close"]), float(self.pending.entry_ce))
            stop = float(self.pending.stop_price)
            return entry - stop

        def _planned_risk_and_size(self, row: pd.Series, direction: str) -> Tuple[float, int, float]:
            risk_points = self._risk_points_for_pending(row, direction)
            if risk_points <= 0 or not np.isfinite(risk_points):
                return 0.0, 0, 0.0
            risk_usd = float(account_cash) * float(risk_pct)
            lots = calculate_lots_for_risk(
                risk_points=risk_points,
                usd_per_point_per_lot=cfg.usd_per_point_per_lot,
                risk_usd=risk_usd,
                min_lot=cfg.min_lot,
                lot_step=cfg.lot_step,
                max_lot=cfg.max_lot,
            )
            units = lots_to_internal_units(lots, cfg.lot_unit)
            planned_risk = risk_points * cfg.usd_per_point_per_lot * internal_units_to_lots(units, cfg.lot_unit)
            return planned_risk, units, lots

        def _update_guard_from_closed_trades(self):
            guard = getattr(self, "prop_guard", None)
            if guard is None:
                return
            try:
                closed = list(self.closed_trades)
            except Exception:
                return
            if len(closed) <= self._guard_seen_closed:
                return
            for t in closed[self._guard_seen_closed:]:
                exit_time = pd.Timestamp(str(t.exit_time))
                if exit_time.tzinfo is None:
                    exit_time = exit_time.tz_localize("UTC")
                # t.pl = price points * internal units. Convert to USD.
                pnl_usd = float(getattr(t, "pl", 0.0)) * cfg.usd_per_point_per_lot * cfg.lot_unit
                guard.on_trade_closed(pnl_usd, exit_time)
            self._guard_seen_closed = len(closed)
            self._sync_debug()

        def _prop_allows_entry(self, row: pd.Series, direction: str) -> bool:
            when = self._now_utc_from_row(row)

            if block_weekend and is_late_friday_cutoff(when, timezone=weekend_timezone, cutoff_hour=weekend_cutoff_hour):
                self.debug_counts["blocked_prop_weekend"] += 1
                self._sync_debug()
                self._clear_pending()
                return False

            if block_news and news_calendar is not None and news_calendar.is_blackout(when, cfg.key):
                self.debug_counts["blocked_prop_news"] += 1
                self._sync_debug()
                self._clear_pending()
                return False

            planned_risk, units, lots = self._planned_risk_and_size(row, direction)
            if units <= 0:
                self.debug_counts["blocked_cfd_size_zero"] += 1
                self._sync_debug()
                self._clear_pending()
                return False

            guard = getattr(self, "prop_guard", None)
            if guard is not None:
                trade_day = row.get("session_date", row.get("et_date", when))
                try:
                    decision = guard.can_open_trade(trade_day, planned_risk_usd=planned_risk)
                except TypeError:
                    decision = guard.can_open_trade(trade_day)
                if not _decision_allowed(decision):
                    key = f"blocked_prop_{_decision_reason(decision)}"
                    self.debug_counts[key] = self.debug_counts.get(key, 0) + 1
                    self._sync_debug()
                    self._clear_pending()
                    return False

            self.fixed_size = int(units)
            self.debug_counts["dynamic_size_updates"] += 1
            self.debug_counts["last_dynamic_lots_x100"] = int(round(lots * 100))
            self._sync_debug()
            return True

        def _enter_short(self, row: pd.Series, i: int):
            self._update_guard_from_closed_trades()
            if not self._prop_allows_entry(row, "short"):
                return
            return super()._enter_short(row, i)

        def _enter_long(self, row: pd.Series, i: int):
            self._update_guard_from_closed_trades()
            if not self._prop_allows_entry(row, "long"):
                return
            return super()._enter_long(row, i)

        def next(self):
            self._update_guard_from_closed_trades()
            return super().next()

    clean_profile = _safe_profile_label(prop_profile_name)
    CFDStrategy.__name__ = f"ICT_TBT_CFD_{cfg.key}_{variant_name}_{clean_profile}"
    return CFDStrategy


def run_symbol_variant(
    cfg: CFDInstrumentConfig,
    variant_name: str,
    base_cls: type,
    prop_profile_name: str,
    profile,
    account_cash: float,
    risk_pct: float,
    report_timezone: str,
    news_calendar: Optional[RestrictedNewsCalendar],
    block_news: bool,
    block_weekend: bool,
    weekend_timezone: str,
    weekend_cutoff_hour: int,
    source: str,
    cache_dir: Path,
    refresh_data: bool,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    StrategyCls = _build_cfd_strategy_class(
        cfg,
        variant_name,
        base_cls,
        prop_profile_name,
        profile,
        account_cash,
        risk_pct,
        news_calendar,
        block_news,
        block_weekend,
        weekend_timezone,
        weekend_cutoff_hour,
    )
    label = prop_profile_name
    print(f"\n=== {cfg.key}/{cfg.mt5_symbol} | {variant_name} | {label} ===")
    df = load_cfd_data(cfg, source, cache_dir, refresh_data)
    if cfg.tail_rows is not None:
        df = df.tail(int(cfg.tail_rows))
    print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()} | days_back={cfg.days_back} | tail_rows={cfg.tail_rows or 'none'}")
    if len(df) < 200:
        print("⚠️ Very few bars. Strategy may not warm up or trade.")
    bt = Backtest(df, StrategyCls, cash=ENGINE_CASH, commission=0.0, exclusive_orders=True, trade_on_close=False)
    stats = bt.run()
    meta = pd.DataFrame(getattr(StrategyCls, "last_trade_log", []))
    meta = _prepare_meta(meta, cfg, variant_name, label, account_cash, report_timezone)
    debug = pd.DataFrame([getattr(StrategyCls, "last_debug_counts", {})])
    if not debug.empty:
        debug.insert(0, "variant", variant_name)
        debug.insert(1, "symbol", cfg.key)
        debug.insert(2, "mt5_symbol", cfg.mt5_symbol)
        debug.insert(3, "prop_profile", label)
    return stats, meta, debug


def summarize_daily(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(["calendar_exit_date_report_tz", "symbol", "variant"], dropna=False)
    return g.agg(
        trades=("realized_pnl_usd", "size"),
        realized_pnl_usd=("realized_pnl_usd", "sum"),
        wins=("realized_pnl_usd", lambda s: int((s > 0).sum())),
        losses=("realized_pnl_usd", lambda s: int((s < 0).sum())),
    ).reset_index()


def summarize_monthly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(["calendar_exit_month_report_tz", "symbol", "variant"], dropna=False)
    return g.agg(
        trades=("realized_pnl_usd", "size"),
        realized_pnl_usd=("realized_pnl_usd", "sum"),
        wins=("realized_pnl_usd", lambda s: int((s > 0).sum())),
        losses=("realized_pnl_usd", lambda s: int((s < 0).sum())),
    ).reset_index()


def summarize_variant(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(["symbol", "variant"], dropna=False)
    return g.agg(
        trades=("realized_pnl_usd", "size"),
        realized_pnl_usd=("realized_pnl_usd", "sum"),
        avg_trade=("realized_pnl_usd", "mean"),
        best_trade=("realized_pnl_usd", "max"),
        worst_trade=("realized_pnl_usd", "min"),
        win_rate=("realized_pnl_usd", lambda s: float((s > 0).mean()) if len(s) else np.nan),
        gross_profit=("realized_pnl_usd", lambda s: float(s[s > 0].sum())),
        gross_loss=("realized_pnl_usd", lambda s: float(s[s < 0].sum())),
    ).reset_index()


def portfolio_summary(trades: pd.DataFrame, label: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([{"label": label, "trades": 0, "realized_pnl_usd": 0.0}])
    pnl = trades["realized_pnl_usd"].astype(float)
    daily = trades.groupby("calendar_exit_date_report_tz")["realized_pnl_usd"].sum().sort_index()
    equity = daily.cumsum()
    dd = equity - equity.cummax()
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = pnl[pnl < 0].sum()
    return pd.DataFrame([{
        "label": label,
        "trades": int(len(trades)),
        "realized_pnl_usd": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(gross_profit / abs(gross_loss)) if gross_loss < 0 else np.inf,
        "avg_trade": float(pnl.mean()),
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
        "best_day": float(daily.max()) if len(daily) else 0.0,
        "worst_day": float(daily.min()) if len(daily) else 0.0,
        "approx_max_daily_drawdown": float(dd.min()) if len(dd) else 0.0,
    }])


def write_outputs(out_dir: Path, suffix: str, raw: pd.DataFrame, filtered: pd.DataFrame, debug_raw: pd.DataFrame, debug_filtered: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {"raw": raw, "filtered": filtered}
    summaries = []
    for label, df in datasets.items():
        df.to_csv(out_dir / f"top_bottom_ticking_cfd_trade_log_{label}_{suffix}.csv", index=False)
        summarize_monthly(df).to_csv(out_dir / f"top_bottom_ticking_cfd_monthly_summary_{label}_{suffix}.csv", index=False)
        summarize_daily(df).to_csv(out_dir / f"top_bottom_ticking_cfd_daily_summary_{label}_{suffix}.csv", index=False)
        summarize_variant(df).to_csv(out_dir / f"top_bottom_ticking_cfd_variant_summary_{label}_{suffix}.csv", index=False)
        ps = portfolio_summary(df, label)
        ps.to_csv(out_dir / f"top_bottom_ticking_cfd_portfolio_summary_{label}_{suffix}.csv", index=False)
        summaries.append(ps)
    pd.concat(summaries, ignore_index=True).to_csv(out_dir / f"top_bottom_ticking_cfd_portfolio_compare_{suffix}.csv", index=False)
    pd.concat([debug_raw, debug_filtered], ignore_index=True).to_csv(out_dir / f"top_bottom_ticking_cfd_debug_counts_{suffix}.csv", index=False)


def apply_symbol_specs_json(configs: Dict[str, CFDInstrumentConfig], json_path: Optional[str]) -> Dict[str, CFDInstrumentConfig]:
    if not json_path:
        return configs
    if load_symbol_specs_json is None:
        raise RuntimeError("MT5 connector not available, cannot load symbol specs JSON.")
    specs = load_symbol_specs_json(json_path)
    out = dict(configs)
    for key, cfg in configs.items():
        spec = specs.get(cfg.mt5_symbol)
        if spec is None:
            continue
        out[key] = with_runtime_overrides(
            cfg,
            usd_per_point_per_lot=spec.usd_per_point_per_lot,
            min_lot=spec.volume_min,
            lot_step=spec.volume_step,
            max_lot=spec.volume_max,
        )
    return out


def main(argv: Optional[list[str]] = None) -> int:
    profile_choices = sorted(set(list(list_prop_profiles()) + ["none"]))
    parser = argparse.ArgumentParser(description="CFD backtest for ICT top-bottom ticking using existing prop profiles.")
    parser.add_argument("--source", choices=["cache", "mt5"], default="cache", help="Use cached parquet or pull from MT5.")
    parser.add_argument("--refresh-data", action="store_true", help="Refresh MT5 cache when --source mt5.")
    parser.add_argument("--mt5-path", default=None, help="Optional MT5 terminal path.")
    parser.add_argument("--mt5-login", type=int, default=None)
    parser.add_argument("--mt5-password", default=None)
    parser.add_argument("--mt5-server", default=None)
    parser.add_argument("--symbols", default="all", help="Comma-separated CFD keys or 'all'. Example: US100,US500,XAUUSD")
    parser.add_argument("--variants", default="all", help="Comma-separated variants or 'all'.")
    parser.add_argument("--days-back", type=int, default=365)
    parser.add_argument("--tail-rows", type=int, default=None)
    parser.add_argument("--no-tail", action="store_true")
    parser.add_argument("--prop-profile", choices=profile_choices, default="none")
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--account-cash", type=float, default=DEFAULT_ACCOUNT_CASH, help="Fallback/reporting account size if profile does not expose one.")
    parser.add_argument("--risk-pct", type=float, default=0.0025, help="Risk per trade, e.g. 0.0025 = 0.25%.")
    parser.add_argument("--symbol-specs-json", default=None, help="Optional JSON exported from MT5 symbol specs.")
    parser.add_argument("--news-blackout-csv", default=None, help="Optional restricted-news CSV.")
    parser.add_argument("--block-news", action="store_true", help="Block new entries around restricted news CSV events.")
    parser.add_argument("--block-weekend", action="store_true", help="Block new entries late Friday for standard prop-style accounts.")
    parser.add_argument("--weekend-timezone", default="Europe/Prague")
    parser.add_argument("--weekend-cutoff-hour", type=int, default=20)
    parser.add_argument("--report-timezone", default=None, help="Timezone used for daily/monthly grouping. Defaults to profile timezone or Europe/Prague.")
    parser.add_argument("--cache-dir", default="src/data/mt5_cfd_cache")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    if args.list_profiles:
        print("Available prop profiles:")
        for p in profile_choices:
            print("-", p)
        return 0

    tail_rows = None if args.no_tail else args.tail_rows
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir) if args.output_dir else ROOT

    configs = apply_symbol_specs_json(CFD_INSTRUMENTS, args.symbol_specs_json)
    configs = {k: with_runtime_overrides(v, days_back=args.days_back, tail_rows=tail_rows) for k, v in configs.items()}

    symbols = list(configs) if args.symbols == "all" else [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    variants = list(VARIANTS) if args.variants == "all" else [v.strip() for v in args.variants.split(",") if v.strip()]

    if args.source == "mt5":
        if mt5_initialize is None:
            raise RuntimeError("MT5 connector not available. Copy mt5_cfd_connector.py into src/exchanges.")
        mt5_initialize(path=args.mt5_path, login=args.mt5_login, password=args.mt5_password, server=args.mt5_server)

    filtered_profile = None if args.prop_profile == "none" else get_prop_profile(args.prop_profile)
    account_cash = profile_account_cash(filtered_profile, args.account_cash)
    report_timezone = args.report_timezone or profile_timezone(filtered_profile, "Europe/Prague")
    news_calendar = RestrictedNewsCalendar(args.news_blackout_csv, blackout_minutes=2) if args.news_blackout_csv else None

    raw_metas: List[pd.DataFrame] = []
    filtered_metas: List[pd.DataFrame] = []
    raw_debugs: List[pd.DataFrame] = []
    filtered_debugs: List[pd.DataFrame] = []

    try:
        for sym in symbols:
            if sym not in configs:
                raise ValueError(f"Unknown CFD symbol key {sym}. Choices: {sorted(configs)}")
            cfg = configs[sym]
            for variant in variants:
                if variant not in VARIANTS:
                    raise ValueError(f"Unknown variant {variant}. Choices: {sorted(VARIANTS)}")

                stats_raw, meta_raw, debug_raw = run_symbol_variant(
                    cfg, variant, VARIANTS[variant], "none", None, account_cash, args.risk_pct,
                    report_timezone, None, False, False, args.weekend_timezone, args.weekend_cutoff_hour,
                    args.source, cache_dir, args.refresh_data,
                )
                print(f"raw | {variant} | {sym} engine trades={stats_raw.get('# Trades', np.nan)}")
                if not meta_raw.empty:
                    raw_metas.append(meta_raw)
                if not debug_raw.empty:
                    raw_debugs.append(debug_raw)

                stats_f, meta_f, debug_f = run_symbol_variant(
                    cfg, variant, VARIANTS[variant], args.prop_profile, filtered_profile, account_cash, args.risk_pct,
                    report_timezone, news_calendar, args.block_news, args.block_weekend, args.weekend_timezone, args.weekend_cutoff_hour,
                    args.source, cache_dir, False,
                )
                print(f"filtered | {variant} | {sym} engine trades={stats_f.get('# Trades', np.nan)}")
                if not meta_f.empty:
                    filtered_metas.append(meta_f)
                if not debug_f.empty:
                    filtered_debugs.append(debug_f)
    finally:
        if args.source == "mt5" and mt5_shutdown is not None:
            mt5_shutdown()

    raw = pd.concat(raw_metas, ignore_index=True) if raw_metas else pd.DataFrame()
    filtered = pd.concat(filtered_metas, ignore_index=True) if filtered_metas else pd.DataFrame()
    debug_raw = pd.concat(raw_debugs, ignore_index=True) if raw_debugs else pd.DataFrame()
    debug_filtered = pd.concat(filtered_debugs, ignore_index=True) if filtered_debugs else pd.DataFrame()

    suffix = _run_suffix(args.prop_profile, args.days_back, tail_rows)
    write_outputs(out_dir, suffix, raw, filtered, debug_raw, debug_filtered)

    raw_pnl = float(raw["realized_pnl_usd"].sum()) if not raw.empty else 0.0
    filtered_pnl = float(filtered["realized_pnl_usd"].sum()) if not filtered.empty else 0.0
    print("\n=== CFD summary ===")
    print(f"Prop profile:               {args.prop_profile}")
    print(f"Account cash used:          ${account_cash:,.2f}")
    print(f"Risk pct per trade:         {args.risk_pct:.4%}")
    print(f"Raw realized PnL USD:       ${raw_pnl:,.2f}")
    print(f"Filtered realized PnL USD:  ${filtered_pnl:,.2f}")
    print(f"Output suffix: {suffix}")
    print(f"Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
