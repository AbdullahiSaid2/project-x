"""
HOW TO USE / WHERE THIS FILE GOES
=================================

Place this file at:
    trading_system/src/strategies/manual/top_bottom_ticking_shared_axl.py

Dependencies that should also exist:
    trading_system/src/strategies/manual/ict_top_bottom_ticking_axl.py
    trading_system/src/strategies/manual/prop_firm_profiles.py
    trading_system/src/strategies/manual/prop_guard.py

Purpose:
- Backtest / forward-test harness for ICT top-bottom ticking
- Keeps strategy logic separate from prop-firm rules
- Lets you toggle prop firms with a command flag
- Applies prop rules as a PRE-ENTRY guard

RUN EXAMPLES
============

1) List available prop profiles
    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared --list-profiles

2) Run with no prop rules
    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared --prop-profile none

3) Run with Apex profile
    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared --prop-profile apex_50k_eval

4) Run with Apex + 6-loss breaker profile
    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared --prop-profile apex_50k_eval_6loss

5) Run only certain symbols
    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared --prop-profile none --symbols MNQ,MES

6) Run only one variant
    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared --prop-profile none --variants type2_baseline

7) Run selected symbols and variants together
    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared --prop-profile apex_50k_eval --symbols MNQ,MES,MGC --variants type1_sniper

Notes:
- This file is intended to replace your current top_bottom_ticking_shared.py
- The strategy file itself should stay pure and unchanged
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT if (ROOT / "src").exists() else ROOT.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fetcher import get_ohlcv
from prop_firm_profiles import get_prop_profile, list_prop_profiles
from prop_guard import PropFirmGuard

mod = importlib.import_module("ict_top_bottom_ticking_axl")

BASE_TYPE2 = None
for name in ("ICTTopBottomTickingType2Baseline", "ICT_TOP_BOTTOM_TICKING_TYPE2", "ICT_TOP_BOTTOM_TICKING"):
    if hasattr(mod, name):
        BASE_TYPE2 = getattr(mod, name)
        break
if BASE_TYPE2 is None:
    raise ImportError("Could not find baseline class in ict_top_bottom_ticking_axl.py")

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

@dataclass(frozen=True)
class InstrumentConfig:
    symbol: str
    exchange: str
    timeframe: str
    days_back: int
    tail_rows: int
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


@dataclass(frozen=True)
class AXLConfig:
    mode: str = "off"
    min_b_score: int = 5
    min_a_score: int = 8
    require_internal: bool = False
    require_htf_anchor: bool = False
    require_mtf_poi: bool = False
    safe_entry_for_b: bool = True
    liquidity_close_ticks: int = 20

INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "MNQ": InstrumentConfig("MNQ", "tradovate", "5m", 365, 120_000, 5, 2.0),
    "MES": InstrumentConfig("MES", "tradovate", "5m", 365, 120_000, 5, 5.0),
    "MYM": InstrumentConfig("MYM", "tradovate", "5m", 365, 120_000, 5, 0.5),
    "MGC": InstrumentConfig("MGC", "tradovate", "5m", 365, 120_000, 5, 10.0),
    "MCL": InstrumentConfig("MCL", "tradovate", "5m", 365, 120_000, 5, 100.0),
}

SYMBOL_SPECS: Dict[str, SymbolSpec] = {
    "MNQ": SymbolSpec(0.25, 6.0, 30.0, 5.0, 34.0, 18, 14, 1, 2, True, False, False, False),
    "MES": SymbolSpec(0.25, 3.0, 15.0, 2.5, 16.0, 18, 14, 1, 2, True, False, False, False),
    "MYM": SymbolSpec(1.0, 20.0, 120.0, 15.0, 140.0, 18, 14, 1, 2, True, False, False, False),
    "MGC": SymbolSpec(0.1, 0.8, 6.0, 0.6, 7.0, 18, 14, 1, 2, True, False, False, False),
    "MCL": SymbolSpec(0.01, 0.05, 0.60, 0.04, 0.80, 18, 14, 1, 2, True, False, False, False),
}

VARIANTS: Dict[str, type] = {"type2_baseline": BASE_TYPE2, "type1_sniper": BASE_TYPE1}


def to_et(ts):
    if pd.isna(ts):
        return ts
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("America/New_York")


def realized_points(row: pd.Series) -> float:
    side = str(row.get("side", "")).upper()
    if side == "LONG":
        return float(row["exit_price"]) - float(row["entry_price"])
    return float(row["entry_price"]) - float(row["exit_price"])


def _prepare_meta(meta: pd.DataFrame, cfg: InstrumentConfig, variant_name: str, prop_profile_name: str) -> pd.DataFrame:
    if meta.empty:
        return meta
    out = meta.copy()
    out["variant"] = variant_name
    out["symbol"] = cfg.symbol
    out["prop_profile"] = prop_profile_name
    out["report_contracts"] = cfg.contracts
    out["dollars_per_point"] = cfg.dollars_per_point
    out["report_account_cash"] = REPORT_ACCOUNT_CASH
    out["entry_time"] = pd.to_datetime(out.get("entry_time"), errors="coerce")
    out["exit_time"] = pd.to_datetime(out.get("exit_time"), errors="coerce")
    out["entry_time_et"] = out["entry_time"].apply(to_et)
    out["exit_time_et"] = out["exit_time"].apply(to_et)
    out["exit_time_et_naive"] = pd.to_datetime(out["exit_time_et"], errors="coerce").dt.tz_localize(None)
    out["calendar_exit_date_et"] = out["exit_time_et_naive"].dt.date
    out["realized_points"] = out.apply(realized_points, axis=1)
    out["gross_pnl_dollars_dynamic"] = out["realized_points"] * cfg.dollars_per_point * cfg.contracts
    out["gross_return_pct_on_50k"] = (out["gross_pnl_dollars_dynamic"] / REPORT_ACCOUNT_CASH) * 100.0
    return out


def _build_guarded_strategy_class(cfg: InstrumentConfig, variant_name: str, base_cls: type, prop_profile_name: str, axl_config: AXLConfig | None = None) -> type:
    spec = SYMBOL_SPECS[cfg.symbol]
    profile = get_prop_profile(prop_profile_name)
    is_sniper = variant_name == "type1_sniper"
    axl = axl_config or AXLConfig()

    class GuardedStrategy(base_cls):
        fixed_size = cfg.contracts
        tick_size = spec.tick_size
        setup_expiry_bars = spec.expiry_sniper if is_sniper else spec.expiry_baseline
        limit_touch_tolerance_ticks = spec.touch_tol_sniper_ticks if is_sniper else spec.touch_tol_baseline_ticks
        require_cos_confirmation = spec.require_cos_sniper if is_sniper else spec.require_cos_baseline
        require_internal_sweep_filter = spec.require_internal_sniper if is_sniper else spec.require_internal_baseline
        min_stop_points = spec.min_stop_sniper if is_sniper else spec.min_stop_baseline
        max_stop_points = spec.max_stop_sniper if is_sniper else spec.max_stop_baseline

        # AXL overlay toggles are passed from CLI and consumed by ict_top_bottom_ticking.py.
        axl_mode = axl.mode
        axl_min_b_score = axl.min_b_score
        axl_min_a_score = axl.min_a_score
        axl_require_internal = axl.require_internal
        axl_require_htf_anchor = axl.require_htf_anchor
        axl_require_mtf_poi = axl.require_mtf_poi
        axl_safe_entry_for_b = axl.safe_entry_for_b
        axl_liquidity_close_ticks = axl.liquidity_close_ticks

        last_trade_log: List[dict] = []
        last_debug_counts: dict = {}

        def init(self):
            self.prop_guard = PropFirmGuard(profile)
            self._guard_seen_closed = 0
            self.__class__.last_trade_log = []
            self.__class__.last_debug_counts = {}
            super().init()
            self.debug_counts.setdefault("blocked_prop_daily_loss", 0)
            self.debug_counts.setdefault("blocked_prop_consecutive_losses", 0)
            self.debug_counts.setdefault("blocked_prop_max_trades", 0)
            self.debug_counts.setdefault("blocked_prop_trailing_drawdown", 0)
            self._sync_debug()

        def _sync_debug(self):
            guard = getattr(self, "prop_guard", None)
            if guard is None:
                self.__class__.last_debug_counts = dict(getattr(self, "debug_counts", {}))
                return
            self.debug_counts["prop_balance"] = float(guard.balance)
            self.debug_counts["prop_day_realized"] = float(guard.day_realized)
            self.debug_counts["prop_consecutive_losses_today"] = int(guard.consecutive_losses_today)
            self.__class__.last_debug_counts = dict(self.debug_counts)

        def _update_guard_from_closed_trades(self):
            try:
                closed = list(self.closed_trades)
            except Exception:
                return
            if len(closed) <= self._guard_seen_closed:
                return
            for t in closed[self._guard_seen_closed:]:
                exit_et = to_et(pd.Timestamp(str(t.exit_time)))
                trade_day = exit_et.tz_localize(None).date()
                points = (float(t.exit_price) - float(t.entry_price)) if float(t.size) > 0 else (float(t.entry_price) - float(t.exit_price))
                pnl_dollars = points * cfg.dollars_per_point * cfg.contracts
                self.prop_guard.on_trade_closed(pnl_dollars, trade_day)
            self._guard_seen_closed = len(closed)
            self._sync_debug()

        def _guard_allows_entry(self, row: pd.Series) -> bool:
            trade_day = row.get("session_date", row.get("et_date"))
            decision = self.prop_guard.can_open_trade(trade_day)
            if decision.allowed:
                return True
            key = f"blocked_prop_{decision.reason}"
            self.debug_counts[key] = self.debug_counts.get(key, 0) + 1
            self._sync_debug()
            self._clear_pending()
            return False

        def _enter_short(self, row: pd.Series, i: int):
            self._update_guard_from_closed_trades()
            if not self._guard_allows_entry(row):
                return
            return super()._enter_short(row, i)

        def _enter_long(self, row: pd.Series, i: int):
            self._update_guard_from_closed_trades()
            if not self._guard_allows_entry(row):
                return
            return super()._enter_long(row, i)

        def next(self):
            self._update_guard_from_closed_trades()
            return super().next()

    GuardedStrategy.__name__ = f"{base_cls.__name__}_{cfg.symbol}_{variant_name}_{prop_profile_name}"
    return GuardedStrategy


def run_symbol_variant(cfg: InstrumentConfig, variant_name: str, base_cls: type, prop_profile_name: str, axl_config: AXLConfig | None = None):
    StrategyCls = _build_guarded_strategy_class(cfg, variant_name, base_cls, prop_profile_name, axl_config)
    print(f"\n=== {cfg.symbol} | {variant_name} | {prop_profile_name} ===")
    df = get_ohlcv(cfg.symbol, exchange=cfg.exchange, timeframe=cfg.timeframe, days_back=cfg.days_back)
    if cfg.tail_rows and cfg.tail_rows > 0:
        df = df.tail(cfg.tail_rows)
        tail_label = str(cfg.tail_rows)
    else:
        tail_label = "none"
    print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()} | days_back={cfg.days_back} | tail_rows={tail_label}")
    bt = Backtest(df, StrategyCls, cash=ENGINE_CASH, commission=0.0, exclusive_orders=True, trade_on_close=False)
    stats = bt.run()
    meta = pd.DataFrame(getattr(StrategyCls, "last_trade_log", []))
    meta = _prepare_meta(meta, cfg, variant_name, prop_profile_name)
    debug = pd.DataFrame([getattr(StrategyCls, "last_debug_counts", {})])
    if not debug.empty:
        debug.insert(0, "variant", variant_name)
        debug.insert(1, "symbol", cfg.symbol)
        debug.insert(2, "prop_profile", prop_profile_name)
    return stats, meta, debug



def _profit_factor(values: pd.Series) -> float:
    wins = values[values > 0].sum()
    losses = -values[values < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def _win_rate(values: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
    return float((values > 0).mean() * 100.0)


def _write_group_summary(df: pd.DataFrame, group_cols: list[str], out_path: Path):
    if df.empty:
        return
    required = {"gross_pnl_dollars_dynamic"}
    if not required.issubset(set(df.columns)):
        return

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        pnl = pd.to_numeric(g["gross_pnl_dollars_dynamic"], errors="coerce").fillna(0.0)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "trades": int(len(g)),
                "gross_pnl_dollars_dynamic": float(pnl.sum()),
                "avg_pnl_dollars": float(pnl.mean()) if len(pnl) else 0.0,
                "win_rate_pct": _win_rate(pnl),
                "profit_factor": _profit_factor(pnl),
                "best_trade": float(pnl.max()) if len(pnl) else 0.0,
                "worst_trade": float(pnl.min()) if len(pnl) else 0.0,
            }
        )
        rows.append(row)

    if rows:
        pd.DataFrame(rows).sort_values(group_cols).to_csv(out_path, index=False)
        print(f"Saved summary -> {out_path}")


def write_enhanced_summaries(combined: pd.DataFrame, suffix: str):
    if combined.empty:
        return

    # Existing trade-level output is still the source of truth. These summaries make it
    # easier to judge whether the AXL overlay actually improves expectancy.
    if "calendar_exit_date_et" in combined.columns:
        daily = combined.copy()
        daily["calendar_exit_date_et"] = pd.to_datetime(daily["calendar_exit_date_et"], errors="coerce").dt.date
        _write_group_summary(
            daily,
            ["symbol", "variant", "calendar_exit_date_et"],
            ROOT / f"top_bottom_ticking_daily_summary{suffix}.csv",
        )

    if "exit_time_et_naive" in combined.columns:
        monthly = combined.copy()
        monthly["exit_month_et"] = pd.to_datetime(monthly["exit_time_et_naive"], errors="coerce").dt.to_period("M").astype(str)
        _write_group_summary(
            monthly,
            ["symbol", "variant", "exit_month_et"],
            ROOT / f"top_bottom_ticking_monthly_summary{suffix}.csv",
        )

    _write_group_summary(
        combined,
        ["symbol", "variant"],
        ROOT / f"top_bottom_ticking_variant_summary{suffix}.csv",
    )

    if "axl_grade" in combined.columns:
        axl_cols = ["symbol", "variant", "axl_grade"]
        _write_group_summary(
            combined,
            axl_cols,
            ROOT / f"top_bottom_ticking_axl_grade_summary{suffix}.csv",
        )



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Top/bottom ticking AXL backtest with separate prop-firm profiles.")
    parser.add_argument("--prop-profile", default="apex_50k_eval", choices=list_prop_profiles())
    parser.add_argument("--symbols", default="all", help="Comma-separated symbols or 'all'")
    parser.add_argument("--variants", default="all", help="Comma-separated variants or 'all'")
    parser.add_argument("--days-back", type=int, default=None, help="Override days_back for all symbols")
    parser.add_argument("--tail-rows", type=int, default=None, help="Override tail row cap for all symbols")
    parser.add_argument("--no-tail", action="store_true", help="Disable tail row cap and use full loaded lookback")

    # AXL overlay controls. Use log_only first to measure before filtering.
    parser.add_argument("--axl-mode", default="off", choices=["off", "log_only", "filter"], help="AXL overlay mode")
    parser.add_argument("--axl-min-b-score", type=int, default=5, help="Minimum score for B-grade AXL setups")
    parser.add_argument("--axl-min-a-score", type=int, default=8, help="Minimum score for A-grade AXL setups")
    parser.add_argument("--axl-require-internal", action="store_true", help="Filter mode: require internal liquidity sweep")
    parser.add_argument("--axl-require-htf-anchor", action="store_true", help="Filter mode: require HTF PD-array anchor")
    parser.add_argument("--axl-require-mtf-poi", action="store_true", help="Filter mode: require MTF PD-array POI")
    parser.add_argument("--axl-disable-safe-entry-for-b", action="store_true", help="Do not move B setups to deeper ODE-style entries")
    parser.add_argument("--axl-liquidity-close-ticks", type=int, default=20, help="Ticks used to decide whether liquidity is close to a PD array")
    parser.add_argument("--list-profiles", action="store_true")
    args = parser.parse_args(argv)

    if args.list_profiles:
        for name in list_prop_profiles():
            print(name)
        return 0

    axl_config = AXLConfig(
        mode=args.axl_mode,
        min_b_score=args.axl_min_b_score,
        min_a_score=args.axl_min_a_score,
        require_internal=bool(args.axl_require_internal),
        require_htf_anchor=bool(args.axl_require_htf_anchor),
        require_mtf_poi=bool(args.axl_require_mtf_poi),
        safe_entry_for_b=not bool(args.axl_disable_safe_entry_for_b),
        liquidity_close_ticks=args.axl_liquidity_close_ticks,
    )

    symbols = list(INSTRUMENTS) if args.symbols == "all" else [s.strip() for s in args.symbols.split(",") if s.strip()]
    variants = list(VARIANTS) if args.variants == "all" else [v.strip() for v in args.variants.split(",") if v.strip()]

    instrument_overrides: Dict[str, InstrumentConfig] = {}
    for sym in symbols:
        base = INSTRUMENTS[sym]
        days_back = args.days_back if args.days_back is not None else base.days_back
        if args.no_tail:
            tail_rows = 0
        elif args.tail_rows is not None:
            tail_rows = args.tail_rows
        else:
            tail_rows = base.tail_rows
        instrument_overrides[sym] = InstrumentConfig(
            symbol=base.symbol,
            exchange=base.exchange,
            timeframe=base.timeframe,
            days_back=days_back,
            tail_rows=tail_rows,
            contracts=base.contracts,
            dollars_per_point=base.dollars_per_point,
        )

    metas = []
    debugs = []
    for sym in symbols:
        for variant in variants:
            stats, meta, debug = run_symbol_variant(instrument_overrides[sym], variant, VARIANTS[variant], args.prop_profile, axl_config)
            print(f"{variant} | {sym} engine trades={stats.get('# Trades', np.nan)}")
            if not meta.empty:
                metas.append(meta)
            if not debug.empty:
                debugs.append(debug)

    combined = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
    debug_df = pd.concat(debugs, ignore_index=True) if debugs else pd.DataFrame()

    tail_suffix = "notail" if args.no_tail else f"tail{args.tail_rows}" if args.tail_rows is not None else f"tail{INSTRUMENTS[symbols[0]].tail_rows}"
    days_suffix = f"{args.days_back}d" if args.days_back is not None else f"{INSTRUMENTS[symbols[0]].days_back}d"
    axl_suffix = f"_axl{args.axl_mode}" if args.axl_mode != "off" else ""
    suffix = f"_{args.prop_profile}_{days_suffix}_{tail_suffix}{axl_suffix}"

    out_trades = ROOT / f"top_bottom_ticking_axl_trade_log{suffix}.csv"
    out_debug = ROOT / f"top_bottom_ticking_axl_debug_counts{suffix}.csv"
    if not combined.empty:
        combined.to_csv(out_trades, index=False)
        print(f"Saved trades -> {out_trades}")
        print(f"Pre-entry-guard realized PnL: ${combined['gross_pnl_dollars_dynamic'].sum():.2f}")
        write_enhanced_summaries(combined, suffix)
    if not debug_df.empty:
        debug_df.to_csv(out_debug, index=False)
        print(f"Saved debug -> {out_debug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
