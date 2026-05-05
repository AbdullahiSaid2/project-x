"""
Top / Bottom Ticking TRUE backtest engine
========================================

This file intentionally strips the old reporting logic down to a small, auditable
calculation path.

Core principle
--------------
Backtesting.py already knows how many contracts were actually closed on each
closed trade row. Therefore final futures-dollar PnL must be calculated as:

    realized_pnl_dollars = engine_pnl_points_contracts * dollars_per_point

Equivalent explicit form:

    realized_pnl_dollars = realized_points * closed_size_contracts * dollars_per_point

Never calculate final PnL with configured_contracts on every closed row when the
strategy uses partial exits. That double/triple counts the same starting size.

This engine creates two worlds:
    raw      = all strategy trades, no prop filter
    filtered = post-trade portfolio filter using corrected dollar PnL

It does not change the strategy model. It only changes how results are measured.

Install:
    cp this file to src/strategies/manual/top_bottom_ticking_truth_backtest_engine.py

Examples:
    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_truth_backtest_engine \
      --prop-profile apex_50k_eval --days-back 365 --no-tail

    PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_truth_backtest_engine \
      --prop-profile apex_50k_eval --days-back 1825 --no-tail
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2] if len(ROOT.parents) >= 3 else ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fetcher import get_ohlcv
from prop_firm_profiles import get_prop_profile, list_prop_profiles


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstrumentConfig:
    symbol: str
    databento_symbol: str
    dollars_per_point: float
    contracts: int
    min_stop_points: float
    max_stop_points: float
    setup_expiry_bars: int
    touch_tolerance_points: float = 0.0
    days_back: int = 365
    tail_rows: Optional[int] = 120_000


INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "MNQ": InstrumentConfig("MNQ", "MNQ.n.0", dollars_per_point=2.0, contracts=5, min_stop_points=10.0, max_stop_points=80.0, setup_expiry_bars=14, touch_tolerance_points=0.25),
    "MES": InstrumentConfig("MES", "MES.n.0", dollars_per_point=5.0, contracts=5, min_stop_points=4.0, max_stop_points=30.0, setup_expiry_bars=14, touch_tolerance_points=0.25),
    "MYM": InstrumentConfig("MYM", "MYM.n.0", dollars_per_point=0.5, contracts=5, min_stop_points=20.0, max_stop_points=150.0, setup_expiry_bars=14, touch_tolerance_points=1.0),
    "MGC": InstrumentConfig("MGC", "MGC.n.0", dollars_per_point=10.0, contracts=5, min_stop_points=2.0, max_stop_points=25.0, setup_expiry_bars=14, touch_tolerance_points=0.1),
    "MCL": InstrumentConfig("MCL", "MCL.n.0", dollars_per_point=100.0, contracts=5, min_stop_points=0.10, max_stop_points=1.50, setup_expiry_bars=14, touch_tolerance_points=0.01),
}


@dataclass(frozen=True)
class VariantConfig:
    name: str
    base_class_name_candidates: Tuple[str, ...]
    require_internal_sweep: bool
    require_cos: bool


VARIANTS: Dict[str, VariantConfig] = {
    "type2_baseline": VariantConfig(
        name="type2_baseline",
        base_class_name_candidates=("ICTTopBottomTickingType2Baseline", "ICT_TOP_BOTTOM_TICKING_TYPE2", "ICT_TOP_BOTTOM_TICKING"),
        require_internal_sweep=True,
        require_cos=True,
    ),
    "type1_sniper": VariantConfig(
        name="type1_sniper",
        base_class_name_candidates=("ICTTopBottomTickingType1Sniper", "ICT_TOP_BOTTOM_TICKING_TYPE1", "ICT_TOP_BOTTOM_TICKING"),
        require_internal_sweep=False,
        require_cos=False,
    ),
}


# ---------------------------------------------------------------------------
# Loading strategy classes
# ---------------------------------------------------------------------------

def _load_base_class(strategy_module: str, candidates: Iterable[str]):
    mod = importlib.import_module(strategy_module)
    for name in candidates:
        if hasattr(mod, name):
            return getattr(mod, name)
    raise ImportError(f"Could not find any strategy class from {list(candidates)} in {strategy_module}")


def build_strategy_class(cfg: InstrumentConfig, variant: VariantConfig, strategy_module: str):
    base_cls = _load_base_class(strategy_module, variant.base_class_name_candidates)

    class Strategy(base_cls):
        pass

    Strategy.__name__ = f"TRUTH_{cfg.symbol}_{variant.name}"

    # Only set attributes that the strategy commonly uses. Missing attributes are
    # harmless; existing strategy defaults remain in place.
    Strategy.fixed_size = int(cfg.contracts)
    Strategy.min_stop_points = float(cfg.min_stop_points)
    Strategy.max_stop_points = float(cfg.max_stop_points)
    Strategy.setup_expiry_bars = int(cfg.setup_expiry_bars)
    Strategy.touch_tolerance_points = float(cfg.touch_tolerance_points)
    Strategy.require_internal_sweep = bool(variant.require_internal_sweep)
    Strategy.require_cos = bool(variant.require_cos)
    Strategy.debug_symbol = cfg.symbol
    Strategy.debug_variant = variant.name

    return Strategy


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(cfg: InstrumentConfig) -> pd.DataFrame:
    df = get_ohlcv(cfg.symbol, cfg.databento_symbol, "1m", days_back=cfg.days_back)
    if df is None or df.empty:
        raise RuntimeError(f"No data loaded for {cfg.symbol}")

    # Standardize column names for Backtesting.py.
    rename = {c: c.capitalize() for c in df.columns if c.lower() in {"open", "high", "low", "close", "volume"}}
    df = df.rename(columns=rename)
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{cfg.symbol} data missing required OHLC columns: {missing}")

    if cfg.tail_rows is not None:
        df = df.tail(int(cfg.tail_rows)).copy()
    else:
        df = df.copy()

    return df


# ---------------------------------------------------------------------------
# True trade accounting
# ---------------------------------------------------------------------------

def _normalize_trades(raw: pd.DataFrame, cfg: InstrumentConfig, variant: str, prop_profile: str) -> pd.DataFrame:
    """Convert Backtesting.py stats._trades into a truth-accounted trade log."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    t = raw.copy()

    # Backtesting.py columns usually include Size, EntryPrice, ExitPrice,
    # EntryTime, ExitTime, PnL, ReturnPct, EntryBar, ExitBar.
    if "Size" not in t.columns:
        raise RuntimeError("Backtesting.py trade log missing Size column; cannot reconstruct actual closed size.")
    if "EntryPrice" not in t.columns or "ExitPrice" not in t.columns:
        raise RuntimeError("Backtesting.py trade log missing EntryPrice/ExitPrice columns.")

    size = pd.to_numeric(t["Size"], errors="coerce")
    entry = pd.to_numeric(t["EntryPrice"], errors="coerce")
    exit_ = pd.to_numeric(t["ExitPrice"], errors="coerce")

    side = np.where(size >= 0, "LONG", "SHORT")
    closed_size = size.abs()
    realized_points = np.where(size >= 0, exit_ - entry, entry - exit_)

    # Engine PnL is points * actual closed size. If Backtesting.py has PnL,
    # use it as the source-of-truth for closed size accounting, but cross-check
    # it against the explicit formula.
    engine_pnl = pd.to_numeric(t.get("PnL", realized_points * closed_size), errors="coerce")
    explicit_engine_pnl = pd.Series(realized_points, index=t.index) * closed_size
    pnl_diff = engine_pnl - explicit_engine_pnl

    realized_pnl_dollars = engine_pnl * float(cfg.dollars_per_point)
    explicit_pnl_dollars = explicit_engine_pnl * float(cfg.dollars_per_point)

    # Commission/slippage placeholders. Keep default zero unless the user passes
    # costs later. The columns make the math auditable.
    commission_dollars = 0.0
    slippage_dollars = 0.0
    net_pnl_dollars = realized_pnl_dollars - commission_dollars - slippage_dollars

    # Risk/R calculations, only if SL exists in stats._trades. If Backtesting.py
    # does not expose SL, R fields stay NaN instead of inventing numbers.
    sl = pd.to_numeric(t["SL"], errors="coerce") if "SL" in t.columns else pd.Series(np.nan, index=t.index)
    risk_points = (entry - sl).abs()
    risk_dollars = risk_points * closed_size * float(cfg.dollars_per_point)
    r_multiple = np.where(risk_dollars > 0, net_pnl_dollars / risk_dollars, np.nan)

    out = pd.DataFrame({
        "symbol": cfg.symbol,
        "variant": variant,
        "prop_profile_requested": prop_profile,
        "side": side,
        "entry_time": pd.to_datetime(t.get("EntryTime", pd.NaT), errors="coerce"),
        "exit_time": pd.to_datetime(t.get("ExitTime", pd.NaT), errors="coerce"),
        "entry_bar": t.get("EntryBar", np.nan),
        "exit_bar": t.get("ExitBar", np.nan),
        "entry_price": entry,
        "exit_price": exit_,
        "closed_size_contracts": closed_size,
        "configured_contracts": int(cfg.contracts),
        "dollars_per_point": float(cfg.dollars_per_point),
        "realized_points": realized_points,
        "engine_pnl_points_x_contracts": engine_pnl,
        "explicit_engine_pnl_points_x_contracts": explicit_engine_pnl,
        "engine_vs_explicit_pnl_diff": pnl_diff,
        "realized_pnl_dollars_gross": realized_pnl_dollars,
        "commission_dollars": commission_dollars,
        "slippage_dollars": slippage_dollars,
        "realized_pnl_dollars_net": net_pnl_dollars,
        "stop_price": sl,
        "risk_points": risk_points,
        "risk_dollars": risk_dollars,
        "r_multiple": r_multiple,
        "return_pct_engine": t.get("ReturnPct", np.nan),
    })

    out["exit_date"] = out["exit_time"].dt.date
    out["exit_month"] = out["exit_time"].dt.to_period("M").astype(str)
    out["exit_year"] = out["exit_time"].dt.year

    # Build a parent position id. This groups partial exits from the same original entry.
    key_cols = ["symbol", "variant", "side", "entry_time", "entry_price"]
    out["position_id"] = pd.factorize(out[key_cols].astype(str).agg("|".join, axis=1))[0] + 1

    # Old wrong formula shown for audit only. Do not use as final PnL.
    out["old_wrong_full_contract_pnl_dollars"] = out["realized_points"] * float(cfg.dollars_per_point) * int(cfg.contracts)
    out["old_wrong_overstatement_dollars"] = out["old_wrong_full_contract_pnl_dollars"] - out["realized_pnl_dollars_net"]

    return out


def run_symbol_variant(cfg: InstrumentConfig, variant_name: str, strategy_module: str, prop_profile: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    variant = VARIANTS[variant_name]
    df = load_data(cfg)
    print(
        f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()} | "
        f"days_back={cfg.days_back} | tail_rows={'none' if cfg.tail_rows is None else cfg.tail_rows}"
    )
    Strategy = build_strategy_class(cfg, variant, strategy_module)
    bt = Backtest(df, Strategy, cash=1_000_000, commission=0, exclusive_orders=False)
    stats = bt.run()
    raw_trades = stats.get("_trades", pd.DataFrame())
    trades = _normalize_trades(raw_trades, cfg, variant_name, prop_profile)
    meta = {
        "engine_trade_rows": float(len(raw_trades)),
        "stats_equity_final": float(stats.get("Equity Final [$]", np.nan)),
        "stats_return_pct": float(stats.get("Return [%]", np.nan)),
        "stats_win_rate_pct": float(stats.get("Win Rate [%]", np.nan)),
        "stats_profit_factor": float(stats.get("Profit Factor", np.nan)),
    }
    return trades, meta


# ---------------------------------------------------------------------------
# Post-trade prop filter based on corrected PnL
# ---------------------------------------------------------------------------

def apply_post_trade_prop_filter(trades: pd.DataFrame, profile_name: str) -> pd.DataFrame:
    """Simple deterministic post-trade filter using corrected net PnL.

    This does not pretend to be live pre-entry gating. It answers: after ordering
    trades chronologically, which trades remain if account-level day loss / max
    loss style rules are applied to corrected PnL?
    """
    if trades.empty or profile_name == "none":
        out = trades.copy()
        out["filter_status"] = "allowed"
        out["filter_reason"] = "none"
        return out

    profile = get_prop_profile(profile_name)
    out_rows = []
    balance = float(profile.starting_balance)
    peak_balance = balance
    current_day = None
    day_pnl = 0.0
    consecutive_losses = 0
    trades_today = 0

    ordered = trades.sort_values(["exit_time", "entry_time", "symbol", "variant"]).copy()

    for _, row in ordered.iterrows():
        day = row["exit_date"]
        if day != current_day:
            current_day = day
            day_pnl = 0.0
            consecutive_losses = 0
            trades_today = 0

        pnl = float(row["realized_pnl_dollars_net"])
        status = "allowed"
        reason = ""

        if profile.daily_loss_limit is not None and day_pnl <= float(profile.daily_loss_limit):
            status = "blocked"
            reason = "daily_loss_already_hit"
        elif profile.max_trades_per_day is not None and trades_today >= int(profile.max_trades_per_day):
            status = "blocked"
            reason = "max_trades_per_day"
        elif profile.max_consecutive_losses_per_day is not None and consecutive_losses >= int(profile.max_consecutive_losses_per_day):
            status = "blocked"
            reason = "max_consecutive_losses_per_day"
        elif profile.trailing_drawdown is not None:
            # Generic conservative drawdown check against peak balance. For FTMO-like
            # static max loss, this may be more conservative than their exact rule.
            if balance <= peak_balance - float(profile.trailing_drawdown):
                status = "blocked"
                reason = "drawdown_limit_already_hit"

        new = row.to_dict()
        new["filter_status"] = status
        new["filter_reason"] = reason
        new["account_balance_before"] = balance
        new["day_pnl_before"] = day_pnl

        if status == "allowed":
            balance += pnl
            day_pnl += pnl
            peak_balance = max(peak_balance, balance)
            trades_today += 1
            consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
            new["realized_pnl_dollars_filtered"] = pnl
        else:
            new["realized_pnl_dollars_filtered"] = 0.0

        new["account_balance_after"] = balance
        new["day_pnl_after"] = day_pnl
        out_rows.append(new)

    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cum = series.cumsum()
    dd = cum - cum.cummax()
    return float(dd.min())


def _max_consecutive_losses(pnl: pd.Series) -> int:
    max_run = 0
    run = 0
    for x in pnl.fillna(0):
        if x < 0:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return int(max_run)


def summarize_pnl(df: pd.DataFrame, pnl_col: str, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{"scope": label, "trades": 0, "total_realized_pnl": 0.0}])
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    total = float(pnl.sum())
    trades = int(len(df))
    win_rate = float((pnl > 0).mean() * 100) if trades else 0.0
    loss_rate = float((pnl < 0).mean() * 100) if trades else 0.0
    pf = gross_profit / abs(gross_loss) if gross_loss < 0 else np.inf
    avg = float(pnl.mean()) if trades else 0.0
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    payoff = avg_win / abs(avg_loss) if avg_loss < 0 else np.inf
    expectancy = avg
    best = float(pnl.max()) if trades else 0.0
    worst = float(pnl.min()) if trades else 0.0

    return pd.DataFrame([{
        "scope": label,
        "trades": trades,
        "winning_trades": int((pnl > 0).sum()),
        "losing_trades": int((pnl < 0).sum()),
        "flat_trades": int((pnl == 0).sum()),
        "win_rate_pct": win_rate,
        "loss_rate_pct": loss_rate,
        "total_realized_pnl": total,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": pf,
        "expectancy_per_trade": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "best_trade": best,
        "worst_trade": worst,
        "max_drawdown_by_trade": _max_drawdown(pnl),
        "max_consecutive_losses": _max_consecutive_losses(pnl),
    }])


def group_summary(df: pd.DataFrame, group_cols: List[str], pnl_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        label = key if isinstance(key, tuple) else (key,)
        summary = summarize_pnl(g, pnl_col, "group").iloc[0].to_dict()
        for col, val in zip(group_cols, label):
            summary[col] = val
        rows.append(summary)
    return pd.DataFrame(rows).sort_values("total_realized_pnl", ascending=False)


def position_summary(trades: pd.DataFrame, pnl_col: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    grouped = trades.groupby(["symbol", "variant", "position_id"], dropna=False).agg(
        entry_time=("entry_time", "min"),
        exit_time=("exit_time", "max"),
        side=("side", "first"),
        entry_price=("entry_price", "first"),
        total_closed_contracts=("closed_size_contracts", "sum"),
        exits=("position_id", "size"),
        realized_pnl_dollars=(pnl_col, "sum"),
        average_r=("r_multiple", "mean"),
    ).reset_index()
    return grouped


def accounting_audit(df: pd.DataFrame, pnl_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    old = pd.to_numeric(df["old_wrong_full_contract_pnl_dollars"], errors="coerce").fillna(0)
    true = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0)
    diff = old - true
    return pd.DataFrame([{
        "rows": len(df),
        "true_pnl_column": pnl_col,
        "true_total_pnl": float(true.sum()),
        "old_wrong_total_full_contract_pnl": float(old.sum()),
        "old_wrong_overstatement": float(diff.sum()),
        "rows_where_old_differs_from_true": int((diff.abs() > 1e-9).sum()),
        "partial_exit_rows_inferred": int((df["closed_size_contracts"] != df["configured_contracts"]).sum()),
        "max_abs_engine_vs_explicit_pnl_diff": float(pd.to_numeric(df["engine_vs_explicit_pnl_diff"], errors="coerce").abs().max() or 0.0),
    }])


def write_outputs(prefix: Path, trades: pd.DataFrame, pnl_col: str) -> Dict[str, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["trade_log"] = prefix.with_name(prefix.name + "_trade_log.csv")
    trades.to_csv(paths["trade_log"], index=False)

    pos = position_summary(trades, pnl_col)
    paths["position_log"] = prefix.with_name(prefix.name + "_position_log.csv")
    pos.to_csv(paths["position_log"], index=False)

    for name, cols in {
        "portfolio_summary": [],
        "symbol_summary": ["symbol"],
        "variant_summary": ["variant"],
        "symbol_variant_summary": ["symbol", "variant"],
        "daily_summary": ["exit_date"],
        "monthly_summary": ["exit_month"],
        "yearly_summary": ["exit_year"],
    }.items():
        if cols:
            out = group_summary(trades, cols, pnl_col)
        else:
            out = summarize_pnl(trades, pnl_col, "portfolio")
        paths[name] = prefix.with_name(prefix.name + f"_{name}.csv")
        out.to_csv(paths[name], index=False)

    paths["accounting_audit"] = prefix.with_name(prefix.name + "_accounting_audit.csv")
    accounting_audit(trades, pnl_col).to_csv(paths["accounting_audit"], index=False)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_contract_overrides(value: Optional[str]) -> Dict[str, int]:
    if not value:
        return {}
    out: Dict[str, int] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        sym, val = part.split(":", 1)
        out[sym.strip().upper()] = int(val)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Truth-accounted ICT top/bottom ticking backtest engine.")
    parser.add_argument("--prop-profile", default="apex_50k_eval", choices=list_prop_profiles())
    parser.add_argument("--symbols", default="MNQ,MES,MYM,MGC,MCL")
    parser.add_argument("--variants", default="type2_baseline,type1_sniper")
    parser.add_argument("--days-back", type=int, default=365)
    parser.add_argument("--tail-rows", type=int, default=120_000)
    parser.add_argument("--no-tail", action="store_true")
    parser.add_argument("--contract-overrides", default=None, help="Example: MNQ:3,MGC:2")
    parser.add_argument("--strategy-module", default="ict_top_bottom_ticking", help="Default: ict_top_bottom_ticking. Can use ict_top_bottom_ticking_management_modes for mode tests.")
    parser.add_argument("--output-dir", default=str(ROOT))
    parser.add_argument("--list-profiles", action="store_true")
    args = parser.parse_args()

    if args.list_profiles:
        print("\n".join(list_prop_profiles()))
        return 0

    symbols = parse_csv_list(args.symbols)
    variants = parse_csv_list(args.variants)
    unknown_symbols = [s for s in symbols if s not in INSTRUMENTS]
    unknown_variants = [v for v in variants if v not in VARIANTS]
    if unknown_symbols:
        raise SystemExit(f"Unknown symbols: {unknown_symbols}. Available: {sorted(INSTRUMENTS)}")
    if unknown_variants:
        raise SystemExit(f"Unknown variants: {unknown_variants}. Available: {sorted(VARIANTS)}")

    contract_overrides = parse_contract_overrides(args.contract_overrides)
    tail_rows = None if args.no_tail else args.tail_rows

    all_raw: List[pd.DataFrame] = []
    meta_rows: List[dict] = []

    for sym in symbols:
        base = INSTRUMENTS[sym]
        cfg = replace(
            base,
            days_back=int(args.days_back),
            tail_rows=tail_rows,
            contracts=int(contract_overrides.get(sym, base.contracts)),
        )
        for variant in variants:
            print(f"\n=== {sym} | {variant} | raw truth run ===")
            try:
                trades, meta = run_symbol_variant(cfg, variant, args.strategy_module, args.prop_profile)
            except Exception as exc:
                print(f"ERROR running {sym} {variant}: {exc}")
                raise
            print(f"{variant} | {sym} closed exit rows={len(trades)} true PnL=${trades['realized_pnl_dollars_net'].sum() if not trades.empty else 0.0:,.2f}")
            all_raw.append(trades)
            meta_rows.append({"symbol": sym, "variant": variant, **meta})

    combined_raw = pd.concat(all_raw, ignore_index=True) if all_raw else pd.DataFrame()
    combined_filtered = apply_post_trade_prop_filter(combined_raw, args.prop_profile)

    suffix = f"{args.prop_profile}_{args.days_back}d_{'notail' if tail_rows is None else f'tail{tail_rows}'}_truth"
    out_dir = Path(args.output_dir)

    raw_prefix = out_dir / f"top_bottom_ticking_raw_{suffix}"
    filtered_prefix = out_dir / f"top_bottom_ticking_filtered_{suffix}"

    raw_paths = write_outputs(raw_prefix, combined_raw, "realized_pnl_dollars_net")
    filtered_paths = write_outputs(filtered_prefix, combined_filtered, "realized_pnl_dollars_filtered")

    meta_path = out_dir / f"top_bottom_ticking_engine_meta_{suffix}.csv"
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    compare = pd.concat([
        summarize_pnl(combined_raw, "realized_pnl_dollars_net", "raw"),
        summarize_pnl(combined_filtered, "realized_pnl_dollars_filtered", "filtered"),
    ], ignore_index=True)
    compare_path = out_dir / f"top_bottom_ticking_compare_{suffix}.csv"
    compare.to_csv(compare_path, index=False)

    print("\n=== TRUE ACCOUNTING SUMMARY ===")
    print(compare.to_string(index=False))
    print("\nSaved key files:")
    print(f"raw trade log      -> {raw_paths['trade_log']}")
    print(f"filtered trade log -> {filtered_paths['trade_log']}")
    print(f"compare            -> {compare_path}")
    print(f"engine meta         -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
