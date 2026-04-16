from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from backtesting import Backtest

from src.data.fetcher import get_ohlcv
from src.strategies.manual.ict_top_bottom_ticking import (
    ICTTopBottomTickingType2Active,
    ICTTopBottomTickingType1Sniper30s,
    ICTTopBottomTickingType1Sniper5s,
)

OUTPUT_DIR = Path(".")
BACKTEST_CASH = 100_000


@dataclass
class RunConfig:
    symbol: str
    exchange: str = "tradovate"
    timeframe: str = "5m"
    days_back: int = 365
    cash: float = BACKTEST_CASH
    commission: float = 0.0
    exclusive_orders: bool = True


VARIANT_MAP = {
    "type2_active": ICTTopBottomTickingType2Active,
    "type1_sniper_30s": ICTTopBottomTickingType1Sniper30s,
    "type1_sniper_5s": ICTTopBottomTickingType1Sniper5s,
}


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for c in out.columns:
        cl = c.lower()
        if cl == "open":
            rename[c] = "Open"
        elif cl == "high":
            rename[c] = "High"
        elif cl == "low":
            rename[c] = "Low"
        elif cl == "close":
            rename[c] = "Close"
        elif cl == "volume":
            rename[c] = "Volume"
    out = out.rename(columns=rename)
    req = ["Open", "High", "Low", "Close"]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    out = out[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out[~out.index.isna()]
    return out.sort_index()


def _load_local_subminute(symbol: str, timeframe: str, days_back: int) -> pd.DataFrame:
    root = Path("src/data/databento_cache")
    candidates = [
        root / f"{symbol}_1s.parquet",
        root / f"{symbol.upper()}_1s.parquet",
        root / "NQ_1s.parquet",
        root / "MNQ_1s.parquet",
    ]
    one_sec = None
    for p in candidates:
        if p.exists():
            one_sec = p
            break
    if one_sec is None:
        raise FileNotFoundError("No local 1-second parquet found in src/data/databento_cache for sub-minute sniper backtests.")
    df = pd.read_parquet(one_sec)
    df = _ensure_ohlcv(df)
    if timeframe == "1s":
        out = df
    elif timeframe == "5s":
        out = df.resample("5s", label="right", closed="right").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna()
    elif timeframe == "30s":
        out = df.resample("30s", label="right", closed="right").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna()
    else:
        raise ValueError(f"Unsupported local sub-minute timeframe: {timeframe}")

    if days_back:
        cutoff = out.index.max() - pd.Timedelta(days=days_back)
        out = out[out.index >= cutoff]
    return out


def load_data(cfg: RunConfig) -> pd.DataFrame:
    if cfg.timeframe in {"30s", "5s", "1s"}:
        return _load_local_subminute(cfg.symbol, cfg.timeframe, cfg.days_back)
    return _ensure_ohlcv(get_ohlcv(cfg.symbol, exchange=cfg.exchange, timeframe=cfg.timeframe, days_back=cfg.days_back))


def _extract_equity_curve(stats) -> pd.DataFrame:
    eq = getattr(stats, "_equity_curve", None)
    if eq is None:
        return pd.DataFrame(columns=["Equity"])
    if not isinstance(eq, pd.DataFrame):
        eq = pd.DataFrame(eq)
    if "Equity" not in eq.columns and len(eq.columns) > 0:
        eq = eq.rename(columns={eq.columns[0]: "Equity"})
    return eq.copy()


def _summary_stats(stats, trades: pd.DataFrame, variant: str, symbol: str) -> dict:
    gross_pnl = float(trades["pnl"].sum()) if not trades.empty else 0.0
    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum()) if not trades.empty else 0.0
    gross_loss = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum()) if not trades.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
    eq = _extract_equity_curve(stats)
    max_closed_dd = np.nan
    if not eq.empty:
        e = eq["Equity"].astype(float)
        max_closed_dd = float((e - e.cummax()).min())
    return {
        "variant": variant,
        "symbol": symbol,
        "trades": len(trades),
        "gross_pnl": gross_pnl,
        "avg_trade": float(trades["pnl"].mean()) if not trades.empty else 0.0,
        "median_trade": float(trades["pnl"].median()) if not trades.empty else 0.0,
        "wins": int((trades["pnl"] > 0).sum()) if not trades.empty else 0,
        "losses": int((trades["pnl"] < 0).sum()) if not trades.empty else 0,
        "breakeven": int((trades["pnl"] == 0).sum()) if not trades.empty else 0,
        "best_trade": float(trades["pnl"].max()) if not trades.empty else np.nan,
        "worst_trade": float(trades["pnl"].min()) if not trades.empty else np.nan,
        "profit_factor": profit_factor,
        "max_closed_trade_drawdown": max_closed_dd,
    }


def _monthly_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce", utc=True)
    out["month"] = out["entry_time"].dt.to_period("M").astype(str)
    out["is_win"] = out["pnl"] > 0
    out["is_loss"] = out["pnl"] < 0
    out["is_be"] = out["pnl"] == 0
    return out.groupby(["variant", "symbol", "month"], dropna=False).agg(
        trades=("pnl", "count"),
        gross_pnl=("pnl", "sum"),
        avg_trade=("pnl", "mean"),
        wins=("is_win", "sum"),
        losses=("is_loss", "sum"),
        breakeven=("is_be", "sum"),
    ).reset_index().sort_values(["variant", "symbol", "month"])


def _daily_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce", utc=True)
    out["day"] = out["entry_time"].dt.date.astype(str)
    out["is_win"] = out["pnl"] > 0
    out["is_loss"] = out["pnl"] < 0
    out["is_be"] = out["pnl"] == 0
    return out.groupby(["variant", "symbol", "day"], dropna=False).agg(
        trades=("pnl", "count"),
        gross_pnl=("pnl", "sum"),
        avg_trade=("pnl", "mean"),
        wins=("is_win", "sum"),
        losses=("is_loss", "sum"),
        breakeven=("is_be", "sum"),
    ).reset_index().sort_values(["variant", "symbol", "day"])


def _losing_clusters(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce", utc=True)
    out = out.sort_values(["variant", "symbol", "entry_time"]).reset_index(drop=True)
    out["is_loss"] = out["pnl"] < 0
    rows = []
    for (variant, symbol), grp in out.groupby(["variant", "symbol"], dropna=False):
        cluster_id = 0
        start = end = None
        count = 0
        pnl = 0.0
        for _, r in grp.iterrows():
            if bool(r["is_loss"]):
                if start is None:
                    cluster_id += 1
                    start = end = r["entry_time"]
                    count = 1
                    pnl = float(r["pnl"])
                else:
                    end = r["entry_time"]
                    count += 1
                    pnl += float(r["pnl"])
            else:
                if start is not None:
                    rows.append({"variant": variant, "symbol": symbol, "cluster_id": cluster_id, "start_time": str(start), "end_time": str(end), "loss_trades": count, "gross_pnl": pnl})
                    start = end = None
                    count = 0
                    pnl = 0.0
        if start is not None:
            rows.append({"variant": variant, "symbol": symbol, "cluster_id": cluster_id, "start_time": str(start), "end_time": str(end), "loss_trades": count, "gross_pnl": pnl})
    return pd.DataFrame(rows)


def run_symbol(cfg: RunConfig, variant_name: str) -> tuple[dict, pd.DataFrame]:
    strategy_cls = VARIANT_MAP[variant_name]
    print(f"\n=== {cfg.symbol} | {variant_name} ===")
    print(f"Loading {cfg.symbol} {cfg.timeframe} data...")
    df = load_data(cfg)
    print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()}")
    bt = Backtest(df, strategy_cls, cash=cfg.cash, commission=cfg.commission, exclusive_orders=cfg.exclusive_orders)
    stats = bt.run()
    trades = pd.DataFrame(getattr(strategy_cls, "last_trade_log", []))
    if trades.empty:
        trades = pd.DataFrame(columns=["side","setup_type","entry_variant","entry_price","exit_price","entry_time","exit_time","pnl","return_pct","external_level","zone_high","zone_low","planned_entry_price","planned_stop_price","planned_target1_price","planned_target2_price","planned_target3_price","internal_sweep"])
    trades["variant"] = variant_name
    trades["symbol"] = cfg.symbol
    return _summary_stats(stats, trades, variant_name, cfg.symbol), trades


def run_all_symbols(symbols: Iterable[str] | None = None) -> None:
    if symbols is None:
        symbols = ["MNQ"]
    variant_cfgs = [
        ("type2_active", RunConfig(symbol="MNQ", timeframe="5m", days_back=365)),
        ("type1_sniper_30s", RunConfig(symbol="MNQ", timeframe="30s", days_back=120)),
        ("type1_sniper_5s", RunConfig(symbol="MNQ", timeframe="5s", days_back=45)),
    ]
    summaries = []
    all_trades = []
    for raw_symbol in symbols:
        symbol = raw_symbol.strip().upper()
        for variant_name, cfg in variant_cfgs:
            cfg.symbol = symbol
            summary, trades = run_symbol(cfg, variant_name)
            summaries.append(summary)
            all_trades.append(trades)
    summary_df = pd.DataFrame(summaries).sort_values(["variant","symbol"]).reset_index(drop=True)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    monthly_df = _monthly_summary(trades_df)
    daily_df = _daily_summary(trades_df)
    losing_df = _losing_clusters(trades_df)
    summary_df.to_csv(OUTPUT_DIR / "top_bottom_ticking_variant_summary.csv", index=False)
    trades_df.to_csv(OUTPUT_DIR / "top_bottom_ticking_all_trades.csv", index=False)
    monthly_df.to_csv(OUTPUT_DIR / "top_bottom_ticking_monthly_summary.csv", index=False)
    daily_df.to_csv(OUTPUT_DIR / "top_bottom_ticking_daily_summary.csv", index=False)
    losing_df.to_csv(OUTPUT_DIR / "top_bottom_ticking_losing_clusters.csv", index=False)
    print("\nSaved:")
    print(" - top_bottom_ticking_variant_summary.csv")
    print(" - top_bottom_ticking_all_trades.csv")
    print(" - top_bottom_ticking_monthly_summary.csv")
    print(" - top_bottom_ticking_daily_summary.csv")
    print(" - top_bottom_ticking_losing_clusters.csv")
    if not summary_df.empty:
        print("\nVariant summary:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_all_symbols(["MNQ"])
