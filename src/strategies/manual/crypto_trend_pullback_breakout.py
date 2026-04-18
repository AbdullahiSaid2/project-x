from __future__ import annotations

"""
Crypto trend-pullback + breakout-retest strategy runner v4.

Intent
------
A fresh crypto-native strategy designed around the patterns that tend to code
more robustly on crypto than pure intraday reversal models:

- H1 trend alignment
- 5m breakout in trend direction
- retest / pullback continuation entries
- ATR-based stops and targets
- partial at 1R, breakeven lock, runner to 2R

This is not guaranteed to be the "most profitable" model. It is the strongest
systematic candidate to test next given the prior backtest results.

Example
-------
PYTHONPATH=. python -m src.strategies.manual.crypto_trend_pullback_breakout \
  --symbols BTC,ETH,SOL --days-back 365 --timeframe 5m \
  --risk-usd 100 --position-usd 15000 --leverage 3
"""

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent


def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "src").exists():
            return p
    return start.parent


PROJECT_ROOT = _find_project_root(THIS_DIR)
for p in (PROJECT_ROOT, THIS_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


def _import_get_crypto_ohlcv():
    try:
        from src.data.ccxt_fetcher import get_crypto_ohlcv
        return get_crypto_ohlcv
    except Exception:
        pass

    try:
        from ccxt_fetcher import get_crypto_ohlcv
        return get_crypto_ohlcv
    except Exception:
        pass

    candidate = PROJECT_ROOT / "src" / "data" / "ccxt_fetcher.py"
    if candidate.exists():
        spec = importlib.util.spec_from_file_location("ccxt_fetcher_local", candidate)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "get_crypto_ohlcv"):
                return getattr(module, "get_crypto_ohlcv")

    raise ImportError("Could not import get_crypto_ohlcv from src.data.ccxt_fetcher or local fallback")


get_crypto_ohlcv = _import_get_crypto_ohlcv()


@dataclass(frozen=True)
class InstrumentConfig:
    symbol: str
    exchange_id: str
    timeframe: str
    days_back: int
    tail_rows: int
    fee_rate: float
    risk_usd: float
    position_usd: float
    leverage: float


@dataclass(frozen=True)
class SymbolSpec:
    tick_size: float
    h1_fast_ema: int
    h1_slow_ema: int
    h1_trend_slope_bars: int
    ltf_fast_ema: int
    breakout_lookback: int
    breakout_buffer_atr: float
    pullback_buffer_atr: float
    retest_tolerance_atr: float
    stop_atr_mult: float
    min_stop_atr: float
    max_stop_atr: float
    min_breakout_body_atr: float
    min_breakout_close_frac: float
    min_volume_ratio: float
    volume_lookback: int
    setup_expiry_bars: int
    max_hold_bars: int
    cooldown_bars: int
    tp1_r: float
    tp2_r: float
    partial1_frac: float
    breakeven_lock_r: float
    trail_activate_r: float
    trail_stop_r: float
    min_h1_distance_from_fast_ema_atr: float
    allow_pullback_to_ema: bool
    entry_touch_tolerance_atr: float
    post_touch_expiry_bars: int
    pending_invalidation_buffer_atr: float


DEFAULT_INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "BTC": InstrumentConfig("BTC", "binance", "5m", 365, 200_000, 0.0006, 100.0, 15_000.0, 3.0),
    "ETH": InstrumentConfig("ETH", "binance", "5m", 365, 200_000, 0.0006, 100.0, 15_000.0, 3.0),
    "SOL": InstrumentConfig("SOL", "binance", "5m", 365, 200_000, 0.0006, 100.0, 15_000.0, 3.0),
}


SYMBOL_SPECS: Dict[str, SymbolSpec] = {
    "BTC": SymbolSpec(
        tick_size=0.10,
        h1_fast_ema=50,
        h1_slow_ema=200,
        h1_trend_slope_bars=3,
        ltf_fast_ema=20,
        breakout_lookback=36,
        breakout_buffer_atr=0.08,
        pullback_buffer_atr=0.18,
        retest_tolerance_atr=0.36,
        stop_atr_mult=0.70,
        min_stop_atr=0.55,
        max_stop_atr=2.40,
        min_breakout_body_atr=0.24,
        min_breakout_close_frac=0.64,
        min_volume_ratio=1.00,
        volume_lookback=24,
        setup_expiry_bars=24,
        max_hold_bars=108,
        cooldown_bars=4,
        tp1_r=1.00,
        tp2_r=2.00,
        partial1_frac=0.65,
        breakeven_lock_r=0.05,
        trail_activate_r=1.60,
        trail_stop_r=0.70,
        min_h1_distance_from_fast_ema_atr=0.05,
        allow_pullback_to_ema=True,
        entry_touch_tolerance_atr=0.28,
        post_touch_expiry_bars=8,
        pending_invalidation_buffer_atr=0.44,
    ),
    "ETH": SymbolSpec(
        tick_size=0.01,
        h1_fast_ema=50,
        h1_slow_ema=200,
        h1_trend_slope_bars=3,
        ltf_fast_ema=20,
        breakout_lookback=32,
        breakout_buffer_atr=0.07,
        pullback_buffer_atr=0.18,
        retest_tolerance_atr=0.32,
        stop_atr_mult=0.68,
        min_stop_atr=0.52,
        max_stop_atr=2.30,
        min_breakout_body_atr=0.21,
        min_breakout_close_frac=0.62,
        min_volume_ratio=0.98,
        volume_lookback=24,
        setup_expiry_bars=24,
        max_hold_bars=108,
        cooldown_bars=4,
        tp1_r=1.00,
        tp2_r=1.90,
        partial1_frac=0.65,
        breakeven_lock_r=0.05,
        trail_activate_r=1.55,
        trail_stop_r=0.68,
        min_h1_distance_from_fast_ema_atr=0.05,
        allow_pullback_to_ema=True,
        entry_touch_tolerance_atr=0.30,
        post_touch_expiry_bars=8,
        pending_invalidation_buffer_atr=0.44,
    ),
    "SOL": SymbolSpec(
        tick_size=0.001,
        h1_fast_ema=50,
        h1_slow_ema=200,
        h1_trend_slope_bars=3,
        ltf_fast_ema=20,
        breakout_lookback=28,
        breakout_buffer_atr=0.06,
        pullback_buffer_atr=0.16,
        retest_tolerance_atr=0.36,
        stop_atr_mult=0.62,
        min_stop_atr=0.48,
        max_stop_atr=2.10,
        min_breakout_body_atr=0.19,
        min_breakout_close_frac=0.59,
        min_volume_ratio=0.95,
        volume_lookback=20,
        setup_expiry_bars=20,
        max_hold_bars=96,
        cooldown_bars=3,
        tp1_r=0.90,
        tp2_r=1.70,
        partial1_frac=0.68,
        breakeven_lock_r=0.06,
        trail_activate_r=1.35,
        trail_stop_r=0.62,
        min_h1_distance_from_fast_ema_atr=0.04,
        allow_pullback_to_ema=True,
        entry_touch_tolerance_atr=0.34,
        post_touch_expiry_bars=7,
        pending_invalidation_buffer_atr=0.48,
    ),
}


OUT_TRADE_CSV = THIS_DIR / "crypto_trend_pullback_breakout_v4_trade_log.csv"
OUT_DEBUG_CSV = THIS_DIR / "crypto_trend_pullback_breakout_v4_debug_counts.csv"
OUT_VARIANT_SUMMARY_CSV = THIS_DIR / "crypto_trend_pullback_breakout_v4_variant_summary.csv"
OUT_MONTHLY_CSV = THIS_DIR / "crypto_trend_pullback_breakout_v4_monthly_summary.csv"
OUT_DAILY_CSV = THIS_DIR / "crypto_trend_pullback_breakout_v4_daily_summary.csv"


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _safe_float(x: float) -> float:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return float("nan")
    return float(x)


def _round_tick(value: float, tick_size: float) -> float:
    if tick_size <= 0:
        return float(value)
    return round(float(value) / tick_size) * tick_size


def build_feature_frame(df: pd.DataFrame, spec: SymbolSpec) -> pd.DataFrame:
    m = df.copy().sort_index()
    m.columns = [c.capitalize() for c in m.columns]
    if m.index.tz is None:
        m.index = pd.DatetimeIndex(m.index).tz_localize("UTC")
    else:
        m.index = pd.DatetimeIndex(m.index).tz_convert("UTC")

    m["atr14"] = _atr(m, 14)
    m["ema20"] = _ema(m["Close"], spec.ltf_fast_ema)
    m["vol_ma"] = m["Volume"].rolling(spec.volume_lookback).mean()
    m["range"] = (m["High"] - m["Low"]).replace(0.0, np.nan)
    m["body"] = (m["Close"] - m["Open"]).abs()
    m["bull_close_frac"] = (m["Close"] - m["Low"]) / m["range"]
    m["bear_close_frac"] = (m["High"] - m["Close"]) / m["range"]
    m["prev_range_high"] = m["High"].rolling(spec.breakout_lookback).max().shift(1)
    m["prev_range_low"] = m["Low"].rolling(spec.breakout_lookback).min().shift(1)
    m["vol_ratio"] = m["Volume"] / m["vol_ma"].replace(0.0, np.nan)

    h1 = (
        m[["Open", "High", "Low", "Close", "Volume"]]
        .resample("1H", label="right", closed="right")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )
    h1["atr14"] = _atr(h1, 14)
    h1["ema_fast"] = _ema(h1["Close"], spec.h1_fast_ema)
    h1["ema_slow"] = _ema(h1["Close"], spec.h1_slow_ema)
    h1["ema_fast_prev"] = h1["ema_fast"].shift(spec.h1_trend_slope_bars)
    h1["bias_bull"] = (
        (h1["Close"] > h1["ema_fast"]) &
        (h1["ema_fast"] > h1["ema_slow"]) &
        ((h1["ema_fast"] - h1["ema_fast_prev"]) > 0) &
        ((h1["Close"] - h1["ema_fast"]).abs() >= h1["atr14"] * spec.min_h1_distance_from_fast_ema_atr)
    )
    h1["bias_bear"] = (
        (h1["Close"] < h1["ema_fast"]) &
        (h1["ema_fast"] < h1["ema_slow"]) &
        ((h1["ema_fast"] - h1["ema_fast_prev"]) < 0) &
        ((h1["Close"] - h1["ema_fast"]).abs() >= h1["atr14"] * spec.min_h1_distance_from_fast_ema_atr)
    )
    h1_ctx = h1[["bias_bull", "bias_bear", "ema_fast", "ema_slow", "atr14"]].rename(
        columns={"ema_fast": "h1_ema_fast", "ema_slow": "h1_ema_slow", "atr14": "h1_atr14"}
    )
    m = pd.merge_asof(m.sort_index(), h1_ctx.sort_index(), left_index=True, right_index=True, direction="backward")

    m["breakout_long"] = (
        m["bias_bull"]
        & (m["Close"] > (m["prev_range_high"] + m["atr14"] * spec.breakout_buffer_atr))
        & (m["body"] >= m["atr14"] * spec.min_breakout_body_atr)
        & (m["bull_close_frac"] >= spec.min_breakout_close_frac)
        & (m["vol_ratio"] >= spec.min_volume_ratio)
    )
    m["breakout_short"] = (
        m["bias_bear"]
        & (m["Close"] < (m["prev_range_low"] - m["atr14"] * spec.breakout_buffer_atr))
        & (m["body"] >= m["atr14"] * spec.min_breakout_body_atr)
        & (m["bear_close_frac"] >= spec.min_breakout_close_frac)
        & (m["vol_ratio"] >= spec.min_volume_ratio)
    )
    return m


@dataclass
class PendingSetup:
    side: str
    reason: str
    created_i: int
    expires_i: int
    trigger_level: float
    entry_level: float
    stop_level: float
    risk_per_unit: float
    tp1: float
    tp2: float
    atr_at_signal: float
    touch_seen: bool = False
    expiry_extended: bool = False


@dataclass
class OpenTrade:
    side: str
    reason: str
    entry_i: int
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    tp1: float
    tp2: float
    risk_per_unit: float
    initial_stop_price: float
    qty: float
    qty_open: float
    qty_partial1: float
    qty_runner: float
    fee_rate: float
    symbol: str
    notional_usd: float
    initial_risk_usd: float
    partial1_taken: bool = False
    be_moved: bool = False
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    gross_pnl_quote: float = 0.0
    fees_quote: float = 0.0
    hold_bars: int = 0
    max_favorable_r: float = 0.0


def _risk_size(entry: float, stop: float, cfg: InstrumentConfig) -> Tuple[float, float, float]:
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0 or not np.isfinite(risk_per_unit):
        return 0.0, 0.0, 0.0
    raw_qty = cfg.risk_usd / risk_per_unit
    max_notional = cfg.position_usd * cfg.leverage
    capped_qty = min(raw_qty, max_notional / max(entry, 1e-9))
    notional = capped_qty * entry
    effective_risk = capped_qty * risk_per_unit
    return capped_qty, notional, effective_risk


def _entry_touched(row: pd.Series, entry: float, tol: float = 0.0) -> bool:
    return row["Low"] <= (entry + tol) and row["High"] >= (entry - tol)


def _effective_fill_price(row: pd.Series, side: str, entry: float, tol: float) -> float:
    low = float(row["Low"])
    high = float(row["High"])
    close = float(row["Close"])
    if low <= entry <= high:
        return float(entry)
    if side == "LONG":
        return min(close, entry + tol)
    return max(close, entry - tol)


def _close_trade(trade: OpenTrade, when: pd.Timestamp, price: float, reason: str) -> dict:
    trade.exit_time = when
    trade.exit_price = float(price)
    trade.exit_reason = reason
    net = trade.gross_pnl_quote - trade.fees_quote
    denom = max(trade.initial_risk_usd, 1e-9)
    return {
        "symbol": trade.symbol,
        "variant": "trend_pullback_breakout",
        "side": trade.side,
        "reason": trade.reason,
        "entry_time": trade.entry_time,
        "exit_time": trade.exit_time,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "stop_price": trade.stop_price,
        "tp1": trade.tp1,
        "tp2": trade.tp2,
        "qty": trade.qty,
        "initial_stop_price": trade.initial_stop_price,
        "initial_risk_usd": trade.initial_risk_usd,
        "notional_usd": trade.notional_usd,
        "gross_pnl_quote": trade.gross_pnl_quote,
        "fees_quote": trade.fees_quote,
        "net_pnl_quote": net,
        "realized_r": net / denom,
        "hold_bars": trade.hold_bars,
        "exit_reason": reason,
        "partial1_taken": trade.partial1_taken,
        "be_moved": trade.be_moved,
        "max_favorable_r": trade.max_favorable_r,
    }


def _update_max_favorable_r(trade: OpenTrade, row: pd.Series) -> None:
    if trade.side == "LONG":
        mfe = float(row["High"]) - trade.entry_price
    else:
        mfe = trade.entry_price - float(row["Low"])
    trade.max_favorable_r = max(trade.max_favorable_r, mfe / max(trade.risk_per_unit, 1e-9))


def _fee_offset_per_unit(entry: float, fee_rate: float) -> float:
    return max(entry * fee_rate * 2.0, 0.0)


def _build_pending_long(row: pd.Series, i: int, spec: SymbolSpec) -> Optional[PendingSetup]:
    atr = _safe_float(row["atr14"])
    trig = _safe_float(row["prev_range_high"])
    if not np.isfinite(atr) or not np.isfinite(trig):
        return None
    entry = max(trig - atr * spec.pullback_buffer_atr, _safe_float(row["ema20"]) - atr * spec.retest_tolerance_atr) if spec.allow_pullback_to_ema else trig - atr * spec.pullback_buffer_atr
    entry = max(entry, trig - atr * (spec.pullback_buffer_atr + spec.retest_tolerance_atr))
    stop_dist = min(max(atr * spec.stop_atr_mult, atr * spec.min_stop_atr), atr * spec.max_stop_atr)
    stop = entry - stop_dist
    risk = entry - stop
    if risk <= 0:
        return None
    return PendingSetup(
        side="LONG",
        reason="breakout_retest",
        created_i=i,
        expires_i=i + spec.setup_expiry_bars,
        trigger_level=trig,
        entry_level=entry,
        stop_level=stop,
        risk_per_unit=risk,
        tp1=entry + risk * spec.tp1_r,
        tp2=entry + risk * spec.tp2_r,
        atr_at_signal=atr,
    )


def _build_pending_short(row: pd.Series, i: int, spec: SymbolSpec) -> Optional[PendingSetup]:
    atr = _safe_float(row["atr14"])
    trig = _safe_float(row["prev_range_low"])
    if not np.isfinite(atr) or not np.isfinite(trig):
        return None
    entry = min(trig + atr * spec.pullback_buffer_atr, _safe_float(row["ema20"]) + atr * spec.retest_tolerance_atr) if spec.allow_pullback_to_ema else trig + atr * spec.pullback_buffer_atr
    entry = min(entry, trig + atr * (spec.pullback_buffer_atr + spec.retest_tolerance_atr))
    stop_dist = min(max(atr * spec.stop_atr_mult, atr * spec.min_stop_atr), atr * spec.max_stop_atr)
    stop = entry + stop_dist
    risk = stop - entry
    if risk <= 0:
        return None
    return PendingSetup(
        side="SHORT",
        reason="breakout_retest",
        created_i=i,
        expires_i=i + spec.setup_expiry_bars,
        trigger_level=trig,
        entry_level=entry,
        stop_level=stop,
        risk_per_unit=risk,
        tp1=entry - risk * spec.tp1_r,
        tp2=entry - risk * spec.tp2_r,
        atr_at_signal=atr,
    )


def _retest_confirm_long(row: pd.Series, pending: PendingSetup, spec: SymbolSpec) -> bool:
    touched_zone = row["Low"] <= (pending.entry_level + pending.atr_at_signal * spec.retest_tolerance_atr)
    strong_close = row["Close"] > row["Open"] and row["bull_close_frac"] >= 0.55
    reclaims_trigger = row["Close"] >= pending.trigger_level
    return bool(touched_zone and strong_close and reclaims_trigger)


def _retest_confirm_short(row: pd.Series, pending: PendingSetup, spec: SymbolSpec) -> bool:
    touched_zone = row["High"] >= (pending.entry_level - pending.atr_at_signal * spec.retest_tolerance_atr)
    strong_close = row["Close"] < row["Open"] and row["bear_close_frac"] >= 0.55
    reclaims_trigger = row["Close"] <= pending.trigger_level
    return bool(touched_zone and strong_close and reclaims_trigger)


def run_symbol(cfg: InstrumentConfig, spec: SymbolSpec) -> Tuple[pd.DataFrame, Dict[str, float]]:
    print(f"\n=== {cfg.symbol} | trend_pullback_breakout | {cfg.exchange_id} | {cfg.timeframe} ===")
    df = get_crypto_ohlcv(cfg.symbol, timeframe=cfg.timeframe, days_back=cfg.days_back, exchange_id=cfg.exchange_id)
    df = df.tail(cfg.tail_rows)
    print(f"Loaded {len(df):,} rows | start={df.index.min()} end={df.index.max()}")
    m = build_feature_frame(df, spec)

    debug: Dict[str, float] = {
        "blocked_h1_bias": 0,
        "breakout_long_seen": 0,
        "breakout_short_seen": 0,
        "pending_long_armed": 0,
        "pending_short_armed": 0,
        "pending_expired": 0,
        "pending_not_touched": 0,
        "pending_invalidated": 0,
        "pending_touch_seen": 0,
        "pending_near_touch": 0,
        "pending_expiry_extended": 0,
        "confirm_long_seen": 0,
        "confirm_short_seen": 0,
        "entry_long": 0,
        "entry_short": 0,
        "partial1": 0,
        "be_move": 0,
        "trail_lock": 0,
        "exit_stop": 0,
        "exit_tp2": 0,
        "exit_timeout": 0,
        "cooldown_blocked": 0,
    }

    trades: List[dict] = []
    pending: Optional[PendingSetup] = None
    open_trade: Optional[OpenTrade] = None
    last_exit_i = -10_000
    idx = list(m.index)

    warmup = max(spec.h1_slow_ema + 10, spec.breakout_lookback + 20, 250)
    for i in range(warmup, len(m)):
        row = m.iloc[i]
        ts = idx[i]

        if open_trade is not None:
            open_trade.hold_bars += 1
            _update_max_favorable_r(open_trade, row)

            if open_trade.partial1_taken and open_trade.max_favorable_r >= spec.trail_activate_r:
                if open_trade.side == "LONG":
                    trail_stop = open_trade.entry_price + open_trade.risk_per_unit * spec.trail_stop_r
                    if trail_stop > open_trade.stop_price:
                        open_trade.stop_price = trail_stop
                        debug["trail_lock"] += 1
                else:
                    trail_stop = open_trade.entry_price - open_trade.risk_per_unit * spec.trail_stop_r
                    if trail_stop < open_trade.stop_price:
                        open_trade.stop_price = trail_stop
                        debug["trail_lock"] += 1

            if open_trade.side == "LONG":
                if row["Low"] <= open_trade.stop_price:
                    exit_px = open_trade.stop_price
                    qty = open_trade.qty_open
                    open_trade.gross_pnl_quote += qty * (exit_px - open_trade.entry_price)
                    open_trade.fees_quote += qty * (open_trade.entry_price + exit_px) * open_trade.fee_rate
                    open_trade.qty_open = 0.0
                    debug["exit_stop"] += 1
                    trades.append(_close_trade(open_trade, ts, exit_px, "stop"))
                    open_trade = None
                    last_exit_i = i
                    pending = None
                    continue
                if (not open_trade.partial1_taken) and row["High"] >= open_trade.tp1:
                    exit_px = open_trade.tp1
                    qty = open_trade.qty_partial1
                    open_trade.gross_pnl_quote += qty * (exit_px - open_trade.entry_price)
                    open_trade.fees_quote += qty * (open_trade.entry_price + exit_px) * open_trade.fee_rate
                    open_trade.qty_open -= qty
                    open_trade.partial1_taken = True
                    debug["partial1"] += 1
                    if not open_trade.be_moved:
                        be_offset = max(open_trade.risk_per_unit * spec.breakeven_lock_r, _fee_offset_per_unit(open_trade.entry_price, open_trade.fee_rate))
                        open_trade.stop_price = open_trade.entry_price + be_offset
                        open_trade.be_moved = True
                        debug["be_move"] += 1
                if row["High"] >= open_trade.tp2:
                    exit_px = open_trade.tp2
                    qty = open_trade.qty_open
                    open_trade.gross_pnl_quote += qty * (exit_px - open_trade.entry_price)
                    open_trade.fees_quote += qty * (open_trade.entry_price + exit_px) * open_trade.fee_rate
                    open_trade.qty_open = 0.0
                    debug["exit_tp2"] += 1
                    trades.append(_close_trade(open_trade, ts, exit_px, "tp2"))
                    open_trade = None
                    last_exit_i = i
                    pending = None
                    continue
            else:
                if row["High"] >= open_trade.stop_price:
                    exit_px = open_trade.stop_price
                    qty = open_trade.qty_open
                    open_trade.gross_pnl_quote += qty * (open_trade.entry_price - exit_px)
                    open_trade.fees_quote += qty * (open_trade.entry_price + exit_px) * open_trade.fee_rate
                    open_trade.qty_open = 0.0
                    debug["exit_stop"] += 1
                    trades.append(_close_trade(open_trade, ts, exit_px, "stop"))
                    open_trade = None
                    last_exit_i = i
                    pending = None
                    continue
                if (not open_trade.partial1_taken) and row["Low"] <= open_trade.tp1:
                    exit_px = open_trade.tp1
                    qty = open_trade.qty_partial1
                    open_trade.gross_pnl_quote += qty * (open_trade.entry_price - exit_px)
                    open_trade.fees_quote += qty * (open_trade.entry_price + exit_px) * open_trade.fee_rate
                    open_trade.qty_open -= qty
                    open_trade.partial1_taken = True
                    debug["partial1"] += 1
                    if not open_trade.be_moved:
                        be_offset = max(open_trade.risk_per_unit * spec.breakeven_lock_r, _fee_offset_per_unit(open_trade.entry_price, open_trade.fee_rate))
                        open_trade.stop_price = open_trade.entry_price - be_offset
                        open_trade.be_moved = True
                        debug["be_move"] += 1
                if row["Low"] <= open_trade.tp2:
                    exit_px = open_trade.tp2
                    qty = open_trade.qty_open
                    open_trade.gross_pnl_quote += qty * (open_trade.entry_price - exit_px)
                    open_trade.fees_quote += qty * (open_trade.entry_price + exit_px) * open_trade.fee_rate
                    open_trade.qty_open = 0.0
                    debug["exit_tp2"] += 1
                    trades.append(_close_trade(open_trade, ts, exit_px, "tp2"))
                    open_trade = None
                    last_exit_i = i
                    pending = None
                    continue

            if open_trade is not None and open_trade.hold_bars >= spec.max_hold_bars:
                exit_px = float(row["Close"])
                qty = open_trade.qty_open
                if open_trade.side == "LONG":
                    open_trade.gross_pnl_quote += qty * (exit_px - open_trade.entry_price)
                else:
                    open_trade.gross_pnl_quote += qty * (open_trade.entry_price - exit_px)
                open_trade.fees_quote += qty * (open_trade.entry_price + exit_px) * open_trade.fee_rate
                open_trade.qty_open = 0.0
                debug["exit_timeout"] += 1
                trades.append(_close_trade(open_trade, ts, exit_px, "timeout"))
                open_trade = None
                last_exit_i = i
                pending = None
                continue

        if open_trade is not None:
            continue

        if pending is not None:
            if i > pending.expires_i:
                debug["pending_expired"] += 1
                debug["pending_not_touched"] += 1
                pending = None
            else:
                if pending.side == "LONG":
                    touch_tol = pending.atr_at_signal * spec.entry_touch_tolerance_atr
                    touched = _entry_touched(row, pending.entry_level, touch_tol)
                    near_touched = (float(row["Low"]) <= pending.entry_level + touch_tol * 1.75) and (float(row["High"]) >= pending.entry_level - touch_tol * 0.50)
                    if near_touched and not pending.touch_seen:
                        debug["pending_near_touch"] += 1
                    if touched and not pending.touch_seen:
                        pending.touch_seen = True
                        debug["pending_touch_seen"] += 1
                        if (not pending.expiry_extended):
                            pending.expires_i += spec.post_touch_expiry_bars
                            pending.expiry_extended = True
                            debug["pending_expiry_extended"] += 1
                    if (not touched) and near_touched and (not pending.expiry_extended):
                        pending.expires_i += max(1, spec.post_touch_expiry_bars // 2)
                        pending.expiry_extended = True
                        debug["pending_expiry_extended"] += 1
                    if row["Close"] < (pending.stop_level - pending.atr_at_signal * spec.pending_invalidation_buffer_atr) and not near_touched:
                        debug["pending_invalidated"] += 1
                        pending = None
                    elif _retest_confirm_long(row, pending, spec) and (touched or near_touched):
                        fill_price = _round_tick(_effective_fill_price(row, "LONG", pending.entry_level, touch_tol), spec.tick_size)
                        stop_price = _round_tick(pending.stop_level, spec.tick_size)
                        if fill_price <= stop_price:
                            pending = None
                            continue
                        qty, notional, effective_risk = _risk_size(fill_price, stop_price, cfg)
                        if qty > 0:
                            actual_risk_per_unit = fill_price - stop_price
                            debug["confirm_long_seen"] += 1
                            debug["entry_long"] += 1
                            open_trade = OpenTrade(
                                side="LONG",
                                reason=pending.reason,
                                entry_i=i,
                                entry_time=ts,
                                entry_price=fill_price,
                                stop_price=stop_price,
                                tp1=_round_tick(fill_price + actual_risk_per_unit * spec.tp1_r, spec.tick_size),
                                tp2=_round_tick(fill_price + actual_risk_per_unit * spec.tp2_r, spec.tick_size),
                                risk_per_unit=actual_risk_per_unit,
                                initial_stop_price=stop_price,
                                qty=qty,
                                qty_open=qty,
                                qty_partial1=qty * spec.partial1_frac,
                                qty_runner=qty * (1.0 - spec.partial1_frac),
                                fee_rate=cfg.fee_rate,
                                symbol=cfg.symbol,
                                notional_usd=notional,
                                initial_risk_usd=effective_risk,
                            )
                            pending = None
                            continue
                else:
                    touch_tol = pending.atr_at_signal * spec.entry_touch_tolerance_atr
                    touched = _entry_touched(row, pending.entry_level, touch_tol)
                    near_touched = (float(row["High"]) >= pending.entry_level - touch_tol * 1.75) and (float(row["Low"]) <= pending.entry_level + touch_tol * 0.50)
                    if near_touched and not pending.touch_seen:
                        debug["pending_near_touch"] += 1
                    if touched and not pending.touch_seen:
                        pending.touch_seen = True
                        debug["pending_touch_seen"] += 1
                        if (not pending.expiry_extended):
                            pending.expires_i += spec.post_touch_expiry_bars
                            pending.expiry_extended = True
                            debug["pending_expiry_extended"] += 1
                    if (not touched) and near_touched and (not pending.expiry_extended):
                        pending.expires_i += max(1, spec.post_touch_expiry_bars // 2)
                        pending.expiry_extended = True
                        debug["pending_expiry_extended"] += 1
                    if row["Close"] > (pending.stop_level + pending.atr_at_signal * spec.pending_invalidation_buffer_atr) and not near_touched:
                        debug["pending_invalidated"] += 1
                        pending = None
                    elif _retest_confirm_short(row, pending, spec) and (touched or near_touched):
                        fill_price = _round_tick(_effective_fill_price(row, "SHORT", pending.entry_level, touch_tol), spec.tick_size)
                        stop_price = _round_tick(pending.stop_level, spec.tick_size)
                        if fill_price >= stop_price:
                            pending = None
                            continue
                        qty, notional, effective_risk = _risk_size(fill_price, stop_price, cfg)
                        if qty > 0:
                            actual_risk_per_unit = stop_price - fill_price
                            debug["confirm_short_seen"] += 1
                            debug["entry_short"] += 1
                            open_trade = OpenTrade(
                                side="SHORT",
                                reason=pending.reason,
                                entry_i=i,
                                entry_time=ts,
                                entry_price=fill_price,
                                stop_price=stop_price,
                                tp1=_round_tick(fill_price - actual_risk_per_unit * spec.tp1_r, spec.tick_size),
                                tp2=_round_tick(fill_price - actual_risk_per_unit * spec.tp2_r, spec.tick_size),
                                risk_per_unit=actual_risk_per_unit,
                                initial_stop_price=stop_price,
                                qty=qty,
                                qty_open=qty,
                                qty_partial1=qty * spec.partial1_frac,
                                qty_runner=qty * (1.0 - spec.partial1_frac),
                                fee_rate=cfg.fee_rate,
                                symbol=cfg.symbol,
                                notional_usd=notional,
                                initial_risk_usd=effective_risk,
                            )
                            pending = None
                            continue

        if pending is not None or open_trade is not None:
            continue

        if i - last_exit_i < spec.cooldown_bars:
            debug["cooldown_blocked"] += 1
            continue

        if bool(row.get("breakout_long", False)):
            debug["breakout_long_seen"] += 1
            setup = _build_pending_long(row, i, spec)
            if setup is not None:
                pending = setup
                debug["pending_long_armed"] += 1
            continue

        if bool(row.get("breakout_short", False)):
            debug["breakout_short_seen"] += 1
            setup = _build_pending_short(row, i, spec)
            if setup is not None:
                pending = setup
                debug["pending_short_armed"] += 1
            continue

        if not bool(row.get("bias_bull", False)) and not bool(row.get("bias_bear", False)):
            debug["blocked_h1_bias"] += 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, debug

    trades_df["symbol"] = cfg.symbol
    trades_df["timeframe"] = cfg.timeframe
    trades_df["exchange"] = cfg.exchange_id
    trades_df["risk_usd"] = cfg.risk_usd
    trades_df["position_usd"] = cfg.position_usd
    trades_df["leverage"] = cfg.leverage
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True)
    trades_df["exit_date"] = trades_df["exit_time"].dt.date
    trades_df["exit_month"] = trades_df["exit_time"].dt.to_period("M").astype(str)
    trades_df["effective_risk_usd"] = trades_df["initial_risk_usd"]
    trades_df["margin_required_usd"] = trades_df["notional_usd"] / max(cfg.leverage, 1e-9)
    return trades_df, debug


def summarize_results(all_trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if all_trades.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    variant_summary = (
        all_trades.groupby(["symbol", "variant"], dropna=False)
        .agg(
            trades=("net_pnl_quote", "size"),
            wins=("net_pnl_quote", lambda s: int((s > 0).sum())),
            total_net_pnl=("net_pnl_quote", "sum"),
            avg_net_pnl=("net_pnl_quote", "mean"),
            avg_realized_r=("realized_r", "mean"),
            median_realized_r=("realized_r", "median"),
            total_initial_risk=("effective_risk_usd", "sum"),
            avg_notional=("notional_usd", "mean"),
            avg_effective_risk=("effective_risk_usd", "mean"),
            avg_margin_required=("margin_required_usd", "mean"),
        )
        .reset_index()
    )
    variant_summary["win_rate_pct"] = 100.0 * variant_summary["wins"] / variant_summary["trades"].clip(lower=1)
    variant_summary["total_realized_r"] = variant_summary["total_net_pnl"] / variant_summary["total_initial_risk"].replace(0.0, np.nan)
    variant_summary["portfolio_r_per_trade"] = variant_summary["total_realized_r"] / variant_summary["trades"].clip(lower=1)

    monthly = (
        all_trades.groupby(["symbol", "variant", "exit_month"], dropna=False)
        .agg(trades=("net_pnl_quote", "size"), total_net_pnl=("net_pnl_quote", "sum"), avg_realized_r=("realized_r", "mean"))
        .reset_index()
    )
    daily = (
        all_trades.groupby(["symbol", "variant", "exit_date"], dropna=False)
        .agg(trades=("net_pnl_quote", "size"), total_net_pnl=("net_pnl_quote", "sum"), avg_realized_r=("realized_r", "mean"))
        .reset_index()
    )
    return variant_summary, monthly, daily


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto trend-pullback + breakout-retest strategy")
    parser.add_argument("--symbols", type=str, default="BTC,ETH,SOL", help="Comma-separated symbols, e.g. BTC,ETH,SOL")
    parser.add_argument("--days-back", type=int, default=365)
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--risk-usd", type=float, default=100.0)
    parser.add_argument("--position-usd", type=float, default=15000.0)
    parser.add_argument("--leverage", type=float, default=3.0)
    parser.add_argument("--exchange-id", type=str, default="binance")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    all_trades_parts: List[pd.DataFrame] = []
    debug_rows: List[dict] = []

    for symbol in symbols:
        if symbol not in SYMBOL_SPECS:
            print(f"Skipping unsupported symbol: {symbol}")
            continue
        base_cfg = DEFAULT_INSTRUMENTS.get(symbol, DEFAULT_INSTRUMENTS["BTC"])
        cfg = InstrumentConfig(
            symbol=symbol,
            exchange_id=args.exchange_id,
            timeframe=args.timeframe,
            days_back=args.days_back,
            tail_rows=base_cfg.tail_rows,
            fee_rate=base_cfg.fee_rate,
            risk_usd=args.risk_usd,
            position_usd=args.position_usd,
            leverage=args.leverage,
        )
        trades_df, debug = run_symbol(cfg, SYMBOL_SPECS[symbol])
        if not trades_df.empty:
            all_trades_parts.append(trades_df)
        debug_rows.append({"symbol": symbol, "variant": "trend_pullback_breakout", **debug})

    all_trades = pd.concat(all_trades_parts, ignore_index=True) if all_trades_parts else pd.DataFrame()
    debug_df = pd.DataFrame(debug_rows)
    variant_summary, monthly, daily = summarize_results(all_trades)

    all_trades.to_csv(OUT_TRADE_CSV, index=False)
    debug_df.to_csv(OUT_DEBUG_CSV, index=False)
    variant_summary.to_csv(OUT_VARIANT_SUMMARY_CSV, index=False)
    monthly.to_csv(OUT_MONTHLY_CSV, index=False)
    daily.to_csv(OUT_DAILY_CSV, index=False)

    total_trades = int(len(all_trades))
    total_net = float(all_trades["net_pnl_quote"].sum()) if not all_trades.empty else 0.0
    avg_notional = float(all_trades["notional_usd"].mean()) if not all_trades.empty else 0.0
    avg_effective_risk = float(all_trades["effective_risk_usd"].mean()) if not all_trades.empty else 0.0
    avg_margin = float(all_trades["margin_required_usd"].mean()) if not all_trades.empty else 0.0
    avg_trade_realized_r = float(all_trades["realized_r"].mean()) if not all_trades.empty else 0.0
    total_net_r = float(all_trades["net_pnl_quote"].sum() / max(all_trades["effective_risk_usd"].sum(), 1e-9)) if not all_trades.empty else 0.0
    portfolio_r_per_trade = float(total_net_r / max(len(all_trades), 1)) if not all_trades.empty else 0.0

    print("\nRun notes")
    print("---------")
    print(f"Total trades: {total_trades}")
    print(f"Total net PnL: {total_net:.2f}")
    print(f"Avg notional per trade: {avg_notional:.2f}")
    print(f"Avg effective risk per trade: {avg_effective_risk:.2f}")
    print(f"Avg margin required per trade: {avg_margin:.2f}")
    risk_weighted_trade_r = total_net_r
    print(f"Avg trade realized R (simple mean): {avg_trade_realized_r:.6f}")
    print(f"Avg trade realized R (risk-weighted): {risk_weighted_trade_r:.6f}")
    print(f"Portfolio R per trade: {portfolio_r_per_trade:.6f}")
    print(f"Total net R: {total_net_r:.6f}")
    if not debug_df.empty:
        numeric_cols = [c for c in debug_df.columns if c not in {"symbol", "variant"}]
        totals = debug_df[numeric_cols].sum().sort_values(ascending=False)
        print("Top friction counts:")
        for k, v in totals.head(10).items():
            print(f"  {k}: {int(v)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
