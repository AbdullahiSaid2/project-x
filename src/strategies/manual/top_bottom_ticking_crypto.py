from __future__ import annotations

"""
Crypto liquidity-sweep + MSS/CISD strategy runner v3.

Changes from v2
---------------
- Confirmation quality is improved modestly instead of broadly loosening entries:
  the confirmation candle must break structure by a small ATR margin and show better close strength.
- ATR stop buffers are widened a little to reduce marginal stop-outs.
- Monetization is earlier:
  TP1 is slightly closer, partial1 is larger, and TP2 is also a bit closer.
- Breakeven and runner protection remain in place.
- Output file names are v3-specific so prior runs are preserved.

Example
-------
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_crypto_liquidity_mss_v3 \
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
    stop_atr_mult: float
    retrace_buffer_atr_mult: float
    entry_expiry_bars: int
    max_hold_bars: int
    h1_fast_ema: int
    h1_slow_ema: int
    h1_lookback: int
    min_sweep_atr_mult: float
    min_confirm_body_atr_mult: float
    min_confirm_close_frac: float
    confirm_break_margin_atr_mult: float
    recent_mss_bars: int
    asia_start_hour_utc: int
    asia_end_hour_utc: int
    be_after_r: float
    tp1_r: float
    tp2_r: float
    partial1_frac: float
    cooldown_bars: int
    allow_countertrend_override_near_external_atr: float
    max_entry_distance_from_level_atr: float
    post_partial_lock_r: float
    trail_activate_r: float
    trail_stop_r: float


DEFAULT_INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "BTC": InstrumentConfig("BTC", "binance", "5m", 365, 200_000, 0.0006, 100.0, 15_000.0, 3.0),
    "ETH": InstrumentConfig("ETH", "binance", "5m", 365, 200_000, 0.0006, 100.0, 15_000.0, 3.0),
    "SOL": InstrumentConfig("SOL", "binance", "5m", 365, 200_000, 100.0 if False else 0.0006, 100.0, 15_000.0, 3.0),
}

# fix accidental positional confusion by rebuilding SOL entry explicitly
DEFAULT_INSTRUMENTS["SOL"] = InstrumentConfig("SOL", "binance", "5m", 365, 200_000, 0.0006, 100.0, 15_000.0, 3.0)


SYMBOL_SPECS: Dict[str, SymbolSpec] = {
    "BTC": SymbolSpec(
        tick_size=0.10,
        stop_atr_mult=0.52,
        retrace_buffer_atr_mult=0.03,
        entry_expiry_bars=9,
        max_hold_bars=60,
        h1_fast_ema=20,
        h1_slow_ema=50,
        h1_lookback=3,
        min_sweep_atr_mult=0.25,
        min_confirm_body_atr_mult=0.18,
        min_confirm_close_frac=0.62,
        confirm_break_margin_atr_mult=0.04,
        recent_mss_bars=6,
        asia_start_hour_utc=0,
        asia_end_hour_utc=8,
        be_after_r=1.0,
        tp1_r=0.68,
        tp2_r=1.18,
        partial1_frac=0.78,
        cooldown_bars=10,
        allow_countertrend_override_near_external_atr=0.15,
        max_entry_distance_from_level_atr=1.10,
        post_partial_lock_r=0.03,
        trail_activate_r=1.15,
        trail_stop_r=0.45,
    ),
    "ETH": SymbolSpec(
        tick_size=0.01,
        stop_atr_mult=0.48,
        retrace_buffer_atr_mult=0.03,
        entry_expiry_bars=9,
        max_hold_bars=60,
        h1_fast_ema=20,
        h1_slow_ema=50,
        h1_lookback=3,
        min_sweep_atr_mult=0.22,
        min_confirm_body_atr_mult=0.16,
        min_confirm_close_frac=0.60,
        confirm_break_margin_atr_mult=0.035,
        recent_mss_bars=6,
        asia_start_hour_utc=0,
        asia_end_hour_utc=8,
        be_after_r=1.0,
        tp1_r=0.68,
        tp2_r=1.18,
        partial1_frac=0.78,
        cooldown_bars=10,
        allow_countertrend_override_near_external_atr=0.18,
        max_entry_distance_from_level_atr=1.15,
        post_partial_lock_r=0.03,
        trail_activate_r=1.10,
        trail_stop_r=0.42,
    ),
    "SOL": SymbolSpec(
        tick_size=0.001,
        stop_atr_mult=0.45,
        retrace_buffer_atr_mult=0.04,
        entry_expiry_bars=9,
        max_hold_bars=54,
        h1_fast_ema=20,
        h1_slow_ema=50,
        h1_lookback=3,
        min_sweep_atr_mult=0.20,
        min_confirm_body_atr_mult=0.14,
        min_confirm_close_frac=0.58,
        confirm_break_margin_atr_mult=0.03,
        recent_mss_bars=6,
        asia_start_hour_utc=0,
        asia_end_hour_utc=8,
        be_after_r=1.0,
        tp1_r=0.62,
        tp2_r=1.05,
        partial1_frac=0.80,
        cooldown_bars=8,
        allow_countertrend_override_near_external_atr=0.20,
        max_entry_distance_from_level_atr=1.20,
        post_partial_lock_r=0.04,
        trail_activate_r=1.00,
        trail_stop_r=0.40,
    ),
}


OUT_TRADE_CSV = THIS_DIR / "crypto_liquidity_mss_v3_trade_log.csv"
OUT_DEBUG_CSV = THIS_DIR / "crypto_liquidity_mss_v3_debug_counts.csv"
OUT_VARIANT_SUMMARY_CSV = THIS_DIR / "crypto_liquidity_mss_v3_variant_summary.csv"
OUT_MONTHLY_CSV = THIS_DIR / "crypto_liquidity_mss_v3_monthly_summary.csv"
OUT_DAILY_CSV = THIS_DIR / "crypto_liquidity_mss_v3_daily_summary.csv"


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


def build_feature_frame(df: pd.DataFrame, spec: SymbolSpec) -> pd.DataFrame:
    m = df.copy().sort_index()
    m.columns = [c.capitalize() for c in m.columns]
    if m.index.tz is None:
        m.index = pd.DatetimeIndex(m.index).tz_localize("UTC")
    else:
        m.index = pd.DatetimeIndex(m.index).tz_convert("UTC")

    m["date_utc"] = pd.DatetimeIndex(m.index).date
    m["hour_utc"] = pd.DatetimeIndex(m.index).hour
    m["atr14"] = _atr(m, 14)

    day_summary = (
        m.groupby("date_utc")
        .agg(day_high=("High", "max"), day_low=("Low", "min"))
        .shift(1)
        .rename(columns={"day_high": "prior_day_high", "day_low": "prior_day_low"})
    )
    m = m.join(day_summary, on="date_utc")

    asia_rows = m[(m["hour_utc"] >= spec.asia_start_hour_utc) & (m["hour_utc"] < spec.asia_end_hour_utc)]
    asia_summary = asia_rows.groupby("date_utc").agg(asia_high=("High", "max"), asia_low=("Low", "min"))
    m = m.join(asia_summary, on="date_utc")

    m["external_buyside"] = m[["prior_day_high", "asia_high"]].max(axis=1, skipna=True)
    m["external_sellside"] = m[["prior_day_low", "asia_low"]].min(axis=1, skipna=True)

    h1 = (
        m[["Open", "High", "Low", "Close", "Volume"]]
        .resample("1H", label="right", closed="right")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )
    h1["h1_ema_fast"] = _ema(h1["Close"], spec.h1_fast_ema)
    h1["h1_ema_slow"] = _ema(h1["Close"], spec.h1_slow_ema)
    h1["h1_prev_high"] = h1["High"].rolling(spec.h1_lookback).max().shift(1)
    h1["h1_prev_low"] = h1["Low"].rolling(spec.h1_lookback).min().shift(1)
    h1["bias_bull"] = (h1["Close"] > h1["h1_ema_fast"]) & (h1["h1_ema_fast"] > h1["h1_ema_slow"])
    h1["bias_bear"] = (h1["Close"] < h1["h1_ema_fast"]) & (h1["h1_ema_fast"] < h1["h1_ema_slow"])
    h1["h1_mss_bull"] = h1["Close"] > h1["h1_prev_high"]
    h1["h1_mss_bear"] = h1["Close"] < h1["h1_prev_low"]
    h1_ctx = h1[["bias_bull", "bias_bear", "h1_mss_bull", "h1_mss_bear"]]

    m = pd.merge_asof(m.sort_index(), h1_ctx.sort_index(), left_index=True, right_index=True, direction="backward")

    m["recent_high"] = m["High"].rolling(spec.recent_mss_bars).max().shift(1)
    m["recent_low"] = m["Low"].rolling(spec.recent_mss_bars).min().shift(1)
    m["body"] = (m["Close"] - m["Open"]).abs()
    m["range"] = (m["High"] - m["Low"]).replace(0.0, np.nan)
    m["bull_close_frac"] = (m["Close"] - m["Low"]) / m["range"]
    m["bear_close_frac"] = (m["High"] - m["Close"]) / m["range"]

    m["dist_above_buyside"] = m["High"] - m["external_buyside"]
    m["dist_below_sellside"] = m["external_sellside"] - m["Low"]

    m["sweep_short"] = (
        pd.notna(m["external_buyside"])
        & (m["High"] > m["external_buyside"])
        & (m["Close"] < m["external_buyside"])
        & (m["dist_above_buyside"] >= m["atr14"] * spec.min_sweep_atr_mult)
    )
    m["sweep_long"] = (
        pd.notna(m["external_sellside"])
        & (m["Low"] < m["external_sellside"])
        & (m["Close"] > m["external_sellside"])
        & (m["dist_below_sellside"] >= m["atr14"] * spec.min_sweep_atr_mult)
    )

    m["confirm_short"] = (
        (m["Close"] < (m["recent_low"] - m["atr14"] * spec.confirm_break_margin_atr_mult))
        & (m["body"] >= m["atr14"] * spec.min_confirm_body_atr_mult)
        & (m["bear_close_frac"] >= spec.min_confirm_close_frac)
    )
    m["confirm_long"] = (
        (m["Close"] > (m["recent_high"] + m["atr14"] * spec.confirm_break_margin_atr_mult))
        & (m["body"] >= m["atr14"] * spec.min_confirm_body_atr_mult)
        & (m["bull_close_frac"] >= spec.min_confirm_close_frac)
    )
    return m


@dataclass
class PendingSetup:
    side: str
    created_i: int
    expires_i: int
    external_level: float
    entry: float
    stop: float
    risk_per_unit: float
    tp1: float
    tp2: float
    atr_at_signal: float
    reason: str


@dataclass
class OpenTrade:
    side: str
    entry_i: int
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    tp1: float
    tp2: float
    risk_per_unit: float
    qty: float
    qty_open: float
    qty_partial1: float
    qty_runner: float
    fee_rate: float
    symbol: str
    partial1_taken: bool = False
    be_moved: bool = False
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    gross_pnl_quote: float = 0.0
    fees_quote: float = 0.0
    hold_bars: int = 0
    max_favorable_r: float = 0.0


def _safe_float(x: float) -> float:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return float("nan")
    return float(x)


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


def _entry_touched(row: pd.Series, entry: float) -> bool:
    return row["Low"] <= entry <= row["High"]


def _close_trade(trade: OpenTrade, when: pd.Timestamp, price: float, reason: str) -> dict:
    trade.exit_time = when
    trade.exit_price = float(price)
    trade.exit_reason = reason
    net = trade.gross_pnl_quote - trade.fees_quote
    denom = max((trade.qty * trade.risk_per_unit), 1e-9)
    return {
        "symbol": trade.symbol,
        "side": trade.side,
        "entry_time": trade.entry_time,
        "exit_time": trade.exit_time,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "stop_price": trade.stop_price,
        "tp1": trade.tp1,
        "tp2": trade.tp2,
        "qty": trade.qty,
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
    # small price buffer to offset at least part of round-trip fees after moving to breakeven
    return max(entry * fee_rate * 2.0, 0.0)


def run_symbol(cfg: InstrumentConfig, spec: SymbolSpec) -> Tuple[pd.DataFrame, Dict[str, float]]:
    print(f"\n=== {cfg.symbol} | liquidity_mss_v3 | {cfg.exchange_id} | {cfg.timeframe} ===")
    df = get_crypto_ohlcv(cfg.symbol, timeframe=cfg.timeframe, days_back=cfg.days_back, exchange_id=cfg.exchange_id)
    df = df.tail(cfg.tail_rows)
    print(f"Loaded {len(df):,} rows | start={df.index.min()} end={df.index.max()}")
    m = build_feature_frame(df, spec)

    debug: Dict[str, float] = {
        "blocked_h1_bias": 0,
        "sweep_long_seen": 0,
        "sweep_short_seen": 0,
        "confirm_long_seen": 0,
        "confirm_short_seen": 0,
        "reject_weak_confirmation": 0,
        "pending_long_armed": 0,
        "pending_short_armed": 0,
        "pending_expired": 0,
        "pending_not_touched": 0,
        "pending_invalidated_far": 0,
        "entry_long": 0,
        "entry_short": 0,
        "partial1": 0,
        "be_move": 0,
        "trail_lock": 0,
        "exit_stop": 0,
        "exit_tp2": 0,
        "exit_timeout": 0,
        "reject_wide_entry": 0,
        "cooldown_blocked": 0,
    }

    trades: List[dict] = []
    pending: Optional[PendingSetup] = None
    open_trade: Optional[OpenTrade] = None
    last_exit_i = -10_000
    idx = list(m.index)

    for i in range(max(80, spec.h1_slow_ema + 5), len(m)):
        row = m.iloc[i]
        ts = idx[i]

        if open_trade is not None:
            open_trade.hold_bars += 1
            _update_max_favorable_r(open_trade, row)

            # tighten runner stop once trade extends enough after partial1
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
                        be_offset = max(open_trade.risk_per_unit * spec.post_partial_lock_r, _fee_offset_per_unit(open_trade.entry_price, open_trade.fee_rate))
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
                        be_offset = max(open_trade.risk_per_unit * spec.post_partial_lock_r, _fee_offset_per_unit(open_trade.entry_price, open_trade.fee_rate))
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
                pnl = qty * (exit_px - open_trade.entry_price) if open_trade.side == "LONG" else qty * (open_trade.entry_price - exit_px)
                open_trade.gross_pnl_quote += pnl
                open_trade.fees_quote += qty * (open_trade.entry_price + exit_px) * open_trade.fee_rate
                open_trade.qty_open = 0.0
                debug["exit_timeout"] += 1
                trades.append(_close_trade(open_trade, ts, exit_px, "timeout"))
                open_trade = None
                last_exit_i = i
                pending = None
                continue

        if pending is not None and open_trade is None:
            if i > pending.expires_i:
                debug["pending_expired"] += 1
                debug["pending_not_touched"] += 1
                pending = None
            else:
                far_invalid = abs(float(row["Close"]) - pending.external_level) > pending.atr_at_signal * spec.max_entry_distance_from_level_atr
                if far_invalid:
                    debug["pending_invalidated_far"] += 1
                    pending = None
                elif _entry_touched(row, pending.entry):
                    qty, _notional, eff_risk = _risk_size(pending.entry, pending.stop, cfg)
                    if qty <= 0 or eff_risk <= 0:
                        pending = None
                    else:
                        q1 = qty * spec.partial1_frac
                        qr = qty - q1
                        open_trade = OpenTrade(
                            side=pending.side,
                            entry_i=i,
                            entry_time=ts,
                            entry_price=float(pending.entry),
                            stop_price=float(pending.stop),
                            tp1=float(pending.tp1),
                            tp2=float(pending.tp2),
                            risk_per_unit=float(pending.risk_per_unit),
                            qty=float(qty),
                            qty_open=float(qty),
                            qty_partial1=float(q1),
                            qty_runner=float(qr),
                            fee_rate=cfg.fee_rate,
                            symbol=cfg.symbol,
                        )
                        if pending.side == "LONG":
                            debug["entry_long"] += 1
                        else:
                            debug["entry_short"] += 1
                        pending = None
                        continue

        if open_trade is not None or pending is not None:
            continue

        if i - last_exit_i < spec.cooldown_bars:
            if bool(row.get("sweep_long", False)) or bool(row.get("sweep_short", False)):
                debug["cooldown_blocked"] += 1
            continue

        atr = _safe_float(row["atr14"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        if bool(row.get("sweep_long", False)):
            debug["sweep_long_seen"] += 1
            external = _safe_float(row["external_sellside"])
            h1_ok = bool(row.get("bias_bull", False) or row.get("h1_mss_bull", False))
            near_external_override = abs(float(row["Close"]) - external) <= atr * spec.allow_countertrend_override_near_external_atr
            if not h1_ok and not near_external_override:
                debug["blocked_h1_bias"] += 1
            else:
                confirm_i = None
                for j in range(i + 1, min(i + 4, len(m))):
                    c = m.iloc[j]
                    if bool(c.get("confirm_long", False)):
                        confirm_i = j
                        debug["confirm_long_seen"] += 1
                        break
                if confirm_i is not None:
                    c = m.iloc[confirm_i]
                    entry = float(c["Low"] + 0.5 * (c["High"] - c["Low"])) - atr * spec.retrace_buffer_atr_mult
                    stop = float(min(row["Low"], c["Low"])) - atr * spec.stop_atr_mult
                    if entry - external > atr * spec.max_entry_distance_from_level_atr:
                        debug["reject_wide_entry"] += 1
                    else:
                        risk = entry - stop
                        if risk > spec.tick_size and stop < entry:
                            pending = PendingSetup(
                                side="LONG",
                                created_i=confirm_i,
                                expires_i=min(confirm_i + spec.entry_expiry_bars, len(m) - 1),
                                external_level=external,
                                entry=entry,
                                stop=stop,
                                risk_per_unit=risk,
                                tp1=entry + risk * spec.tp1_r,
                                tp2=entry + risk * spec.tp2_r,
                                atr_at_signal=atr,
                                reason="sweep_long_confirm_long",
                            )
                            debug["pending_long_armed"] += 1
                else:
                    debug["reject_weak_confirmation"] += 1

        if pending is None and bool(row.get("sweep_short", False)):
            debug["sweep_short_seen"] += 1
            external = _safe_float(row["external_buyside"])
            h1_ok = bool(row.get("bias_bear", False) or row.get("h1_mss_bear", False))
            near_external_override = abs(float(row["Close"]) - external) <= atr * spec.allow_countertrend_override_near_external_atr
            if not h1_ok and not near_external_override:
                debug["blocked_h1_bias"] += 1
            else:
                confirm_i = None
                for j in range(i + 1, min(i + 4, len(m))):
                    c = m.iloc[j]
                    if bool(c.get("confirm_short", False)):
                        confirm_i = j
                        debug["confirm_short_seen"] += 1
                        break
                if confirm_i is not None:
                    c = m.iloc[confirm_i]
                    entry = float(c["High"] - 0.5 * (c["High"] - c["Low"])) + atr * spec.retrace_buffer_atr_mult
                    stop = float(max(row["High"], c["High"])) + atr * spec.stop_atr_mult
                    if external - entry > atr * spec.max_entry_distance_from_level_atr:
                        debug["reject_wide_entry"] += 1
                    else:
                        risk = stop - entry
                        if risk > spec.tick_size and stop > entry:
                            pending = PendingSetup(
                                side="SHORT",
                                created_i=confirm_i,
                                expires_i=min(confirm_i + spec.entry_expiry_bars, len(m) - 1),
                                external_level=external,
                                entry=entry,
                                stop=stop,
                                risk_per_unit=risk,
                                tp1=entry - risk * spec.tp1_r,
                                tp2=entry - risk * spec.tp2_r,
                                atr_at_signal=atr,
                                reason="sweep_short_confirm_short",
                            )
                            debug["pending_short_armed"] += 1
                else:
                    debug["reject_weak_confirmation"] += 1

    if open_trade is not None:
        last_ts = idx[-1]
        last_close = float(m.iloc[-1]["Close"])
        qty = open_trade.qty_open
        pnl = qty * (last_close - open_trade.entry_price) if open_trade.side == "LONG" else qty * (open_trade.entry_price - last_close)
        open_trade.gross_pnl_quote += pnl
        open_trade.fees_quote += qty * (open_trade.entry_price + last_close) * open_trade.fee_rate
        open_trade.qty_open = 0.0
        trades.append(_close_trade(open_trade, last_ts, last_close, "eod"))

    trade_df = pd.DataFrame(trades)
    if not trade_df.empty:
        trade_df["entry_time"] = pd.to_datetime(trade_df["entry_time"], utc=True)
        trade_df["exit_time"] = pd.to_datetime(trade_df["exit_time"], utc=True)
        trade_df["exit_date"] = trade_df["exit_time"].dt.date
        trade_df["exit_month"] = trade_df["exit_time"].dt.to_period("M").astype(str)
        trade_df["notional_at_entry"] = trade_df["qty"] * trade_df["entry_price"]
        trade_df["effective_risk_quote"] = trade_df["qty"] * (trade_df["entry_price"] - trade_df["stop_price"]).abs()
        trade_df["return_pct_on_margin_budget"] = trade_df["net_pnl_quote"] / cfg.position_usd * 100.0
        trade_df["margin_required_est"] = trade_df["notional_at_entry"] / cfg.leverage
        trade_df["variant"] = "liquidity_mss_h1_bias_v2"
        trade_df["timeframe"] = cfg.timeframe
        trade_df["exchange_id"] = cfg.exchange_id
    return trade_df, debug


def build_variant_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=[
            "symbol", "variant", "trades", "win_rate", "gross_pnl_quote", "fees_quote",
            "net_pnl_quote", "avg_net_pnl", "avg_realized_r", "profit_factor",
            "avg_notional", "avg_effective_risk", "avg_margin_required"
        ])

    g = trades.groupby(["symbol", "variant"], dropna=False)
    rows = []
    for (symbol, variant), x in g:
        wins = x.loc[x["net_pnl_quote"] > 0, "net_pnl_quote"].sum()
        losses = x.loc[x["net_pnl_quote"] < 0, "net_pnl_quote"].sum()
        profit_factor = wins / abs(losses) if losses < 0 else np.nan
        rows.append({
            "symbol": symbol,
            "variant": variant,
            "trades": int(len(x)),
            "win_rate": float((x["net_pnl_quote"] > 0).mean() * 100.0),
            "gross_pnl_quote": float(x["gross_pnl_quote"].sum()),
            "fees_quote": float(x["fees_quote"].sum()),
            "net_pnl_quote": float(x["net_pnl_quote"].sum()),
            "avg_net_pnl": float(x["net_pnl_quote"].mean()),
            "avg_realized_r": float(x["realized_r"].mean()),
            "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else np.nan,
            "avg_notional": float(x["notional_at_entry"].mean()),
            "avg_effective_risk": float(x["effective_risk_quote"].mean()),
            "avg_margin_required": float(x["margin_required_est"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["net_pnl_quote", "symbol"], ascending=[False, True]).reset_index(drop=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crypto liquidity sweep + MSS strategy backtest v3")
    p.add_argument("--symbols", default="BTC,ETH,SOL", help="Comma-separated symbols, e.g. BTC,ETH,SOL")
    p.add_argument("--days-back", type=int, default=365)
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--risk-usd", type=float, default=100.0)
    p.add_argument("--position-usd", type=float, default=15000.0)
    p.add_argument("--leverage", type=float, default=3.0)
    p.add_argument("--exchange-id", default="binance")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]

    all_trades: List[pd.DataFrame] = []
    debug_rows: List[dict] = []

    for symbol in symbols:
        if symbol not in SYMBOL_SPECS:
            print(f"Skipping unsupported symbol: {symbol}")
            continue
        base = DEFAULT_INSTRUMENTS[symbol]
        cfg = InstrumentConfig(
            symbol=symbol,
            exchange_id=args.exchange_id,
            timeframe=args.timeframe,
            days_back=args.days_back,
            tail_rows=base.tail_rows,
            fee_rate=base.fee_rate,
            risk_usd=args.risk_usd,
            position_usd=args.position_usd,
            leverage=args.leverage,
        )
        trades, debug = run_symbol(cfg, SYMBOL_SPECS[symbol])
        if not trades.empty:
            all_trades.append(trades)
        debug_rows.append({"symbol": symbol, **debug})

    trade_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    debug_df = pd.DataFrame(debug_rows)
    variant_summary = build_variant_summary(trade_df)

    if not trade_df.empty:
        monthly = (
            trade_df.groupby(["symbol", "variant", "exit_month"], dropna=False)
            .agg(
                trades=("net_pnl_quote", "size"),
                net_pnl_quote=("net_pnl_quote", "sum"),
                gross_pnl_quote=("gross_pnl_quote", "sum"),
                fees_quote=("fees_quote", "sum"),
                avg_realized_r=("realized_r", "mean"),
            )
            .reset_index()
        )
        daily = (
            trade_df.groupby(["symbol", "variant", "exit_date"], dropna=False)
            .agg(
                trades=("net_pnl_quote", "size"),
                net_pnl_quote=("net_pnl_quote", "sum"),
                gross_pnl_quote=("gross_pnl_quote", "sum"),
                fees_quote=("fees_quote", "sum"),
                avg_realized_r=("realized_r", "mean"),
            )
            .reset_index()
        )
    else:
        monthly = pd.DataFrame()
        daily = pd.DataFrame()

    trade_df.to_csv(OUT_TRADE_CSV, index=False)
    debug_df.to_csv(OUT_DEBUG_CSV, index=False)
    variant_summary.to_csv(OUT_VARIANT_SUMMARY_CSV, index=False)
    monthly.to_csv(OUT_MONTHLY_CSV, index=False)
    daily.to_csv(OUT_DAILY_CSV, index=False)

    total_trades = int(len(trade_df))
    total_net = float(trade_df["net_pnl_quote"].sum()) if not trade_df.empty else 0.0
    avg_notional = float(trade_df["notional_at_entry"].mean()) if not trade_df.empty else 0.0
    avg_risk = float(trade_df["effective_risk_quote"].mean()) if not trade_df.empty else 0.0
    avg_margin = float(trade_df["margin_required_est"].mean()) if not trade_df.empty else 0.0
    avg_r = float(trade_df["realized_r"].mean()) if not trade_df.empty else 0.0

    print("\nRun notes")
    print("---------")
    print(f"Total trades: {total_trades}")
    print(f"Total net PnL: {total_net:.2f}")
    print(f"Avg notional per trade: {avg_notional:.2f}")
    print(f"Avg effective risk per trade: {avg_risk:.2f}")
    print(f"Avg margin required per trade: {avg_margin:.2f}")
    print(f"Avg net R per trade: {avg_r:.4f}")
    if not debug_df.empty:
        top_friction = debug_df.drop(columns=["symbol"], errors="ignore").sum(axis=0).sort_values(ascending=False).head(10)
        print("Top friction counts:")
        for k, v in top_friction.items():
            print(f"  {k}: {int(v)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
