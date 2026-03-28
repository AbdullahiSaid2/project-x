# ============================================================
# 🌙 ICT Multi-Timeframe Backtester
#
# A custom backtesting engine built specifically for the ICT
# A+ strategy. Standard backtesting.py can't do this because
# it only supports a single timeframe. This engine runs all
# three timeframes simultaneously.
#
# WHAT IT TESTS:
#   Same exact logic as ict_executor.py:
#   1. D1  — Direction (PDH/PDL bias + structure)
#   2. D1  — Premium / Discount zone
#   3. H1  — FVG / OB + Displacement
#   4. TIME — Kill Zone filter (London / NY AM)
#   5. M5  — CISD quality (Strong / Moderate)
#
# HOW TO RUN:
#   python src/agents/ict_backtester.py
#   python src/agents/ict_backtester.py --symbol ETH
#   python src/agents/ict_backtester.py --symbol BTC --days 120
# ============================================================

import sys
import csv
import json
import pytz
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config      import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION
from src.data.fetcher import get_ohlcv

# ── paths ─────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "src" / "data" / "ict_backtest"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EST = pytz.timezone("America/New_York")

# ── backtest config ───────────────────────────────────────────
INITIAL_CASH          = BACKTEST_INITIAL_CASH
COMMISSION_PCT        = BACKTEST_COMMISSION
RR_MINIMUM            = 2.0
CISD_STRENGTH_REQUIRED = "STRONG"   # "STRONG" | "MODERATE"
REQUIRE_KILL_ZONE     = True
MAX_TRADES_PER_SESSION = 2

# Kill Zone hours (EST)
KILL_ZONES = {
    "London Open": (2,  5),
    "New York AM": (7, 10),
}


# ── Trade record ──────────────────────────────────────────────
@dataclass
class BacktestTrade:
    symbol:        str
    entry_time:    str
    exit_time:     str = ""
    direction:     str = ""        # LONG | SHORT
    entry_price:   float = 0.0
    exit_price:    float = 0.0
    stop_loss:     float = 0.0
    take_profit:   float = 0.0
    risk_reward:   float = 0.0
    pnl_usd:       float = 0.0
    pnl_pct:       float = 0.0
    outcome:       str = ""        # WIN | LOSS | OPEN
    exit_reason:   str = ""        # TP_HIT | SL_HIT | END_OF_DATA
    kill_zone:     str = ""
    cisd_strength: str = ""
    d1_bias:       str = ""
    price_zone:    str = ""
    fvg_found:     bool = False
    ob_found:      bool = False


@dataclass
class BacktestResult:
    symbol:         str
    days_tested:    int
    total_trades:   int = 0
    wins:           int = 0
    losses:         int = 0
    win_rate:       float = 0.0
    total_return:   float = 0.0
    total_return_pct: float = 0.0
    avg_win:        float = 0.0
    avg_loss:       float = 0.0
    profit_factor:  float = 0.0
    max_drawdown:   float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe:         float = 0.0
    avg_rr:         float = 0.0
    trades:         list = field(default_factory=list)
    equity_curve:   list = field(default_factory=list)


# ════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ════════════════════════════════════════════════════════════════

def load_data(symbol: str, days: int = 90) -> dict:
    """Load D1, H1, and M5 data for backtesting.
    Uses Databento for M5 (precise) with 15m fallback."""
    print(f"  📡 Loading data for {symbol}...")
    sym = symbol.replace("-USD", "")

    data = {}

    # D1 and H1 — use standard fetcher (Databento auto-routes for futures)
    for tf, label in [("1D", "d1"), ("1H", "h1")]:
        try:
            df = get_ohlcv(sym, exchange=EXCHANGE, timeframe=tf, days_back=days)
            df.index = pd.to_datetime(df.index)
            data[label] = df
            print(f"     {tf}: {len(df)} candles")
        except Exception as e:
            print(f"     ❌ {tf} failed: {e}")
            data[label] = pd.DataFrame()

    # M5 — try Databento first, fall back to 15m
    m5_loaded = False
    try:
        from src.data.databento_fetcher import get_databento_ohlcv, FUTURES_DB_MAP
        import os
        if os.getenv("DATABENTO_API_KEY") and sym.upper() in FUTURES_DB_MAP:
            df = get_databento_ohlcv(sym, "5m", days_back=min(days, 90))
            df.index = pd.to_datetime(df.index)
            data["m15"] = df   # stored as m15 key for compatibility
            print(f"     5m (Databento): {len(df)} candles ← precise M5")
            m5_loaded = True
    except Exception:
        pass

    if not m5_loaded:
        try:
            df = get_ohlcv(sym, exchange=EXCHANGE, timeframe="15m", days_back=days)
            df.index = pd.to_datetime(df.index)
            data["m15"] = df
            print(f"     15m (proxy): {len(df)} candles")
        except Exception as e:
            print(f"     ❌ 15m failed: {e}")
            data["m15"] = pd.DataFrame()

    return data


def get_d1_candle_at(d1_df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    """Get the D1 candle for the previous day relative to timestamp."""
    try:
        day  = ts.normalize()
        prev = d1_df[d1_df.index < day]
        if prev.empty:
            return None
        return prev.iloc[-1]
    except Exception:
        return None


def get_h1_slice(h1_df: pd.DataFrame, ts: pd.Timestamp, lookback_hours: int = 30) -> pd.DataFrame:
    """Get the last N H1 candles before a timestamp."""
    return h1_df[h1_df.index < ts].tail(lookback_hours)


def get_m15_slice(m15_df: pd.DataFrame, ts: pd.Timestamp, lookback: int = 20) -> pd.DataFrame:
    """Get the last N 15m candles before a timestamp."""
    return m15_df[m15_df.index < ts].tail(lookback)


# ════════════════════════════════════════════════════════════════
# ICT CHECKS (same logic as ict_scanner / ict_executor)
# ════════════════════════════════════════════════════════════════

def check_d1_bias(d1_candle: pd.Series, current_price: float,
                  d1_history: pd.DataFrame) -> dict:
    """D1 directional bias from previous day high/low."""
    pdh = float(d1_candle["High"])
    pdl = float(d1_candle["Low"])
    pdc = float(d1_candle["Close"])

    # 5-day structure
    recent_d1 = d1_history.tail(5)
    if len(recent_d1) >= 3:
        highs = recent_d1["High"].values
        lows  = recent_d1["Low"].values
        hh = len(highs) >= 3 and highs[-1] > highs[-2] > highs[-3]
        hl = len(lows)  >= 3 and lows[-1]  > lows[-2]  > lows[-3]
        lh = len(highs) >= 3 and highs[-1] < highs[-2] < highs[-3]
        ll = len(lows)  >= 3 and lows[-1]  < lows[-2]  < lows[-3]
    else:
        hh = hl = lh = ll = False

    if current_price > pdh:
        bias = "BULLISH"
    elif current_price < pdl:
        bias = "BEARISH"
    elif hh and hl:
        bias = "BULLISH"
    elif lh and ll:
        bias = "BEARISH"
    elif current_price > pdc:
        bias = "BULLISH"
    elif current_price < pdc:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {"bias": bias, "pdh": pdh, "pdl": pdl}


def check_premium_discount(price: float, pdh: float, pdl: float) -> str:
    """Return price zone relative to daily range."""
    rng = pdh - pdl
    if rng <= 0:
        return "EQUILIBRIUM"
    pct = (price - pdl) / rng * 100
    if pct <= 35:   return "DEEP DISCOUNT"
    if pct <= 45:   return "DISCOUNT"
    if pct <= 55:   return "EQUILIBRIUM"
    if pct <= 65:   return "PREMIUM"
    return "DEEP PREMIUM"


def find_h1_fvg(h1_slice: pd.DataFrame, bias: str) -> dict:
    """Find most recent unfilled FVG aligned with bias."""
    if len(h1_slice) < 3:
        return {}
    curr_price = float(h1_slice["Close"].iloc[-1])
    for i in range(len(h1_slice)-3, -1, -1):
        c1 = h1_slice.iloc[i]
        c3 = h1_slice.iloc[i+2]
        if bias == "BULLISH" and float(c1["High"]) < float(c3["Low"]):
            gap_high = float(c3["Low"])
            gap_low  = float(c1["High"])
            gap_mid  = (gap_high + gap_low) / 2
            if gap_low <= curr_price <= gap_high * 1.003:
                return {"found": True, "mid": gap_mid,
                        "high": gap_high, "low": gap_low, "type": "BULLISH"}
        elif bias == "BEARISH" and float(c1["Low"]) > float(c3["High"]):
            gap_low  = float(c3["High"])
            gap_high = float(c1["Low"])
            gap_mid  = (gap_high + gap_low) / 2
            if gap_low * 0.997 <= curr_price <= gap_high:
                return {"found": True, "mid": gap_mid,
                        "high": gap_high, "low": gap_low, "type": "BEARISH"}
    return {}


def find_h1_ob(h1_slice: pd.DataFrame, bias: str) -> dict:
    """Find most recent Order Block aligned with bias."""
    if len(h1_slice) < 4:
        return {}
    curr_price = float(h1_slice["Close"].iloc[-1])
    avg_range  = float((h1_slice["High"] - h1_slice["Low"]).mean())

    for i in range(len(h1_slice)-4, -1, -1):
        c     = h1_slice.iloc[i]
        next3 = h1_slice.iloc[i+1:i+4]
        if bias == "BULLISH" and float(c["Close"]) < float(c["Open"]):
            move = (next3["Close"].max() - float(c["Low"])) / float(c["Low"])
            if move > 0.004:
                ob_high = float(c["Open"])
                ob_low  = float(c["Low"])
                if ob_low <= curr_price <= ob_high * 1.002:
                    return {"found": True, "mid": (ob_high+ob_low)/2,
                            "high": ob_high, "low": ob_low, "type": "BULLISH"}
        elif bias == "BEARISH" and float(c["Close"]) > float(c["Open"]):
            move = (float(c["High"]) - next3["Close"].min()) / float(c["High"])
            if move > 0.004:
                ob_high = float(c["High"])
                ob_low  = float(c["Close"])
                if ob_low * 0.998 <= curr_price <= ob_high:
                    return {"found": True, "mid": (ob_high+ob_low)/2,
                            "high": ob_high, "low": ob_low, "type": "BEARISH"}
    return {}


def check_h1_displacement(h1_slice: pd.DataFrame, bias: str) -> str:
    """Detect displacement candle on H1. Returns STRONG | MODERATE | NONE."""
    if len(h1_slice) < 5:
        return "NONE"
    avg_range = float((h1_slice["High"] - h1_slice["Low"]).tail(20).mean())
    recent    = h1_slice.tail(5)
    for i in range(-5, 0):
        c    = recent.iloc[i]
        rng  = float(c["High"] - c["Low"])
        body = abs(float(c["Close"]) - float(c["Open"]))
        if rng == 0:
            continue
        body_pct   = body / rng
        size_ratio = rng / max(avg_range, 0.001)

        is_bull = float(c["Close"]) > float(c["Open"])
        is_bear = float(c["Close"]) < float(c["Open"])
        match   = (bias == "BULLISH" and is_bull) or (bias == "BEARISH" and is_bear)

        if match and body_pct >= 0.60 and size_ratio >= 1.3:
            if body_pct >= 0.75 and size_ratio >= 1.8:
                return "STRONG"
            return "MODERATE"
    return "NONE"


def check_m15_cisd(m15_slice: pd.DataFrame, bias: str) -> dict:
    """Detect CISD quality on M5 (or 15m proxy if Databento unavailable)."""
    if len(m15_slice) < 5:
        return {"found": False, "strength": "NONE", "entry": 0}

    avg_range = float((m15_slice["High"] - m15_slice["Low"]).mean())
    recent    = m15_slice.tail(8)

    for i in range(-8, 0):
        c    = recent.iloc[i]
        rng  = float(c["High"] - c["Low"])
        body = abs(float(c["Close"]) - float(c["Open"]))
        if rng == 0:
            continue
        body_pct   = body / rng
        size_ratio = rng / max(avg_range, 0.001)

        is_bull = float(c["Close"]) > float(c["Open"])
        is_bear = float(c["Close"]) < float(c["Open"])
        match   = (bias == "BULLISH" and is_bull) or (bias == "BEARISH" and is_bear)

        if not match:
            continue

        score = 0
        if body_pct >= 0.75:  score += 3
        elif body_pct >= 0.60: score += 2
        elif body_pct >= 0.45: score += 1
        if size_ratio >= 2.0:  score += 3
        elif size_ratio >= 1.5: score += 2
        elif size_ratio >= 1.2: score += 1

        if bias == "BULLISH":
            close_pos = (float(c["Close"]) - float(c["Low"])) / rng
        else:
            close_pos = (float(c["High"]) - float(c["Close"])) / rng
        if close_pos >= 0.80:  score += 2
        elif close_pos >= 0.65: score += 1

        if score >= 7:
            strength = "STRONG"
        elif score >= 4:
            strength = "MODERATE"
        else:
            continue   # too weak — skip

        return {
            "found":    True,
            "strength": strength,
            "entry":    round(float(c["Close"]), 4),
            "score":    score,
        }

    return {"found": False, "strength": "NONE", "entry": 0}


def in_kill_zone(ts: pd.Timestamp) -> str | None:
    """Returns Kill Zone name if timestamp is in one, else None."""
    try:
        ts_est = ts.tz_localize("UTC").astimezone(EST)
        hour   = ts_est.hour
        for name, (start, end) in KILL_ZONES.items():
            if start <= hour < end:
                return name
    except Exception:
        pass
    return None


# ════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════

def run_backtest(symbol: str, days: int = 90) -> BacktestResult:
    """
    Walk-forward ICT backtest on historical data.
    Iterates through every H1 candle, runs all 4 ICT checks,
    and simulates trade execution + management.
    """
    result = BacktestResult(symbol=symbol, days_tested=days)
    data   = load_data(symbol, days)

    h1_df  = data.get("h1",  pd.DataFrame())
    d1_df  = data.get("d1",  pd.DataFrame())
    m15_df = data.get("m15", pd.DataFrame())

    if h1_df.empty or d1_df.empty:
        print(f"  ❌ Insufficient data for {symbol}")
        return result

    cash          = float(INITIAL_CASH)
    peak_cash     = cash
    max_drawdown  = 0.0
    equity_curve  = [cash]
    trades        = []
    open_trade    = None
    session_counts: dict[str, int] = {}
    pnl_list      = []

    print(f"\n  🔁 Walking through {len(h1_df)} H1 candles...")

    for idx, (ts, h1_row) in enumerate(h1_df.iterrows()):
        curr_price = float(h1_row["Close"])

        # ── Manage open trade first ────────────────────────────
        if open_trade is not None:
            high = float(h1_row["High"])
            low  = float(h1_row["Low"])

            hit_tp = (open_trade.direction == "LONG"  and high >= open_trade.take_profit) or \
                     (open_trade.direction == "SHORT" and low  <= open_trade.take_profit)
            hit_sl = (open_trade.direction == "LONG"  and low  <= open_trade.stop_loss) or \
                     (open_trade.direction == "SHORT" and high >= open_trade.stop_loss)

            if hit_tp or hit_sl:
                exit_price    = open_trade.take_profit if hit_tp else open_trade.stop_loss
                open_trade.exit_price  = exit_price
                open_trade.exit_time   = str(ts)
                open_trade.exit_reason = "TP_HIT" if hit_tp else "SL_HIT"

                # Calculate PnL
                size_units = (cash * 0.95) / open_trade.entry_price
                if open_trade.direction == "LONG":
                    raw_pnl = (exit_price - open_trade.entry_price) * size_units
                else:
                    raw_pnl = (open_trade.entry_price - exit_price) * size_units
                commission  = cash * 0.95 * COMMISSION_PCT * 2
                net_pnl     = raw_pnl - commission

                open_trade.pnl_usd  = round(net_pnl, 2)
                open_trade.pnl_pct  = round(net_pnl / cash * 100, 3)
                open_trade.outcome  = "WIN" if net_pnl > 0 else "LOSS"

                cash += net_pnl
                cash  = max(cash, 0)

                result.total_trades += 1
                if net_pnl > 0:
                    result.wins += 1
                else:
                    result.losses += 1

                pnl_list.append(net_pnl / (cash - net_pnl) * 100 if (cash - net_pnl) > 0 else 0)
                trades.append(open_trade)
                open_trade = None

                # Drawdown tracking
                peak_cash    = max(peak_cash, cash)
                dd           = (peak_cash - cash) / peak_cash * 100
                max_drawdown = max(max_drawdown, dd)
                equity_curve.append(round(cash, 2))
            continue   # don't look for new trades while one is open

        # ── Look for new trade setup ───────────────────────────
        # Skip if we don't have enough historical data yet
        if idx < 24:
            continue

        # Step 4 — Kill Zone check first (fast filter)
        kz = in_kill_zone(ts)
        if REQUIRE_KILL_ZONE and kz is None:
            continue

        # Session trade limit
        session_key = kz or "outside"
        if session_counts.get(session_key, 0) >= MAX_TRADES_PER_SESSION:
            continue

        # Step 1 — D1 Bias
        d1_before   = d1_df[d1_df.index < ts]
        if len(d1_before) < 2:
            continue
        d1_candle   = d1_before.iloc[-1]
        d1_info     = check_d1_bias(d1_candle, curr_price, d1_before)
        bias        = d1_info["bias"]
        pdh         = d1_info["pdh"]
        pdl         = d1_info["pdl"]

        if bias == "NEUTRAL":
            continue

        # Step 2 — Premium / Discount
        zone = check_premium_discount(curr_price, pdh, pdl)
        if bias == "BULLISH" and "PREMIUM" in zone:
            continue
        if bias == "BEARISH" and "DISCOUNT" in zone:
            continue

        # Step 3 — H1 FVG / OB + Displacement
        h1_slice = h1_df[h1_df.index <= ts].tail(30)
        fvg      = find_h1_fvg(h1_slice, bias)
        ob       = find_h1_ob(h1_slice, bias)
        disp     = check_h1_displacement(h1_slice, bias)

        has_pd   = bool(fvg) or bool(ob)
        has_disp = disp in ("STRONG", "MODERATE")

        if not (has_pd and has_disp):
            continue

        # Step 5 — M15 CISD
        m15_slice = m15_df[m15_df.index <= ts].tail(20) if not m15_df.empty else pd.DataFrame()
        if m15_slice.empty:
            continue

        cisd = check_m15_cisd(m15_slice, bias)
        if not cisd.get("found"):
            continue

        strength_order = {"STRONG": 2, "MODERATE": 1}
        required = strength_order.get(CISD_STRENGTH_REQUIRED, 2)
        actual   = strength_order.get(cisd.get("strength",""), 0)
        if actual < required:
            continue

        # All checks passed — build the trade
        entry  = cisd.get("entry", curr_price)
        if entry == 0:
            entry = curr_price

        if bias == "BULLISH":
            sl = round(entry * 0.995, 4)    # 0.5% below
            tp = round(pdh, 4)              # target PDH
        else:
            sl = round(entry * 1.005, 4)    # 0.5% above
            tp = round(pdl, 4)              # target PDL

        # RR check
        if bias == "BULLISH":
            risk_pts   = entry - sl
            reward_pts = tp - entry
        else:
            risk_pts   = sl - entry
            reward_pts = entry - tp

        if risk_pts <= 0 or reward_pts <= 0:
            continue

        rr = reward_pts / risk_pts
        if rr < RR_MINIMUM:
            continue

        # Get best PD level
        pd_level = fvg.get("mid", ob.get("mid", entry)) if fvg else ob.get("mid", entry)

        # Open the trade
        direction  = "LONG" if bias == "BULLISH" else "SHORT"
        open_trade = BacktestTrade(
            symbol       = symbol,
            entry_time   = str(ts),
            direction    = direction,
            entry_price  = round(entry, 4),
            stop_loss    = sl,
            take_profit  = tp,
            risk_reward  = round(rr, 2),
            kill_zone    = kz or "Outside KZ",
            cisd_strength= cisd.get("strength",""),
            d1_bias      = bias,
            price_zone   = zone,
            fvg_found    = bool(fvg),
            ob_found     = bool(ob),
        )

        # Increment session count
        session_counts[session_key] = session_counts.get(session_key, 0) + 1

    # Close any open trade at end of data
    if open_trade is not None:
        last_price           = float(h1_df["Close"].iloc[-1])
        open_trade.exit_price  = last_price
        open_trade.exit_time   = str(h1_df.index[-1])
        open_trade.exit_reason = "END_OF_DATA"
        open_trade.outcome     = "OPEN"
        trades.append(open_trade)

    # ── Calculate final stats ─────────────────────────────────
    result.trades       = trades
    result.equity_curve = equity_curve
    result.total_trades = len([t for t in trades if t.outcome != "OPEN"])
    result.wins         = len([t for t in trades if t.outcome == "WIN"])
    result.losses       = len([t for t in trades if t.outcome == "LOSS"])
    result.total_return = round(cash - INITIAL_CASH, 2)
    result.total_return_pct = round((cash - INITIAL_CASH) / INITIAL_CASH * 100, 2)
    result.max_drawdown     = round(max_drawdown, 2)

    if result.total_trades > 0:
        result.win_rate  = round(result.wins / result.total_trades * 100, 1)

        win_pnls  = [t.pnl_usd for t in trades if t.outcome == "WIN"]
        loss_pnls = [abs(t.pnl_usd) for t in trades if t.outcome == "LOSS"]
        result.avg_win  = round(sum(win_pnls)  / len(win_pnls),  2) if win_pnls  else 0
        result.avg_loss = round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0
        result.profit_factor = round(sum(win_pnls) / max(sum(loss_pnls), 0.01), 2)

        rr_vals     = [t.risk_reward for t in trades if t.risk_reward > 0]
        result.avg_rr = round(sum(rr_vals) / len(rr_vals), 2) if rr_vals else 0

        # Sharpe (annualised from trade returns)
        if len(pnl_list) > 1:
            pnl_arr = np.array(pnl_list)
            result.sharpe = round(
                float(np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-9) * np.sqrt(252)), 2
            )

    return result


# ════════════════════════════════════════════════════════════════
# REPORTING
# ════════════════════════════════════════════════════════════════

def print_results(r: BacktestResult):
    ret_col = "+" if r.total_return >= 0 else ""
    print(f"\n{'═'*60}")
    print(f"📊 ICT BACKTEST RESULTS — {r.symbol} ({r.days_tested} days)")
    print(f"{'═'*60}")
    print(f"  Total trades   : {r.total_trades}")
    print(f"  Win / Loss     : {r.wins}W / {r.losses}L")
    print(f"  Win rate       : {r.win_rate}%")
    print(f"  Total return   : {ret_col}{r.total_return_pct}%  (${ret_col}{r.total_return:,.2f})")
    print(f"  Max drawdown   : -{r.max_drawdown}%")
    print(f"  Sharpe ratio   : {r.sharpe}")
    print(f"  Profit factor  : {r.profit_factor}")
    print(f"  Avg win        : ${r.avg_win:,.2f}")
    print(f"  Avg loss       : ${r.avg_loss:,.2f}")
    print(f"  Avg R:R        : {r.avg_rr}")
    print(f"{'─'*60}")

    if r.trades:
        print(f"\n  📋 Trade Log:")
        print(f"  {'Time':<20} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'PnL':>9} {'Outcome':<8} {'CISD':<10} {'KZ'}")
        print(f"  {'─'*95}")
        for t in r.trades[-20:]:   # show last 20
            pnl_str  = f"${t.pnl_usd:>+.2f}" if t.outcome != "OPEN" else "open"
            out_icon = "✅" if t.outcome=="WIN" else "❌" if t.outcome=="LOSS" else "⏳"
            print(f"  {t.entry_time[:16]:<20} {t.direction:<6} "
                  f"${t.entry_price:>9,.2f} ${t.exit_price:>9,.2f} "
                  f"{pnl_str:>9} {out_icon} {t.outcome:<6} "
                  f"{t.cisd_strength:<10} {t.kill_zone}")


def save_results(r: BacktestResult):
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save trade log CSV
    trade_csv = RESULTS_DIR / f"ict_trades_{r.symbol}_{today}.csv"
    if r.trades:
        rows = []
        for t in r.trades:
            rows.append({
                "symbol":      t.symbol,
                "entry_time":  t.entry_time,
                "exit_time":   t.exit_time,
                "direction":   t.direction,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "stop_loss":   t.stop_loss,
                "take_profit": t.take_profit,
                "risk_reward": t.risk_reward,
                "pnl_usd":     t.pnl_usd,
                "pnl_pct":     t.pnl_pct,
                "outcome":     t.outcome,
                "exit_reason": t.exit_reason,
                "kill_zone":   t.kill_zone,
                "cisd":        t.cisd_strength,
                "d1_bias":     t.d1_bias,
                "zone":        t.price_zone,
                "fvg":         t.fvg_found,
                "ob":          t.ob_found,
            })
        df = pd.DataFrame(rows)
        df.to_csv(trade_csv, index=False)
        print(f"\n  💾 Trade log saved: {trade_csv}")

    # Append summary to master CSV
    summary_csv = RESULTS_DIR / "ict_backtest_summary.csv"
    summary_row = {
        "date":           today,
        "symbol":         r.symbol,
        "days":           r.days_tested,
        "trades":         r.total_trades,
        "wins":           r.wins,
        "losses":         r.losses,
        "win_rate":       r.win_rate,
        "return_pct":     r.total_return_pct,
        "return_usd":     r.total_return,
        "max_drawdown":   r.max_drawdown,
        "sharpe":         r.sharpe,
        "profit_factor":  r.profit_factor,
        "avg_rr":         r.avg_rr,
        "cisd_required":  CISD_STRENGTH_REQUIRED,
        "kill_zone_only": REQUIRE_KILL_ZONE,
        "rr_minimum":     RR_MINIMUM,
    }
    write_header = not summary_csv.exists()
    with open(summary_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_row.keys())
        if write_header:
            w.writeheader()
        w.writerow(summary_row)
    print(f"  💾 Summary saved: {summary_csv}")


# ════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="🎯 ICT Multi-Timeframe Backtester")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Symbol to test e.g. BTC, ETH, SOL (default: all)")
    parser.add_argument("--days", type=int, default=90,
                        help="Days of history to test (default: 90)")
    parser.add_argument("--cisd", type=str, default="STRONG",
                        choices=["STRONG","MODERATE"],
                        help="Required CISD strength (default: STRONG)")
    parser.add_argument("--no-killzone", action="store_true",
                        help="Test without Kill Zone filter")
    parser.add_argument("--rr", type=float, default=2.0,
                        help="Minimum R:R ratio (default: 2.0)")
    args = parser.parse_args()

    # Apply args
    CISD_STRENGTH_REQUIRED = args.cisd
    REQUIRE_KILL_ZONE      = not args.no_killzone
    RR_MINIMUM             = args.rr

    symbols = [args.symbol.upper()] if args.symbol else ["BTC", "ETH", "SOL"]

    print(f"""
🎯 ICT Multi-Timeframe Backtester
{'═'*60}
  Symbols     : {symbols}
  Days        : {args.days}
  CISD needed : {CISD_STRENGTH_REQUIRED}
  Kill Zone   : {'Yes' if REQUIRE_KILL_ZONE else 'No'}
  Min R:R     : {RR_MINIMUM}
  Start cash  : ${INITIAL_CASH:,}
{'═'*60}
""")

    all_results = []
    for sym in symbols:
        print(f"\n{'═'*60}")
        print(f"🔍 Backtesting {sym}...")
        result = run_backtest(sym, days=args.days)
        print_results(result)
        save_results(result)
        all_results.append(result)

    # Combined summary
    if len(all_results) > 1:
        print(f"\n{'═'*60}")
        print("📊 COMBINED SUMMARY")
        print(f"{'─'*60}")
        for r in all_results:
            col = "+" if r.total_return_pct >= 0 else ""
            print(f"  {r.symbol:<6} {col}{r.total_return_pct:>7.1f}%  "
                  f"Sharpe {r.sharpe:>5.2f}  "
                  f"Win {r.win_rate:>5.1f}%  "
                  f"{r.total_trades:>3} trades  "
                  f"DD -{r.max_drawdown:.1f}%")

    print(f"\n✅ Results saved to: {RESULTS_DIR}")
    print(f"   Open ict_backtest_summary.csv to review")
