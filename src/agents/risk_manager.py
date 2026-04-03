#!/usr/bin/env python3
# ============================================================
# 🌙 Risk Manager
#
# Handles all risk calculations before any trade is placed:
#   - Position sizing from SL distance
#   - Daily loss tracking and kill switch
#   - Prop firm limit enforcement
#   - R:R filter
#   - Concurrent position limits
# ============================================================

import json
import csv
from pathlib import Path
from datetime import datetime, date, timezone

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = REPO_ROOT / "src" / "data"
RISK_LOG  = DATA_DIR / "risk_log.csv"
PNL_FILE  = DATA_DIR / "daily_pnl.json"

FUTURES_SYMS = {"MES", "MNQ", "MYM"}


def kelly_fraction(win_rate: float, rr_ratio: float,
                   fraction: float = 0.5) -> float:
    """
    Kelly Criterion position sizing.
    Returns the optimal fraction of account to risk per trade.

    Formula: Kelly % = Win Rate - (1 - Win Rate) / R:R
    We use half-Kelly (fraction=0.5) to reduce volatility.

    Args:
        win_rate:  Historical win rate as decimal (e.g. 0.66 for 66%)
        rr_ratio:  Reward-to-risk ratio (e.g. 2.0 for 2:1)
        fraction:  Kelly fraction — 0.5 = half-Kelly (recommended)

    Examples:
        HistogramFade: 66.2% WR, 2:1 RR → Kelly = 49.3% → half-Kelly = 24.6%
        ZScoreReversion: 89.1% WR, 2:1 RR → Kelly = 83.7% → half-Kelly = 41.9%
        RSIOverbought: 62.9% WR, 2:1 RR → Kelly = 44.4% → half-Kelly = 22.2%

    Note: Always cap at MAX_RISK_PCT to stay within prop firm limits.
    """
    if win_rate <= 0 or rr_ratio <= 0:
        return 0.01  # fallback to 1%

    kelly = win_rate - (1 - win_rate) / rr_ratio
    kelly = max(0.005, kelly)   # never below 0.5%
    kelly = min(0.25,  kelly)   # never above 25% (insane leverage)
    return round(kelly * fraction, 4)


def get_position_risk_pct(signal: dict) -> float:
    """
    Determine risk % for this signal.
    Uses Kelly if strategy has reliable stats, otherwise falls back to config.

    Priority:
      1. Kelly from signal's own win_rate + rr (most accurate)
      2. Flat rate from config (RISK_PER_TRADE_PCT)
    """
    from src.config import RISK_PER_TRADE_PCT, USE_KELLY_SIZING

    if not USE_KELLY_SIZING:
        return RISK_PER_TRADE_PCT

    win_rate = float(signal.get("win_rate", 0))
    rr       = float(signal.get("rr", 0))

    # Need at least 20 trades of history and valid stats to use Kelly
    num_trades = int(signal.get("num_trades", 0))
    if win_rate > 0.3 and rr >= 1.5 and num_trades >= 20:
        kelly = kelly_fraction(win_rate, rr, fraction=0.5)
        # Cap at prop firm safe limit (1% for eval, 0.5% for PA)
        from src.config import PROP_FIRM_ACTIVE, PROP_FIRM_ACCOUNT_TYPE
        if PROP_FIRM_ACTIVE:
            cap = 0.01 if PROP_FIRM_ACCOUNT_TYPE == "eval" else 0.005
            kelly = min(kelly, cap)
        return kelly

    return RISK_PER_TRADE_PCT

# Contract specs for position sizing
CONTRACT_SPECS = {
    "MES": {"tick_size": 0.25, "tick_value": 1.25,   "point_value": 5.0},
    "MNQ": {"tick_size": 0.25, "tick_value": 0.50,   "point_value": 2.0},
    "MYM": {"tick_size": 1.0,  "tick_value": 0.50,   "point_value": 0.50},
}


def load_daily_pnl() -> dict:
    today = str(date.today())
    if not PNL_FILE.exists():
        return {"date": today, "futures_pnl": 0.0, "crypto_pnl": 0.0,
                "total_pnl": 0.0, "trades": 0, "kill_switch": False}
    try:
        data = json.loads(PNL_FILE.read_text())
        if data.get("date") != today:
            # New day — reset
            return {"date": today, "futures_pnl": 0.0, "crypto_pnl": 0.0,
                    "total_pnl": 0.0, "trades": 0, "kill_switch": False}
        return data
    except Exception:
        return {"date": today, "futures_pnl": 0.0, "crypto_pnl": 0.0,
                "total_pnl": 0.0, "trades": 0, "kill_switch": False}


def save_daily_pnl(data: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PNL_FILE.write_text(json.dumps(data, indent=2))


def update_pnl(pnl_usd: float, market: str = "crypto"):
    data = load_daily_pnl()
    if market == "futures":
        data["futures_pnl"] += pnl_usd
    else:
        data["crypto_pnl"] += pnl_usd
    data["total_pnl"] = data["futures_pnl"] + data["crypto_pnl"]
    data["trades"]    += 1
    save_daily_pnl(data)
    return data


def log_risk_event(event: str, detail: str, signal: dict = None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    exists = RISK_LOG.exists()
    with open(RISK_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp","event","detail",
                                           "strategy","symbol","direction"])
        if not exists:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event":     event,
            "detail":    detail,
            "strategy":  (signal or {}).get("strategy", ""),
            "symbol":    (signal or {}).get("symbol", ""),
            "direction": (signal or {}).get("direction", ""),
        })


def calculate_position_size(signal: dict, account_size: float) -> dict:
    """
    Calculate position size based on stop loss distance.
    Also applies macro regime size adjustment from weekly briefing.

    For crypto:
        risk_pct    = get_position_risk_pct(signal)
    risk_amount = account_size * risk_pct
        position_usd = risk_amount / (abs(entry - sl) / entry)

    For futures:
        risk_pct    = get_position_risk_pct(signal)
    risk_amount = account_size * risk_pct
        contracts   = risk_amount / (ticks_to_sl * tick_value)
    """
    from src.config import (RISK_PER_TRADE_PCT, AUTO_SIZE_FROM_SL,
                             MAX_POSITION_SIZE_USD)

    symbol    = signal.get("symbol", "").upper()
    entry     = float(signal.get("entry", 0))
    sl        = float(signal.get("sl", 0))
    direction = signal.get("direction", "LONG")

    if entry <= 0 or sl <= 0:
        return {"size_usd": MAX_POSITION_SIZE_USD, "contracts": 1,
                "risk_usd": 0, "error": "Invalid entry/SL"}

    risk_pct    = get_position_risk_pct(signal)
    risk_amount = account_size * risk_pct

    if symbol in FUTURES_SYMS:
        spec = CONTRACT_SPECS.get(symbol, CONTRACT_SPECS["MES"])
        sl_points     = abs(entry - sl)
        sl_ticks      = sl_points / spec["tick_size"]
        risk_per_cont = sl_ticks * spec["tick_value"]

        if risk_per_cont <= 0:
            contracts = 1
        else:
            contracts = max(1, int(risk_amount / risk_per_cont))

        size_usd  = contracts * spec["point_value"] * entry
        actual_risk = contracts * risk_per_cont

        return {
            "contracts":   contracts,
            "size_usd":    round(size_usd, 2),
            "risk_usd":    round(actual_risk, 2),
            "risk_pct":    round(actual_risk / account_size * 100, 2),
            "sl_ticks":    round(sl_ticks, 1),
            "sl_points":   round(sl_points, 2),
        }
    else:
        # Crypto — size in USD
        sl_pct    = abs(entry - sl) / entry
        if sl_pct <= 0:
            size_usd = risk_amount
        elif AUTO_SIZE_FROM_SL:
            size_usd = risk_amount / sl_pct
        else:
            size_usd = MAX_POSITION_SIZE_USD

        size_usd    = min(size_usd, MAX_POSITION_SIZE_USD)

        # Apply macro regime adjustment from weekly briefing
        try:
            from src.agents.weekly_briefing import get_size_adjustment
            macro_adj = get_size_adjustment()
            if macro_adj != 1.0:
                size_usd = size_usd * macro_adj
        except Exception:
            macro_adj = 1.0

        actual_risk = size_usd * sl_pct

        return {
            "contracts":  None,
            "size_usd":   round(size_usd, 2),
            "risk_usd":   round(actual_risk, 2),
            "risk_pct":   round(actual_risk / account_size * 100, 2),
            "sl_pct":     round(sl_pct * 100, 2),
            "macro_adj":  macro_adj,
        }


def check_risk(signal: dict, open_positions: list = None) -> tuple[bool, str]:
    """
    Full risk check before placing a trade.
    Returns (approved: bool, reason: str).
    """
    from src.config import (
        RISK_PER_TRADE_PCT, MIN_RR_RATIO,
        MAX_DAILY_LOSS_PCT, MAX_DAILY_LOSS_FUTURES, MAX_DAILY_LOSS_CRYPTO,
        MAX_CONCURRENT_FUTURES, MAX_CONCURRENT_CRYPTO,
        PROP_FIRM_ACTIVE, PROP_FIRM_MAX_DAILY_LOSS_PCT,
        KILL_SWITCH_DAILY_LOSS_PCT,
        HYPERLIQUID_ACCOUNT_SIZE, TRADOVATE_ACCOUNT_SIZE,
    )

    symbol    = signal.get("symbol", "").upper()
    is_futures = symbol in FUTURES_SYMS
    account   = TRADOVATE_ACCOUNT_SIZE if is_futures else HYPERLIQUID_ACCOUNT_SIZE
    market    = "futures" if is_futures else "crypto"
    open_pos  = open_positions or []

    # ── 1. Kill switch check ───────────────────────────────────
    pnl_data = load_daily_pnl()
    if pnl_data.get("kill_switch"):
        return False, "🚫 KILL SWITCH ACTIVE — daily loss limit hit, no more trades today"

    # ── 2. Daily loss limit ────────────────────────────────────
    total_loss  = min(0, pnl_data["total_pnl"])
    loss_pct    = abs(total_loss) / account if account > 0 else 0

    daily_limit = PROP_FIRM_MAX_DAILY_LOSS_PCT if PROP_FIRM_ACTIVE else MAX_DAILY_LOSS_PCT

    if loss_pct >= KILL_SWITCH_DAILY_LOSS_PCT:
        pnl_data["kill_switch"] = True
        save_daily_pnl(pnl_data)
        log_risk_event("KILL_SWITCH", f"Daily loss {loss_pct*100:.1f}% hit limit", signal)
        return False, f"🚫 KILL SWITCH — daily loss {loss_pct*100:.1f}% ≥ {KILL_SWITCH_DAILY_LOSS_PCT*100:.1f}%"

    if is_futures and abs(pnl_data["futures_pnl"]) >= MAX_DAILY_LOSS_FUTURES:
        return False, f"⛔ Futures daily loss ${abs(pnl_data['futures_pnl']):.0f} ≥ ${MAX_DAILY_LOSS_FUTURES} limit"

    if not is_futures and abs(pnl_data["crypto_pnl"]) >= MAX_DAILY_LOSS_CRYPTO:
        return False, f"⛔ Crypto daily loss ${abs(pnl_data['crypto_pnl']):.0f} ≥ ${MAX_DAILY_LOSS_CRYPTO} limit"

    # ── 3. R:R check ──────────────────────────────────────────
    rr = float(signal.get("rr", 0))
    if rr < MIN_RR_RATIO:
        return False, f"⛔ R:R {rr:.1f} < minimum {MIN_RR_RATIO} — skipping"

    # ── 4. Concurrent positions ────────────────────────────────
    open_futures = [p for p in open_pos if p.get("symbol","") in FUTURES_SYMS]
    open_crypto  = [p for p in open_pos if p.get("symbol","") not in FUTURES_SYMS]

    if is_futures and len(open_futures) >= MAX_CONCURRENT_FUTURES:
        return False, f"⛔ Already have {len(open_futures)} futures position(s) open (max {MAX_CONCURRENT_FUTURES})"

    if not is_futures and len(open_crypto) >= MAX_CONCURRENT_CRYPTO:
        return False, f"⛔ Already have {len(open_crypto)} crypto position(s) open (max {MAX_CONCURRENT_CRYPTO})"

    # ── 5. Prop firm total drawdown ────────────────────────────
    if PROP_FIRM_ACTIVE:
        from src.config import PROP_FIRM_MAX_TOTAL_LOSS_PCT
        # Would need equity tracking for total drawdown — skip for now
        pass

    return True, "✅ Risk checks passed"


def print_risk_summary(signal: dict, sizing: dict, check_result: tuple):
    approved, reason = check_result
    pnl = load_daily_pnl()
    symbol = signal.get("symbol","")
    is_fut = symbol.upper() in FUTURES_SYMS

    from src.config import HYPERLIQUID_ACCOUNT_SIZE, TRADOVATE_ACCOUNT_SIZE
    account = TRADOVATE_ACCOUNT_SIZE if is_fut else HYPERLIQUID_ACCOUNT_SIZE

    print(f"""
  ⚖️  RISK ASSESSMENT
  ─────────────────────────────────────────────
  Account     : ${account:,.0f} ({'Tradovate SIM' if is_fut else 'Hyperliquid'})
  Risk/trade  : {sizing.get('risk_pct',0):.1f}% = ${sizing.get('risk_usd',0):.2f}
  Position    : ${sizing.get('size_usd',0):.2f}
  {'Contracts  : ' + str(sizing.get('contracts','')) if is_fut else f"SL distance: {sizing.get('sl_pct',0):.1f}%"}
  R:R ratio   : {signal.get('rr',0):.1f}
  Daily P&L   : ${pnl['total_pnl']:+.2f} ({pnl['trades']} trades today)
  Decision    : {reason}
  ─────────────────────────────────────────────""")
    return approved