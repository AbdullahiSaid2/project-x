# ============================================================
# 🌙 ICT Auto Executor
#
# Extends the ICT Scanner with automated M5 CISD detection
# and trade execution. When an A+ setup fires on D1/H1,
# this agent drops to M5, confirms CISD quality, then
# executes the trade automatically with SL and TP.
#
# EXECUTION LOGIC:
#   A+ Scanner alert
#        ↓
#   M5 CISD check (must be STRONG, not weak/messy)
#        ↓
#   Risk agent approval
#        ↓
#   Entry at FVG/OB level via limit order
#   SL below/above M5 structure
#   TP at PDH (bullish) or PDL (bearish)
#
# HOW TO RUN:
#   python src/agents/ict_executor.py
#   python src/agents/ict_executor.py --once
#   python src/agents/ict_executor.py --symbol ETH
#
# ⚠️  IMPORTANT: Only run after forward testing for 2-4 weeks.
#     Start with ICT_TRADE_SIZE_USD = 10 to test execution.
# ============================================================

import sys
import csv
import json
import time
import traceback
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config               import (EXCHANGE, SLEEP_BETWEEN_RUNS_SEC,
                                       STOP_LOSS_PCT, TAKE_PROFIT_PCT)
from src.data.fetcher          import get_ohlcv
from src.models.llm_router import model
from src.exchanges.router      import get_price, buy, sell, active_symbols
from src.agents.risk_agent     import risk
from src.agents.ict_scanner    import (
    scan_symbol, ICTSetup, SYMBOLS,
    check_kill_zone, print_setup, log_setup, GRADE_ICONS
)

# ── paths ─────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[2]
EXEC_LOG     = REPO_ROOT / "src" / "data" / "ict_exec_log.csv"
EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── execution config ──────────────────────────────────────────
ICT_TRADE_SIZE_USD     = 50       # ← start small, increase when confident
MIN_SETUP_GRADE        = "A+"     # only execute on A+ (change to "A" for more trades)
REQUIRE_KILL_ZONE      = True     # only trade during Kill Zones
CISD_STRENGTH_REQUIRED = "STRONG" # "STRONG" | "MODERATE" — STRONG = fewer, cleaner trades
MAX_TRADES_PER_SESSION = 2        # max trades per Kill Zone session
RR_MINIMUM             = 2.0      # minimum risk:reward ratio required
SCAN_INTERVAL_SECS     = 300      # scan every 5 minutes


# ── trade result dataclass ────────────────────────────────────
@dataclass
class ICTTrade:
    symbol:          str
    timestamp:       str
    direction:       str   # LONG | SHORT
    setup_grade:     str
    entry_price:     float = 0.0
    stop_loss:       float = 0.0
    take_profit:     float = 0.0
    risk_reward:     float = 0.0
    size_usd:        float = 0.0
    cisd_strength:   str   = ""
    cisd_candle:     str   = ""
    fvg_level:       float = 0.0
    ob_level:        float = 0.0
    kill_zone:       str   = ""
    executed:        bool  = False
    blocked_reason:  str   = ""
    ai_confirmation: str   = ""


# ════════════════════════════════════════════════════════════════
# M5 CISD DETECTOR
# The critical quality filter — this is what separates A+ from noise
# ════════════════════════════════════════════════════════════════

def detect_m5_cisd(symbol: str, bias: str) -> dict:
    """
    Detect Change in State of Delivery (CISD) on M5.

    A strong CISD requires:
    1. A displacement candle: large body (>60% of range), closes near extreme
    2. Break of the most recent M5 swing structure
    3. Candle size significantly above M5 average (confirms momentum)
    4. NOT followed immediately by an equal opposite candle (no indecision)

    Returns dict with:
      found:    bool
      strength: STRONG | MODERATE | WEAK
      candle:   description of the CISD candle
      entry:    suggested entry price (50% of CISD candle or FVG level)
      reason:   why it passed or failed
    """
    try:
        # Get last 60 M5 candles
        # Priority: Databento 5m (precise) → yfinance 15m (proxy)
        m5_data = None
        try:
            from src.data.databento_fetcher import get_databento_ohlcv, FUTURES_DB_MAP
            import os
            if os.getenv("DATABENTO_API_KEY") and symbol.upper() in FUTURES_DB_MAP:
                m5_data = get_databento_ohlcv(symbol, "5m", days_back=5)
                print(f"     📡 M5 data: Databento ({len(m5_data)} candles)")
        except Exception:
            pass

        if m5_data is None or m5_data.empty:
            # Fall back to 15m as proxy (yfinance doesn't have reliable 5m)
            m5_data = get_ohlcv(
                symbol.replace("-USD", ""),
                exchange=EXCHANGE,
                timeframe="15m",
                days_back=3,
            )
            print(f"     📡 M5 data: 15m proxy ({len(m5_data)} candles)")

        df = m5_data
        if len(df) < 20:
            return {"found": False, "strength": "NONE", "reason": "Insufficient M5 data"}

        recent   = df.tail(20)
        avg_range = float((df["High"] - df["Low"]).tail(40).mean())

        # Find the most recent candle matching the bias
        cisd_candles = []
        for i in range(-10, 0):
            c     = recent.iloc[i]
            rng   = float(c["High"] - c["Low"])
            body  = abs(float(c["Close"]) - float(c["Open"]))
            if rng == 0:
                continue

            body_pct  = body / rng
            size_ratio = rng / max(avg_range, 0.001)

            is_bullish = float(c["Close"]) > float(c["Open"])
            is_bearish = float(c["Close"]) < float(c["Open"])

            matched = (bias == "BULLISH" and is_bullish) or (bias == "BEARISH" and is_bearish)
            if not matched:
                continue

            # Score the candle quality
            score = 0
            if body_pct >= 0.75:  score += 3   # strong body
            elif body_pct >= 0.60: score += 2
            elif body_pct >= 0.45: score += 1

            if size_ratio >= 2.0:  score += 3   # much larger than average
            elif size_ratio >= 1.5: score += 2
            elif size_ratio >= 1.2: score += 1

            # Check close near extreme (bullish: close near high, bearish: close near low)
            if bias == "BULLISH":
                close_pos = (float(c["Close"]) - float(c["Low"])) / rng
                if close_pos >= 0.80: score += 2
                elif close_pos >= 0.65: score += 1
            else:
                close_pos = (float(c["High"]) - float(c["Close"])) / rng
                if close_pos >= 0.80: score += 2
                elif close_pos >= 0.65: score += 1

            if score >= 7:
                strength = "STRONG"
            elif score >= 4:
                strength = "MODERATE"
            else:
                strength = "WEAK"

            entry_price = float(c["Close"])
            candle_desc = (
                f"{bias} displacement: body {body_pct:.0%}, "
                f"{size_ratio:.1f}x avg size, score {score}/8"
            )

            cisd_candles.append({
                "found":    True,
                "strength": strength,
                "score":    score,
                "candle":   candle_desc,
                "entry":    round(entry_price, 4),
                "body_pct": round(body_pct, 3),
                "size_ratio": round(size_ratio, 2),
                "reason":   f"CISD detected — {strength}",
            })

        if not cisd_candles:
            return {
                "found":    False,
                "strength": "NONE",
                "entry":    0.0,
                "reason":   f"No {bias} displacement candle found on M5",
            }

        # Return the most recent strong one, or best available
        cisd_candles.sort(key=lambda x: x["score"], reverse=True)
        return cisd_candles[0]

    except Exception as e:
        return {"found": False, "strength": "NONE", "entry": 0.0, "reason": f"M5 error: {e}"}


# ════════════════════════════════════════════════════════════════
# RISK:REWARD CALCULATOR
# ════════════════════════════════════════════════════════════════

def calculate_rr(entry: float, stop: float, target: float, direction: str) -> dict:
    """
    Calculate risk:reward ratio.
    Returns dict with risk_pips, reward_pips, rr_ratio.
    """
    if direction == "LONG":
        risk_pts   = entry - stop
        reward_pts = target - entry
    else:
        risk_pts   = stop - entry
        reward_pts = entry - target

    if risk_pts <= 0:
        return {"valid": False, "rr": 0, "risk_pts": 0, "reward_pts": 0}

    rr = reward_pts / risk_pts
    return {
        "valid":      rr >= RR_MINIMUM,
        "rr":         round(rr, 2),
        "risk_pts":   round(risk_pts, 4),
        "reward_pts": round(reward_pts, 4),
        "risk_pct":   round(risk_pts / entry * 100, 2),
    }


# ════════════════════════════════════════════════════════════════
# AI CONFIRMATION (optional final check)
# ════════════════════════════════════════════════════════════════

EXEC_CONFIRM_PROMPT = """You are an ICT execution specialist making a final go/no-go decision.

SETUP DETAILS:
{data}

Answer with ONLY valid JSON:
{{
  "execute": true | false,
  "confidence": 0.0-1.0,
  "reason": "one sentence",
  "concern": "one risk to be aware of or null"
}}

Be conservative — only approve if the setup is genuinely clean."""


def ai_execution_check(trade: ICTTrade, setup: ICTSetup) -> dict:
    """Final AI sanity check before execution."""
    try:
        data = {
            "symbol":       trade.symbol,
            "direction":    trade.direction,
            "grade":        trade.setup_grade,
            "d1_bias":      setup.d1_bias,
            "price_zone":   setup.price_zone,
            "entry":        trade.entry_price,
            "stop_loss":    trade.stop_loss,
            "take_profit":  trade.take_profit,
            "risk_reward":  trade.risk_reward,
            "cisd_strength":trade.cisd_strength,
            "kill_zone":    trade.kill_zone,
            "fvg_found":    setup.h1_fvg_found,
            "ob_found":     setup.h1_ob_found,
            "displacement": setup.h1_disp_strength,
        }
        raw    = model.chat(
            system_prompt="You are a conservative ICT trade execution specialist. Return only valid JSON.",
            user_prompt=EXEC_CONFIRM_PROMPT.format(data=json.dumps(data, indent=2)),
        )
        raw    = raw.replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception as e:
        return {"execute": False, "confidence": 0, "reason": f"AI check failed: {e}"}


# ════════════════════════════════════════════════════════════════
# EXECUTION ENGINE
# ════════════════════════════════════════════════════════════════

def build_trade(setup: ICTSetup, cisd: dict, price: float) -> ICTTrade:
    """Build a trade object from the ICT setup and CISD data."""
    direction = "LONG" if setup.d1_bias == "BULLISH" else "SHORT"

    # Entry: use CISD entry or best PD array level
    entry = cisd.get("entry", price)
    if entry == 0:
        entry = price

    # Stop loss: below M5 structure for longs, above for shorts
    # Use a 0.5% buffer below/above the CISD candle extreme
    if direction == "LONG":
        sl = round(entry * (1 - 0.005), 4)   # 0.5% below entry
    else:
        sl = round(entry * (1 + 0.005), 4)   # 0.5% above entry

    # Target: PDH for longs, PDL for shorts (ERL from D1)
    tp = setup.prev_day_high if direction == "LONG" else setup.prev_day_low
    if tp == 0:
        # Fallback if PDH/PDL not available
        tp = entry * (1 + 0.02) if direction == "LONG" else entry * (1 - 0.02)

    # Get best PD array level
    pd_level = 0.0
    if setup.h1_fvg_found and setup.h1_fvg_level:
        pd_level = setup.h1_fvg_level
    elif setup.h1_ob_found and setup.h1_ob_level:
        pd_level = setup.h1_ob_level

    # RR calculation
    rr_data = calculate_rr(entry, sl, tp, direction)

    return ICTTrade(
        symbol=      setup.symbol,
        timestamp=   datetime.now().isoformat(),
        direction=   direction,
        setup_grade= setup.setup_grade,
        entry_price= round(entry, 4),
        stop_loss=   sl,
        take_profit= round(tp, 4),
        risk_reward= rr_data.get("rr", 0),
        size_usd=    ICT_TRADE_SIZE_USD,
        cisd_strength= cisd.get("strength", ""),
        cisd_candle= cisd.get("candle", ""),
        fvg_level=   setup.h1_fvg_level,
        ob_level=    setup.h1_ob_level,
        kill_zone=   setup.kill_zone_name,
    )


def execute_trade(trade: ICTTrade) -> bool:
    """Execute the trade via the exchange router."""
    try:
        if trade.direction == "LONG":
            result = buy(trade.symbol, trade.size_usd)
        else:
            result = sell(trade.symbol, trade.size_usd)

        trade.executed = True
        print(f"\n  ✅ TRADE EXECUTED")
        print(f"     {trade.direction} {trade.symbol}")
        print(f"     Entry : ${trade.entry_price:,.4f}")
        print(f"     SL    : ${trade.stop_loss:,.4f}  ({abs(trade.entry_price-trade.stop_loss)/trade.entry_price*100:.2f}% risk)")
        print(f"     TP    : ${trade.take_profit:,.4f}  (R:R {trade.risk_reward:.1f})")
        print(f"     Size  : ${trade.size_usd}")
        return True

    except Exception as e:
        trade.executed      = False
        trade.blocked_reason = f"Exchange error: {e}"
        print(f"\n  ❌ Execution failed: {e}")
        traceback.print_exc()
        return False


def log_execution(trade: ICTTrade):
    """Log every execution attempt (successful or blocked) to CSV."""
    row = {
        "timestamp":       trade.timestamp,
        "symbol":          trade.symbol,
        "direction":       trade.direction,
        "grade":           trade.setup_grade,
        "entry":           trade.entry_price,
        "stop_loss":       trade.stop_loss,
        "take_profit":     trade.take_profit,
        "risk_reward":     trade.risk_reward,
        "size_usd":        trade.size_usd,
        "cisd_strength":   trade.cisd_strength,
        "kill_zone":       trade.kill_zone,
        "executed":        trade.executed,
        "blocked_reason":  trade.blocked_reason,
        "ai_confirmation": trade.ai_confirmation,
    }
    write_header = not EXEC_LOG.exists()
    with open(EXEC_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


# ════════════════════════════════════════════════════════════════
# GATE CHECKS — every condition that must pass before execution
# ════════════════════════════════════════════════════════════════

def run_gate_checks(setup: ICTSetup, cisd: dict, trade: ICTTrade) -> tuple[bool, str]:
    """
    Run all pre-execution gates in order.
    Returns (passed: bool, reason: str).
    """

    # Gate 1 — Setup grade
    grade_order = {"A+": 3, "A": 2, "B": 1, "NO SETUP": 0}
    min_order   = grade_order.get(MIN_SETUP_GRADE, 3)
    if grade_order.get(setup.setup_grade, 0) < min_order:
        return False, f"Grade {setup.setup_grade} below minimum {MIN_SETUP_GRADE}"

    # Gate 2 — Kill Zone (if required)
    if REQUIRE_KILL_ZONE and not setup.in_kill_zone:
        kz = check_kill_zone()
        return False, f"Not in Kill Zone — next: {kz.get('next_session','?')} in {kz.get('mins_to_next','?')}m"

    # Gate 3 — CISD found and strong enough
    if not cisd.get("found"):
        return False, f"No M5 CISD found: {cisd.get('reason','')}"

    strength_order = {"STRONG": 2, "MODERATE": 1, "WEAK": 0, "NONE": -1}
    required       = strength_order.get(CISD_STRENGTH_REQUIRED, 2)
    actual         = strength_order.get(cisd.get("strength","NONE"), -1)
    if actual < required:
        return False, f"CISD strength {cisd.get('strength')} below required {CISD_STRENGTH_REQUIRED}"

    # Gate 4 — D1 bias + zone alignment
    if setup.d1_bias == "BULLISH" and "PREMIUM" in setup.price_zone:
        return False, f"Bullish bias but price in PREMIUM zone ({setup.zone_pct:.0f}%) — wait for discount"
    if setup.d1_bias == "BEARISH" and "DISCOUNT" in setup.price_zone:
        return False, f"Bearish bias but price in DISCOUNT zone ({setup.zone_pct:.0f}%) — wait for premium"

    # Gate 5 — Risk:Reward
    if trade.risk_reward < RR_MINIMUM:
        return False, f"R:R {trade.risk_reward:.2f} below minimum {RR_MINIMUM}"

    # Gate 6 — Risk agent
    direction = "buy" if trade.direction == "LONG" else "sell"
    allowed, reason = risk.check_trade(trade.symbol, trade.size_usd, direction)
    if not allowed:
        return False, f"Risk agent blocked: {reason}"

    return True, "All gates passed"


# ════════════════════════════════════════════════════════════════
# MAIN EXECUTOR CLASS
# ════════════════════════════════════════════════════════════════

class ICTExecutor:

    def __init__(self):
        self.trades_this_session: dict[str, int] = {}   # kill_zone → trade count
        self.last_trade_symbol:   dict[str, str] = {}   # symbol → timestamp of last trade

        print("🎯⚡ ICT Auto Executor initialised")
        print(f"   Min grade       : {MIN_SETUP_GRADE}")
        print(f"   CISD required   : {CISD_STRENGTH_REQUIRED}")
        print(f"   Kill Zone only  : {REQUIRE_KILL_ZONE}")
        print(f"   Min R:R         : {RR_MINIMUM}")
        print(f"   Trade size      : ${ICT_TRADE_SIZE_USD}")
        print(f"   Max trades/sess : {MAX_TRADES_PER_SESSION}")
        print(f"\n  ⚠️  LIVE EXECUTION ACTIVE — real trades will be placed\n")

    def _session_trade_count(self) -> int:
        kz = check_kill_zone()
        key = kz.get("name", "outside")
        return self.trades_this_session.get(key, 0)

    def _increment_session_count(self):
        kz  = check_kill_zone()
        key = kz.get("name", "outside")
        self.trades_this_session[key] = self.trades_this_session.get(key, 0) + 1

    def process_symbol(self, symbol: str) -> ICTTrade | None:
        """
        Full pipeline for one symbol:
        Scan → Gate checks → M5 CISD → AI confirm → Execute
        """
        print(f"\n{'─'*60}")
        print(f"  🔍 Processing {symbol}...")

        # 1 — Run D1/H1 scan
        setup = scan_symbol(symbol)
        print_setup(setup)
        log_setup(setup)

        # Quick pre-filter before expensive M5 check
        if setup.setup_grade not in ("A+", "A"):
            print(f"  ⏸️  Grade {setup.setup_grade} — skipping execution")
            return None

        if REQUIRE_KILL_ZONE and not setup.in_kill_zone:
            kz = check_kill_zone()
            print(f"  ⏸️  Not in Kill Zone — next in {kz.get('mins_to_next','?')}m")
            return None

        # 2 — Check session trade limit
        if self._session_trade_count() >= MAX_TRADES_PER_SESSION:
            print(f"  🚫 Max trades ({MAX_TRADES_PER_SESSION}) reached this session")
            return None

        # 3 — Get current price
        try:
            price = get_price(symbol)
        except Exception as e:
            print(f"  ❌ Could not get price: {e}")
            return None

        # 4 — M5 CISD check
        print(f"\n  📐 Checking M5 for CISD...")
        cisd = detect_m5_cisd(symbol, setup.d1_bias)
        cisd_icon = "✅" if cisd.get("found") else "❌"
        print(f"  {cisd_icon} CISD: {cisd.get('strength','NONE')} — {cisd.get('reason','')}")

        # 5 — Build trade object
        trade = build_trade(setup, cisd, price)

        # 6 — Run all gate checks
        passed, gate_reason = run_gate_checks(setup, cisd, trade)
        if not passed:
            trade.blocked_reason = gate_reason
            print(f"\n  🚫 BLOCKED: {gate_reason}")
            log_execution(trade)
            return None

        print(f"\n  ✅ All gates passed!")
        print(f"     Direction  : {trade.direction}")
        print(f"     Entry      : ${trade.entry_price:,.4f}")
        print(f"     Stop Loss  : ${trade.stop_loss:,.4f}")
        print(f"     Take Profit: ${trade.take_profit:,.4f}")
        print(f"     R:R        : {trade.risk_reward:.2f}")
        print(f"     CISD       : {trade.cisd_strength}")

        # 7 — AI final confirmation
        print(f"\n  🤖 Running AI final check...")
        ai_result = ai_execution_check(trade, setup)
        trade.ai_confirmation = ai_result.get("reason","")
        ai_approved = ai_result.get("execute", False)
        ai_conf     = ai_result.get("confidence", 0)

        print(f"  {'✅' if ai_approved else '❌'} AI: {ai_result.get('reason','')} (confidence: {ai_conf:.0%})")
        if ai_result.get("concern"):
            print(f"  ⚠️  Risk note: {ai_result['concern']}")

        if not ai_approved:
            trade.blocked_reason = f"AI rejected: {trade.ai_confirmation}"
            log_execution(trade)
            return None

        # 8 — EXECUTE
        print(f"\n  🚀 Executing trade...")
        success = execute_trade(trade)
        log_execution(trade)

        if success:
            self._increment_session_count()
            self.last_trade_symbol[symbol] = trade.timestamp

        return trade if success else None

    def run_once(self) -> list:
        kz     = check_kill_zone()
        kz_str = f"🔥 {kz['name']}" if kz.get("active") else f"Next KZ: {kz.get('next_session','—')} in {kz.get('mins_to_next','?')}m"

        print(f"\n{'═'*60}")
        print(f"🎯⚡ ICT Executor scan — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   {kz_str} | Trades this session: {self._session_trade_count()}/{MAX_TRADES_PER_SESSION}")
        print(f"{'═'*60}")

        # Show portfolio first
        risk.portfolio_summary()

        executed = []
        for symbol in SYMBOLS:
            result = self.process_symbol(symbol)
            if result:
                executed.append(result)
            time.sleep(1)

        print(f"\n{'═'*60}")
        if executed:
            print(f"✅ {len(executed)} trade(s) executed this scan:")
            for t in executed:
                print(f"   {t.direction} {t.symbol} @ ${t.entry_price:,.4f} | R:R {t.risk_reward:.1f} | {t.cisd_strength} CISD")
        else:
            print("⏸️  No trades executed this scan.")

        return executed

    def run(self):
        print("🚀 ICT Auto Executor running. Press Ctrl+C to stop.\n")
        try:
            while True:
                self.run_once()
                print(f"\n😴 Next scan in {SCAN_INTERVAL_SECS//60} minutes...")
                time.sleep(SCAN_INTERVAL_SECS)
        except KeyboardInterrupt:
            print("\n🛑 ICT Executor stopped.")


# ── entrypoint ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="🎯⚡ ICT Auto Executor")
    parser.add_argument("--once",   action="store_true",
                        help="Single scan then exit")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Process single symbol only e.g. ETH")
    parser.add_argument("--dryrun", action="store_true",
                        help="Run all checks but do NOT execute trades (safe testing)")
    args = parser.parse_args()

    if args.dryrun:
        # Monkey-patch execute_trade to simulate without real orders
        def fake_execute(trade):
            print(f"\n  🧪 DRY RUN — would have executed {trade.direction} {trade.symbol} @ ${trade.entry_price:,.4f}")
            trade.executed      = True
            trade.blocked_reason = "DRY RUN"
            return True
        import src.agents.ict_executor as _self
        _self.execute_trade = fake_execute
        print("🧪 DRY RUN MODE — no real trades will be placed\n")

    executor = ICTExecutor()

    if args.symbol:
        sym = args.symbol.upper()
        executor.process_symbol(sym)
    elif args.once:
        executor.run_once()
    else:
        executor.run()
