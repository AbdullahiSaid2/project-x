# ============================================================
# 🌙 ICT Setup Scanner
#
# Scans for ICT A+ setups across D1 → H1 confluence.
# YOU make the final M5 entry decision — this handles scanning.
#
# THE 4-STEP CHECK (per scan):
#   1. D1  — Direction: PDH/PDL bias + structure
#   2. D1  — Objective: Are we in premium or discount?
#   3. H1  — Confirmation: FVG / OB tapped + displacement?
#   4. TIME — Kill Zone: London or New York AM only?
#
# HOW TO RUN:
#   python src/agents/ict_scanner.py
#   python src/agents/ict_scanner.py --once
#   python src/agents/ict_scanner.py --symbol ETH
# ============================================================

import sys
import csv
import json
import time
import pytz
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config               import EXCHANGE, HYPERLIQUID_TOKENS, COINBASE_TOKENS
from src.data.fetcher          import get_ohlcv
from src.models.llm_router import model

# ── paths ─────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
SCANNER_LOG = REPO_ROOT / "src" / "data" / "ict_scanner_log.csv"
SCANNER_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── config ────────────────────────────────────────────────────
SYMBOLS       = ["BTC", "ETH", "SOL"] if EXCHANGE == "hyperliquid" else ["BTC-USD", "ETH-USD", "SOL-USD"]
SCAN_INTERVAL = 300    # scan every 5 minutes

# ── Kill Zone times (EST / UTC-5) ─────────────────────────────
# London Open:  02:00 – 05:00 EST
# New York AM:  07:00 – 10:00 EST
EST = pytz.timezone("America/New_York")

KILL_ZONES = {
    "London Open":    (2, 5),
    "New York AM":    (7, 10),
    "London Close":   (10, 12),   # secondary
}


# ── Setup result dataclass ────────────────────────────────────
@dataclass
class ICTSetup:
    symbol:           str
    timestamp:        str
    # Step 1 — D1 Direction
    d1_bias:          str = "NEUTRAL"      # BULLISH | BEARISH | NEUTRAL
    d1_structure:     str = ""
    prev_day_high:    float = 0.0
    prev_day_low:     float = 0.0
    current_price:    float = 0.0
    # Step 2 — Premium / Discount
    daily_range_mid:  float = 0.0
    price_zone:       str = ""             # DISCOUNT | PREMIUM | EQUILIBRIUM
    zone_pct:         float = 0.0          # how far into discount/premium (0-100%)
    # Step 3 — H1 Confirmation
    h1_fvg_found:     bool = False
    h1_fvg_level:     float = 0.0
    h1_fvg_type:      str = ""             # BULLISH | BEARISH
    h1_ob_found:      bool = False
    h1_ob_level:      float = 0.0
    h1_ob_type:       str = ""
    h1_displacement:  bool = False
    h1_disp_strength: str = ""             # STRONG | WEAK | NONE
    h1_swept_liq:     bool = False
    # Step 4 — Kill Zone
    in_kill_zone:     bool = False
    kill_zone_name:   str = ""
    current_session:  str = ""
    # Overall
    setup_score:      int = 0              # 0-4 (how many steps confirmed)
    setup_grade:      str = "NO SETUP"    # A+ | A | B | NO SETUP
    alert:            bool = False
    action:           str = ""            # what to do on M5
    reasoning:        str = ""
    ai_notes:         str = ""


# ════════════════════════════════════════════════════════════════
# STEP 1 — D1 DIRECTION
# ════════════════════════════════════════════════════════════════

def analyse_d1_direction(symbol: str) -> dict:
    """
    Determine D1 bias using:
    - Previous day high/low (PDH/PDL)
    - Whether price broke PDH (bullish) or PDL (bearish)
    - Last 5-day swing structure
    """
    try:
        df = get_ohlcv(symbol, exchange=EXCHANGE, timeframe="1D", days_back=20)
        if len(df) < 5:
            return {"bias": "NEUTRAL", "structure": "Insufficient data"}

        # Previous day candle
        prev_day    = df.iloc[-2]
        today_open  = float(df.iloc[-1]["Open"])
        curr_price  = float(df.iloc[-1]["Close"])
        pdh         = float(prev_day["High"])
        pdl         = float(prev_day["Low"])
        pdc         = float(prev_day["Close"])

        # 5-day structure: higher highs / lower lows
        highs  = df["High"].iloc[-6:-1].values
        lows   = df["Low"].iloc[-6:-1].values
        hh     = highs[-1] > highs[-2] > highs[-3]   # higher highs
        hl     = lows[-1]  > lows[-2]  > lows[-3]    # higher lows
        lh     = highs[-1] < highs[-2] < highs[-3]   # lower highs
        ll     = lows[-1]  < lows[-2]  < lows[-3]    # lower lows

        # D1 bias logic
        if curr_price > pdh:
            bias      = "BULLISH"
            structure = f"Price broke PDH ${pdh:,.2f} — bullish expansion"
        elif curr_price < pdl:
            bias      = "BEARISH"
            structure = f"Price broke PDL ${pdl:,.2f} — bearish expansion"
        elif hh and hl:
            bias      = "BULLISH"
            structure = "Higher highs + higher lows — uptrend structure"
        elif lh and ll:
            bias      = "BEARISH"
            structure = "Lower highs + lower lows — downtrend structure"
        elif curr_price > pdc:
            bias      = "BULLISH"
            structure = f"Above prev day close ${pdc:,.2f} — mild bullish"
        elif curr_price < pdc:
            bias      = "BEARISH"
            structure = f"Below prev day close ${pdc:,.2f} — mild bearish"
        else:
            bias      = "NEUTRAL"
            structure = "No clear D1 direction"

        return {
            "bias":      bias,
            "structure": structure,
            "pdh":       pdh,
            "pdl":       pdl,
            "price":     curr_price,
        }
    except Exception as e:
        return {"bias": "NEUTRAL", "structure": f"D1 error: {e}",
                "pdh": 0, "pdl": 0, "price": 0}


# ════════════════════════════════════════════════════════════════
# STEP 2 — PREMIUM / DISCOUNT
# ════════════════════════════════════════════════════════════════

def analyse_premium_discount(price: float, pdh: float, pdl: float) -> dict:
    """
    ICT Premium/Discount:
    - Below 50% of PDH-PDL range = DISCOUNT (look to buy)
    - Above 50% of PDH-PDL range = PREMIUM  (look to sell)
    - 40-60% range = EQUILIBRIUM (avoid)
    """
    if pdh == 0 or pdl == 0:
        return {"zone": "UNKNOWN", "pct": 50.0, "mid": 0}

    day_range = pdh - pdl
    if day_range <= 0:
        return {"zone": "EQUILIBRIUM", "pct": 50.0, "mid": (pdh + pdl) / 2}

    mid       = pdl + (day_range * 0.5)
    pct       = ((price - pdl) / day_range) * 100   # 0% = at PDL, 100% = at PDH

    if pct <= 35:
        zone = "DEEP DISCOUNT"
    elif pct <= 45:
        zone = "DISCOUNT"
    elif pct <= 55:
        zone = "EQUILIBRIUM"
    elif pct <= 65:
        zone = "PREMIUM"
    else:
        zone = "DEEP PREMIUM"

    return {"zone": zone, "pct": round(pct, 1), "mid": round(mid, 2)}


# ════════════════════════════════════════════════════════════════
# STEP 3 — H1 CONFIRMATION
# ════════════════════════════════════════════════════════════════

def find_fvg(df: pd.DataFrame, bias: str, lookback: int = 20) -> dict:
    """
    Fair Value Gap detection.
    Bullish FVG: candle[i-2].high < candle[i].low  (gap between them)
    Bearish FVG: candle[i-2].low  > candle[i].high (gap between them)
    Returns the most recent unfilled FVG aligned with bias.
    """
    recent = df.tail(lookback + 2)
    fvgs   = []

    for i in range(2, len(recent)):
        c1 = recent.iloc[i-2]
        c2 = recent.iloc[i-1]   # displacement candle
        c3 = recent.iloc[i]

        if bias == "BULLISH":
            # Bullish FVG: gap between c1 high and c3 low
            if c1["High"] < c3["Low"]:
                gap_high = float(c3["Low"])
                gap_low  = float(c1["High"])
                gap_mid  = (gap_high + gap_low) / 2
                # Check if current price is at/near the FVG
                curr = float(df["Close"].iloc[-1])
                if gap_low <= curr <= gap_high * 1.002:
                    fvgs.append({
                        "type":     "BULLISH",
                        "high":     round(gap_high, 4),
                        "low":      round(gap_low,  4),
                        "mid":      round(gap_mid,  4),
                        "filled":   curr > gap_high,
                        "at_level": abs(curr - gap_mid) / gap_mid < 0.005,
                    })

        elif bias == "BEARISH":
            # Bearish FVG: gap between c1 low and c3 high
            if c1["Low"] > c3["High"]:
                gap_low  = float(c3["High"])
                gap_high = float(c1["Low"])
                gap_mid  = (gap_high + gap_low) / 2
                curr     = float(df["Close"].iloc[-1])
                if gap_low * 0.998 <= curr <= gap_high:
                    fvgs.append({
                        "type":     "BEARISH",
                        "high":     round(gap_high, 4),
                        "low":      round(gap_low,  4),
                        "mid":      round(gap_mid,  4),
                        "filled":   curr < gap_low,
                        "at_level": abs(curr - gap_mid) / gap_mid < 0.005,
                    })

    return fvgs[-1] if fvgs else {}


def find_order_block(df: pd.DataFrame, bias: str, lookback: int = 30) -> dict:
    """
    Order Block detection.
    Bullish OB: last bearish candle (close < open) before a strong 3-bar bullish move
    Bearish OB: last bullish candle (close > open) before a strong 3-bar bearish move
    """
    recent = df.tail(lookback)
    obs    = []

    for i in range(1, len(recent) - 3):
        c     = recent.iloc[i]
        next3 = recent.iloc[i+1 : i+4]

        if bias == "BULLISH":
            # Last bearish candle before strong bullish move
            if c["Close"] < c["Open"]:
                move = (next3["Close"].max() - c["Low"]) / c["Low"]
                if move > 0.005:   # at least 0.5% move after
                    ob_high = float(c["Open"])   # top of bearish OB body
                    ob_low  = float(c["Low"])
                    curr    = float(df["Close"].iloc[-1])
                    if ob_low <= curr <= ob_high * 1.002:
                        obs.append({
                            "type":  "BULLISH",
                            "high":  round(ob_high, 4),
                            "low":   round(ob_low,  4),
                            "mid":   round((ob_high + ob_low) / 2, 4),
                            "strength": round(move * 100, 2),
                        })

        elif bias == "BEARISH":
            # Last bullish candle before strong bearish move
            if c["Close"] > c["Open"]:
                move = (c["High"] - next3["Close"].min()) / c["High"]
                if move > 0.005:
                    ob_high = float(c["High"])
                    ob_low  = float(c["Close"])  # bottom of bullish OB body
                    curr    = float(df["Close"].iloc[-1])
                    if ob_low * 0.998 <= curr <= ob_high:
                        obs.append({
                            "type":  "BEARISH",
                            "high":  round(ob_high, 4),
                            "low":   round(ob_low,  4),
                            "mid":   round((ob_high + ob_low) / 2, 4),
                            "strength": round(move * 100, 2),
                        })

    return obs[-1] if obs else {}


def detect_displacement(df: pd.DataFrame, bias: str, lookback: int = 5) -> dict:
    """
    Displacement / CISD detection on the last N candles.
    Strong displacement: large body candle (body > 60% of range),
    closes near high/low, above average size.
    """
    recent    = df.tail(lookback + 1)
    avg_range = float((df["High"] - df["Low"]).tail(20).mean())

    displacements = []
    for i in range(-lookback, 0):
        c     = recent.iloc[i]
        rng   = float(c["High"] - c["Low"])
        body  = abs(float(c["Close"]) - float(c["Open"]))
        if rng == 0:
            continue
        body_pct = body / rng

        is_bullish_disp = (
            float(c["Close"]) > float(c["Open"])   # bullish candle
            and body_pct > 0.6                      # strong body
            and rng > avg_range * 1.3               # larger than average
        )
        is_bearish_disp = (
            float(c["Close"]) < float(c["Open"])
            and body_pct > 0.6
            and rng > avg_range * 1.3
        )

        if bias == "BULLISH" and is_bullish_disp:
            strength = "STRONG" if body_pct > 0.75 and rng > avg_range * 1.8 else "MODERATE"
            displacements.append({"direction": "BULLISH", "strength": strength,
                                  "body_pct": round(body_pct, 2), "size_ratio": round(rng/avg_range, 2)})
        elif bias == "BEARISH" and is_bearish_disp:
            strength = "STRONG" if body_pct > 0.75 and rng > avg_range * 1.8 else "MODERATE"
            displacements.append({"direction": "BEARISH", "strength": strength,
                                  "body_pct": round(body_pct, 2), "size_ratio": round(rng/avg_range, 2)})

    return displacements[-1] if displacements else {}


def check_liquidity_swept(df: pd.DataFrame, bias: str, lookback: int = 10) -> bool:
    """Check if price recently swept a swing high/low (stop hunt)."""
    recent = df.tail(lookback)
    if bias == "BULLISH":
        # Look for wick below recent swing low that recovered
        swing_low = float(recent["Low"].iloc[:-2].min())
        last_low  = float(recent["Low"].iloc[-1])
        last_close= float(recent["Close"].iloc[-1])
        swept = last_low < swing_low and last_close > swing_low
        return swept
    elif bias == "BEARISH":
        swing_high = float(recent["High"].iloc[:-2].max())
        last_high  = float(recent["High"].iloc[-1])
        last_close = float(recent["Close"].iloc[-1])
        swept = last_high > swing_high and last_close < swing_high
        return swept
    return False


# ════════════════════════════════════════════════════════════════
# STEP 4 — KILL ZONE
# ════════════════════════════════════════════════════════════════

def check_kill_zone() -> dict:
    """Check if current time is inside a Kill Zone (EST)."""
    now_est  = datetime.now(EST)
    hour_est = now_est.hour

    for name, (start, end) in KILL_ZONES.items():
        if start <= hour_est < end:
            mins_left = (end - hour_est) * 60 - now_est.minute
            return {
                "active":    True,
                "name":      name,
                "time_est":  now_est.strftime("%H:%M EST"),
                "mins_left": mins_left,
                "priority":  "HIGH" if name in ["London Open", "New York AM"] else "MEDIUM",
            }

    # Calculate time to next Kill Zone
    for name, (start, end) in KILL_ZONES.items():
        if hour_est < start:
            mins_to = (start - hour_est) * 60 - now_est.minute
            return {
                "active":       False,
                "next_session": name,
                "mins_to_next": mins_to,
                "time_est":     now_est.strftime("%H:%M EST"),
                "priority":     "NONE",
            }

    return {"active": False, "next_session": "London Open (tomorrow)",
            "time_est": now_est.strftime("%H:%M EST"), "priority": "NONE"}


# ════════════════════════════════════════════════════════════════
# AI NOTES GENERATOR
# ════════════════════════════════════════════════════════════════

ICT_NOTES_PROMPT = """You are an ICT (Inner Circle Trader) analyst.
Based on the setup data below, give a brief trader's briefing.
Be concise and actionable. Max 3 sentences.

SETUP DATA:
{data}

Focus on:
1. What the trader should be watching for on M5
2. Key price levels to monitor
3. One risk warning if applicable

Respond in plain text, no JSON, no bullet points."""


def get_ai_notes(setup: ICTSetup) -> str:
    """Get a brief AI commentary on the setup."""
    try:
        data = {
            "symbol":       setup.symbol,
            "d1_bias":      setup.d1_bias,
            "price":        setup.current_price,
            "pdh":          setup.prev_day_high,
            "pdl":          setup.prev_day_low,
            "zone":         setup.price_zone,
            "fvg":          f"{setup.h1_fvg_type} FVG at {setup.h1_fvg_level}" if setup.h1_fvg_found else "None",
            "ob":           f"{setup.h1_ob_type} OB at {setup.h1_ob_level}" if setup.h1_ob_found else "None",
            "displacement": setup.h1_disp_strength,
            "kill_zone":    setup.kill_zone_name or "Not active",
            "grade":        setup.setup_grade,
        }
        notes = model.chat(
            system_prompt="You are a concise ICT trading analyst.",
            user_prompt=ICT_NOTES_PROMPT.format(data=json.dumps(data, indent=2)),
        )
        return notes.strip()
    except Exception:
        return ""


# ════════════════════════════════════════════════════════════════
# MAIN SCANNER
# ════════════════════════════════════════════════════════════════

def score_setup(setup: ICTSetup) -> tuple[int, str]:
    """
    Score the setup 0-4 based on how many steps are confirmed.
    Returns (score, grade).
    """
    score = 0

    # Step 1 — D1 bias clear
    if setup.d1_bias in ("BULLISH", "BEARISH"):
        score += 1

    # Step 2 — In correct zone (discount for buy, premium for sell)
    if setup.d1_bias == "BULLISH" and "DISCOUNT" in setup.price_zone:
        score += 1
    elif setup.d1_bias == "BEARISH" and "PREMIUM" in setup.price_zone:
        score += 1

    # Step 3 — H1 confirmation (FVG or OB + displacement)
    has_pd_array  = setup.h1_fvg_found or setup.h1_ob_found
    has_disp      = setup.h1_displacement
    has_liq_sweep = setup.h1_swept_liq
    if has_pd_array and (has_disp or has_liq_sweep):
        score += 1
    elif has_pd_array:
        score += 0.5   # partial — PD array but no displacement yet

    # Step 4 — Kill Zone
    if setup.in_kill_zone and setup.kill_zone_name in ("London Open", "New York AM"):
        score += 1

    score = int(score)

    if score == 4:
        grade = "A+"
    elif score == 3:
        grade = "A"
    elif score == 2:
        grade = "B"
    else:
        grade = "NO SETUP"

    return score, grade


def build_action(setup: ICTSetup) -> str:
    """Generate the trader's action instruction for M5."""
    if setup.setup_grade == "A+":
        direction = "LONG" if setup.d1_bias == "BULLISH" else "SHORT"
        pd_level  = setup.h1_fvg_level if setup.h1_fvg_found else setup.h1_ob_level
        return (f"⚡ GO TO M5 NOW — Look for {direction} CISD. "
                f"Entry near ${pd_level:,.2f}. "
                f"SL below M5 structure. Target PDH ${setup.prev_day_high:,.2f}" 
                if setup.d1_bias == "BULLISH" 
                else f"⚡ GO TO M5 NOW — Look for {direction} CISD. "
                f"Entry near ${pd_level:,.2f}. "
                f"SL above M5 structure. Target PDL ${setup.prev_day_low:,.2f}")
    elif setup.setup_grade == "A":
        return "👁️ MONITOR — Setup developing but not complete. Wait for Kill Zone or displacement."
    elif setup.setup_grade == "B":
        return "📋 WATCH — Early stage setup. 1-2 conditions missing. Do not enter yet."
    else:
        return "⏸️ STAND DOWN — No valid ICT setup present. Wait for better conditions."


def scan_symbol(symbol: str) -> ICTSetup:
    """Run full 4-step ICT scan for one symbol."""
    setup           = ICTSetup(symbol=symbol, timestamp=datetime.now().isoformat())

    # ── Step 1: D1 Direction ────────────────────────────────
    d1 = analyse_d1_direction(symbol)
    setup.d1_bias       = d1.get("bias", "NEUTRAL")
    setup.d1_structure  = d1.get("structure", "")
    setup.prev_day_high = d1.get("pdh", 0)
    setup.prev_day_low  = d1.get("pdl", 0)
    setup.current_price = d1.get("price", 0)

    # ── Step 2: Premium / Discount ──────────────────────────
    pd_zone = analyse_premium_discount(
        setup.current_price, setup.prev_day_high, setup.prev_day_low
    )
    setup.price_zone      = pd_zone.get("zone", "UNKNOWN")
    setup.zone_pct        = pd_zone.get("pct", 50.0)
    setup.daily_range_mid = pd_zone.get("mid", 0)

    # ── Step 3: H1 Confirmation ─────────────────────────────
    if setup.d1_bias != "NEUTRAL":
        try:
            h1_df = get_ohlcv(symbol, exchange=EXCHANGE, timeframe="1H", days_back=10)

            # FVG
            fvg = find_fvg(h1_df, setup.d1_bias)
            if fvg:
                setup.h1_fvg_found = True
                setup.h1_fvg_level = fvg.get("mid", 0)
                setup.h1_fvg_type  = fvg.get("type", "")

            # Order Block
            ob = find_order_block(h1_df, setup.d1_bias)
            if ob:
                setup.h1_ob_found = True
                setup.h1_ob_level = ob.get("mid", 0)
                setup.h1_ob_type  = ob.get("type", "")

            # Displacement
            disp = detect_displacement(h1_df, setup.d1_bias)
            if disp:
                setup.h1_displacement  = True
                setup.h1_disp_strength = disp.get("strength", "MODERATE")

            # Liquidity sweep
            setup.h1_swept_liq = check_liquidity_swept(h1_df, setup.d1_bias)

        except Exception as e:
            setup.d1_structure += f" | H1 error: {e}"

    # ── Step 4: Kill Zone ────────────────────────────────────
    kz = check_kill_zone()
    setup.in_kill_zone   = kz.get("active", False)
    setup.kill_zone_name = kz.get("name", "")
    setup.current_session= kz.get("time_est", "")

    # ── Score & Grade ────────────────────────────────────────
    setup.setup_score, setup.setup_grade = score_setup(setup)
    setup.alert   = setup.setup_grade in ("A+", "A")
    setup.action  = build_action(setup)

    # ── AI Notes (only for A/A+ setups to save API cost) ────
    if setup.setup_grade in ("A+", "A"):
        setup.ai_notes = get_ai_notes(setup)

    return setup


# ── printing ──────────────────────────────────────────────────
GRADE_ICONS = {"A+": "🚨", "A": "⚡", "B": "👁️", "NO SETUP": "⏸️"}
BIAS_ICONS  = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}

def print_setup(setup: ICTSetup):
    icon  = GRADE_ICONS.get(setup.setup_grade, "⏸️")
    bicon = BIAS_ICONS.get(setup.d1_bias, "⚪")

    print(f"\n  {'─'*58}")
    print(f"  {icon} {setup.symbol}  Grade: {setup.setup_grade}  Score: {setup.setup_score}/4")
    print(f"  {'─'*58}")
    print(f"  {bicon} D1 Bias    : {setup.d1_bias}")
    print(f"     Structure : {setup.d1_structure}")
    print(f"     PDH: ${setup.prev_day_high:>12,.2f}   PDL: ${setup.prev_day_low:>12,.2f}")
    print(f"     Price    : ${setup.current_price:>12,.2f}")

    zone_icon = "🟢" if "DISCOUNT" in setup.price_zone else "🔴" if "PREMIUM" in setup.price_zone else "⚪"
    print(f"\n  {zone_icon} Zone       : {setup.price_zone} ({setup.zone_pct:.1f}% of daily range)")
    print(f"     Range mid : ${setup.daily_range_mid:,.2f}")

    fvg_icon = "✅" if setup.h1_fvg_found else "❌"
    ob_icon  = "✅" if setup.h1_ob_found  else "❌"
    dp_icon  = "✅" if setup.h1_displacement else "❌"
    lq_icon  = "✅" if setup.h1_swept_liq   else "❌"
    print(f"\n  📊 H1 Checks:")
    print(f"     {fvg_icon} FVG        : {'${:,.2f} ({})'.format(setup.h1_fvg_level, setup.h1_fvg_type) if setup.h1_fvg_found else 'Not found'}")
    print(f"     {ob_icon} Order Block: {'${:,.2f} ({})'.format(setup.h1_ob_level, setup.h1_ob_type) if setup.h1_ob_found else 'Not found'}")
    print(f"     {dp_icon} Displacement: {setup.h1_disp_strength or 'None'}")
    print(f"     {lq_icon} Liq Swept  : {'Yes' if setup.h1_swept_liq else 'No'}")

    kz_icon = "✅" if setup.in_kill_zone else "❌"
    print(f"\n  ⏰ {kz_icon} Kill Zone  : {setup.kill_zone_name if setup.in_kill_zone else 'Not active'} ({setup.current_session})")

    print(f"\n  📋 ACTION: {setup.action}")
    if setup.ai_notes:
        print(f"\n  🤖 AI Notes: {setup.ai_notes}")


# ── CSV logging ───────────────────────────────────────────────
def log_setup(setup: ICTSetup):
    row = {
        "timestamp":       setup.timestamp,
        "symbol":          setup.symbol,
        "grade":           setup.setup_grade,
        "score":           setup.setup_score,
        "d1_bias":         setup.d1_bias,
        "price":           setup.current_price,
        "pdh":             setup.prev_day_high,
        "pdl":             setup.prev_day_low,
        "price_zone":      setup.price_zone,
        "zone_pct":        setup.zone_pct,
        "fvg_found":       setup.h1_fvg_found,
        "fvg_level":       setup.h1_fvg_level,
        "ob_found":        setup.h1_ob_found,
        "ob_level":        setup.h1_ob_level,
        "displacement":    setup.h1_disp_strength,
        "liq_swept":       setup.h1_swept_liq,
        "kill_zone":       setup.kill_zone_name,
        "in_kill_zone":    setup.in_kill_zone,
        "alert":           setup.alert,
        "action":          setup.action[:120],
        "ai_notes":        setup.ai_notes[:200] if setup.ai_notes else "",
    }
    write_header = not SCANNER_LOG.exists()
    with open(SCANNER_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


# ════════════════════════════════════════════════════════════════
# MAIN CLASS
# ════════════════════════════════════════════════════════════════

class ICTScanner:

    def __init__(self):
        print("🎯 ICT Setup Scanner initialised")
        print(f"   Symbols    : {SYMBOLS}")
        print(f"   Kill Zones : London Open (2-5am EST) | New York AM (7-10am EST)")
        print(f"   Scan every : {SCAN_INTERVAL//60} minutes")
        print(f"   Grades     : A+ (all 4 steps) | A (3 steps) | B (2 steps)\n")

    def scan_all(self) -> list[ICTSetup]:
        kz     = check_kill_zone()
        kz_str = f"🔥 {kz['name']}" if kz.get("active") else f"Next: {kz.get('next_session','—')} in {kz.get('mins_to_next','?')}m"

        print(f"\n{'═'*60}")
        print(f"🎯 ICT Scan — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Session: {kz.get('time_est','—')}  |  {kz_str}")
        print(f"{'═'*60}")

        setups  = []
        alerts  = []

        for symbol in SYMBOLS:
            print(f"\n── Scanning {symbol} ──")
            try:
                setup = scan_symbol(symbol)
                print_setup(setup)
                log_setup(setup)
                setups.append(setup)
                if setup.alert:
                    alerts.append(setup)
            except Exception as e:
                print(f"  ❌ Scan failed for {symbol}: {e}")

        # Summary
        print(f"\n{'═'*60}")
        if alerts:
            print(f"🚨 {len(alerts)} ALERT(S) FIRED:")
            for s in alerts:
                print(f"   {GRADE_ICONS.get(s.setup_grade,'⏸️')} {s.symbol} — Grade {s.setup_grade} — {s.d1_bias}")
        else:
            print("✅ No A/A+ setups right now. Monitoring...")

        return setups

    def run(self):
        print("🚀 ICT Scanner running. Press Ctrl+C to stop.\n")
        try:
            while True:
                self.scan_all()
                print(f"\n😴 Next scan in {SCAN_INTERVAL//60} minutes...")
                time.sleep(SCAN_INTERVAL)
        except KeyboardInterrupt:
            print("\n🛑 ICT Scanner stopped.")


# ── entrypoint ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="🎯 ICT Setup Scanner")
    parser.add_argument("--once",   action="store_true", help="Single scan then exit")
    parser.add_argument("--symbol", type=str, default=None, help="Scan single symbol e.g. ETH")
    args = parser.parse_args()

    scanner = ICTScanner()

    if args.symbol:
        sym   = args.symbol.upper()
        setup = scan_symbol(sym)
        print_setup(setup)
        log_setup(setup)
    elif args.once:
        scanner.scan_all()
    else:
        scanner.run()
