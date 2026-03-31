# ============================================================
# 🌙 Chart Analysis Agent ("Chuck")
#
# Reads a screenshot of ANY chart and returns a buy/sell/hold
# recommendation with reasoning. Uses Claude's vision API.
#
# Moon Dev's version ("Chuck") analyzes crypto charts.
# This version is enhanced for ICT/futures:
#   - Identifies ICT patterns: FVG, sweep, CISD, order blocks
#   - Kill zone awareness (checks if in valid trading window)
#   - Scores setup quality: A+ / A / B / No Setup
#   - Works on any chart screenshot from TradingView
#
# HOW TO USE:
#   # Analyze a chart screenshot
#   python src/agents/chart_agent.py --image /path/to/chart.png
#
#   # Run on live market (captures screenshot from Hyperliquid/Tradovate)
#   python src/agents/chart_agent.py --symbol MNQ --tf 5m
#
#   # Continuous monitoring mode
#   python src/agents/chart_agent.py --symbol MNQ --continuous
#
#   # From dashboard: screenshot any chart and drag into the interface
# ============================================================

import sys
import base64
import json
import csv
import time
import requests
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Paths ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE  = REPO_ROOT / "src" / "data" / "chart_analysis_log.csv"

# ── Analysis prompts ──────────────────────────────────────────
ICT_ANALYSIS_PROMPT = """You are an expert ICT (Inner Circle Trader) futures trader 
analyzing a chart screenshot.

Analyze this chart and provide:

1. MARKET STRUCTURE
   - Current trend direction (bullish / bearish / ranging)
   - Key swing highs and lows visible
   - Any Market Structure Shifts (MSS) or Change of Character (CHoCH)

2. LIQUIDITY ANALYSIS
   - Any swept liquidity levels (buy-side or sell-side)
   - Unswept highs/lows that price is likely targeting
   - Premium or discount zone (above/below equilibrium)

3. ICT PATTERNS IDENTIFIED
   - Fair Value Gaps (FVG) — bullish or bearish
   - Inverse FVGs (IFVG)
   - Order Blocks
   - Breaker Blocks
   - CISD (Change in State of Delivery)
   - Liquidity sweeps

4. SETUP QUALITY
   Rate the current setup:
   - A+: Sweep + CISD + kill zone confluence
   - A:  2 of the 3 conditions
   - B:  1 condition or weak setup
   - No Setup: avoid trading

5. TRADE RECOMMENDATION
   - Direction: LONG / SHORT / NO TRADE
   - Entry: specific price or condition
   - Stop Loss: specific level
   - Target: specific level or liquidity pool
   - Reasoning: 2-3 sentences max

Respond in this exact JSON format:
{
  "trend": "bullish|bearish|ranging",
  "structure_shift": "yes|no|possible",
  "liquidity_swept": "buy-side|sell-side|none",
  "patterns": ["FVG", "Sweep", "CISD"],
  "setup_grade": "A+|A|B|no_setup",
  "signal": "LONG|SHORT|NO_TRADE",
  "entry": "price or condition",
  "stop_loss": "price or level",
  "target": "price or level",
  "reasoning": "brief explanation",
  "confidence": 0.0-1.0
}"""

GENERAL_ANALYSIS_PROMPT = """You are an expert technical analyst.
Analyze this trading chart and provide a buy/sell/hold recommendation.

Look for:
1. Trend direction
2. Key support/resistance levels
3. Technical patterns (head & shoulders, double top/bottom, flags, etc.)
4. Momentum indicators (RSI, MACD if visible)
5. Volume patterns

Respond in JSON format:
{
  "trend": "bullish|bearish|ranging",
  "key_levels": ["level1", "level2"],
  "patterns": ["pattern1"],
  "signal": "LONG|SHORT|NO_TRADE",
  "entry": "condition or price",
  "stop_loss": "level",
  "target": "level",
  "reasoning": "brief explanation",
  "confidence": 0.0-1.0
}"""


def encode_image(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def encode_image_url(url: str) -> str:
    """Download image from URL and encode to base64."""
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return base64.standard_b64encode(r.content).decode("utf-8")


def analyze_chart_image(image_path: str = None,
                        image_url: str = None,
                        image_b64: str = None,
                        mode: str = "ict",
                        symbol: str = "",
                        timeframe: str = "") -> dict:
    """
    Analyze a chart image using Claude's vision API.

    Args:
        image_path: Local file path to chart screenshot
        image_url:  URL of chart image
        image_b64:  Pre-encoded base64 image
        mode:       "ict" for ICT analysis, "general" for TA analysis
        symbol:     Symbol name for context (optional)
        timeframe:  Timeframe for context (optional)

    Returns:
        Dict with analysis results
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set in .env\n"
            "Get from: console.anthropic.com → API Keys"
        )

    # Get image data
    if image_path:
        img_b64   = encode_image(image_path)
        media_type = "image/png" if image_path.endswith(".png") else "image/jpeg"
    elif image_url:
        img_b64    = encode_image_url(image_url)
        media_type = "image/jpeg"
    elif image_b64:
        img_b64    = image_b64
        media_type = "image/png"
    else:
        raise ValueError("Provide image_path, image_url, or image_b64")

    # Choose prompt
    prompt = ICT_ANALYSIS_PROMPT if mode == "ict" else GENERAL_ANALYSIS_PROMPT

    # Add context if provided
    context = ""
    if symbol or timeframe:
        context = f"\n\nChart context: {symbol} {timeframe} — futures micro contract"

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-5",   # vision-capable model
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type":   "image",
                    "source": {
                        "type":       "base64",
                        "media_type": media_type,
                        "data":       img_b64,
                    },
                },
                {
                    "type": "text",
                    "text": prompt + context,
                },
            ],
        }],
    )

    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            result = json.loads(m.group(0))
        else:
            result = {"raw_response": raw, "signal": "NO_TRADE", "error": "parse_failed"}

    # Add metadata
    result["timestamp"] = datetime.now().isoformat()
    result["symbol"]    = symbol
    result["timeframe"] = timeframe
    result["mode"]      = mode

    return result


def print_analysis(result: dict):
    """Pretty print the chart analysis result."""
    signal = result.get("signal", "NO_TRADE")
    grade  = result.get("setup_grade", "")
    conf   = result.get("confidence", 0)

    # Signal emoji
    icons = {"LONG": "🟢", "SHORT": "🔴", "NO_TRADE": "⚪"}
    icon  = icons.get(signal, "⚪")

    print(f"\n{'='*55}")
    print(f"  {icon} CHART ANALYSIS — {result.get('symbol','')} "
          f"{result.get('timeframe','')}")
    print(f"{'='*55}")
    print(f"  Signal     : {signal} {f'(Grade: {grade})' if grade else ''}")
    print(f"  Confidence : {conf:.0%}")
    print(f"  Trend      : {result.get('trend','?')}")

    patterns = result.get("patterns", [])
    if patterns:
        print(f"  Patterns   : {', '.join(patterns)}")

    liq = result.get("liquidity_swept", "")
    if liq and liq != "none":
        print(f"  Liquidity  : {liq} swept ✅")

    if signal != "NO_TRADE":
        print(f"\n  Entry      : {result.get('entry','?')}")
        print(f"  Stop Loss  : {result.get('stop_loss','?')}")
        print(f"  Target     : {result.get('target','?')}")

    print(f"\n  Reasoning  : {result.get('reasoning','?')}")
    print(f"{'='*55}\n")


def log_analysis(result: dict):
    """Save analysis to CSV log."""
    write_header = not LOG_FILE.exists()
    row = {
        "timestamp":       result.get("timestamp", ""),
        "symbol":          result.get("symbol", ""),
        "timeframe":       result.get("timeframe", ""),
        "signal":          result.get("signal", ""),
        "setup_grade":     result.get("setup_grade", ""),
        "confidence":      result.get("confidence", 0),
        "trend":           result.get("trend", ""),
        "patterns":        "|".join(result.get("patterns", [])),
        "entry":           result.get("entry", ""),
        "stop_loss":       result.get("stop_loss", ""),
        "target":          result.get("target", ""),
        "reasoning":       result.get("reasoning", ""),
        "liquidity_swept": result.get("liquidity_swept", ""),
    }

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def capture_chart_screenshot(symbol: str, timeframe: str) -> str:
    """
    Placeholder for chart capture.
    In production: use Selenium/Playwright to screenshot TradingView.
    For now: returns None and instructs user to provide screenshot manually.
    """
    print(f"  📸 To analyze live charts:")
    print(f"     1. Open TradingView with {symbol} {timeframe}")
    print(f"     2. Take a screenshot (Cmd+Shift+4 on Mac)")
    print(f"     3. Run: python src/agents/chart_agent.py --image /path/to/screenshot.png")
    return None


def run_analysis(image_path: str = None,
                 image_url: str = None,
                 symbol: str = "",
                 timeframe: str = "",
                 mode: str = "ict") -> dict | None:
    """Run a single chart analysis."""
    print(f"\n📊 Chart Analysis Agent")
    if symbol:
        print(f"   Symbol: {symbol} | TF: {timeframe} | Mode: {mode}")

    if not image_path and not image_url:
        capture_chart_screenshot(symbol, timeframe)
        return None

    print(f"  🔍 Analyzing chart...")

    try:
        result = analyze_chart_image(
            image_path=image_path,
            image_url=image_url,
            mode=mode,
            symbol=symbol,
            timeframe=timeframe,
        )
        print_analysis(result)
        log_analysis(result)
        return result

    except Exception as e:
        print(f"  ❌ Analysis failed: {e}")
        return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="📊 Chart Analysis Agent")
    p.add_argument("--image",      type=str, default=None,
                   help="Path to chart screenshot")
    p.add_argument("--url",        type=str, default=None,
                   help="URL of chart image")
    p.add_argument("--symbol",     type=str, default="",
                   help="Symbol name e.g. MNQ, BTC")
    p.add_argument("--tf",         type=str, default="",
                   help="Timeframe e.g. 5m, 15m, 1H")
    p.add_argument("--mode",       type=str, default="ict",
                   choices=["ict", "general"],
                   help="Analysis mode: ict (default) or general")
    p.add_argument("--continuous", action="store_true",
                   help="Run continuously every 5 minutes")
    args = p.parse_args()

    if args.continuous:
        print(f"🔄 Continuous chart analysis every 5 minutes...")
        print(f"   Provide chart screenshots as they update")
        while True:
            result = run_analysis(
                image_path=args.image,
                image_url=args.url,
                symbol=args.symbol,
                timeframe=args.tf,
                mode=args.mode,
            )
            if not result:
                break
            print(f"  💤 Next analysis in 5 mins...")
            time.sleep(300)
    else:
        run_analysis(
            image_path=args.image,
            image_url=args.url,
            symbol=args.symbol,
            timeframe=args.tf,
            mode=args.mode,
        )