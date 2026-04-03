#!/usr/bin/env python3
# ============================================================
# 🌙 Weekly Briefing Agent
#
# Runs every Monday before market open.
# Searches the web for:
#   1. Top macro events this week (feeds news filter)
#   2. Current macro regime (risk-on/risk-off)
#   3. Unusual asset correlations (regime warning)
#   4. Crypto sentiment vs fundamentals divergence
#   5. Key levels for your vault symbols this week
#
# Output:
#   - Saves to src/data/weekly_brief.json
#   - Dashboard reads this for the Regime panel
#   - Forward tester reads macro regime to adjust sizing
#   - News filter reads events for kill-zone logic
#
# HOW TO RUN:
#   python src/agents/weekly_briefing.py           # run now
#   python src/agents/weekly_briefing.py --schedule # auto every Monday 6am
#
# Works for BOTH prop firm (Apex futures) and crypto (Hyperliquid).
# ============================================================

import os, sys, json, time
from pathlib import Path
from datetime import datetime, timezone, date

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "src" / "data"
BRIEF_FILE = DATA_DIR / "weekly_brief.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Search helper (reuse from websearch_agent) ────────────────
def search(query: str, num: int = 5) -> list[str]:
    """Search web, return list of result snippets."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if api_key:
        try:
            import requests
            r = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query,
                      "max_results": num, "search_depth": "basic"},
                timeout=15
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            return [f"{x.get('title','')}: {x.get('content','')[:300]}"
                    for x in results]
        except Exception as e:
            print(f"  Tavily error: {e}")

    # Fallback: Serper
    api_key = os.getenv("SERPER_API_KEY", "")
    if api_key:
        try:
            import requests
            r = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": num},
                timeout=15
            )
            r.raise_for_status()
            items = r.json().get("organic", [])
            return [f"{x.get('title','')}: {x.get('snippet','')[:300]}"
                    for x in items]
        except Exception as e:
            print(f"  Serper error: {e}")

    return []


def ask_llm(prompt: str, max_tokens: int = 800) -> str:
    """Call LLM router for analysis."""
    try:
        from src.models.llm_router import model
        return model(prompt, max_tokens=max_tokens)
    except Exception as e:
        return f"LLM error: {e}"


# ══════════════════════════════════════════════════════════════
# SECTION 1 — HIGH IMPACT NEWS EVENTS THIS WEEK
# ══════════════════════════════════════════════════════════════

def get_weekly_events() -> dict:
    """
    Pull this week's high-impact macro events.
    Used by forward tester's news filter.
    """
    print("  📅 Fetching this week's macro events...")
    today = date.today()
    week  = today.strftime("%B %d %Y")

    results = search(f"high impact economic events calendar week {week} Fed FOMC CPI NFP", 6)
    if not results:
        return {"events": [], "summary": "Could not fetch events"}

    prompt = f"""Today is {week}. Based on these search results, list the 5 most important 
high-impact economic events happening THIS WEEK that could move futures markets (ES, NQ, MES, MNQ).

Search results:
{chr(10).join(results[:5])}

For each event return ONLY this JSON format (no other text):
{{
  "events": [
    {{"day": "Monday", "time_est": "8:30 AM", "event": "CPI Report", "impact": "HIGH", "affects": ["MES","MNQ","crypto"]}},
    ...
  ],
  "highest_risk_day": "Wednesday",
  "summary": "One sentence summary of week's macro risk"
}}"""

    raw = ask_llm(prompt, 600)
    try:
        # Extract JSON from response
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        pass
    return {"events": [], "summary": raw[:200]}


# ══════════════════════════════════════════════════════════════
# SECTION 2 — MACRO REGIME (risk-on / risk-off)
# ══════════════════════════════════════════════════════════════

def get_macro_regime() -> dict:
    """
    Current macro environment — risk-on or risk-off.
    Affects position sizing: reduce size in risk-off, full size in risk-on.
    """
    print("  🌍 Analysing macro regime...")

    results = search("current macroeconomic environment inflation Fed rates recession 2026", 5)
    results += search("S&P 500 market trend risk on risk off current", 3)

    prompt = f"""Based on current macro data, assess whether we're in a RISK-ON or RISK-OFF environment.

Search results:
{chr(10).join(results[:6])}

Respond ONLY in this JSON format:
{{
  "regime": "RISK-ON",
  "confidence": "HIGH",
  "reasoning": "2-3 sentence explanation",
  "signals": {{
    "rates": "rising/falling/stable",
    "inflation": "high/moderate/low",
    "growth": "expanding/contracting/stable",
    "sentiment": "bullish/bearish/neutral"
  }},
  "impact_on_trading": "How this affects futures and crypto trading this week",
  "size_adjustment": 1.0
}}

size_adjustment: 1.0 = normal, 0.75 = reduce 25%, 0.5 = reduce 50%, 1.25 = increase 25%"""

    raw = ask_llm(prompt, 500)
    try:
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        pass
    return {"regime": "UNKNOWN", "confidence": "LOW", "reasoning": raw[:200],
            "size_adjustment": 1.0}


# ══════════════════════════════════════════════════════════════
# SECTION 3 — CORRELATION MAP (unusual relationships)
# ══════════════════════════════════════════════════════════════

def get_correlation_signals() -> dict:
    """
    Detect unusual asset correlations that historically signal regime changes.
    Gold + stocks rising together, bonds + equities falling simultaneously, etc.
    """
    print("  🔗 Checking asset correlations...")

    results = search("gold stocks bonds correlations unusual current market 2026", 4)
    results += search("BTC crypto correlation S&P 500 current", 3)

    prompt = f"""Analyse current asset correlations based on these search results.
Identify any UNUSUAL correlations that historically signal regime changes or volatility.

Search results:
{chr(10).join(results[:5])}

Respond ONLY in this JSON format:
{{
  "unusual_correlations": [
    {{
      "assets": "Gold + Stocks",
      "observation": "Both rising simultaneously",
      "historical_signal": "Historically precedes...",
      "trading_implication": "For MES/crypto this means..."
    }}
  ],
  "crypto_correlation": {{
    "btc_spy_correlation": "high/low/negative",
    "implication": "What this means for crypto trades"
  }},
  "warning_level": "LOW",
  "summary": "One sentence overall correlation regime summary"
}}

warning_level: LOW, MEDIUM, HIGH, CRITICAL"""

    raw = ask_llm(prompt, 600)
    try:
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        pass
    return {"unusual_correlations": [], "warning_level": "LOW", "summary": raw[:200]}


# ══════════════════════════════════════════════════════════════
# SECTION 4 — CRYPTO SENTIMENT vs FUNDAMENTALS
# ══════════════════════════════════════════════════════════════

def get_crypto_divergence() -> dict:
    """
    Find vault crypto tokens where sentiment diverges from fundamentals.
    Negative sentiment + strong fundamentals = higher confidence on mean reversion entries.
    """
    print("  🪙 Checking crypto sentiment divergence...")

    # Your vaulted crypto tokens
    tokens = ["TAO", "PEPE", "WIF", "SOL", "BNB", "ETH", "AVAX", "FET", "RNDR", "BTC"]
    token_str = " ".join(tokens)

    results = search(f"crypto sentiment {token_str} market outlook this week 2026", 5)
    results += search("AI crypto tokens TAO FET RNDR fundamentals on-chain 2026", 3)

    prompt = f"""Analyse sentiment vs fundamentals for these crypto tokens: {token_str}

These are tokens in an algorithmic trading system. We want to know:
1. Which tokens have NEGATIVE sentiment but STRONG fundamentals (best for mean reversion longs)
2. Which tokens have POSITIVE sentiment but WEAKENING fundamentals (caution on longs)
3. Overall crypto market risk this week

Search results:
{chr(10).join(results[:6])}

Respond ONLY in this JSON format:
{{
  "high_conviction_longs": [
    {{"token": "TAO", "reason": "Why sentiment negative but fundamentals strong", "confidence": "HIGH"}}
  ],
  "caution_tokens": [
    {{"token": "WIF", "reason": "Why to be cautious", "risk": "HIGH"}}
  ],
  "market_regime": "BULLISH",
  "btc_outlook": "Short summary of BTC outlook this week",
  "summary": "One sentence crypto market summary"
}}"""

    raw = ask_llm(prompt, 600)
    try:
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        pass
    return {"high_conviction_longs": [], "caution_tokens": [],
            "market_regime": "UNKNOWN", "summary": raw[:200]}


# ══════════════════════════════════════════════════════════════
# SECTION 5 — KEY LEVELS FOR VAULT SYMBOLS
# ══════════════════════════════════════════════════════════════

def get_key_levels() -> dict:
    """
    Key support/resistance levels for vault futures symbols this week.
    Used as context for manual trade review.
    """
    print("  📊 Fetching key levels...")

    results = search("MES ES S&P 500 futures key levels support resistance this week", 4)
    results += search("NQ Nasdaq futures key levels support resistance current", 3)

    prompt = f"""Based on these search results, identify the key price levels for MES (Micro S&P 500) 
and MNQ (Micro Nasdaq) futures this week.

Search results:
{chr(10).join(results[:5])}

Respond ONLY in this JSON format:
{{
  "MES": {{
    "key_resistance": [5300, 5350],
    "key_support": [5200, 5150],
    "weekly_range": "5150-5350",
    "bias": "BULLISH",
    "notes": "Brief context"
  }},
  "MNQ": {{
    "key_resistance": [19000, 19500],
    "key_support": [18500, 18000],
    "weekly_range": "18000-19500",
    "bias": "NEUTRAL",
    "notes": "Brief context"
  }}
}}"""

    raw = ask_llm(prompt, 400)
    try:
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════════════════
# MASTER BRIEFING FUNCTION
# ══════════════════════════════════════════════════════════════

def run_weekly_briefing(force: bool = False) -> dict:
    """
    Run the full weekly briefing. Saves to weekly_brief.json.
    Skipped if already ran today (unless force=True).
    """
    today = str(date.today())

    # Check if already ran today
    if not force and BRIEF_FILE.exists():
        try:
            existing = json.loads(BRIEF_FILE.read_text())
            if existing.get("date") == today:
                print(f"  ✅ Weekly brief already generated today ({today})")
                return existing
        except Exception:
            pass

    print(f"\n🌙 Weekly Briefing Agent — {today}")
    print("=" * 50)

    brief = {
        "date":      today,
        "generated": datetime.now(timezone.utc).isoformat(),
        "week_of":   date.today().strftime("%B %d, %Y"),
    }

    # Run all 5 sections
    try:
        brief["events"]       = get_weekly_events()
    except Exception as e:
        brief["events"]       = {"events": [], "summary": str(e)}

    try:
        brief["macro_regime"] = get_macro_regime()
    except Exception as e:
        brief["macro_regime"] = {"regime": "UNKNOWN", "size_adjustment": 1.0}

    try:
        brief["correlations"] = get_correlation_signals()
    except Exception as e:
        brief["correlations"] = {"warning_level": "LOW", "unusual_correlations": []}

    try:
        brief["crypto"]       = get_crypto_divergence()
    except Exception as e:
        brief["crypto"]       = {"high_conviction_longs": [], "market_regime": "UNKNOWN"}

    try:
        brief["key_levels"]   = get_key_levels()
    except Exception as e:
        brief["key_levels"]   = {}

    # Save
    BRIEF_FILE.write_text(json.dumps(brief, indent=2))
    print(f"\n  ✅ Brief saved to {BRIEF_FILE}")

    # Print summary
    _print_summary(brief)

    return brief


def _print_summary(brief: dict):
    """Print a human-readable summary to terminal."""
    print("\n" + "=" * 50)
    print(f"  📋 WEEKLY BRIEF — {brief.get('week_of','')}")
    print("=" * 50)

    # Macro regime
    regime = brief.get("macro_regime", {})
    r_col  = "🟢" if regime.get("regime") == "RISK-ON" else "🔴" if regime.get("regime") == "RISK-OFF" else "🟡"
    print(f"\n  {r_col} Macro Regime: {regime.get('regime','?')} "
          f"({regime.get('confidence','?')} confidence)")
    print(f"     {regime.get('reasoning','')[:120]}")
    size_adj = regime.get("size_adjustment", 1.0)
    if size_adj != 1.0:
        print(f"  ⚖️  Size adjustment: {size_adj}x (auto-applied to all trades)")

    # Key events
    events = brief.get("events", {})
    evt_list = events.get("events", [])
    if evt_list:
        print(f"\n  📅 Key Events This Week:")
        for e in evt_list[:5]:
            impact = "🔴" if e.get("impact") == "HIGH" else "🟡"
            print(f"     {impact} {e.get('day','')} {e.get('time_est','')} — "
                  f"{e.get('event','')} "
                  f"[affects: {', '.join(e.get('affects',[]))}]")
        if events.get("highest_risk_day"):
            print(f"  ⚠️  Highest risk day: {events['highest_risk_day']}")

    # Correlation warnings
    corr = brief.get("correlations", {})
    warn = corr.get("warning_level", "LOW")
    if warn in ("HIGH", "CRITICAL"):
        print(f"\n  ⚠️  Correlation Warning: {warn}")
        for c in corr.get("unusual_correlations", []):
            print(f"     • {c.get('assets')}: {c.get('observation')}")
            print(f"       → {c.get('trading_implication','')[:100]}")

    # Crypto
    crypto = brief.get("crypto", {})
    longs  = crypto.get("high_conviction_longs", [])
    if longs:
        print(f"\n  🪙 High Conviction Crypto Longs:")
        for t in longs[:3]:
            print(f"     🟢 {t.get('token')}: {t.get('reason','')[:100]}")
    caution = crypto.get("caution_tokens", [])
    if caution:
        print(f"  ⚠️  Caution Tokens:")
        for t in caution[:3]:
            print(f"     🔴 {t.get('token')}: {t.get('reason','')[:100]}")

    print(f"\n  📈 Crypto Market: {crypto.get('market_regime','?')}")
    print(f"     {crypto.get('btc_outlook','')[:120]}")
    print("\n" + "=" * 50)


# ══════════════════════════════════════════════════════════════
# INTEGRATION — forward tester reads this
# ══════════════════════════════════════════════════════════════

def get_current_brief() -> dict:
    """
    Called by vault_forward_test.py and apex_risk.py to get current brief.
    Returns cached brief if today's exists, runs new brief if Monday or missing.
    """
    today = date.today()

    # Always run on Monday
    if today.weekday() == 0:  # Monday
        return run_weekly_briefing()

    # Return cached if exists
    if BRIEF_FILE.exists():
        try:
            brief = json.loads(BRIEF_FILE.read_text())
            return brief
        except Exception:
            pass

    # No brief yet — run now
    return run_weekly_briefing()


def get_size_adjustment() -> float:
    """
    Returns size multiplier based on macro regime.
    Called by risk_manager.py before sizing each trade.
    0.5 = risk-off (half size) | 1.0 = normal | 1.25 = risk-on (slightly larger)
    """
    try:
        brief = json.loads(BRIEF_FILE.read_text()) if BRIEF_FILE.exists() else {}
        adj   = brief.get("macro_regime", {}).get("size_adjustment", 1.0)
        # Cap adjustment to reasonable range
        return max(0.5, min(1.25, float(adj)))
    except Exception:
        return 1.0  # default to normal size if can't read


def get_news_events_this_week() -> list[dict]:
    """
    Returns list of high-impact events for this week.
    Called by apex_risk.py news filter as supplementary data.
    """
    try:
        brief  = json.loads(BRIEF_FILE.read_text()) if BRIEF_FILE.exists() else {}
        events = brief.get("events", {}).get("events", [])
        return events
    except Exception:
        return []


def get_caution_tokens() -> list[str]:
    """
    Returns list of crypto tokens to be cautious on this week.
    Called by vault_forward_test.py — reduces conviction on these signals.
    """
    try:
        brief   = json.loads(BRIEF_FILE.read_text()) if BRIEF_FILE.exists() else {}
        caution = brief.get("crypto", {}).get("caution_tokens", [])
        return [t.get("token", "") for t in caution]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════
# SCHEDULER — auto-run every Monday
# ══════════════════════════════════════════════════════════════

def run_scheduler():
    """
    Run weekly briefing every Monday at 6:00 AM EST.
    Keep this running in a background terminal:
      python src/agents/weekly_briefing.py --schedule
    """
    import zoneinfo
    est = zoneinfo.ZoneInfo("America/New_York")
    print("🗓️  Weekly Briefing Scheduler running...")
    print("   Fires every Monday at 6:00 AM EST")
    print("   Press Ctrl+C to stop\n")

    while True:
        now = datetime.now(est)
        # Monday = 0, check if it's between 6:00 and 6:15 AM
        if now.weekday() == 0 and now.hour == 6 and now.minute < 15:
            print(f"\n⏰ Monday 6 AM — Running weekly briefing...")
            run_weekly_briefing(force=True)
            time.sleep(3600)  # Sleep 1hr to avoid running twice
        else:
            # Next Monday check
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0 and now.hour < 6:
                days_until_monday = 0
            elif days_until_monday == 0:
                days_until_monday = 7

            next_run = f"Monday" if days_until_monday <= 1 else f"in {days_until_monday} days"
            print(f"\r  💤 Next brief: {next_run} at 6 AM EST | "
                  f"Now: {now.strftime('%a %H:%M')}   ", end="", flush=True)
            time.sleep(60)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="🌙 Weekly Briefing Agent")
    p.add_argument("--schedule", action="store_true",
                   help="Run scheduler (auto every Monday 6 AM EST)")
    p.add_argument("--force",    action="store_true",
                   help="Force run even if already ran today")
    p.add_argument("--show",     action="store_true",
                   help="Show latest saved brief without re-running")
    args = p.parse_args()

    if args.schedule:
        run_scheduler()
    elif args.show:
        if BRIEF_FILE.exists():
            brief = json.loads(BRIEF_FILE.read_text())
            _print_summary(brief)
        else:
            print("No brief saved yet. Run without --show to generate one.")
    else:
        run_weekly_briefing(force=args.force)
