# ============================================================
# 🌙 Funding Rate Arbitrage Agent
#
# Moon Dev's favourite strategy. Exploits the difference
# between perpetual futures funding rates and spot prices.
#
# LOGIC:
#   High positive funding  → Longs paying shorts → SHORT perp + HOLD spot
#   High negative funding  → Shorts paying longs → LONG perp + SHORT spot
#   Funding resets every 8 hours on most exchanges.
#
# HOW TO RUN:
#   python src/agents/funding_arb_agent.py
#   python src/agents/funding_arb_agent.py --once
# ============================================================

import sys
import csv
import json
import time
import requests
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config               import HYPERLIQUID_TOKENS, SLEEP_BETWEEN_RUNS_SEC
from src.models.llm_router import model

REPO_ROOT = Path(__file__).resolve().parents[2]
ARB_LOG   = REPO_ROOT / "src" / "data" / "funding_arb_log.csv"
ARB_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────
MIN_FUNDING_RATE   = 0.0005    # 0.05% per 8h = ~54% annualised — minimum to act
ANNUALISED_TARGET  = 0.20      # 20% annualised minimum threshold
TRADE_SIZE_USD     = 100       # size per leg


def annualise_funding(rate_8h: float) -> float:
    """Convert 8-hour funding rate to annualised percentage."""
    return rate_8h * 3 * 365 * 100   # 3 periods/day × 365 days × 100


def get_all_funding_rates() -> list[dict]:
    """Fetch current funding rates from Hyperliquid for all assets."""
    try:
        r    = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "metaAndAssetCtxs"},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        data     = r.json()
        universe = data[0].get("universe", [])
        ctxs     = data[1]

        rates = []
        for i, asset in enumerate(universe):
            ctx          = ctxs[i]
            funding_8h   = float(ctx.get("funding", 0))
            mark_px      = float(ctx.get("markPx", 0))
            oi           = float(ctx.get("openInterest", 0)) * mark_px
            annualised   = annualise_funding(funding_8h)

            rates.append({
                "symbol":      asset.get("name"),
                "funding_8h":  round(funding_8h * 100, 5),    # as %
                "annualised":  round(annualised, 2),
                "mark_price":  mark_px,
                "oi_millions": round(oi / 1e6, 2),
                "direction":   "SHORT_PERP" if funding_8h > 0 else "LONG_PERP",
            })

        # Sort by absolute annualised rate
        rates.sort(key=lambda x: abs(x["annualised"]), reverse=True)
        return rates

    except Exception as e:
        print(f"  ❌ Failed to fetch funding rates: {e}")
        return []


ANALYSIS_PROMPT = """You are a crypto funding rate arbitrage specialist.
Analyse these funding rate opportunities and rank the best ones.

DATA:
{data}

For each opportunity assess:
1. Is the annualised rate above 20%? (minimum threshold)
2. Is OI large enough for liquid execution?
3. Any risks (e.g. rate about to reset, low OI)?

Respond ONLY with valid JSON:
{{
  "top_opportunities": [
    {{
      "symbol": "...",
      "action": "SHORT_PERP_HOLD_SPOT or LONG_PERP_SHORT_SPOT",
      "annualised_pct": 0.0,
      "confidence": 0.0-1.0,
      "reasoning": "one sentence",
      "risk": "one sentence"
    }}
  ],
  "market_summary": "one sentence on overall funding environment"
}}"""


def analyse_opportunities(rates: list[dict]) -> dict:
    """Ask DeepSeek to rank the best funding arb opportunities."""
    # Only send top 10 to save tokens
    top = [r for r in rates if abs(r["annualised"]) >= ANNUALISED_TARGET * 100][:10]
    if not top:
        return {"top_opportunities": [], "market_summary": "No opportunities above threshold"}
    try:
        raw    = model.chat(
            system_prompt="You are a funding rate arbitrage analyst. Return only valid JSON.",
            user_prompt=ANALYSIS_PROMPT.format(data=json.dumps(top, indent=2)),
        )
        raw    = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"top_opportunities": top[:3], "market_summary": f"Analysis error: {e}"}


def log_opportunity(opp: dict):
    row = {
        "timestamp":   datetime.now().isoformat(),
        "symbol":      opp.get("symbol", ""),
        "action":      opp.get("action", ""),
        "annualised":  opp.get("annualised_pct", 0),
        "confidence":  opp.get("confidence", 0),
        "reasoning":   opp.get("reasoning", ""),
        "risk":        opp.get("risk", ""),
    }
    write_header = not ARB_LOG.exists()
    with open(ARB_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


class FundingArbAgent:

    def scan(self):
        print(f"\n{'═'*60}")
        print(f"💰 Funding Arb Scan — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Threshold: {ANNUALISED_TARGET*100:.0f}% annualised")

        rates = get_all_funding_rates()
        if not rates:
            return

        # Show top 10
        print(f"\n  📊 Top Funding Rates (all assets):")
        print(f"  {'Symbol':<8} {'8h Rate':>10} {'Annualised':>12} {'Direction':<15} {'OI $M':>8}")
        print(f"  {'─'*58}")
        for r in rates[:10]:
            flag = "🔥" if abs(r["annualised"]) >= ANNUALISED_TARGET * 100 else "  "
            print(f"  {flag} {r['symbol']:<6} {r['funding_8h']:>+9.4f}% "
                  f"{r['annualised']:>+11.1f}%  {r['direction']:<15} {r['oi_millions']:>7.1f}")

        # Filter actionable
        actionable = [r for r in rates if abs(r["annualised"]) >= ANNUALISED_TARGET * 100]
        if not actionable:
            print(f"\n  ⏸️  No opportunities above {ANNUALISED_TARGET*100:.0f}% threshold")
            return

        print(f"\n  🎯 {len(actionable)} opportunities above threshold — analysing...")
        analysis = analyse_opportunities(rates)

        print(f"\n  🌍 Market: {analysis.get('market_summary','')}")
        print(f"\n  🏆 Top Opportunities:")
        for opp in analysis.get("top_opportunities", []):
            icon = "🟢" if opp.get("annualised_pct", 0) > 0 else "🔴"
            print(f"\n  {icon} {opp.get('symbol')} — {opp.get('annualised_pct',0):+.1f}% annualised")
            print(f"     Action  : {opp.get('action','')}")
            print(f"     Reason  : {opp.get('reasoning','')}")
            print(f"     Risk    : {opp.get('risk','')}")
            print(f"     Confidence: {opp.get('confidence',0):.0%}")
            log_opportunity(opp)

        print(f"\n  ⚠️  Remember: Funding arb requires BOTH legs (perp + spot/hedge)")
        print(f"     Net income ≈ ${TRADE_SIZE_USD * ANNUALISED_TARGET:.0f}/year on ${TRADE_SIZE_USD} per trade")

    def run(self):
        print("🚀 Funding Arb Agent running. Press Ctrl+C to stop.\n")
        try:
            while True:
                self.scan()
                # Funding resets every 8h — scan every hour
                sleep = min(SLEEP_BETWEEN_RUNS_SEC, 3600)
                print(f"\n😴 Next scan in {sleep//60} minutes...")
                time.sleep(sleep)
        except KeyboardInterrupt:
            print("\n🛑 Funding Arb Agent stopped.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    agent = FundingArbAgent()
    if args.once:
        agent.scan()
    else:
        agent.run()
