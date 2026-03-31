# ============================================================
# 🌙 Listing Arbitrage Agent
#
# Identifies promising new tokens on CoinGecko before they
# reach major exchanges like Binance and Coinbase.
# When a token gets listed on a major exchange, price often
# pumps 20-100% — this agent finds them early.
#
# Moon Dev's version does this for Solana tokens specifically.
# This version covers all chains via CoinGecko.
#
# HOW TO RUN:
#   python src/agents/listing_arb_agent.py
#   python src/agents/listing_arb_agent.py --chain solana
#   python src/agents/listing_arb_agent.py --min-volume 1000000
# ============================================================

import sys
import csv
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.llm_router import model

# ── Paths ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE  = REPO_ROOT / "src" / "data" / "listing_arb_log.csv"

# ── Config ────────────────────────────────────────────────────
COINGECKO_BASE  = "https://api.coingecko.com/api/v3"
MIN_VOLUME_USD  = 500_000     # minimum 24h volume
MIN_MARKET_CAP  = 1_000_000   # minimum market cap
MAX_MARKET_CAP  = 500_000_000 # max — avoid already-large tokens
MIN_PRICE_CHANGE = 5.0        # minimum 24h price change %
SLEEP_BETWEEN_RUNS = 300      # 5 minutes

# Major exchanges — token is NOT on these yet = opportunity
MAJOR_EXCHANGES = {
    "binance", "coinbase", "kraken", "okx", "bybit"
}

# ── AI analysis prompt ────────────────────────────────────────
ANALYSIS_PROMPT = """You are a crypto market analyst specializing in new token listings.
Analyze these new tokens and identify the top 3 most promising for a listing arbitrage play.

A good candidate has:
- Growing volume and social activity
- Not yet on Binance or Coinbase
- Strong community/narrative (AI, gaming, DeFi, meme)
- Realistic market cap (not too large)
- Recent price momentum

TOKENS:
{tokens}

Respond with JSON:
{{
  "top_picks": [
    {{
      "symbol": "TOKEN",
      "reason": "One sentence why this is a good listing arb candidate",
      "risk": "low/medium/high",
      "potential_exchange": "binance/coinbase/both"
    }}
  ],
  "summary": "Brief market summary"
}}"""


def get_new_tokens(chain: str = "all",
                   min_volume: float = MIN_VOLUME_USD) -> list[dict]:
    """
    Fetch new/trending tokens from CoinGecko that aren't on
    major exchanges yet.
    """
    tokens = []

    try:
        # Get trending coins
        print("  📡 Fetching trending tokens from CoinGecko...")
        r = requests.get(
            f"{COINGECKO_BASE}/search/trending",
            timeout=10
        )
        trending = r.json().get("coins", [])

        for item in trending:
            coin = item.get("item", {})
            tokens.append({
                "id":       coin.get("id", ""),
                "symbol":   coin.get("symbol", "").upper(),
                "name":     coin.get("name", ""),
                "rank":     coin.get("market_cap_rank", 9999),
                "source":   "trending",
            })

        time.sleep(1)

        # Get recently added tokens
        print("  📡 Fetching recently added tokens...")
        r = requests.get(
            f"{COINGECKO_BASE}/coins/list/new",
            timeout=10
        )
        new_coins = r.json()

        for coin in new_coins[:50]:  # check top 50 newest
            tokens.append({
                "id":     coin.get("id", ""),
                "symbol": coin.get("symbol", "").upper(),
                "name":   coin.get("name", ""),
                "rank":   9999,
                "source": "new_listing",
            })

        time.sleep(1)

    except Exception as e:
        print(f"  ⚠️  CoinGecko fetch error: {e}")

    return tokens


def get_token_details(token_ids: list[str]) -> list[dict]:
    """Get detailed market data for a list of token IDs."""
    if not token_ids:
        return []

    details = []
    batch_size = 50

    for i in range(0, len(token_ids), batch_size):
        batch = token_ids[i:i+batch_size]
        ids   = ",".join(batch)

        try:
            r = requests.get(
                f"{COINGECKO_BASE}/coins/markets",
                params={
                    "vs_currency":           "usd",
                    "ids":                   ids,
                    "order":                 "volume_desc",
                    "price_change_percentage": "24h",
                },
                timeout=15
            )
            data = r.json()
            if isinstance(data, list):
                details.extend(data)
            time.sleep(1)

        except Exception as e:
            print(f"  ⚠️  Detail fetch error: {e}")

    return details


def check_exchange_listings(token_id: str) -> set:
    """Check which major exchanges a token is listed on."""
    try:
        r = requests.get(
            f"{COINGECKO_BASE}/coins/{token_id}/tickers",
            params={"depth": False},
            timeout=10
        )
        tickers = r.json().get("tickers", [])
        exchanges = {
            t.get("market", {}).get("identifier", "").lower()
            for t in tickers
        }
        return exchanges & MAJOR_EXCHANGES
    except Exception:
        return set()


def filter_candidates(details: list[dict],
                      min_volume: float) -> list[dict]:
    """Filter tokens to find listing arb candidates."""
    candidates = []

    for token in details:
        try:
            volume     = float(token.get("total_volume", 0) or 0)
            mkt_cap    = float(token.get("market_cap", 0) or 0)
            price_chg  = float(token.get("price_change_percentage_24h", 0) or 0)
            symbol     = token.get("symbol", "").upper()
            name       = token.get("name", "")
            token_id   = token.get("id", "")

            # Basic filters
            if volume < min_volume:
                continue
            if mkt_cap < MIN_MARKET_CAP or mkt_cap > MAX_MARKET_CAP:
                continue
            if price_chg < MIN_PRICE_CHANGE:
                continue

            # Check if already on major exchanges
            listed_on = check_exchange_listings(token_id)
            time.sleep(0.5)

            if MAJOR_EXCHANGES.issubset(listed_on):
                continue  # already on all major exchanges

            candidates.append({
                "id":         token_id,
                "symbol":     symbol,
                "name":       name,
                "price":      token.get("current_price", 0),
                "volume_24h": round(volume),
                "market_cap": round(mkt_cap),
                "change_24h": round(price_chg, 2),
                "listed_on":  list(listed_on),
                "missing_from": list(MAJOR_EXCHANGES - listed_on),
            })

        except Exception:
            continue

    return candidates


def analyze_with_ai(candidates: list[dict]) -> dict:
    """Use AI to rank and analyze the candidates."""
    if not candidates:
        return {}

    token_summary = json.dumps([{
        "symbol":    c["symbol"],
        "name":      c["name"],
        "volume_24h": f"${c['volume_24h']:,}",
        "market_cap": f"${c['market_cap']:,}",
        "change_24h": f"{c['change_24h']}%",
        "missing_from": c["missing_from"],
    } for c in candidates[:15]], indent=2)

    try:
        raw = model.chat(
            system_prompt="You are a crypto market analyst. Return only valid JSON.",
            user_prompt=ANALYSIS_PROMPT.format(tokens=token_summary),
        )
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


def log_results(candidates: list[dict], analysis: dict):
    """Save results to CSV log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for c in candidates:
        rows.append({
            "timestamp":     timestamp,
            "symbol":        c["symbol"],
            "name":          c["name"],
            "price":         c["price"],
            "volume_24h":    c["volume_24h"],
            "market_cap":    c["market_cap"],
            "change_24h":    c["change_24h"],
            "listed_on":     "|".join(c["listed_on"]),
            "missing_from":  "|".join(c["missing_from"]),
        })

    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_once(chain: str = "all",
             min_volume: float = MIN_VOLUME_USD) -> list[dict]:
    """Single scan run."""
    print(f"\n💎 Listing Arbitrage Agent")
    print(f"   Min volume: ${min_volume:,.0f}")
    print(f"   Checking:   {MAJOR_EXCHANGES}")
    print(f"{'='*55}")

    # Get tokens
    tokens     = get_new_tokens(chain, min_volume)
    token_ids  = list({t["id"] for t in tokens if t["id"]})
    print(f"\n  Found {len(token_ids)} tokens to analyze")

    # Get details
    print(f"  Fetching market data...")
    details    = get_token_details(token_ids)

    # Filter candidates
    print(f"  Filtering candidates...")
    candidates = filter_candidates(details, min_volume)
    print(f"  Candidates: {len(candidates)}")

    if not candidates:
        print(f"\n  ℹ️  No candidates found this scan")
        return []

    # AI analysis
    print(f"\n  🤖 AI analyzing top candidates...")
    analysis   = analyze_with_ai(candidates)

    # Display results
    print(f"\n{'='*55}")
    print(f"  📊 LISTING ARB CANDIDATES ({len(candidates)} found)")
    print(f"{'='*55}\n")

    for c in sorted(candidates, key=lambda x: x["volume_24h"], reverse=True)[:10]:
        missing = ", ".join(c["missing_from"])
        print(f"  {c['symbol']:<8} ${c['price']:<12.4f} "
              f"Vol: ${c['volume_24h']:>12,}  "
              f"+{c['change_24h']}%  "
              f"Missing from: {missing}")

    if "top_picks" in analysis:
        print(f"\n  🎯 AI TOP PICKS:")
        for pick in analysis["top_picks"]:
            print(f"\n    {pick['symbol']} — {pick['reason']}")
            print(f"    Risk: {pick['risk']} | "
                  f"Likely listing: {pick.get('potential_exchange','?')}")

    if "summary" in analysis:
        print(f"\n  Market summary: {analysis['summary']}")

    # Log
    if candidates:
        log_results(candidates, analysis)

    return candidates


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="💎 Listing Arbitrage Agent")
    p.add_argument("--chain",      default="all",
                   help="blockchain filter (default: all)")
    p.add_argument("--min-volume", type=float, default=MIN_VOLUME_USD,
                   help=f"min 24h volume USD (default: {MIN_VOLUME_USD:,.0f})")
    p.add_argument("--once",       action="store_true",
                   help="Single scan then exit")
    args = p.parse_args()

    if args.once:
        run_once(args.chain, args.min_volume)
    else:
        print(f"🔄 Running continuously every {SLEEP_BETWEEN_RUNS}s...")
        while True:
            try:
                run_once(args.chain, args.min_volume)
            except KeyboardInterrupt:
                print("\n🛑 Stopped.")
                break
            except Exception as e:
                print(f"  ❌ Error: {e}")
            print(f"\n  💤 Next scan in {SLEEP_BETWEEN_RUNS}s...")
            time.sleep(SLEEP_BETWEEN_RUNS)
