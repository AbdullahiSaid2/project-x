# ============================================================
# 🌙 Sentiment Agent
#
# Pulls recent crypto news headlines via free RSS feeds
# and uses DeepSeek to score overall market sentiment.
# No Twitter API key needed — uses public RSS sources.
#
# HOW TO RUN:
#   python src/agents/sentiment_agent.py
#   python src/agents/sentiment_agent.py --once
# ============================================================

import sys
import csv
import json
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config               import SLEEP_BETWEEN_RUNS_SEC, HYPERLIQUID_TOKENS
from src.models.llm_router import model

# ── paths ─────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parents[2]
SENTIMENT_LOG = REPO_ROOT / "src" / "data" / "sentiment_log.csv"
SENTIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── RSS news sources (free, no API key) ───────────────────────
RSS_FEEDS = {
    "CoinDesk":      "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "CryptoSlate":   "https://cryptoslate.com/feed/",
    "Decrypt":       "https://decrypt.co/feed",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"}


def fetch_headlines(max_per_feed: int = 5) -> list[dict]:
    """Fetch recent headlines from crypto RSS feeds."""
    headlines = []
    for source, url in RSS_FEEDS.items():
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            root = ET.fromstring(r.content)
            items = root.findall(".//item")[:max_per_feed]
            for item in items:
                title = item.findtext("title", "").strip()
                pub   = item.findtext("pubDate", "").strip()
                if title:
                    headlines.append({
                        "source": source,
                        "title":  title,
                        "date":   pub,
                    })
        except Exception as e:
            print(f"   ⚠️  Failed to fetch {source}: {e}")
        time.sleep(0.2)
    return headlines


# ── AI sentiment analysis ─────────────────────────────────────
SENTIMENT_PROMPT = """You are a crypto market sentiment analyst.
Analyse the following recent news headlines and score the overall market sentiment.

Focus on how these headlines would affect crypto traders' psychology and positioning.

HEADLINES:
{headlines}

TOKEN FOCUS: {tokens}

Respond ONLY with valid JSON:
{{
  "overall_sentiment": "VERY_BULLISH" | "BULLISH" | "NEUTRAL" | "BEARISH" | "VERY_BEARISH",
  "score": -1.0 to +1.0,
  "confidence": 0.0-1.0,
  "token_sentiments": {{
    "BTC": "BULLISH" | "NEUTRAL" | "BEARISH",
    "ETH": "BULLISH" | "NEUTRAL" | "BEARISH",
    "SOL": "BULLISH" | "NEUTRAL" | "BEARISH"
  }},
  "key_themes": ["theme1", "theme2", "theme3"],
  "top_headline_impact": "Which headline has the most market impact and why",
  "trading_bias": "SHORT_TERM" | "MEDIUM_TERM" | "BOTH" | "NONE",
  "summary": "Two sentence market sentiment summary"
}}"""


def analyse_sentiment(headlines: list[dict], tokens: list[str]) -> dict:
    """Ask DeepSeek to score overall sentiment from headlines."""
    formatted = "\n".join([
        f"[{h['source']}] {h['title']}"
        for h in headlines
    ])
    try:
        raw    = model.chat(
            system_prompt="You are a crypto sentiment analyst. Return only valid JSON.",
            user_prompt=SENTIMENT_PROMPT.format(
                headlines=formatted,
                tokens=", ".join(tokens),
            ),
        )
        raw    = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return result
    except Exception as e:
        return {
            "overall_sentiment": "NEUTRAL",
            "score": 0.0,
            "confidence": 0.0,
            "summary": f"Analysis failed: {e}",
        }


# ── logging ───────────────────────────────────────────────────
def _log(row: dict):
    write_header = not SENTIMENT_LOG.exists()
    with open(SENTIMENT_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


def _sentiment_bar(score: float) -> str:
    """Visual bar for sentiment score -1 to +1."""
    filled = int((score + 1) / 2 * 20)
    return "░" * (20 - filled) + "█" * filled


# ═══════════════════════════════════════════════════════════════
class SentimentAgent:

    def __init__(self):
        self.tokens = HYPERLIQUID_TOKENS[:5]
        print("📰 Sentiment Agent initialised")
        print(f"   Sources : {list(RSS_FEEDS.keys())}")
        print(f"   Tokens  : {self.tokens}\n")

    def scan(self) -> dict:
        print(f"\n{'═'*60}")
        print(f"📰 Sentiment scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1 — Fetch headlines
        print("\n  📡 Fetching headlines...")
        headlines = fetch_headlines(max_per_feed=6)
        print(f"  ✅ {len(headlines)} headlines fetched from {len(RSS_FEEDS)} sources")

        if not headlines:
            print("  ❌ No headlines fetched — check network connection")
            return {}

        # Show sample headlines
        print("\n  📋 Recent headlines:")
        for h in headlines[:5]:
            print(f"     [{h['source']}] {h['title'][:70]}...")

        # 2 — AI analysis
        print("\n  🤖 Analysing sentiment with DeepSeek...")
        result = analyse_sentiment(headlines, self.tokens)

        # 3 — Display results
        sentiment = result.get("overall_sentiment", "NEUTRAL")
        score     = result.get("score", 0.0)
        conf      = result.get("confidence", 0.0)
        summary   = result.get("summary", "")
        themes    = result.get("key_themes", [])
        top_hl    = result.get("top_headline_impact", "")

        icon_map = {
            "VERY_BULLISH": "🚀", "BULLISH": "🟢",
            "NEUTRAL": "⚪",
            "BEARISH": "🔴", "VERY_BEARISH": "💀",
        }
        icon = icon_map.get(sentiment, "⚪")

        print(f"\n  {icon} Overall Sentiment: {sentiment}")
        print(f"     Score     : {score:+.2f}  [{_sentiment_bar(score)}]")
        print(f"     Confidence: {conf:.0%}")
        print(f"     Summary   : {summary}")

        # Per-token breakdown
        token_sentiments = result.get("token_sentiments", {})
        if token_sentiments:
            print("\n  📊 Token breakdown:")
            for token, ts in token_sentiments.items():
                ti = icon_map.get(ts, "⚪")
                print(f"     {token:4s}: {ti} {ts}")

        if themes:
            print(f"\n  🏷️  Key themes: {', '.join(themes)}")
        if top_hl:
            print(f"\n  ⚡ Biggest impact: {top_hl}")

        # 4 — Log
        row = {
            "timestamp":       datetime.now().isoformat(),
            "sentiment":       sentiment,
            "score":           round(score, 3),
            "confidence":      round(conf, 3),
            "headlines_count": len(headlines),
            "key_themes":      "|".join(themes),
            "summary":         summary[:200],
            "btc_sentiment":   token_sentiments.get("BTC", "NEUTRAL"),
            "eth_sentiment":   token_sentiments.get("ETH", "NEUTRAL"),
            "sol_sentiment":   token_sentiments.get("SOL", "NEUTRAL"),
        }
        _log(row)

        print(f"\n✅ Sentiment scan complete.")
        return result

    def run(self):
        print("🚀 Sentiment Agent running. Press Ctrl+C to stop.\n")
        try:
            while True:
                self.scan()
                # Sentiment changes slowly — scan every 30 mins
                sleep_time = max(SLEEP_BETWEEN_RUNS_SEC, 1800)
                print(f"\n😴 Next scan in {sleep_time//60} minutes...")
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n🛑 Sentiment Agent stopped.")


# ── entrypoint ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="📰 Sentiment Agent")
    parser.add_argument("--once", action="store_true", help="Single scan then exit")
    args = parser.parse_args()

    agent = SentimentAgent()
    if args.once:
        agent.scan()
    else:
        agent.run()
