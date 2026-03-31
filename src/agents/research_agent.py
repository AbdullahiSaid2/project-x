# ============================================================
# 🌙 Research Agent
#
# Automatically finds new trading strategy ideas from the web
# and adds them to ideas.txt so the RBI backtester can run
# continuously discovering new strategies.
#
# This is the key Moon Dev feature that lets the RBI agent
# run forever — the Research Agent keeps feeding it new ideas.
#
# WHAT IT DOES:
#   1. Searches the web for trading strategy resources
#   2. Reads the content of each page
#   3. Uses Claude to extract concrete trading ideas
#   4. Converts them to code-able backtesting format
#   5. Deduplicates and adds to ideas.txt
#
# HOW TO RUN:
#   python src/agents/research_agent.py
#   python src/agents/research_agent.py --queries 5
#   python src/agents/research_agent.py --dry-run  (preview only)
# ============================================================

import sys
import json
import time
import random
import requests
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.llm_router import rbi_model as model

# ── Paths ─────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
IDEAS_FILE = REPO_ROOT / "src" / "data" / "ideas.txt"
LOG_FILE   = REPO_ROOT / "src" / "data" / "research_agent_log.json"

# ── Search queries to find strategy ideas ─────────────────────
SEARCH_QUERIES = [
    "profitable futures trading strategies RSI MACD backtested",
    "momentum trading strategies NQ ES futures",
    "mean reversion trading strategy futures 2024 2025",
    "volume spread analysis trading strategy",
    "ICT trading concepts systematic rules",
    "quantitative trading strategies futures markets",
    "VWAP trading strategy intraday futures",
    "order flow trading strategies futures",
    "breakout trading strategy NQ futures",
    "trend following strategies micro futures",
    "crypto trading strategies BTC ETH backtested Sharpe",
    "algorithmic trading strategy ideas python backtesting",
    "supply demand trading strategy systematic rules",
    "fair value gap trading strategy rules entry exit",
    "liquidity sweep trading strategy systematic",
    "opening range breakout strategy futures",
    "ATR based trading strategies futures",
    "Bollinger bands trading strategy futures",
    "stochastic oscillator trading strategy",
    "market profile trading strategies futures",
]

# ── Extraction prompt ─────────────────────────────────────────
EXTRACT_PROMPT = """You are a quantitative trading strategy researcher.

Read the following content from a trading website/article and extract
concrete, specific trading strategy ideas that can be backtested.

CONTENT:
{content}

RULES FOR EXTRACTION:
1. Each idea must be ONE specific entry condition + exit condition
2. Maximum 2 conditions per idea (e.g. "when X AND Y, enter long")
3. Must be measurable with technical indicators (RSI, MACD, EMA, ATR, volume etc)
4. No discretionary concepts ("when price looks strong") — must be objective
5. Must specify direction: Long or Short
6. No multi-timeframe (single timeframe only)
7. Write in plain English, not code

FORMAT — return ONLY a JSON array, no markdown:
[
  "Long when [specific condition 1] and [optional condition 2], exit when [exit condition]",
  "Short when [specific condition 1], exit when [exit condition]"
]

Extract 3-8 ideas. If no concrete backtestable ideas found, return empty array: []"""

SIMPLIFY_PROMPT = """You are a quantitative trading researcher.

Convert this trading idea into a simple, backtestable format.
The idea must:
- Use standard technical indicators only (RSI, MACD, EMA, ATR, Bollinger Bands, volume etc)
- Have at most 2 entry conditions
- Specify a clear exit condition
- Be measurable/objective — no subjective language
- Specify direction (Long or Short)

IDEA: {idea}

Return ONE clean sentence in this format:
"Long/Short when [condition 1] and optionally [condition 2], exit when [exit condition]"

Return ONLY the sentence, nothing else."""


def search_web(query: str, num_results: int = 3) -> list[dict]:
    """
    Simple web search using DuckDuckGo (no API key needed).
    Returns list of {title, url, snippet} dicts.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        r   = requests.get(url, headers=headers, timeout=10)

        results = []
        # Parse DuckDuckGo HTML results
        from html.parser import HTMLParser

        class DDGParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.in_result = False
                self.current = {}
                self.capture_title = False
                self.capture_snippet = False

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "a" and "result__a" in attrs.get("class",""):
                    self.capture_title = True
                    self.current["url"] = attrs.get("href","")
                if tag == "a" and "result__snippet" in attrs.get("class",""):
                    self.capture_snippet = True

            def handle_data(self, data):
                if self.capture_title:
                    self.current["title"] = data.strip()
                    self.capture_title = False
                if self.capture_snippet:
                    self.current["snippet"] = data.strip()
                    self.capture_snippet = False
                    if self.current.get("url"):
                        self.results.append(self.current.copy())
                    self.current = {}

        parser = DDGParser()
        parser.feed(r.text)
        return parser.results[:num_results]

    except Exception as e:
        print(f"    ⚠️  Search failed: {e}")
        return []


def fetch_page_content(url: str, max_chars: int = 3000) -> str:
    """Fetch and clean text content from a URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; TradingResearcher/1.0)"
        }
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()

        # Basic HTML cleaning
        text = re.sub(r'<script[^>]*>.*?</script>', '', r.text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>',  '', text,   flags=re.DOTALL)
        text = re.sub(r'<[^>]+>',                  ' ', text)
        text = re.sub(r'\s+',                      ' ', text).strip()
        text = re.sub(r'[^\x00-\x7F]+',            ' ', text)  # ASCII only

        return text[:max_chars]
    except Exception:
        return ""


def extract_ideas_from_content(content: str) -> list[str]:
    """Use Claude to extract trading ideas from page content."""
    if len(content) < 100:
        return []

    try:
        raw = model.chat(
            system_prompt="You extract trading strategy ideas. Return only valid JSON.",
            user_prompt=EXTRACT_PROMPT.format(content=content),
            max_tokens=1000,
        )
        raw   = re.sub(r"```json|```", "", raw).strip()
        ideas = json.loads(raw)
        return [i for i in ideas if isinstance(i, str) and len(i) > 20]
    except Exception:
        return []


def load_existing_ideas() -> set:
    """Load existing ideas from ideas.txt for deduplication."""
    if not IDEAS_FILE.exists():
        return set()
    lines = IDEAS_FILE.read_text().splitlines()
    return {l.strip().lower()[:60] for l in lines
            if l.strip() and not l.startswith("#")}


def is_duplicate(idea: str, existing: set) -> bool:
    """Check if idea is already in ideas.txt."""
    idea_key = idea.strip().lower()[:60]
    for e in existing:
        if idea_key[:40] == e[:40]:
            return True
    return False


def save_ideas(new_ideas: list[str], dry_run: bool = False) -> int:
    """Append new ideas to ideas.txt. Returns count added."""
    if not new_ideas:
        return 0

    if dry_run:
        print(f"\n  [DRY RUN] Would add {len(new_ideas)} ideas:")
        for idea in new_ideas:
            print(f"    → {idea[:90]}")
        return len(new_ideas)

    timestamp = datetime.now().strftime("%Y-%m-%d")
    header    = f"\n\n# ── Research Agent — {timestamp} ──────────────────\n"

    with open(IDEAS_FILE, "a") as f:
        f.write(header)
        for idea in new_ideas:
            f.write(idea + "\n")

    return len(new_ideas)


def log_run(queries_used: int, ideas_found: int, ideas_added: int):
    """Log this research run."""
    log = []
    if LOG_FILE.exists():
        try:
            log = json.loads(LOG_FILE.read_text())
        except Exception:
            log = []

    log.append({
        "timestamp":    datetime.now().isoformat(),
        "queries_used": queries_used,
        "ideas_found":  ideas_found,
        "ideas_added":  ideas_added,
    })

    # Keep last 50 runs
    LOG_FILE.write_text(json.dumps(log[-50:], indent=2))


def run_research(num_queries: int = 3,
                 dry_run: bool = False,
                 verbose: bool = True) -> int:
    """
    Main research loop. Returns number of new ideas added.
    """
    print(f"\n🔬 Research Agent — Finding New Strategy Ideas")
    print(f"   Queries: {num_queries} | Dry run: {dry_run}")
    print(f"   Target : {IDEAS_FILE}")
    print(f"{'='*55}\n")

    existing_ideas = load_existing_ideas()
    print(f"  Existing ideas in file: {len(existing_ideas)}\n")

    # Pick random queries to avoid repetition
    queries = random.sample(SEARCH_QUERIES, min(num_queries, len(SEARCH_QUERIES)))

    all_new_ideas = []
    total_found   = 0

    for i, query in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] Searching: \"{query}\"")

        results = search_web(query, num_results=3)
        if not results:
            print(f"    ⚠️  No results found")
            continue

        for result in results:
            url = result.get("url", "")
            if not url or "duckduckgo" in url:
                continue

            # Skip known low-quality sources
            skip_domains = ["reddit.com", "youtube.com", "twitter.com",
                           "facebook.com", "instagram.com", "tiktok.com"]
            if any(d in url for d in skip_domains):
                continue

            if verbose:
                print(f"    📄 Reading: {url[:60]}...")

            content = fetch_page_content(url)
            if not content:
                continue

            ideas = extract_ideas_from_content(content)
            total_found += len(ideas)

            # Filter duplicates and validate
            new_ideas = []
            for idea in ideas:
                idea = idea.strip()
                if (len(idea) > 30 and
                        not is_duplicate(idea, existing_ideas) and
                        any(w in idea.lower() for w in
                            ["long", "short", "buy", "sell"]) and
                        any(w in idea.lower() for w in
                            ["rsi", "macd", "ema", "atr", "volume",
                             "bollinger", "vwap", "stoch", "momentum",
                             "bar", "candle", "price", "close", "high", "low"])):
                    new_ideas.append(idea)
                    # Add to existing to prevent duplicates within this run
                    existing_ideas.add(idea.lower()[:60])

            if new_ideas:
                all_new_ideas.extend(new_ideas)
                print(f"    ✅ Found {len(new_ideas)} new ideas")
            else:
                print(f"    ○  No new ideas from this page")

            time.sleep(1)  # polite delay

        time.sleep(2)  # between queries

    # Save results
    print(f"\n{'='*55}")
    print(f"  Total ideas found  : {total_found}")
    print(f"  New (deduplicated) : {len(all_new_ideas)}")

    added = save_ideas(all_new_ideas, dry_run=dry_run)

    log_run(len(queries), total_found, added)

    if added > 0 and not dry_run:
        print(f"\n  ✅ Added {added} new ideas to ideas.txt")
        print(f"     Run RBI backtester to test them:")
        print(f"     python src/agents/rbi_parallel.py --market crypto")
    elif added == 0:
        print(f"\n  ℹ️  No new ideas to add this run")

    return added


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="🔬 Research Agent")
    p.add_argument("--queries", type=int, default=3,
                   help="Number of search queries to run (default: 3)")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview ideas without saving")
    p.add_argument("--verbose", action="store_true", default=True)
    args = p.parse_args()

    run_research(
        num_queries=args.queries,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
