# ============================================================
# 🌙 Websearch Agent — Fully Automated
#
# Zero manual work required. Runs automatically:
#   1. Searches web for trading strategy articles
#   2. Finds and downloads PDFs automatically
#   3. Finds YouTube videos and extracts strategy info
#   4. Extracts concrete rules from all sources
#   5. Deduplicates and adds to ideas.txt
#   6. Logs processed URLs to avoid repeats
#
# HOW TO RUN:
#   python src/agents/websearch_agent.py            # one-shot
#   python src/agents/websearch_agent.py --queries 10
#   python src/agents/websearch_agent.py --continuous
# ============================================================

import sys, json, time, re, io, os, requests, random
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.llm_router import model  # default router: Claude → DeepSeek → Groq → Gemini

REPO_ROOT     = Path(__file__).resolve().parents[2]
IDEAS_FILE    = REPO_ROOT / "src" / "data" / "ideas.txt"
LOG_FILE      = REPO_ROOT / "src" / "data" / "websearch_log.json"
PDF_CACHE_DIR = REPO_ROOT / "src" / "data" / "pdf_cache"
PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

WEB_QUERIES = [
    "trading strategy rules entry exit systematic backtested NQ futures",
    "RSI momentum trading strategy rules entry exit futures crypto",
    "MACD crossover trading strategy systematic futures",
    "mean reversion trading strategy crypto BTC rules",
    "volume breakout trading strategy systematic futures",
    "ICT liquidity sweep trading strategy rules entry exit",
    "EMA crossover trading strategy entry exit stop loss",
    "Bollinger bands trading strategy systematic rules",
    "VWAP intraday trading strategy futures rules",
    "momentum trading strategy crypto ETH SOL systematic",
    "order block trading strategy rules entry exit",
    "ATR breakout trading strategy futures systematic",
    "stochastic oscillator trading strategy systematic rules",
    "divergence trading strategy RSI MACD systematic",
    "gap fill trading strategy futures systematic",
]

PDF_QUERIES = [
    "algorithmic trading strategy rules backtesting PDF",
    "futures trading strategy systematic rules PDF",
    "crypto trading strategy technical analysis PDF",
    "quantitative trading strategy research PDF",
    "momentum trading strategy backtested PDF",
]

YOUTUBE_QUERIES = [
    "site:youtube.com futures trading strategy systematic NQ ES",
    "site:youtube.com crypto trading strategy BTC ETH rules entry exit",
    "site:youtube.com ICT trading strategy systematic rules",
    "site:youtube.com algo trading strategy backtesting python",
    "site:youtube.com mean reversion trading strategy systematic",
]

EXTRACT_PROMPT = """You are a quantitative trading strategy researcher.
Extract backtestable trading ideas from this content.

SOURCE: {source_type} — {title}
CONTENT: {content}

RULES:
1. Each idea = direction (Long/Short) + entry condition + exit condition
2. Max 2 entry conditions per idea
3. Use only measurable indicators: RSI, MACD, EMA, SMA, ATR, Bollinger, VWAP, volume, Stochastic, CCI, momentum, price levels
4. No vague language — must be algorithmic
5. Format: "Long/Short when [condition], exit when [condition]"

Return ONLY a JSON array:
["Long when RSI crosses below 30 and close is above 20 EMA, exit when RSI crosses above 50"]

Extract 3-8 ideas. Return [] if none found."""

PDF_EXTRACT_PROMPT = """Extract ALL trading strategy rules from this document.

DOCUMENT: {title}
CONTENT: {content}

Each rule must have direction (Long/Short), entry condition, exit condition.
Format as JSON array:
["Long when [entry], exit when [exit]", "Short when [entry], exit when [exit]"]
Return [] if none found."""

YOUTUBE_PROMPT = """Analyze this YouTube trading video and extract strategy rules.

TITLE: {title}
DESCRIPTION: {content}

Extract concrete entry/exit rules. Return JSON array:
["Long when..., exit when...", "Short when..., exit when..."]
Return [] if insufficient info."""


def load_processed_urls():
    if not LOG_FILE.exists(): return set()
    try: return set(json.loads(LOG_FILE.read_text()).get("urls", []))
    except: return set()

def save_processed_url(url):
    data = {"urls": []}
    if LOG_FILE.exists():
        try: data = json.loads(LOG_FILE.read_text())
        except: pass
    urls = set(data.get("urls", []))
    urls.add(url)
    data["urls"] = list(urls)[-500:]
    LOG_FILE.write_text(json.dumps(data, indent=2))

def load_existing_ideas():
    if not IDEAS_FILE.exists(): return set()
    return {l.strip().lower()[:50] for l in IDEAS_FILE.read_text().splitlines()
            if l.strip() and not l.startswith("#")}

def is_valid_idea(idea, existing):
    if len(idea) < 30 or len(idea) > 300: return False
    idea_l = idea.lower()
    if not any(w in idea_l for w in ["long", "short", "buy", "sell"]): return False
    inds = ["rsi","macd","ema","sma","atr","vwap","volume","bollinger","stoch",
            "momentum","price","close","high","low","bar","candle","cross","cci",
            "moving average","average","oscillator","breakout","reversal","trend",
            "support","resistance","fibonacci","pivot","pattern","signal"]
    if not any(i in idea_l for i in inds): return False
    vague = ["looks","seems","appears","feels","discretion","visual","subjective"]
    if any(v in idea_l for v in vague): return False
    key = idea_l[:50]
    return not any(key[:35] == e[:35] for e in existing)

def save_ideas(ideas, label=""):
    if not ideas: return 0
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(IDEAS_FILE, "a") as f:
        f.write(f"\n\n# ── Websearch Agent [{label}] — {ts} ──\n")
        for idea in ideas:
            f.write(idea + "\n")
    return len(ideas)

def extract_ideas(content, title, source_type, is_youtube=False, is_pdf=False):
    if len(content) < 80: return []
    prompt_tmpl = YOUTUBE_PROMPT if is_youtube else PDF_EXTRACT_PROMPT if is_pdf else EXTRACT_PROMPT
    try:
        prompt = prompt_tmpl.format(source_type=source_type, title=title[:100], content=content[:4000])
    except:
        prompt = EXTRACT_PROMPT.format(source_type=source_type, title=title[:100], content=content[:4000])
    try:
        raw = model.chat(
            system_prompt="Extract trading ideas. Return only valid JSON arrays.",
            user_prompt=prompt,
            max_tokens=1500
        )
        raw   = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        ideas = json.loads(raw)
        return [i.strip() for i in ideas if isinstance(i, str) and len(i) > 20]
    except RuntimeError as e:
        # All LLM providers failed
        err = str(e)
        if "All LLM providers failed" in err:
            print(f"        ❌ All AI providers failed — check API credits")
            print(f"           DeepSeek: platform.deepseek.com/usage")
            print(f"           Groq:     console.groq.com (free tier)")
            print(f"           Gemini:   aistudio.google.com (free tier)")
        else:
            print(f"        ⚠️  LLM error: {err[:100]}")
        return []
    except json.JSONDecodeError:
        # LLM returned non-JSON — try to extract manually
        if raw and len(raw) > 20:
            lines = [l.strip().strip('"').strip("'") for l in raw.split('\n')
                     if l.strip() and len(l.strip()) > 30
                     and any(w in l.lower() for w in ['long','short','when','exit'])]
            return lines[:8]
        return []
    except Exception as e:
        print(f"        ⚠️  Extract error: {str(e)[:80]}")
        return []

def clean_html(html):
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>',   ' ', text, flags=re.DOTALL)
    text = re.sub(r'<nav[^>]*>.*?</nav>',        ' ', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

def tavily_search(query, num=5, with_content=False):
    """
    Tavily — #1 search API for AI agents.
    Built specifically for LLM/agent workflows.
    Free: 1,000 searches/month.
    Get key: https://app.tavily.com → API Keys
    Set in .env: TAVILY_API_KEY=tvly-...

    with_content=True: returns list of {url, title, content} dicts
                       so we skip the second fetch entirely.
    with_content=False: returns list of URLs only (for PDF/YT detection).
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return [] if not with_content else []
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key":              api_key,
                "query":                query,
                "search_depth":         "basic",
                "max_results":          num,
                "include_answer":       False,
                "include_raw_content":  with_content,
            },
            timeout=15
        )
        r.raise_for_status()
        data    = r.json()
        results = data.get("results", [])
        if with_content:
            # Return rich dicts with content already included
            out = []
            for item in results:
                if item.get("url"):
                    out.append({
                        "url":         item["url"],
                        "title":       item.get("title", item["url"]),
                        "content":     item.get("raw_content") or item.get("content", ""),
                        "source_type": "article",
                    })
            return out
        else:
            return [item["url"] for item in results if item.get("url")]
    except Exception:
        return []


def serper_search(query, num=5):
    """
    Serper — Google results via API.
    Free: 2,500 searches/month.
    Get key: https://serper.dev → Dashboard
    Set in .env: SERPER_API_KEY=...
    """
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        return []
    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=15
        )
        r.raise_for_status()
        data    = r.json()
        organic = data.get("organic", [])
        return [item["link"] for item in organic if item.get("link")]
    except Exception:
        return []


def brave_search(query, num=5):
    """
    Brave Search — independent index.
    Free: 2,000 searches/month.
    Get key: https://api.search.brave.com → Dashboard
    Set in .env: BRAVE_API_KEY=...
    """
    api_key = os.getenv("BRAVE_API_KEY", "")
    if not api_key:
        return []
    try:
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json",
                     "Accept-Encoding": "gzip",
                     "X-Subscription-Token": api_key},
            params={"q": query, "count": num},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        return [item["url"] for item in data.get("web", {}).get("results", [])
                if item.get("url")]
    except Exception:
        return []


def ddg_search(query, num=5):
    """DuckDuckGo scrape — no API key, least reliable."""
    for url in [
        f"https://lite.duckduckgo.com/lite/?q={requests.utils.quote(query)}",
        f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}&kl=us-en",
    ]:
        try:
            r     = requests.get(url, headers=HEADERS, timeout=12)
            links = []
            for pat in [
                r'href="(https://[^"&]{20,})"[^>]*class="[^"]*result[^"]*"',
                r'uddg=(https?[^&"]+)',
                r'<a[^>]+href="(https://(?!duckduckgo)[^"&]{20,})">',
            ]:
                found = [f for f in re.findall(pat, r.text)
                         if not any(s in f for s in ['duckduckgo','duck.com'])]
                if found:
                    links.extend(found)
                    break
            if links:
                seen = set(); unique = []
                for l in links:
                    if l not in seen: seen.add(l); unique.append(l)
                return unique[:num]
        except Exception:
            continue
    return []


def search_web(query, num=5):
    """
    1. Tavily  (TAVILY_API_KEY)  — best for AI agents
    2. Serper  (SERPER_API_KEY)  — real Google results
    3. Brave   (BRAVE_API_KEY)   — independent index
    DuckDuckGo removed — it blocks automated requests.
    """
    results = tavily_search(query, num)
    if results:
        return results

    results = serper_search(query, num)
    if results:
        return results

    results = brave_search(query, num)
    if results:
        return results

    print(f"      ⚠️  All search engines failed — check API keys in .env")
    return []


def is_pdf_url(url):
    return ".pdf" in url.lower() or "pdf" in urlparse(url).path.lower()

def is_youtube_url(url):
    return "youtube.com/watch" in url or "youtu.be/" in url



def fetch_article(url, max_chars=5000):
    """Fetch and clean text from a web article."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        m     = re.search(r'<title[^>]*>(.*?)</title>', r.text, re.DOTALL)
        title = re.sub(r'<[^>]+>', '', m.group(1)).strip() if m else url
        return {"title": title, "content": clean_html(r.text)[:max_chars],
                "source_type": "article", "url": url}
    except Exception:
        return {}


def fetch_youtube(url):
    """Extract title + description from a YouTube video page."""
    try:
        r    = requests.get(url, headers=HEADERS, timeout=12)
        text = r.text
        title = ""
        for pat in [r'"title":"(.*?)"', r'<title>(.*?)</title>']:
            m = re.search(pat, text)
            if m:
                title = re.sub(r'\\u[\da-fA-F]{4}',
                               lambda x: chr(int(x.group(0)[2:], 16)), m.group(1))
                title = title.replace("\\n", " ").strip()
                if title and "YouTube" not in title[:8]: break
        desc = ""
        m = re.search(r'"shortDescription":"(.*?)"(?=,")', text, re.DOTALL)
        if m:
            desc = m.group(1).replace("\\n", "\n").replace('\\"', '"')[:3000]
        if not title and not desc: return {}
        return {"title": title or url,
                "content": f"Title: {title}\n\nDescription:\n{desc}",
                "source_type": "YouTube video", "is_youtube": True, "url": url}
    except Exception:
        return {}


def fetch_pdf(url, max_chars=6000):
    """Download and extract text from a PDF URL."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "pdf" not in ct.lower() and ".pdf" not in url.lower(): return {}
        if len(r.content) < 1000: return {}
        text = ""
        try:
            import pdfplumber, io as _io
            with pdfplumber.open(_io.BytesIO(r.content)) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages[:30])
        except ImportError:
            try:
                from pypdf import PdfReader
                import io as _io
                reader = PdfReader(_io.BytesIO(r.content))
                text   = "\n".join(p.extract_text() or "" for p in reader.pages[:30])
            except ImportError:
                print("      ⚠️  Install pdfplumber: pip install pdfplumber")
                return {}
        if len(text.strip()) < 100: return {}
        title = urlparse(url).path.split("/")[-1].replace(".pdf", "")
        return {"title": title, "content": text[:max_chars],
                "source_type": "PDF document", "is_pdf": True, "url": url}
    except Exception:
        return {}

def run_websearch(num_queries=3, continuous=False):
    total_added = 0
    while True:
        print(f"\n🔍 Websearch Agent — Fully Automated")
        print(f"{'='*55}")
        existing       = load_existing_ideas()
        processed_urls = load_processed_urls()
        print(f"  Existing ideas : {len(existing)}")
        print(f"  URLs processed : {len(processed_urls)}\n")

        web_qs = random.sample(WEB_QUERIES, min(num_queries, len(WEB_QUERIES)))
        pdf_qs = random.sample(PDF_QUERIES, min(max(1, num_queries//2), len(PDF_QUERIES)))
        yt_qs  = random.sample(YOUTUBE_QUERIES, min(max(1, num_queries//2), len(YOUTUBE_QUERIES)))

        run_ideas = []

        # ── Articles ──────────────────────────────────────────
        print(f"  📰 ARTICLES ({len(web_qs)} queries)")
        for q in web_qs:
            print(f"    → {q[:60]}")

            # Tavily with content — get URL + text in one call
            if os.getenv("TAVILY_API_KEY"):
                rich = tavily_search(q, num=6, with_content=True)
                if rich:
                    print(f"      Found {len(rich)} results")
                    for src in rich:
                        url = src.get("url","")
                        if not url or url in processed_urls: continue
                        if is_pdf_url(url):
                            src2 = fetch_pdf(url)
                            if src2:
                                ideas = extract_ideas(src2["content"], src2["title"], src2["source_type"], is_pdf=True)
                                new   = [i for i in ideas if is_valid_idea(i, existing)]
                                if new:
                                    run_ideas.extend(new); existing.update(i.lower()[:50] for i in new)
                                    print(f"      ✅ PDF {url[:50]}: {len(new)} ideas")
                                save_processed_url(url)
                        elif is_youtube_url(url):
                            src2 = fetch_youtube(url)
                            if src2:
                                ideas = extract_ideas(src2["content"], src2["title"], src2["source_type"], is_youtube=True)
                                new   = [i for i in ideas if is_valid_idea(i, existing)]
                                if new:
                                    run_ideas.extend(new); existing.update(i.lower()[:50] for i in new)
                                    print(f"      ✅ YouTube {url[:50]}: {len(new)} ideas")
                                save_processed_url(url)
                        else:
                            content = src.get("content","")
                            title   = src.get("title", url)
                            if content:
                                ideas = extract_ideas(content, title, "article")
                                new   = [i for i in ideas if is_valid_idea(i, existing)]
                                if new:
                                    run_ideas.extend(new); existing.update(i.lower()[:50] for i in new)
                                    print(f"      ✅ {url[:55]}: {len(new)} ideas")
                                save_processed_url(url)
                            time.sleep(0.3)
                    time.sleep(1)
                    continue

            # Fallback to Serper URL-only
            urls = search_web(q, 6)
            urls = [u for u in urls if not is_pdf_url(u) and not is_youtube_url(u)]
            if not urls:
                print(f"    ⚠️  No URLs found for this query")
            for url in urls[:4]:
                if url in processed_urls: continue
                src = fetch_article(url)
                if src:
                    ideas = extract_ideas(src["content"], src["title"], src["source_type"])
                    new   = [i for i in ideas if is_valid_idea(i, existing)]
                    if new:
                        run_ideas.extend(new)
                        existing.update(i.lower()[:50] for i in new)
                        print(f"      ✅ {url[:55]}: {len(new)} ideas")
                # Only mark as processed if we got content
                # (don't block URL if search just returned no results)
                if src:
                    save_processed_url(url)
                time.sleep(0.5)
            time.sleep(1)

        # ── PDFs ──────────────────────────────────────────────
        print(f"\n  📄 PDFs ({len(pdf_qs)} queries)")
        for q in pdf_qs:
            print(f"    → {q[:60]}")
            # Search for PDF links specifically
            all_urls = search_web(q + " filetype:pdf", 8)
            pdf_urls = [u for u in all_urls if is_pdf_url(u)][:3]
            for url in pdf_urls:
                if url in processed_urls: continue
                print(f"      📥 {url[:60]}")
                src = fetch_pdf(url)
                if src:
                    ideas = extract_ideas(src["content"], src["title"],
                                         src["source_type"], is_pdf=True)
                    new   = [i for i in ideas if is_valid_idea(i, existing)]
                    if new:
                        run_ideas.extend(new)
                        existing.update(i.lower()[:50] for i in new)
                        print(f"      ✅ {len(new)} ideas from PDF")
                # Only mark processed if we got content
                if src:
                    save_processed_url(url)
                time.sleep(1)
            time.sleep(1)

        # ── YouTube ───────────────────────────────────────────
        print(f"\n  🎥 YOUTUBE ({len(yt_qs)} queries)")
        for q in yt_qs:
            print(f"    → {q[:60]}")
            all_urls = search_web(q, 8)
            yt_urls  = [u for u in all_urls if is_youtube_url(u)][:3]
            for url in yt_urls:
                if url in processed_urls: continue
                print(f"      📺 {url[:60]}")
                src = fetch_youtube(url)
                if src:
                    ideas = extract_ideas(src["content"], src["title"],
                                         src["source_type"], is_youtube=True)
                    new   = [i for i in ideas if is_valid_idea(i, existing)]
                    if new:
                        run_ideas.extend(new)
                        existing.update(i.lower()[:50] for i in new)
                        print(f"      ✅ {len(new)} ideas from video")
                # Only mark processed if we got content
                if src:
                    save_processed_url(url)
                time.sleep(1)
            time.sleep(1)

        # ── Save ──────────────────────────────────────────────
        print(f"\n{'='*55}")
        if run_ideas:
            added = save_ideas(run_ideas, f"auto-{num_queries}q")
            total_added += added
            print(f"  ✅ Added {added} new ideas this run")
            print(f"  📁 Total in ideas.txt: {len(existing)}")
            print(f"\n  Run backtester:")
            print(f"  python src/agents/rbi_parallel.py --market futures")
        else:
            print(f"  ℹ️  No new ideas — all URLs already processed")

        if not continuous: break
        print(f"\n  💤 Next search in 30 mins... (Ctrl+C to stop)")
        time.sleep(1800)

    return total_added


def reprocess_urls(n: int = 150):
    """
    Re-extract ideas from the first N processed URLs.
    Used when URLs were fetched but idea extraction failed (e.g. no API credits).
    Removes them from the log so the next run re-processes them.
    """
    if not LOG_FILE.exists():
        print("No websearch log found — nothing to reprocess.")
        return

    data = json.loads(LOG_FILE.read_text())
    all_urls = data.get("urls", [])

    if not all_urls:
        print("Log is empty — nothing to reprocess.")
        return

    # Take first N, keep the rest
    to_reprocess = all_urls[:n]
    keep         = all_urls[n:]

    print(f"\n🔄 Reprocessing {len(to_reprocess)} URLs")
    print(f"   (removing from log so they get re-extracted)")

    # Write back with those URLs removed
    data["urls"] = keep
    LOG_FILE.write_text(json.dumps(data, indent=2))

    # Now run extraction on those URLs directly
    existing = load_existing_ideas()
    all_new  = []
    skip     = ["youtube.com","youtu.be",".pdf"]

    print(f"\n  📰 Re-extracting from {len(to_reprocess)} URLs...\n")

    for i, url in enumerate(to_reprocess, 1):
        print(f"  [{i}/{len(to_reprocess)}] {url[:70]}")
        try:
            if is_youtube_url(url):
                src = fetch_youtube(url)
                if src:
                    ideas = extract_ideas(src["content"], src["title"],
                                          src["source_type"], is_youtube=True)
            elif is_pdf_url(url):
                src = fetch_pdf(url)
                if src:
                    ideas = extract_ideas(src["content"], src["title"],
                                          src["source_type"], is_pdf=True)
            else:
                src = fetch_article(url)
                if src:
                    ideas = extract_ideas(src["content"], src["title"],
                                          src["source_type"])
                else:
                    ideas = []

            new = [i for i in ideas if is_valid_idea(i, existing)]
            if new:
                all_new.extend(new)
                existing.update(i.lower()[:50] for i in new)
                print(f"    ✅ {len(new)} ideas extracted")
                save_processed_url(url)  # mark as done again
            else:
                print(f"    ○  No ideas")
                save_processed_url(url)
        except Exception as e:
            print(f"    ⚠️  Error: {str(e)[:60]}")

    if all_new:
        added = save_ideas(all_new, f"reprocess-{n}")
        print(f"\n✅ Added {added} new ideas to ideas.txt")
    else:
        print(f"\n  ℹ️  No new ideas extracted from reprocessed URLs")
        print(f"     Check that Claude/DeepSeek API credits are available")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="🔍 Websearch Agent — Fully Automated")
    p.add_argument("--queries",    type=int, default=3)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--reprocess",  type=int, default=0,
                   metavar="N",
                   help="Re-extract ideas from first N previously processed URLs")
    args = p.parse_args()

    if args.reprocess > 0:
        reprocess_urls(args.reprocess)
    else:
        run_websearch(num_queries=args.queries, continuous=args.continuous)