# ============================================================
# 🌙 Parallel RBI Backtester
#
# Runs the full RBI pipeline across 25 datasets simultaneously
# using multi-threading. Moon Dev style — full parallel power.
#
# vs Standard RBI:
#   Standard  → 3 symbols × 1 timeframe = 3 tests per idea
#   Parallel  → 25 symbol/timeframe combos tested simultaneously
#   Speed     → ~8x faster, much more robust results
#
# HOW TO RUN:
#   python src/agents/rbi_parallel.py
#   python src/agents/rbi_parallel.py --workers 5
#   python src/agents/rbi_parallel.py --idea "RSI below 30 buy"
# ============================================================

import os
import re
import sys
import json
import hashlib
import textwrap
import traceback
import concurrent.futures
import threading
from pathlib import Path
from datetime import datetime

import pandas as pd
from backtesting import Backtest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config        import (BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION,
                                EXCHANGE, DEFAULT_TIMEFRAME)
from src.models.llm_router import model
from src.data.fetcher  import get_ohlcv

# ── paths ─────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
IDEAS_FILE  = REPO_ROOT / "src" / "data" / "ideas.txt"
RESULTS_DIR = REPO_ROOT / "src" / "data" / "rbi_results"
PROCESSED   = REPO_ROOT / "src" / "data" / "processed_ideas.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 25 dataset configurations ─────────────────────────────────
# symbol × timeframe combinations — covers full market breadth
ALL_DATASETS = [
    # BTC — multiple timeframes
    ("BTC", "15m"), ("BTC", "1H"), ("BTC", "4H"), ("BTC", "1D"),
    # ETH — multiple timeframes
    ("ETH", "15m"), ("ETH", "1H"), ("ETH", "4H"), ("ETH", "1D"),
    # SOL — multiple timeframes
    ("SOL", "15m"), ("SOL", "1H"), ("SOL", "4H"), ("SOL", "1D"),
    # Alt coins — 1H
    ("ARB",  "1H"), ("ARB",  "4H"),
    ("AVAX", "1H"), ("AVAX", "4H"),
    ("MATIC","1H"),
    ("LINK", "1H"),
    ("DOT",  "1H"),
    ("ADA",  "1H"),
    ("DOGE", "1H"),
    ("BNB",  "1H"),
    ("XRP",  "1H"),
    ("LTC",  "1H"),
    ("ATOM", "1H"),
]

# Thread safety for printing
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


# ── Prompts (same as rbi_agent.py) ───────────────────────────
RESEARCH_PROMPT = """You are an expert quantitative trading researcher.
Analyse the following trading idea and produce a concise strategy specification.

TRADING IDEA:
{idea}

Respond with ONLY a JSON object (no markdown fences) in this exact format:
{{
  "name": "TwoWordStrategyName",
  "description": "One sentence description",
  "entry_long":  "Precise entry condition for a LONG trade",
  "entry_short": "Precise entry condition for a SHORT trade (or null if long-only)",
  "exit_long":   "Exit condition for long",
  "exit_short":  "Exit condition for short (or null)",
  "indicators":  ["list", "of", "indicators", "needed"],
  "timeframe":   "preferred timeframe e.g. 1H",
  "notes":       "Any important caveats"
}}"""

BACKTEST_PROMPT = """You are an expert Python quant developer.
Write a backtesting.py Strategy class for the following specification.

SPECIFICATION:
{spec}

RULES:
- Import ONLY from: backtesting, backtesting.lib, backtesting.test, ta, numpy
- Class name must be exactly: GeneratedStrategy
- Use self.I() for all indicator calculations
- Use ta library for indicators: import ta as ta_lib, then ta_lib.trend.ema_indicator(close, window=9)
- size parameter in buy()/sell() must be a fraction between 0 and 1 (e.g. 0.95)
- Include stop-loss and take-profit on every trade
- Keep it simple — no complex ML, no external data calls, no multi-timeframe
- Return ONLY the Python class, no imports, no markdown

Start your response with: class GeneratedStrategy(Strategy):"""

FIX_PROMPT = """The following backtesting.py Strategy class produced this error:

ERROR: {error}

CODE:
{code}

Fix the error. Return ONLY the corrected class, no markdown, starting with:
class GeneratedStrategy(Strategy):"""


# ── Research phase ────────────────────────────────────────────
def research_strategy(idea: str) -> dict:
    """Ask the LLM to research the idea and return a spec dict."""
    raw = model.chat(
        system_prompt="You are a quantitative trading researcher. Return only valid JSON.",
        user_prompt=RESEARCH_PROMPT.format(idea=idea),
    )
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
    return json.loads(raw)


def generate_backtest_code(spec: dict) -> str:
    """Ask the LLM to write backtesting.py code for the spec."""
    code = model.chat(
        system_prompt="You are an expert Python quant developer. Return only valid Python code.",
        user_prompt=BACKTEST_PROMPT.format(spec=json.dumps(spec, indent=2)),
    )
    code = re.sub(r"```python|```", "", code).strip()
    if not code.startswith("class GeneratedStrategy"):
        match = re.search(r"(class GeneratedStrategy.*)", code, re.DOTALL)
        code  = match.group(1) if match else code
    return code


# ── Single dataset backtest ───────────────────────────────────
def run_single_backtest(args: tuple) -> dict | None:
    """
    Run backtest for one (symbol, timeframe, code) combination.
    Returns result dict or None on failure.
    Called in parallel by thread pool.
    """
    symbol, timeframe, code, spec_name = args

    imports = textwrap.dedent("""
        import numpy as np
        import pandas as pd
        import ta as ta_lib
        from backtesting import Strategy
        from backtesting.lib import crossover
    """)
    full_code = imports + "\n" + code

    max_retries = 2
    current_code = code

    for attempt in range(1, max_retries + 1):
        try:
            df = get_ohlcv(symbol, exchange=EXCHANGE,
                           timeframe=timeframe, days_back=59)
            if df is None or len(df) < 50:
                return None

            namespace = {}
            exec(compile(imports + "\n" + current_code, "<strategy>", "exec"), namespace)
            StrategyClass = namespace["GeneratedStrategy"]

            bt    = Backtest(df, StrategyClass,
                             cash=BACKTEST_INITIAL_CASH,
                             commission=BACKTEST_COMMISSION,
                             exclusive_orders=True)
            stats = bt.run()

            result = {
                "symbol":       symbol,
                "timeframe":    timeframe,
                "return_pct":   round(float(stats["Return [%]"]), 2),
                "buy_hold_pct": round(float(stats["Buy & Hold Return [%]"]), 2),
                "max_drawdown": round(float(stats["Max. Drawdown [%]"]), 2),
                "sharpe":       round(float(stats.get("Sharpe Ratio", 0) or 0), 3),
                "num_trades":   int(stats["# Trades"]),
                "win_rate":     round(float(stats.get("Win Rate [%]", 0) or 0), 2),
            }
            safe_print(f"    ✅ {spec_name} | {symbol} {timeframe}: "
                       f"return={result['return_pct']:+.1f}% "
                       f"sharpe={result['sharpe']:.2f} "
                       f"trades={result['num_trades']}")
            return result

        except Exception as e:
            err_msg = str(e)
            if attempt < max_retries:
                # Ask LLM to fix
                try:
                    fixed = model.chat(
                        system_prompt="You are a Python developer. Return only valid Python code.",
                        user_prompt=FIX_PROMPT.format(error=err_msg, code=current_code),
                    )
                    fixed = re.sub(r"```python|```", "", fixed).strip()
                    if "class GeneratedStrategy" in fixed:
                        current_code = fixed
                except Exception:
                    pass
            else:
                safe_print(f"    ❌ {spec_name} | {symbol} {timeframe}: failed — {err_msg[:60]}")
    return None


# ── Persistence ───────────────────────────────────────────────
def load_processed() -> dict:
    if PROCESSED.exists():
        return json.loads(PROCESSED.read_text())
    return {}

def save_processed(data: dict):
    PROCESSED.write_text(json.dumps(data, indent=2))

def idea_hash(idea: str) -> str:
    return hashlib.md5(idea.strip().encode()).hexdigest()[:12]

def load_ideas() -> list[str]:
    if not IDEAS_FILE.exists():
        return []
    ideas = []
    for line in IDEAS_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ideas.append(line)
    return ideas

def save_results(idea: str, spec: dict, all_stats: list):
    today   = datetime.now().strftime("%Y-%m-%d")
    out_dir = RESULTS_DIR / today
    out_dir.mkdir(parents=True, exist_ok=True)
    name    = spec.get("name", "Unknown")

    # Per-strategy JSON
    result = {"idea": idea, "spec": spec, "results": all_stats,
               "timestamp": datetime.now().isoformat()}
    (out_dir / f"{name}_parallel.json").write_text(json.dumps(result, indent=2))

    # Append to master CSV
    csv_path = RESULTS_DIR / "backtest_stats.csv"
    rows = []
    for s in all_stats:
        rows.append({"strategy": name, "idea": idea[:80], **s, "date": today})
    df = pd.DataFrame(rows)
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


# ── Main parallel engine ──────────────────────────────────────
class ParallelRBIAgent:

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.processed   = load_processed()
        print(f"\n🚀 Parallel RBI Backtester initialised")
        print(f"   Datasets : {len(ALL_DATASETS)} (vs 3 in standard RBI)")
        print(f"   Workers  : {max_workers} parallel threads")
        print(f"   Speedup  : ~{min(max_workers, len(ALL_DATASETS))}x faster")
        print(f"   LLM      : {model.available[0]} (with fallbacks: {model.available[1:]})\n")

    def process_idea(self, idea: str):
        h = idea_hash(idea)
        if h in self.processed:
            safe_print(f"  ⏭️  Skipping (already done): {idea[:50]}...")
            return

        safe_print(f"\n{'─'*60}")
        safe_print(f"  💡 IDEA: {idea}")

        # Phase 1 — Research (single call, sets the spec)
        safe_print(f"  🔬 Researching...")
        try:
            spec = research_strategy(idea)
            safe_print(f"  📋 Strategy: {spec.get('name')} — {spec.get('description','')}")
        except Exception as e:
            safe_print(f"  ❌ Research failed: {e}")
            return

        # Phase 2 — Generate code (single call)
        safe_print(f"  💻 Generating backtest code...")
        try:
            code = generate_backtest_code(spec)
        except Exception as e:
            safe_print(f"  ❌ Code generation failed: {e}")
            return

        spec_name = spec.get("name", "Unknown")

        # Phase 3 — Run all 25 datasets IN PARALLEL
        safe_print(f"\n  ⚡ Running {len(ALL_DATASETS)} parallel backtests...")
        t_start = datetime.now()

        # Build argument list
        tasks = [(sym, tf, code, spec_name) for sym, tf in ALL_DATASETS]

        all_stats = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(run_single_backtest, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    all_stats.append(result)

        elapsed = (datetime.now() - t_start).seconds
        safe_print(f"\n  ⏱️  Completed {len(all_stats)} successful backtests in {elapsed}s")

        # Summary
        if all_stats:
            profitable   = [s for s in all_stats if s["return_pct"] > 0]
            best         = max(all_stats, key=lambda x: x["sharpe"])
            avg_return   = sum(s["return_pct"] for s in all_stats) / len(all_stats)
            avg_sharpe   = sum(s["sharpe"] for s in all_stats) / len(all_stats)

            safe_print(f"\n  📊 Summary: {len(profitable)}/{len(all_stats)} profitable")
            safe_print(f"     Avg return  : {avg_return:+.1f}%")
            safe_print(f"     Avg Sharpe  : {avg_sharpe:.2f}")
            safe_print(f"     🏆 Best: {best['symbol']} {best['timeframe']} "
                       f"→ {best['return_pct']:+.1f}% | Sharpe {best['sharpe']:.2f}")

            save_results(idea, spec, all_stats)

        self.processed[h] = {
            "idea":      idea,
            "strategy":  spec.get("name"),
            "datasets":  len(all_stats),
            "timestamp": datetime.now().isoformat(),
        }
        save_processed(self.processed)

    def run(self):
        ideas = load_ideas()
        if not ideas:
            print("❌ No ideas found in ideas.txt")
            return

        pending = [i for i in ideas if idea_hash(i) not in self.processed]
        print(f"📋 {len(ideas)} total ideas | {len(pending)} pending | "
              f"{len(ideas)-len(pending)} already done")

        if not pending:
            print("✅ All ideas already processed!")
            return

        print(f"\n{'═'*60}")
        for idea in pending:
            self.process_idea(idea)

        print(f"\n{'═'*60}")
        print(f"✅ All done! Results in: {RESULTS_DIR}")
        model.print_status()


# ── entrypoint ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="⚡ Parallel RBI Backtester")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker threads (default: 8)")
    parser.add_argument("--idea", type=str, default=None,
                        help="Test a single idea directly")
    parser.add_argument("--datasets", type=int, default=25,
                        help="Number of datasets to use (default: 25, max: 25)")
    args = parser.parse_args()

    # Trim dataset list if requested
    global ALL_DATASETS
    ALL_DATASETS = ALL_DATASETS[:args.datasets]

    agent = ParallelRBIAgent(max_workers=args.workers)

    if args.idea:
        agent.process_idea(args.idea)
    else:
        agent.run()
