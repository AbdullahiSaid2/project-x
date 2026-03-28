# ============================================================
# 🌙 RBI Backtester Agent
# Research → Backtest → Implement
#
# HOW TO USE:
#   1. Add strategy ideas (plain text or YouTube/PDF URLs)
#      to  src/data/ideas.txt
#   2. Run:  python src/agents/rbi_agent.py
#   3. Results saved to src/data/rbi_results/
# ============================================================

import os
import re
import sys
import json
import hashlib
import textwrap
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
from backtesting import Backtest

# ── project imports ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config          import (BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION,
                                  HYPERLIQUID_TOKENS, COINBASE_TOKENS,
                                  EXCHANGE, DEFAULT_TIMEFRAME)
from src.models.llm_router import rbi_model as model
from src.data.fetcher    import get_ohlcv

# ── paths ────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[2]
IDEAS_FILE   = REPO_ROOT / "src" / "data" / "ideas.txt"
RESULTS_DIR  = REPO_ROOT / "src" / "data" / "rbi_results"
PROCESSED    = REPO_ROOT / "src" / "data" / "processed_ideas.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── which symbols to test ─────────────────────────────────────
TEST_SYMBOLS = (
    HYPERLIQUID_TOKENS[:3] if EXCHANGE == "hyperliquid" else COINBASE_TOKENS[:3]
)

# ── prompts ──────────────────────────────────────────────────
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
- Use ta library for indicators. Import like: import ta as ta_lib
- Example EMA: ta_lib.trend.ema_indicator(close, window=9)
- Example RSI: ta_lib.momentum.rsi(close, window=14)
- Example MACD: ta_lib.trend.macd(close), ta_lib.trend.macd_signal(close)
- size parameter in buy()/sell() must be a fraction between 0 and 1 (e.g. 0.95)
- Include stop-loss and take-profit on every trade
- Keep it simple — no complex ML, no external data calls
- Return ONLY the Python class, no imports outside the class, no markdown

Start your response with: class GeneratedStrategy(Strategy):"""

FIX_PROMPT = """The following backtesting.py Strategy class produced this error:

ERROR: {error}

CODE:
{code}

Fix the error. Return ONLY the corrected class, no markdown, starting with:
class GeneratedStrategy(Strategy):"""


# ═══════════════════════════════════════════════════════════════
class RBIAgent:

    def __init__(self):
        self.processed = self._load_processed()
        print("🌙 RBI Backtester Agent initialised")
        print(f"   Exchange : {EXCHANGE}")
        print(f"   Symbols  : {TEST_SYMBOLS}")
        print(f"   Ideas    : {IDEAS_FILE}")

    # ── persistence ──────────────────────────────────────────
    def _load_processed(self) -> dict:
        if PROCESSED.exists():
            return json.loads(PROCESSED.read_text())
        return {}

    def _save_processed(self):
        PROCESSED.write_text(json.dumps(self.processed, indent=2))

    def _idea_hash(self, idea: str) -> str:
        return hashlib.md5(idea.strip().encode()).hexdigest()[:12]

    # ── idea loading ─────────────────────────────────────────
    def load_ideas(self) -> list[str]:
        if not IDEAS_FILE.exists():
            IDEAS_FILE.parent.mkdir(parents=True, exist_ok=True)
            IDEAS_FILE.write_text(
                "# Add one trading idea per line (plain text, YouTube URL, or PDF URL)\n"
                "Buy when RSI is below 30 and price is above the 200 EMA\n"
                "MACD crossover with volume confirmation\n"
                "Bollinger Band squeeze breakout\n"
            )
            print(f"  📝 Created example ideas file at {IDEAS_FILE}")

        ideas = []
        for line in IDEAS_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ideas.append(line)
        return ideas

    # ── phase 1: research ────────────────────────────────────
    def research(self, idea: str) -> dict:
        print(f"\n  🔬 Researching: {idea[:60]}...")
        raw = model.chat(
            system_prompt="You are a quantitative trading researcher. Respond only with valid JSON.",
            user_prompt=RESEARCH_PROMPT.format(idea=idea),
        )
        # strip any accidental markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        spec = json.loads(raw)
        print(f"  📋 Strategy: {spec.get('name')} — {spec.get('description')}")
        return spec

    # ── phase 2: generate backtest code ─────────────────────
    def generate_code(self, spec: dict) -> str:
        print("  💻 Generating backtest code...")
        code = model.chat(
            system_prompt="You are an expert Python quant developer. Return only valid Python code.",
            user_prompt=BACKTEST_PROMPT.format(spec=json.dumps(spec, indent=2)),
        )
        code = re.sub(r"```python|```", "", code).strip()
        if not code.startswith("class GeneratedStrategy"):
            # try to extract the class
            match = re.search(r"(class GeneratedStrategy.*)", code, re.DOTALL)
            code = match.group(1) if match else code
        return code

    # ── phase 3: run backtest ────────────────────────────────
    def run_backtest(self, code: str, df: pd.DataFrame,
                     symbol: str, timeframe: str,
                     max_retries: int = 3) -> dict | None:
        imports = textwrap.dedent("""
            import numpy as np
            import pandas as pd
            import ta as ta_lib
        import ta
            from backtesting import Strategy
            from backtesting.lib import crossover
        """)
        full_code = imports + "\n" + code

        for attempt in range(1, max_retries + 1):
            try:
                namespace = {}
                exec(compile(full_code, "<strategy>", "exec"), namespace)
                StrategyClass = namespace["GeneratedStrategy"]

                bt = Backtest(
                    df,
                    StrategyClass,
                    cash=BACKTEST_INITIAL_CASH,
                    commission=BACKTEST_COMMISSION,
                    exclusive_orders=True,
                )
                stats = bt.run()

                return {
                    "symbol":        symbol,
                    "timeframe":     timeframe,
                    "return_pct":    round(float(stats["Return [%]"]), 2),
                    "buy_hold_pct":  round(float(stats["Buy & Hold Return [%]"]), 2),
                    "max_drawdown":  round(float(stats["Max. Drawdown [%]"]), 2),
                    "sharpe":        round(float(stats.get("Sharpe Ratio", 0) or 0), 3),
                    "num_trades":    int(stats["# Trades"]),
                    "win_rate":      round(float(stats.get("Win Rate [%]", 0) or 0), 2),
                }

            except Exception as e:
                err_msg = str(e)
                print(f"    ⚠️  Attempt {attempt} failed: {err_msg[:80]}")
                if attempt < max_retries:
                    print("    🔧 Asking DeepSeek to fix the code...")
                    fixed = model.chat(
                        system_prompt="You are an expert Python developer. Return only valid Python code.",
                        user_prompt=FIX_PROMPT.format(error=err_msg, code=code),
                    )
                    fixed = re.sub(r"```python|```", "", fixed).strip()
                    if "class GeneratedStrategy" in fixed:
                        code = fixed
                        full_code = imports + "\n" + code
        return None

    # ── save results ─────────────────────────────────────────
    def save_results(self, idea: str, spec: dict, all_stats: list):
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out_dir = RESULTS_DIR / today
        out_dir.mkdir(parents=True, exist_ok=True)

        name = spec.get("name", "Unknown")

        # per-strategy JSON
        result = {
            "idea":     idea,
            "spec":     spec,
            "results":  all_stats,
            "timestamp": datetime.now().isoformat(),
        }
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2))

        # append to master CSV
        csv_path = RESULTS_DIR / "backtest_stats.csv"
        rows = []
        for s in all_stats:
            rows.append({
                "strategy":     name,
                "idea":         idea[:80],
                **s,
                "date": today,
            })
        df = pd.DataFrame(rows)
        if csv_path.exists():
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

        # summary
        if all_stats:
            best = max(all_stats, key=lambda x: x["return_pct"])
            print(f"\n  🏆 Best result → {best['symbol']} {best['timeframe']}: "
                  f"{best['return_pct']}% return | "
                  f"{best['sharpe']} Sharpe | "
                  f"{best['num_trades']} trades")

    # ── main loop ────────────────────────────────────────────
    def run(self):
        ideas = self.load_ideas()
        print(f"\n🌙 Processing {len(ideas)} idea(s)...\n{'═'*60}")

        for idea in ideas:
            idea_hash = self._idea_hash(idea)
            if idea_hash in self.processed:
                print(f"  ⏭️  Skipping (already done): {idea[:50]}...")
                continue

            print(f"\n{'─'*60}")
            print(f"  💡 IDEA: {idea}")

            try:
                # Phase 1 — Research
                spec = self.research(idea)

                # Phase 2 — Generate code
                code = self.generate_code(spec)

                # Phase 3 — Backtest across symbols × timeframes
                all_stats = []
                tf = spec.get("timeframe", DEFAULT_TIMEFRAME)

                for symbol in TEST_SYMBOLS:
                    print(f"\n  📊 Testing {symbol} @ {tf}...")
                    try:
                        df = get_ohlcv(symbol, exchange=EXCHANGE, timeframe=tf, days_back=59)
                        stats = self.run_backtest(code, df, symbol, tf)
                        if stats:
                            all_stats.append(stats)
                            print(f"     Return: {stats['return_pct']}% | "
                                  f"Drawdown: {stats['max_drawdown']}% | "
                                  f"Trades: {stats['num_trades']}")
                    except Exception as e:
                        print(f"     ❌ Data error for {symbol}: {e}")

                # Save
                self.save_results(idea, spec, all_stats)
                self.processed[idea_hash] = {
                    "idea": idea,
                    "strategy": spec.get("name"),
                    "timestamp": datetime.now().isoformat(),
                }
                self._save_processed()

            except Exception as e:
                print(f"  ❌ Failed to process idea: {e}")
                traceback.print_exc()

        print(f"\n{'═'*60}")
        print(f"✅ Done! Results in: {RESULTS_DIR}")
        print(f"   Open backtest_stats.csv to review all results.")


# ── entrypoint ───────────────────────────────────────────────
if __name__ == "__main__":
    agent = RBIAgent()
    agent.run()
