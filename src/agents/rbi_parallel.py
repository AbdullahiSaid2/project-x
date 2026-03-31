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
from src.models.llm_router import rbi_model as model
from src.data.fetcher  import get_ohlcv

# ── paths ─────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
IDEAS_FILE  = REPO_ROOT / "src" / "data" / "ideas.txt"
RESULTS_DIR = REPO_ROOT / "src" / "data" / "rbi_results"
PROCESSED   = REPO_ROOT / "src" / "data" / "processed_ideas.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset configurations ────────────────────────────────────
# Two separate market groups — run with --market crypto or --market futures

CRYPTO_DATASETS = [
    # Core — always included (deepest liquidity, best data)
    ("BTC",  "15m"), ("BTC",  "1H"), ("BTC",  "4H"),
    ("ETH",  "15m"), ("ETH",  "1H"), ("ETH",  "4H"),
    ("SOL",  "15m"), ("SOL",  "1H"), ("SOL",  "4H"),
    # Large cap alts
    ("BNB",  "1H"),  ("BNB",  "4H"),
    ("AVAX", "1H"),  ("AVAX", "4H"),
    ("LINK", "1H"),  ("LINK", "4H"),
    # DeFi / L2
    ("ARB",  "15m"), ("ARB",  "1H"),
    ("OP",   "15m"), ("OP",   "1H"),
    ("UNI",  "1H"),
    # High volatility (more signals)
    ("PEPE", "15m"), ("PEPE", "1H"),
    ("WIF",  "15m"), ("WIF",  "1H"),
    # AI tokens (trending sector)
    ("FET",  "1H"),  ("FET",  "4H"),
    ("RNDR", "1H"),
    ("TAO",  "4H"),
]

FUTURES_DATASETS = [
    # CME Micro futures — all timeframes (Databento data)
    ("MNQ", "15m"), ("MNQ", "1H"), ("MNQ", "4H"), ("MNQ", "1D"),
    ("MES", "15m"), ("MES", "1H"), ("MES", "4H"), ("MES", "1D"),
    ("MYM", "15m"), ("MYM", "1H"), ("MYM", "4H"), ("MYM", "1D"),
]

ALL_DATASETS = CRYPTO_DATASETS   # default — overridden by --market flag

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

CRITICAL REQUIREMENTS — your spec must follow these or the backtest will be useless:
1. TRADE FREQUENCY: Entry conditions must fire at least 50-200 times over 5 years of data.
   Do NOT stack more than 2 conditions simultaneously — overly specific entries rarely fire.
   Good: "RSI crosses above 70" (fires often)
   Bad:  "RSI above 70 AND price above 200 EMA AND volume spike AND ATR expanding" (fires rarely)

2. ENTRY SIMPLICITY: Use at most 2 conditions for entry. Pick the most important signal only.
   The second condition can be a broad filter (e.g. RSI > 50 for direction bias).

3. EXIT CONDITIONS: Use ATR-based exits. Take profit = 2-3x ATR. Stop loss = 1-1.5x ATR.
   This ensures meaningful returns when trades win.

4. DIRECTION: If the idea is directional (e.g. short reversal), focus on that direction.
   Only include both long and short if the idea explicitly covers both.

Respond with ONLY a JSON object (no markdown fences) in this exact format:
{{
  "name": "TwoWordStrategyName",
  "description": "One sentence description",
  "entry_long":  "Precise entry condition for a LONG trade (max 2 conditions)",
  "entry_short": "Precise entry condition for a SHORT trade (or null if long-only)",
  "exit_long":   "ATR-based exit: take profit at 2-3x ATR above entry, stop at 1-1.5x ATR below",
  "exit_short":  "ATR-based exit: take profit at 2-3x ATR below entry, stop at 1-1.5x ATR above",
  "indicators":  ["list", "of", "indicators", "needed"],
  "timeframe":   "preferred timeframe e.g. 15m or 1H",
  "notes":       "Expected trade frequency over 5 years of futures data"
}}"""

BACKTEST_PROMPT = """You are an expert Python quant developer.
Write a backtesting.py Strategy class for the following specification.

SPECIFICATION:
{spec}

STRICT RULES — follow every one or the code will fail:
- Import ONLY from: backtesting, backtesting.lib, ta, numpy
- NEVER use talib, TA-Lib, or pandas_ta — use only the ta package
- Class name must be exactly: GeneratedStrategy
- Use self.I() for ALL indicator calculations
- CRITICAL: Always wrap close/high/low/open in pd.Series() before passing to ta:
    self.I(lambda c: ta_lib.momentum.rsi(pd.Series(c), window=14).values, self.data.Close)
- Use ta library like this inside self.I() lambdas:
    ta_lib.trend.ema_indicator(pd.Series(close), window=9).values
    ta_lib.momentum.rsi(pd.Series(close), window=14).values
    ta_lib.volatility.bollinger_hband(pd.Series(close), window=20).values
    ta_lib.trend.macd_diff(pd.Series(close)).values
    ta_lib.volatility.average_true_range(pd.Series(h), pd.Series(l), pd.Series(c), window=14).values

POSITION SIZING — CRITICAL:
- Always use fixed size: self.buy(size=0.1) or self.sell(size=0.1)
- NEVER use size > 0.2 — causes overflow
- NEVER calculate size dynamically

STOP LOSS — MANDATORY:
- Always include a stop loss on every trade to protect capital
- Use ATR-based stop: 1.0 to 2.0x ATR from entry
- Example: sl = price - 1.5 * atr  (for long)

EXIT STRATEGY — USE LOGICAL EXITS, NOT FIXED ATR TP:
- Use a LOGICAL exit condition based on the strategy's signal reversing
- Good exits: RSI crosses back through 50, indicator reverses, opposite signal fires
- Example for RSI strategy: exit long when RSI drops below 50
- Example for momentum: exit when the momentum indicator reverses
- This lets winners run naturally rather than capping them at a fixed target
- Only use a fixed TP as a safety net (e.g. 5-10x ATR) not as the primary exit
- The logical exit is what produced +200% returns — fixed ATR TP produces tiny returns

ENTRY CONDITIONS — KEEP SIMPLE:
- Maximum 2 entry conditions — do not stack 3+ conditions
- Entry must fire at least 30-50 times over 5 years (not ultra-rare setups)
- Simple crossover or threshold signals work better than complex multi-condition logic

TRADE MANAGEMENT:
- Close existing position before opening opposite direction

- No ML, no external data, no multi-timeframe
- Return ONLY the class body, no imports, no markdown

Start your response with: class GeneratedStrategy(Strategy):"""

FIX_PROMPT = """The following backtesting.py Strategy class produced this error:

ERROR: {error}

CODE:
{code}

Fix ALL errors. Key rules:
- NEVER use talib — only use the ta package
- All indicators must use self.I()
- ta usage: ta_lib.trend.ema_indicator(pd.Series(close), window=9).values
- Return ONLY the corrected class starting with:
class GeneratedStrategy(Strategy):
- CRITICAL: Always use small fixed size e.g. self.buy(size=0.1) — never size > 0.2"""


# ── Phase 3: Package check ────────────────────────────────────
# Scans generated code for known bad patterns and fixes them
# BEFORE running the backtest. No AI call — pure regex.
# Catches ~60% of failures for free.

def package_check(code: str) -> str:
    """
    Scan and fix known bad patterns in generated code.
    Applied immediately after code generation AND before each retry.

    Fixes:
      1. Wrong imports (talib/pandas_ta → ta)
      2. Float position sizes > 0.2 → 0.1
      3. Equity-based dynamic sizing → 0.1
      4. Missing pd.Series() wraps in ta calls
      5. Wrong class name → GeneratedStrategy
      6. Forbidden backtesting.lib imports
    """
    import re

    # Fix 1: Wrong library imports
    replacements = [
        (r'import talib\b',          'import ta as ta_lib'),
        (r'import TA-Lib\b',         'import ta as ta_lib'),
        (r'import pandas_ta\b',      'import ta as ta_lib'),
        (r'from talib\b',            'from ta'),
        (r'from pandas_ta import.*', 'import ta as ta_lib'),
        (r'talib\.',                  'ta_lib.'),
        (r'pandas_ta\.',              'ta_lib.'),
    ]
    for bad, good in replacements:
        code = re.sub(bad, good, code)

    # Fix 2: Float position sizes — any size > 0.2 → 0.1
    def fix_size(m):
        try:
            val = float(m.group(2))
            if val > 0.2:
                return f'self.{m.group(1)}(size=0.1'
        except ValueError:
            pass
        return m.group(0)
    code = re.sub(r'self\.(buy|sell)\(size=([0-9.]+)', fix_size, code)

    # Fix 3: Dynamic sizing patterns → fixed 0.1
    code = re.sub(r'size\s*=\s*self\.equity[^,)\n]*',   'size=0.1', code)
    code = re.sub(r'size\s*=\s*self\.position[^,)\n]*', 'size=0.1', code)
    code = re.sub(r'size\s*=\s*int\([^)]+\)',           'size=0.1', code)
    code = re.sub(r'size\s*=\s*round\([^)]+\)',         'size=0.1', code)

    # Fix 4: Missing pd.Series() around self.data.X in ta calls
    # ta_lib.xxx(self.data.Close) → ta_lib.xxx(pd.Series(self.data.Close))
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        # Only fix when inside a ta_lib call (not standalone references)
        code = re.sub(
            rf'(ta_lib\.\w+\.\w+\([^)]*?)self\.data\.{col}',
            rf'\1pd.Series(self.data.{col})',
            code
        )

    # Fix 5: Wrong class name
    if 'class GeneratedStrategy' not in code:
        code = re.sub(
            r'class\s+\w+\s*\(Strategy\)',
            'class GeneratedStrategy(Strategy)',
            code, count=1
        )

    # Fix 6: Forbidden backtesting.lib imports (only crossover is safe)
    code = re.sub(
        r'from backtesting\.lib import (?!crossover)(\w+)',
        r'# removed forbidden import: \1',
        code
    )

    return code


def validate_code(code: str) -> tuple:
    """
    Quick structural validation before running backtest.
    Returns (is_valid: bool, reason: str).
    """
    import ast

    required = [
        ('class GeneratedStrategy', "Missing GeneratedStrategy class"),
        ('def init',                "Missing init() method"),
        ('def next',                "Missing next() method"),
    ]
    for pattern, msg in required:
        if pattern not in code:
            return False, msg

    # Parse check
    imports = (
        "import numpy as np\nimport pandas as pd\n"
        "import ta as ta_lib\nimport ta\n"
        "from backtesting import Strategy\n"
        "from backtesting.lib import crossover\n"
    )
    try:
        ast.parse(imports + "\n" + code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    return True, "OK"


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

    # Phase 3: Immediately clean the generated code
    code = package_check(code)
    return code


# ── Single dataset backtest ───────────────────────────────────
def run_single_backtest(args: tuple) -> dict | None:
    """
    Run backtest for one (symbol, timeframe, code) combination.
    Returns result dict or None on failure.
    Called in parallel by thread pool.
    """
    symbol, timeframe, code, spec_name = args

    # Suppress noisy backtesting.py warnings that flood the terminal
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning,  module="backtesting")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="backtesting")
    warnings.filterwarnings("ignore", message=".*contingent SL/TP.*")
    warnings.filterwarnings("ignore", message=".*divide by zero.*")
    warnings.filterwarnings("ignore", message=".*invalid value.*")

    imports = textwrap.dedent("""
        import numpy as np
        import pandas as pd
        import ta as ta_lib
        import ta
        from backtesting import Strategy
        from backtesting.lib import crossover
    """)
    full_code = imports + "\n" + code

    max_retries = 10   # Moon Dev uses 10 iterations
    current_code = code

    for attempt in range(1, max_retries + 1):
        try:
            df = get_ohlcv(symbol, exchange=EXCHANGE,
                           timeframe=timeframe, days_back=1825)
            if df is None or len(df) < 50:
                return None

            # Sanitise — convert all columns to plain float64
            # Databento returns custom array types that backtesting.py
            # cannot handle. This ensures clean numpy compatibility.
            df = pd.DataFrame({
                "Open":   df["Open"].astype(float).values,
                "High":   df["High"].astype(float).values,
                "Low":    df["Low"].astype(float).values,
                "Close":  df["Close"].astype(float).values,
                "Volume": df["Volume"].astype(float).values,
            }, index=df.index)

            # Phase 3: Package check — fix known bad patterns
            # before running (no AI call needed)
            current_code = package_check(current_code)

            # Phase 3b: Validate code structure
            valid, err = validate_code(current_code)
            if not valid:
                raise ValueError(f"Code validation failed: {err}")

            namespace = {}
            exec(compile(imports + "\n" + current_code, "<strategy>", "exec"), namespace)
            StrategyClass = namespace["GeneratedStrategy"]

            bt    = Backtest(df, StrategyClass,
                             cash=BACKTEST_INITIAL_CASH,
                             commission=BACKTEST_COMMISSION,
                             exclusive_orders=True)

            # Timeout — kill runaway backtests after 90 seconds
            import signal as _signal
            def _timeout_handler(signum, frame):
                raise TimeoutError("Backtest exceeded 90 second limit")

            # Only use signal on main thread (signal only works on Unix main thread)
            import threading as _threading
            if _threading.current_thread() is _threading.main_thread():
                _signal.signal(_signal.SIGALRM, _timeout_handler)
                _signal.alarm(90)

            try:
                stats = bt.run()
            finally:
                if _threading.current_thread() is _threading.main_thread():
                    _signal.alarm(0)   # cancel alarm

            return_val  = float(stats["Return [%]"])
            drawdown    = float(stats["Max. Drawdown [%]"])
            sharpe      = float(stats.get("Sharpe Ratio", 0) or 0)
            num_trades  = int(stats["# Trades"])
            win_rate    = float(stats.get("Win Rate [%]", 0) or 0)

            # Sanity check — reject overflow results
            # Real strategies cannot return more than 100,000% over 5 years
            # or have Sharpe > 50. These are position sizing bugs in AI code.
            if abs(return_val) > 100_000 or abs(sharpe) > 50:
                safe_print(f"    ⚠️  {spec_name} | {symbol} {timeframe}: "
                           f"overflow result rejected (return={return_val:.2e}%) — "
                           f"likely position sizing bug in generated code")
                return None

            # Target return filter — don't save dead results
            # Moon Dev: SAVE_IF_OVER_RETURN = 1.0
            MIN_SAVE_RETURN = 0.0   # save anything profitable
            # (set to 1.0 to only save results with >1% return)

            result = {
                "symbol":       symbol,
                "timeframe":    timeframe,
                "return_pct":   round(return_val, 2),
                "buy_hold_pct": round(float(stats["Buy & Hold Return [%]"]), 2),
                "max_drawdown": round(drawdown, 2),
                "sharpe":       round(sharpe, 3),
                "num_trades":   num_trades,
                "win_rate":     round(win_rate, 2),
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

def load_vaulted_ideas() -> set:
    """
    Load ideas that are already in the strategy vault.
    RBI will skip these — no point regenerating code for
    ideas that already have a proven vault implementation.
    """
    vault_index = REPO_ROOT / "src" / "strategies" / "vault" / "vault_index.json"
    if not vault_index.exists():
        return set()

    try:
        data     = json.loads(vault_index.read_text())
        vaulted  = set()
        for s in data.get("strategies", []):
            idea = s.get("idea", "").strip()
            if idea:
                # Store first 60 chars — enough to match against ideas.txt
                vaulted.add(idea[:60].lower())
        return vaulted
    except Exception:
        return set()


def is_vaulted(idea: str, vaulted_ideas: set) -> bool:
    """Check if this idea is already represented in the vault."""
    idea_lower = idea[:60].lower()
    for v in vaulted_ideas:
        # Match if ideas share at least 80% of their start
        # (handles minor wording differences)
        if idea_lower[:40] == v[:40]:
            return True
    return False


def save_results(idea: str, spec: dict, all_stats: list):
    today   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        ideas         = load_ideas()
        vaulted_ideas = load_vaulted_ideas()
        if not ideas:
            print("❌ No ideas found in ideas.txt")
            return

        # Skip ideas already in vault OR already processed this session
        vaulted_skipped = [i for i in ideas if is_vaulted(i, vaulted_ideas)]
        pending = [i for i in ideas
                   if idea_hash(i) not in self.processed
                   and not is_vaulted(i, vaulted_ideas)]

        if vaulted_skipped:
            safe_print(f"  🏛️  Skipping {len(vaulted_skipped)} vaulted ideas "
                       f"(already in vault — no need to regenerate)")
        print(f"📋 {len(ideas)} total ideas | {len(pending)} pending | "
              f"{len(ideas)-len(pending)-len(vaulted_skipped)} already processed, "
              f"{len(vaulted_skipped)} vaulted)")

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
    parser.add_argument("--workers", type=int, default=18,
                        help="Parallel worker threads (default: 8)")
    parser.add_argument("--idea", type=str, default=None,
                        help="Test a single idea directly")
    parser.add_argument("--datasets", type=int, default=None,
                        help="Limit number of datasets")
    parser.add_argument("--market",   type=str, default="crypto",
                        choices=["crypto", "futures", "all"],
                        help="Market to backtest: crypto | futures | all (default: crypto)")
    args = parser.parse_args()

    # Select dataset group
    if args.market == "futures":
        ALL_DATASETS[:] = FUTURES_DATASETS
        print(f"📊 Market: FUTURES (MNQ, MES, MYM) — using Databento data")
    elif args.market == "all":
        ALL_DATASETS[:] = CRYPTO_DATASETS + FUTURES_DATASETS
        print(f"📊 Market: ALL ({len(ALL_DATASETS)} datasets)")
    else:
        ALL_DATASETS[:] = CRYPTO_DATASETS
        print(f"📊 Market: CRYPTO (BTC, ETH, SOL + alts)")

    if args.datasets:
        ALL_DATASETS[:] = ALL_DATASETS[:args.datasets]

    agent = ParallelRBIAgent(max_workers=args.workers)

    if args.idea:
        agent.process_idea(args.idea)
    else:
        agent.run()

    # Auto-run Strategy Vault after completing
    if not args.idea:
        try:
            from src.strategies.strategy_vault import run_vault
            print(f"\n{'='*55}")
            print(f"\U0001f3db  Running Strategy Vault...")
            print(f"   Saving strategies with Sharpe > 1.5 permanently")
            run_vault()
        except Exception as e:
            print(f"\n\u26a0\ufe0f  Strategy Vault skipped: {e}")
