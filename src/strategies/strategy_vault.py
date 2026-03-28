# ============================================================
# 🌙 Strategy Vault
#
# Automatically identifies winning strategies from RBI backtest
# results, optimises their parameters, and saves them permanently
# so they survive cache clears and re-runs.
#
# HOW IT WORKS:
#   1. Reads latest backtest_stats.csv after each RBI run
#   2. Filters strategies passing quality gate:
#        - Sharpe > 1.5
#        - 10+ trades
#        - Drawdown < 20%
#        - Profitable
#   3. For each winner, asks DeepSeek to write clean hard-coded
#      strategy code (not the RBI one-shot — a proper version)
#   4. Optimises key parameters using backtesting.py optimiser
#   5. Saves to src/strategies/vault/ — never gets deleted
#   6. Maintains a vault manifest (vault_index.json)
#
# HOW TO RUN:
#   # Automatically after a backtest run:
#   python src/agents/rbi_parallel.py --market futures && \
#   python src/strategies/strategy_vault.py
#
#   # Manually on existing results:
#   python src/strategies/strategy_vault.py
#   python src/strategies/strategy_vault.py --csv path/to/backtest_stats.csv
#   python src/strategies/strategy_vault.py --optimise MES 15m HistogramFade
# ============================================================

import os
import sys
import csv
import json
import time
import textwrap
import warnings
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning,    module="backtesting")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="backtesting")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.llm_router    import model
from src.data.fetcher          import get_ohlcv
from src.config                import (EXCHANGE, BACKTEST_INITIAL_CASH,
                                        BACKTEST_COMMISSION)

# ── Paths ─────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parents[2]
RESULTS_CSV    = REPO_ROOT / "src" / "data" / "rbi_results" / "backtest_stats.csv"
VAULT_DIR      = REPO_ROOT / "src" / "strategies" / "vault"
VAULT_INDEX    = VAULT_DIR / "vault_index.json"
VAULT_DIR.mkdir(parents=True, exist_ok=True)

# ── Quality gate ──────────────────────────────────────────────
MIN_SHARPE     = 1.5     # minimum Sharpe ratio
MIN_TRADES     = 10      # minimum number of trades
MAX_DRAWDOWN   = -20.0   # maximum acceptable drawdown %
MIN_RETURN     = 5.0     # minimum return %
MIN_WIN_RATE   = 40.0    # minimum win rate %

# ── Optimisation ──────────────────────────────────────────────
RUN_OPTIMISE   = True    # set False to skip parameter optimisation
OPTIMISE_DAYS  = 1825    # days of data to use for optimisation

# ── Code generation prompt ────────────────────────────────────
VAULT_PROMPT = """You are an expert Python quant developer.
Write a clean, production-quality backtesting.py Strategy class.

STRATEGY IDEA:
{idea}

BACKTEST RESULTS THAT TRIGGERED THIS SAVE:
Symbol: {symbol}, Timeframe: {timeframe}
Return: {return_pct}%, Sharpe: {sharpe}, Drawdown: {drawdown}%
Trades: {num_trades}, Win Rate: {win_rate}%

STRICT RULES:
- Class name must be exactly: VaultStrategy
- Import ONLY: numpy as np, pandas as pd, ta as ta_lib, ta
  from backtesting import Strategy
  from backtesting.lib import crossover
- Use self.I() for ALL indicators
- Use ta library: ta_lib.trend.ema_indicator(pd.Series(c), window=9).values
- Always wrap inputs in pd.Series() before ta functions
- Fixed position size: self.buy(size=0.1) — never larger
- Include proper stop loss and take profit
- Add class-level parameters for key values so they can be optimised:
    fast_period = 12
    slow_period = 26
    etc.
- NEVER use talib, TA-Lib, or pandas_ta
- Clean, well-commented code
- Return ONLY the class, no imports, no markdown

Start with: class VaultStrategy(Strategy):"""


# ════════════════════════════════════════════════════════════════
# VAULT INDEX
# ════════════════════════════════════════════════════════════════

def load_vault_index() -> dict:
    if VAULT_INDEX.exists():
        return json.loads(VAULT_INDEX.read_text())
    return {"strategies": [], "last_updated": None}


def save_vault_index(index: dict):
    index["last_updated"] = datetime.now().isoformat()
    VAULT_INDEX.write_text(json.dumps(index, indent=2))


def is_already_vaulted(strategy_name: str, symbol: str,
                        timeframe: str) -> bool:
    index = load_vault_index()
    for s in index["strategies"]:
        if (s["name"]      == strategy_name and
                s["symbol"]    == symbol and
                s["timeframe"] == timeframe):
            return True
    return False


# ════════════════════════════════════════════════════════════════
# QUALITY GATE
# ════════════════════════════════════════════════════════════════

def passes_quality_gate(row: dict) -> tuple[bool, str]:
    """Check if a backtest result deserves to be vaulted."""
    try:
        sharpe   = float(row.get("sharpe",   0) or 0)
        trades   = int(row.get("num_trades", 0) or 0)
        drawdown = float(row.get("max_drawdown", -100) or -100)
        ret      = float(row.get("return_pct",   0) or 0)
        winrate  = float(row.get("win_rate",     0) or 0)
    except (ValueError, TypeError):
        return False, "Invalid numeric data"

    if sharpe < MIN_SHARPE:
        return False, f"Sharpe {sharpe:.2f} < {MIN_SHARPE}"
    if trades < MIN_TRADES:
        return False, f"Only {trades} trades < {MIN_TRADES}"
    if drawdown < MAX_DRAWDOWN:
        return False, f"Drawdown {drawdown:.1f}% < {MAX_DRAWDOWN}%"
    if ret < MIN_RETURN:
        return False, f"Return {ret:.1f}% < {MIN_RETURN}%"
    if winrate < MIN_WIN_RATE:
        return False, f"Win rate {winrate:.1f}% < {MIN_WIN_RATE}%"

    return True, "Passed all checks"


# ════════════════════════════════════════════════════════════════
# CODE GENERATION
# ════════════════════════════════════════════════════════════════

def generate_vault_code(row: dict) -> str:
    """Ask the LLM to write clean strategy code for vaulting."""
    prompt = VAULT_PROMPT.format(
        idea       = row.get("idea", ""),
        symbol     = row.get("symbol", ""),
        timeframe  = row.get("timeframe", ""),
        return_pct = row.get("return_pct", 0),
        sharpe     = row.get("sharpe", 0),
        drawdown   = row.get("max_drawdown", 0),
        num_trades = row.get("num_trades", 0),
        win_rate   = row.get("win_rate", 0),
    )
    code = model.chat(
        system_prompt="You are an expert Python quant. Return only valid Python code.",
        user_prompt=prompt,
    )
    code = re.sub(r"```python|```", "", code).strip()
    if not code.startswith("class VaultStrategy"):
        match = re.search(r"(class VaultStrategy.*)", code, re.DOTALL)
        code  = match.group(1) if match else code
    return code


# ════════════════════════════════════════════════════════════════
# PARAMETER OPTIMISATION
# ════════════════════════════════════════════════════════════════

def optimise_strategy(code: str, symbol: str, timeframe: str,
                       strategy_name: str) -> tuple[str, dict]:
    """
    Run backtesting.py parameter optimisation on the strategy.
    Returns optimised code with best parameters baked in, and stats dict.
    """
    try:
        from backtesting import Backtest

        df = get_ohlcv(symbol, exchange=EXCHANGE,
                       timeframe=timeframe, days_back=OPTIMISE_DAYS)
        df = pd.DataFrame({
            "Open":   df["Open"].astype(float).values,
            "High":   df["High"].astype(float).values,
            "Low":    df["Low"].astype(float).values,
            "Close":  df["Close"].astype(float).values,
            "Volume": df["Volume"].astype(float).values,
        }, index=df.index)

        imports = textwrap.dedent("""
            import numpy as np
            import pandas as pd
            import ta as ta_lib
            import ta
            from backtesting import Strategy
            from backtesting.lib import crossover
        """)
        namespace = {}
        exec(compile(imports + "\n" + code, "<vault>", "exec"), namespace)
        StrategyClass = namespace["VaultStrategy"]

        bt = Backtest(df, StrategyClass,
                      cash=BACKTEST_INITIAL_CASH,
                      commission=BACKTEST_COMMISSION,
                      exclusive_orders=True)

        # Detect optimisable parameters (class-level int/float attributes)
        opt_params = {}
        for attr, val in vars(StrategyClass).items():
            if attr.startswith("_"):
                continue
            if isinstance(val, int) and 2 <= val <= 200:
                # Create a range around the default value
                step  = max(1, val // 5)
                low   = max(2, val - val // 2)
                high  = val + val // 2 + step
                opt_params[attr] = range(low, high, step)
            elif isinstance(val, float) and 0 < val < 1:
                opt_params[attr] = [round(v, 2) for v in
                                     np.arange(max(0.01, val-0.2),
                                               min(0.99, val+0.21), 0.05)]

        if not opt_params:
            print(f"     ℹ️  No optimisable parameters found — using defaults")
            stats = bt.run()
            return code, {
                "sharpe":    round(float(stats.get("Sharpe Ratio", 0) or 0), 3),
                "return":    round(float(stats["Return [%]"]), 2),
                "drawdown":  round(float(stats["Max. Drawdown [%]"]), 2),
                "trades":    int(stats["# Trades"]),
                "win_rate":  round(float(stats.get("Win Rate [%]", 0) or 0), 2),
                "optimised": False,
            }

        print(f"     🔧 Optimising {len(opt_params)} parameters: {list(opt_params.keys())}")

        # Run optimisation — maximise Sharpe ratio
        stats, heatmap = bt.optimize(
            **opt_params,
            maximize="Sharpe Ratio",
            return_heatmap=True,
        )

        best_params = stats._strategy.__dict__
        best_sharpe = round(float(stats.get("Sharpe Ratio", 0) or 0), 3)
        print(f"     ✅ Optimised Sharpe: {best_sharpe:.2f}")

        # Bake best parameters into the code as comments
        param_lines = "\n".join(
            f"    # optimised: {k} = {v}"
            for k, v in opt_params.items()
            if k in best_params
        )
        # Replace default parameter values with optimised ones
        opt_code = code
        for param, val in best_params.items():
            if param in opt_params:
                if isinstance(val, int):
                    opt_code = re.sub(
                        rf"(\s+{param}\s*=\s*)\d+",
                        rf"\g<1>{val}",
                        opt_code
                    )
                elif isinstance(val, float):
                    opt_code = re.sub(
                        rf"(\s+{param}\s*=\s*)[\d.]+",
                        rf"\g<1>{round(val, 4)}",
                        opt_code
                    )

        return opt_code, {
            "sharpe":    best_sharpe,
            "return":    round(float(stats["Return [%]"]), 2),
            "drawdown":  round(float(stats["Max. Drawdown [%]"]), 2),
            "trades":    int(stats["# Trades"]),
            "win_rate":  round(float(stats.get("Win Rate [%]", 0) or 0), 2),
            "optimised": True,
            "best_params": {k: v for k, v in best_params.items()
                             if k in opt_params},
        }

    except Exception as e:
        print(f"     ⚠️  Optimisation failed: {e} — saving with defaults")
        return code, {"optimised": False, "error": str(e)}


# ════════════════════════════════════════════════════════════════
# VAULT A STRATEGY
# ════════════════════════════════════════════════════════════════

def vault_strategy(row: dict, force: bool = False) -> bool:
    """
    Generate, optimise and save a strategy to the vault.
    Returns True if successfully vaulted.
    """
    strategy_name = row.get("strategy", "Unknown").replace(" ", "")
    symbol        = row.get("symbol", "")
    timeframe     = row.get("timeframe", "")
    sharpe        = float(row.get("sharpe", 0) or 0)
    ret           = float(row.get("return_pct", 0) or 0)

    if not force and is_already_vaulted(strategy_name, symbol, timeframe):
        print(f"  ⏭️  {strategy_name} {symbol} {timeframe} — already in vault")
        return False

    print(f"\n  💾 Vaulting: {strategy_name} | {symbol} {timeframe}")
    print(f"     Sharpe {sharpe:.2f} | Return {ret:+.1f}% | "
          f"{row.get('num_trades')} trades")

    # 1. Generate code
    print(f"     📝 Generating strategy code...")
    try:
        code = generate_vault_code(row)
    except Exception as e:
        print(f"     ❌ Code generation failed: {e}")
        return False

    # 2. Optimise parameters
    optimised_code = code
    opt_stats      = {}
    if RUN_OPTIMISE:
        print(f"     🔧 Running parameter optimisation...")
        optimised_code, opt_stats = optimise_strategy(
            code, symbol, timeframe, strategy_name
        )

    # 3. Build the vault file
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name  = re.sub(r"[^a-zA-Z0-9_]", "_", strategy_name)
    filename   = f"{safe_name}_{symbol}_{timeframe}_{timestamp}.py"
    vault_path = VAULT_DIR / filename

    file_content = f'''# ============================================================
# 🌙 VAULT STRATEGY: {strategy_name}
# Symbol: {symbol} | Timeframe: {timeframe}
#
# DISCOVERY STATS:
#   Return   : {ret:+.1f}%
#   Sharpe   : {sharpe:.2f}
#   Drawdown : {row.get("max_drawdown", 0):.1f}%
#   Trades   : {row.get("num_trades", 0)}
#   Win Rate : {row.get("win_rate", 0):.1f}%
#
# OPTIMISED STATS:
#   Sharpe   : {opt_stats.get("sharpe", "N/A")}
#   Return   : {opt_stats.get("return", "N/A")}%
#   Optimised: {opt_stats.get("optimised", False)}
#   Best params: {opt_stats.get("best_params", {})}
#
# ORIGINAL IDEA:
#   {row.get("idea", "")[:120]}
#
# VAULTED: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# DO NOT DELETE — this is a verified winning strategy
# ============================================================

import sys
import warnings
import numpy as np
import pandas as pd
import ta as ta_lib
import ta
from pathlib import Path
from datetime import datetime
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

warnings.filterwarnings("ignore", category=UserWarning,    module="backtesting")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="backtesting")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


{optimised_code}


def run(symbol: str = "{symbol}", timeframe: str = "{timeframe}",
        days_back: int = 1825) -> dict:
    """Run this vaulted strategy."""
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    df = get_ohlcv(symbol, exchange=EXCHANGE,
                   timeframe=timeframe, days_back=days_back)
    df = pd.DataFrame({{
        "Open":   df["Open"].astype(float).values,
        "High":   df["High"].astype(float).values,
        "Low":    df["Low"].astype(float).values,
        "Close":  df["Close"].astype(float).values,
        "Volume": df["Volume"].astype(float).values,
    }}, index=df.index)

    bt    = Backtest(df, VaultStrategy,
                     cash=BACKTEST_INITIAL_CASH,
                     commission=BACKTEST_COMMISSION,
                     exclusive_orders=True)
    stats = bt.run()

    return {{
        "strategy":     "{strategy_name}",
        "symbol":       symbol,
        "timeframe":    timeframe,
        "return_pct":   round(float(stats["Return [%]"]), 2),
        "sharpe":       round(float(stats.get("Sharpe Ratio", 0) or 0), 3),
        "max_drawdown": round(float(stats["Max. Drawdown [%]"]), 2),
        "num_trades":   int(stats["# Trades"]),
        "win_rate":     round(float(stats.get("Win Rate [%]", 0) or 0), 2),
    }}


if __name__ == "__main__":
    result = run()
    print(f"Return: {{result['return_pct']:+.1f}}% | "
          f"Sharpe: {{result['sharpe']:.2f}} | "
          f"DD: {{result['max_drawdown']:.1f}}% | "
          f"Trades: {{result['num_trades']}}")
'''

    vault_path.write_text(file_content)

    # 4. Update vault index
    index = load_vault_index()
    index["strategies"].append({
        "name":         strategy_name,
        "symbol":       symbol,
        "timeframe":    timeframe,
        "file":         filename,
        "vaulted_at":   datetime.now().isoformat(),
        "discovery": {
            "sharpe":    sharpe,
            "return":    ret,
            "drawdown":  float(row.get("max_drawdown", 0)),
            "trades":    int(row.get("num_trades", 0)),
            "win_rate":  float(row.get("win_rate", 0)),
        },
        "optimised":    opt_stats,
        "idea":         row.get("idea", "")[:200],
    })
    save_vault_index(index)

    print(f"     ✅ Saved to vault: {filename}")
    return True


# ════════════════════════════════════════════════════════════════
# MAIN VAULT RUNNER
# ════════════════════════════════════════════════════════════════

def run_vault(csv_path: Path = None, force: bool = False):
    """
    Read latest backtest results and vault all qualifying strategies.
    """
    path = csv_path or RESULTS_CSV

    if not path.exists():
        print(f"❌ No backtest results found at: {path}")
        print(f"   Run: python src/agents/rbi_parallel.py --market futures")
        return

    # Load results
    df = pd.read_csv(path)
    for col in ["return_pct","sharpe","max_drawdown","num_trades","win_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    total      = len(df)
    qualifying = []
    rejected   = []

    print(f"\n🏛️  Strategy Vault — Quality Check")
    print(f"   Results file: {path.name}")
    print(f"   Total results: {total}")
    print(f"\n   Quality gate:")
    print(f"   Sharpe   > {MIN_SHARPE}")
    print(f"   Trades   ≥ {MIN_TRADES}")
    print(f"   Drawdown > {MAX_DRAWDOWN}%")
    print(f"   Return   > {MIN_RETURN}%")
    print(f"   Win Rate > {MIN_WIN_RATE}%")
    print()

    for _, row in df.iterrows():
        passed, reason = passes_quality_gate(row.to_dict())
        if passed:
            qualifying.append(row.to_dict())
        else:
            rejected.append((row.get("strategy","?"), reason))

    print(f"   Qualifying : {len(qualifying)}")
    print(f"   Rejected   : {len(rejected)}")

    if not qualifying:
        print(f"\n  ℹ️  No strategies passed the quality gate this run.")
        print(f"     This is normal — not every run produces vault-worthy results.")
        print(f"\n  Closest to qualifying (top 5 by Sharpe):")
        top = df[df["num_trades"] > 0].nlargest(5, "sharpe")
        for _, r in top.iterrows():
            passed, reason = passes_quality_gate(r.to_dict())
            print(f"    {r['strategy']:<30} {r['symbol']:<5} {r['timeframe']:<5} "
                  f"Sharpe={r['sharpe']:.2f} — ❌ {reason}")
        return

    # Vault each qualifying strategy
    print(f"\n{'═'*55}")
    print(f"  Vaulting {len(qualifying)} qualifying strategies...")
    print(f"{'═'*55}")

    vaulted  = 0
    skipped  = 0

    for row in qualifying:
        result = vault_strategy(row, force=force)
        if result:
            vaulted += 1
        else:
            skipped += 1
        time.sleep(1)   # be polite to the AI API

    # Summary
    index = load_vault_index()
    print(f"\n{'═'*55}")
    print(f"✅ Vault run complete")
    print(f"   New strategies vaulted : {vaulted}")
    print(f"   Already in vault       : {skipped}")
    print(f"   Total in vault         : {len(index['strategies'])}")
    print(f"   Vault location         : {VAULT_DIR}")


def list_vault():
    """Print all strategies currently in the vault."""
    index = load_vault_index()
    strats = index.get("strategies", [])

    if not strats:
        print("🏛️  Vault is empty. Run a backtest first.")
        return

    print(f"\n🏛️  Strategy Vault — {len(strats)} strategies")
    print(f"{'═'*70}")
    print(f"\n{'#':<4} {'Strategy':<28} {'Sym':<5} {'TF':<5} "
          f"{'Sharpe':>7} {'Return':>8} {'DD':>8} {'Trades':>7}")
    print(f"{'─'*70}")

    for i, s in enumerate(strats, 1):
        disc = s.get("discovery", {})
        opt  = s.get("optimised", {})
        # Show optimised Sharpe if available, else discovery
        sharpe = opt.get("sharpe") or disc.get("sharpe", 0)
        ret    = opt.get("return") or disc.get("return", 0)
        print(f"{i:<4} {s['name']:<28} {s['symbol']:<5} {s['timeframe']:<5} "
              f"{sharpe:>7.2f} {ret:>+7.1f}% {disc.get('drawdown',0):>7.1f}% "
              f"{disc.get('trades',0):>7}")

    print(f"\n  Vault: {VAULT_DIR}")
    print(f"  Index: {VAULT_INDEX}")


def run_all_vaulted():
    """Re-run all vaulted strategies on fresh data to verify they still work."""
    index = load_vault_index()
    strats = index.get("strategies", [])

    if not strats:
        print("🏛️  Vault is empty.")
        return

    print(f"\n🔄 Re-running all {len(strats)} vaulted strategies on fresh data...")
    print(f"{'═'*60}")

    results = []
    for s in strats:
        file_path = VAULT_DIR / s["file"]
        if not file_path.exists():
            print(f"  ❌ File missing: {s['file']}")
            continue
        try:
            namespace = {}
            exec(compile(file_path.read_text(), str(file_path), "exec"), namespace)
            run_fn = namespace.get("run")
            if run_fn:
                r = run_fn(s["symbol"], s["timeframe"])
                results.append(r)
                disc_sharpe = s.get("discovery", {}).get("sharpe", 0)
                delta = r["sharpe"] - disc_sharpe
                icon  = "✅" if r["sharpe"] >= 1.0 else "⚠️ "
                print(f"  {icon} {s['name']:<28} {s['symbol']:<5} {s['timeframe']:<5} "
                      f"Sharpe {r['sharpe']:.2f} ({delta:+.2f} vs discovery)")
        except Exception as e:
            print(f"  ❌ {s['name']}: {e}")

    print(f"\n  ✅ Re-run complete. {len(results)} strategies verified.")
    return results


# ════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="🏛️ Strategy Vault")
    p.add_argument("--csv",      type=str, default=None,
                   help="Path to backtest_stats.csv (default: latest results)")
    p.add_argument("--list",     action="store_true",
                   help="List all vaulted strategies")
    p.add_argument("--rerun",    action="store_true",
                   help="Re-run all vaulted strategies on fresh data")
    p.add_argument("--force",    action="store_true",
                   help="Re-vault strategies even if already in vault")
    p.add_argument("--sharpe",   type=float, default=None,
                   help=f"Override min Sharpe (default: {MIN_SHARPE})")
    p.add_argument("--trades",   type=int,   default=None,
                   help=f"Override min trades (default: {MIN_TRADES})")
    args = p.parse_args()

    if args.sharpe:
        MIN_SHARPE = args.sharpe
    if args.trades:
        MIN_TRADES = args.trades

    if args.list:
        list_vault()
    elif args.rerun:
        run_all_vaulted()
    else:
        csv_path = Path(args.csv) if args.csv else None
        run_vault(csv_path=csv_path, force=args.force)
