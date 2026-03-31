#!/usr/bin/env python3
# ============================================================
# 🌙 Vault Strategy from Backtest Results
#
# Vaults a strategy directly from the RBI results —
# reads the generated code from disk and creates a proper
# vault file without needing to re-run the backtest.
#
# USAGE:
#   # Vault by strategy name (finds best result automatically)
#   python src/agents/vault_strategy.py --name "Imbalance Reversal"
#
#   # Vault with specific symbol/timeframe
#   python src/agents/vault_strategy.py --name "Structure Break" --symbol TAO --tf 4H
#
#   # List all vaultable candidates from latest results
#   python src/agents/vault_strategy.py --list
#
#   # Vault multiple at once
#   python src/agents/vault_strategy.py --name "Structure Break" --name "Bollinger Breakout"
# ============================================================

import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT    = Path(__file__).resolve().parents[2]
RESULTS_DIR  = REPO_ROOT / "src" / "data" / "rbi_results"
VAULT_DIR    = REPO_ROOT / "src" / "strategies" / "vault"
VAULT_INDEX  = VAULT_DIR / "vault_index.json"

VAULT_DIR.mkdir(parents=True, exist_ok=True)


def load_vault_index() -> dict:
    if VAULT_INDEX.exists():
        try:
            return json.loads(VAULT_INDEX.read_text())
        except Exception:
            pass
    return {"strategies": []}


def save_vault_index(index: dict):
    VAULT_INDEX.write_text(json.dumps(index, indent=2))


def find_result_json(strategy_name: str,
                     symbol: str = None,
                     timeframe: str = None) -> dict | None:
    """
    Search all result JSON files for a strategy by name.
    Returns the best matching result dict.
    """
    if not RESULTS_DIR.exists():
        return None

    # Search all date folders
    candidates = []
    for json_file in sorted(RESULTS_DIR.rglob("*.json"), reverse=True):
        try:
            data = json.loads(json_file.read_text())
            spec = data.get("spec", {})
            name = spec.get("name", "")

            # Match by name (case-insensitive, partial ok)
            if strategy_name.lower() not in name.lower():
                continue

            results = data.get("results", [])
            for r in results:
                # Filter by symbol/timeframe if specified
                if symbol and r.get("symbol", "").upper() != symbol.upper():
                    continue
                if timeframe and r.get("timeframe", "") != timeframe:
                    continue

                sharpe = float(r.get("sharpe", 0) or 0)
                candidates.append({
                    "file":      json_file,
                    "data":      data,
                    "result":    r,
                    "sharpe":    sharpe,
                    "name":      name,
                })
        except Exception:
            continue

    if not candidates:
        return None

    # Return best by Sharpe
    return max(candidates, key=lambda x: x["sharpe"])


def find_code(strategy_name: str, result_file: Path) -> str:
    """
    Find the generated Python code for a strategy.
    Looks in:
    1. Same folder as result JSON — as a .py file
    2. Embedded in the JSON as "code" key
    3. Falls back to template
    """
    folder = result_file.parent
    name   = result_file.stem.replace("_parallel", "")

    # 1. Look for .py file in same folder
    for py_file in folder.glob("*.py"):
        if strategy_name.lower().replace(" ", "") in py_file.stem.lower().replace(" ", ""):
            code = py_file.read_text()
            if "class GeneratedStrategy" in code or "class VaultStrategy" in code:
                return code

    # 2. Check if embedded in JSON
    try:
        data = json.loads(result_file.read_text())
        code = data.get("code", "")
        if code and "class" in code:
            return code
    except Exception:
        pass

    return ""


def vault_strategy(strategy_name: str,
                   symbol: str = None,
                   timeframe: str = None,
                   force: bool = False) -> bool:
    """
    Vault a strategy from backtest results.
    Returns True if successful.
    """
    print(f"\n🏛️  Vaulting: {strategy_name}")
    if symbol:
        print(f"   Symbol: {symbol} | TF: {timeframe}")

    # Find the result
    match = find_result_json(strategy_name, symbol, timeframe)
    if not match:
        print(f"   ❌ No results found for '{strategy_name}'")
        print(f"      Make sure the backtest has run and results are in:")
        print(f"      {RESULTS_DIR}")
        return False

    result  = match["result"]
    data    = match["data"]
    name    = match["name"]
    sharpe  = match["sharpe"]
    sym     = result.get("symbol", symbol or "UNKNOWN")
    tf      = result.get("timeframe", timeframe or "UNKNOWN")
    ret     = float(result.get("return_pct", 0) or 0)
    dd      = float(result.get("max_drawdown", 0) or 0)
    trades  = int(result.get("num_trades", 0) or 0)
    wr      = float(result.get("win_rate", 0) or 0)
    idea    = data.get("idea", "")
    spec    = data.get("spec", {})

    print(f"   Found: {name} | {sym} {tf}")
    print(f"   Return={ret:+.1f}% | Sharpe={sharpe:.2f} | "
          f"DD={dd:.1f}% | Trades={trades} | Win={wr:.1f}%")

    # Check if already vaulted
    index = load_vault_index()
    existing_names = [s["name"] for s in index["strategies"]]
    if name in existing_names and not force:
        print(f"   ⚠️  Already vaulted. Use --force to overwrite.")
        return False

    # Get generated code
    code = find_code(strategy_name, match["file"])

    # Build vault file
    safe_name = re.sub(r'[^a-zA-Z0-9]', '', name)
    vault_file = VAULT_DIR / f"{safe_name}_{sym}_{tf}.py"

    if code and "class" in code:
        # Wrap the generated code as a proper vault strategy
        vault_code = f'''# ============================================================
# 🏛️  VAULT: {name}
# Symbol:    {sym}
# Timeframe: {tf}
# Vaulted:   {datetime.now().strftime("%Y-%m-%d")}
#
# BACKTEST RESULTS:
#   Return    : {ret:+.1f}%
#   Sharpe    : {sharpe:.2f}
#   Drawdown  : {dd:.1f}%
#   Trades    : {trades}
#   Win Rate  : {wr:.1f}%
#
# IDEA: {idea[:120]}
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

warnings.filterwarnings("ignore", category=UserWarning, module="backtesting")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="backtesting")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ── Generated Strategy Code (do not modify) ──────────────────
{code if "import" not in code[:50] else chr(10).join(
    l for l in code.split(chr(10))
    if not l.strip().startswith("import") and not l.strip().startswith("from")
)}


# ── Vault runner ──────────────────────────────────────────────
def run(symbol: str = "{sym}", timeframe: str = "{tf}",
        days_back: int = 1825) -> dict:
    from src.data.fetcher import get_ohlcv
    from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION

    print(f"\\n📊 {name} — {{symbol}} {{timeframe}} ({{days_back}} days)")

    df = get_ohlcv(symbol, exchange=EXCHANGE,
                   timeframe=timeframe, days_back=days_back)
    df = pd.DataFrame({{
        "Open":   df["Open"].astype(float).values,
        "High":   df["High"].astype(float).values,
        "Low":    df["Low"].astype(float).values,
        "Close":  df["Close"].astype(float).values,
        "Volume": df["Volume"].astype(float).values,
    }}, index=df.index)

    bt    = Backtest(df, GeneratedStrategy,
                     cash=BACKTEST_INITIAL_CASH,
                     commission=BACKTEST_COMMISSION,
                     exclusive_orders=True)
    stats = bt.run()
    result = {{
        "strategy":     "{name}",
        "symbol":       symbol, "timeframe": timeframe,
        "return_pct":   round(float(stats["Return [%]"]), 2),
        "max_drawdown": round(float(stats["Max. Drawdown [%]"]), 2),
        "sharpe":       round(float(stats.get("Sharpe Ratio", 0) or 0), 3),
        "num_trades":   int(stats["# Trades"]),
        "win_rate":     round(float(stats.get("Win Rate [%]", 0) or 0), 2),
        "date":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }}
    print(f"  Return   : {{result['return_pct']:>+8.1f}}%")
    print(f"  Sharpe   : {{result['sharpe']:>8.2f}}")
    print(f"  Drawdown : {{result['max_drawdown']:>8.1f}}%")
    print(f"  Trades   : {{result['num_trades']:>8}}")
    print(f"  Win Rate : {{result['win_rate']:>8.1f}}%")
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="{sym}")
    p.add_argument("--tf",     default="{tf}")
    p.add_argument("--days",   type=int, default=1825)
    args = p.parse_args()
    run(args.symbol, args.tf, args.days)
'''
    else:
        print(f"   ⚠️  No generated code found — saving spec-only vault file")
        print(f"      The strategy was run but code wasn't saved to disk.")
        print(f"      Future runs will save code automatically (rbi_parallel.py updated).")
        vault_code = f'''# ============================================================
# 🏛️  VAULT: {name}  (spec-only — code not recovered)
# Symbol:    {sym} | Timeframe: {tf}
# Vaulted:   {datetime.now().strftime("%Y-%m-%d")}
# Return: {ret:+.1f}% | Sharpe: {sharpe:.2f} | Trades: {trades}
# IDEA: {idea[:120]}
# ============================================================
# NOTE: The generated backtesting.py code was not saved during
# the original run. Re-generate by running:
#   rm src/data/processed_ideas.json
#   python src/agents/rbi_parallel.py --market all
# The updated rbi_parallel.py now saves code automatically.
# ============================================================
'''

    vault_file.write_text(vault_code)
    print(f"   ✅ Vault file written: {vault_file.name}")

    # Update vault index
    if name in existing_names:
        index["strategies"] = [s for s in index["strategies"] if s["name"] != name]

    index["strategies"].append({
        "name":         name,
        "symbol":       sym,
        "symbols":      [sym],
        "timeframe":    tf,
        "return_pct":   round(ret, 2),
        "max_drawdown": round(dd, 2),
        "sharpe":       round(sharpe, 3),
        "num_trades":   trades,
        "win_rate":     round(wr, 2),
        "idea":         idea,
        "description":  spec.get("description", ""),
        "vault_file":   vault_file.name,
        "date_added":   datetime.now().strftime("%Y-%m-%d"),
    })
    save_vault_index(index)
    print(f"   ✅ Vault index updated ({len(index['strategies'])} strategies total)")
    return True


def list_candidates(min_sharpe: float = 1.0,
                    min_trades: int = 10) -> list:
    """List vaultable candidates from latest results."""
    if not RESULTS_DIR.exists():
        print("No results directory found. Run the backtester first.")
        return []

    candidates = []
    seen = set()

    for json_file in sorted(RESULTS_DIR.rglob("*.json"), reverse=True):
        try:
            data    = json.loads(json_file.read_text())
            spec    = data.get("spec", {})
            name    = spec.get("name", "")
            results = data.get("results", [])

            for r in results:
                key    = f"{name}_{r.get('symbol')}_{r.get('timeframe')}"
                if key in seen: continue
                seen.add(key)

                sharpe = float(r.get("sharpe", 0) or 0)
                trades = int(r.get("num_trades", 0) or 0)
                ret    = float(r.get("return_pct", 0) or 0)

                if sharpe >= min_sharpe and trades >= min_trades and ret > 0:
                    candidates.append({
                        "name":      name,
                        "symbol":    r.get("symbol"),
                        "timeframe": r.get("timeframe"),
                        "sharpe":    sharpe,
                        "return":    ret,
                        "trades":    trades,
                        "win_rate":  float(r.get("win_rate", 0) or 0),
                        "dd":        float(r.get("max_drawdown", 0) or 0),
                        "file":      json_file,
                    })
        except Exception:
            continue

    candidates.sort(key=lambda x: x["sharpe"], reverse=True)
    return candidates


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="🏛️  Vault strategies from RBI backtest results"
    )
    p.add_argument("--name",   type=str,   action="append", default=[],
                   help="Strategy name to vault (can use multiple times)")
    p.add_argument("--symbol", type=str,   default=None)
    p.add_argument("--tf",     type=str,   default=None)
    p.add_argument("--list",   action="store_true",
                   help="List all vaultable candidates")
    p.add_argument("--min-sharpe", type=float, default=1.0)
    p.add_argument("--min-trades", type=int,   default=10)
    p.add_argument("--force",  action="store_true",
                   help="Overwrite existing vault entries")
    args = p.parse_args()

    if args.list or not args.name:
        candidates = list_candidates(args.min_sharpe, args.min_trades)
        if not candidates:
            print("No candidates found. Run backtester first.")
        else:
            print(f"\n🏛️  VAULTABLE CANDIDATES "
                  f"(Sharpe > {args.min_sharpe}, {args.min_trades}+ trades)\n")
            print(f"{'Strategy':<28} {'Sym':<6} {'TF':<5} "
                  f"{'Return':>8} {'Sharpe':>7} {'DD':>7} "
                  f"{'Trades':>7} {'Win%':>6}")
            print("-" * 75)
            for c in candidates:
                print(f"{c['name']:<28} {c['symbol']:<6} {c['timeframe']:<5} "
                      f"{c['return']:>+7.1f}% {c['sharpe']:>7.2f} "
                      f"{c['dd']:>6.1f}% {c['trades']:>7} {c['win_rate']:>5.1f}%")
            print(f"\nTo vault: python src/agents/vault_strategy.py --name \"Strategy Name\"")

    for name in args.name:
        vault_strategy(name, args.symbol, args.tf, args.force)
