# ============================================================
# Strategy Vault v2
#
# Deterministic vault compiler for RBI v2 artifacts.
#
# HOW IT WORKS:
# 1. Reads RBI v2 json result files from src/data/rbi_results/
# 2. Applies aggregate quality + robustness gate
# 3. Compiles deterministic VaultStrategy from schema/family
# 4. Optimizes class parameters using backtesting.py optimizer
# 5. Saves to src/strategies/vault/ and updates vault_index.json
# ============================================================

from __future__ import annotations

import json
import re
import textwrap
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from backtesting import Backtest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.fetcher import get_ohlcv
from src.config import EXCHANGE, BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION
from src.strategies.families.schema import schema_from_dict
from src.strategies.families.compiler import compile_strategy_class, RUNTIME_IMPORT_BLOCK

warnings.filterwarnings("ignore", category=UserWarning, module="backtesting")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="backtesting")

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "src" / "data" / "rbi_results"
VAULT_DIR = REPO_ROOT / "src" / "strategies" / "vault"
VAULT_INDEX = VAULT_DIR / "vault_index.json"
VAULT_DIR.mkdir(parents=True, exist_ok=True)

FUTURES_SYMBOLS = {"MES", "MNQ", "MYM", "ES", "NQ", "YM", "CL", "GC", "ZB"}

MIN_SHARPE_FUTURES = 1.3
MIN_RETURN_FUTURES = 4.0
MIN_SHARPE_CRYPTO = 1.0
MIN_RETURN_CRYPTO = 4.0

MIN_TRADES = 12
MAX_DRAWDOWN = -25.0
MIN_WIN_RATE = 35.0
MIN_DATASET_BREADTH = 2
OPTIMISE_DAYS = 1825
RUN_OPTIMISE = True


def get_thresholds(symbol: str) -> Tuple[float, float]:
    if symbol.upper() in FUTURES_SYMBOLS:
        return MIN_SHARPE_FUTURES, MIN_RETURN_FUTURES
    return MIN_SHARPE_CRYPTO, MIN_RETURN_CRYPTO


def load_vault_index() -> Dict[str, Any]:
    if VAULT_INDEX.exists():
        return json.loads(VAULT_INDEX.read_text())
    return {"strategies": [], "last_updated": None}


def save_vault_index(index: Dict[str, Any]):
    index["last_updated"] = datetime.now().isoformat()
    VAULT_INDEX.write_text(json.dumps(index, indent=2))


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_")
    return text[:80] if text else "strategy"


def is_already_vaulted(name: str, family: str) -> bool:
    idx = load_vault_index()
    for s in idx["strategies"]:
        if s.get("name") == name and s.get("family") == family:
            return True
    return False


def load_candidate_files() -> List[Path]:
    return sorted(
        [
            p for p in RESULTS_DIR.glob("*.json")
            if p.name != "vault_index.json"
        ]
    )


def summarize_candidate(payload: Dict[str, Any]) -> Dict[str, Any]:
    results = payload.get("results", [])
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return {
            "best": None,
            "valid_count": 0,
            "dataset_breadth": 0,
            "mean_sharpe": 0.0,
            "mean_return": 0.0,
            "mean_win_rate": 0.0,
            "mean_drawdown": 0.0,
        }

    valid_sorted = sorted(
        valid,
        key=lambda r: (
            r.get("sharpe", 0.0),
            r.get("return_pct", 0.0),
            r.get("win_rate", 0.0),
            -(abs(r.get("max_drawdown", 0.0))),
        ),
        reverse=True,
    )

    dataset_breadth = sum(1 for r in valid if (r.get("return_pct", 0.0) or 0.0) > 0)
    return {
        "best": valid_sorted[0],
        "valid_count": len(valid),
        "dataset_breadth": dataset_breadth,
        "mean_sharpe": float(sum(r["sharpe"] for r in valid) / len(valid)),
        "mean_return": float(sum(r["return_pct"] for r in valid) / len(valid)),
        "mean_win_rate": float(sum(r["win_rate"] for r in valid) / len(valid)),
        "mean_drawdown": float(sum(r["max_drawdown"] for r in valid) / len(valid)),
    }


def passes_quality_gate(payload: Dict[str, Any]) -> Tuple[bool, str]:
    summary = summarize_candidate(payload)
    best = summary["best"]
    if not best:
        return False, "No valid backtests"

    min_sharpe, min_return = get_thresholds(best["symbol"])

    if summary["dataset_breadth"] < MIN_DATASET_BREADTH:
        return False, f"Dataset breadth {summary['dataset_breadth']} < {MIN_DATASET_BREADTH}"

    if (best.get("sharpe", 0.0) or 0.0) < min_sharpe:
        return False, f"Sharpe {best.get('sharpe', 0.0):.2f} < {min_sharpe}"

    if (best.get("return_pct", 0.0) or 0.0) < min_return:
        return False, f"Return {best.get('return_pct', 0.0):.2f}% < {min_return}%"

    if (best.get("num_trades", 0) or 0) < MIN_TRADES:
        return False, f"Trades {best.get('num_trades', 0)} < {MIN_TRADES}"

    if (best.get("max_drawdown", 0.0) or 0.0) < MAX_DRAWDOWN:
        return False, f"Drawdown {best.get('max_drawdown', 0.0):.2f}% < {MAX_DRAWDOWN}%"

    if (best.get("win_rate", 0.0) or 0.0) < MIN_WIN_RATE:
        return False, f"Win rate {best.get('win_rate', 0.0):.2f}% < {MIN_WIN_RATE}%"

    return True, "Passed"


def compile_vault_code(payload: Dict[str, Any]) -> str:
    schema = schema_from_dict(payload["schema"], source_idea=payload.get("idea", ""))
    code = compile_strategy_class(schema, class_name="VaultStrategy")
    return code


def exec_vault_code(code: str):
    namespace: Dict[str, Any] = {}
    compiled = compile(RUNTIME_IMPORT_BLOCK + "\n\n" + code, "<vault_strategy>", "exec")
    exec(compiled, namespace)
    return namespace["VaultStrategy"]


def optimise_strategy(code: str, symbol: str, timeframe: str) -> Tuple[str, Dict[str, Any]]:
    try:
        df = get_ohlcv(symbol, exchange=EXCHANGE, timeframe=timeframe, days_back=OPTIMISE_DAYS)
        df = pd.DataFrame(
            {
                "Open": df["Open"].astype(float).values,
                "High": df["High"].astype(float).values,
                "Low": df["Low"].astype(float).values,
                "Close": df["Close"].astype(float).values,
                "Volume": df["Volume"].astype(float).values,
            },
            index=df.index,
        )

        StrategyClass = exec_vault_code(code)
        bt = Backtest(
            df,
            StrategyClass,
            cash=BACKTEST_INITIAL_CASH,
            commission=BACKTEST_COMMISSION,
            exclusive_orders=True,
        )

        opt_params = {}
        for attr, val in vars(StrategyClass).items():
            if attr.startswith("_"):
                continue
            if isinstance(val, int) and 2 <= val <= 200:
                step = max(1, val // 5)
                low = max(2, val - val // 2)
                high = val + val // 2 + step
                if high > low:
                    opt_params[attr] = range(low, high, step)
            elif isinstance(val, float) and 0 < val <= 5:
                vals = np.arange(max(0.05, val * 0.5), min(5.0, val * 1.5) + 0.001, max(0.05, val / 5))
                vals = [round(float(v), 3) for v in vals]
                if len(vals) > 1:
                    opt_params[attr] = vals

        if not opt_params:
            stats = bt.run()
            return code, {
                "optimised": False,
                "sharpe": round(float(stats.get("Sharpe Ratio", 0) or 0), 3),
                "return": round(float(stats.get("Return [%]", 0) or 0), 2),
                "drawdown": round(float(stats.get("Max. Drawdown [%]", 0) or 0), 2),
                "trades": int(stats.get("# Trades", 0) or 0),
                "win_rate": round(float(stats.get("Win Rate [%]", 0) or 0), 2),
            }

        stats, _ = bt.optimize(
            **opt_params,
            maximize="Sharpe Ratio",
            return_heatmap=True,
        )
        best_params = stats._strategy.__dict__

        opt_code = code
        for param, val in best_params.items():
            if param not in opt_params:
                continue
            if isinstance(val, int):
                opt_code = re.sub(rf"(\s+{param}\s*=\s*)\d+", rf"\g<1>{val}", opt_code)
            elif isinstance(val, float):
                opt_code = re.sub(rf"(\s+{param}\s*=\s*)[\d.]+", rf"\g<1>{round(val, 4)}", opt_code)

        return opt_code, {
            "optimised": True,
            "sharpe": round(float(stats.get("Sharpe Ratio", 0) or 0), 3),
            "return": round(float(stats.get("Return [%]", 0) or 0), 2),
            "drawdown": round(float(stats.get("Max. Drawdown [%]", 0) or 0), 2),
            "trades": int(stats.get("# Trades", 0) or 0),
            "win_rate": round(float(stats.get("Win Rate [%]", 0) or 0), 2),
            "best_params": {k: v for k, v in best_params.items() if k in opt_params},
        }

    except Exception as e:
        return code, {"optimised": False, "error": str(e)}


def vault_strategy(payload: Dict[str, Any], force: bool = False) -> bool:
    schema = payload["schema"]
    name = schema["name"].replace(" ", "")
    family = schema["family"]

    if not force and is_already_vaulted(name, family):
        print(f"⏭️ {name} ({family}) — already vaulted")
        return False

    summary = summarize_candidate(payload)
    best = summary["best"]
    if not best:
        print(f"❌ {name} — no valid backtests")
        return False

    print(f"\n🏛️ Vaulting: {name} | {family}")
    print(
        f"Best: {best['symbol']} {best['timeframe']} | "
        f"Sharpe {best['sharpe']:.2f} | Return {best['return_pct']:.1f}% | "
        f"Trades {best['num_trades']}"
    )

    try:
        code = compile_vault_code(payload)
    except Exception as e:
        print(f"❌ Compile failed: {e}")
        return False

    opt_code = code
    opt_stats = {}
    if RUN_OPTIMISE:
        print("⚙️ Optimizing parameters...")
        opt_code, opt_stats = optimise_strategy(code, best["symbol"], best["timeframe"])

    filename = f"{slugify(name)}_{family}.py"
    out_path = VAULT_DIR / filename

    file_text = textwrap.dedent(
        f"""\
        # Auto-generated by strategy_vault.py
        # Source idea: {payload.get("idea", "")}
        # Family: {family}
        # Best dataset: {best["symbol"]} {best["timeframe"]}
        # Summary: {json.dumps(summary)}
        # Optimization: {json.dumps(opt_stats, default=str)}

        {RUNTIME_IMPORT_BLOCK}

        {opt_code}
        """
    )
    out_path.write_text(file_text)

    index = load_vault_index()
    index["strategies"].append(
        {
            "name": name,
            "family": family,
            "filename": filename,
            "source_idea": payload.get("idea", ""),
            "symbol": best["symbol"],
            "timeframe": best["timeframe"],
            "summary": summary,
            "optimisation": opt_stats,
            "vaulted_at": datetime.now().isoformat(),
        }
    )
    save_vault_index(index)

    print(f"✅ Saved to {out_path}")
    return True


def run_vault(force: bool = False):
    files = load_candidate_files()
    if not files:
        print("❌ No RBI result files found")
        return

    vaulted = 0
    skipped = 0

    for fp in files:
        try:
            payload = json.loads(fp.read_text())
        except Exception as e:
            print(f"⚠️ Skipping {fp.name}: {e}")
            skipped += 1
            continue

        ok, reason = passes_quality_gate(payload)
        strategy_name = payload.get("schema", {}).get("name", fp.stem)

        if not ok:
            print(f"⏭️ {strategy_name}: {reason}")
            skipped += 1
            continue

        success = vault_strategy(payload, force=force)
        vaulted += int(success)
        skipped += int(not success)

    print(f"\n✅ Vault complete | vaulted: {vaulted} | skipped: {skipped}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="🏛️ Strategy Vault v2")
    parser.add_argument("--force", action="store_true", help="Force vaulting even if already vaulted")
    args = parser.parse_args()

    run_vault(force=args.force)