# ============================================================
# RBI Parallel Backtester v2
#
# Strict mode:
# raw idea -> parameter-aware family/schema -> deterministic compiler
# -> smoke test -> parallel backtest sweep -> save results
#
# Upgrades in this version:
# - Adds 5m futures datasets
# - Prioritizes datasets by timeframe_hint and symbol mentioned in idea
# - Filters out undersampled "best" results
# - Labels low-trade strategies as under-sampled
# ============================================================

from __future__ import annotations

import csv
import json
import hashlib
import concurrent.futures
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import pandas as pd
from backtesting import Backtest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    BACKTEST_INITIAL_CASH,
    BACKTEST_COMMISSION,
    EXCHANGE,
)
from src.data.fetcher import get_ohlcv
from src.strategies.families.registry import heuristic_schema_from_idea

try:
    from src.strategies.families.compiler import compile_strategy_class, RUNTIME_IMPORT_BLOCK
except Exception:
    print("⚠️ Families module failed — using fallback compiler")

    def compile_strategy_class(*args, **kwargs):
        return """
class GeneratedStrategy(Strategy):
    def init(self):
        pass
    def next(self):
        return
"""
    RUNTIME_IMPORT_BLOCK = "from backtesting import Strategy"

from src.strategies.families.schema import schema_from_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
IDEAS_FILE = REPO_ROOT / "src" / "data" / "ideas.txt"
RESULTS_DIR = REPO_ROOT / "src" / "data" / "rbi_results"
PROCESSED = REPO_ROOT / "src" / "data" / "processed_ideas_v2.json"
SUMMARY_CSV = RESULTS_DIR / "backtest_stats_v2.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRADES_FOR_BEST = 20

CRYPTO_DATASETS = [
    ("BTC", "15m"), ("BTC", "1H"), ("BTC", "4H"),
    ("ETH", "15m"), ("ETH", "1H"), ("ETH", "4H"),
    ("SOL", "15m"), ("SOL", "1H"), ("SOL", "4H"),
    ("BNB", "1H"), ("BNB", "4H"),
    ("AVAX", "1H"), ("AVAX", "4H"),
    ("LINK", "1H"), ("LINK", "4H"),
    ("ARB", "15m"), ("ARB", "1H"),
    ("OP", "15m"), ("OP", "1H"),
    ("UNI", "1H"),
    ("PEPE", "15m"), ("PEPE", "1H"),
    ("WIF", "15m"), ("WIF", "1H"),
    ("FET", "1H"), ("FET", "4H"),
    ("RNDR", "1H"), ("TAO", "4H"),
]

FUTURES_DATASETS = [
    ("MNQ", "5m"), ("MNQ", "15m"), ("MNQ", "1H"), ("MNQ", "4H"), ("MNQ", "1D"),
    ("MES", "5m"), ("MES", "15m"), ("MES", "1H"), ("MES", "4H"), ("MES", "1D"),
    ("MYM", "5m"), ("MYM", "15m"), ("MYM", "1H"), ("MYM", "4H"), ("MYM", "1D"),
]

ALL_DATASETS = CRYPTO_DATASETS
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def idea_hash(idea: str) -> str:
    return hashlib.sha256(idea.strip().encode("utf-8")).hexdigest()[:16]


def load_processed() -> set[str]:
    if not PROCESSED.exists():
        return set()
    try:
        return set(json.loads(PROCESSED.read_text()))
    except Exception:
        return set()


def save_processed(processed: set[str]) -> None:
    PROCESSED.write_text(json.dumps(sorted(processed), indent=2))


def load_ideas() -> List[str]:
    if not IDEAS_FILE.exists():
        return []
    ideas = []
    for line in IDEAS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ideas.append(line)
    return ideas


def exec_strategy_code(code: str, class_name: str = "GeneratedStrategy"):
    namespace: Dict[str, Any] = {}
    compiled = compile(RUNTIME_IMPORT_BLOCK + "\n\n" + code, "<generated_strategy>", "exec")
    exec(compiled, namespace)
    return namespace[class_name]


def load_dataframe(symbol: str, timeframe: str, days_back: int = 1825) -> pd.DataFrame | None:
    df = get_ohlcv(symbol, exchange=EXCHANGE, timeframe=timeframe, days_back=days_back)
    if df is None or len(df) < 80:
        return None
    return pd.DataFrame(
        {
            "Open": df["Open"].astype(float).values,
            "High": df["High"].astype(float).values,
            "Low": df["Low"].astype(float).values,
            "Close": df["Close"].astype(float).values,
            "Volume": df["Volume"].astype(float).values,
        },
        index=df.index,
    )


def choose_smoke_dataset(schema_dict: Dict[str, Any]) -> Tuple[str, str]:
    desc = (schema_dict.get("description") or "").lower()
    tf = (schema_dict.get("timeframe_hint") or "15m").lower()

    if "mes" in desc:
        return ("MES", tf if tf in {"5m", "15m", "1h", "4h", "1d"} else "15m")
    if "mnq" in desc:
        return ("MNQ", tf if tf in {"5m", "15m", "1h", "4h", "1d"} else "15m")
    if "mym" in desc:
        return ("MYM", tf if tf in {"5m", "15m", "1h", "4h", "1d"} else "15m")

    family = (schema_dict.get("family") or "").lower()
    if family in {"ict_fvg", "ict_liquidity_sweep", "breakout"}:
        return ("MES", tf if tf in {"5m", "15m", "1h", "4h", "1d"} else "15m")

    return ("ETH", "15m")


def build_schema(idea: str) -> Dict[str, Any]:
    raw = heuristic_schema_from_idea(idea)
    schema = schema_from_dict(raw, source_idea=idea)
    return schema.to_dict()


def dataset_priority_key(dataset: Tuple[str, str], schema_dict: Dict[str, Any]) -> Tuple[int, int, int]:
    symbol, timeframe = dataset
    desc = (schema_dict.get("description") or "").lower()
    tf_hint = (schema_dict.get("timeframe_hint") or "15m").lower()

    symbol_score = 2
    if symbol.lower() in desc:
        symbol_score = 0
    elif any(x in desc for x in ["mes", "mnq", "mym"]):
        symbol_score = 1

    tf_order = {
        tf_hint: 0,
        "5m": 1,
        "15m": 2,
        "1h": 3,
        "4h": 4,
        "1d": 5,
    }
    tf_score = tf_order.get(timeframe.lower(), 9)

    market_score = 0
    if symbol in {"MES", "MNQ", "MYM"}:
        market_score = 0
    return (symbol_score, tf_score, market_score)


def order_datasets(schema_dict: Dict[str, Any], datasets: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    return sorted(datasets, key=lambda d: dataset_priority_key(d, schema_dict))


def smoke_test_strategy(code: str, dataset: Tuple[str, str]) -> Tuple[bool, str]:
    symbol, timeframe = dataset
    try:
        df = load_dataframe(symbol, timeframe, days_back=180)
        if df is None:
            return False, "No smoke-test data"

        StrategyClass = exec_strategy_code(code, "GeneratedStrategy")
        df = df.iloc[: min(len(df), 300)].copy()

        bt = Backtest(
            df,
            StrategyClass,
            cash=max(BACKTEST_INITIAL_CASH, 1_000_000),
            commission=BACKTEST_COMMISSION,
            exclusive_orders=True,
        )
        _ = bt.run()
        return True, "OK"
    except Exception as e:
        return False, str(e)


def run_single_backtest(args: Tuple[str, str, str, str]) -> Dict[str, Any] | None:
    symbol, timeframe, code, strategy_name = args
    try:
        safe_print(f"  📡 Fetching {symbol} {timeframe}...")
        df = load_dataframe(symbol, timeframe, days_back=1825)
        if df is None:
            return None

        StrategyClass = exec_strategy_code(code, "GeneratedStrategy")
        bt = Backtest(
            df,
            StrategyClass,
            cash=max(BACKTEST_INITIAL_CASH, 1_000_000),
            commission=BACKTEST_COMMISSION,
            exclusive_orders=True,
        )
        stats = bt.run()
        trades_df = getattr(stats, "_trades", pd.DataFrame())
        trades = len(trades_df) if trades_df is not None else 0

        return {
            "strategy": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "return_pct": float(stats.get("Return [%]", 0.0) or 0.0),
            "sharpe": float(stats.get("Sharpe Ratio", 0.0) or 0.0) if stats.get("Sharpe Ratio", 0.0) == stats.get("Sharpe Ratio", 0.0) else 0.0,
            "max_drawdown": float(stats.get("Max. Drawdown [%]", 0.0) or 0.0),
            "win_rate": float(stats.get("Win Rate [%]", 0.0) or 0.0) if stats.get("Win Rate [%]", 0.0) == stats.get("Win Rate [%]", 0.0) else 0.0,
            "num_trades": int(trades),
            "equity_final": float(stats.get("Equity Final [$]", 0.0) or 0.0),
            "buy_hold_return_pct": float(stats.get("Buy & Hold Return [%]", 0.0) or 0.0),
            "under_sampled": trades < MIN_TRADES_FOR_BEST,
            "error": "",
        }
    except Exception as e:
        return {
            "strategy": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "return_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
            "equity_final": 0.0,
            "buy_hold_return_pct": 0.0,
            "under_sampled": True,
            "error": str(e),
        }


def rank_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def score(r: Dict[str, Any]):
        if r.get("error"):
            return (-999999, -999999, -999999, -999999, -999999)
        trades = r.get("num_trades", 0) or 0
        min_trade_boost = min(trades, 50) / 50.0
        undersampled_penalty = -5 if trades < MIN_TRADES_FOR_BEST else 0
        return (
            undersampled_penalty + (r.get("sharpe", 0.0) or 0.0) + min_trade_boost,
            r.get("return_pct", 0.0) or 0.0,
            r.get("win_rate", 0.0) or 0.0,
            trades,
            -(abs(r.get("max_drawdown", 0.0) or 0.0)),
        )
    return sorted(results, key=score, reverse=True)


def select_best_result(valid: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not valid:
        return None

    eligible = [r for r in valid if (r.get("num_trades", 0) or 0) >= MIN_TRADES_FOR_BEST]
    if eligible:
        return rank_results(eligible)[0]

    return rank_results(valid)[0]


def summarize_result_set(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]

    if not valid:
        return {
            "valid_count": 0,
            "failed_count": len(failed),
            "best": None,
            "mean_sharpe": 0.0,
            "mean_return": 0.0,
            "mean_win_rate": 0.0,
            "dataset_breadth": 0,
            "eligible_count": 0,
        }

    mean_sharpe = sum(r["sharpe"] for r in valid) / len(valid)
    mean_return = sum(r["return_pct"] for r in valid) / len(valid)
    mean_win = sum(r["win_rate"] for r in valid) / len(valid)
    positive_datasets = sum(1 for r in valid if r["return_pct"] > 0)
    eligible_count = sum(1 for r in valid if (r.get("num_trades", 0) or 0) >= MIN_TRADES_FOR_BEST)

    return {
        "valid_count": len(valid),
        "failed_count": len(failed),
        "best": select_best_result(valid),
        "mean_sharpe": round(mean_sharpe, 4),
        "mean_return": round(mean_return, 4),
        "mean_win_rate": round(mean_win, 4),
        "dataset_breadth": positive_datasets,
        "eligible_count": eligible_count,
    }


def write_summary_csv(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "strategy", "symbol", "timeframe", "return_pct", "sharpe", "max_drawdown",
        "win_rate", "num_trades", "equity_final", "buy_hold_return_pct", "under_sampled", "error"
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


class ParallelRBIAgentV2:
    def __init__(self, max_workers: int = 12):
        self.max_workers = max_workers
        self.processed = load_processed()
        self.summary_rows: List[Dict[str, Any]] = []

    def process_idea(self, idea: str):
        print("\n" + "─" * 60)
        print(f"💡 IDEA: {idea}")
        print("🔬 Building schema...")

        idea_id = idea_hash(idea)

        try:
            schema_dict = build_schema(idea)
            strategy_name = schema_dict["name"]
            print(f"🧠 Family: {schema_dict['family']}")
            print(f"📝 Strategy: {strategy_name}")
            print(
                f"⚙️ Params: lookback={schema_dict['setup_params']['lookback']}, "
                f"retest={schema_dict['setup_params']['retest_required']}, "
                f"volume={schema_dict['setup_params']['volume_confirmation']}, "
                f"large_bar={schema_dict['setup_params']['large_bar_confirmation']}, "
                f"rr={schema_dict['risk_params']['tp_r_multiple']}, "
                f"direction={schema_dict['direction']}, "
                f"timeframe={schema_dict['timeframe_hint']}"
            )
        except Exception as e:
            print(f"❌ Schema build failed: {e}")
            self.processed.add(idea_id)
            save_processed(self.processed)
            return

        try:
            code = compile_strategy_class(schema_from_dict(schema_dict, source_idea=idea), class_name="GeneratedStrategy")
        except Exception as e:
            print(f"❌ Compile failed: {e}")
            self.processed.add(idea_id)
            save_processed(self.processed)
            return

        smoke_dataset = choose_smoke_dataset(schema_dict)
        ok, smoke_msg = smoke_test_strategy(code, smoke_dataset)
        if not ok:
            print(f"❌ Smoke test failed on {smoke_dataset[0]} {smoke_dataset[1]}: {smoke_msg}")
            self.processed.add(idea_id)
            save_processed(self.processed)
            return

        print(f"✅ Smoke test passed on {smoke_dataset[0]} {smoke_dataset[1]}")

        ordered = order_datasets(schema_dict, ALL_DATASETS)
        print(f"⚙️ Running parallel backtests across {len(ordered)} datasets...")

        jobs = [(symbol, timeframe, code, strategy_name) for symbol, timeframe in ordered]
        results: List[Dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(run_single_backtest, job) for job in jobs]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    result = fut.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    safe_print(f"⚠️ Worker failed: {e}")

        summary = summarize_result_set(results)

        print(f"\n⏱️ Completed {summary['valid_count']} successful backtests")
        if summary["best"]:
            best = summary["best"]
            sample_note = " | under-sampled" if best.get("under_sampled") else ""
            print(
                f"⭐ BEST: {best['symbol']} {best['timeframe']} | "
                f"Sharpe {best['sharpe']:.2f} | Return {best['return_pct']:.1f}% | "
                f"WR {best['win_rate']:.1f}% | Trades {best['num_trades']}{sample_note}"
            )
        print(f"📊 Eligible results (>= {MIN_TRADES_FOR_BEST} trades): {summary['eligible_count']}")

        artifact = {
            "idea": idea,
            "schema": schema_dict,
            "compiled_code": code,
            "summary": summary,
            "results": results,
            "created_at_utc": datetime.utcnow().isoformat(),
        }

        out_path = RESULTS_DIR / f"{idea_id}_{strategy_name.replace(' ', '_')}.json"
        out_path.write_text(json.dumps(artifact, indent=2))
        print(f"💾 Saved: {out_path}")

        self.summary_rows.extend(results)
        write_summary_csv(self.summary_rows)

        self.processed.add(idea_id)
        save_processed(self.processed)

    def run(self):
        ideas = load_ideas()
        if not ideas:
            print(f"❌ No ideas found in: {IDEAS_FILE}")
            return

        pending = [i for i in ideas if idea_hash(i) not in self.processed]
        print(f"{len(ideas)} total ideas | {len(pending)} pending | {len(ideas) - len(pending)} already processed")

        if not pending:
            print("✅ All ideas already processed!")
            return

        print(f"\n{'═' * 60}")
        for idea in pending:
            self.process_idea(idea)
        print(f"\n{'═' * 60}")
        print(f"✅ All done! Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="⚡ RBI Parallel Backtester v2")
    parser.add_argument("--workers", type=int, default=12, help="Parallel worker threads")
    parser.add_argument("--idea", type=str, default=None, help="Test a single idea directly")
    parser.add_argument("--datasets", type=int, default=None, help="Limit number of datasets")
    parser.add_argument(
        "--market",
        type=str,
        default="crypto",
        choices=["crypto", "futures", "all"],
        help="Market to backtest: crypto | futures | all",
    )
    args = parser.parse_args()

    if args.market == "futures":
        ALL_DATASETS[:] = FUTURES_DATASETS
        print("📈 Market: FUTURES")
    elif args.market == "all":
        ALL_DATASETS[:] = CRYPTO_DATASETS + FUTURES_DATASETS
        print(f"📈 Market: ALL ({len(ALL_DATASETS)} datasets)")
    else:
        ALL_DATASETS[:] = CRYPTO_DATASETS
        print("📈 Market: CRYPTO")

    if args.datasets:
        ALL_DATASETS[:] = ALL_DATASETS[: args.datasets]

    agent = ParallelRBIAgentV2(max_workers=args.workers)

    if args.idea:
        agent.process_idea(args.idea)
    else:
        agent.run()