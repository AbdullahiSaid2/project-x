# ============================================================
# RBI Parallel Backtester v2
#
# Features:
# - Idea quality gate before schema build
# - Dual-mode scoring: discovery + faithful
# - Automatic rewrite-variant generation when discovery passes
#   but faithful fails
# - Separate promotion lanes:
#     1) faithful promotion
#     2) discovery promotion
# - Resilient smoke testing with fallback datasets
# - Discovery quality gate
# - Final idea classification:
#     reject
#     watchlist_discovery
#     watchlist_faithful
#     research_candidate
#     vault_candidate
# - Market-aware behavior for futures vs crypto
# - Futures sub-market routing:
#     * indices -> MNQ / MES / MYM
#     * oil     -> MCL
#     * gold    -> MGC
# - Auto-routing by ideas file:
#     * ideas_futures.txt / ideas_indices.txt -> futures indices
#     * ideas_oil.txt -> futures oil
#     * ideas_gold.txt -> futures gold
#     * ideas_crypto.txt -> crypto
#     * mixed files -> infer per idea / all
# - Trade-frequency penalties / hard overtrade filter
# - Expectancy gate
# - Timeframe bias
# - Regime annotations (trend + ATR expansion)
# - Futures session-aware filters
# ============================================================

from __future__ import annotations

import csv
import json
import hashlib
import concurrent.futures
import itertools
import math
import threading
import traceback
from pathlib import Path
from datetime import datetime, time
from typing import Dict, Any, List, Tuple, Optional

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
DEFAULT_IDEAS_FILE = REPO_ROOT / "src" / "data" / "ideas.txt"
RESULTS_DIR = REPO_ROOT / "src" / "data" / "rbi_results"
PROCESSED = REPO_ROOT / "src" / "data" / "processed_ideas_v2.json"
SUMMARY_CSV = RESULTS_DIR / "backtest_stats_v2.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRADES_FOR_BEST = 20
MIN_RELIABLE_TRADES = 50
MIN_RETURN_FOR_RESEARCH_PASS = 0.5
MIN_SHARPE_FOR_RESEARCH_PASS = 1.0

MIN_DISCOVERY_RETURN = 1.0
MIN_DISCOVERY_SHARPE = 1.2
MIN_DISCOVERY_EXPECTANCY_PROXY = 0.02
MIN_DISCOVERY_TRADES = 20

MAX_TRADES_SOFT = 500
MAX_TRADES_HARD = 1500
MAX_TRADES_15M = 300

CRYPTO_SYMBOLS = {
    "BTC", "ETH", "SOL", "BNB", "AVAX", "LINK", "ARB", "OP", "UNI",
    "PEPE", "WIF", "FET", "RNDR", "TAO"
}

INDEX_FUTURES_SYMBOLS = {"MES", "MNQ", "MYM"}
OIL_FUTURES_SYMBOLS = {"MCL"}
GOLD_FUTURES_SYMBOLS = {"MGC"}
FUTURES_SYMBOLS = INDEX_FUTURES_SYMBOLS | OIL_FUTURES_SYMBOLS | GOLD_FUTURES_SYMBOLS

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
    ("RNDR", "1H"), ("RNDR", "4H"),
    ("TAO", "1H"), ("TAO", "4H"),
]

INDEX_FUTURES_DATASETS = [
    ("MNQ", "5m"), ("MNQ", "15m"), ("MNQ", "1H"), ("MNQ", "4H"), ("MNQ", "1D"),
    ("MES", "5m"), ("MES", "15m"), ("MES", "1H"), ("MES", "4H"), ("MES", "1D"),
    ("MYM", "5m"), ("MYM", "15m"), ("MYM", "1H"), ("MYM", "4H"), ("MYM", "1D"),
]

OIL_FUTURES_DATASETS = [
    ("MCL", "5m"), ("MCL", "15m"), ("MCL", "1H"), ("MCL", "4H"), ("MCL", "1D"),
]

GOLD_FUTURES_DATASETS = [
    ("MGC", "5m"), ("MGC", "15m"), ("MGC", "1H"), ("MGC", "4H"), ("MGC", "1D"),
]

FUTURES_DATASETS = INDEX_FUTURES_DATASETS + OIL_FUTURES_DATASETS + GOLD_FUTURES_DATASETS
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


def load_ideas(ideas_file: Path) -> List[str]:
    if not ideas_file.exists():
        return []
    ideas = []
    for line in ideas_file.read_text(encoding="utf-8").splitlines():
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


def infer_market_from_schema(schema_dict: Dict[str, Any]) -> str:
    locked_symbol = str(schema_dict.get("locked_symbol") or "").upper()
    if locked_symbol in FUTURES_SYMBOLS:
        return "futures"
    if locked_symbol in CRYPTO_SYMBOLS:
        return "crypto"

    desc = (schema_dict.get("description") or "").upper()
    explicit = [s for s in FUTURES_SYMBOLS.union(CRYPTO_SYMBOLS) if s in desc]

    if any(s in FUTURES_SYMBOLS for s in explicit):
        return "futures"
    if any(s in CRYPTO_SYMBOLS for s in explicit):
        return "crypto"

    current_symbols = {s for s, _ in ALL_DATASETS}
    if current_symbols and current_symbols.issubset(CRYPTO_SYMBOLS):
        return "crypto"
    if current_symbols and current_symbols.issubset(FUTURES_SYMBOLS):
        return "futures"

    return "futures"


def infer_futures_submarket_from_symbol(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol in INDEX_FUTURES_SYMBOLS:
        return "indices"
    if symbol in OIL_FUTURES_SYMBOLS:
        return "oil"
    if symbol in GOLD_FUTURES_SYMBOLS:
        return "gold"
    return "indices"


def infer_market_from_idea_text(idea: str) -> str:
    text = idea.upper()
    if any(sym in text for sym in FUTURES_SYMBOLS):
        return "futures"
    if any(sym in text for sym in CRYPTO_SYMBOLS):
        return "crypto"

    futures_terms = [
        "NEW YORK", "RTH", "OPENING RANGE", "PREVIOUS DAY HIGH", "PREVIOUS DAY LOW",
        "FUTURES", "SESSION", "CRUDE", "GOLD", "OIL", "COMEX", "NYMEX"
    ]
    crypto_terms = [
        "BTC", "ETH", "SOL", "BNB", "AVAX", "LINK", "ARB", "OP", "UNI",
        "PEPE", "WIF", "FET", "RNDR", "TAO", "CRYPTO"
    ]

    if any(term in text for term in futures_terms):
        return "futures"
    if any(term in text for term in crypto_terms):
        return "crypto"

    return "futures"


def infer_futures_submarket_from_idea_text(idea: str) -> str:
    text = idea.upper()
    if any(sym in text for sym in OIL_FUTURES_SYMBOLS):
        return "oil"
    if any(sym in text for sym in GOLD_FUTURES_SYMBOLS):
        return "gold"
    if any(sym in text for sym in INDEX_FUTURES_SYMBOLS):
        return "indices"

    if any(term in text for term in ["CRUDE", "WTI", "OIL", "ENERGY"]):
        return "oil"
    if any(term in text for term in ["GOLD", "METALS", "COMEX", "XAU"]):
        return "gold"
    return "indices"


def infer_market_from_ideas_file(ideas_file: Path, ideas: List[str]) -> str:
    name = ideas_file.name.lower()
    if "crypto" in name:
        return "crypto"
    if any(x in name for x in ["futures", "indices", "oil", "gold"]):
        return "futures"
    if "mixed" in name:
        return "all"

    if not ideas:
        return "futures"

    markets = {infer_market_from_idea_text(i) for i in ideas}
    if markets == {"futures"}:
        return "futures"
    if markets == {"crypto"}:
        return "crypto"
    return "all"


def infer_futures_submarket_from_ideas_file(ideas_file: Path, ideas: List[str]) -> str:
    name = ideas_file.name.lower()
    if "oil" in name:
        return "oil"
    if "gold" in name:
        return "gold"
    if any(x in name for x in ["futures", "indices"]):
        return "indices"

    if not ideas:
        return "indices"

    submarkets = {infer_futures_submarket_from_idea_text(i) for i in ideas if infer_market_from_idea_text(i) == "futures"}
    if len(submarkets) == 1:
        return list(submarkets)[0]
    return "mixed"


def market_default_symbol(market: str) -> str:
    return "BTC" if market == "crypto" else "MES"


def market_default_timeframe() -> str:
    return "15m"


def market_neighbor_symbols(symbol: str, market: str) -> List[str]:
    symbol = symbol.upper()
    if market == "crypto":
        ordered = ["BTC", "ETH", "SOL", "BNB", "AVAX", "LINK", "ARB", "OP"]
    else:
        submarket = infer_futures_submarket_from_symbol(symbol)
        if submarket == "oil":
            ordered = ["MCL"]
        elif submarket == "gold":
            ordered = ["MGC"]
        else:
            ordered = ["MES", "MNQ", "MYM"]

    if symbol in ordered:
        rest = [x for x in ordered if x != symbol]
        return [symbol] + rest
    return ordered


def _explicit_symbols_from_desc(desc: str) -> List[str]:
    desc_up = desc.upper()
    ordered = [
        "MES", "MNQ", "MYM", "MCL", "MGC",
        "BTC", "ETH", "SOL", "BNB", "AVAX", "LINK",
        "ARB", "OP", "UNI", "PEPE", "WIF", "FET", "RNDR", "TAO"
    ]
    return [s for s in ordered if s in desc_up]


def evaluate_idea_quality(idea: str) -> Dict[str, Any]:
    text = idea.lower()
    score = 0
    reasons: List[str] = []

    if "stop" in text:
        score += 2
    else:
        reasons.append("missing stop")

    if "target" in text or "r:" in text or "2r" in text or "2.5r" in text or "3r" in text:
        score += 2
    else:
        reasons.append("missing target")

    if any(tf in text for tf in ["5m", "15m", "1h", "4h", "1d"]):
        score += 2
    else:
        reasons.append("missing timeframe")

    if any(x in text for x in ["ema", "trend", "bullish regime", "bearish regime", "rising", "falling"]):
        score += 2
    else:
        reasons.append("missing trend context")

    if any(x in text for x in ["atr", "volatility", "expanding", "compression", "session", "rth", "new york"]):
        score += 2
    else:
        reasons.append("missing volatility/regime/session context")

    if "no trade" in text:
        score += 2
    else:
        reasons.append("missing no-trade condition")

    if any(x in text for x in ["retest", "pullback", "reclaim", "fails", "holds"]):
        score += 1

    if any(x in text for x in ["within 2 bars", "within 3 bars", "within 4 bars", "within 5 bars"]):
        score += 1

    vague_terms = [
        "large candle", "strong move", "momentum", "with volume", "reversal",
        "breakout", "bullish", "bearish"
    ]
    vague_hits = sum(1 for term in vague_terms if term in text)
    if vague_hits >= 4:
        score -= 2
        reasons.append("too generic")

    weak_patterns = [
        "double top", "double bottom", "inside bar", "engulfing",
        "doji", "rsi divergence", "volume spike"
    ]
    if any(x in text for x in weak_patterns) and not any(x in text for x in ["ema", "atr", "no trade", "4h", "1h", "session", "rth"]):
        score -= 4
        reasons.append("retail-pattern-only")

    passed = score >= 7
    return {"score": score, "pass": passed, "reasons": reasons}


def load_dataframe(symbol: str, timeframe: str, days_back: int = 1825) -> pd.DataFrame | None:
    df = get_ohlcv(symbol, exchange=EXCHANGE, timeframe=timeframe, days_back=days_back)
    if df is None or len(df) < 80:
        return None

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            return None

    clean = pd.DataFrame(
        {
            "Open": df["Open"].astype(float).values,
            "High": df["High"].astype(float).values,
            "Low": df["Low"].astype(float).values,
            "Close": df["Close"].astype(float).values,
            "Volume": df["Volume"].astype(float).values,
        },
        index=df.index,
    ).dropna()

    if len(clean) < 80:
        return None
    return clean


def choose_smoke_dataset(schema_dict: Dict[str, Any]) -> Tuple[str, str]:
    market = infer_market_from_schema(schema_dict)
    desc = (schema_dict.get("description") or "").upper()
    tf = (schema_dict.get("timeframe_hint") or "15m").lower()

    explicit_symbols = _explicit_symbols_from_desc(desc)
    if explicit_symbols:
        for sym in explicit_symbols:
            if market == "crypto" and sym in CRYPTO_SYMBOLS:
                return (sym, tf if tf in {"15m", "1h", "4h"} else "15m")
            if market == "futures" and sym in FUTURES_SYMBOLS:
                return (sym, tf if tf in {"5m", "15m", "1h", "4h", "1d"} else "15m")

    if market == "crypto":
        return ("BTC", tf if tf in {"15m", "1h", "4h"} else "15m")
    return ("MES", tf if tf in {"5m", "15m", "1h", "4h", "1d"} else "15m")


def build_schema(idea: str) -> Dict[str, Any]:
    raw = heuristic_schema_from_idea(idea)
    schema = schema_from_dict(raw, source_idea=idea)
    return schema.to_dict()


def dataset_priority_key(dataset: Tuple[str, str], schema_dict: Dict[str, Any]) -> Tuple[int, int, int]:
    symbol, timeframe = dataset
    desc = (schema_dict.get("description") or "").upper()
    tf_hint = (schema_dict.get("timeframe_hint") or "15m").lower()
    market = infer_market_from_schema(schema_dict)

    explicit_symbols = _explicit_symbols_from_desc(desc)

    symbol_score = 3
    if explicit_symbols and symbol in explicit_symbols:
        symbol_score = 0
    elif market == "futures" and symbol in FUTURES_SYMBOLS:
        symbol_score = 1
    elif market == "crypto" and symbol in CRYPTO_SYMBOLS:
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
    if market == "futures" and symbol not in FUTURES_SYMBOLS:
        market_score = 2
    elif market == "crypto" and symbol not in CRYPTO_SYMBOLS:
        market_score = 2

    return (symbol_score, tf_score, market_score)


def order_datasets(schema_dict: Dict[str, Any], datasets: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    market = infer_market_from_schema(schema_dict)
    if market == "crypto":
        datasets = [d for d in datasets if d[0] in CRYPTO_SYMBOLS]
    elif market == "futures":
        datasets = [d for d in datasets if d[0] in FUTURES_SYMBOLS]
    return sorted(datasets, key=lambda d: dataset_priority_key(d, schema_dict))


def smoke_test_strategy(code: str, dataset: Tuple[str, str]) -> Tuple[bool, str]:
    symbol, timeframe = dataset
    try:
        df = load_dataframe(symbol, timeframe, days_back=180)
        if df is None:
            return False, f"No smoke-test data for {symbol} {timeframe}"

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
        short = str(e) or e.__class__.__name__
        return False, f"{short}\n{traceback.format_exc(limit=2)}"


def smoke_test_with_fallbacks(code: str, primary: Tuple[str, str], schema_dict: Dict[str, Any]) -> Tuple[bool, Tuple[str, str], str]:
    candidates: List[Tuple[str, str]] = []
    seen = set()
    market = infer_market_from_schema(schema_dict)
    tf = (schema_dict.get("timeframe_hint") or "15m").lower()

    def add(ds: Tuple[str, str]):
        if ds not in seen:
            seen.add(ds)
            candidates.append(ds)

    add(primary)

    if market == "crypto":
        for ds in [
            ("BTC", "15m"),
            ("ETH", "15m"),
            ("SOL", "15m"),
            ("BTC", "1H"),
            ("ETH", "1H"),
            ("SOL", "1H"),
            ("BTC", tf if tf in {"15m", "1h", "4h"} else "15m"),
        ]:
            add(ds)
    else:
        submarket = "indices"
        explicit = _explicit_symbols_from_desc(schema_dict.get("description", ""))
        if explicit:
            submarket = infer_futures_submarket_from_symbol(explicit[0])

        if submarket == "oil":
            fallbacks = [("MCL", "15m"), ("MCL", "1H"), ("MCL", tf if tf in {"5m", "15m", "1h", "4h", "1d"} else "15m")]
        elif submarket == "gold":
            fallbacks = [("MGC", "15m"), ("MGC", "1H"), ("MGC", tf if tf in {"5m", "15m", "1h", "4h", "1d"} else "15m")]
        else:
            fallbacks = [
                ("MES", "15m"),
                ("MNQ", "15m"),
                ("MYM", "15m"),
                ("MES", "1H"),
                ("MNQ", "1H"),
                ("MYM", "1H"),
                ("MES", tf if tf in {"5m", "15m", "1h", "4h", "1d"} else "15m"),
            ]
        for ds in fallbacks:
            add(ds)

    last_msg = "No smoke datasets attempted"
    for ds in candidates:
        ok, msg = smoke_test_strategy(code, ds)
        if ok:
            return True, ds, msg
        last_msg = f"{ds[0]} {ds[1]} -> {msg}"
        safe_print(f"  ⚠️ Smoke test failed on {ds[0]} {ds[1]}: {str(msg).splitlines()[0]}")

    return False, primary, last_msg


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _to_new_york_index(index: pd.Index) -> pd.DatetimeIndex | None:
    try:
        dt = pd.DatetimeIndex(index)
        if dt.tz is None:
            dt = dt.tz_localize("UTC")
        return dt.tz_convert("America/New_York")
    except Exception:
        return None


def compute_session_features(df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
    if symbol.upper() not in FUTURES_SYMBOLS or timeframe.upper() in {"4H", "1D"}:
        return {
            "session_rth_share": 0.0,
            "session_first_hour_share": 0.0,
            "session_trend_quality": 0.0,
            "session_score": 0.0,
            "session_aware": False,
        }

    ny_index = _to_new_york_index(df.index)
    if ny_index is None or len(ny_index) == 0:
        return {
            "session_rth_share": 0.0,
            "session_first_hour_share": 0.0,
            "session_trend_quality": 0.0,
            "session_score": 0.0,
            "session_aware": False,
        }

    times = pd.Series(ny_index.time, index=df.index)
    rth_mask = times.between(time(9, 30), time(16, 15))
    first_hour_mask = times.between(time(9, 30), time(10, 30))

    rth_share = float(rth_mask.mean()) if len(rth_mask) else 0.0
    first_hour_share = float(first_hour_mask.mean()) if len(first_hour_mask) else 0.0

    if rth_mask.any():
        rth_df = df.loc[rth_mask.values]
        rth_returns = rth_df["Close"].pct_change().dropna()
        if len(rth_returns) > 5 and rth_returns.std() > 0:
            session_trend_quality = float(abs(rth_returns.mean()) / rth_returns.std())
        else:
            session_trend_quality = 0.0
    else:
        session_trend_quality = 0.0

    session_score = 0.0
    if rth_share >= 0.20:
        session_score += 1.0
    if first_hour_share >= 0.05:
        session_score += 0.5
    if session_trend_quality >= 0.05:
        session_score += 1.0

    return {
        "session_rth_share": round(rth_share, 6),
        "session_first_hour_share": round(first_hour_share, 6),
        "session_trend_quality": round(session_trend_quality, 6),
        "session_score": round(session_score, 6),
        "session_aware": True,
    }


def compute_regime_features(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["Close"]
    if len(df) < 60:
        return {
            "trend_regime": "unknown",
            "trend_strength_pct": 0.0,
            "atr_pct": 0.0,
            "atr_expansion_ratio": 1.0,
            "regime_score": 0.0,
            "trend_aligned": False,
            "vol_expanding": False,
        }

    ema_fast = _ema(close, 20)
    ema_slow = _ema(close, 50)
    atr14 = _atr(df, 14)
    atr50 = atr14.rolling(50).mean()

    last_close = float(close.iloc[-1])
    last_fast = float(ema_fast.iloc[-1]) if not math.isnan(ema_fast.iloc[-1]) else last_close
    last_slow = float(ema_slow.iloc[-1]) if not math.isnan(ema_slow.iloc[-1]) else last_close
    last_atr = float(atr14.iloc[-1]) if not math.isnan(atr14.iloc[-1]) else 0.0
    last_atr50 = float(atr50.iloc[-1]) if not math.isnan(atr50.iloc[-1]) and atr50.iloc[-1] != 0 else max(last_atr, 1e-9)

    trend_strength_pct = abs(last_fast - last_slow) / max(abs(last_close), 1e-9) * 100.0
    atr_pct = last_atr / max(abs(last_close), 1e-9) * 100.0
    atr_expansion_ratio = last_atr / max(last_atr50, 1e-9)

    fast_slope = float(last_fast - ema_fast.iloc[-5]) if len(ema_fast) >= 5 else 0.0
    slow_slope = float(last_slow - ema_slow.iloc[-5]) if len(ema_slow) >= 5 else 0.0

    if last_fast > last_slow and fast_slope > 0 and slow_slope > 0:
        trend_regime = "bullish"
    elif last_fast < last_slow and fast_slope < 0 and slow_slope < 0:
        trend_regime = "bearish"
    else:
        trend_regime = "ranging"

    trend_aligned = trend_regime in {"bullish", "bearish"}
    vol_expanding = atr_expansion_ratio >= 1.05

    regime_score = 0.0
    if trend_aligned:
        regime_score += 1.5
    if vol_expanding:
        regime_score += 1.0
    if trend_strength_pct >= 1.0:
        regime_score += 1.0
    if trend_regime == "ranging":
        regime_score -= 1.0

    return {
        "trend_regime": trend_regime,
        "trend_strength_pct": round(trend_strength_pct, 6),
        "atr_pct": round(atr_pct, 6),
        "atr_expansion_ratio": round(atr_expansion_ratio, 6),
        "regime_score": round(regime_score, 6),
        "trend_aligned": trend_aligned,
        "vol_expanding": vol_expanding,
    }


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

        sharpe_raw = stats.get("Sharpe Ratio", 0.0)
        win_raw = stats.get("Win Rate [%]", 0.0)

        sharpe = float(sharpe_raw or 0.0) if sharpe_raw == sharpe_raw else 0.0
        win_rate = float(win_raw or 0.0) if win_raw == win_raw else 0.0
        ret = float(stats.get("Return [%]", 0.0) or 0.0)
        expectancy_proxy = ret / max(trades, 1)

        regime = compute_regime_features(df)
        session = compute_session_features(df, symbol, timeframe)

        overtraded = trades > MAX_TRADES_HARD or (timeframe.lower() == "15m" and trades > MAX_TRADES_15M)
        soft_overtraded = trades > MAX_TRADES_SOFT
        unreliable_sample = trades < MIN_RELIABLE_TRADES

        return {
            "strategy": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "return_pct": ret,
            "sharpe": sharpe,
            "max_drawdown": float(stats.get("Max. Drawdown [%]", 0.0) or 0.0),
            "win_rate": win_rate,
            "num_trades": int(trades),
            "equity_final": float(stats.get("Equity Final [$]", 0.0) or 0.0),
            "buy_hold_return_pct": float(stats.get("Buy & Hold Return [%]", 0.0) or 0.0),
            "expectancy_proxy": expectancy_proxy,
            "under_sampled": trades < MIN_TRADES_FOR_BEST,
            "unreliable_sample": unreliable_sample,
            "overtraded": overtraded,
            "soft_overtraded": soft_overtraded,
            "trend_regime": regime["trend_regime"],
            "trend_strength_pct": regime["trend_strength_pct"],
            "atr_pct": regime["atr_pct"],
            "atr_expansion_ratio": regime["atr_expansion_ratio"],
            "regime_score": regime["regime_score"],
            "trend_aligned": regime["trend_aligned"],
            "vol_expanding": regime["vol_expanding"],
            "session_rth_share": session["session_rth_share"],
            "session_first_hour_share": session["session_first_hour_share"],
            "session_trend_quality": session["session_trend_quality"],
            "session_score": session["session_score"],
            "session_aware": session["session_aware"],
            "ranking_score_discovery": 0.0,
            "ranking_score_faithful": 0.0,
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
            "expectancy_proxy": 0.0,
            "under_sampled": True,
            "unreliable_sample": True,
            "overtraded": False,
            "soft_overtraded": False,
            "trend_regime": "unknown",
            "trend_strength_pct": 0.0,
            "atr_pct": 0.0,
            "atr_expansion_ratio": 1.0,
            "regime_score": 0.0,
            "trend_aligned": False,
            "vol_expanding": False,
            "session_rth_share": 0.0,
            "session_first_hour_share": 0.0,
            "session_trend_quality": 0.0,
            "session_score": 0.0,
            "session_aware": False,
            "ranking_score_discovery": -999999.0,
            "ranking_score_faithful": -999999.0,
            "error": str(e),
        }


def _locked_symbol(schema_dict: Dict[str, Any]) -> str | None:
    v = schema_dict.get("locked_symbol")
    return str(v).lower() if v else None


def _locked_timeframe(schema_dict: Dict[str, Any]) -> str | None:
    v = schema_dict.get("locked_timeframe")
    return str(v).lower() if v else None


def _symbol_match_score(result: Dict[str, Any], schema_dict: Dict[str, Any], faithful: bool) -> float:
    symbol = (result.get("symbol") or "").lower()

    if faithful and _locked_symbol(schema_dict):
        return 3.0 if symbol == _locked_symbol(schema_dict) else -4.0

    desc = (schema_dict.get("description") or "").upper()
    explicit_symbols = _explicit_symbols_from_desc(desc)
    if not explicit_symbols:
        return 0.0
    if symbol.upper() in explicit_symbols:
        return 2.5 if faithful else 1.5
    return -2.0 if faithful else -0.5


def _timeframe_match_score(result: Dict[str, Any], schema_dict: Dict[str, Any], faithful: bool) -> float:
    tf = (result.get("timeframe") or "").lower()
    tf_hint = _locked_timeframe(schema_dict) if faithful and _locked_timeframe(schema_dict) else (schema_dict.get("timeframe_hint") or "15m").lower()

    if tf == tf_hint:
        return 3.0 if faithful else 1.5

    if faithful:
        adjacency = {
            "5m": {"15m": -0.5, "1h": -2.0, "4h": -4.0, "1d": -5.0},
            "15m": {"5m": -0.5, "1h": -0.8, "4h": -2.5, "1d": -4.0},
            "1h": {"15m": -0.8, "4h": -0.8, "5m": -2.0, "1d": -2.0},
            "4h": {"1h": -0.8, "1d": -0.8, "15m": -2.5, "5m": -4.0},
            "1d": {"4h": -0.8, "1h": -2.0, "15m": -4.0, "5m": -5.0},
        }
    else:
        adjacency = {
            "5m": {"15m": 0.4, "1h": -0.3, "4h": -0.8, "1d": -1.2},
            "15m": {"5m": 0.3, "1h": 0.25, "4h": -0.5, "1d": -0.9},
            "1h": {"15m": 0.15, "4h": 0.15, "5m": -0.4, "1d": -0.4},
            "4h": {"1h": 0.1, "1d": 0.1, "15m": -0.5, "5m": -0.9},
            "1d": {"4h": 0.1, "1h": -0.4, "15m": -0.8, "5m": -1.2},
        }

    return adjacency.get(tf_hint, {}).get(tf, -0.75 if faithful else -0.25)


def _timeframe_bias_score(result: Dict[str, Any], schema_dict: Dict[str, Any], faithful: bool) -> float:
    tf = str(result.get("timeframe") or "").lower()
    market = infer_market_from_schema(schema_dict)

    if market == "crypto":
        if tf == "15m":
            return -3.0
        if tf == "1h":
            return 1.5
        if tf == "4h":
            return 2.0
    else:
        symbol = str(result.get("symbol", "")).upper()
        submarket = infer_futures_submarket_from_symbol(symbol)

        if submarket == "indices":
            if tf == "15m":
                return -1.5
            if tf == "1h":
                return 1.75
            if tf == "4h":
                return 2.0
            if tf == "1d":
                return 0.75

        if submarket == "oil":
            if tf == "5m":
                return 0.5
            if tf == "15m":
                return 1.0
            if tf == "1h":
                return 1.75
            if tf == "4h":
                return 1.25

        if submarket == "gold":
            if tf == "5m":
                return -0.5
            if tf == "15m":
                return 0.5
            if tf == "1h":
                return 1.5
            if tf == "4h":
                return 2.0

    return 0.0


def _trade_quality_penalty(result: Dict[str, Any]) -> float:
    trades = int(result.get("num_trades", 0) or 0)
    tf = str(result.get("timeframe") or "").lower()

    penalty = 0.0
    if trades > MAX_TRADES_SOFT:
        penalty += 4.0
    if trades > MAX_TRADES_HARD:
        penalty += 8.0
    if tf == "15m" and trades > MAX_TRADES_15M:
        penalty += 6.0
    if bool(result.get("unreliable_sample", False)):
        penalty += 1.5
    return penalty


def _regime_bonus(result: Dict[str, Any]) -> float:
    regime_score = float(result.get("regime_score", 0.0) or 0.0)
    trend_aligned = bool(result.get("trend_aligned", False))
    vol_expanding = bool(result.get("vol_expanding", False))

    bonus = regime_score * 0.8
    if trend_aligned:
        bonus += 0.75
    if vol_expanding:
        bonus += 0.50
    return bonus


def _session_bonus(result: Dict[str, Any], schema_dict: Dict[str, Any]) -> float:
    if infer_market_from_schema(schema_dict) != "futures":
        return 0.0

    symbol = str(result.get("symbol", "")).upper()
    submarket = infer_futures_submarket_from_symbol(symbol)

    score = float(result.get("session_score", 0.0) or 0.0)
    rth_share = float(result.get("session_rth_share", 0.0) or 0.0)
    first_hour = float(result.get("session_first_hour_share", 0.0) or 0.0)
    trend_quality = float(result.get("session_trend_quality", 0.0) or 0.0)

    if submarket == "oil":
        bonus = score * 0.6
        if first_hour >= 0.05:
            bonus += 0.15
        return bonus

    if submarket == "gold":
        bonus = score * 0.7
        if rth_share >= 0.20:
            bonus += 0.25
        return bonus

    bonus = score * 0.9
    if rth_share >= 0.20:
        bonus += 0.75
    if first_hour >= 0.05:
        bonus += 0.25
    if trend_quality >= 0.05:
        bonus += 0.5
    return bonus


def compute_ranking_score(result: Dict[str, Any], schema_dict: Dict[str, Any], faithful: bool = False) -> float:
    if result.get("error"):
        return -999999.0

    sharpe = result.get("sharpe", 0.0) or 0.0
    ret = result.get("return_pct", 0.0) or 0.0
    win = result.get("win_rate", 0.0) or 0.0
    trades = result.get("num_trades", 0) or 0
    dd = abs(result.get("max_drawdown", 0.0) or 0.0)
    expectancy_proxy = result.get("expectancy_proxy", 0.0) or 0.0

    score = 0.0
    score += sharpe * (1.8 if faithful else 2.0)
    score += ret * (0.40 if faithful else 0.45)
    score += win * 0.03
    score += min(trades, 100) * 0.02
    score += expectancy_proxy * 25.0
    score -= dd * 0.08

    score += _symbol_match_score(result, schema_dict, faithful=faithful)
    score += _timeframe_match_score(result, schema_dict, faithful=faithful)
    score += _timeframe_bias_score(result, schema_dict, faithful=faithful)
    score += _regime_bonus(result)
    score += _session_bonus(result, schema_dict)
    score -= _trade_quality_penalty(result)

    if trades < MIN_TRADES_FOR_BEST:
        score -= 3.0
    if bool(result.get("overtraded", False)):
        score -= 12.0
    if expectancy_proxy < 0:
        score -= 3.0
    elif expectancy_proxy < MIN_DISCOVERY_EXPECTANCY_PROXY:
        score -= 2.0

    return round(score, 6)


def enrich_rankings(results: List[Dict[str, Any]], schema_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    enriched = []
    for r in results:
        row = dict(r)
        row["ranking_score_discovery"] = compute_ranking_score(row, schema_dict, faithful=False)
        row["ranking_score_faithful"] = compute_ranking_score(row, schema_dict, faithful=True)
        enriched.append(row)
    return enriched


def rank_results(results: List[Dict[str, Any]], schema_dict: Dict[str, Any], faithful: bool = False) -> List[Dict[str, Any]]:
    enriched = enrich_rankings(results, schema_dict)
    key = "ranking_score_faithful" if faithful else "ranking_score_discovery"
    return sorted(enriched, key=lambda r: r.get(key, -999999.0), reverse=True)


def select_best_result(valid: List[Dict[str, Any]], schema_dict: Dict[str, Any], faithful: bool = False) -> Dict[str, Any] | None:
    if not valid:
        return None
    ranked = rank_results(valid, schema_dict, faithful=faithful)
    eligible = [
        r for r in ranked
        if (r.get("num_trades", 0) or 0) >= MIN_TRADES_FOR_BEST and not bool(r.get("overtraded", False))
    ]
    return eligible[0] if eligible else ranked[0]


def _passes_research(best: Dict[str, Any] | None) -> bool:
    return bool(
        best
        and not bool(best.get("overtraded", False))
        and not bool(best.get("unreliable_sample", False))
        and (best.get("num_trades", 0) or 0) >= MIN_TRADES_FOR_BEST
        and (best.get("sharpe", 0.0) or 0.0) >= MIN_SHARPE_FOR_RESEARCH_PASS
        and (best.get("return_pct", 0.0) or 0.0) >= MIN_RETURN_FOR_RESEARCH_PASS
        and (best.get("expectancy_proxy", 0.0) or 0.0) >= 0.0
    )


def _passes_discovery_quality(best: Dict[str, Any] | None) -> bool:
    if not best:
        return False

    base_ok = bool(
        not bool(best.get("overtraded", False))
        and not bool(best.get("unreliable_sample", False))
        and (best.get("num_trades", 0) or 0) >= MIN_DISCOVERY_TRADES
        and (best.get("sharpe", 0.0) or 0.0) >= MIN_DISCOVERY_SHARPE
        and (best.get("return_pct", 0.0) or 0.0) >= MIN_DISCOVERY_RETURN
        and (best.get("expectancy_proxy", 0.0) or 0.0) >= MIN_DISCOVERY_EXPECTANCY_PROXY
        and bool(best.get("trend_aligned", False))
        and (best.get("regime_score", 0.0) or 0.0) >= 1.0
    )
    if not base_ok:
        return False

    symbol = str(best.get("symbol", "")).upper()
    submarket = infer_futures_submarket_from_symbol(symbol)
    if submarket == "indices":
        return (best.get("session_score", 0.0) or 0.0) >= 1.0
    return True


def _vault_hint(best: Dict[str, Any] | None) -> bool:
    if not best:
        return False

    base_ok = bool(
        not bool(best.get("overtraded", False))
        and not bool(best.get("unreliable_sample", False))
        and (best.get("return_pct", 0.0) or 0.0) >= 2.0
        and (best.get("sharpe", 0.0) or 0.0) >= 1.2
        and (best.get("expectancy_proxy", 0.0) or 0.0) >= MIN_DISCOVERY_EXPECTANCY_PROXY
        and (best.get("num_trades", 0) or 0) >= MIN_RELIABLE_TRADES
    )
    if not base_ok:
        return False

    symbol = str(best.get("symbol", "")).upper()
    submarket = infer_futures_submarket_from_symbol(symbol)
    if submarket == "indices":
        return (best.get("session_score", 0.0) or 0.0) >= 1.0
    return True


def summarize_result_set(results: List[Dict[str, Any]], schema_dict: Dict[str, Any]) -> Dict[str, Any]:
    valid = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]

    if not valid:
        return {
            "valid_count": 0,
            "failed_count": len(failed),
            "best_discovery": None,
            "best_faithful": None,
            "best": None,
            "mean_sharpe": 0.0,
            "mean_return": 0.0,
            "mean_win_rate": 0.0,
            "dataset_breadth": 0,
            "eligible_count": 0,
            "research_pass_discovery": False,
            "research_pass_faithful": False,
            "discovery_quality_pass": False,
            "vault_ready_hint_discovery": False,
            "vault_ready_hint_faithful": False,
            "overtraded_count": 0,
            "reliable_count": 0,
        }

    enriched = enrich_rankings(valid, schema_dict)
    ranked_discovery = sorted(enriched, key=lambda r: r.get("ranking_score_discovery", -999999.0), reverse=True)
    ranked_faithful = sorted(enriched, key=lambda r: r.get("ranking_score_faithful", -999999.0), reverse=True)

    mean_sharpe = sum(r["sharpe"] for r in enriched) / len(enriched)
    mean_return = sum(r["return_pct"] for r in enriched) / len(enriched)
    mean_win = sum(r["win_rate"] for r in enriched) / len(enriched)
    positive_datasets = sum(1 for r in enriched if r["return_pct"] > 0)
    eligible_count = sum(
        1 for r in enriched
        if (r.get("num_trades", 0) or 0) >= MIN_TRADES_FOR_BEST and not bool(r.get("overtraded", False))
    )
    overtraded_count = sum(1 for r in enriched if bool(r.get("overtraded", False)))
    reliable_count = sum(1 for r in enriched if not bool(r.get("unreliable_sample", False)))

    best_discovery = select_best_result(enriched, schema_dict, faithful=False)
    best_faithful = select_best_result(enriched, schema_dict, faithful=True)

    return {
        "valid_count": len(enriched),
        "failed_count": len(failed),
        "best_discovery": best_discovery,
        "best_faithful": best_faithful,
        "best": best_faithful or best_discovery,
        "mean_sharpe": round(mean_sharpe, 4),
        "mean_return": round(mean_return, 4),
        "mean_win_rate": round(mean_win, 4),
        "dataset_breadth": positive_datasets,
        "eligible_count": eligible_count,
        "overtraded_count": overtraded_count,
        "reliable_count": reliable_count,
        "research_pass_discovery": _passes_research(best_discovery),
        "research_pass_faithful": _passes_research(best_faithful),
        "discovery_quality_pass": _passes_discovery_quality(best_discovery),
        "vault_ready_hint_discovery": _vault_hint(best_discovery),
        "vault_ready_hint_faithful": _vault_hint(best_faithful),
        "ranked_preview_discovery": ranked_discovery[:5],
        "ranked_preview_faithful": ranked_faithful[:5],
    }


def classify_idea(summary: Dict[str, Any]) -> str:
    best_faithful = summary.get("best_faithful")
    faithful_research = summary.get("research_pass_faithful", False)
    discovery_research = summary.get("research_pass_discovery", False)
    discovery_quality = summary.get("discovery_quality_pass", False)
    vault_faithful = summary.get("vault_ready_hint_faithful", False)
    vault_discovery = summary.get("vault_ready_hint_discovery", False)

    if vault_faithful or vault_discovery:
        return "vault_candidate"

    if faithful_research and discovery_research and discovery_quality:
        return "research_candidate"

    if faithful_research:
        return "watchlist_faithful"

    if discovery_research and discovery_quality:
        return "watchlist_discovery"

    if best_faithful and (best_faithful.get("num_trades", 0) or 0) >= MIN_RELIABLE_TRADES:
        if not bool(best_faithful.get("overtraded", False)) and (best_faithful.get("return_pct", 0.0) or 0.0) > 0:
            return "watchlist_faithful"

    return "reject"


def write_summary_csv(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "strategy", "symbol", "timeframe", "return_pct", "sharpe", "max_drawdown",
        "win_rate", "num_trades", "equity_final", "buy_hold_return_pct",
        "expectancy_proxy", "under_sampled", "unreliable_sample", "overtraded",
        "soft_overtraded", "trend_regime", "trend_strength_pct", "atr_pct",
        "atr_expansion_ratio", "regime_score", "trend_aligned", "vol_expanding",
        "session_rth_share", "session_first_hour_share", "session_trend_quality",
        "session_score", "session_aware",
        "ranking_score_discovery", "ranking_score_faithful", "error"
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _apply_schema_overrides(base_schema: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    schema = json.loads(json.dumps(base_schema))
    for section, params in overrides.items():
        if section not in schema:
            schema[section] = {}
        schema[section].update(params)
    return schema


def _family_param_grid(schema_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    family = (schema_dict.get("family") or "").lower()
    if family != "breakout":
        return [{}]

    base = schema_dict
    base_lookback = int(base["setup_params"].get("lookback", 10))
    base_rr = float(base["risk_params"].get("tp_r_multiple", 2.0))

    lookbacks = sorted(set([max(5, base_lookback - 2), base_lookback, base_lookback + 2, base_lookback + 5]))
    retest_bars = [2, 3, 4, 5]
    retest_tols = [0.0005, 0.001, 0.0015]
    volume_mults = [1.0, 1.05, 1.1, 1.2]
    large_bar_mults = [0.1, 0.2, 0.3]
    sl_mults = [0.5, 0.75, 1.0, 1.25]
    rr_mults = sorted(set([1.5, 2.0, 2.5, 3.0, round(base_rr, 2)]))
    min_breakout_mults = [0.8, 1.0, 1.2]
    move_to_be_vals = [0.0, 1.0]
    trail_after_vals = [0.0, 1.0]
    failure_exit_vals = [False, True]

    combos = []
    for combo in itertools.product(
        lookbacks, retest_bars, retest_tols, volume_mults, large_bar_mults,
        sl_mults, rr_mults, min_breakout_mults, move_to_be_vals, trail_after_vals, failure_exit_vals
    ):
        lookback, max_retest_bars, retest_tol, volume_mult, large_bar_mult, sl_mult, rr_mult, min_breakout_mult, move_to_be, trail_after, failure_exit = combo
        combos.append({
            "setup_params": {
                "lookback": lookback,
                "max_retest_bars": max_retest_bars,
                "retest_tolerance_pct": retest_tol,
                "volume_multiplier": volume_mult,
                "large_bar_atr_mult": large_bar_mult,
                "min_breakout_range_mult": min_breakout_mult,
                "move_to_be_at_r": move_to_be,
                "trail_atr_after_r": trail_after,
                "failure_exit_on_level_reclaim": failure_exit,
            },
            "risk_params": {
                "sl_atr_mult": sl_mult,
                "tp_r_multiple": rr_mult,
            },
        })

    limited = []
    seen = set()
    for c in combos:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            limited.append(c)

    return limited[:96]


def _run_variant_across_datasets(schema_dict: Dict[str, Any], ordered: List[Tuple[str, str]], max_workers: int) -> Tuple[List[Dict[str, Any]], str]:
    schema_obj = schema_from_dict(schema_dict, source_idea=schema_dict.get("description", ""))
    code = compile_strategy_class(schema_obj, class_name="GeneratedStrategy")
    jobs = [(symbol, timeframe, code, schema_dict["name"]) for symbol, timeframe in ordered]
    results: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_backtest, job) for job in jobs]
        for fut in concurrent.futures.as_completed(futures):
            try:
                result = fut.result()
                if result:
                    results.append(result)
            except Exception as e:
                safe_print(f"⚠️ Worker failed: {e}")

    ranked_results = enrich_rankings(results, schema_dict)
    return ranked_results, code


def optimize_schema_variant(base_schema: Dict[str, Any], ordered: List[Tuple[str, str]], max_workers: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], str]:
    family = (base_schema.get("family") or "").lower()
    grid = _family_param_grid(base_schema)

    if family != "breakout" or len(grid) == 1:
        ranked_results, code = _run_variant_across_datasets(base_schema, ordered, max_workers)
        summary = summarize_result_set(ranked_results, base_schema)
        return base_schema, ranked_results, summary, code

    best_schema = base_schema
    best_code = ""
    best_score = -999999.0
    preview_datasets = ordered[: min(6, len(ordered))]

    print(f"🧪 Optimizing {family} family across {len(grid)} parameter variants...")

    for idx, overrides in enumerate(grid, start=1):
        variant_schema = _apply_schema_overrides(base_schema, overrides)

        try:
            variant_results, code = _run_variant_across_datasets(variant_schema, preview_datasets, min(max_workers, 6))
            variant_summary = summarize_result_set(variant_results, variant_schema)
            candidate = variant_summary.get("best_faithful") or variant_summary.get("best_discovery")
            candidate_score = candidate.get("ranking_score_faithful", -999999.0) if candidate else -999999.0

            if candidate_score > best_score:
                best_score = candidate_score
                best_schema = variant_schema
                best_code = code

            if idx % 10 == 0:
                print(f"  ↳ checked {idx}/{len(grid)} variants | current best faithful score {best_score:.2f}")

        except Exception as e:
            safe_print(f"⚠️ Variant optimization error: {e}")

    print("✅ Re-running best variant across full ordered dataset set...")
    full_results, full_code = _run_variant_across_datasets(best_schema, ordered, max_workers)
    full_summary = summarize_result_set(full_results, best_schema)
    return best_schema, full_results, full_summary, full_code


def _primary_symbol_from_desc(desc: str) -> str | None:
    syms = _explicit_symbols_from_desc(desc)
    return syms[0].upper() if syms else None


def _neighbor_timeframes(tf: str, market: str) -> List[str]:
    tf = tf.lower()
    if market == "crypto":
        mapping = {
            "15m": ["15m", "1h", "4h"],
            "1h": ["1h", "15m", "4h"],
            "4h": ["4h", "1h", "15m"],
        }
        return mapping.get(tf, [tf, "15m", "1h"])
    mapping = {
        "5m": ["5m", "15m", "1h"],
        "15m": ["15m", "5m", "1h"],
        "1h": ["1h", "15m", "4h"],
        "4h": ["4h", "1h", "1d"],
        "1d": ["1d", "4h", "1h"],
    }
    return mapping.get(tf, [tf, "15m", "1h"])


def _build_rewrite_idea(base_idea: str, symbol: str, timeframe: str, rewrite_label: str) -> str:
    return f"{base_idea} | rewrite: {rewrite_label} | adapted to {symbol} {timeframe}"


def generate_rewrite_variants(base_schema: Dict[str, Any], summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    market = infer_market_from_schema(base_schema)
    desc = (base_schema.get("description") or "").upper()
    faithful_symbol = _primary_symbol_from_desc(desc) or (summary.get("best_faithful") or {}).get("symbol") or market_default_symbol(market)
    faithful_tf = (base_schema.get("timeframe_hint") or market_default_timeframe()).lower()

    best_discovery = summary.get("best_discovery") or {}
    discovery_symbol = best_discovery.get("symbol", faithful_symbol)
    discovery_tf = (best_discovery.get("timeframe") or faithful_tf).lower()

    neighbors = market_neighbor_symbols(faithful_symbol, market)
    alt_symbol = neighbors[1] if len(neighbors) > 1 else faithful_symbol
    shifted_tf = _neighbor_timeframes(faithful_tf, market)[1]

    variants: List[Dict[str, Any]] = []

    candidate_specs = [
        {
            "lane": "faithful",
            "label": "faithful_tighter_retest",
            "symbol": faithful_symbol,
            "timeframe": faithful_tf,
            "setup_params": {
                "close_confirmation": True,
                "rejection_confirmation": True,
                "max_retest_bars": 2,
                "retest_tolerance_pct": 0.0005,
                "volume_multiplier": 1.10,
                "min_breakout_range_mult": 1.2,
                "failure_exit_on_level_reclaim": True,
            },
            "risk_params": {"sl_atr_mult": 0.75, "tp_r_multiple": 2.0},
        },
        {
            "lane": "faithful",
            "label": "faithful_shifted_tf",
            "symbol": faithful_symbol,
            "timeframe": shifted_tf,
            "setup_params": {
                "close_confirmation": True,
                "rejection_confirmation": True,
                "max_retest_bars": 3,
                "retest_tolerance_pct": 0.001,
                "volume_multiplier": 1.05,
                "min_breakout_range_mult": 1.0,
            },
            "risk_params": {"sl_atr_mult": 0.75, "tp_r_multiple": 2.0},
        },
        {
            "lane": "discovery",
            "label": "discovery_market_mapped",
            "symbol": discovery_symbol,
            "timeframe": discovery_tf,
            "setup_params": {
                "close_confirmation": True,
                "rejection_confirmation": True,
                "max_retest_bars": 2,
                "retest_tolerance_pct": 0.0005,
                "volume_multiplier": 1.10,
                "min_breakout_range_mult": 1.2,
            },
            "risk_params": {"sl_atr_mult": 0.75, "tp_r_multiple": 2.0},
        },
        {
            "lane": "discovery",
            "label": "discovery_neighbor_symbol",
            "symbol": alt_symbol if discovery_symbol == faithful_symbol else discovery_symbol,
            "timeframe": shifted_tf if discovery_tf == faithful_tf else discovery_tf,
            "setup_params": {
                "close_confirmation": True,
                "rejection_confirmation": True,
                "max_retest_bars": 2,
                "retest_tolerance_pct": 0.0007,
                "volume_multiplier": 1.05,
                "min_breakout_range_mult": 1.1,
            },
            "risk_params": {"sl_atr_mult": 0.75, "tp_r_multiple": 2.0},
        },
        {
            "lane": "faithful",
            "label": "faithful_balanced",
            "symbol": faithful_symbol,
            "timeframe": faithful_tf,
            "setup_params": {
                "close_confirmation": True,
                "rejection_confirmation": True,
                "max_retest_bars": 3,
                "retest_tolerance_pct": 0.001,
                "volume_multiplier": 1.05,
                "min_breakout_range_mult": 1.0,
                "move_to_be_at_r": 1.0,
                "trail_atr_after_r": 1.0,
            },
            "risk_params": {"sl_atr_mult": 0.5, "tp_r_multiple": 1.5},
        },
    ]

    seen = set()
    for spec in candidate_specs:
        schema = json.loads(json.dumps(base_schema))
        schema["description"] = _build_rewrite_idea(base_schema.get("description", ""), spec["symbol"], spec["timeframe"], spec["label"])
        schema["normalized_idea"] = schema["description"]
        schema["timeframe_hint"] = spec["timeframe"]
        schema["locked_symbol"] = spec["symbol"]
        schema["locked_timeframe"] = spec["timeframe"]
        schema.setdefault("setup_params", {})
        schema.setdefault("risk_params", {})
        schema["setup_params"].update(spec["setup_params"])
        schema["risk_params"].update(spec["risk_params"])

        key = json.dumps(
            {
                "lane": spec["lane"],
                "label": spec["label"],
                "symbol": spec["symbol"],
                "timeframe": spec["timeframe"],
                "setup_params": schema["setup_params"],
                "risk_params": schema["risk_params"],
            },
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)

        variants.append(
            {
                "lane": spec["lane"],
                "label": spec["label"],
                "target_symbol": spec["symbol"],
                "target_timeframe": spec["timeframe"],
                "rewrite_idea": schema["description"],
                "schema": schema,
            }
        )

    return variants[:5]


def _rewrite_ordered_datasets(schema_dict: Dict[str, Any], datasets: List[Tuple[str, str]], target_symbol: str, target_timeframe: str) -> List[Tuple[str, str]]:
    exact = []
    same_symbol = []
    same_timeframe = []
    rest = []

    market = infer_market_from_schema(schema_dict)
    if market == "crypto":
        datasets = [d for d in datasets if d[0] in CRYPTO_SYMBOLS]
    else:
        datasets = [d for d in datasets if d[0] in FUTURES_SYMBOLS]

    for ds in datasets:
        symbol, timeframe = ds
        tf = timeframe.lower()
        target_tf = target_timeframe.lower()

        if symbol == target_symbol and tf == target_tf:
            exact.append(ds)
        elif symbol == target_symbol:
            same_symbol.append(ds)
        elif tf == target_tf:
            same_timeframe.append(ds)
        else:
            rest.append(ds)

    rest = order_datasets(schema_dict, rest)
    return exact + same_symbol + same_timeframe + rest


def test_rewrite_variants(base_schema: Dict[str, Any], base_summary: Dict[str, Any], datasets: List[Tuple[str, str]], max_workers: int) -> List[Dict[str, Any]]:
    variants = generate_rewrite_variants(base_schema, base_summary)
    if not variants:
        return []

    print(f"🧬 Testing {len(variants)} rewrite variants...")

    outputs: List[Dict[str, Any]] = []
    for idx, variant in enumerate(variants, start=1):
        label = variant["label"]
        schema = variant["schema"]
        ordered = _rewrite_ordered_datasets(schema, datasets, variant["target_symbol"], variant["target_timeframe"])

        print(
            f"  ↳ rewrite {idx}/{len(variants)}: {label} "
            f"[{variant['lane']}] [{variant['target_symbol']} {variant['target_timeframe']}]"
        )

        try:
            ranked_results, code = _run_variant_across_datasets(schema, ordered[: min(8, len(ordered))], min(max_workers, 8))
            summary = summarize_result_set(ranked_results, schema)
            outputs.append(
                {
                    "lane": variant["lane"],
                    "label": label,
                    "target_symbol": variant["target_symbol"],
                    "target_timeframe": variant["target_timeframe"],
                    "rewrite_idea": variant["rewrite_idea"],
                    "schema": schema,
                    "summary": summary,
                    "results_preview": ranked_results[:5],
                    "compiled_code": code,
                }
            )
        except Exception as e:
            outputs.append(
                {
                    "lane": variant["lane"],
                    "label": label,
                    "target_symbol": variant["target_symbol"],
                    "target_timeframe": variant["target_timeframe"],
                    "rewrite_idea": variant["rewrite_idea"],
                    "schema": schema,
                    "summary": {"best_faithful": None, "best_discovery": None},
                    "results_preview": [],
                    "error": str(e),
                }
            )

    return outputs


def _score_rewrite_candidate(candidate: Dict[str, Any], item: Dict[str, Any]) -> float:
    base = candidate.get("ranking_score_faithful", -999999.0)
    symbol_bonus = 4.0 if candidate.get("symbol") == item.get("target_symbol") else -6.0
    timeframe_bonus = 4.0 if str(candidate.get("timeframe", "")).lower() == str(item.get("target_timeframe", "")).lower() else -6.0
    return base + symbol_bonus + timeframe_bonus


def _choose_best_rewrite_by_lane(rewrite_variants_output: List[Dict[str, Any]], lane: str) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    best_item = None
    best_candidate = None
    best_score = -999999.0

    for item in rewrite_variants_output:
        if item.get("lane") != lane:
            continue
        summary = item.get("summary", {})
        candidate = summary.get("best_faithful") or summary.get("best_discovery")
        if not candidate:
            continue
        score = _score_rewrite_candidate(candidate, item)
        if score > best_score:
            best_score = score
            best_item = item
            best_candidate = candidate

    return best_item, best_candidate


def promote_rewrite_lane_and_optimize(
    rewrite_variants_output: List[Dict[str, Any]],
    datasets: List[Tuple[str, str]],
    max_workers: int,
    lane: str,
) -> Dict[str, Any] | None:
    best_item, best_candidate = _choose_best_rewrite_by_lane(rewrite_variants_output, lane)
    if not best_item or not best_candidate:
        return None

    promoted_schema = json.loads(json.dumps(best_item["schema"]))
    promoted_schema["name"] = f"{promoted_schema.get('name', 'Strategy')} Rewrite {lane.title()}"
    promoted_schema["description"] = best_item["rewrite_idea"]
    promoted_schema["normalized_idea"] = best_item["rewrite_idea"]
    promoted_schema["timeframe_hint"] = best_item["target_timeframe"]
    promoted_schema["locked_symbol"] = best_item["target_symbol"]
    promoted_schema["locked_timeframe"] = best_item["target_timeframe"]

    target_symbol = best_item["target_symbol"]
    target_timeframe = best_item["target_timeframe"]
    ordered = _rewrite_ordered_datasets(promoted_schema, datasets, target_symbol, target_timeframe)

    print(
        f"🚀 Promoting {lane} rewrite '{best_item['label']}' into second-stage optimization "
        f"anchored to {target_symbol} {target_timeframe}..."
    )

    try:
        best_schema, ranked_results, summary, code = optimize_schema_variant(promoted_schema, ordered[: min(8, len(ordered))], max_workers)
        return {
            "lane": lane,
            "label": best_item["label"],
            "rewrite_idea": best_item["rewrite_idea"],
            "schema": best_schema,
            "summary": summary,
            "results": ranked_results,
            "compiled_code": code,
            "target_symbol": target_symbol,
            "target_timeframe": target_timeframe,
            "anchored": True,
        }
    except Exception as e:
        return {
            "lane": lane,
            "label": best_item["label"],
            "rewrite_idea": best_item["rewrite_idea"],
            "schema": promoted_schema,
            "summary": {"best_faithful": None, "best_discovery": None},
            "results": [],
            "error": str(e),
            "target_symbol": target_symbol,
            "target_timeframe": target_timeframe,
            "anchored": True,
        }


class ParallelRBIAgentV2:
    def __init__(self, max_workers: int = 12, ideas_file: Optional[Path] = None, forced_market: str = "auto"):
        self.max_workers = max_workers
        self.processed = load_processed()
        self.summary_rows: List[Dict[str, Any]] = []
        self.ideas_file = ideas_file or DEFAULT_IDEAS_FILE
        self.forced_market = forced_market

    def resolve_run_market(self, ideas: List[str]) -> str:
        if self.forced_market in {"crypto", "futures", "all"}:
            return self.forced_market
        return infer_market_from_ideas_file(self.ideas_file, ideas)

    def datasets_for_schema(self, schema_dict: Dict[str, Any]) -> List[Tuple[str, str]]:
        if self.forced_market == "all":
            return order_datasets(schema_dict, CRYPTO_DATASETS + FUTURES_DATASETS)
        if self.forced_market == "crypto":
            return order_datasets(schema_dict, CRYPTO_DATASETS)
        if self.forced_market == "futures":
            return order_datasets(schema_dict, FUTURES_DATASETS)

        market = infer_market_from_schema(schema_dict)
        if market == "crypto":
            return order_datasets(schema_dict, CRYPTO_DATASETS)
        if market == "futures":
            explicit = _explicit_symbols_from_desc(schema_dict.get("description", ""))
            if explicit:
                submarket = infer_futures_submarket_from_symbol(explicit[0])
            else:
                submarket = infer_futures_submarket_from_ideas_file(self.ideas_file, [schema_dict.get("description", "")])

            if submarket == "oil":
                return order_datasets(schema_dict, OIL_FUTURES_DATASETS)
            if submarket == "gold":
                return order_datasets(schema_dict, GOLD_FUTURES_DATASETS)
            if submarket == "indices":
                return order_datasets(schema_dict, INDEX_FUTURES_DATASETS)
            return order_datasets(schema_dict, FUTURES_DATASETS)

        return order_datasets(schema_dict, CRYPTO_DATASETS + FUTURES_DATASETS)

    def process_idea(self, idea: str):
        print("\n" + "─" * 60)
        print(f"💡 IDEA: {idea}")
        print("🔬 Building schema...")

        idea_id = idea_hash(idea)
        rewrite_variants_output: List[Dict[str, Any]] = []
        promoted_rewrite_faithful: Dict[str, Any] | None = None
        promoted_rewrite_discovery: Dict[str, Any] | None = None

        quality = evaluate_idea_quality(idea)
        if not quality["pass"]:
            print(f"❌ Idea rejected before schema build | quality score: {quality['score']} | reasons: {', '.join(quality['reasons'])}")
            self.processed.add(idea_id)
            save_processed(self.processed)
            return

        try:
            schema_dict = build_schema(idea)
            strategy_name = schema_dict["name"]
            market = infer_market_from_schema(schema_dict)
            print(f"🧠 Family: {schema_dict['family']}")
            print(f"📝 Strategy: {strategy_name}")
            print(f"🌍 Market context: {market}")
            print(f"🧪 Idea quality score: {quality['score']}")
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
            base_code = compile_strategy_class(schema_from_dict(schema_dict, source_idea=idea), class_name="GeneratedStrategy")
        except Exception as e:
            print(f"❌ Compile failed: {e}")
            self.processed.add(idea_id)
            save_processed(self.processed)
            return

        smoke_dataset = choose_smoke_dataset(schema_dict)
        ok, smoke_used, smoke_msg = smoke_test_with_fallbacks(base_code, smoke_dataset, schema_dict)
        if not ok:
            print(f"❌ Smoke test failed after fallbacks. Last error: {str(smoke_msg).splitlines()[0]}")
            self.processed.add(idea_id)
            save_processed(self.processed)
            return

        print(f"✅ Smoke test passed on {smoke_used[0]} {smoke_used[1]}")

        ordered = self.datasets_for_schema(schema_dict)
        print(f"⚙️ Running research across {len(ordered)} datasets...")

        best_schema, ranked_results, summary, code = optimize_schema_variant(schema_dict, ordered, self.max_workers)
        classification = classify_idea(summary)

        print(f"\n⏱️ Completed {summary['valid_count']} successful backtests")

        best_discovery = summary.get("best_discovery")
        if best_discovery:
            print(
                f"🌐 BEST DISCOVERY: {best_discovery['symbol']} {best_discovery['timeframe']} | "
                f"Sharpe {best_discovery['sharpe']:.2f} | Return {best_discovery['return_pct']:.1f}% | "
                f"WR {best_discovery['win_rate']:.1f}% | Trades {best_discovery['num_trades']} | "
                f"Score {best_discovery['ranking_score_discovery']:.2f}"
            )

        best_faithful = summary.get("best_faithful")
        if best_faithful:
            print(
                f"🎯 BEST FAITHFUL: {best_faithful['symbol']} {best_faithful['timeframe']} | "
                f"Sharpe {best_faithful['sharpe']:.2f} | Return {best_faithful['return_pct']:.1f}% | "
                f"WR {best_faithful['win_rate']:.1f}% | Trades {best_faithful['num_trades']} | "
                f"Score {best_faithful['ranking_score_faithful']:.2f}"
            )

        print(f"📊 Eligible results (>= {MIN_TRADES_FOR_BEST} trades): {summary['eligible_count']}")
        print(f"📦 Reliable results (>= {MIN_RELIABLE_TRADES} trades): {summary['reliable_count']}")
        print(f"🚫 Overtraded results filtered: {summary['overtraded_count']}")
        print(
            f"🧠 Research pass — discovery: {summary['research_pass_discovery']} | "
            f"faithful: {summary['research_pass_faithful']}"
        )
        print(f"🧪 Discovery quality pass: {summary['discovery_quality_pass']}")
        print(
            f"🏛️ Vault-ready hint — discovery: {summary['vault_ready_hint_discovery']} | "
            f"faithful: {summary['vault_ready_hint_faithful']}"
        )
        print(f"🏷️ Classification: {classification}")

        if (
            summary["research_pass_discovery"]
            and summary["discovery_quality_pass"]
            and not summary["research_pass_faithful"]
            and classification != "vault_candidate"
        ):
            rewrite_variants_output = test_rewrite_variants(
                best_schema,
                summary,
                ordered,
                self.max_workers,
            )

            best_faithful_item, best_faithful_candidate = _choose_best_rewrite_by_lane(
                rewrite_variants_output, "faithful"
            )
            if best_faithful_item and best_faithful_candidate:
                print(
                    f"🧬 BEST REWRITE FAITHFUL: {best_faithful_item['label']} | "
                    f"{best_faithful_candidate['symbol']} {best_faithful_candidate['timeframe']} | "
                    f"Sharpe {best_faithful_candidate['sharpe']:.2f} | Return {best_faithful_candidate['return_pct']:.1f}% | "
                    f"WR {best_faithful_candidate['win_rate']:.1f}% | Trades {best_faithful_candidate['num_trades']}"
                )

            best_discovery_item, best_discovery_candidate = _choose_best_rewrite_by_lane(
                rewrite_variants_output, "discovery"
            )
            if best_discovery_item and best_discovery_candidate:
                print(
                    f"🧬 BEST REWRITE DISCOVERY: {best_discovery_item['label']} | "
                    f"{best_discovery_candidate['symbol']} {best_discovery_candidate['timeframe']} | "
                    f"Sharpe {best_discovery_candidate['sharpe']:.2f} | Return {best_discovery_candidate['return_pct']:.1f}% | "
                    f"WR {best_discovery_candidate['win_rate']:.1f}% | Trades {best_discovery_candidate['num_trades']}"
                )

            promoted_rewrite_faithful = promote_rewrite_lane_and_optimize(
                rewrite_variants_output,
                ordered,
                self.max_workers,
                lane="faithful",
            )
            promoted_rewrite_discovery = promote_rewrite_lane_and_optimize(
                rewrite_variants_output,
                ordered,
                self.max_workers,
                lane="discovery",
            )

            if promoted_rewrite_faithful:
                promoted_summary = promoted_rewrite_faithful.get("summary", {})
                promoted_best = promoted_summary.get("best_faithful") or promoted_summary.get("best_discovery")
                if promoted_best:
                    print(
                        f"🚀 PROMOTED FAITHFUL BEST: {promoted_best['symbol']} {promoted_best['timeframe']} | "
                        f"Sharpe {promoted_best['sharpe']:.2f} | Return {promoted_best['return_pct']:.1f}% | "
                        f"WR {promoted_best['win_rate']:.1f}% | Trades {promoted_best['num_trades']}"
                    )

            if promoted_rewrite_discovery:
                promoted_summary = promoted_rewrite_discovery.get("summary", {})
                promoted_best = promoted_summary.get("best_faithful") or promoted_summary.get("best_discovery")
                if promoted_best:
                    print(
                        f"🚀 PROMOTED DISCOVERY BEST: {promoted_best['symbol']} {promoted_best['timeframe']} | "
                        f"Sharpe {promoted_best['sharpe']:.2f} | Return {promoted_best['return_pct']:.1f}% | "
                        f"WR {promoted_best['win_rate']:.1f}% | Trades {promoted_best['num_trades']}"
                    )
        else:
            if summary["research_pass_discovery"] and not summary["discovery_quality_pass"]:
                print("⏭️ Discovery branch did not clear the economic quality gate, so rewrites were skipped.")
            elif classification == "vault_candidate":
                print("🏛️ Vault candidate already found, so rewrite expansion was skipped.")

        artifact = {
            "idea": idea,
            "schema": best_schema,
            "compiled_code": code,
            "summary": summary,
            "classification": classification,
            "idea_quality": quality,
            "results": ranked_results,
            "rewrite_variants": rewrite_variants_output,
            "promoted_rewrite_faithful": promoted_rewrite_faithful,
            "promoted_rewrite_discovery": promoted_rewrite_discovery,
            "created_at_utc": datetime.utcnow().isoformat(),
        }

        out_path = RESULTS_DIR / f"{idea_id}_{strategy_name.replace(' ', '_')}.json"
        out_path.write_text(json.dumps(artifact, indent=2))
        print(f"💾 Saved: {out_path}")

        self.summary_rows.extend(ranked_results)
        if promoted_rewrite_faithful and promoted_rewrite_faithful.get("results"):
            self.summary_rows.extend(promoted_rewrite_faithful["results"])
        if promoted_rewrite_discovery and promoted_rewrite_discovery.get("results"):
            self.summary_rows.extend(promoted_rewrite_discovery["results"])
        write_summary_csv(self.summary_rows)

        self.processed.add(idea_id)
        save_processed(self.processed)

    def run(self):
        ideas = load_ideas(self.ideas_file)
        if not ideas:
            print(f"❌ No ideas found in: {self.ideas_file}")
            return

        pending = [i for i in ideas if idea_hash(i) not in self.processed]
        resolved_market = self.resolve_run_market(ideas)
        resolved_submarket = infer_futures_submarket_from_ideas_file(self.ideas_file, ideas) if resolved_market == "futures" else "-"

        print(f"📄 Ideas file: {self.ideas_file}")
        print(f"🧭 Auto-routed market: {resolved_market.upper()}")
        if resolved_market == "futures":
            print(f"🧭 Futures sub-market: {resolved_submarket.upper()}")
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
    parser.add_argument("--ideas-file", type=str, default=str(DEFAULT_IDEAS_FILE), help="Ideas file path")
    parser.add_argument(
        "--market",
        type=str,
        default="auto",
        choices=["auto", "crypto", "futures", "all"],
        help="Market routing: auto | crypto | futures | all",
    )
    args = parser.parse_args()

    ideas_file = Path(args.ideas_file)
    agent = ParallelRBIAgentV2(max_workers=args.workers, ideas_file=ideas_file, forced_market=args.market)

    if args.market == "futures":
        ALL_DATASETS[:] = FUTURES_DATASETS
        print("📈 Forced market: FUTURES")
    elif args.market == "all":
        ALL_DATASETS[:] = CRYPTO_DATASETS + FUTURES_DATASETS
        print(f"📈 Forced market: ALL ({len(ALL_DATASETS)} datasets)")
    elif args.market == "crypto":
        ALL_DATASETS[:] = CRYPTO_DATASETS
        print("📈 Forced market: CRYPTO")
    else:
        loaded_ideas = load_ideas(ideas_file)
        inferred = infer_market_from_ideas_file(ideas_file, loaded_ideas)
        if inferred == "futures":
            submarket = infer_futures_submarket_from_ideas_file(ideas_file, loaded_ideas)
            if submarket == "oil":
                ALL_DATASETS[:] = OIL_FUTURES_DATASETS
                print(f"📈 Auto market: FUTURES OIL ({len(ALL_DATASETS)} datasets)")
            elif submarket == "gold":
                ALL_DATASETS[:] = GOLD_FUTURES_DATASETS
                print(f"📈 Auto market: FUTURES GOLD ({len(ALL_DATASETS)} datasets)")
            elif submarket == "indices":
                ALL_DATASETS[:] = INDEX_FUTURES_DATASETS
                print(f"📈 Auto market: FUTURES INDICES ({len(ALL_DATASETS)} datasets)")
            else:
                ALL_DATASETS[:] = FUTURES_DATASETS
                print(f"📈 Auto market: FUTURES MIXED ({len(ALL_DATASETS)} datasets)")
        elif inferred == "crypto":
            ALL_DATASETS[:] = CRYPTO_DATASETS
            print("📈 Auto market: CRYPTO")
        else:
            ALL_DATASETS[:] = CRYPTO_DATASETS + FUTURES_DATASETS
            print(f"📈 Auto market: ALL ({len(ALL_DATASETS)} datasets)")

    if args.datasets:
        ALL_DATASETS[:] = ALL_DATASETS[: args.datasets]

    if args.idea:
        agent.process_idea(args.idea)
    else:
        agent.run()