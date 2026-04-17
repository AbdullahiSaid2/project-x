from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from backtesting import Backtest

from config import BACKTEST_CASH, DEFAULT_QTY, MODEL_NAME, SIGNAL_LOOKBACK_BARS, SYMBOLS

DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LIVE_TIMEFRAME = "1m"
MIN_ROWS_BY_TIMEFRAME = {"1m": 11, "5m": 11, "15m": 11}
PROP_PROFILE_NAME = os.getenv("PROP_PROFILE", "apex_pa_50k")


def _find_file(filename: str) -> Path:
    candidates = [
        PROJECT_ROOT / "src" / "strategies" / "manual" / filename,
        PROJECT_ROOT / "src" / "strategies" / filename,
        PROJECT_ROOT / "src" / filename,
        PROJECT_ROOT / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = sorted(PROJECT_ROOT.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not locate {filename} anywhere under project root: {PROJECT_ROOT}")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FETCHER_PATH = _find_file("fetcher.py")
SHARED_PATH = _find_file("top_bottom_ticking_shared.py")
STRAT_PATH = _find_file("ict_top_bottom_ticking.py")

_fetcher = _load_module("top_bottom_fetcher_local", FETCHER_PATH)
_shared = _load_module("top_bottom_shared_local", SHARED_PATH)
_strat = _load_module("top_bottom_strategy_local", STRAT_PATH)

get_ohlcv = _fetcher.get_ohlcv
estimate_days_back_for_live_window = getattr(_fetcher, "estimate_days_back_for_live_window", None)

# Optional live-aware hooks if your fetcher exposes them.
get_live_ohlcv = getattr(_fetcher, "get_live_ohlcv", None)
get_latest_live_ohlcv = getattr(_fetcher, "get_latest_live_ohlcv", None)

INSTRUMENTS = _shared.INSTRUMENTS
_prepare_meta = _shared._prepare_meta

ICT_TOP_BOTTOM_TICKING_TYPE2 = None
for name in (
    "ICT_TOP_BOTTOM_TICKING_TYPE2",
    "ICTTopBottomTickingType2",
    "ICTTopBottomTickingType2Baseline",
    "ICT_TOP_BOTTOM_TICKING",
):
    if hasattr(_strat, name):
        ICT_TOP_BOTTOM_TICKING_TYPE2 = getattr(_strat, name)
        break
if ICT_TOP_BOTTOM_TICKING_TYPE2 is None:
    raise ImportError("Could not find top_bottom_ticking baseline strategy class in ict_top_bottom_ticking.py")

REQUIRED_COLUMNS = [
    "entry_time_et_naive",
    "planned_entry_price",
    "planned_stop_price",
    "planned_target3_price",
    "side",
    "setup_type",
    "symbol",
]


def _estimate_days_back_for_live_window(timeframe: str, tail_rows: int) -> int:
    if callable(estimate_days_back_for_live_window):
        return int(estimate_days_back_for_live_window(timeframe, tail_rows))
    tf = timeframe.lower()
    minutes_per_bar = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}.get(tf, 1)
    minutes = tail_rows * minutes_per_bar
    return max(1, int(minutes / (60 * 24)) + 3)


def build_signal_id(signal: dict[str, Any]) -> str:
    raw = f"{signal['symbol']}|{signal['side']}|{signal['entry']}|{signal['stop']}|{signal['target']}|{signal['timestamp_et']}|{signal['setup_type']}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def normalize_signal(raw: dict[str, Any]) -> dict[str, Any]:
    signal = {
        "model_name": MODEL_NAME,
        "symbol": str(raw["symbol"]).upper(),
        "side": str(raw["side"]).upper(),
        "entry": float(raw["entry"]),
        "stop": float(raw["stop"]),
        "target": float(raw["target"]),
        "timestamp_et": str(raw["timestamp_et"]),
        "session_date_et": str(raw["session_date_et"]),
        "setup_type": str(raw["setup_type"]),
        "qty": int(raw.get("qty", DEFAULT_QTY)),
    }
    signal["signal_id"] = str(raw.get("signal_id") or build_signal_id(signal))
    return signal


def validate_signal(signal: dict[str, Any]) -> bool:
    if signal["qty"] <= 0 or signal["entry"] <= 0 or signal["stop"] <= 0 or signal["target"] <= 0:
        return False
    if signal["side"] not in {"LONG", "SHORT", "BUY", "SELL"}:
        return False

    side = signal["side"]
    if side in {"LONG", "BUY"}:
        return signal["stop"] < signal["entry"] < signal["target"]
    return signal["target"] < signal["entry"] < signal["stop"]


def _live_days_back_for_cfg(cfg: Any) -> int:
    configured_days = int(getattr(cfg, "days_back", 30))
    tail_rows = int(getattr(cfg, "tail_rows", 500))
    estimated_days = _estimate_days_back_for_live_window(cfg.timeframe, tail_rows)
    return min(configured_days, estimated_days)


def _required_rows_for_timeframe(tf: str) -> int:
    return int(MIN_ROWS_BY_TIMEFRAME.get(tf, 11))


def _fetch_live_ohlcv(symbol: str, exchange: str, timeframe: str, days_back: int, tail_rows: int) -> pd.DataFrame:
    """
    Live-only wrapper.
    Priority:
    1) get_latest_live_ohlcv(...)
    2) get_live_ohlcv(...)
    3) plain get_ohlcv(...)
    No historical fallback/merge.
    """
    if callable(get_latest_live_ohlcv):
        try:
            df = get_latest_live_ohlcv(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                days_back=days_back,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df.tail(tail_rows)
        except Exception as exc:
            print(f"[live_model] latest live fetch failed for {symbol}: {exc}")

    if callable(get_live_ohlcv):
        try:
            df = get_live_ohlcv(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                days_back=days_back,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df.tail(tail_rows)
        except Exception as exc:
            print(f"[live_model] live fetch failed for {symbol}: {exc}")

    try:
        df = get_ohlcv(
            symbol,
            exchange=exchange,
            timeframe=timeframe,
            days_back=days_back,
        )
        return df.tail(tail_rows)
    except Exception as exc:
        print(f"[live_model] fetch failed for {symbol}: {exc}")
        return pd.DataFrame()


def _load_live_df(cfg: Any) -> pd.DataFrame:
    tf = str(getattr(cfg, "timeframe", LIVE_TIMEFRAME))
    live_days_back = _live_days_back_for_cfg(cfg)
    required_rows = _required_rows_for_timeframe(tf)
    tail_rows = int(getattr(cfg, "tail_rows", 500))

    live_df = _fetch_live_ohlcv(
        cfg.symbol,
        exchange=cfg.exchange,
        timeframe=tf,
        days_back=live_days_back,
        tail_rows=tail_rows,
    )

    if live_df.empty:
        print(f"[live_model] skipping {cfg.symbol}: live path returned 0 rows for {tf}")
        return pd.DataFrame()

    if len(live_df) < required_rows:
        print(
            f"[live_model] skipping {cfg.symbol}: live rows insufficient "
            f"({len(live_df)} rows for {tf}; need at least {required_rows}). "
            f"No historical fallback in live trading mode."
        )
        return pd.DataFrame()

    print(f"[live_model] using live candles for {cfg.symbol} ({tf}) ({len(live_df)} rows)")
    return live_df


def _make_strategy_class(cfg: Any):
    return type(
        f"TopBottomLive_{cfg.symbol}",
        (ICT_TOP_BOTTOM_TICKING_TYPE2,),
        {
            "fixed_size": int(getattr(cfg, "contracts", DEFAULT_QTY)),
            "last_trade_log": [],
            "last_debug_counts": {},
        },
    )


def run_top_bottom_for_symbol(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = INSTRUMENTS[symbol]
    strategy_cls = _make_strategy_class(cfg)
    df = _load_live_df(cfg)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    bt = Backtest(df, strategy_cls, cash=BACKTEST_CASH, commission=0.0, exclusive_orders=True)
    bt.run()

    meta = pd.DataFrame(getattr(strategy_cls, "last_trade_log", []))
    if meta.empty:
        print(f"[live_model] {symbol} produced no fresh trade log rows from the strategy")
        return pd.DataFrame(), df

    meta = _prepare_meta(
        meta,
        cfg,
        variant_name="type2_baseline",
        prop_profile_name=PROP_PROFILE_NAME,
    )
    if meta.empty:
        print(f"[live_model] {symbol} produced trade rows but none survived _prepare_meta")
        return pd.DataFrame(), df
    return meta, df


def extract_fresh_entries(meta: pd.DataFrame, df: pd.DataFrame) -> list[dict[str, Any]]:
    if meta.empty or df.empty:
        return []

    latest_bar_ts = pd.Timestamp(df.index[-1])
    if latest_bar_ts.tzinfo is not None:
        latest_bar_ts = latest_bar_ts.tz_convert("America/New_York").tz_localize(None)

    lookback_start = pd.Timestamp(df.index[max(0, len(df) - SIGNAL_LOOKBACK_BARS)])
    if lookback_start.tzinfo is not None:
        lookback_start = lookback_start.tz_convert("America/New_York").tz_localize(None)

    subset = meta.copy().dropna(subset=[c for c in REQUIRED_COLUMNS if c in meta.columns])
    subset = subset[(subset["entry_time_et_naive"] >= lookback_start) & (subset["entry_time_et_naive"] <= latest_bar_ts)]

    if subset.empty:
        print(
            f"[live_model] no eligible fresh entries: latest_bar_et={latest_bar_ts.isoformat()} "
            f"lookback_start_et={lookback_start.isoformat()}"
        )
        return []

    out = []
    for _, row in subset.iterrows():
        target_col = "planned_target3_price" if "planned_target3_price" in row else "planned_target_price"
        raw = {
            "symbol": row["symbol"],
            "side": row["side"],
            "entry": float(row["planned_entry_price"]),
            "stop": float(row["planned_stop_price"]),
            "target": float(row[target_col]),
            "timestamp_et": pd.Timestamp(row["entry_time_et_naive"]).isoformat(),
            "session_date_et": str(pd.Timestamp(row["entry_time_et_naive"]).date()),
            "setup_type": str(row.get("setup_type", "UNKNOWN")),
            "qty": DEFAULT_QTY,
        }
        try:
            signal = normalize_signal(raw)
            if validate_signal(signal):
                out.append(signal)
            else:
                print(f"[live_model] invalid signal rejected for {symbol}: {raw}")
        except Exception as exc:
            print(f"[live_model] skipped malformed {symbol} signal: {exc}")
    return out


def generate_live_signals() -> list[dict[str, Any]]:
    signals = []
    for symbol in SYMBOLS:
        if symbol not in INSTRUMENTS:
            print(f"[live_model] unsupported symbol skipped: {symbol}")
            continue
        try:
            meta, df = run_top_bottom_for_symbol(symbol)
            symbol_signals = extract_fresh_entries(meta, df)
            if not symbol_signals:
                print(f"[live_model] {symbol} yielded 0 fresh signals this cycle")
            signals.extend(symbol_signals)
        except Exception as exc:
            print(f"[live_model] {symbol} scan failed: {exc}")
    return signals