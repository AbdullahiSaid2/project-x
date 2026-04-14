from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from backtesting import Backtest
from zoneinfo import ZoneInfo

from config import (
    BACKTEST_CASH,
    DEFAULT_QTY,
    MODEL_NAME,
    SIGNAL_LOOKBACK_BARS,
    SYMBOLS,
    TRADINGVIEW_BARS_DIR,
    TRADINGVIEW_FALLBACK_TO_FETCHER,
    TRADINGVIEW_MIN_BARS,
    TRADINGVIEW_SYMBOL_MAP,
    USE_TRADINGVIEW_BARS,
)

ET = ZoneInfo('America/New_York')
BASE_DIR = Path(__file__).resolve().parents[4]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data.fetcher import get_ohlcv  # noqa: E402
from src.strategies.manual.v473_shared import INSTRUMENTS, _make_strategy_class, _prepare_meta  # noqa: E402


REQUIRED_COLUMNS = [
    'entry_time_et_naive',
    'planned_entry_price',
    'planned_stop_price',
    'planned_target_price',
    'side',
    'setup_type',
    'setup_tier',
    'bridge_type',
    'symbol',
]


def now_et() -> datetime:
    return datetime.now(timezone.utc).astimezone(ET)


def build_signal_id(signal: dict[str, Any]) -> str:
    raw = (
        f"{signal['symbol']}|{signal['side']}|{signal['entry']}|{signal['stop']}|"
        f"{signal['target']}|{signal['timestamp_et']}|{signal['setup_type']}|{signal['bridge_type']}"
    )
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]


def normalize_signal(raw: dict[str, Any]) -> dict[str, Any]:
    signal = {
        'model_name': MODEL_NAME,
        'symbol': str(raw['symbol']).upper(),
        'side': str(raw['side']).upper(),
        'entry': float(raw['entry']),
        'stop': float(raw['stop']),
        'target': float(raw['target']),
        'timestamp_et': str(raw['timestamp_et']),
        'session_date_et': str(raw['session_date_et']),
        'setup_type': str(raw['setup_type']),
        'setup_tier': str(raw.get('setup_tier', 'B')),
        'bridge_type': str(raw.get('bridge_type', 'UNKNOWN')),
        'qty': int(raw.get('qty', DEFAULT_QTY)),
    }
    signal['signal_id'] = str(raw.get('signal_id') or build_signal_id(signal))
    return signal


def validate_signal(signal: dict[str, Any]) -> bool:
    if signal['qty'] <= 0:
        return False
    if signal['entry'] <= 0 or signal['stop'] <= 0 or signal['target'] <= 0:
        return False
    if signal['side'] not in {'LONG', 'SHORT'}:
        return False
    if signal['side'] == 'LONG':
        return signal['stop'] < signal['entry'] < signal['target']
    return signal['target'] < signal['entry'] < signal['stop']


def _tv_timeframe_key(strategy_timeframe: str) -> str:
    tf = str(strategy_timeframe).strip().lower()
    mapping = {
        '1m': '1',
        '3m': '3',
        '5m': '5',
        '15m': '15',
        '30m': '30',
        '45m': '45',
        '60m': '60',
        '1h': '60',
        '2h': '120',
        '4h': '240',
        '1d': '1D',
        'd': '1D',
    }
    return mapping.get(tf, strategy_timeframe)


def _parse_bar_time(value: Any) -> pd.Timestamp:
    text = str(value).strip()
    if text.isdigit():
        return pd.to_datetime(int(text), unit='ms', utc=True).tz_convert(ET).tz_localize(None)
    ts = pd.to_datetime(text, utc=True, errors='coerce')
    if pd.isna(ts):
        raise ValueError(f'Unparseable TradingView bar_time: {value}')
    return ts.tz_convert(ET).tz_localize(None)


def _tv_file_for_symbol(strategy_symbol: str) -> tuple[Path, str, str]:
    cfg = INSTRUMENTS[strategy_symbol]
    tv_symbol = TRADINGVIEW_SYMBOL_MAP.get(strategy_symbol.upper(), f'{strategy_symbol.upper()}1!')
    tv_timeframe = _tv_timeframe_key(cfg.timeframe)
    path = TRADINGVIEW_BARS_DIR / f'{tv_symbol}__{tv_timeframe}.json'
    return path, tv_symbol, tv_timeframe


def load_tradingview_ohlcv(strategy_symbol: str) -> pd.DataFrame:
    path, tv_symbol, tv_timeframe = _tv_file_for_symbol(strategy_symbol)
    if not path.exists():
        raise FileNotFoundError(f'TradingView bars file not found: {path}')

    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'TradingView bars file is empty: {path}')

    cleaned: list[dict[str, Any]] = []
    for row in rows:
        try:
            cleaned.append(
                {
                    'Date': _parse_bar_time(row.get('bar_time')),
                    'Open': float(row['open']),
                    'High': float(row['high']),
                    'Low': float(row['low']),
                    'Close': float(row['close']),
                    'Volume': float(row.get('volume', 0.0)),
                }
            )
        except Exception as exc:
            print(f'[live_model] skipped malformed TradingView candle for {strategy_symbol}: {exc}')

    if not cleaned:
        raise ValueError(f'No usable TradingView candles found in: {path}')

    df = pd.DataFrame(cleaned).drop_duplicates(subset=['Date']).sort_values('Date').set_index('Date')
    if len(df) < TRADINGVIEW_MIN_BARS:
        raise ValueError(
            f'TradingView candle history too short for {strategy_symbol}: '
            f'{len(df)} < {TRADINGVIEW_MIN_BARS} from {tv_symbol}/{tv_timeframe}'
        )

    cfg = INSTRUMENTS[strategy_symbol]
    df = df.tail(cfg.tail_rows)
    print(f'[live_model] using TradingView candles for {strategy_symbol} from {path} ({len(df)} rows)')
    return df


def load_model_ohlcv(strategy_symbol: str) -> tuple[pd.DataFrame, str]:
    cfg = INSTRUMENTS[strategy_symbol]

    if USE_TRADINGVIEW_BARS:
        try:
            return load_tradingview_ohlcv(strategy_symbol), 'tradingview'
        except Exception as exc:
            print(f'[live_model] TradingView fallback for {strategy_symbol}: {exc}')
            if not TRADINGVIEW_FALLBACK_TO_FETCHER:
                raise

    df = get_ohlcv(cfg.symbol, exchange=cfg.exchange, timeframe=cfg.timeframe, days_back=cfg.days_back)
    return df, 'fetcher'


def run_exact_v473_for_symbol(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = INSTRUMENTS[symbol]
    strategy_cls = _make_strategy_class(cfg)
    df, source = load_model_ohlcv(symbol)
    df = df.tail(cfg.tail_rows)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    bt = Backtest(df, strategy_cls, cash=BACKTEST_CASH, commission=0.0, exclusive_orders=True)
    bt.run()

    meta = pd.DataFrame(getattr(strategy_cls, 'last_trade_log', []))
    if meta.empty:
        print(f'[live_model] no entries for {symbol} using {source}')
        return pd.DataFrame(), pd.DataFrame()
    meta = _prepare_meta(meta, cfg)
    if meta.empty:
        print(f'[live_model] prepared meta empty for {symbol} using {source}')
        return pd.DataFrame(), pd.DataFrame()
    return meta, df


def extract_fresh_entries(meta: pd.DataFrame, df: pd.DataFrame) -> list[dict[str, Any]]:
    if meta.empty or df.empty:
        return []

    latest_bar_ts = pd.Timestamp(df.index[-1])
    if latest_bar_ts.tzinfo is not None:
        latest_bar_ts = latest_bar_ts.tz_convert('America/New_York').tz_localize(None)
    lookback_start = pd.Timestamp(df.index[max(0, len(df) - SIGNAL_LOOKBACK_BARS)])
    if lookback_start.tzinfo is not None:
        lookback_start = lookback_start.tz_convert('America/New_York').tz_localize(None)

    out: list[dict[str, Any]] = []
    subset = meta.copy()
    subset = subset.dropna(subset=[c for c in REQUIRED_COLUMNS if c in subset.columns])
    subset = subset[subset['entry_time_et_naive'] >= lookback_start]
    subset = subset[subset['entry_time_et_naive'] <= latest_bar_ts]

    for _, row in subset.iterrows():
        raw = {
            'symbol': row['symbol'],
            'side': row['side'],
            'entry': float(row['planned_entry_price']),
            'stop': float(row['planned_stop_price']),
            'target': float(row['planned_target_price']),
            'timestamp_et': pd.Timestamp(row['entry_time_et_naive']).isoformat(),
            'session_date_et': str(pd.Timestamp(row['entry_time_et_naive']).date()),
            'setup_type': str(row.get('setup_type', 'UNKNOWN')),
            'setup_tier': str(row.get('setup_tier', 'B')),
            'bridge_type': str(row.get('bridge_type', 'UNKNOWN')),
            'qty': DEFAULT_QTY,
        }
        try:
            signal = normalize_signal(raw)
            if validate_signal(signal):
                out.append(signal)
        except Exception as exc:
            print(f'[live_model] skipped malformed signal row: {exc}')
    return out


def generate_live_signals() -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        if symbol not in INSTRUMENTS:
            print(f'[live_model] unsupported symbol skipped: {symbol}')
            continue
        try:
            meta, df = run_exact_v473_for_symbol(symbol)
            signals.extend(extract_fresh_entries(meta, df))
        except Exception as exc:
            print(f'[live_model] {symbol} scan failed: {exc}')
    return signals
