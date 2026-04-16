
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

ENTRY_REQUIRED_COLUMNS = [
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


def build_action_id(action: dict[str, Any]) -> str:
    raw = json.dumps(action, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]


def normalize_side(side: str) -> str:
    side_u = str(side).upper()
    if side_u in {'LONG', 'BUY'}:
        return 'buy'
    if side_u in {'SHORT', 'SELL'}:
        return 'sell'
    if side_u == 'EXIT':
        return 'exit'
    return side_u.lower()


def normalize_entry_signal(raw: dict[str, Any]) -> dict[str, Any]:
    signal = {
        'action_type': 'entry',
        'model_name': MODEL_NAME,
        'symbol': str(raw['symbol']).upper(),
        'side': normalize_side(str(raw['side'])),
        'entry': float(raw['entry']),
        'stop': float(raw['stop']),
        'target': float(raw['target']),
        'partial_target': float(raw.get('partial_target')) if raw.get('partial_target') not in (None, '') else None,
        'runner_target': float(raw.get('runner_target')) if raw.get('runner_target') not in (None, '') else None,
        'timestamp_et': str(raw['timestamp_et']),
        'session_date_et': str(raw['session_date_et']),
        'setup_type': str(raw.get('setup_type', 'UNKNOWN')),
        'setup_tier': str(raw.get('setup_tier', 'B')),
        'bridge_type': str(raw.get('bridge_type', 'UNKNOWN')),
        'qty': int(raw.get('qty', DEFAULT_QTY)),
    }
    signal['signal_id'] = str(raw.get('signal_id') or build_action_id(signal))
    return signal


def normalize_v473_event(symbol: str, raw: dict[str, Any]) -> dict[str, Any]:
    event_type = str(raw.get('event_type', '')).strip()
    action = {
        'action_type': event_type,
        'model_name': MODEL_NAME,
        'symbol': symbol,
        'timestamp_et': str(raw.get('timestamp_et', '')),
        'setup_type': str(raw.get('setup_type', '')),
        'setup_tier': str(raw.get('setup_tier', '')),
        'bridge_type': str(raw.get('bridge_type', '')),
        'qty': int(raw.get('qty', DEFAULT_QTY) or DEFAULT_QTY),
        'side': normalize_side(str(raw.get('side', ''))),
        'entry': float(raw['planned_entry_price']) if raw.get('planned_entry_price') not in (None, '') and pd.notna(raw.get('planned_entry_price')) else 0.0,
        'stop': float(raw['planned_stop_price']) if raw.get('planned_stop_price') not in (None, '') and pd.notna(raw.get('planned_stop_price')) else 0.0,
        'target': float(raw['planned_target_price']) if raw.get('planned_target_price') not in (None, '') and pd.notna(raw.get('planned_target_price')) else 0.0,
        'partial_target': float(raw['partial_target_price']) if raw.get('partial_target_price') not in (None, '') and pd.notna(raw.get('partial_target_price')) else None,
        'runner_target': float(raw['runner_target_price']) if raw.get('runner_target_price') not in (None, '') and pd.notna(raw.get('runner_target_price')) else None,
        'portion': float(raw['portion']) if raw.get('portion') not in (None, '') and pd.notna(raw.get('portion')) else None,
        'new_stop': float(raw['new_stop']) if raw.get('new_stop') not in (None, '') and pd.notna(raw.get('new_stop')) else None,
        'rr_now': float(raw['rr_now']) if raw.get('rr_now') not in (None, '') and pd.notna(raw.get('rr_now')) else None,
        'entry_price': float(raw['entry_price']) if raw.get('entry_price') not in (None, '') and pd.notna(raw.get('entry_price')) else None,
        'exit_price': float(raw['exit_price']) if raw.get('exit_price') not in (None, '') and pd.notna(raw.get('exit_price')) else None,
        'pnl': float(raw['pnl']) if raw.get('pnl') not in (None, '') and pd.notna(raw.get('pnl')) else None,
    }
    action['signal_id'] = build_action_id(action)
    return action


def validate_entry_signal(signal: dict[str, Any]) -> bool:
    if signal['qty'] <= 0:
        return False
    if signal['entry'] <= 0 or signal['stop'] <= 0 or signal['target'] <= 0:
        return False
    if signal['side'] not in {'buy', 'sell'}:
        return False
    if signal['side'] == 'buy':
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
        raise ValueError(f'TradingView bars file empty or malformed: {path}')
    df = pd.DataFrame(rows)
    needed = {'open', 'high', 'low', 'close', 'volume', 'bar_time'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'TradingView bars missing columns {sorted(missing)} in {path}')
    df['Date'] = df['bar_time'].apply(_parse_bar_time)
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().drop_duplicates(subset=['Date']).sort_values('Date').set_index('Date')
    if len(df) < TRADINGVIEW_MIN_BARS:
        raise ValueError(
            f'TradingView bars for {strategy_symbol} have only {len(df)} rows; '
            f'need at least {TRADINGVIEW_MIN_BARS}'
        )
    print(f'[live_model] using TradingView candles for {strategy_symbol} from {path} ({len(df)} rows)')
    return df


def get_live_ohlcv_for_symbol(strategy_symbol: str) -> pd.DataFrame:
    cfg = INSTRUMENTS[strategy_symbol]
    if USE_TRADINGVIEW_BARS:
        try:
            return load_tradingview_ohlcv(strategy_symbol)
        except Exception as exc:
            if not TRADINGVIEW_FALLBACK_TO_FETCHER:
                raise
            print(f'[live_model] TradingView fallback for {strategy_symbol}: {exc}')
    return get_ohlcv(cfg.symbol, exchange=cfg.exchange, timeframe=cfg.timeframe, days_back=cfg.days_back)


def run_exact_v473_for_symbol(symbol: str) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    cfg = INSTRUMENTS[symbol]
    strategy_cls = _make_strategy_class(cfg)
    df = get_live_ohlcv_for_symbol(symbol)
    df = df.tail(cfg.tail_rows)
    if df.empty:
        return pd.DataFrame(), [], pd.DataFrame()

    bt = Backtest(df, strategy_cls, cash=BACKTEST_CASH, commission=0.0, exclusive_orders=True)
    bt.run()

    meta = pd.DataFrame(getattr(strategy_cls, 'last_trade_log', []))
    if not meta.empty:
        meta = _prepare_meta(meta, cfg)
    events = list(getattr(strategy_cls, 'last_event_log', []))
    return meta, events, df


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
    subset = subset.dropna(subset=[c for c in ENTRY_REQUIRED_COLUMNS if c in subset.columns])
    subset = subset[subset['entry_time_et_naive'] >= lookback_start]
    subset = subset[subset['entry_time_et_naive'] <= latest_bar_ts]

    for _, row in subset.iterrows():
        raw = {
            'symbol': row['symbol'],
            'side': row['side'],
            'entry': float(row['planned_entry_price']),
            'stop': float(row['planned_stop_price']),
            'target': float(row['planned_target_price']),
            'partial_target': row.get('partial_target_price'),
            'runner_target': row.get('runner_target_price'),
            'timestamp_et': pd.Timestamp(row['entry_time_et_naive']).isoformat(),
            'session_date_et': str(pd.Timestamp(row['entry_time_et_naive']).date()),
            'setup_type': str(row.get('setup_type', 'UNKNOWN')),
            'setup_tier': str(row.get('setup_tier', 'B')),
            'bridge_type': str(row.get('bridge_type', 'UNKNOWN')),
            'qty': DEFAULT_QTY,
        }
        signal = normalize_entry_signal(raw)
        if validate_entry_signal(signal):
            out.append(signal)
    return out


def extract_fresh_events(symbol: str, events: list[dict[str, Any]], df: pd.DataFrame) -> list[dict[str, Any]]:
    if not events or df.empty:
        return []
    latest_bar_ts = pd.Timestamp(df.index[-1])
    if latest_bar_ts.tzinfo is not None:
        latest_bar_ts = latest_bar_ts.tz_convert('America/New_York').tz_localize(None)
    lookback_start = pd.Timestamp(df.index[max(0, len(df) - SIGNAL_LOOKBACK_BARS)])
    if lookback_start.tzinfo is not None:
        lookback_start = lookback_start.tz_convert('America/New_York').tz_localize(None)

    out = []
    for ev in events:
        ts = pd.to_datetime(ev.get('timestamp_et'), errors='coerce')
        if pd.isna(ts):
            continue
        if ts.tzinfo is not None:
            ts = ts.tz_convert('America/New_York').tz_localize(None)
        if ts < lookback_start or ts > latest_bar_ts:
            continue
        out.append(normalize_v473_event(symbol, ev))
    return out


def generate_live_actions() -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        if symbol not in INSTRUMENTS:
            print(f'[live_model] unsupported symbol skipped: {symbol}')
            continue
        try:
            meta, events, df = run_exact_v473_for_symbol(symbol)
            actions.extend(extract_fresh_entries(meta, df))
            actions.extend(extract_fresh_events(symbol, events, df))
        except Exception as exc:
            print(f'[live_model] {symbol} scan failed: {exc}')
    return actions


def generate_live_signals() -> list[dict[str, Any]]:
    return generate_live_actions()
