from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from backtesting import Backtest

from config import DEFAULT_QTY, MODEL_NAME, SIGNAL_LOOKBACK_BARS, SYMBOLS

warnings.filterwarnings(
    'ignore',
    message='Some prices are larger than initial cash value.*',
    category=UserWarning,
)

ET = ZoneInfo('America/New_York')
DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parents[3] if len(DEPLOY_DIR.parents) >= 4 else DEPLOY_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MIN_ROWS_BY_TIMEFRAME = {'1m': 40, '5m': 40, '15m': 30, '30m': 20, '1h': 12}
PROP_PROFILE_NAME = os.getenv('PROP_PROFILE', 'apex_pa_50k')
LIVE_BACKTEST_CASH = 1_000_000.0
LIVE_STALE_MINUTES = int(os.getenv('TOP_BOTTOM_TICKING_LIVE_STALE_MINUTES', '20'))
ENABLE_FILE_LOGGING = os.getenv('TOP_BOTTOM_TICKING_FILE_LOGGING', '1').lower() not in {'0', 'false', 'no'}
LOG_DIR = DEPLOY_DIR / 'logs'

LIVE_SYMBOL_STOP_OVERRIDES = {
    'MNQ': {'min_stop_points': None, 'max_stop_points': 220.0},
    'MES': {'min_stop_points': None, 'max_stop_points': 30.0},
    'MYM': {'min_stop_points': None, 'max_stop_points': 250.0},
    'MGC': {'min_stop_points': None, 'max_stop_points': 20.0},
    'MCL': {'min_stop_points': None, 'max_stop_points': 1.20},
}

LIVE_SYMBOL_SANITY_BANDS = {
    'MNQ': {'min_price': 5000.0, 'max_price': 50000.0, 'max_bar_return': 0.08},
    'MES': {'min_price': 1000.0, 'max_price': 20000.0, 'max_bar_return': 0.08},
    'MYM': {'min_price': 5000.0, 'max_price': 100000.0, 'max_bar_return': 0.08},
    'MGC': {'min_price': 500.0, 'max_price': 10000.0, 'max_bar_return': 0.08},
    'MCL': {'min_price': 20.0, 'max_price': 200.0, 'max_bar_return': 0.20},
}

REQUIRED_COLUMNS = [
    'entry_time_et_naive',
    'planned_entry_price',
    'planned_stop_price',
    'side',
    'setup_type',
    'symbol',
]

TARGET_PRICE_CANDIDATES = [
    'planned_target_price',
    'planned_target3_price',
    'planned_target2_price',
    'planned_target1_price',
]

QTY_CANDIDATES = ['qty', 'report_contracts', 'contracts', 'fixed_size']
ENTRY_TIME_CANDIDATES = [
    'entry_time_et_naive',
    'entry_time_et',
    'entry_time',
    'planned_entry_time_et',
    'planned_entry_time',
    'timestamp_et',
    'timestamp',
]


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz='UTC')


def _today_log_path() -> Path:
    now_et = datetime.now(timezone.utc).astimezone(ET)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f'live_model_{now_et:%Y%m%d}.log'


def get_log_file_path() -> Path:
    return _today_log_path()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return 'NaN'
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
    return value


def _debug(msg: str) -> None:
    line = f'[live_model] {msg}'
    print(line)
    if ENABLE_FILE_LOGGING:
        try:
            with _today_log_path().open('a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception:
            pass


# ---------- module loading ----------
def _find_file(filename: str) -> Path:
    candidates = [
        DEPLOY_DIR / filename,
        PROJECT_ROOT / 'src' / 'strategies' / 'deployed' / 'top_bottom_ticking' / filename,
        PROJECT_ROOT / 'src' / 'strategies' / 'manual' / filename,
        PROJECT_ROOT / 'src' / 'strategies' / filename,
        PROJECT_ROOT / 'src' / 'data' / filename,
        PROJECT_ROOT / 'src' / filename,
        PROJECT_ROOT / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = sorted(PROJECT_ROOT.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f'Could not locate {filename} anywhere under project root: {PROJECT_ROOT}')


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load module {name} from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SHARED_PATH = _find_file('top_bottom_ticking_shared.py')
STRAT_PATH = _find_file('ict_top_bottom_ticking.py')
DATABENTO_LIVE_PATH = _find_file('databento_live.py')

_shared = _load_module('top_bottom_shared_local_runtime', SHARED_PATH)
_strat = _load_module('ict_top_bottom_ticking_local_runtime', STRAT_PATH)
_db_live = _load_module('databento_live_local_runtime', DATABENTO_LIVE_PATH)

get_live_ohlcv = _db_live.get_live_ohlcv
INSTRUMENTS = _shared.INSTRUMENTS
SYMBOL_SPECS = _shared.SYMBOL_SPECS
_prepare_meta = _shared._prepare_meta

ICT_TOP_BOTTOM_TICKING = None
for name in ('ICT_TOP_BOTTOM_TICKING', 'ICTTopBottomTickingType2Baseline', 'ICT_TOP_BOTTOM_TICKING_TYPE2'):
    if hasattr(_strat, name):
        ICT_TOP_BOTTOM_TICKING = getattr(_strat, name)
        break
if ICT_TOP_BOTTOM_TICKING is None:
    raise ImportError(
        f'Could not find ICT strategy class in {STRAT_PATH}. '
        f'Expected one of: ICT_TOP_BOTTOM_TICKING, '
        f'ICTTopBottomTickingType2Baseline, ICT_TOP_BOTTOM_TICKING_TYPE2'
    )


def _to_naive_et_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert(ET).tz_localize(None)


def _coerce_naive_et(values: Any):
    ts = pd.to_datetime(values, errors='coerce')

    if isinstance(ts, pd.Series):
        try:
            if ts.dt.tz is not None:
                return ts.dt.tz_convert(ET).dt.tz_localize(None)
        except Exception:
            pass
        return ts

    if isinstance(ts, pd.DatetimeIndex):
        if ts.tz is not None:
            return ts.tz_convert(ET).tz_localize(None)
        return ts

    if isinstance(ts, pd.Timestamp):
        if pd.isna(ts):
            return ts
        if ts.tzinfo is not None:
            return ts.tz_convert(ET).tz_localize(None)
        return ts

    return ts


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    idx = pd.DatetimeIndex(out.index)

    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    else:
        idx = idx.tz_convert('UTC')

    out.index = idx
    out = out.sort_index()
    out = out[~out.index.duplicated(keep='last')]
    return out


def _to_bt_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)

    if idx.tz is None:
        idx = idx.tz_localize('UTC')

    idx = idx.tz_convert(ET).tz_localize(None)
    out.index = idx
    return out


def _heartbeat(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    if df.empty:
        _debug(f'{symbol} heartbeat | timeframe={timeframe} | rows=0')
        return

    last_ts = pd.Timestamp(df.index[-1])
    last_ts_utc = last_ts.tz_localize('UTC') if last_ts.tzinfo is None else last_ts.tz_convert('UTC')
    last_ts_et = last_ts_utc.tz_convert(ET)
    last_close = float(df['Close'].iloc[-1])

    _debug(
        f'{symbol} heartbeat | timeframe={timeframe} | rows={len(df)} | '
        f'latest_bar_utc={last_ts_utc.isoformat()} | latest_bar_et={last_ts_et.isoformat()} | '
        f'latest_close={last_close}'
    )


def _clean_live_df(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = _ensure_utc_index(df)

    required = ['Open', 'High', 'Low', 'Close']
    missing = [c for c in required if c not in out.columns]
    if missing:
        _debug(f'skipping {symbol}: missing OHLC columns {missing}')
        return pd.DataFrame()

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    before = len(out)
    out = out.dropna(subset=required)

    bad_non_positive = (out[required] <= 0).any(axis=1)
    if bad_non_positive.any():
        _debug(f'{symbol} cleansed {int(bad_non_positive.sum())} poisoned live rows before scan')
        out = out.loc[~bad_non_positive].copy()

    if out.empty:
        return out

    bad_structure = (
        (out['High'] < out[['Open', 'Close', 'Low']].max(axis=1))
        | (out['Low'] > out[['Open', 'Close', 'High']].min(axis=1))
        | (out['High'] < out['Low'])
    )
    if bad_structure.any():
        _debug(f'{symbol} cleansed {int(bad_structure.sum())} structurally invalid live rows before scan')
        out = out.loc[~bad_structure].copy()

    if out.empty:
        return out

    band = LIVE_SYMBOL_SANITY_BANDS.get(symbol, {})
    min_price = band.get('min_price')
    max_price = band.get('max_price')
    max_bar_return = band.get('max_bar_return')

    if min_price is not None:
        bad_low = out['Close'] < float(min_price)
        if bad_low.any():
            _debug(f'{symbol} cleansed {int(bad_low.sum())} rows below sanity floor {min_price}')
            out = out.loc[~bad_low].copy()

    if out.empty:
        return out

    if max_price is not None:
        bad_high = out['Close'] > float(max_price)
        if bad_high.any():
            _debug(f'{symbol} cleansed {int(bad_high.sum())} rows above sanity ceiling {max_price}')
            out = out.loc[~bad_high].copy()

    if out.empty:
        return out

    if max_bar_return is not None and len(out) >= 2:
        rets = out['Close'].pct_change().abs().replace([np.inf, -np.inf], np.nan)
        bad_jump = rets > float(max_bar_return)
        if bad_jump.any():
            _debug(f'{symbol} cleansed {int(bad_jump.sum())} jumpy live rows before scan')
            out = out.loc[~bad_jump].copy()

    out = out.sort_index()
    out = out[~out.index.duplicated(keep='last')]

    removed_total = before - len(out)
    if removed_total > 0:
        _debug(f'{symbol} total live rows removed during cleansing: {removed_total}')

    return out


def _latest_bar_is_fresh(df: pd.DataFrame, symbol: str) -> bool:
    if df.empty:
        return False

    latest_bar_utc = pd.Timestamp(df.index[-1])
    latest_bar_utc = latest_bar_utc.tz_localize('UTC') if latest_bar_utc.tzinfo is None else latest_bar_utc.tz_convert('UTC')
    latest_bar_et = latest_bar_utc.tz_convert(ET)
    age_minutes = (_utc_now() - latest_bar_utc).total_seconds() / 60.0

    if age_minutes > LIVE_STALE_MINUTES:
        _debug(
            f'skipping {symbol}: refusing stale data for live mode | '
            f'latest_bar_et={latest_bar_et.isoformat()} | age_minutes={age_minutes:.1f}'
        )
        return False

    return True


def _safe_simple(v: Any) -> bool:
    return isinstance(v, (int, float, bool, str)) or v is None


def _snapshot_debug_from_instance(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}

    try:
        if hasattr(obj, 'debug_counts'):
            for k, v in dict(getattr(obj, 'debug_counts', {})).items():
                if _safe_simple(v):
                    out[k] = _json_safe(v)
    except Exception:
        pass

    if hasattr(obj, 'pending'):
        try:
            pending = getattr(obj, 'pending')
            if pending is not None:
                for k, v in getattr(pending, '__dict__', {}).items():
                    if _safe_simple(v):
                        out[f'pending_{k}'] = _json_safe(v)
        except Exception:
            pass
    return out


def build_signal_id(signal: dict[str, Any]) -> str:
    raw = f"{signal['symbol']}|{signal['side']}|{signal['entry']}|{signal['stop']}|{signal['target']}|{signal['timestamp_et']}|{signal['setup_type']}|{signal['qty']}"
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
        'qty': int(raw.get('qty', DEFAULT_QTY)),
    }
    signal['signal_id'] = str(raw.get('signal_id') or build_signal_id(signal))
    return signal


def validate_signal(signal: dict[str, Any]) -> bool:
    if signal['qty'] <= 0 or signal['entry'] <= 0 or signal['stop'] <= 0 or signal['target'] <= 0:
        return False
    if signal['side'] not in {'LONG', 'SHORT', 'BUY', 'SELL'}:
        return False
    if signal['side'] in {'LONG', 'BUY'}:
        return signal['stop'] < signal['entry'] < signal['target']
    return signal['target'] < signal['entry'] < signal['stop']


def _required_rows_for_timeframe(tf: str) -> int:
    return int(MIN_ROWS_BY_TIMEFRAME.get(tf, 40))


def _load_live_df(cfg: Any) -> pd.DataFrame:
    tf = str(getattr(cfg, 'timeframe', '5m'))
    tail_rows = min(int(getattr(cfg, 'tail_rows', 120000)), 5000)
    required_rows = max(_required_rows_for_timeframe(tf), SIGNAL_LOOKBACK_BARS + 5)

    try:
        live_df = get_live_ohlcv(symbol=cfg.symbol, timeframe=tf, tail_rows=tail_rows)
    except Exception as exc:
        _debug(f'{cfg.symbol} live fetch failed: {exc}')
        return pd.DataFrame()

    if live_df.empty:
        _debug(f'skipping {cfg.symbol}: true live feed returned 0 rows')
        return pd.DataFrame()

    live_df = _clean_live_df(cfg.symbol, live_df)
    if live_df.empty:
        _debug(f'skipping {cfg.symbol}: no usable live rows after cleansing')
        return pd.DataFrame()

    _heartbeat(cfg.symbol, tf, live_df)

    if not _latest_bar_is_fresh(live_df, cfg.symbol):
        return pd.DataFrame()

    if len(live_df) < required_rows:
        _debug(
            f'skipping {cfg.symbol}: insufficient live rows after cleansing | '
            f'rows={len(live_df)} timeframe={tf}'
        )
        return pd.DataFrame()

    _debug(
        f'using TRUE live candles for {cfg.symbol} ({tf}) ({len(live_df)} rows) '
        f'start={live_df.index[0]} end={live_df.index[-1]}'
    )
    return live_df


def _ensure_meta_columns(meta: pd.DataFrame, symbol: str, cfg: Any) -> pd.DataFrame:
    if meta.empty:
        return meta

    out = meta.copy()

    if 'symbol' not in out.columns or out['symbol'].isna().all():
        out['symbol'] = symbol

    if 'entry_time_et_naive' not in out.columns:
        filled = False
        for col in ENTRY_TIME_CANDIDATES:
            if col in out.columns:
                out['entry_time_et_naive'] = _coerce_naive_et(out[col])
                filled = True
                break
        if not filled:
            out['entry_time_et_naive'] = pd.NaT
    else:
        out['entry_time_et_naive'] = _coerce_naive_et(out['entry_time_et_naive'])

    if 'planned_target_price' not in out.columns:
        for col in TARGET_PRICE_CANDIDATES:
            if col in out.columns:
                out['planned_target_price'] = pd.to_numeric(out[col], errors='coerce')
                break
        if 'planned_target_price' not in out.columns:
            out['planned_target_price'] = np.nan

    for col in ['planned_entry_price', 'planned_stop_price', 'planned_target_price']:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    qty = None
    for col in QTY_CANDIDATES:
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors='coerce').dropna()
            if not vals.empty:
                qty = int(vals.iloc[0])
                break
    if qty is None:
        qty = int(getattr(cfg, 'contracts', DEFAULT_QTY))
    out['qty'] = qty

    return out


def _make_strategy_class(cfg: Any):
    base = ICT_TOP_BOTTOM_TICKING
    spec = SYMBOL_SPECS[cfg.symbol]
    is_sniper = False
    live_override = LIVE_SYMBOL_STOP_OVERRIDES.get(cfg.symbol, {})

    manual_min_stop = spec.min_stop_sniper if is_sniper else spec.min_stop_baseline
    manual_max_stop = spec.max_stop_sniper if is_sniper else spec.max_stop_baseline
    effective_min_stop = manual_min_stop if live_override.get('min_stop_points') is None else live_override['min_stop_points']
    effective_max_stop = manual_max_stop if live_override.get('max_stop_points') is None else live_override['max_stop_points']

    class StrategyCls(base):
        fixed_size = int(getattr(cfg, 'contracts', DEFAULT_QTY))
        tick_size = spec.tick_size
        setup_expiry_bars = spec.expiry_sniper if is_sniper else spec.expiry_baseline
        limit_touch_tolerance_ticks = spec.touch_tol_sniper_ticks if is_sniper else spec.touch_tol_baseline_ticks
        require_cos_confirmation = spec.require_cos_sniper if is_sniper else spec.require_cos_baseline
        require_internal_sweep_filter = spec.require_internal_sniper if is_sniper else spec.require_internal_baseline
        min_stop_points = effective_min_stop
        max_stop_points = effective_max_stop
        last_trade_log = []
        last_debug_counts = {}

        def next(self):
            super().next()
            try:
                snap = _snapshot_debug_from_instance(self)
                snap['effective_min_stop_points'] = float(type(self).min_stop_points)
                snap['effective_max_stop_points'] = float(type(self).max_stop_points)
                snap['manual_min_stop_points'] = float(manual_min_stop)
                snap['manual_max_stop_points'] = float(manual_max_stop)
                type(self).last_debug_counts = snap
            except Exception:
                type(self).last_debug_counts = {}

    StrategyCls.__name__ = f'ICT_TOP_BOTTOM_TICKING_{cfg.symbol}'
    return StrategyCls


def _run_backtest_pass(symbol: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], Any]:
    cfg = INSTRUMENTS[symbol]
    StrategyCls = _make_strategy_class(cfg)
    bt_df = _to_bt_index(df)
    bt = Backtest(bt_df, StrategyCls, cash=LIVE_BACKTEST_CASH, commission=0.0, exclusive_orders=True, trade_on_close=False)
    bt.run()

    raw_meta = pd.DataFrame(getattr(StrategyCls, 'last_trade_log', []))
    raw_debug = getattr(StrategyCls, 'last_debug_counts', {}) or {}
    try:
        meta = _prepare_meta(raw_meta, cfg, 'type2_baseline', PROP_PROFILE_NAME)
    except TypeError:
        meta = _prepare_meta(raw_meta, cfg)
    meta = _ensure_meta_columns(meta, symbol, cfg)
    return meta, raw_debug, cfg


def run_top_bottom_for_symbol(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = INSTRUMENTS[symbol]
    df = _load_live_df(cfg)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    meta, raw_debug, _ = _run_backtest_pass(symbol, df)
    if raw_debug:
        _debug(f'{symbol} strategy debug snapshot: {json.dumps(_json_safe(raw_debug), ensure_ascii=False)}')
        if not raw_debug.get('pending_direction'):
            _debug(f'{symbol} no active pending setup remains after strategy pass')
    _debug(f'{symbol} raw trade log rows before prepare_meta: {len(meta)}')

    if meta.empty:
        _debug(f'{symbol} produced no fresh trade log rows from the strategy')
        return pd.DataFrame(), df

    _debug(f'{symbol} prepared trade log rows after prepare_meta: {len(meta)}')
    return meta, df


def extract_fresh_entries(meta: pd.DataFrame, df: pd.DataFrame) -> list[dict[str, Any]]:
    if meta.empty or df.empty:
        return []

    latest_bar_ts = _to_naive_et_timestamp(df.index[-1])
    lookback_idx = max(0, len(df) - SIGNAL_LOOKBACK_BARS)
    lookback_start = _to_naive_et_timestamp(df.index[lookback_idx])

    _debug(
        f'fresh-entry window: latest_bar_et={latest_bar_ts.isoformat()} '
        f'lookback_start_et={lookback_start.isoformat()} total_meta_rows={len(meta)}'
    )

    subset = meta.copy().dropna(subset=[c for c in REQUIRED_COLUMNS if c in meta.columns])
    _debug(f'meta rows after required-column dropna: {len(subset)}')

    if 'entry_time_et_naive' not in subset.columns:
        _debug('meta rows missing entry_time_et_naive after normalization')
        return []

    subset = subset[(subset['entry_time_et_naive'] >= lookback_start) & (subset['entry_time_et_naive'] <= latest_bar_ts)]
    _debug(f'meta rows inside fresh-entry window: {len(subset)}')

    if subset.empty:
        return []

    out = []
    for _, row in subset.iterrows():
        entry_ts = pd.Timestamp(row['entry_time_et_naive'])
        raw = {
            'symbol': row['symbol'],
            'side': row['side'],
            'entry': float(row['planned_entry_price']),
            'stop': float(row['planned_stop_price']),
            'target': float(row['planned_target_price']),
            'timestamp_et': entry_ts.isoformat(),
            'session_date_et': str(entry_ts.date()),
            'setup_type': str(row.get('setup_type', 'UNKNOWN')),
            'qty': int(row.get('qty', DEFAULT_QTY)),
        }
        try:
            signal = normalize_signal(raw)
            if validate_signal(signal):
                out.append(signal)
            else:
                _debug(f"invalid signal rejected for {signal['symbol']}: {raw}")
        except Exception as exc:
            _debug(f"skipped malformed signal for {row.get('symbol', 'UNKNOWN')}: {exc}")

    _debug(f'fresh live signals built: {len(out)}')
    return out


def generate_live_signals() -> list[dict[str, Any]]:
    _debug(
        f'cycle start | model={MODEL_NAME} | symbols={list(SYMBOLS)} | '
        f'log_file={get_log_file_path()}'
    )
    signals: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        if symbol not in INSTRUMENTS:
            _debug(f'unsupported symbol skipped: {symbol}')
            continue
        try:
            meta, df = run_top_bottom_for_symbol(symbol)
            symbol_signals = extract_fresh_entries(meta, df)
            if not symbol_signals:
                _debug(f'{symbol} yielded 0 fresh signals this cycle')
            else:
                _debug(f'{symbol} yielded {len(symbol_signals)} fresh signals this cycle')
            signals.extend(symbol_signals)
        except Exception as exc:
            _debug(f'{symbol} scan failed: {exc}')
    _debug(f'cycle complete | total_signals={len(signals)}')
    return signals
