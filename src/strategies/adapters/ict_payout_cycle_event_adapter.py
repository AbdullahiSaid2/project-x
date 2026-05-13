
from __future__ import annotations

"""
ICT Payout Cycle Model v2.

Fixes:
- symbol-aware minimum stop distances
- rejects tiny stops before sizing
- broader relaxed-mode setup option for smoke testing
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
from src.backtesting.event_engine.models import OrderPlan, SymbolSpec, PropProfile


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    prev = close.shift(1)
    tr = pd.concat([high-low, (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()


def _last_swing_high(high: pd.Series) -> pd.Series:
    is_swing = (high > high.shift(1)) & (high > high.shift(2)) & (high >= high.shift(-1)) & (high >= high.shift(-2))
    return high.where(is_swing).ffill().shift(1)


def _last_swing_low(low: pd.Series) -> pd.Series:
    is_swing = (low < low.shift(1)) & (low < low.shift(2)) & (low <= low.shift(-1)) & (low <= low.shift(-2))
    return low.where(is_swing).ffill().shift(1)


@dataclass
class PendingFVG:
    direction: str = ''
    created_index: int = -1
    expiry_index: int = -1
    sweep_level: float = np.nan
    sweep_extreme: float = np.nan
    fvg_low: float = np.nan
    fvg_high: float = np.nan
    fvg_ce: float = np.nan
    stop: float = np.nan
    setup_score: float = 0.0
    reason: str = ''


class ICTPayoutCycleAdapter:
    name = 'ict_payout_cycle_v2'

    MIN_STOP_POINTS = {'MNQ': 8.0, 'MES': 6.0, 'MYM': 60.0, 'MGC': 6.0, 'MCL': 0.25}
    MAX_STOP_POINTS = {'MNQ': 80.0, 'MES': 60.0, 'MYM': 350.0, 'MGC': 40.0, 'MCL': 2.5}

    def __init__(self):
        self.pending: Dict[str, PendingFVG] = {}

    def _state(self, symbol: str) -> PendingFVG:
        if symbol not in self.pending:
            self.pending[symbol] = PendingFVG()
        return self.pending[symbol]

    def build_features(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out['atr14'] = _atr(out, 14)
        out['ema20'] = _ema(out['Close'], 20)
        out['ema50'] = _ema(out['Close'], 50)
        out['ema200'] = _ema(out['Close'], 200)
        out['body'] = (out['Close'] - out['Open']).abs()
        out['range'] = (out['High'] - out['Low']).replace(0, np.nan)
        out['close_pos'] = (out['Close'] - out['Low']) / out['range']
        out['prior_high_60'] = out['High'].rolling(60, min_periods=60).max().shift(1)
        out['prior_low_60'] = out['Low'].rolling(60, min_periods=60).min().shift(1)
        out['swing_high'] = _last_swing_high(out['High'])
        out['swing_low'] = _last_swing_low(out['Low'])
        out['bull_fvg_low'] = out['High'].shift(2)
        out['bull_fvg_high'] = out['Low']
        out['bull_fvg'] = out['bull_fvg_high'] > out['bull_fvg_low']
        out['bear_fvg_low'] = out['High']
        out['bear_fvg_high'] = out['Low'].shift(2)
        out['bear_fvg'] = out['bear_fvg_high'] > out['bear_fvg_low']
        out['bull_displacement'] = (out['Close'] > out['Open']) & (out['body'] >= out['atr14'] * 0.50) & (out['close_pos'] >= 0.60)
        out['bear_displacement'] = (out['Close'] < out['Open']) & (out['body'] >= out['atr14'] * 0.50) & (out['close_pos'] <= 0.40)

        idx = out.index
        et_index = idx.tz_localize('UTC').tz_convert('America/New_York') if idx.tz is None else idx.tz_convert('America/New_York')
        tmp = out.copy()
        tmp['_et_date'] = et_index.date
        tmp['_et_hour'] = et_index.hour
        daily = tmp.groupby('_et_date').agg(day_high=('High','max'), day_low=('Low','min'))
        daily['prev_day_high'] = daily['day_high'].shift(1)
        daily['prev_day_low'] = daily['day_low'].shift(1)
        out['prev_day_high'] = tmp['_et_date'].map(daily['prev_day_high'])
        out['prev_day_low'] = tmp['_et_date'].map(daily['prev_day_low'])
        session_dates = [(t + pd.Timedelta(days=1)).date() if t.hour >= 18 else t.date() for t in et_index]
        tmp['_session_date'] = session_dates
        asia = tmp[tmp['_et_hour'].between(18, 23)].groupby('_session_date').agg(asia_high=('High','max'), asia_low=('Low','min'))
        out['asia_high'] = pd.Series(session_dates, index=out.index).map(asia['asia_high'])
        out['asia_low'] = pd.Series(session_dates, index=out.index).map(asia['asia_low'])

        for rule, prefix in [('1h','h1'), ('4h','h4')]:
            frame = _resample_ohlcv(out[['Open','High','Low','Close']], rule)
            frame[f'{prefix}_ema20'] = _ema(frame['Close'], 20)
            frame[f'{prefix}_ema50'] = _ema(frame['Close'], 50)
            shifted = frame[[f'{prefix}_ema20', f'{prefix}_ema50']].shift(1)
            out = out.join(shifted.reindex(out.index, method='ffill'))
        out['bull_htf_bias'] = (out['Close'] > out['h1_ema50']) & (out['h1_ema20'] > out['h1_ema50']) & (out['h4_ema20'] >= out['h4_ema50'])
        out['bear_htf_bias'] = (out['Close'] < out['h1_ema50']) & (out['h1_ema20'] < out['h1_ema50']) & (out['h4_ema20'] <= out['h4_ema50'])

        mins = et_index.hour * 60 + et_index.minute
        out['is_valid_entry_window'] = ((mins >= 2*60) & (mins <= 5*60)) | ((mins >= 8*60+30) & (mins <= 11*60)) | ((mins >= 13*60+30) & (mins <= 15*60+30))
        return out

    def _score_long(self, row: pd.Series) -> float:
        return float(int(bool(row.get('bull_htf_bias', False))) + int(bool(row.get('bull_displacement', False))) + int(bool(row.get('bull_fvg', False))) + int(bool(row.get('is_valid_entry_window', False))) + int(float(row.get('Close', 0)) > float(row.get('ema50', np.inf))))

    def _score_short(self, row: pd.Series) -> float:
        return float(int(bool(row.get('bear_htf_bias', False))) + int(bool(row.get('bear_displacement', False))) + int(bool(row.get('bear_fvg', False))) + int(bool(row.get('is_valid_entry_window', False))) + int(float(row.get('Close', 0)) < float(row.get('ema50', -np.inf))))

    def _long_sweep_level(self, row: pd.Series) -> Optional[float]:
        low = float(row['Low'])
        candidates = [row.get('prev_day_low', np.nan), row.get('asia_low', np.nan), row.get('prior_low_60', np.nan), row.get('swing_low', np.nan)]
        valid = [float(x) for x in candidates if pd.notna(x) and low < float(x)]
        return max(valid) if valid else None

    def _short_sweep_level(self, row: pd.Series) -> Optional[float]:
        high = float(row['High'])
        candidates = [row.get('prev_day_high', np.nan), row.get('asia_high', np.nan), row.get('prior_high_60', np.nan), row.get('swing_high', np.nan)]
        valid = [float(x) for x in candidates if pd.notna(x) and high > float(x)]
        return min(valid) if valid else None

    def _stop_ok(self, symbol: str, entry: float, stop: float) -> bool:
        dist = abs(entry - stop)
        return self.MIN_STOP_POINTS.get(symbol, 1.0) <= dist <= self.MAX_STOP_POINTS.get(symbol, 999999.0)

    def _arm_long_if_valid(self, symbol: str, row: pd.Series, i: int, args) -> None:
        sweep = self._long_sweep_level(row)
        if sweep is None:
            return
        relaxed = bool(getattr(args, 'relaxed_mode', False))
        has_confirmation = float(row['Close']) > sweep and bool(row.get('bull_displacement', False))
        if not relaxed:
            has_confirmation = has_confirmation and bool(row.get('bull_fvg', False))
        if not has_confirmation or self._score_long(row) < float(getattr(args, 'min_ict_payout_score', 3.0)):
            return
        if bool(row.get('bull_fvg', False)):
            fvg_low, fvg_high = float(row['bull_fvg_low']), float(row['bull_fvg_high'])
        else:
            fvg_low, fvg_high = min(float(row['Open']), float(row['Close'])), max(float(row['Open']), float(row['Close']))
        fvg_ce = (fvg_low + fvg_high) / 2.0
        atr = float(row['atr14']) if pd.notna(row.get('atr14')) else 0.0
        stop = float(row['Low']) - max(atr * float(getattr(args, 'stop_buffer_atr', 0.25)), self.MIN_STOP_POINTS.get(symbol, 1.0) * 0.25)
        self.pending[symbol] = PendingFVG('LONG', i, i + int(getattr(args, 'fvg_expiry_bars', 30)), sweep, float(row['Low']), fvg_low, fvg_high, fvg_ce, stop, self._score_long(row), 'bull_sweep_reclaim_displacement_fvg')

    def _arm_short_if_valid(self, symbol: str, row: pd.Series, i: int, args) -> None:
        sweep = self._short_sweep_level(row)
        if sweep is None:
            return
        relaxed = bool(getattr(args, 'relaxed_mode', False))
        has_confirmation = float(row['Close']) < sweep and bool(row.get('bear_displacement', False))
        if not relaxed:
            has_confirmation = has_confirmation and bool(row.get('bear_fvg', False))
        if not has_confirmation or self._score_short(row) < float(getattr(args, 'min_ict_payout_score', 3.0)):
            return
        if bool(row.get('bear_fvg', False)):
            fvg_low, fvg_high = float(row['bear_fvg_low']), float(row['bear_fvg_high'])
        else:
            fvg_low, fvg_high = min(float(row['Open']), float(row['Close'])), max(float(row['Open']), float(row['Close']))
        fvg_ce = (fvg_low + fvg_high) / 2.0
        atr = float(row['atr14']) if pd.notna(row.get('atr14')) else 0.0
        stop = float(row['High']) + max(atr * float(getattr(args, 'stop_buffer_atr', 0.25)), self.MIN_STOP_POINTS.get(symbol, 1.0) * 0.25)
        self.pending[symbol] = PendingFVG('SHORT', i, i + int(getattr(args, 'fvg_expiry_bars', 30)), sweep, float(row['High']), fvg_low, fvg_high, fvg_ce, stop, self._score_short(row), 'bear_sweep_reclaim_displacement_fvg')

    def _pending_to_order(self, symbol: str, pending: PendingFVG, row: pd.Series, spec: SymbolSpec, args) -> Optional[OrderPlan]:
        target_r = float(getattr(args, 'target_r', 5.0))
        entry_mode = str(getattr(args, 'entry_mode', 'ce')).lower()
        touched = float(row['Low']) <= pending.fvg_high and float(row['High']) >= pending.fvg_low
        if not touched:
            return None
        if pending.direction == 'LONG':
            entry = pending.fvg_ce if entry_mode == 'ce' else min(float(row['Close']), pending.fvg_high)
            stop = min(pending.stop, pending.sweep_extreme - 0.25)
            if not self._stop_ok(symbol, entry, stop):
                return None
            return OrderPlan(symbol, 'LONG', float(entry), float(stop), float(entry + (entry-stop)*target_r), 'ict_payout_cycle_long', pending.reason, self.name, pending.setup_score)
        if pending.direction == 'SHORT':
            entry = pending.fvg_ce if entry_mode == 'ce' else max(float(row['Close']), pending.fvg_low)
            stop = max(pending.stop, pending.sweep_extreme + 0.25)
            if not self._stop_ok(symbol, entry, stop):
                return None
            return OrderPlan(symbol, 'SHORT', float(entry), float(stop), float(entry - (stop-entry)*target_r), 'ict_payout_cycle_short', pending.reason, self.name, pending.setup_score)
        return None

    def signal_for_row(self, symbol: str, row: pd.Series, history: pd.DataFrame, spec: SymbolSpec, profile: PropProfile, args):
        i = len(history) - 1
        if i < int(getattr(args, 'warmup_bars', 500)):
            return None
        state = self._state(symbol)
        if state.direction and i > state.expiry_index:
            self.pending[symbol] = PendingFVG(); state = self.pending[symbol]
        if state.direction:
            order = self._pending_to_order(symbol, state, row, spec, args)
            if order is not None:
                self.pending[symbol] = PendingFVG()
                return order
            return None
        if not bool(row.get('is_valid_entry_window', False)):
            return None
        if bool(row.get('bull_htf_bias', False)):
            self._arm_long_if_valid(symbol, row, i, args)
        if bool(row.get('bear_htf_bias', False)):
            self._arm_short_if_valid(symbol, row, i, args)
        return None
