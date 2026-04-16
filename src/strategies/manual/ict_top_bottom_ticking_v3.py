from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Optional

import numpy as np
import pandas as pd
from backtesting import Strategy


@dataclass
class PendingSetup:
    side: str
    armed_bar: int
    zone_low: float
    zone_high: float
    entry_price: float
    stop_price: float
    target1: float
    target2: float
    target3: float
    external_sweep: bool
    internal_sweep: bool
    session_tag: str


class ICTTopBottomTickingBase(Strategy):
    # sizing and management
    fixed_size = 5
    tp1_size = 2
    tp2_size = 2
    tp3_size = 1
    rr_target1 = 1.0
    rr_target2 = 2.0
    rr_target3 = 3.5
    be_after_tp1 = True

    # setup logic
    require_internal_sweep = False
    use_strict_session_windows = False
    use_type1_sniper = False
    session_open_hour = 8   # NY session focus default
    session_close_hour = 12
    max_setup_age_bars = 6
    sweep_lookback_bars = 24
    min_sweep_ticks = 1
    tick_size = 0.25
    zone_buffer_ticks = 1
    min_rr_multiple = 1.5

    def init(self) -> None:
        self.pending: Optional[PendingSetup] = None
        self._tp1_done = False
        self._tp2_done = False
        self._trade_meta = {
            'entry_time': None,
            'entry_price': None,
            'stop_price': None,
            'target1': None,
            'target2': None,
            'target3': None,
            'side': None,
            'setup_type': None,
            'external_sweep': None,
            'internal_sweep': None,
            'session_tag': None,
        }

        idx = pd.DatetimeIndex(self.data.df.index)
        self._df = self.data.df.copy()
        self._df.index = idx

        self._roll_high = self._df['High'].rolling(self.sweep_lookback_bars, min_periods=self.sweep_lookback_bars).max().shift(1)
        self._roll_low = self._df['Low'].rolling(self.sweep_lookback_bars, min_periods=self.sweep_lookback_bars).min().shift(1)
        self._body = (self._df['Close'] - self._df['Open']).abs()
        self._range = (self._df['High'] - self._df['Low']).replace(0, np.nan)
        self._body_ratio = (self._body / self._range).fillna(0)
        self._atr = self._atr_like(14)

    def _atr_like(self, n: int) -> pd.Series:
        h = self._df['High']
        l = self._df['Low']
        c = self._df['Close'].shift(1)
        tr = pd.concat([(h - l).abs(), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
        return tr.rolling(n, min_periods=n).mean()

    def _in_session(self, ts: pd.Timestamp) -> bool:
        if not self.use_strict_session_windows:
            return True
        t = ts.time()
        return time(self.session_open_hour, 0) <= t <= time(self.session_close_hour, 59)

    def _session_tag(self, ts: pd.Timestamp) -> str:
        h = ts.hour
        if 0 <= h < 5:
            return 'asia'
        if 5 <= h < 8:
            return 'london'
        if 8 <= h < 12:
            return 'ny_am'
        if 12 <= h < 16:
            return 'ny_pm'
        return 'globex'

    def _internal_sweep_short(self, i: int) -> bool:
        if i < 3:
            return False
        highs = self._df['High']
        return highs.iloc[i] > highs.iloc[i-1] and self._df['Close'].iloc[i] < highs.iloc[i-1]

    def _internal_sweep_long(self, i: int) -> bool:
        if i < 3:
            return False
        lows = self._df['Low']
        return lows.iloc[i] < lows.iloc[i-1] and self._df['Close'].iloc[i] > lows.iloc[i-1]

    def _micro_cos_short(self, i: int) -> bool:
        if i < 2:
            return False
        return self._df['Close'].iloc[i] < self._df['Low'].iloc[i-1]

    def _micro_cos_long(self, i: int) -> bool:
        if i < 2:
            return False
        return self._df['Close'].iloc[i] > self._df['High'].iloc[i-1]

    def _arm_short(self, i: int) -> Optional[PendingSetup]:
        row = self._df.iloc[i]
        prev_high = self._roll_high.iloc[i]
        atr = self._atr.iloc[i]
        if pd.isna(prev_high) or pd.isna(atr):
            return None
        ext = row['High'] >= prev_high + self.min_sweep_ticks * self.tick_size
        if not ext:
            return None
        internal = self._internal_sweep_short(i)
        if self.require_internal_sweep and not internal:
            return None
        # rejection block approximation from sweep bar body/zone
        zone_high = float(row['High'])
        zone_low = float(max(row['Open'], row['Close']))
        if zone_high <= zone_low:
            zone_low = float(row['Low'] + (row['High'] - row['Low']) * 0.5)
        entry = (zone_high + zone_low) / 2.0
        if self.use_type1_sniper:
            # Type1: tighter micro entry and CoS required later
            entry = zone_low + (zone_high - zone_low) * 0.25
        stop = zone_high + self.zone_buffer_ticks * self.tick_size
        risk = stop - entry
        if risk <= 0:
            return None
        t1 = entry - self.rr_target1 * risk
        t2 = entry - self.rr_target2 * risk
        t3 = entry - self.rr_target3 * risk
        return PendingSetup('short', i, zone_low, zone_high, entry, stop, t1, t2, t3, ext, internal, self._session_tag(row.name))

    def _arm_long(self, i: int) -> Optional[PendingSetup]:
        row = self._df.iloc[i]
        prev_low = self._roll_low.iloc[i]
        atr = self._atr.iloc[i]
        if pd.isna(prev_low) or pd.isna(atr):
            return None
        ext = row['Low'] <= prev_low - self.min_sweep_ticks * self.tick_size
        if not ext:
            return None
        internal = self._internal_sweep_long(i)
        if self.require_internal_sweep and not internal:
            return None
        zone_low = float(row['Low'])
        zone_high = float(min(row['Open'], row['Close']))
        if zone_high <= zone_low:
            zone_high = float(row['Low'] + (row['High'] - row['Low']) * 0.5)
        entry = (zone_high + zone_low) / 2.0
        if self.use_type1_sniper:
            entry = zone_high - (zone_high - zone_low) * 0.25
        stop = zone_low - self.zone_buffer_ticks * self.tick_size
        risk = entry - stop
        if risk <= 0:
            return None
        t1 = entry + self.rr_target1 * risk
        t2 = entry + self.rr_target2 * risk
        t3 = entry + self.rr_target3 * risk
        return PendingSetup('long', i, zone_low, zone_high, entry, stop, t1, t2, t3, ext, internal, self._session_tag(row.name))

    def _validate_pending(self) -> bool:
        p = self.pending
        if p is None:
            return False
        if p.side == 'short':
            return p.target3 < p.entry_price < p.stop_price
        return p.stop_price < p.entry_price < p.target3

    def _try_enter(self, i: int) -> None:
        if self.pending is None or self.position:
            return
        row = self._df.iloc[i]
        # pending timeout
        if i - self.pending.armed_bar > self.max_setup_age_bars:
            self.pending = None
            return
        p = self.pending
        touched = row['Low'] <= p.entry_price <= row['High']
        if not touched:
            return
        if self.use_type1_sniper:
            if p.side == 'short' and not self._micro_cos_short(i):
                return
            if p.side == 'long' and not self._micro_cos_long(i):
                return
        if not self._validate_pending():
            self.pending = None
            return

        self._tp1_done = False
        self._tp2_done = False
        self._trade_meta = {
            'entry_time': row.name,
            'entry_price': p.entry_price,
            'stop_price': p.stop_price,
            'target1': p.target1,
            'target2': p.target2,
            'target3': p.target3,
            'side': p.side,
            'setup_type': 'type1' if self.use_type1_sniper else 'type2',
            'external_sweep': p.external_sweep,
            'internal_sweep': p.internal_sweep,
            'session_tag': p.session_tag,
        }
        if p.side == 'short':
            self.sell(size=self.fixed_size, sl=float(p.stop_price), tp=float(p.target3))
        else:
            self.buy(size=self.fixed_size, sl=float(p.stop_price), tp=float(p.target3))
        self.pending = None

    def _manage_open_trade(self, i: int) -> None:
        if not self.position:
            return
        row = self._df.iloc[i]
        side = self._trade_meta['side']
        t1 = self._trade_meta['target1']
        t2 = self._trade_meta['target2']
        if side == 'long':
            if (not self._tp1_done) and row['High'] >= t1:
                self.position.close(portion=self.tp1_size / self.fixed_size)
                self._tp1_done = True
                if self.be_after_tp1:
                    for trade in self.trades:
                        if trade.is_long:
                            trade.sl = self._trade_meta['entry_price']
            if self._tp1_done and (not self._tp2_done) and row['High'] >= t2:
                portion = self.tp2_size / max(self.fixed_size - self.tp1_size, 1)
                self.position.close(portion=min(1.0, portion))
                self._tp2_done = True
        else:
            if (not self._tp1_done) and row['Low'] <= t1:
                self.position.close(portion=self.tp1_size / self.fixed_size)
                self._tp1_done = True
                if self.be_after_tp1:
                    for trade in self.trades:
                        if trade.is_short:
                            trade.sl = self._trade_meta['entry_price']
            if self._tp1_done and (not self._tp2_done) and row['Low'] <= t2:
                portion = self.tp2_size / max(self.fixed_size - self.tp1_size, 1)
                self.position.close(portion=min(1.0, portion))
                self._tp2_done = True

    def next(self) -> None:
        i = len(self.data) - 1
        row = self._df.iloc[i]
        if not self._in_session(row.name):
            return

        self._manage_open_trade(i)
        if self.position:
            return

        # try existing pending first
        self._try_enter(i)
        if self.position:
            return

        # arm new setup
        short_pending = self._arm_short(i)
        long_pending = self._arm_long(i)

        # prefer the tighter body ratio / more selective setup if both appear
        if short_pending and long_pending:
            # choose reversal opposite candle close direction
            self.pending = short_pending if self._df['Close'].iloc[i] < self._df['Open'].iloc[i] else long_pending
        elif short_pending:
            self.pending = short_pending
        elif long_pending:
            self.pending = long_pending


class ICTTopBottomTickingType2(ICTTopBottomTickingBase):
    require_internal_sweep = False
    use_strict_session_windows = False
    use_type1_sniper = False


class ICTTopBottomTickingType2Strict(ICTTopBottomTickingBase):
    require_internal_sweep = True
    use_strict_session_windows = False
    use_type1_sniper = False


class ICTTopBottomTickingType2StrictSession(ICTTopBottomTickingBase):
    require_internal_sweep = True
    use_strict_session_windows = True
    use_type1_sniper = False
    session_open_hour = 8
    session_close_hour = 11


class ICTTopBottomTickingType1Sniper(ICTTopBottomTickingBase):
    require_internal_sweep = True
    use_strict_session_windows = True
    use_type1_sniper = True
    session_open_hour = 8
    session_close_hour = 11
    max_setup_age_bars = 4
