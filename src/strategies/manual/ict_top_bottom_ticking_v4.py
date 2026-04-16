from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd
from backtesting import Strategy

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[3] if len(ROOT.parents) >= 4 else ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TICK_SIZE = 0.25
FORCE_FLAT_HOUR_ET = 16
FORCE_FLAT_MINUTE_ET = 50
GLOBEX_REOPEN_HOUR_ET = 18


@dataclass
class PendingSetup:
    direction: str = ""
    created_bar: int = -1
    expiry_bar: int = -1
    entry_ce: float = np.nan
    zone_high: float = np.nan
    zone_low: float = np.nan
    stop_price: float = np.nan
    target1: float = np.nan
    target2: float = np.nan
    target3: float = np.nan
    external_level: float = np.nan
    setup_type: str = ""
    entry_variant: str = ""
    internal_sweep: bool = False


def _to_et(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ts = pd.DatetimeIndex(idx)
    if ts.tz is None:
        ts = ts.tz_localize('UTC')
    return ts.tz_convert('America/New_York')


def _session_date_for_et(ts: pd.Series) -> pd.Series:
    return (
        ts.dt.tz_localize(None)
        + pd.to_timedelta((ts.dt.hour >= GLOBEX_REOPEN_HOUR_ET).astype(int), unit='D')
    ).dt.date


def build_model_frame(df: pd.DataFrame, context_tf: str = '15min') -> pd.DataFrame:
    m = df.copy()
    m.columns = [c.capitalize() for c in m.columns]
    et = _to_et(pd.DatetimeIndex(m.index))
    m['et_time'] = et
    m['et_date'] = et.date
    m['et_hour'] = et.hour
    m['et_minute'] = et.minute
    m['session_date'] = _session_date_for_et(pd.Series(et, index=m.index))

    session_summary = (
        m.groupby('session_date')
        .agg(session_high=('High', 'max'), session_low=('Low', 'min'))
        .shift(1)
        .rename(columns={'session_high': 'prior_session_high', 'session_low': 'prior_session_low'})
    )
    m = m.join(session_summary, on='session_date')

    asia_rows = m[m['et_hour'] >= 18]
    asia_summary = (
        asia_rows.groupby('session_date')
        .agg(asia_high=('High', 'max'), asia_low=('Low', 'min'))
        .rename(columns={'asia_high': 'asia_high', 'asia_low': 'asia_low'})
    )
    m = m.join(asia_summary, on='session_date')

    m['external_buyside'] = m[['prior_session_high', 'asia_high']].max(axis=1, skipna=True)
    m['external_sellside'] = m[['prior_session_low', 'asia_low']].min(axis=1, skipna=True)

    ctx = (
        m[['Open', 'High', 'Low', 'Close', 'Volume']]
        .resample(context_tf, label='right', closed='right')
        .agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        .dropna()
    )
    ctx['swing_high_ctx'] = ctx['High'].rolling(5, center=True).max().eq(ctx['High'])
    ctx['swing_low_ctx'] = ctx['Low'].rolling(5, center=True).min().eq(ctx['Low'])
    ctx = ctx[['High', 'Low', 'Close', 'swing_high_ctx', 'swing_low_ctx']].rename(
        columns={'High': 'ctx_high', 'Low': 'ctx_low', 'Close': 'ctx_close'}
    )
    m = pd.merge_asof(
        m.sort_index(), ctx.sort_index(), left_index=True, right_index=True, direction='backward'
    )

    m['recent_high_6'] = m['High'].rolling(6).max().shift(1)
    m['recent_low_6'] = m['Low'].rolling(6).min().shift(1)
    m['sweep_short'] = (
        pd.notna(m['external_buyside'])
        & (m['High'] >= m['external_buyside'] + TICK_SIZE)
        & (m['Close'] <= m['external_buyside'])
    )
    m['sweep_long'] = (
        pd.notna(m['external_sellside'])
        & (m['Low'] <= m['external_sellside'] - TICK_SIZE)
        & (m['Close'] >= m['external_sellside'])
    )
    m['internal_sweep_short'] = m['High'] >= (m['recent_high_6'] + TICK_SIZE)
    m['internal_sweep_long'] = m['Low'] <= (m['recent_low_6'] - TICK_SIZE)
    m['cos_short'] = m['Close'] < m['recent_low_6']
    m['cos_long'] = m['Close'] > m['recent_high_6']
    return m


class ICTTopBottomTickingBase(Strategy):
    fixed_size = 5
    min_warmup_bars = 150
    setup_expiry_bars = 18
    require_cos_confirmation = True
    require_internal_sweep_filter = False
    move_stop_to_be_after_t1 = True
    partial_1_fraction = 2 / 5
    partial_2_fraction_of_remaining = 2 / 3
    target1_r = 1.0
    target2_r = 2.25
    target3_r = 4.25
    min_stop_points = 6.0
    max_stop_points = 30.0
    context_tf = '15min'
    session_filter = None
    entry_tolerance_ticks = 1
    prefer_limit_style = False
    last_trade_log: List[dict] = []
    last_debug_counts: dict = {}

    def init(self):
        self.m = build_model_frame(self.data.df.copy(), context_tf=self.context_tf)
        self.pending = PendingSetup()
        self.partial1_taken = False
        self.partial2_taken = False
        self.be_moved = False
        self.active_risk = np.nan
        self.open_trade_meta = None
        self.prev_closed_count = 0
        self.debug_counts = {
            'sweep_short_seen': 0,
            'sweep_long_seen': 0,
            'pending_short_armed': 0,
            'pending_long_armed': 0,
            'entry_short': 0,
            'entry_long': 0,
            'partial1': 0,
            'partial2': 0,
            'be_move': 0,
            'forced_flat': 0,
            'expired_pending': 0,
            'blocked_internal_filter': 0,
            'invalid_target_order': 0,
        }
        self.__class__.last_trade_log = []
        self.__class__.last_debug_counts = {}

    def _sync_debug(self):
        self.__class__.last_debug_counts = dict(self.debug_counts)

    def _i(self) -> int:
        return len(self.data) - 1

    def _latest_trade(self):
        try:
            if self.trades:
                return self.trades[-1]
        except Exception:
            pass
        return None

    def _clear_pending(self):
        self.pending = PendingSetup()

    def _after_force_flat_cutoff(self, row: pd.Series) -> bool:
        hour = int(row.get('et_hour', -1))
        minute = int(row.get('et_minute', -1))
        return (hour > FORCE_FLAT_HOUR_ET) or (hour == FORCE_FLAT_HOUR_ET and minute >= FORCE_FLAT_MINUTE_ET)

    def _session_ok(self, row: pd.Series) -> bool:
        if self.session_filter is None:
            return True
        hour = int(row.get('et_hour', -1))
        return self.session_filter[0] <= hour <= self.session_filter[1]

    def _force_flat_if_needed(self, row: pd.Series):
        if self.position and self._after_force_flat_cutoff(row):
            try:
                self.position.close()
                self.debug_counts['forced_flat'] += 1
                self._sync_debug()
            except Exception:
                pass

    def _record_open_trade_meta(self, entry: float):
        self.open_trade_meta = {
            'setup_type': self.pending.setup_type,
            'entry_variant': self.pending.entry_variant,
            'external_level': self.pending.external_level,
            'zone_high': self.pending.zone_high,
            'zone_low': self.pending.zone_low,
            'planned_entry_price': entry,
            'planned_stop_price': self.pending.stop_price,
            'planned_target1_price': self.pending.target1,
            'planned_target2_price': self.pending.target2,
            'planned_target3_price': self.pending.target3,
            'internal_sweep': self.pending.internal_sweep,
        }

    def _log_newly_closed_trades(self):
        try:
            closed = list(self.closed_trades)
        except Exception:
            return
        if len(closed) <= self.prev_closed_count:
            return
        new_items = closed[self.prev_closed_count:]
        for t in new_items:
            meta = self.open_trade_meta or {}
            self.__class__.last_trade_log.append(
                {
                    'side': 'LONG' if float(t.size) > 0 else 'SHORT',
                    'setup_type': meta.get('setup_type', ''),
                    'entry_variant': meta.get('entry_variant', ''),
                    'entry_price': float(t.entry_price),
                    'exit_price': float(t.exit_price),
                    'entry_time': str(t.entry_time),
                    'exit_time': str(t.exit_time),
                    'pnl': float(getattr(t, 'pl', np.nan)),
                    'return_pct': float(getattr(t, 'pl_pct', np.nan)) if hasattr(t, 'pl_pct') else np.nan,
                    **meta,
                }
            )
        if new_items and not self.position:
            self.open_trade_meta = None
        self.prev_closed_count = len(closed)

    def _entry_zone_short(self, row: pd.Series):
        zone_high = float(row['High'])
        zone_low = min(float(row['Open']), float(row['Close']))
        return zone_high, zone_low, (zone_high + zone_low) / 2.0

    def _entry_zone_long(self, row: pd.Series):
        zone_low = float(row['Low'])
        zone_high = max(float(row['Open']), float(row['Close']))
        return zone_low, zone_high, (zone_high + zone_low) / 2.0

    def _arm_short_from_sweep(self, row: pd.Series, i: int):
        if self.require_internal_sweep_filter and not bool(row.get('internal_sweep_short', False)):
            self.debug_counts['blocked_internal_filter'] += 1
            self._sync_debug()
            return
        zone_high, zone_low, entry_ce = self._entry_zone_short(row)
        stop = zone_high + TICK_SIZE
        risk = stop - entry_ce
        if risk < self.min_stop_points or risk > self.max_stop_points:
            return
        self.pending = PendingSetup(
            direction='short',
            created_bar=i,
            expiry_bar=i + self.setup_expiry_bars,
            entry_ce=entry_ce,
            zone_high=zone_high,
            zone_low=zone_low,
            stop_price=stop,
            target1=entry_ce - (risk * self.target1_r),
            target2=entry_ce - (risk * self.target2_r),
            target3=entry_ce - (risk * self.target3_r),
            external_level=float(row.get('external_buyside', np.nan)),
            setup_type=self.__class__.__name__.upper() + '_SHORT',
            entry_variant='CE_LIMIT' if self.prefer_limit_style else 'CE_PLUS_COS',
            internal_sweep=bool(row.get('internal_sweep_short', False)),
        )
        self.debug_counts['pending_short_armed'] += 1
        self._sync_debug()

    def _arm_long_from_sweep(self, row: pd.Series, i: int):
        if self.require_internal_sweep_filter and not bool(row.get('internal_sweep_long', False)):
            self.debug_counts['blocked_internal_filter'] += 1
            self._sync_debug()
            return
        zone_low, zone_high, entry_ce = self._entry_zone_long(row)
        stop = zone_low - TICK_SIZE
        risk = entry_ce - stop
        if risk < self.min_stop_points or risk > self.max_stop_points:
            return
        self.pending = PendingSetup(
            direction='long',
            created_bar=i,
            expiry_bar=i + self.setup_expiry_bars,
            entry_ce=entry_ce,
            zone_high=zone_high,
            zone_low=zone_low,
            stop_price=stop,
            target1=entry_ce + (risk * self.target1_r),
            target2=entry_ce + (risk * self.target2_r),
            target3=entry_ce + (risk * self.target3_r),
            external_level=float(row.get('external_sellside', np.nan)),
            setup_type=self.__class__.__name__.upper() + '_LONG',
            entry_variant='CE_LIMIT' if self.prefer_limit_style else 'CE_PLUS_COS',
            internal_sweep=bool(row.get('internal_sweep_long', False)),
        )
        self.debug_counts['pending_long_armed'] += 1
        self._sync_debug()

    def _pending_short_ready(self, row: pd.Series) -> bool:
        touched = float(row['High']) >= (self.pending.entry_ce - (TICK_SIZE * self.entry_tolerance_ticks))
        if not touched:
            return False
        if self.require_cos_confirmation:
            return bool(row.get('cos_short', False)) and float(row['Close']) < self.pending.entry_ce
        return float(row['Close']) <= (self.pending.entry_ce + TICK_SIZE)

    def _pending_long_ready(self, row: pd.Series) -> bool:
        touched = float(row['Low']) <= (self.pending.entry_ce + (TICK_SIZE * self.entry_tolerance_ticks))
        if not touched:
            return False
        if self.require_cos_confirmation:
            return bool(row.get('cos_long', False)) and float(row['Close']) > self.pending.entry_ce
        return float(row['Close']) >= (self.pending.entry_ce - TICK_SIZE)

    def _enter_short(self, row: pd.Series):
        entry = float(self.pending.entry_ce) if self.prefer_limit_style else min(float(row['Close']), float(self.pending.entry_ce))
        stop = float(self.pending.stop_price)
        risk = stop - entry
        if risk <= 0:
            self._clear_pending()
            return
        target1 = entry - (risk * self.target1_r)
        target2 = entry - (risk * self.target2_r)
        target3 = entry - (risk * self.target3_r)
        if not (target3 < entry < stop):
            self.debug_counts['invalid_target_order'] += 1
            self._sync_debug()
            self._clear_pending()
            return
        self.pending.target1 = target1
        self.pending.target2 = target2
        self.pending.target3 = target3
        self.active_risk = risk
        self.sell(size=self.fixed_size, sl=stop, tp=target3)
        self.partial1_taken = self.partial2_taken = self.be_moved = False
        self._record_open_trade_meta(entry)
        self.debug_counts['entry_short'] += 1
        self._sync_debug()
        self._clear_pending()

    def _enter_long(self, row: pd.Series):
        entry = float(self.pending.entry_ce) if self.prefer_limit_style else max(float(row['Close']), float(self.pending.entry_ce))
        stop = float(self.pending.stop_price)
        risk = entry - stop
        if risk <= 0:
            self._clear_pending()
            return
        target1 = entry + (risk * self.target1_r)
        target2 = entry + (risk * self.target2_r)
        target3 = entry + (risk * self.target3_r)
        if not (stop < entry < target3):
            self.debug_counts['invalid_target_order'] += 1
            self._sync_debug()
            self._clear_pending()
            return
        self.pending.target1 = target1
        self.pending.target2 = target2
        self.pending.target3 = target3
        self.active_risk = risk
        self.buy(size=self.fixed_size, sl=stop, tp=target3)
        self.partial1_taken = self.partial2_taken = self.be_moved = False
        self._record_open_trade_meta(entry)
        self.debug_counts['entry_long'] += 1
        self._sync_debug()
        self._clear_pending()

    def _manage_trade(self):
        trade = self._latest_trade()
        if trade is None or not self.position or not np.isfinite(self.active_risk):
            return
        price = float(self.data.Close[-1])
        if self.position.is_long:
            r_now = (price - float(trade.entry_price)) / self.active_risk
            if not self.partial1_taken and r_now >= self.target1_r:
                try:
                    self.position.close(portion=self.partial_1_fraction)
                    self.partial1_taken = True
                    self.debug_counts['partial1'] += 1
                    self._sync_debug()
                except Exception:
                    pass
            if self.move_stop_to_be_after_t1 and self.partial1_taken and not self.be_moved:
                try:
                    if trade.sl is not None:
                        trade.sl = max(float(trade.sl), float(trade.entry_price))
                    self.be_moved = True
                    self.debug_counts['be_move'] += 1
                    self._sync_debug()
                except Exception:
                    pass
            if self.partial1_taken and not self.partial2_taken and r_now >= self.target2_r:
                try:
                    self.position.close(portion=self.partial_2_fraction_of_remaining)
                    self.partial2_taken = True
                    self.debug_counts['partial2'] += 1
                    self._sync_debug()
                except Exception:
                    pass
        else:
            r_now = (float(trade.entry_price) - price) / self.active_risk
            if not self.partial1_taken and r_now >= self.target1_r:
                try:
                    self.position.close(portion=self.partial_1_fraction)
                    self.partial1_taken = True
                    self.debug_counts['partial1'] += 1
                    self._sync_debug()
                except Exception:
                    pass
            if self.move_stop_to_be_after_t1 and self.partial1_taken and not self.be_moved:
                try:
                    if trade.sl is not None:
                        trade.sl = min(float(trade.sl), float(trade.entry_price))
                    self.be_moved = True
                    self.debug_counts['be_move'] += 1
                    self._sync_debug()
                except Exception:
                    pass
            if self.partial1_taken and not self.partial2_taken and r_now >= self.target2_r:
                try:
                    self.position.close(portion=self.partial_2_fraction_of_remaining)
                    self.partial2_taken = True
                    self.debug_counts['partial2'] += 1
                    self._sync_debug()
                except Exception:
                    pass

    def next(self):
        self._log_newly_closed_trades()
        i = self._i()
        if i < self.min_warmup_bars:
            return
        row = self.m.iloc[i]
        self._manage_trade()
        self._force_flat_if_needed(row)
        if self.pending.expiry_bar >= 0 and i > self.pending.expiry_bar:
            self.debug_counts['expired_pending'] += 1
            self._sync_debug()
            self._clear_pending()
        if self.position:
            return
        if self._after_force_flat_cutoff(row):
            self._clear_pending()
            return
        if not self._session_ok(row):
            return
        if self.pending.direction == 'short' and self._pending_short_ready(row):
            self._enter_short(row)
            return
        if self.pending.direction == 'long' and self._pending_long_ready(row):
            self._enter_long(row)
            return
        if self.pending.direction:
            return
        if bool(row.get('sweep_short', False)):
            self.debug_counts['sweep_short_seen'] += 1
            self._sync_debug()
            self._arm_short_from_sweep(row, i)
            return
        if bool(row.get('sweep_long', False)):
            self.debug_counts['sweep_long_seen'] += 1
            self._sync_debug()
            self._arm_long_from_sweep(row, i)


class ICTTopBottomTickingType2(ICTTopBottomTickingBase):
    pass


class ICTTopBottomTickingType2Active(ICTTopBottomTickingBase):
    require_cos_confirmation = False
    require_internal_sweep_filter = False
    setup_expiry_bars = 24
    entry_tolerance_ticks = 2
    prefer_limit_style = True
    min_stop_points = 4.0
    max_stop_points = 34.0


class ICTTopBottomTickingType1Sniper30s(ICTTopBottomTickingBase):
    require_cos_confirmation = True
    require_internal_sweep_filter = True
    setup_expiry_bars = 20
    min_warmup_bars = 300
    min_stop_points = 2.0
    max_stop_points = 18.0


class ICTTopBottomTickingType1Sniper5s(ICTTopBottomTickingBase):
    require_cos_confirmation = True
    require_internal_sweep_filter = True
    setup_expiry_bars = 60
    min_warmup_bars = 900
    min_stop_points = 1.0
    max_stop_points = 12.0


ICT_TOP_BOTTOM_TICKING_TYPE1 = ICTTopBottomTickingType1Sniper30s
ICT_TOP_BOTTOM_TICKING_TYPE2 = ICTTopBottomTickingType2
