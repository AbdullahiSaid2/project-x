from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.strategies.manual.ict_multi_setup_v452 import (
    ICT_MULTI_SETUP_V452,
    PendingSignal,
)


class ICT_MULTI_SETUP_V453(ICT_MULTI_SETUP_V452):
    """
    V45.3
    - Hard stop cap for prop use: 30 NQ points max risk.
    - Apex PA 50k oriented guardrails.
    - Qualifying-day and consistency tracking.
    - High-impact news blackout support via optional CSV.

    Optional CSV location:
    src/strategies/manual/high_impact_news_et.csv

    Expected columns:
    - event_time_et   (example: 2026-03-18 14:00:00)
    - event_name      (optional)
    - impact          (optional)
    """

    # --- risk / target profile ---
    risk_multiple = 2.0
    min_target_points = 50.0
    min_stop_points = 12.0
    max_stop_points = 30.0
    stop_buffer_atr = 0.10
    fixed_size = 0.10

    # --- entry mechanics ---
    require_pullback_entry = True
    pullback_entry_tolerance_points = 4.0
    pending_expiry_bars = 15

    # --- trade management ---
    enable_partial = True
    partial_rr = 1.0
    partial_close_fraction = 0.50
    enable_runner_to_liquidity = True
    breakeven_only_after_confirmation = True
    be_confirm_rr = 1.25
    min_bars_before_be = 3

    # --- prop mode ---
    prop_mode = True
    prop_daily_loss_limit = -1000.0
    prop_daily_max_trades = 3
    prop_max_consecutive_losses = 2
    prop_reduce_size_after_drawdown = True
    prop_drawdown_reduce_threshold = -600.0
    prop_reduced_size_multiplier = 0.5
    payout_defense_mode = False
    payout_defense_daily_profit_lock = 900.0

    # --- Apex PA 50k style controls ---
    prop_max_drawdown_limit = -2000.0
    prop_consistency_limit = 0.50
    prop_consistency_warning = 0.45
    prop_qualifying_day_target = 300.0
    prop_required_qualifying_days = 5

    # --- news blackout ---
    enable_news_filter = True
    news_flatten_minutes_before = 5
    news_resume_minutes_after = 5
    news_csv_filename = 'high_impact_news_et.csv'

    # --- ranking ---
    prop_trade_only_ranked_setups = True
    allowed_prop_setup_tiers = {"A", "B"}
    prop_b_tier_allowed_setups = {"NYPM_CONTINUATION", "ASIA_CONTINUATION"}

    last_debug_counts = {}
    last_trade_log = []

    def init(self):
        super().init()
        self.qualifying_days = set()
        self.daily_pnl_history = {}
        self.cycle_realized_pnl = 0.0
        self.best_day_pnl_in_cycle = 0.0
        self.peak_cycle_pnl = 0.0
        self.drawdown_from_cycle_peak = 0.0
        self.max_drawdown_paused = False
        self.news_events_et = self._load_news_events()

        self.debug_counts.update({
            'reject_stop_cap_long': 0,
            'reject_stop_cap_short': 0,
            'prop_block_max_drawdown': 0,
            'prop_block_consistency': 0,
            'prop_block_news': 0,
            'news_forced_flatten': 0,
            'qualifying_day_count': 0,
        })
        self._sync_debug()
        self.__class__.last_trade_log = []
        self._sync_trade_log()

    def _load_news_events(self):
        if not self.enable_news_filter:
            return []
        candidates = [
            Path(__file__).resolve().with_name(self.news_csv_filename),
            Path.cwd() / self.news_csv_filename,
        ]
        for path in candidates:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    if 'event_time_et' not in df.columns:
                        continue
                    out = []
                    for _, r in df.iterrows():
                        ts = pd.Timestamp(r['event_time_et'])
                        if ts.tzinfo is None:
                            ts = ts.tz_localize('America/New_York')
                        else:
                            ts = ts.tz_convert('America/New_York')
                        out.append({
                            'event_time_et': ts,
                            'event_name': str(r.get('event_name', '')),
                            'impact': str(r.get('impact', 'high')),
                        })
                    return out
                except Exception:
                    return []
        return []

    def _row_ts_et(self, row: pd.Series):
        try:
            ts = pd.Timestamp(row.name)
        except Exception:
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        return ts.tz_convert('America/New_York')

    def _current_news_event(self, row: pd.Series):
        ts = self._row_ts_et(row)
        if ts is None or not self.news_events_et:
            return None
        for event in self.news_events_et:
            et = event['event_time_et']
            if et - pd.Timedelta(minutes=self.news_flatten_minutes_before) <= ts <= et + pd.Timedelta(minutes=self.news_resume_minutes_after):
                return event
        return None

    def _must_flatten_for_news(self, row: pd.Series) -> bool:
        ts = self._row_ts_et(row)
        if ts is None or not self.news_events_et:
            return False
        for event in self.news_events_et:
            et = event['event_time_et']
            if et - pd.Timedelta(minutes=self.news_flatten_minutes_before) <= ts < et:
                return True
        return False

    def _consistency_ratio(self) -> float:
        cycle_pnl = float(self.cycle_realized_pnl)
        if cycle_pnl <= 0:
            return 0.0
        return float(self.best_day_pnl_in_cycle) / cycle_pnl

    def _finalize_previous_day(self):
        if self.current_et_day is None:
            return
        day_pnl = float(self.realized_pnl_today)
        self.daily_pnl_history[self.current_et_day] = day_pnl
        if day_pnl >= self.prop_qualifying_day_target:
            self.qualifying_days.add(self.current_et_day)
            self.debug_counts['qualifying_day_count'] = len(self.qualifying_days)

        if day_pnl > self.best_day_pnl_in_cycle:
            self.best_day_pnl_in_cycle = day_pnl

    def _update_day_reset(self, row: pd.Series):
        if self.current_et_day != row['et_date']:
            self._finalize_previous_day()
        super()._update_day_reset(row)
        self.debug_counts['qualifying_day_count'] = len(self.qualifying_days)
        self._sync_debug()

    def _log_newly_closed_trades(self):
        before = len(self.__class__.last_trade_log)
        super()._log_newly_closed_trades()
        after_items = self.__class__.last_trade_log[before:]
        if not after_items:
            return
        for item in after_items:
            pnl = float(item.get('pnl', 0.0) or 0.0)
            self.cycle_realized_pnl += pnl
            self.peak_cycle_pnl = max(self.peak_cycle_pnl, self.cycle_realized_pnl)
            self.drawdown_from_cycle_peak = self.cycle_realized_pnl - self.peak_cycle_pnl
            if self.drawdown_from_cycle_peak <= self.prop_max_drawdown_limit:
                self.max_drawdown_paused = True

            item['cycle_realized_pnl'] = self.cycle_realized_pnl
            item['best_day_pnl_in_cycle'] = self.best_day_pnl_in_cycle
            item['consistency_ratio'] = self._consistency_ratio()
            item['qualifying_days_count'] = len(self.qualifying_days)
            item['prop_daily_loss_limit'] = self.prop_daily_loss_limit
            item['prop_max_drawdown_limit'] = self.prop_max_drawdown_limit
            item['prop_qualifying_day_target'] = self.prop_qualifying_day_target
        self._sync_trade_log()

    def _prop_setup_allowed(self, setup_type: str, tier: str) -> bool:
        if not self.prop_mode:
            return True
        ratio = self._consistency_ratio()
        if ratio > self.prop_consistency_limit:
            return False
        if tier == 'A':
            return True
        if tier == 'B' and setup_type in self.prop_b_tier_allowed_setups:
            if ratio >= self.prop_consistency_warning:
                return False
            return True
        return False

    def _prop_can_trade(self) -> bool:
        if not self.prop_mode:
            return True
        if self.max_drawdown_paused or self.drawdown_from_cycle_peak <= self.prop_max_drawdown_limit:
            self.debug_counts['prop_block_max_drawdown'] += 1
            self._sync_debug()
            return False
        if self.realized_pnl_today <= self.prop_daily_loss_limit:
            self.debug_counts['prop_block_daily_loss'] += 1
            self._sync_debug()
            return False
        if self.daily_trade_count >= self.prop_daily_max_trades:
            self.debug_counts['prop_block_daily_trade_cap'] += 1
            self._sync_debug()
            return False
        if self.consecutive_losses >= self.prop_max_consecutive_losses:
            self.debug_counts['prop_block_consecutive_losses'] += 1
            self._sync_debug()
            return False
        if self._consistency_ratio() > self.prop_consistency_limit:
            self.debug_counts['prop_block_consistency'] += 1
            self._sync_debug()
            return False
        if self.payout_defense_mode and self.realized_pnl_today >= self.payout_defense_daily_profit_lock:
            self.debug_counts['prop_block_daily_trade_cap'] += 1
            self._sync_debug()
            return False
        return True

    def _effective_size(self) -> float:
        size = self.fixed_size
        if self.prop_mode and self.prop_reduce_size_after_drawdown and self.realized_pnl_today <= self.prop_drawdown_reduce_threshold:
            size *= self.prop_reduced_size_multiplier
        if self._consistency_ratio() >= self.prop_consistency_warning:
            size *= 0.5
        if len(self.qualifying_days) >= self.prop_required_qualifying_days:
            size *= 0.5
        return max(size, 0.01)

    def _setup_tier(self, setup_type: str, bridge_type: str, side: str) -> str:
        if bridge_type == 'CISD':
            return 'C'
        if setup_type == 'NYPM_CONTINUATION' and bridge_type in {'C2C3', 'MSS'} and side == 'SHORT':
            return 'A'
        if setup_type == 'NYPM_CONTINUATION' and bridge_type in {'C2C3', 'iFVG'} and side == 'LONG':
            return 'A'
        if setup_type == 'NYAM_CONTINUATION' and bridge_type in {'MSS', 'C2C3'} and side == 'SHORT':
            return 'A'
        if setup_type == 'NYAM_CONTINUATION' and bridge_type == 'MSS' and side == 'LONG':
            return 'A'
        if setup_type == 'LONDON_CONTINUATION' and bridge_type in {'MSS', 'C2C3', 'iFVG'}:
            return 'A'
        if setup_type == 'ASIA_CONTINUATION' and bridge_type in {'C2C3', 'iFVG'}:
            return 'B'
        if setup_type == 'NYPM_CONTINUATION' and bridge_type == 'iFVG':
            return 'B'
        if setup_type == 'NYAM_CONTINUATION' and bridge_type == 'iFVG':
            return 'B'
        return 'C'

    def _arm_long_pullback(self, row: pd.Series, high_now: float, low_now: float, atr3: float, i: int):
        if not self._bridge_allowed_for_session(self.state.setup_type, self.state.bridge_type):
            if self.state.bridge_type == 'CISD':
                self.debug_counts['blocked_cisd_session'] += 1
            elif self.state.bridge_type == 'iFVG':
                self.debug_counts['blocked_ifvg_session'] += 1
            self._sync_debug()
            self._clear_state()
            return

        tier = self._setup_tier(self.state.setup_type, self.state.bridge_type, 'LONG')
        if self.prop_trade_only_ranked_setups and self.prop_mode and not self._prop_setup_allowed(self.state.setup_type, tier):
            self.debug_counts['blocked_by_ranking'] += 1
            self._sync_debug()
            self._clear_state()
            return

        pullback_low = float(row.get('bull_bridge_low_30m', np.nan))
        pullback_high = float(row.get('bull_bridge_high_30m', np.nan))
        if not (np.isfinite(pullback_low) and np.isfinite(pullback_high)):
            return

        structural_stop = min(low_now, pullback_low) - (atr3 * self.stop_buffer_atr if atr3 > 0 else 0.0)
        entry_trigger = pullback_high + 0.25
        stop = max(structural_stop, entry_trigger - self.max_stop_points)
        stop = min(stop, entry_trigger - self.min_stop_points)
        if stop >= entry_trigger:
            return

        risk = entry_trigger - stop
        if risk > self.max_stop_points:
            self.debug_counts['reject_stop_cap_long'] += 1
            self._sync_debug()
            self._clear_state()
            return

        runner_target = self._hybrid_target_above(row, entry_trigger, risk)
        partial_target = entry_trigger + (risk * self.partial_rr)

        self.pending = PendingSignal(
            direction='long',
            entry_trigger=entry_trigger,
            stop_price=stop,
            target_price=runner_target,
            expiry_bar=i + self.pending_expiry_bars,
            setup_type=self.state.setup_type,
            bridge_type=self.state.bridge_type,
            entry_variant='PULLBACK_1M',
            pullback_low=pullback_low,
            pullback_high=pullback_high,
            partial_target=partial_target,
            runner_target=runner_target,
            setup_tier=tier,
        )
        self.debug_counts['arm_pending_long'] += 1
        self._sync_debug()

    def _arm_short_pullback(self, row: pd.Series, high_now: float, low_now: float, atr3: float, i: int):
        if not self._bridge_allowed_for_session(self.state.setup_type, self.state.bridge_type):
            if self.state.bridge_type == 'CISD':
                self.debug_counts['blocked_cisd_session'] += 1
            elif self.state.bridge_type == 'iFVG':
                self.debug_counts['blocked_ifvg_session'] += 1
            self._sync_debug()
            self._clear_state()
            return

        tier = self._setup_tier(self.state.setup_type, self.state.bridge_type, 'SHORT')
        if self.prop_trade_only_ranked_setups and self.prop_mode and not self._prop_setup_allowed(self.state.setup_type, tier):
            self.debug_counts['blocked_by_ranking'] += 1
            self._sync_debug()
            self._clear_state()
            return

        pullback_low = float(row.get('bear_bridge_low_30m', np.nan))
        pullback_high = float(row.get('bear_bridge_high_30m', np.nan))
        if not (np.isfinite(pullback_low) and np.isfinite(pullback_high)):
            return

        structural_stop = max(high_now, pullback_high) + (atr3 * self.stop_buffer_atr if atr3 > 0 else 0.0)
        entry_trigger = pullback_low - 0.25
        stop = min(structural_stop, entry_trigger + self.max_stop_points)
        stop = max(stop, entry_trigger + self.min_stop_points)
        if stop <= entry_trigger:
            return

        risk = stop - entry_trigger
        if risk > self.max_stop_points:
            self.debug_counts['reject_stop_cap_short'] += 1
            self._sync_debug()
            self._clear_state()
            return

        runner_target = self._hybrid_target_below(row, entry_trigger, risk)
        partial_target = entry_trigger - (risk * self.partial_rr)

        self.pending = PendingSignal(
            direction='short',
            entry_trigger=entry_trigger,
            stop_price=stop,
            target_price=runner_target,
            expiry_bar=i + self.pending_expiry_bars,
            setup_type=self.state.setup_type,
            bridge_type=self.state.bridge_type,
            entry_variant='PULLBACK_1M',
            pullback_low=pullback_low,
            pullback_high=pullback_high,
            partial_target=partial_target,
            runner_target=runner_target,
            setup_tier=tier,
        )
        self.debug_counts['arm_pending_short'] += 1
        self._sync_debug()

    def next(self):
        i = self._i()
        if i < 200:
            return super().next()
        row = self.m.iloc[i]

        if self.enable_news_filter:
            if self._must_flatten_for_news(row) and self.position:
                try:
                    self.position.close()
                    self.debug_counts['news_forced_flatten'] += 1
                    self._sync_debug()
                except Exception:
                    pass
            if self._current_news_event(row) is not None:
                self.debug_counts['prop_block_news'] += 1
                self._sync_debug()

        return super().next()
