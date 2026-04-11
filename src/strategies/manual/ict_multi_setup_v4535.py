
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.strategies.manual.ict_multi_setup_v452 import (
    ICT_MULTI_SETUP_V452,
    PendingSignal,
)


class ICT_MULTI_SETUP_V4535(ICT_MULTI_SETUP_V452):
    """
    V45.35

    Built from V45.2 as the base engine, with only the cleaner controls kept:
    - hard 30-point stop cap
    - optional high-impact news blackout support
    - soft Apex-style prop tracking
    - softer ranking and throttling than V45.3
    """

    # --- risk / target profile ---
    max_stop_points = 30.0

    # --- prop profile ---
    prop_mode = True
    prop_daily_loss_limit = -1000.0
    prop_daily_max_trades = 5
    prop_max_consecutive_losses = 3
    prop_reduce_size_after_drawdown = True
    prop_drawdown_reduce_threshold = -700.0
    prop_reduced_size_multiplier = 0.5

    # --- Apex-style tracking ---
    prop_max_drawdown_limit = -2000.0
    prop_consistency_limit = 0.50
    prop_consistency_soft_warning = 0.45
    prop_consistency_hard_block = 0.65
    prop_qualifying_day_target = 300.0
    prop_required_qualifying_days = 5

    # --- news blackout ---
    enable_news_filter = True
    news_flatten_minutes_before = 5
    news_resume_minutes_after = 5
    news_csv_filename = "high_impact_news_et.csv"

    # --- ranking ---
    prop_trade_only_ranked_setups = True
    allowed_prop_setup_tiers = {"A", "B"}
    prop_b_tier_allowed_setups = {
        "NYPM_CONTINUATION",
        "NYAM_CONTINUATION",
        "LONDON_CONTINUATION",
    }

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
            "reject_stop_cap_long": 0,
            "reject_stop_cap_short": 0,
            "prop_block_max_drawdown": 0,
            "prop_block_consistency": 0,
            "prop_block_news": 0,
            "news_forced_flatten": 0,
            "qualifying_day_count": 0,
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
                    if "event_time_et" not in df.columns:
                        continue
                    ts = pd.to_datetime(df["event_time_et"], errors="coerce")
                    ts = ts.dropna()
                    ts = [
                        t.tz_localize("America/New_York")
                        if t.tzinfo is None else t.tz_convert("America/New_York")
                        for t in ts
                    ]
                    return sorted(ts)
                except Exception:
                    return []
        return []

    def _row_ts_et(self, row: pd.Series):
        ts = row.name
        try:
            ts = pd.Timestamp(ts)
        except Exception:
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("America/New_York")

    def _current_news_event(self, row: pd.Series):
        if not self.news_events_et:
            return None
        row_ts = self._row_ts_et(row)
        if row_ts is None:
            return None
        for event_ts in self.news_events_et:
            if (event_ts - pd.Timedelta(minutes=self.news_flatten_minutes_before)) <= row_ts <= (
                event_ts + pd.Timedelta(minutes=self.news_resume_minutes_after)
            ):
                return event_ts
        return None

    def _must_flatten_for_news(self, row: pd.Series) -> bool:
        if not self.enable_news_filter:
            return False
        event_ts = self._current_news_event(row)
        if event_ts is None:
            return False
        row_ts = self._row_ts_et(row)
        return row_ts is not None and row_ts >= (event_ts - pd.Timedelta(minutes=self.news_flatten_minutes_before))

    def _consistency_ratio(self) -> float:
        cycle_profit = max(self.cycle_realized_pnl, 0.0)
        if cycle_profit <= 0:
            return 0.0
        return float(self.best_day_pnl_in_cycle) / float(cycle_profit)

    def _finalize_previous_day(self):
        if self.current_et_day is None:
            return
        self.daily_pnl_history[self.current_et_day] = self.realized_pnl_today
        if self.realized_pnl_today >= self.prop_qualifying_day_target:
            self.qualifying_days.add(self.current_et_day)
        self.best_day_pnl_in_cycle = max(self.best_day_pnl_in_cycle, self.realized_pnl_today)
        self.drawdown_from_cycle_peak = self.cycle_realized_pnl - self.peak_cycle_pnl
        self.debug_counts["qualifying_day_count"] = len(self.qualifying_days)
        self._sync_debug()

    def _update_day_reset(self, row: pd.Series):
        if self.current_et_day != row["et_date"]:
            if self.current_et_day is not None:
                self._finalize_previous_day()
            self.current_et_day = row["et_date"]
            self.realized_pnl_today = 0.0
            self.daily_trade_count = 0
            self.consecutive_losses = 0

    def _log_newly_closed_trades(self):
        try:
            closed = list(self.closed_trades)
        except Exception:
            return

        if len(closed) <= self.prev_closed_count:
            return

        new_items = closed[self.prev_closed_count:]
        for t in new_items:
            pnl = float(getattr(t, "pl", 0.0))
            self.realized_pnl_today += pnl
            self.cycle_realized_pnl += pnl
            self.peak_cycle_pnl = max(self.peak_cycle_pnl, self.cycle_realized_pnl)
            self.drawdown_from_cycle_peak = self.cycle_realized_pnl - self.peak_cycle_pnl

            if pnl < 0:
                self.consecutive_losses += 1
            elif pnl > 0:
                self.consecutive_losses = 0

            meta = self.open_trade_meta or {}
            planned_entry = meta.get("planned_entry_price", np.nan)
            planned_stop = meta.get("planned_stop_price", np.nan)
            planned_target = meta.get("planned_target_price", np.nan)

            stop_points = np.nan
            tp_points = np.nan
            planned_rr = np.nan
            if pd.notna(planned_entry) and pd.notna(planned_stop):
                stop_points = abs(float(planned_entry) - float(planned_stop))
            if pd.notna(planned_entry) and pd.notna(planned_target):
                tp_points = abs(float(planned_target) - float(planned_entry))
            if pd.notna(stop_points) and stop_points > 0 and pd.notna(tp_points):
                planned_rr = float(tp_points) / float(stop_points)

            self.__class__.last_trade_log.append({
                "side": "LONG" if float(t.size) > 0 else "SHORT",
                "setup_type": meta.get("setup_type", ""),
                "bridge_type": meta.get("bridge_type", ""),
                "entry_variant": meta.get("entry_variant", ""),
                "setup_tier": meta.get("setup_tier", ""),
                "planned_entry_price": planned_entry,
                "planned_stop_price": planned_stop,
                "planned_target_price": planned_target,
                "partial_target_price": meta.get("partial_target_price", np.nan),
                "runner_target_price": meta.get("runner_target_price", np.nan),
                "stop_points": stop_points,
                "tp_points": tp_points,
                "planned_rr": planned_rr,
                "entry_price": float(t.entry_price),
                "exit_price": float(t.exit_price),
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "return_pct": float(getattr(t, "pl_pct", np.nan)) if hasattr(t, "pl_pct") else np.nan,
                "pnl": pnl,
                "cycle_realized_pnl": self.cycle_realized_pnl,
                "best_day_pnl_in_cycle": self.best_day_pnl_in_cycle,
                "consistency_ratio": self._consistency_ratio(),
                "qualifying_days_count": len(self.qualifying_days),
                "prop_daily_loss_limit": self.prop_daily_loss_limit,
                "prop_max_drawdown_limit": self.prop_max_drawdown_limit,
                "prop_qualifying_day_target": self.prop_qualifying_day_target,
            })

        if new_items and not self.position:
            self.open_trade_meta = None

        self.prev_closed_count = len(closed)
        self._sync_trade_log()

    def _prop_setup_allowed(self, setup_type: str, tier: str) -> bool:
        if tier == "A":
            return True
        if tier == "B":
            return setup_type in self.prop_b_tier_allowed_setups
        return False

    def _prop_can_trade(self) -> bool:
        if not self.prop_mode:
            return True

        if self.max_drawdown_paused or self.drawdown_from_cycle_peak <= self.prop_max_drawdown_limit:
            self.max_drawdown_paused = True
            self.debug_counts["prop_block_max_drawdown"] += 1
            self._sync_debug()
            return False

        # Hard blocks only at actual limits.
        if self.realized_pnl_today <= self.prop_daily_loss_limit:
            self.debug_counts["prop_block_daily_loss"] += 1
            self._sync_debug()
            return False

        if self.daily_trade_count >= self.prop_daily_max_trades:
            self.debug_counts["prop_block_daily_trade_cap"] += 1
            self._sync_debug()
            return False

        if self.consecutive_losses >= self.prop_max_consecutive_losses:
            self.debug_counts["prop_block_consecutive_losses"] += 1
            self._sync_debug()
            return False

        ratio = self._consistency_ratio()
        if self.cycle_realized_pnl > 1500 and ratio >= self.prop_consistency_hard_block:
            self.debug_counts["prop_block_consistency"] += 1
            self._sync_debug()
            return False

        return True

    def _effective_size(self) -> float:
        size = self.fixed_size
        if not self.prop_mode:
            return size

        if self.prop_reduce_size_after_drawdown and self.realized_pnl_today <= self.prop_drawdown_reduce_threshold:
            size *= self.prop_reduced_size_multiplier

        ratio = self._consistency_ratio()
        if self.cycle_realized_pnl > 0 and ratio >= self.prop_consistency_soft_warning:
            size *= 0.75

        if len(self.qualifying_days) >= self.prop_required_qualifying_days:
            size *= 0.75

        return max(size, 0.01)

    def _setup_tier(self, setup_type: str, bridge_type: str, side: str) -> str:
        # Keep V45.2 ranking but slightly favor proven NYAM/NYPM continuations.
        base = super()._setup_tier(setup_type, bridge_type, side)
        if setup_type in {"NYPM_CONTINUATION", "NYAM_CONTINUATION"} and bridge_type in {"C2C3", "iFVG", "MSS"}:
            return "A"
        return base

    def _arm_long_pullback(self, row: pd.Series, high_now: float, low_now: float, atr3: float, i: int):
        if not self._bridge_allowed_for_session(self.state.setup_type, self.state.bridge_type):
            if self.state.bridge_type == "CISD":
                self.debug_counts["blocked_cisd_session"] += 1
            elif self.state.bridge_type == "iFVG":
                self.debug_counts["blocked_ifvg_session"] += 1
            self._sync_debug()
            self._clear_state()
            return

        tier = self._setup_tier(self.state.setup_type, self.state.bridge_type, "LONG")
        if self.prop_trade_only_ranked_setups and self.prop_mode and not self._prop_setup_allowed(self.state.setup_type, tier):
            self.debug_counts["blocked_by_ranking"] += 1
            self._sync_debug()
            self._clear_state()
            return

        pullback_low = float(row.get("bull_bridge_low_30m", np.nan))
        pullback_high = float(row.get("bull_bridge_high_30m", np.nan))
        if not (np.isfinite(pullback_low) and np.isfinite(pullback_high)):
            return

        structural_stop = min(low_now, pullback_low) - (atr3 * self.stop_buffer_atr if atr3 > 0 else 0.0)
        entry_trigger = pullback_high + 0.25
        stop = min(structural_stop, entry_trigger - self.min_stop_points)
        if stop >= entry_trigger:
            return

        risk = entry_trigger - stop
        if risk > self.max_stop_points:
            self.debug_counts["reject_stop_cap_long"] += 1
            self._sync_debug()
            self._clear_state()
            return

        runner_target = self._hybrid_target_above(row, entry_trigger, risk)
        partial_target = entry_trigger + (risk * self.partial_rr)

        self.pending = PendingSignal(
            direction="long",
            entry_trigger=entry_trigger,
            stop_price=stop,
            target_price=runner_target,
            expiry_bar=i + self.pending_expiry_bars,
            setup_type=self.state.setup_type,
            bridge_type=self.state.bridge_type,
            entry_variant="PULLBACK_1M",
            pullback_low=pullback_low,
            pullback_high=pullback_high,
            partial_target=partial_target,
            runner_target=runner_target,
            setup_tier=tier,
        )
        self.debug_counts["arm_pending_long"] += 1
        self._sync_debug()

    def _arm_short_pullback(self, row: pd.Series, high_now: float, low_now: float, atr3: float, i: int):
        if not self._bridge_allowed_for_session(self.state.setup_type, self.state.bridge_type):
            if self.state.bridge_type == "CISD":
                self.debug_counts["blocked_cisd_session"] += 1
            elif self.state.bridge_type == "iFVG":
                self.debug_counts["blocked_ifvg_session"] += 1
            self._sync_debug()
            self._clear_state()
            return

        tier = self._setup_tier(self.state.setup_type, self.state.bridge_type, "SHORT")
        if self.prop_trade_only_ranked_setups and self.prop_mode and not self._prop_setup_allowed(self.state.setup_type, tier):
            self.debug_counts["blocked_by_ranking"] += 1
            self._sync_debug()
            self._clear_state()
            return

        pullback_low = float(row.get("bear_bridge_low_30m", np.nan))
        pullback_high = float(row.get("bear_bridge_high_30m", np.nan))
        if not (np.isfinite(pullback_low) and np.isfinite(pullback_high)):
            return

        structural_stop = max(high_now, pullback_high) + (atr3 * self.stop_buffer_atr if atr3 > 0 else 0.0)
        entry_trigger = pullback_low - 0.25
        stop = max(structural_stop, entry_trigger + self.min_stop_points)
        if stop <= entry_trigger:
            return

        risk = stop - entry_trigger
        if risk > self.max_stop_points:
            self.debug_counts["reject_stop_cap_short"] += 1
            self._sync_debug()
            self._clear_state()
            return

        runner_target = self._hybrid_target_below(row, entry_trigger, risk)
        partial_target = entry_trigger - (risk * self.partial_rr)

        self.pending = PendingSignal(
            direction="short",
            entry_trigger=entry_trigger,
            stop_price=stop,
            target_price=runner_target,
            expiry_bar=i + self.pending_expiry_bars,
            setup_type=self.state.setup_type,
            bridge_type=self.state.bridge_type,
            entry_variant="PULLBACK_1M",
            pullback_low=pullback_low,
            pullback_high=pullback_high,
            partial_target=partial_target,
            runner_target=runner_target,
            setup_tier=tier,
        )
        self.debug_counts["arm_pending_short"] += 1
        self._sync_debug()

    def next(self):
        i = self._i()
        if i < 200:
            return

        row = self.m.iloc[i]
        self._update_day_reset(row)
        self._log_newly_closed_trades()

        if self.position and self._must_flatten_for_news(row):
            try:
                self.position.close()
                self.debug_counts["news_forced_flatten"] += 1
                self._sync_debug()
            except Exception:
                pass

        if self._current_news_event(row) is not None and not self.position:
            self.debug_counts["prop_block_news"] += 1
            self._sync_debug()

        super().next()
