import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class EvalConfig:
    account_size: float
    profit_target: float
    max_drawdown: float
    drawdown_type: str

    risk_per_trade: float
    daily_profit_target: float
    daily_soft_loss_stop: float
    max_trades_per_day: int
    pause_after_consecutive_losses: int


@dataclass
class PAConfig:
    account_size: float
    max_drawdown: float
    drawdown_type: str

    risk_per_trade: float
    daily_profit_target: float
    daily_soft_loss_stop: float
    max_trades_per_day: int
    pause_after_consecutive_losses: int

    payout_buffer: float
    payout_lock_days: int

    reduce_risk_after_payout: bool
    reduced_risk_per_trade: float


@dataclass
class DualModeProfile:
    name: str
    eval: EvalConfig
    pa: PAConfig


# ============================================================
# HELPERS
# ============================================================

def load_profiles(profile_path: str) -> dict:

    with open(profile_path, "r") as f:
        raw = yaml.safe_load(f)

    out = {}

    for name, cfg in raw.items():

        eval_cfg = EvalConfig(**cfg["eval"])
        pa_cfg = PAConfig(**cfg["pa"])

        out[name] = DualModeProfile(
            name=name,
            eval=eval_cfg,
            pa=pa_cfg,
        )

    return out


# ============================================================
# TRADE LOG LOADER
# ============================================================

def load_trade_log(path: str) -> pd.DataFrame:
    """
    Load a trade log from either the older research backtests or the newer
    event engine.

    Important fix:
        Event-engine logs use columns such as exit_time_et / entry_time_et.
        These values can contain mixed timezone offsets around DST changes.
        Pandas may parse that as object dtype unless utc=True is used, which
        then breaks `.dt.date`.

    This function therefore:
        1. Preferentially uses exit_time_et / exit_time for closed-trade date.
        2. Parses timestamps with utc=True and errors="coerce".
        3. Drops rows where timestamp parsing failed.
        4. Normalizes PnL to a common `pnl` column.
    """

    df = pd.read_csv(path)

    # Prefer exit timestamp because payout/eval lifecycle should be based on
    # when the trade actually realized PnL.
    timestamp_candidates = [
        # NEW EVENT ENGINE
        "exit_time_et",
        "exit_time",
        "exit_timestamp",
        "closed_at",

        # FALLBACKS
        "timestamp",
        "entry_time_et",
        "entry_time",
        "entry_timestamp",
        "opened_at",
    ]

    pnl_candidates = [
        "net_pnl_dollars",
        "net_pnl",
        "pnl",
        "realized_pnl",
    ]

    timestamp_col = None
    pnl_col = None

    # =====================================================
    # FIND TIMESTAMP COLUMN
    # =====================================================

    for col in timestamp_candidates:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        print("\nAvailable columns:")
        print(df.columns.tolist())
        raise ValueError("No timestamp column found")

    # =====================================================
    # FIND PNL COLUMN
    # =====================================================

    for col in pnl_candidates:
        if col in df.columns:
            pnl_col = col
            break

    if pnl_col is None:
        print("\nAvailable columns:")
        print(df.columns.tolist())
        raise ValueError("No pnl column found")

    # =====================================================
    # NORMALIZE
    # =====================================================

    df["timestamp"] = pd.to_datetime(
        df[timestamp_col],
        utc=True,
        errors="coerce",
    )

    bad_timestamp_rows = int(df["timestamp"].isna().sum())
    if bad_timestamp_rows:
        print(
            f"⚠️ Dropping {bad_timestamp_rows} rows with unparseable timestamps "
            f"from column {timestamp_col!r}"
        )

    df = df.dropna(subset=["timestamp"]).copy()

    df["pnl"] = pd.to_numeric(
        df[pnl_col],
        errors="coerce",
    ).fillna(0.0)

    df = df.sort_values("timestamp").reset_index(drop=True)

    # `.dt` is safe now because utc=True guarantees datetime64[ns, UTC].
    df["trade_date"] = df["timestamp"].dt.date

    return df




# ============================================================
# MAIN SIMULATOR
# ============================================================

class DualModeLifecycleSimulator:
    """
    Correct account lifecycle and payout-cycle accounting:

        eval pass
        -> PA account starts
        -> payout cycle starts
        -> trade until PA payout rules are met
        -> receive payout capped by max payout
        -> reset payout cycle
        -> continue same PA
        -> if PA blows, start new eval
        -> repeat

    Payout eligibility is based on profit SINCE LAST PAYOUT, not on fixed
    reporting windows. The target payout frequency is only used for reporting.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        profile: DualModeProfile,
        pa_df: pd.DataFrame | None = None,
        eval_risk_multiplier: float = 1.0,
        pa_risk_multiplier: float = 1.0,
        target_payout_frequency_days: int = 10,
        min_weekly_payout_amount: float = 1000.0,
        pa_min_payout_trading_days: int = 5,
        pa_consistency_rule_pct: float = 50.0,
        pa_min_payout_amount: float = 250.0,
        pa_max_payout_amount: float = 2000.0,
        enable_first_payout_buffer_mode: bool = False,
        first_payout_build_balance: float = 53000.0,
        first_payout_amount: float = 1000.0,
        retained_cushion_after_payout: float = 2000.0,
        pa_payout_buffer_override: float | None = None,
        pa_acceleration_daily_profit_target: float = 0.0,
        enable_pa_volatility_smoother: bool = True,
        smoother_after_profit_day_threshold: float = 750.0,
        smoother_after_profit_risk_multiplier: float = 0.75,
        smoother_loss_streak_threshold: int = 2,
        smoother_after_loss_risk_multiplier: float = 0.50,
        smoother_post_payout_risk_multiplier: float = 0.50,
        smoother_buffer_floor: float = 1000.0,
        smoother_buffer_risk_multiplier: float = 0.50,
        enable_consistency_repair: bool = True,
        consistency_repair_risk_multiplier: float = 0.50,
        consistency_repair_daily_profit_target: float = 250.0,
        aggressive_eval_until_pass: bool = False,
    ):
        self.df = df
        self.pa_df = pa_df if pa_df is not None else df
        self.profile = profile

        self.eval_risk_multiplier = float(eval_risk_multiplier)
        self.pa_risk_multiplier = float(pa_risk_multiplier)
        self.target_payout_frequency_days = int(target_payout_frequency_days)
        self.min_weekly_payout_amount = float(min_weekly_payout_amount)

        self.pa_min_payout_trading_days = int(pa_min_payout_trading_days)
        self.pa_consistency_rule_pct = float(pa_consistency_rule_pct)
        self.pa_min_payout_amount = float(pa_min_payout_amount)
        self.pa_max_payout_amount = float(pa_max_payout_amount)

        self.enable_first_payout_buffer_mode = bool(enable_first_payout_buffer_mode)
        self.first_payout_build_balance = float(first_payout_build_balance)
        self.first_payout_amount = float(first_payout_amount)
        self.retained_cushion_after_payout = float(retained_cushion_after_payout)

        self.pa_payout_buffer_override = (
            None
            if pa_payout_buffer_override is None or float(pa_payout_buffer_override) < 0
            else float(pa_payout_buffer_override)
        )
        self.pa_acceleration_daily_profit_target = float(pa_acceleration_daily_profit_target)

        self.enable_pa_volatility_smoother = bool(enable_pa_volatility_smoother)
        self.smoother_after_profit_day_threshold = float(smoother_after_profit_day_threshold)
        self.smoother_after_profit_risk_multiplier = float(smoother_after_profit_risk_multiplier)
        self.smoother_loss_streak_threshold = int(smoother_loss_streak_threshold)
        self.smoother_after_loss_risk_multiplier = float(smoother_after_loss_risk_multiplier)
        self.smoother_post_payout_risk_multiplier = float(smoother_post_payout_risk_multiplier)
        self.smoother_buffer_floor = float(smoother_buffer_floor)
        self.smoother_buffer_risk_multiplier = float(smoother_buffer_risk_multiplier)

        self.enable_consistency_repair = bool(enable_consistency_repair)
        self.consistency_repair_risk_multiplier = float(consistency_repair_risk_multiplier)
        self.consistency_repair_daily_profit_target = float(consistency_repair_daily_profit_target)
        self.aggressive_eval_until_pass = bool(aggressive_eval_until_pass)

        self.cycles = []
        self.events = []
        self.daily_rows = []
        self.payout_rows = []
        self.payout_cycle_rows = []

    def log_event(self, ts, event_type, cycle_id, details=None):
        self.events.append({
            "timestamp": ts,
            "event_type": event_type,
            "cycle_id": cycle_id,
            "details": details,
        })

    @staticmethod
    def _first_index_at_or_after(trades, ts):
        for i, trade in enumerate(trades):
            if trade["timestamp"] >= ts:
                return i
        return len(trades)

    @staticmethod
    def _first_index_after(trades, ts):
        for i, trade in enumerate(trades):
            if trade["timestamp"] > ts:
                return i
        return len(trades)

    def run(self):
        eval_trades = self.df.to_dict("records")
        pa_trades = self.pa_df.to_dict("records")

        eval_idx = 0
        cycle_id = 1

        while eval_idx < len(eval_trades):
            result = self.run_one_account_lifecycle(
                eval_trades=eval_trades,
                pa_trades=pa_trades,
                eval_start_idx=eval_idx,
                cycle_id=cycle_id,
            )

            self.cycles.append(result["cycle"])

            next_idx = int(result["next_eval_idx"])
            if next_idx <= eval_idx:
                next_idx = eval_idx + 1

            eval_idx = next_idx
            cycle_id += 1

        return self.build_summary()

    def _day_lock(self, ts, cycle_id, cfg, daily_pnl, daily_trade_count, consecutive_losses):
        if daily_trade_count >= cfg.max_trades_per_day:
            self.log_event(ts, "MAX_TRADES_DAY_LOCK", cycle_id)
            return True, "max_trades_per_day"

        if daily_pnl <= -abs(cfg.daily_soft_loss_stop):
            self.log_event(ts, "DAILY_SOFT_LOSS_LOCK", cycle_id)
            return True, "daily_soft_loss_stop"

        if daily_pnl >= cfg.daily_profit_target:
            self.log_event(ts, "DAILY_PROFIT_TARGET_LOCK", cycle_id)
            return True, "daily_profit_target"

        if consecutive_losses >= cfg.pause_after_consecutive_losses:
            self.log_event(ts, "CONSECUTIVE_LOSS_LOCK", cycle_id)
            return True, "consecutive_losses"

        return False, ""

    def _daily_row(self, cycle_id, trade_day, mode, balance, daily_pnl, daily_trade_count, locked, reason):
        self.daily_rows.append({
            "cycle_id": cycle_id,
            "date": trade_day,
            "mode": mode,
            "balance": balance,
            "daily_pnl": daily_pnl,
            "daily_trade_count": daily_trade_count,
            "day_locked": locked,
            "day_lock_reason": reason,
        })

    def _payout_cycle_stats(self, cycle_day_pnls, cycle_profit):
        best_day = max(cycle_day_pnls.values()) if cycle_day_pnls else 0.0
        consistency_limit = max(0.0, cycle_profit) * (self.pa_consistency_rule_pct / 100.0)
        consistency_ok = cycle_profit > 0 and best_day <= consistency_limit
        return best_day, consistency_limit, consistency_ok

    def run_one_account_lifecycle(self, eval_trades, pa_trades, eval_start_idx, cycle_id):
        eval_cfg = self.profile.eval
        pa_cfg = self.profile.pa

        # =====================================================
        # EVAL PHASE
        # =====================================================

        eval_balance = eval_cfg.account_size
        eval_high_watermark = eval_balance

        eval_start_ts = eval_trades[eval_start_idx]["timestamp"]
        eval_end_ts = None
        eval_pass_ts = None

        eval_active_days = set()
        eval_trade_count = 0
        eval_daily_pnl = 0.0
        eval_daily_trade_count = 0
        eval_consecutive_losses = 0
        eval_current_day = None
        eval_day_locked = False
        eval_day_lock_reason = ""

        self.log_event(eval_start_ts, "EVAL_STARTED", cycle_id)

        eval_idx = eval_start_idx

        while eval_idx < len(eval_trades):
            trade = eval_trades[eval_idx]
            ts = trade["timestamp"]
            trade_day = trade["trade_date"]

            if eval_current_day != trade_day:
                eval_current_day = trade_day
                eval_daily_pnl = 0.0
                eval_daily_trade_count = 0
                eval_consecutive_losses = 0
                eval_day_locked = False
                eval_day_lock_reason = ""

            if eval_day_locked:
                eval_idx += 1
                continue

            scaled_pnl = float(trade["pnl"]) * self.eval_risk_multiplier
            eval_balance += scaled_pnl
            eval_daily_pnl += scaled_pnl
            eval_daily_trade_count += 1
            eval_trade_count += 1
            eval_active_days.add(trade_day)
            eval_end_ts = ts

            if scaled_pnl < 0:
                eval_consecutive_losses += 1
            else:
                eval_consecutive_losses = 0

            eval_high_watermark = max(eval_high_watermark, eval_balance)
            eval_drawdown = eval_balance - eval_high_watermark

            self._daily_row(
                cycle_id, trade_day, "eval", eval_balance, eval_daily_pnl,
                eval_daily_trade_count, eval_day_locked, eval_day_lock_reason
            )

            if eval_drawdown <= -abs(eval_cfg.max_drawdown):
                self.log_event(ts, "EVAL_FAILED_MAX_LOSS", cycle_id)
                return {
                    "next_eval_idx": eval_idx + 1,
                    "cycle": {
                        "cycle_id": cycle_id,
                        "result": "eval_failed",
                        "eval_passed": False,
                        "pa_blown": False,
                        "payout_count": 0,
                        "payout_total": 0.0,
                        "final_balance": eval_balance,
                        "net_pnl": eval_balance - eval_cfg.account_size,
                        "eval_start": eval_start_ts,
                        "eval_end": ts,
                        "eval_calendar_days": (pd.Timestamp(ts).date() - pd.Timestamp(eval_start_ts).date()).days,
                        "eval_active_days": len(eval_active_days),
                        "eval_trades": eval_trade_count,
                        "pa_start": None,
                        "pa_end": None,
                        "pa_calendar_days": 0,
                        "pa_active_days": 0,
                        "pa_trades": 0,
                    }
                }

            if eval_balance >= eval_cfg.account_size + eval_cfg.profit_target:
                eval_pass_ts = ts
                eval_end_ts = ts
                self.log_event(ts, "EVAL_PASSED", cycle_id)
                break

            eval_day_locked, eval_day_lock_reason = self._day_lock(
                ts, cycle_id, eval_cfg, eval_daily_pnl, eval_daily_trade_count, eval_consecutive_losses
            )

            eval_idx += 1

        if eval_pass_ts is None:
            last_ts = eval_end_ts or eval_start_ts
            return {
                "next_eval_idx": len(eval_trades),
                "cycle": {
                    "cycle_id": cycle_id,
                    "result": "data_end_before_eval_pass",
                    "eval_passed": False,
                    "pa_blown": False,
                    "payout_count": 0,
                    "payout_total": 0.0,
                    "final_balance": eval_balance,
                    "net_pnl": eval_balance - eval_cfg.account_size,
                    "eval_start": eval_start_ts,
                    "eval_end": last_ts,
                    "eval_calendar_days": (pd.Timestamp(last_ts).date() - pd.Timestamp(eval_start_ts).date()).days,
                    "eval_active_days": len(eval_active_days),
                    "eval_trades": eval_trade_count,
                    "pa_start": None,
                    "pa_end": None,
                    "pa_calendar_days": 0,
                    "pa_active_days": 0,
                    "pa_trades": 0,
                }
            }

        next_eval_idx_after_this_account = eval_idx + 1

        # =====================================================
        # PA PHASE
        # =====================================================

        pa_start_ts = eval_pass_ts
        pa_balance = pa_cfg.account_size
        pa_start_balance = pa_balance
        pa_high_watermark = pa_balance

        pa_idx = self._first_index_at_or_after(pa_trades, pa_start_ts)

        pa_current_day = None
        pa_daily_pnl = 0.0
        pa_daily_trade_count = 0
        pa_consecutive_losses = 0
        pa_day_locked = False
        pa_day_lock_reason = ""
        payout_lock_days_remaining = 0

        last_completed_pa_day_pnl = 0.0
        prior_day_profit_smoother_active = False
        loss_streak_smoother_active = False
        post_payout_smoother_active = False
        buffer_smoother_active = False
        smoother_trade_count = 0
        smoother_events = 0

        pa_active_days = set()
        pa_trade_count = 0
        pa_end_ts = pa_start_ts

        payout_count = 0
        payout_total = 0.0
        last_payout_ts = None

        # Payout cycle state. This resets ONLY after an actual payout.
        payout_cycle_number = 1
        payout_cycle_start_ts = pa_start_ts
        payout_cycle_start_balance = pa_balance
        payout_cycle_trade_count = 0
        payout_cycle_active_days = set()
        payout_cycle_day_pnls = {}

        consistency_repair_active = False
        consistency_repair_activations = 0
        consistency_repair_trades = 0
        consistency_repair_days = set()

        # Reporting window state. This is independent from payout eligibility.
        report_window_start_ts = pa_start_ts
        report_window_start_balance = pa_balance
        report_window_trade_count = 0
        report_window_active_days = set()
        report_window_day_pnls = {}
        report_window_number = 1

        self.log_event(pa_start_ts, "PA_STARTED", cycle_id)

        while pa_idx < len(pa_trades):
            trade = pa_trades[pa_idx]
            ts = trade["timestamp"]

            if ts < pa_start_ts:
                raise RuntimeError(f"PA timestamp leak: PA trade {ts} before PA start {pa_start_ts}")

            trade_day = trade["trade_date"]
            pa_end_ts = ts

            if pa_current_day != trade_day:
                # Reporting-only target window.
                if self.target_payout_frequency_days > 0:
                    days_elapsed = (pd.Timestamp(ts).date() - pd.Timestamp(report_window_start_ts).date()).days
                    if days_elapsed >= self.target_payout_frequency_days:
                        report_profit = pa_balance - report_window_start_balance
                        best_day, consistency_limit, consistency_ok = self._payout_cycle_stats(
                            report_window_day_pnls, report_profit
                        )
                        self.payout_cycle_rows.append({
                            "cycle_id": cycle_id,
                            "pa_start_timestamp": pa_start_ts,
                            "timestamp_valid": ts >= pa_start_ts,
                            "weekly_cycle_number": report_window_number,
                            "cycle_start": report_window_start_ts,
                            "cycle_end": ts,
                            "target_frequency_days": self.target_payout_frequency_days,
                            "calendar_days": days_elapsed,
                            "active_trading_days": len(report_window_active_days),
                            "trades": report_window_trade_count,
                            "cycle_pnl": report_profit,
                            "target_met": report_profit >= self.min_weekly_payout_amount,
                            "min_weekly_payout_amount": self.min_weekly_payout_amount,
                            "payout_triggered": False,
                            "payout_amount": 0.0,
                            "best_payout_cycle_day": best_day,
                            "consistency_limit": consistency_limit,
                            "consistency_ok": consistency_ok,
                            "min_days_ok": len(report_window_active_days) >= self.pa_min_payout_trading_days,
                            "min_payout_ok": False,
                            "consistency_rule_pct": self.pa_consistency_rule_pct,
                            "min_trading_days_required": self.pa_min_payout_trading_days,
                            "min_payout_amount": self.pa_min_payout_amount,
                            "max_payout_amount": self.pa_max_payout_amount,
                            "row_type": "reporting_window",
                        })

                        report_window_number += 1
                        report_window_start_ts = ts
                        report_window_start_balance = pa_balance
                        report_window_trade_count = 0
                        report_window_active_days = set()
                        report_window_day_pnls = {}

                if pa_current_day is not None:
                    last_completed_pa_day_pnl = pa_daily_pnl

                prior_day_profit_smoother_active = (
                    self.enable_pa_volatility_smoother
                    and last_completed_pa_day_pnl >= self.smoother_after_profit_day_threshold
                )

                if prior_day_profit_smoother_active:
                    smoother_events += 1
                    self.log_event(
                        ts,
                        "PA_SMOOTHER_AFTER_PROFIT_DAY",
                        cycle_id,
                        details=f"prior_day_pnl=${last_completed_pa_day_pnl:.2f}",
                    )

                pa_current_day = trade_day
                pa_daily_pnl = 0.0
                pa_daily_trade_count = 0
                pa_consecutive_losses = 0
                pa_day_locked = False
                pa_day_lock_reason = ""

                if payout_lock_days_remaining > 0:
                    payout_lock_days_remaining -= 1

                post_payout_smoother_active = (
                    self.enable_pa_volatility_smoother
                    and payout_lock_days_remaining > 0
                )

            if pa_day_locked:
                pa_idx += 1
                continue

            effective_risk = pa_cfg.risk_per_trade * self.pa_risk_multiplier

            if payout_lock_days_remaining > 0 and pa_cfg.reduce_risk_after_payout:
                effective_risk = min(effective_risk, pa_cfg.reduced_risk_per_trade)

            retained_profit = pa_balance - pa_start_balance
            if retained_profit >= 2000:
                effective_risk = min(effective_risk, pa_cfg.reduced_risk_per_trade)

            if consistency_repair_active and self.enable_consistency_repair:
                effective_risk = min(
                    effective_risk,
                    pa_cfg.risk_per_trade * self.consistency_repair_risk_multiplier,
                )

            if self.enable_pa_volatility_smoother:
                retained_profit_for_smoother = pa_balance - pa_start_balance
                effective_payout_buffer_for_smoother = (
                    self.pa_payout_buffer_override
                    if self.pa_payout_buffer_override is not None
                    else pa_cfg.payout_buffer
                )
                buffer_above_floor = retained_profit_for_smoother - effective_payout_buffer_for_smoother

                buffer_smoother_active = buffer_above_floor < self.smoother_buffer_floor
                loss_streak_smoother_active = (
                    pa_consecutive_losses >= self.smoother_loss_streak_threshold
                )

                if prior_day_profit_smoother_active:
                    effective_risk = min(
                        effective_risk,
                        pa_cfg.risk_per_trade * self.smoother_after_profit_risk_multiplier,
                    )

                if loss_streak_smoother_active:
                    effective_risk = min(
                        effective_risk,
                        pa_cfg.risk_per_trade * self.smoother_after_loss_risk_multiplier,
                    )

                if post_payout_smoother_active:
                    effective_risk = min(
                        effective_risk,
                        pa_cfg.risk_per_trade * self.smoother_post_payout_risk_multiplier,
                    )

                if buffer_smoother_active:
                    effective_risk = min(
                        effective_risk,
                        pa_cfg.risk_per_trade * self.smoother_buffer_risk_multiplier,
                    )

            if self.enable_pa_volatility_smoother and effective_risk < (pa_cfg.risk_per_trade * self.pa_risk_multiplier):
                smoother_trade_count += 1

            scaled_pnl = float(trade["pnl"]) * (effective_risk / 150.0)

            pa_balance += scaled_pnl
            pa_daily_pnl += scaled_pnl
            pa_daily_trade_count += 1
            pa_trade_count += 1
            pa_active_days.add(trade_day)

            payout_cycle_trade_count += 1
            payout_cycle_active_days.add(trade_day)
            payout_cycle_day_pnls[trade_day] = payout_cycle_day_pnls.get(trade_day, 0.0) + scaled_pnl

            report_window_trade_count += 1
            report_window_active_days.add(trade_day)
            report_window_day_pnls[trade_day] = report_window_day_pnls.get(trade_day, 0.0) + scaled_pnl

            if consistency_repair_active:
                consistency_repair_trades += 1
                consistency_repair_days.add(trade_day)

            if scaled_pnl < 0:
                pa_consecutive_losses += 1
            else:
                pa_consecutive_losses = 0

            pa_high_watermark = max(pa_high_watermark, pa_balance)
            pa_drawdown = pa_balance - pa_high_watermark

            self._daily_row(
                cycle_id, trade_day, "pa", pa_balance, pa_daily_pnl,
                pa_daily_trade_count, pa_day_locked, pa_day_lock_reason
            )

            if pa_drawdown <= -abs(pa_cfg.max_drawdown):
                self.log_event(ts, "PA_BLOWN_MAX_LOSS", cycle_id)
                self.log_event(
                    ts,
                    "NEXT_EVAL_ALLOWED_AFTER_PA_BLOW",
                    cycle_id,
                    details=f"next_eval_idx={self._first_index_after(eval_trades, ts)}",
                )
                return {
                    "next_eval_idx": self._first_index_after(eval_trades, ts),
                    "cycle": {
                        "cycle_id": cycle_id,
                        "result": "pa_blown",
                        "eval_passed": True,
                        "pa_blown": True,
                        "payout_count": payout_count,
                        "payout_total": payout_total,
                        "final_balance": pa_balance,
                        "net_pnl": pa_balance - pa_start_balance,
                        "eval_start": eval_start_ts,
                        "eval_end": eval_pass_ts,
                        "eval_calendar_days": (pd.Timestamp(eval_pass_ts).date() - pd.Timestamp(eval_start_ts).date()).days,
                        "eval_active_days": len(eval_active_days),
                        "eval_trades": eval_trade_count,
                        "pa_start": pa_start_ts,
                        "pa_end": ts,
                        "pa_blow_timestamp": ts,
                        "next_eval_start_idx": self._first_index_after(eval_trades, ts),
                        "pa_calendar_days": (pd.Timestamp(ts).date() - pd.Timestamp(pa_start_ts).date()).days,
                        "pa_active_days": len(pa_active_days),
                        "pa_trades": pa_trade_count,
                        "consistency_repair_activations": consistency_repair_activations,
                        "pa_volatility_smoother_enabled": self.enable_pa_volatility_smoother,
                        "smoother_trade_count": smoother_trade_count,
                        "smoother_events": smoother_events,
                    }
                }

            pa_day_locked, pa_day_lock_reason = self._day_lock(
                ts, cycle_id, pa_cfg, pa_daily_pnl, pa_daily_trade_count, pa_consecutive_losses
            )

            if (
                consistency_repair_active
                and self.enable_consistency_repair
                and pa_daily_pnl >= self.consistency_repair_daily_profit_target
            ):
                pa_day_locked = True
                pa_day_lock_reason = "consistency_repair_daily_profit_target"
                self.log_event(
                    ts,
                    "CONSISTENCY_REPAIR_DAILY_TARGET_LOCK",
                    cycle_id,
                    details=f"daily_pnl=${pa_daily_pnl:.2f}",
                )

            if (
                self.pa_acceleration_daily_profit_target > 0
                and pa_daily_pnl >= self.pa_acceleration_daily_profit_target
            ):
                pa_day_locked = True
                pa_day_lock_reason = "pa_acceleration_daily_profit_target"
                self.log_event(
                    ts,
                    "PA_ACCELERATION_DAILY_TARGET_LOCK",
                    cycle_id,
                    details=f"daily_pnl=${pa_daily_pnl:.2f}",
                )

            # =================================================
            # TRUE PAYOUT ELIGIBILITY
            # =================================================

            retained_profit = pa_balance - pa_start_balance

            # =================================================
            # PAYOUT BUFFER MODEL
            # =================================================
            #
            # Default model:
            #   payout availability = retained profit - payout buffer.
            #
            # First-payout buffer mode:
            #   1. First payout waits until PA reaches first_payout_build_balance,
            #      e.g. $53,000.
            #   2. First payout is capped to first_payout_amount, e.g. $1,000.
            #      This leaves a $2,000 cushion on a $50k PA.
            #   3. Later payouts do NOT rebuild the whole first buffer again.
            #      They only protect retained_cushion_after_payout, e.g. $2,000.
            #
            if self.enable_first_payout_buffer_mode:
                if payout_count == 0:
                    first_payout_ready = pa_balance >= self.first_payout_build_balance
                    available_payout = (
                        min(
                            max(0.0, pa_balance - pa_start_balance),
                            self.first_payout_amount,
                        )
                        if first_payout_ready
                        else 0.0
                    )
                    effective_payout_buffer = self.first_payout_build_balance - pa_start_balance
                    payout_cap_for_this_request = self.first_payout_amount
                    min_balance_after_payout = pa_start_balance + self.retained_cushion_after_payout
                else:
                    min_balance_after_payout = pa_start_balance + self.retained_cushion_after_payout
                    available_payout = max(0.0, pa_balance - min_balance_after_payout)
                    effective_payout_buffer = self.retained_cushion_after_payout
                    payout_cap_for_this_request = self.pa_max_payout_amount
            else:
                effective_payout_buffer = (
                    self.pa_payout_buffer_override
                    if self.pa_payout_buffer_override is not None
                    else pa_cfg.payout_buffer
                )
                available_payout = max(0.0, retained_profit - effective_payout_buffer)
                payout_cap_for_this_request = self.pa_max_payout_amount
                min_balance_after_payout = pa_start_balance + effective_payout_buffer

            payout_cycle_profit = max(0.0, pa_balance - payout_cycle_start_balance)
            best_cycle_day, consistency_limit, consistency_ok = self._payout_cycle_stats(
                payout_cycle_day_pnls, payout_cycle_profit
            )

            min_days_ok = len(payout_cycle_active_days) >= self.pa_min_payout_trading_days
            min_payout_ok = available_payout >= self.pa_min_payout_amount

            if (
                min_days_ok
                and min_payout_ok
                and not consistency_ok
                and self.enable_consistency_repair
                and not consistency_repair_active
            ):
                consistency_repair_active = True
                consistency_repair_activations += 1
                needed_total = (
                    best_cycle_day / (self.pa_consistency_rule_pct / 100.0)
                    if self.pa_consistency_rule_pct > 0
                    else 0.0
                )
                self.log_event(
                    ts,
                    "CONSISTENCY_REPAIR_STARTED",
                    cycle_id,
                    details=(
                        f"cycle_profit=${payout_cycle_profit:.2f}; "
                        f"best_day=${best_cycle_day:.2f}; "
                        f"needed_total=${needed_total:.2f}"
                    ),
                )

            if min_days_ok and consistency_ok and min_payout_ok:
                payout_amount = min(available_payout, payout_cap_for_this_request)

                # Never allow a payout to reduce the PA below the retained cushion.
                payout_amount = min(
                    payout_amount,
                    max(0.0, pa_balance - min_balance_after_payout),
                )

                if ts < pa_start_ts:
                    raise RuntimeError(f"Invalid payout timestamp {ts}: before PA start {pa_start_ts}")

                payout_total += payout_amount
                payout_count += 1
                pa_balance -= payout_amount
                payout_lock_days_remaining = pa_cfg.payout_lock_days

                days_since_pa_start = (pd.Timestamp(ts).date() - pd.Timestamp(pa_start_ts).date()).days
                days_since_last_payout = (
                    (pd.Timestamp(ts).date() - pd.Timestamp(last_payout_ts).date()).days
                    if last_payout_ts is not None
                    else days_since_pa_start
                )
                payout_cycle_days = (pd.Timestamp(ts).date() - pd.Timestamp(payout_cycle_start_ts).date()).days

                self.payout_rows.append({
                    "cycle_id": cycle_id,
                    "timestamp": ts,
                    "pa_start_timestamp": pa_start_ts,
                    "timestamp_valid": ts >= pa_start_ts,
                    "payout_number": payout_count,
                    "payout_amount": payout_amount,
                    "days_since_pa_start": days_since_pa_start,
                    "days_since_last_payout": days_since_last_payout,
                    "pa_trades_to_payout": pa_trade_count,
                    "pa_active_days_to_payout": len(pa_active_days),
                    "payout_cycle_active_days": len(payout_cycle_active_days),
                    "payout_cycle_trades": payout_cycle_trade_count,
                    "payout_cycle_calendar_days": payout_cycle_days,
                    "payout_cycle_profit": payout_cycle_profit,
                    "best_payout_cycle_day": best_cycle_day,
                    "consistency_limit": consistency_limit,
                    "consistency_ok": consistency_ok,
                    "min_days_ok": min_days_ok,
                    "min_payout_ok": min_payout_ok,
                    "available_payout": available_payout,
                    "retained_profit": retained_profit,
                    "payout_buffer": effective_payout_buffer,
                    "first_payout_buffer_mode": self.enable_first_payout_buffer_mode,
                    "first_payout_build_balance": self.first_payout_build_balance,
                    "first_payout_amount": self.first_payout_amount,
                    "retained_cushion_after_payout": self.retained_cushion_after_payout,
                    "min_balance_after_payout": min_balance_after_payout,
                    "payout_cap_for_this_request": payout_cap_for_this_request,
                    "min_payout_amount": self.pa_min_payout_amount,
                    "max_payout_amount": self.pa_max_payout_amount,
                    "consistency_rule_pct": self.pa_consistency_rule_pct,
                    "consistency_repair_active_before_payout": consistency_repair_active,
                    "consistency_repair_activations": consistency_repair_activations,
                    "consistency_repair_trades": consistency_repair_trades,
                    "consistency_repair_active_days": len(consistency_repair_days),
                    "pa_volatility_smoother_enabled": self.enable_pa_volatility_smoother,
                    "smoother_trade_count": smoother_trade_count,
                    "smoother_events": smoother_events,
                    "prior_day_profit_smoother_active": prior_day_profit_smoother_active,
                    "loss_streak_smoother_active": loss_streak_smoother_active,
                    "post_payout_smoother_active": post_payout_smoother_active,
                    "buffer_smoother_active": buffer_smoother_active,
                })

                self.payout_cycle_rows.append({
                    "cycle_id": cycle_id,
                    "pa_start_timestamp": pa_start_ts,
                    "timestamp_valid": ts >= pa_start_ts,
                    "weekly_cycle_number": payout_cycle_number,
                    "cycle_start": payout_cycle_start_ts,
                    "cycle_end": ts,
                    "target_frequency_days": self.target_payout_frequency_days,
                    "calendar_days": payout_cycle_days,
                    "active_trading_days": len(payout_cycle_active_days),
                    "trades": payout_cycle_trade_count,
                    "cycle_pnl": payout_cycle_profit,
                    "target_met": payout_cycle_profit >= self.min_weekly_payout_amount,
                    "min_weekly_payout_amount": self.min_weekly_payout_amount,
                    "payout_triggered": True,
                    "payout_amount": payout_amount,
                    "best_payout_cycle_day": best_cycle_day,
                    "consistency_limit": consistency_limit,
                    "consistency_ok": consistency_ok,
                    "min_days_ok": min_days_ok,
                    "min_payout_ok": min_payout_ok,
                    "available_payout": available_payout,
                    "retained_profit": retained_profit,
                    "payout_buffer": effective_payout_buffer,
                    "first_payout_buffer_mode": self.enable_first_payout_buffer_mode,
                    "first_payout_build_balance": self.first_payout_build_balance,
                    "first_payout_amount": self.first_payout_amount,
                    "retained_cushion_after_payout": self.retained_cushion_after_payout,
                    "min_balance_after_payout": min_balance_after_payout,
                    "payout_cap_for_this_request": payout_cap_for_this_request,
                    "consistency_rule_pct": self.pa_consistency_rule_pct,
                    "min_trading_days_required": self.pa_min_payout_trading_days,
                    "min_payout_amount": self.pa_min_payout_amount,
                    "max_payout_amount": self.pa_max_payout_amount,
                    "row_type": "actual_payout_cycle",
                    "consistency_repair_active_before_payout": consistency_repair_active,
                    "consistency_repair_activations": consistency_repair_activations,
                    "consistency_repair_trades": consistency_repair_trades,
                    "consistency_repair_active_days": len(consistency_repair_days),
                    "pa_volatility_smoother_enabled": self.enable_pa_volatility_smoother,
                    "smoother_trade_count": smoother_trade_count,
                    "smoother_events": smoother_events,
                    "prior_day_profit_smoother_active": prior_day_profit_smoother_active,
                    "loss_streak_smoother_active": loss_streak_smoother_active,
                    "post_payout_smoother_active": post_payout_smoother_active,
                    "buffer_smoother_active": buffer_smoother_active,
                })

                self.log_event(ts, "PAYOUT_APPROVED", cycle_id, details=f"${payout_amount:.2f}")
                self.log_event(ts, "POST_PAYOUT_LOCK", cycle_id)

                last_payout_ts = ts
                payout_cycle_number += 1

                # Reset only the actual payout cycle state after payout.
                payout_cycle_start_ts = ts
                payout_cycle_start_balance = pa_balance
                payout_cycle_trade_count = 0
                payout_cycle_active_days = set()
                payout_cycle_day_pnls = {}

                consistency_repair_active = False
                consistency_repair_trades = 0
                consistency_repair_days = set()

                post_payout_smoother_active = self.enable_pa_volatility_smoother

                pa_day_locked = True
                pa_day_lock_reason = "post_payout_lock"

            pa_idx += 1

        return {
            "next_eval_idx": len(eval_trades),
            "cycle": {
                "cycle_id": cycle_id,
                "result": "data_end",
                "eval_passed": True,
                "pa_blown": False,
                "payout_count": payout_count,
                "payout_total": payout_total,
                "final_balance": pa_balance,
                "net_pnl": pa_balance - pa_start_balance,
                "eval_start": eval_start_ts,
                "eval_end": eval_pass_ts,
                "eval_calendar_days": (pd.Timestamp(eval_pass_ts).date() - pd.Timestamp(eval_start_ts).date()).days,
                "eval_active_days": len(eval_active_days),
                "eval_trades": eval_trade_count,
                "pa_start": pa_start_ts,
                "pa_end": pa_end_ts,
                "next_eval_start_idx": len(eval_trades),
                "pa_calendar_days": (pd.Timestamp(pa_end_ts).date() - pd.Timestamp(pa_start_ts).date()).days,
                "pa_active_days": len(pa_active_days),
                "pa_trades": pa_trade_count,
                "consistency_repair_activations": consistency_repair_activations,
                "pa_volatility_smoother_enabled": self.enable_pa_volatility_smoother,
                "smoother_trade_count": smoother_trade_count,
                "smoother_events": smoother_events,
            }
        }

    def build_summary(self):
        cycles_df = pd.DataFrame(self.cycles)
        payout_cycles_df = pd.DataFrame(self.payout_cycle_rows)

        eval_passed = cycles_df["eval_passed"].sum() if len(cycles_df) else 0
        pa_blown = cycles_df["pa_blown"].sum() if len(cycles_df) else 0
        payout_total = cycles_df["payout_total"].sum() if len(cycles_df) else 0

        actual_payout_cycles = (
            payout_cycles_df[payout_cycles_df["row_type"] == "actual_payout_cycle"]
            if len(payout_cycles_df) and "row_type" in payout_cycles_df.columns
            else pd.DataFrame()
        )

        payout_cycle_hit_rate = (
            float(payout_cycles_df["target_met"].mean())
            if len(payout_cycles_df) else 0.0
        )
        avg_days_per_actual_payout = (
            float(actual_payout_cycles["calendar_days"].mean())
            if len(actual_payout_cycles) else 0.0
        )

        return {
            "eval_attempts": len(cycles_df),
            "eval_passed": int(eval_passed),
            "eval_pass_rate": float(eval_passed) / len(cycles_df) if len(cycles_df) else 0.0,
            "pa_blown": int(pa_blown),
            "total_payouts": payout_total,
            "avg_payouts_per_account": cycles_df["payout_count"].mean() if len(cycles_df) else 0,
            "avg_pa_net_pnl": cycles_df["net_pnl"].mean() if len(cycles_df) else 0,
            "avg_eval_calendar_days": cycles_df["eval_calendar_days"].mean() if "eval_calendar_days" in cycles_df.columns and len(cycles_df) else 0,
            "avg_eval_active_days": cycles_df["eval_active_days"].mean() if "eval_active_days" in cycles_df.columns and len(cycles_df) else 0,
            "avg_eval_trades": cycles_df["eval_trades"].mean() if "eval_trades" in cycles_df.columns and len(cycles_df) else 0,
            "avg_pa_calendar_days": cycles_df["pa_calendar_days"].mean() if "pa_calendar_days" in cycles_df.columns and len(cycles_df) else 0,
            "avg_pa_active_days": cycles_df["pa_active_days"].mean() if "pa_active_days" in cycles_df.columns and len(cycles_df) else 0,
            "avg_pa_trades": cycles_df["pa_trades"].mean() if "pa_trades" in cycles_df.columns and len(cycles_df) else 0,
            "payout_cycle_hit_rate": payout_cycle_hit_rate,
            "avg_days_per_actual_payout": avg_days_per_actual_payout,
            "pa_min_payout_trading_days": self.pa_min_payout_trading_days,
            "pa_consistency_rule_pct": self.pa_consistency_rule_pct,
            "pa_min_payout_amount": self.pa_min_payout_amount,
            "pa_max_payout_amount": self.pa_max_payout_amount,
            "first_payout_buffer_mode": self.enable_first_payout_buffer_mode,
            "first_payout_build_balance": self.first_payout_build_balance,
            "first_payout_amount": self.first_payout_amount,
            "retained_cushion_after_payout": self.retained_cushion_after_payout,
            "pa_payout_buffer_override": (
                self.pa_payout_buffer_override
                if self.pa_payout_buffer_override is not None
                else "profile_default"
            ),
            "pa_acceleration_daily_profit_target": self.pa_acceleration_daily_profit_target,
            "consistency_repair_enabled": self.enable_consistency_repair,
            "consistency_repair_risk_multiplier": self.consistency_repair_risk_multiplier,
            "consistency_repair_daily_profit_target": self.consistency_repair_daily_profit_target,
            "total_consistency_repair_activations": (
                cycles_df["consistency_repair_activations"].sum()
                if "consistency_repair_activations" in cycles_df.columns and len(cycles_df) else 0
            ),
            "pa_volatility_smoother_enabled": self.enable_pa_volatility_smoother,
            "total_smoother_trades": (
                cycles_df["smoother_trade_count"].sum()
                if "smoother_trade_count" in cycles_df.columns and len(cycles_df) else 0
            ),
            "total_smoother_events": (
                cycles_df["smoother_events"].sum()
                if "smoother_events" in cycles_df.columns and len(cycles_df) else 0
            ),
            "smoother_after_profit_day_threshold": self.smoother_after_profit_day_threshold,
            "smoother_after_profit_risk_multiplier": self.smoother_after_profit_risk_multiplier,
            "smoother_loss_streak_threshold": self.smoother_loss_streak_threshold,
            "smoother_after_loss_risk_multiplier": self.smoother_after_loss_risk_multiplier,
            "smoother_post_payout_risk_multiplier": self.smoother_post_payout_risk_multiplier,
            "smoother_buffer_floor": self.smoother_buffer_floor,
            "smoother_buffer_risk_multiplier": self.smoother_buffer_risk_multiplier,
        }


# ============================================================
# OUTPUTS
# ============================================================

def write_outputs(sim, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(sim.cycles).to_csv(output_dir / "dual_mode_cycles.csv", index=False)
    pd.DataFrame(sim.events).to_csv(output_dir / "dual_mode_events.csv", index=False)
    pd.DataFrame(sim.daily_rows).to_csv(output_dir / "dual_mode_daily.csv", index=False)
    pd.DataFrame(sim.payout_rows).to_csv(output_dir / "dual_mode_payouts.csv", index=False)
    pd.DataFrame(sim.payout_cycle_rows).to_csv(output_dir / "dual_mode_payout_cycles.csv", index=False)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--trade-log", required=False, default="")
    parser.add_argument("--eval-trade-log", default="")
    parser.add_argument("--pa-trade-log", default="")

    parser.add_argument("--profile", required=True)

    parser.add_argument(
        "--profiles-file",
        default="src/strategies/manual/researched_prop_trend/dual_mode_profiles.yaml",
    )

    parser.add_argument(
        "--output-dir",
        default="src/strategies/manual/researched_prop_trend/dual_mode_outputs",
    )

    parser.add_argument("--eval-risk-multiplier", type=float, default=1.0)
    parser.add_argument("--pa-risk-multiplier", type=float, default=1.0)
    parser.add_argument("--target-payout-frequency-days", type=int, default=10)
    parser.add_argument("--min-weekly-payout-amount", type=float, default=1000.0)
    parser.add_argument("--pa-min-payout-trading-days", type=int, default=5)
    parser.add_argument("--pa-consistency-rule-pct", type=float, default=50.0)
    parser.add_argument("--pa-min-payout-amount", type=float, default=250.0)
    parser.add_argument("--pa-max-payout-amount", type=float, default=2000.0)
    parser.add_argument("--enable-first-payout-buffer-mode", action="store_true")
    parser.add_argument("--first-payout-build-balance", type=float, default=53000.0)
    parser.add_argument("--first-payout-amount", type=float, default=1000.0)
    parser.add_argument("--retained-cushion-after-payout", type=float, default=2000.0)
    parser.add_argument(
        "--pa-payout-buffer-override",
        type=float,
        default=-1.0,
        help="Override PA retained payout buffer. Use -1 to keep profile default.",
    )
    parser.add_argument(
        "--pa-acceleration-daily-profit-target",
        type=float,
        default=0.0,
        help="Optional PA daily profit lock for faster/smoother payout cycles. 0 disables.",
    )
    parser.add_argument("--disable-pa-volatility-smoother", action="store_true")
    parser.add_argument("--smoother-after-profit-day-threshold", type=float, default=750.0)
    parser.add_argument("--smoother-after-profit-risk-multiplier", type=float, default=0.75)
    parser.add_argument("--smoother-loss-streak-threshold", type=int, default=2)
    parser.add_argument("--smoother-after-loss-risk-multiplier", type=float, default=0.50)
    parser.add_argument("--smoother-post-payout-risk-multiplier", type=float, default=0.50)
    parser.add_argument("--smoother-buffer-floor", type=float, default=1000.0)
    parser.add_argument("--smoother-buffer-risk-multiplier", type=float, default=0.50)

    parser.add_argument("--disable-consistency-repair", action="store_true")
    parser.add_argument("--consistency-repair-risk-multiplier", type=float, default=0.50)
    parser.add_argument("--consistency-repair-daily-profit-target", type=float, default=250.0)
    parser.add_argument("--aggressive-eval-until-pass", action="store_true")

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    profiles = load_profiles(args.profiles_file)

    if args.profile not in profiles:
        raise ValueError(f"Unknown profile: {args.profile}")

    profile = profiles[args.profile]

    eval_trade_log = args.eval_trade_log or args.trade_log
    pa_trade_log = args.pa_trade_log or args.trade_log

    if not eval_trade_log:
        raise ValueError("Provide --trade-log or --eval-trade-log")

    if not pa_trade_log:
        raise ValueError("Provide --trade-log or --pa-trade-log")

    df = load_trade_log(eval_trade_log)
    pa_df = load_trade_log(pa_trade_log)

    sim = DualModeLifecycleSimulator(
        df=df,
        pa_df=pa_df,
        profile=profile,
        eval_risk_multiplier=args.eval_risk_multiplier,
        pa_risk_multiplier=args.pa_risk_multiplier,
        target_payout_frequency_days=args.target_payout_frequency_days,
        min_weekly_payout_amount=args.min_weekly_payout_amount,
        pa_min_payout_trading_days=args.pa_min_payout_trading_days,
        pa_consistency_rule_pct=args.pa_consistency_rule_pct,
        pa_min_payout_amount=args.pa_min_payout_amount,
        pa_max_payout_amount=args.pa_max_payout_amount,
        enable_first_payout_buffer_mode=args.enable_first_payout_buffer_mode,
        first_payout_build_balance=args.first_payout_build_balance,
        first_payout_amount=args.first_payout_amount,
        retained_cushion_after_payout=args.retained_cushion_after_payout,
        pa_payout_buffer_override=args.pa_payout_buffer_override,
        pa_acceleration_daily_profit_target=args.pa_acceleration_daily_profit_target,
        enable_pa_volatility_smoother=not args.disable_pa_volatility_smoother,
        smoother_after_profit_day_threshold=args.smoother_after_profit_day_threshold,
        smoother_after_profit_risk_multiplier=args.smoother_after_profit_risk_multiplier,
        smoother_loss_streak_threshold=args.smoother_loss_streak_threshold,
        smoother_after_loss_risk_multiplier=args.smoother_after_loss_risk_multiplier,
        smoother_post_payout_risk_multiplier=args.smoother_post_payout_risk_multiplier,
        smoother_buffer_floor=args.smoother_buffer_floor,
        smoother_buffer_risk_multiplier=args.smoother_buffer_risk_multiplier,
        enable_consistency_repair=not args.disable_consistency_repair,
        consistency_repair_risk_multiplier=args.consistency_repair_risk_multiplier,
        consistency_repair_daily_profit_target=args.consistency_repair_daily_profit_target,
        aggressive_eval_until_pass=args.aggressive_eval_until_pass,
    )

    summary = sim.run()

    output_dir = Path(args.output_dir)
    write_outputs(sim, output_dir)

    print("\n================ DUAL MODE SUMMARY ================")

    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    print("\nWrote files:")
    print(output_dir / "dual_mode_cycles.csv")
    print(output_dir / "dual_mode_events.csv")
    print(output_dir / "dual_mode_daily.csv")
    print(output_dir / "dual_mode_payouts.csv")
    print(output_dir / "dual_mode_payout_cycles.csv")


if __name__ == "__main__":
    main()
