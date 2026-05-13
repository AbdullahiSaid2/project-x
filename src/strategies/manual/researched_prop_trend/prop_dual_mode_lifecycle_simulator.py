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

    df = pd.read_csv(path)

    timestamp_candidates = [
    "exit_time",
    "entry_time",

    # NEW EVENT ENGINE
    "exit_time_et",
    "entry_time_et",

    # OTHER FORMATS
    "closed_at",
    "opened_at",
    "exit_timestamp",
    "entry_timestamp",
    "timestamp",
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

        raise ValueError(
            "No timestamp column found"
        )

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

        raise ValueError(
            "No pnl column found"
        )

    # =====================================================
    # NORMALIZE
    # =====================================================

    df["timestamp"] = pd.to_datetime(
        df[timestamp_col]
    )

    df["pnl"] = pd.to_numeric(
        df[pnl_col],
        errors="coerce",
    ).fillna(0)

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    df["trade_date"] = (
        df["timestamp"].dt.date
    )

    return df


# ============================================================
# MAIN SIMULATOR
# ============================================================

class DualModeLifecycleSimulator:

    def __init__(self, df: pd.DataFrame, profile: DualModeProfile):

        self.df = df
        self.profile = profile

        self.cycles = []
        self.events = []
        self.daily_rows = []
        self.payout_rows = []

    # ========================================================

    def log_event(self, ts, event_type, cycle_id, details=None):

        self.events.append({
            "timestamp": ts,
            "event_type": event_type,
            "cycle_id": cycle_id,
            "details": details,
        })

    # ========================================================

    def run(self):

        trades = self.df.to_dict("records")

        idx = 0
        cycle_id = 1

        while idx < len(trades):

            result = self.run_single_cycle(
                trades=trades,
                start_idx=idx,
                cycle_id=cycle_id,
            )

            self.cycles.append(result["cycle"])

            idx = result["next_idx"]

            cycle_id += 1

        return self.build_summary()

    # ========================================================

    def run_single_cycle(
        self,
        trades,
        start_idx,
        cycle_id,
    ):

        mode = "eval"

        eval_cfg = self.profile.eval
        pa_cfg = self.profile.pa

        balance = eval_cfg.account_size

        high_watermark = balance

        current_idx = start_idx

        current_day = None

        daily_pnl = 0.0
        daily_trade_count = 0
        consecutive_losses = 0
        trading_locked = False

        payout_count = 0
        payout_total = 0.0

        payout_lock_days_remaining = 0

        pa_start_balance = None

        self.log_event(
            trades[start_idx]["timestamp"],
            "EVAL_STARTED",
            cycle_id,
        )

        while current_idx < len(trades):

            trade = trades[current_idx]

            ts = trade["timestamp"]
            pnl = float(trade["pnl"])
            trade_day = trade["trade_date"]

            # =================================================
            # NEW DAY RESET
            # =================================================

            if current_day != trade_day:

                current_day = trade_day

                daily_pnl = 0.0
                daily_trade_count = 0
                consecutive_losses = 0
                trading_locked = False

                if payout_lock_days_remaining > 0:
                    payout_lock_days_remaining -= 1

            # =================================================
            # CONFIG
            # =================================================

            if mode == "eval":
                cfg = eval_cfg
            else:
                cfg = pa_cfg

            # =================================================
            # DAY LOCKS
            # =================================================

            if trading_locked:
                current_idx += 1
                continue

            if daily_trade_count >= cfg.max_trades_per_day:

                trading_locked = True

                self.log_event(
                    ts,
                    "MAX_TRADES_DAY_LOCK",
                    cycle_id,
                )

                current_idx += 1
                continue

            if daily_pnl <= -cfg.daily_soft_loss_stop:

                trading_locked = True

                self.log_event(
                    ts,
                    "DAILY_SOFT_LOSS_LOCK",
                    cycle_id,
                )

                current_idx += 1
                continue

            # =================================================
            # POST-PAYOUT RISK REDUCTION
            # =================================================

            effective_risk = cfg.risk_per_trade

            if mode == "pa":

                if (
                    payout_lock_days_remaining > 0
                    and pa_cfg.reduce_risk_after_payout
                ):
                    effective_risk = (
                        pa_cfg.reduced_risk_per_trade
                    )

                retained_profit = balance - pa_start_balance

                if retained_profit >= 2000:

                    effective_risk = min(
                        effective_risk,
                        pa_cfg.reduced_risk_per_trade
                    )

            # =================================================
            # APPLY TRADE
            # =================================================

            scale_factor = effective_risk / 150.0

            scaled_pnl = pnl * scale_factor

            balance += scaled_pnl

            daily_pnl += scaled_pnl

            daily_trade_count += 1

            # =================================================
            # CONSECUTIVE LOSSES
            # =================================================

            if scaled_pnl < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0

            if (
                consecutive_losses
                >= cfg.pause_after_consecutive_losses
            ):

                trading_locked = True

                self.log_event(
                    ts,
                    "CONSECUTIVE_LOSS_LOCK",
                    cycle_id,
                )

            # =================================================
            # DAILY PROFIT LOCK
            # =================================================

            if daily_pnl >= cfg.daily_profit_target:

                trading_locked = True

                self.log_event(
                    ts,
                    "DAILY_PROFIT_TARGET_LOCK",
                    cycle_id,
                )

            # =================================================
            # DRAWDOWN
            # =================================================

            high_watermark = max(
                high_watermark,
                balance,
            )

            drawdown = balance - high_watermark

            # =================================================
            # EVAL MODE
            # =================================================

            if mode == "eval":

                if drawdown <= -eval_cfg.max_drawdown:

                    self.log_event(
                        ts,
                        "EVAL_FAILED_MAX_LOSS",
                        cycle_id,
                    )

                    return {
                        "next_idx": current_idx + 1,
                        "cycle": {
                            "cycle_id": cycle_id,
                            "result": "eval_failed",
                            "eval_passed": False,
                            "pa_blown": False,
                            "payout_count": 0,
                            "payout_total": 0.0,
                            "final_balance": balance,
                            "net_pnl": (
                                balance
                                - eval_cfg.account_size
                            ),
                        }
                    }

                if balance >= (
                    eval_cfg.account_size
                    + eval_cfg.profit_target
                ):

                    mode = "pa"

                    pa_start_balance = balance

                    high_watermark = balance

                    self.log_event(
                        ts,
                        "EVAL_PASSED",
                        cycle_id,
                    )

                    self.log_event(
                        ts,
                        "PA_STARTED",
                        cycle_id,
                    )

            # =================================================
            # PA MODE
            # =================================================

            else:

                if drawdown <= -pa_cfg.max_drawdown:

                    self.log_event(
                        ts,
                        "PA_BLOWN_MAX_LOSS",
                        cycle_id,
                    )

                    return {
                        "next_idx": current_idx + 1,
                        "cycle": {
                            "cycle_id": cycle_id,
                            "result": "pa_blown",
                            "eval_passed": True,
                            "pa_blown": True,
                            "payout_count": payout_count,
                            "payout_total": payout_total,
                            "final_balance": balance,
                            "net_pnl": (
                                balance
                                - pa_start_balance
                            ),
                        }
                    }

                # =============================================
                # PAYOUT PROTECTION
                # =============================================

                retained_profit = (
                    balance - pa_start_balance
                )

                if retained_profit >= 2500:

                    trading_locked = True

                    self.log_event(
                        ts,
                        "PA_PROFIT_LOCK",
                        cycle_id,
                    )

                # =============================================
                # PAYOUT LOGIC
                # =============================================

                if retained_profit >= (
                    pa_cfg.payout_buffer + 2000
                ):

                    payout_amount = (
                        retained_profit
                        - pa_cfg.payout_buffer
                    )

                    payout_total += payout_amount

                    payout_count += 1

                    balance -= payout_amount

                    payout_lock_days_remaining = (
                        pa_cfg.payout_lock_days
                    )

                    self.payout_rows.append({
                        "cycle_id": cycle_id,
                        "timestamp": ts,
                        "payout_number": payout_count,
                        "payout_amount": payout_amount,
                    })

                    self.log_event(
                        ts,
                        "PAYOUT_APPROVED",
                        cycle_id,
                        details=f"${payout_amount:.2f}",
                    )

                    # =========================================
                    # POST PAYOUT ACCOUNT LOCK
                    # =========================================

                    trading_locked = True

                    self.log_event(
                        ts,
                        "POST_PAYOUT_LOCK",
                        cycle_id,
                    )

            # =================================================
            # DAILY ROWS
            # =================================================

            self.daily_rows.append({
                "cycle_id": cycle_id,
                "date": trade_day,
                "mode": mode,
                "balance": balance,
                "daily_pnl": daily_pnl,
            })

            current_idx += 1

        # =====================================================
        # END OF DATA
        # =====================================================

        return {
            "next_idx": current_idx,
            "cycle": {
                "cycle_id": cycle_id,
                "result": "data_end",
                "eval_passed": (
                    mode == "pa"
                ),
                "pa_blown": False,
                "payout_count": payout_count,
                "payout_total": payout_total,
                "final_balance": balance,
                "net_pnl": (
                    balance - eval_cfg.account_size
                ),
            }
        }

    # ========================================================

    def build_summary(self):

        cycles_df = pd.DataFrame(self.cycles)

        eval_passed = (
            cycles_df["eval_passed"].sum()
            if len(cycles_df) > 0 else 0
        )

        pa_blown = (
            cycles_df["pa_blown"].sum()
            if len(cycles_df) > 0 else 0
        )

        payout_total = (
            cycles_df["payout_total"].sum()
            if len(cycles_df) > 0 else 0
        )

        return {
            "eval_attempts": len(cycles_df),
            "eval_passed": int(eval_passed),
            "eval_pass_rate": (
                float(eval_passed) / len(cycles_df)
                if len(cycles_df) > 0 else 0
            ),
            "pa_blown": int(pa_blown),
            "total_payouts": payout_total,
            "avg_payouts_per_account": (
                cycles_df["payout_count"].mean()
                if len(cycles_df) > 0 else 0
            ),
            "avg_pa_net_pnl": (
                cycles_df["net_pnl"].mean()
                if len(cycles_df) > 0 else 0
            ),
        }


# ============================================================
# OUTPUTS
# ============================================================

def write_outputs(sim, output_dir):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(sim.cycles).to_csv(
        output_dir / "dual_mode_cycles.csv",
        index=False,
    )

    pd.DataFrame(sim.events).to_csv(
        output_dir / "dual_mode_events.csv",
        index=False,
    )

    pd.DataFrame(sim.daily_rows).to_csv(
        output_dir / "dual_mode_daily.csv",
        index=False,
    )

    pd.DataFrame(sim.payout_rows).to_csv(
        output_dir / "dual_mode_payouts.csv",
        index=False,
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trade-log",
        required=True,
    )

    parser.add_argument(
        "--profile",
        required=True,
    )

    parser.add_argument(
        "--profiles-file",
        default=(
            "src/strategies/manual/"
            "researched_prop_trend/"
            "dual_mode_profiles.yaml"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "src/strategies/manual/"
            "researched_prop_trend/"
            "dual_mode_outputs"
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    profiles = load_profiles(
        args.profiles_file
    )

    if args.profile not in profiles:
        raise ValueError(
            f"Unknown profile: {args.profile}"
        )

    profile = profiles[args.profile]

    df = load_trade_log(
        args.trade_log
    )

    sim = DualModeLifecycleSimulator(
        df=df,
        profile=profile,
    )

    summary = sim.run()

    output_dir = Path(
        args.output_dir
    )

    write_outputs(
        sim,
        output_dir,
    )

    print(
        "\n================ "
        "DUAL MODE SUMMARY "
        "================"
    )

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


if __name__ == "__main__":
    main()