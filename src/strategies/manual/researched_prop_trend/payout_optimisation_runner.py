
from __future__ import annotations

"""
Payout Optimisation Runner.

Tests many strategy / symbol / target / risk / PA-management combinations
and ranks them by payout lifecycle performance.

Supported strategies:
  - ict_fractal
  - top_bottom_ticking
  - researched_prop_trend
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import itertools
import json
import shutil
import subprocess
import sys
from typing import Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "payout_optimisation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class OptimisationConfig:
    strategy: str
    symbols: str
    days_back: int
    target_r: float
    risk_per_trade: float
    daily_profit_target: float
    daily_soft_loss_stop: float
    max_trades_per_day: int
    pause_after_consecutive_losses: int
    lifecycle_profile: str
    min_planned_target_dollars: float
    min_trend_score: Optional[float] = None


def _run(cmd: list[str], cwd: Path, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)


def _slug(s: str) -> str:
    return s.replace(" ", "-").replace(",", "_").replace("/", "-").replace(":", "-").replace(".", "p")


def _parse_number(value):
    if value is None:
        return None
    raw = str(value).replace("$", "").replace(",", "").replace("%", "").strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except Exception:
        return None


def parse_lifecycle_summary(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = (
            k.strip()
            .lower()
            .replace("/", "_")
            .replace(" ", "_")
            .replace("+", "plus")
            .replace("-", "_")
        )
        out[key] = v.strip()
    return out


def parse_event_trade_log(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {
            "event_trades": 0,
            "event_net_pnl": 0.0,
            "event_gross_pnl": 0.0,
            "event_commissions": 0.0,
            "event_win_rate_pct": 0.0,
            "event_profit_factor": 0.0,
            "event_worst_trade": 0.0,
            "event_best_trade": 0.0,
        }

    pnl = df["net_pnl_dollars"] if "net_pnl_dollars" in df.columns else df["gross_pnl_dollars"]
    gross = df["gross_pnl_dollars"].sum() if "gross_pnl_dollars" in df.columns else pnl.sum()
    commissions = df["commissions_dollars"].sum() if "commissions_dollars" in df.columns else 0.0
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    pf = float(wins / abs(losses)) if losses < 0 else float("inf")

    return {
        "event_trades": int(len(df)),
        "event_net_pnl": float(pnl.sum()),
        "event_gross_pnl": float(gross),
        "event_commissions": float(commissions),
        "event_win_rate_pct": float((pnl > 0).mean() * 100),
        "event_profit_factor": pf,
        "event_worst_trade": float(pnl.min()),
        "event_best_trade": float(pnl.max()),
    }


def parse_interval_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {
            "payout_accounts": 0,
            "avg_days_to_first_payout": None,
            "median_days_to_first_payout": None,
            "avg_days_between_payouts": None,
            "median_days_between_payouts": None,
            "max_days_between_payouts": None,
        }

    return {
        "payout_accounts": int(len(df)),
        "avg_days_to_first_payout": float(df["first_payout_days"].mean()) if "first_payout_days" in df else None,
        "median_days_to_first_payout": float(df["first_payout_days"].median()) if "first_payout_days" in df else None,
        "avg_days_between_payouts": float(df["avg_days_between_payouts"].mean()) if "avg_days_between_payouts" in df else None,
        "median_days_between_payouts": float(df["median_days_between_payouts"].median()) if "median_days_between_payouts" in df else None,
        "max_days_between_payouts": float(df["max_days_between_payouts"].max()) if "max_days_between_payouts" in df else None,
    }


def build_event_command(cfg: OptimisationConfig, output_prefix: str, news_events: str, timeframe: str, commission: float, disable_account_drawdown_lock: bool) -> tuple[list[str], str]:
    symbols = cfg.symbols.split(",")

    if cfg.strategy in {"ict_fractal", "top_bottom_ticking"}:
        cmd = [
            sys.executable, "-m", "src.backtesting.event_engine.run_event_backtest",
            "--strategy", cfg.strategy,
            "--symbols", *symbols,
            "--prop-profile", "apex_50k_pa",
            "--days-back", str(cfg.days_back),
            "--timeframe", timeframe,
            "--no-tail",
            "--commission-per-contract-side", str(commission),
            "--target-r", str(cfg.target_r),
            "--min-planned-target-dollars", str(cfg.min_planned_target_dollars),
            "--risk-per-trade", str(cfg.risk_per_trade),
            "--daily-profit-target", str(cfg.daily_profit_target),
            "--daily-soft-loss-stop", str(cfg.daily_soft_loss_stop),
            "--max-trades-per-day", str(cfg.max_trades_per_day),
            "--pause-after-consecutive-losses", str(cfg.pause_after_consecutive_losses),
            "--news-events", news_events,
            "--output-prefix", output_prefix,
        ]
        if disable_account_drawdown_lock:
            cmd.append("--disable-account-drawdown-lock")
        return cmd, f"src/backtesting/event_engine/outputs/{output_prefix}_event_trade_log.csv"

    if cfg.strategy == "researched_prop_trend":
        cmd = [
            sys.executable, "-m", "src.strategies.manual.researched_prop_trend.prop_event_engine_backtest",
            "--symbols", *symbols,
            "--prop-profile", "apex_50k_pa",
            "--days-back", str(cfg.days_back),
            "--timeframe", timeframe,
            "--no-tail",
            "--commission-per-contract-side", str(commission),
            "--target-r", str(cfg.target_r),
            "--min-planned-target-dollars", str(cfg.min_planned_target_dollars),
            "--risk-per-trade", str(cfg.risk_per_trade),
            "--news-events", news_events,
        ]
        if cfg.min_trend_score is not None:
            cmd.extend(["--min-trend-score", str(cfg.min_trend_score)])
        return cmd, "src/strategies/manual/researched_prop_trend/event_trade_log.csv"

    raise ValueError(f"Unsupported strategy: {cfg.strategy}")


def run_lifecycle(trade_log: str, lifecycle_profile: str) -> subprocess.CompletedProcess:
    return _run(
        [
            sys.executable, "-m", "src.strategies.manual.researched_prop_trend.prop_lifecycle_payout_simulator",
            "--trade-log", trade_log,
            "--lifecycle-profile", lifecycle_profile,
        ],
        cwd=ROOT,
    )


def run_interval_report(run_dir: Path) -> subprocess.CompletedProcess:
    payouts = THIS_DIR / "prop_lifecycle_payouts.csv"
    cycles = THIS_DIR / "prop_lifecycle_payout_cycles.csv"
    out_detail = run_dir / "payout_interval_report.csv"
    out_summary = run_dir / "payout_interval_summary.csv"

    if not payouts.exists() or not cycles.exists():
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="Missing payouts/cycles")

    return _run(
        [
            sys.executable, "-m", "src.strategies.manual.researched_prop_trend.payout_interval_report",
            "--payouts", str(payouts.relative_to(ROOT)),
            "--cycles", str(cycles.relative_to(ROOT)),
            "--out-detail", str(out_detail.relative_to(ROOT)),
            "--out-summary", str(out_summary.relative_to(ROOT)),
        ],
        cwd=ROOT,
    )


def copy_lifecycle_outputs(run_dir: Path):
    for name in [
        "prop_lifecycle_payout_summary.txt",
        "prop_lifecycle_payout_cycles.csv",
        "prop_lifecycle_payout_events.csv",
        "prop_lifecycle_payout_daily.csv",
        "prop_lifecycle_payouts.csv",
    ]:
        src = THIS_DIR / name
        if src.exists():
            shutil.copy2(src, run_dir / name)


def compute_score(row: dict) -> float:
    payout_total = float(row.get("approved_payout_total") or 0)
    net_plus = float(row.get("total_pa_net_plus_payouts") or 0)
    pa_blown = float(row.get("pa_blown") or 0)
    pa_completed = float(row.get("pa_completed_max_payouts") or 0)

    days_first = row.get("median_days_to_first_payout")
    gap = row.get("median_days_between_payouts")

    days_penalty = float(days_first) * 10 if days_first is not None and pd.notna(days_first) else 0
    gap_penalty = float(gap) * 5 if gap is not None and pd.notna(gap) else 0

    return payout_total + (0.50 * net_plus) + (2500 * pa_completed) - (750 * pa_blown) - days_penalty - gap_penalty


def configs_from_grid(args) -> list[OptimisationConfig]:
    configs = []
    for combo in itertools.product(
        args.strategies,
        args.symbol_sets,
        args.target_r,
        args.risk_per_trade,
        args.daily_profit_target,
        args.daily_soft_loss_stop,
        args.max_trades_per_day,
        args.pause_after_consecutive_losses,
        args.lifecycle_profiles,
    ):
        strategy, symbols, target_r, risk, dpt, soft, max_trades, pause_losses, lifecycle = combo
        configs.append(
            OptimisationConfig(
                strategy=strategy,
                symbols=symbols,
                days_back=args.days_back,
                target_r=float(target_r),
                risk_per_trade=float(risk),
                daily_profit_target=float(dpt),
                daily_soft_loss_stop=float(soft),
                max_trades_per_day=int(max_trades),
                pause_after_consecutive_losses=int(pause_losses),
                lifecycle_profile=lifecycle,
                min_planned_target_dollars=float(args.min_planned_target_dollars),
                min_trend_score=float(args.min_trend_score) if args.min_trend_score is not None else None,
            )
        )
    return configs


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--strategies", nargs="+", default=["ict_fractal"])
    p.add_argument("--symbol-sets", nargs="+", default=["MNQ,MYM", "MNQ,MES,MYM"])
    p.add_argument("--target-r", nargs="+", type=float, default=[3.0, 4.0, 5.0])
    p.add_argument("--risk-per-trade", nargs="+", type=float, default=[75, 100, 125, 150])
    p.add_argument("--daily-profit-target", nargs="+", type=float, default=[350, 500, 650])
    p.add_argument("--daily-soft-loss-stop", nargs="+", type=float, default=[250, 300, 350])
    p.add_argument("--max-trades-per-day", nargs="+", type=int, default=[3, 5, 6])
    p.add_argument("--pause-after-consecutive-losses", nargs="+", type=int, default=[1, 2])
    p.add_argument("--lifecycle-profiles", nargs="+", default=["apex_50k_eod_lifecycle_safe", "apex_50k_eod_lifecycle_balanced"])
    p.add_argument("--days-back", type=int, default=365)
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--commission-per-contract-side", type=float, default=2.0)
    p.add_argument("--min-planned-target-dollars", type=float, default=250)
    p.add_argument("--min-trend-score", type=float, default=3)
    p.add_argument("--news-events", default="src/strategies/manual/researched_prop_trend/news_events.csv")
    p.add_argument("--max-runs", type=int, default=20)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--disable-account-drawdown-lock", action="store_true")
    p.add_argument("--skip-event-backtest", action="store_true")
    p.add_argument("--existing-trade-log", default="")
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args()

    configs = configs_from_grid(args)
    configs = configs[args.start_index : args.start_index + args.max_runs]

    print("================ PAYOUT OPTIMISATION RUNNER ================")
    print(f"Total selected configs: {len(configs)}")
    print(f"Output dir: {OUT_DIR}")

    rows = []

    for run_number, cfg in enumerate(configs, start=args.start_index):
        run_id = (
            f"{run_number:04d}__{cfg.strategy}__{_slug(cfg.symbols)}__"
            f"r{cfg.target_r}__risk{cfg.risk_per_trade}__"
            f"dpt{cfg.daily_profit_target}__soft{cfg.daily_soft_loss_stop}__"
            f"mt{cfg.max_trades_per_day}__loss{cfg.pause_after_consecutive_losses}__"
            f"{_slug(cfg.lifecycle_profile)}"
        )
        run_dir = OUT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== RUN {run_number} ===")
        print(json.dumps(asdict(cfg), indent=2))

        output_prefix = f"opt_{run_number:04d}_{cfg.strategy}"
        trade_log = args.existing_trade_log

        if not args.skip_event_backtest:
            event_cmd, trade_log = build_event_command(
                cfg,
                output_prefix=output_prefix,
                news_events=args.news_events,
                timeframe=args.timeframe,
                commission=args.commission_per_contract_side,
                disable_account_drawdown_lock=args.disable_account_drawdown_lock,
            )
            (run_dir / "event_command.txt").write_text(" ".join(event_cmd) + "\n")
            event_result = _run(event_cmd, cwd=ROOT)
            (run_dir / "event_stdout.txt").write_text(event_result.stdout)
            (run_dir / "event_stderr.txt").write_text(event_result.stderr)

            print(event_result.stdout[-2000:])

            if event_result.returncode != 0:
                print(event_result.stderr[-2000:])
                rows.append(asdict(cfg) | {
                    "run_id": run_id,
                    "status": "event_failed",
                    "event_returncode": event_result.returncode,
                    "event_error_tail": event_result.stderr[-500:],
                })
                if args.fail_fast:
                    break
                continue

            src_trade = ROOT / trade_log
            if src_trade.exists():
                shutil.copy2(src_trade, run_dir / "event_trade_log.csv")
        else:
            if not trade_log:
                raise ValueError("--skip-event-backtest requires --existing-trade-log")

        event_metrics = parse_event_trade_log(ROOT / trade_log)

        lifecycle_result = run_lifecycle(trade_log, cfg.lifecycle_profile)
        (run_dir / "lifecycle_stdout.txt").write_text(lifecycle_result.stdout)
        (run_dir / "lifecycle_stderr.txt").write_text(lifecycle_result.stderr)
        print(lifecycle_result.stdout[-2000:])

        if lifecycle_result.returncode != 0:
            print(lifecycle_result.stderr[-2000:])
            rows.append(asdict(cfg) | {
                "run_id": run_id,
                "status": "lifecycle_failed",
                "lifecycle_returncode": lifecycle_result.returncode,
                "lifecycle_error_tail": lifecycle_result.stderr[-500:],
            } | event_metrics)
            if args.fail_fast:
                break
            continue

        copy_lifecycle_outputs(run_dir)
        interval_result = run_interval_report(run_dir)
        (run_dir / "interval_stdout.txt").write_text(interval_result.stdout)
        (run_dir / "interval_stderr.txt").write_text(interval_result.stderr)

        summary = parse_lifecycle_summary(run_dir / "prop_lifecycle_payout_summary.txt")
        interval_metrics = parse_interval_summary(run_dir / "payout_interval_summary.csv")

        row = asdict(cfg) | {
            "run_id": run_id,
            "status": "ok",
            "eval_attempts": _parse_number(summary.get("eval_attempts")),
            "eval_passed": _parse_number(summary.get("eval_passed")),
            "eval_pass_rate_pct": _parse_number(summary.get("eval_pass_rate")),
            "pa_accounts_started": _parse_number(summary.get("pa_accounts_started")),
            "pa_blown": _parse_number(summary.get("pa_blown")),
            "pa_completed_max_payouts": _parse_number(summary.get("pa_completed_max_payouts")),
            "pa_active_data_ended": _parse_number(summary.get("pa_active_data_ended")),
            "approved_payout_count": _parse_number(summary.get("approved_payout_count")),
            "approved_payout_total": _parse_number(summary.get("approved_payout_total")),
            "avg_payout": _parse_number(summary.get("avg_payout")),
            "median_payout": _parse_number(summary.get("median_payout")),
            "total_pa_retained_net_pnl": _parse_number(summary.get("total_pa_retained_net_pnl")),
            "total_pa_net_plus_payouts": _parse_number(summary.get("total_pa_net_plus_payouts")),
            "avg_pa_net_plus_payouts": _parse_number(summary.get("avg_pa_net_plus_payouts")),
            "median_pa_net_plus_payouts": _parse_number(summary.get("median_pa_net_plus_payouts")),
            "avg_pa_days_survived": _parse_number(summary.get("avg_pa_days_survived")),
            "median_pa_days_survived": _parse_number(summary.get("median_pa_days_survived")),
        } | event_metrics | interval_metrics

        row["payout_score"] = compute_score(row)
        rows.append(row)

        results = pd.DataFrame(rows)
        if not results.empty and "payout_score" in results.columns:
            results = results.sort_values("payout_score", ascending=False, na_position="last")
        results.to_csv(OUT_DIR / "payout_optimisation_results.csv", index=False)

        show_cols = [
            "strategy", "symbols", "target_r", "risk_per_trade", "daily_profit_target",
            "daily_soft_loss_stop", "max_trades_per_day", "pause_after_consecutive_losses",
            "lifecycle_profile", "approved_payout_count", "approved_payout_total",
            "pa_blown", "median_days_to_first_payout", "median_days_between_payouts",
            "total_pa_net_plus_payouts", "payout_score"
        ]
        present = [c for c in show_cols if c in results.columns]
        print("Current best:")
        print(results[present].head(10).to_string(index=False))

    final = pd.DataFrame(rows)
    if not final.empty and "payout_score" in final.columns:
        final = final.sort_values("payout_score", ascending=False, na_position="last")
    final.to_csv(OUT_DIR / "payout_optimisation_results.csv", index=False)

    print("\n================ FINAL OPTIMISATION RESULTS ================")
    if not final.empty:
        present = [c for c in [
            "strategy", "symbols", "target_r", "risk_per_trade", "daily_profit_target",
            "daily_soft_loss_stop", "max_trades_per_day", "pause_after_consecutive_losses",
            "lifecycle_profile", "event_net_pnl", "event_trades", "approved_payout_count",
            "approved_payout_total", "pa_blown", "median_days_to_first_payout",
            "median_days_between_payouts", "total_pa_net_plus_payouts", "payout_score", "run_id"
        ] if c in final.columns]
        print(final[present].head(20).to_string(index=False))
    print(f"\nWrote: {OUT_DIR / 'payout_optimisation_results.csv'}")


if __name__ == "__main__":
    main()
