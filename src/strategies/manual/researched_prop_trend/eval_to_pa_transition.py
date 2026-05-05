
from __future__ import annotations

"""
Eval-to-PA journey report.

This script:
  1. Reads event_trade_log.csv
  2. Finds when eval passes using apex_eval_simulator logic
  3. Reports the next trade index/time that would begin PA mode

Then you can run the event engine separately using apex_50k_pa + news rules for the PA phase.

Run:

PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.eval_to_pa_transition \
  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv
"""

from pathlib import Path
import argparse
import pandas as pd

from .apex_eval_simulator import load_profile, prepare_trades, simulate_attempt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eval_to_pa_transition_report.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-log", default=str(ROOT / "event_trade_log.csv"))
    parser.add_argument("--prop-profile", default="apex_50k_eod_eval")
    args = parser.parse_args()

    trades = prepare_trades(Path(args.trade_log))
    profile = load_profile(args.prop_profile)
    result = simulate_attempt(trades, profile, attempt_id=1, start_index=0)
    daily = result.pop("_daily_records", [])

    status = result.get("status")
    end_idx = int(result.get("end_index", 0))
    next_idx = end_idx + 1
    next_time = trades.iloc[next_idx]["exit_dt_et"] if next_idx < len(trades) else pd.NaT

    lines = []
    lines.append("================ EVAL TO PA TRANSITION ================")
    lines.append(f"Eval status: {status}")
    lines.append(f"Eval end index: {end_idx}")
    lines.append(f"Next PA start index: {next_idx}")
    lines.append(f"Next PA start time ET: {next_time}")
    lines.append(f"Eval net PnL: ${float(result.get('net_pnl', 0.0)):,.2f}")
    lines.append(f"Eval trades: {result.get('trades')}")
    lines.append(f"Eval calendar days: {result.get('calendar_days')}")
    lines.append(f"Worst intraday DD during eval: ${float(result.get('worst_intraday_drawdown', 0.0)):,.2f}")
    lines.append(f"Worst EOD DD during eval: ${float(result.get('worst_eod_drawdown', 0.0)):,.2f}")
    lines.append("")
    lines.append("To simulate PA from that point using an existing event_trade_log:")
    lines.append("")
    lines.append("PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.performance_account_simulator \\")
    lines.append("  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv \\")
    lines.append("  --prop-profile apex_50k_pa \\")
    lines.append(f"  --start-index {next_idx}")
    text = "\n".join(lines)
    print(text)
    OUT.write_text(text + "\n")


if __name__ == "__main__":
    main()
