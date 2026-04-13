from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from backtesting import Backtest

from src.strategies.manual.ict_multi_setup_v456 import ICT_MULTI_SETUP_V456
from src.strategies.manual.tmp_test_ict_multi_setup_v456 import load_nq_data


def main() -> None:
    df = load_nq_data()
    print("2) Running backtest...\n")
    bt = Backtest(df, ICT_MULTI_SETUP_V456, cash=1_000_000, commission=0.0, exclusive_orders=False, trade_on_close=False)
    stats = bt.run()
    print("3) Headline stats")
    print(stats)

    meta = pd.DataFrame(ICT_MULTI_SETUP_V456.TRADE_METADATA_LOG)
    if meta.empty:
        raise ValueError("No trade metadata log available.")

    meta["calendar_exit_date_et"] = pd.to_datetime(meta["calendar_exit_date_et"])
    meta["exit_apex_session_date"] = pd.to_datetime(meta["exit_apex_session_date"])

    cal = (
        meta.groupby("calendar_exit_date_et", dropna=False)
        .agg(
            trades=("realized_dollars_5_mnq", "size"),
            gross_pnl_dollars=("realized_dollars_5_mnq", "sum"),
            gross_points=("realized_points", "sum"),
            avg_trade_dollars=("realized_dollars_5_mnq", "mean"),
            win_rate_pct=("realized_dollars_5_mnq", lambda s: (s.gt(0).mean() * 100.0) if len(s) else 0.0),
        )
        .reset_index()
        .sort_values("calendar_exit_date_et")
    )
    cal["cumulative_pnl_dollars"] = cal["gross_pnl_dollars"].cumsum()

    apex = (
        meta.groupby("exit_apex_session_date", dropna=False)
        .agg(
            trades=("realized_dollars_5_mnq", "size"),
            gross_pnl_dollars=("realized_dollars_5_mnq", "sum"),
            gross_points=("realized_points", "sum"),
            avg_trade_dollars=("realized_dollars_5_mnq", "mean"),
            win_rate_pct=("realized_dollars_5_mnq", lambda s: (s.gt(0).mean() * 100.0) if len(s) else 0.0),
        )
        .reset_index()
        .sort_values("exit_apex_session_date")
    )
    apex["cumulative_pnl_dollars"] = apex["gross_pnl_dollars"].cumsum()

    cal_csv = Path("v456_daily_pnl_calendar_et.csv")
    apex_csv = Path("v456_daily_pnl_apex_session.csv")
    cal.to_csv(cal_csv, index=False)
    apex.to_csv(apex_csv, index=False)

    print("\n4) Export complete")
    print(f"Calendar daily CSV: {cal_csv.resolve()}")
    print(f"Apex-session daily CSV: {apex_csv.resolve()}")
    print("\n5) Calendar ET daily PnL")
    print(cal.to_string(index=False))
    print("\n6) Apex session daily PnL")
    print(apex.to_string(index=False))


if __name__ == "__main__":
    main()
