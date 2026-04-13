from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from pathlib import Path
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

    meta["entry_time_et"] = pd.to_datetime(meta["entry_time_et"])
    meta["exit_time_et"] = pd.to_datetime(meta["exit_time_et"])
    meta["entry_time_et_naive"] = meta["entry_time_et"].dt.tz_localize(None)
    meta["exit_time_et_naive"] = meta["exit_time_et"].dt.tz_localize(None)
    meta["entry_month_et"] = meta["entry_time_et"].dt.to_period("M").astype(str)
    meta["exit_month_et"] = meta["exit_time_et"].dt.to_period("M").astype(str)

    monthly = (
        meta.groupby("exit_month_et", dropna=False)
        .agg(
            trades=("realized_dollars_5_mnq", "size"),
            gross_pnl_dollars_5_mnq=("realized_dollars_5_mnq", "sum"),
            avg_trade_dollars=("realized_dollars_5_mnq", "mean"),
            gross_points=("realized_points", "sum"),
            avg_points_per_trade=("realized_points", "mean"),
            win_rate_pct=("realized_dollars_5_mnq", lambda s: (s.gt(0).mean() * 100.0) if len(s) else 0.0),
        )
        .reset_index()
        .sort_values("exit_month_et")
    )
    monthly["cumulative_pnl_dollars_5_mnq"] = monthly["gross_pnl_dollars_5_mnq"].cumsum()

    out_dir = Path(".")
    excel_path = out_dir / "v456_monthly_pnl_export.xlsx"
    monthly_csv = out_dir / "v456_monthly_pnl_summary.csv"
    trades_csv = out_dir / "v456_trade_log.csv"

    monthly.to_csv(monthly_csv, index=False)
    meta.drop(columns=["entry_time_et", "exit_time_et"]).to_csv(trades_csv, index=False)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        monthly.to_excel(writer, sheet_name="Monthly", index=False)
        meta_to_excel = meta.copy()
        meta_to_excel["entry_time_et"] = meta_to_excel["entry_time_et"].dt.tz_localize(None)
        meta_to_excel["exit_time_et"] = meta_to_excel["exit_time_et"].dt.tz_localize(None)
        meta_to_excel.to_excel(writer, sheet_name="Trades", index=False)

    print("\n4) Export complete")
    print(f"Excel: {excel_path.resolve()}")
    print(f"Monthly CSV: {monthly_csv.resolve()}")
    print(f"Trades CSV: {trades_csv.resolve()}")

    print("\n5) Monthly summary")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
