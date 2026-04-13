from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtesting import Backtest

from src.strategies.manual.ict_multi_setup_v463 import ICT_MULTI_SETUP_V463
from src.strategies.manual.reporting_v463 import load_strategy_meta, save_excel_with_naive_datetimes
from src.strategies.manual.tmp_test_ict_multi_setup_v463 import load_nq_data


def main() -> None:
    df = load_nq_data()
    print("2) Running backtest...\n")
    bt = Backtest(df, ICT_MULTI_SETUP_V463, cash=1_000_000, commission=0.0, exclusive_orders=False, trade_on_close=False)
    stats = bt.run()
    print("3) Headline stats")
    print(stats)

    meta = load_strategy_meta(ICT_MULTI_SETUP_V463)
    if meta.empty:
        print("\n4) No metadata log found.")
        return

    monthly = (
        meta.groupby("exit_month_et", dropna=False)
        .agg(
            trades=("realized_dollars_dynamic_contracts", "size"),
            gross_pnl_dollars_dynamic=("realized_dollars_dynamic_contracts", "sum"),
            gross_pnl_dollars_5_mnq=("realized_dollars_5_mnq", "sum"),
            gross_pnl_dollars_10_mnq=("realized_dollars_10_mnq", "sum"),
            avg_trade_dollars_dynamic=("realized_dollars_dynamic_contracts", "mean"),
            gross_points=("realized_points", "sum"),
            avg_points_per_trade=("realized_points", "mean"),
            win_rate_pct=("realized_dollars_dynamic_contracts", lambda s: (s.gt(0).mean() * 100.0) if len(s) else 0.0),
        )
        .reset_index()
        .sort_values("exit_month_et")
    )
    monthly["cumulative_pnl_dollars_dynamic"] = monthly["gross_pnl_dollars_dynamic"].cumsum()

    output_excel = Path("v463_monthly_pnl_export.xlsx")
    output_monthly_csv = Path("v463_monthly_pnl_summary.csv")
    output_trades_csv = Path("v463_trade_log.csv")

    save_excel_with_naive_datetimes(
        output_excel,
        {
            "monthly_summary": monthly,
            "trade_log": meta,
        },
    )

    monthly.to_csv(output_monthly_csv, index=False)
    meta.to_csv(output_trades_csv, index=False)

    print("\n4) Export complete")
    print(f"Excel: {output_excel.resolve()}")
    print(f"Monthly CSV: {output_monthly_csv.resolve()}")
    print(f"Trades CSV: {output_trades_csv.resolve()}")
    print("\n5) Monthly summary")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
