from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from backtesting import Backtest

from src.strategies.manual.ict_multi_setup_v468 import ICT_MULTI_SETUP_V468
from src.strategies.manual.reporting_v468 import load_strategy_meta, save_excel_with_naive_datetimes
from src.strategies.manual.tmp_test_ict_multi_setup_v468 import load_nq_data


def main() -> None:
    df = load_nq_data()
    print("2) Running backtest...\n")
    bt = Backtest(df, ICT_MULTI_SETUP_V468, cash=1_000_000, commission=0.0, exclusive_orders=False, trade_on_close=False)
    stats = bt.run()
    print("3) Headline stats")
    print(stats)

    meta = load_strategy_meta(ICT_MULTI_SETUP_V468)
    if meta.empty:
        print("\n4) No metadata log found.")
        return

    daily = (
        meta.dropna(subset=["exit_apex_session_date"])
        .groupby("exit_apex_session_date", dropna=False)
        .agg(
            trades=("realized_dollars_dynamic_contracts", "size"),
            gross_pnl_dollars=("realized_dollars_dynamic_contracts", "sum"),
            gross_points=("realized_points", "sum"),
            avg_trade_dollars=("realized_dollars_dynamic_contracts", "mean"),
            win_rate_pct=("realized_dollars_dynamic_contracts", lambda s: (s.gt(0).mean() * 100.0) if len(s) else 0.0),
        )
        .reset_index()
        .sort_values("exit_apex_session_date")
    )
    daily["cumulative_pnl_dollars"] = daily["gross_pnl_dollars"].cumsum()

    monthly = (
        meta.groupby("exit_month_et", dropna=False)
        .agg(
            trades=("realized_dollars_dynamic_contracts", "size"),
            gross_pnl_dollars=("realized_dollars_dynamic_contracts", "sum"),
            gross_points=("realized_points", "sum"),
            avg_trade_dollars=("realized_dollars_dynamic_contracts", "mean"),
            avg_points_per_trade=("realized_points", "mean"),
            win_rate_pct=("realized_dollars_dynamic_contracts", lambda s: (s.gt(0).mean() * 100.0) if len(s) else 0.0),
            trading_days=("exit_apex_session_date", pd.Series.nunique),
        )
        .reset_index()
        .sort_values("exit_month_et")
    )

    monthly_extra = (
        daily.assign(exit_month_et=pd.to_datetime(daily["exit_apex_session_date"]).dt.to_period("M").astype(str))
        .groupby("exit_month_et", dropna=False)
        .agg(
            green_days=("gross_pnl_dollars", lambda s: int((s > 0).sum())),
            red_days=("gross_pnl_dollars", lambda s: int((s < 0).sum())),
            worst_day_dollars=("gross_pnl_dollars", "min"),
            best_day_dollars=("gross_pnl_dollars", "max"),
            avg_day_dollars=("gross_pnl_dollars", "mean"),
            soft_loss_cap_breach_days=("gross_pnl_dollars", lambda s: int((s <= -1000.0).sum())),
        )
        .reset_index()
    )

    monthly = monthly.merge(monthly_extra, on="exit_month_et", how="left")
    monthly["cumulative_pnl_dollars"] = monthly["gross_pnl_dollars"].cumsum()
    monthly["account_balance_est"] = 50000.0 + monthly["cumulative_pnl_dollars"]
    monthly["month_green"] = monthly["gross_pnl_dollars"] > 0
    monthly["strong_month_flag"] = monthly["gross_pnl_dollars"] >= 3000.0
    monthly["equity_peak_est"] = monthly["account_balance_est"].cummax()
    monthly["drawdown_from_peak_dollars"] = monthly["account_balance_est"] - monthly["equity_peak_est"]

    output_excel = Path("v468_apex_50k_monthly_payout_export.xlsx")
    output_monthly_csv = Path("v468_apex_50k_monthly_summary.csv")
    output_daily_csv = Path("v468_apex_50k_daily_summary.csv")
    output_trades_csv = Path("v468_apex_50k_trade_log.csv")

    save_excel_with_naive_datetimes(
        output_excel,
        {
            "monthly_summary": monthly,
            "daily_summary": daily,
            "trade_log": meta,
        },
    )

    monthly.to_csv(output_monthly_csv, index=False)
    daily.to_csv(output_daily_csv, index=False)
    meta.to_csv(output_trades_csv, index=False)

    print("\n4) Export complete")
    print(f"Excel: {output_excel.resolve()}")
    print(f"Monthly CSV: {output_monthly_csv.resolve()}")
    print(f"Daily CSV: {output_daily_csv.resolve()}")
    print(f"Trades CSV: {output_trades_csv.resolve()}")
    print("\n5) Apex 50k monthly payout-style summary")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
