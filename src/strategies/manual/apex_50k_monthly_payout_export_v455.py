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

from src.strategies.manual.ict_multi_setup_v455 import ICT_MULTI_SETUP_V455
from src.strategies.manual.tmp_test_ict_multi_setup_v455 import load_nq_data


def main() -> None:
    df = load_nq_data()
    print("2) Running backtest...\n")
    bt = Backtest(df, ICT_MULTI_SETUP_V455, cash=1_000_000, commission=0.0, exclusive_orders=False, trade_on_close=False)
    stats = bt.run()

    print("3) Headline stats")
    print(stats)

    meta = pd.DataFrame(ICT_MULTI_SETUP_V455.TRADE_METADATA_LOG)
    if meta.empty:
        raise ValueError("No trade metadata log available.")

    meta["entry_time_et"] = pd.to_datetime(meta["entry_time_et"])
    meta["exit_time_et"] = pd.to_datetime(meta["exit_time_et"])
    meta["entry_time_et_naive"] = meta["entry_time_et"].dt.tz_localize(None)
    meta["exit_time_et_naive"] = meta["exit_time_et"].dt.tz_localize(None)
    meta["entry_date_et"] = meta["entry_time_et"].dt.date
    meta["exit_date_et"] = meta["exit_time_et"].dt.date
    meta["exit_month_et"] = meta["exit_time_et"].dt.to_period("M").astype(str)

    daily = (
        meta.groupby("exit_date_et", dropna=False)
        .agg(
            trades=("realized_dollars_5_mnq", "size"),
            gross_pnl_dollars=("realized_dollars_5_mnq", "sum"),
            gross_points=("realized_points", "sum"),
            win_rate_pct=("realized_dollars_5_mnq", lambda s: (s.gt(0).mean() * 100.0) if len(s) else 0.0),
        )
        .reset_index()
        .sort_values("exit_date_et")
    )
    daily["qualifying_day_flag"] = daily["gross_pnl_dollars"] >= 150.0
    daily["month"] = pd.to_datetime(daily["exit_date_et"]).dt.to_period("M").astype(str)

    monthly = (
        meta.groupby("exit_month_et", dropna=False)
        .agg(
            trades=("realized_dollars_5_mnq", "size"),
            gross_pnl_dollars=("realized_dollars_5_mnq", "sum"),
            gross_points=("realized_points", "sum"),
            avg_trade_dollars=("realized_dollars_5_mnq", "mean"),
            avg_points_per_trade=("realized_points", "mean"),
            win_rate_pct=("realized_dollars_5_mnq", lambda s: (s.gt(0).mean() * 100.0) if len(s) else 0.0),
        )
        .reset_index()
        .sort_values("exit_month_et")
    )

    day_rollup = (
        daily.groupby("month", dropna=False)
        .agg(
            trading_days=("exit_date_et", "size"),
            green_days=("gross_pnl_dollars", lambda s: int((s > 0).sum())),
            red_days=("gross_pnl_dollars", lambda s: int((s < 0).sum())),
            worst_day_dollars=("gross_pnl_dollars", "min"),
            best_day_dollars=("gross_pnl_dollars", "max"),
            avg_day_dollars=("gross_pnl_dollars", "mean"),
            soft_loss_cap_breach_days=("gross_pnl_dollars", lambda s: int((s <= -1000).sum())),
            qualifying_days=("qualifying_day_flag", "sum"),
        )
        .reset_index()
        .rename(columns={"month": "exit_month_et"})
    )

    monthly = monthly.merge(day_rollup, on="exit_month_et", how="left")
    monthly["cumulative_pnl_dollars"] = monthly["gross_pnl_dollars"].cumsum()
    monthly["account_balance_est"] = 50000.0 + monthly["cumulative_pnl_dollars"]
    monthly["month_green"] = monthly["gross_pnl_dollars"] > 0
    monthly["eligible_for_review_flag"] = monthly["qualifying_days"] >= 5
    monthly["strong_month_flag"] = monthly["gross_pnl_dollars"] >= 2000
    monthly["conservative_payout_view"] = monthly["gross_pnl_dollars"].clip(lower=0) * 0.8
    monthly["retained_buffer_view"] = monthly["gross_pnl_dollars"].clip(lower=0) * 0.2
    monthly["equity_peak_est"] = monthly["account_balance_est"].cummax()
    monthly["drawdown_from_peak_dollars"] = monthly["account_balance_est"] - monthly["equity_peak_est"]

    out_dir = Path(".")
    excel_path = out_dir / "v455_apex_50k_monthly_payout_export.xlsx"
    monthly_csv = out_dir / "v455_apex_50k_monthly_summary.csv"
    daily_csv = out_dir / "v455_apex_50k_daily_summary.csv"
    trades_csv = out_dir / "v455_apex_50k_trade_log.csv"

    monthly.to_csv(monthly_csv, index=False)
    daily.to_csv(daily_csv, index=False)
    meta.drop(columns=["entry_time_et", "exit_time_et"]).to_csv(trades_csv, index=False)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        monthly.to_excel(writer, sheet_name="Monthly", index=False)
        daily.to_excel(writer, sheet_name="Daily", index=False)
        meta_to_excel = meta.copy()
        meta_to_excel["entry_time_et"] = meta_to_excel["entry_time_et"].dt.tz_localize(None)
        meta_to_excel["exit_time_et"] = meta_to_excel["exit_time_et"].dt.tz_localize(None)
        meta_to_excel.to_excel(writer, sheet_name="Trades", index=False)

    print("\n4) Export complete")
    print(f"Excel: {excel_path.resolve()}")
    print(f"Monthly CSV: {monthly_csv.resolve()}")
    print(f"Daily CSV: {daily_csv.resolve()}")
    print(f"Trades CSV: {trades_csv.resolve()}")

    print("\n5) Apex 50k monthly payout-style summary")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
