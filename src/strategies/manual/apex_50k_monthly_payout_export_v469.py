from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TRADES_CSV = ROOT / "v469_trade_log.csv"
MONTHLY_CSV = ROOT / "v469_apex_50k_monthly_summary.csv"
DAILY_CSV = ROOT / "v469_apex_50k_daily_summary.csv"
START_BALANCE = 50_000.0


def main() -> None:
    if not TRADES_CSV.exists():
        print(f"Missing {TRADES_CSV}")
        return

    df = pd.read_csv(TRADES_CSV)
    pnl_col = "realized_dollars_dynamic_contracts" if "realized_dollars_dynamic_contracts" in df.columns else "pnl"
    day_col = "exit_apex_session_date" if "exit_apex_session_date" in df.columns else "calendar_exit_date_et"

    if day_col not in df.columns:
        raise ValueError("No daily grouping column found")

    daily = (
        df.groupby(day_col, dropna=False)
        .agg(
            trades=(pnl_col, "size"),
            gross_pnl_dollars=(pnl_col, "sum"),
        )
        .reset_index()
        .sort_values(day_col)
    )
    daily["cumulative_pnl_dollars"] = daily["gross_pnl_dollars"].cumsum()
    daily.to_csv(DAILY_CSV, index=False)

    daily["exit_month_et"] = daily[day_col].astype(str).str.slice(0, 7)
    monthly = (
        daily.groupby("exit_month_et", dropna=False)
        .agg(
            trading_days=(day_col, "size"),
            gross_pnl_dollars=("gross_pnl_dollars", "sum"),
            best_day_dollars=("gross_pnl_dollars", "max"),
            worst_day_dollars=("gross_pnl_dollars", "min"),
            avg_day_dollars=("gross_pnl_dollars", "mean"),
        )
        .reset_index()
        .sort_values("exit_month_et")
    )
    monthly["cumulative_pnl_dollars"] = monthly["gross_pnl_dollars"].cumsum()
    monthly["account_balance_est"] = START_BALANCE + monthly["cumulative_pnl_dollars"]
    monthly["month_green"] = monthly["gross_pnl_dollars"] > 0
    monthly.to_csv(MONTHLY_CSV, index=False)

    print("Export complete")
    print("Monthly CSV:", MONTHLY_CSV)
    print("Daily CSV:", DAILY_CSV)
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
