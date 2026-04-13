from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TRADES_CSV = ROOT / "v469_trade_log.csv"
OUT_CSV = ROOT / "v469_monthly_pnl_summary.csv"


def main() -> None:
    if not TRADES_CSV.exists():
        print(f"Missing {TRADES_CSV}")
        return

    df = pd.read_csv(TRADES_CSV)
    pnl_col = "realized_dollars_dynamic_contracts" if "realized_dollars_dynamic_contracts" in df.columns else "pnl"
    month_col = "exit_month_et" if "exit_month_et" in df.columns else None

    if month_col is None:
        if "exit_time_et" in df.columns:
            dt = pd.to_datetime(df["exit_time_et"], errors="coerce")
        elif "ExitTime" in df.columns:
            dt = pd.to_datetime(df["ExitTime"], errors="coerce")
        else:
            raise ValueError("No usable exit-time column found")
        df["exit_month_et"] = dt.dt.strftime("%Y-%m")
        month_col = "exit_month_et"

    monthly = (
        df.groupby(month_col, dropna=False)
        .agg(
            trades=(pnl_col, "size"),
            gross_pnl_dollars_dynamic=(pnl_col, "sum"),
            avg_trade_dollars_dynamic=(pnl_col, "mean"),
        )
        .reset_index()
        .rename(columns={month_col: "exit_month_et"})
        .sort_values("exit_month_et")
    )
    monthly["cumulative_pnl_dollars_dynamic"] = monthly["gross_pnl_dollars_dynamic"].cumsum()
    monthly.to_csv(OUT_CSV, index=False)
    print("Export complete")
    print("Monthly CSV:", OUT_CSV)
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
