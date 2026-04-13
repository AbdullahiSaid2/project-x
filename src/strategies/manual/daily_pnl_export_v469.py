from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TRADES_CSV = ROOT / "v469_trade_log.csv"
CAL_CSV = ROOT / "v469_daily_pnl_calendar_et.csv"
APEX_CSV = ROOT / "v469_daily_pnl_apex_session.csv"


def export_one(df: pd.DataFrame, day_col: str, out_path: Path, pnl_col: str) -> None:
    if day_col not in df.columns:
        print(f"Skipping {day_col}: missing column")
        return
    out = (
        df.groupby(day_col, dropna=False)
        .agg(
            trades=(pnl_col, "size"),
            gross_pnl_dollars=(pnl_col, "sum"),
        )
        .reset_index()
        .sort_values(day_col)
    )
    out["cumulative_pnl_dollars"] = out["gross_pnl_dollars"].cumsum()
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))


def main() -> None:
    if not TRADES_CSV.exists():
        print(f"Missing {TRADES_CSV}")
        return

    df = pd.read_csv(TRADES_CSV)
    pnl_col = "realized_dollars_dynamic_contracts" if "realized_dollars_dynamic_contracts" in df.columns else "pnl"

    print("Calendar ET daily PnL")
    export_one(df, "calendar_exit_date_et", CAL_CSV, pnl_col)
    print("\nApex-session daily PnL")
    export_one(df, "exit_apex_session_date", APEX_CSV, pnl_col)


if __name__ == "__main__":
    main()
