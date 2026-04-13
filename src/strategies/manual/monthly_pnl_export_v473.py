import pandas as pd
from v473_shared import OUT_MONTHLY_CSV, OUT_MONTHLY_XLSX, OUT_TRADE_CSV, build_monthly_summary


def main():
    df = pd.read_csv(OUT_TRADE_CSV)
    monthly = build_monthly_summary(df)
    monthly["cumulative_pnl_dollars_dynamic"] = monthly.groupby("symbol")["gross_pnl_dollars_dynamic"].cumsum()
    monthly.to_csv(OUT_MONTHLY_CSV, index=False)
    with pd.ExcelWriter(OUT_MONTHLY_XLSX, engine="openpyxl") as writer:
        monthly.to_excel(writer, sheet_name="MonthlySummary", index=False)
        df.to_excel(writer, sheet_name="Trades", index=False)
    print(monthly.to_string(index=False))
    print(f"Saved: {OUT_MONTHLY_CSV.resolve()}")
    print(f"Saved: {OUT_MONTHLY_XLSX.resolve()}")

if __name__ == "__main__":
    main()
