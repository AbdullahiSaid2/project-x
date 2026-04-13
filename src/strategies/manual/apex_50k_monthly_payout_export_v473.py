import pandas as pd
from v473_shared import OUT_APEX_MONTHLY_CSV, OUT_APEX_DAILY_CSV, OUT_APEX_XLSX, OUT_TRADE_CSV, build_apex_monthly_summary, build_daily_summary


def main():
    df = pd.read_csv(OUT_TRADE_CSV)
    monthly = build_apex_monthly_summary(df)
    daily = build_daily_summary(df, by_apex_session=True)
    monthly.to_csv(OUT_APEX_MONTHLY_CSV, index=False)
    daily.to_csv(OUT_APEX_DAILY_CSV, index=False)
    with pd.ExcelWriter(OUT_APEX_XLSX, engine="openpyxl") as writer:
        monthly.to_excel(writer, sheet_name="MonthlySummary", index=False)
        daily.to_excel(writer, sheet_name="DailySummary", index=False)
        df.to_excel(writer, sheet_name="Trades", index=False)
    print(monthly.to_string(index=False))
    print(f"Saved: {OUT_APEX_MONTHLY_CSV.resolve()}")
    print(f"Saved: {OUT_APEX_DAILY_CSV.resolve()}")
    print(f"Saved: {OUT_APEX_XLSX.resolve()}")

if __name__ == "__main__":
    main()
