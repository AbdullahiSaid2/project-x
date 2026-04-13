import pandas as pd
from v472_shared import OUT_CALENDAR_DAILY_CSV, OUT_APEX_SESSION_DAILY_CSV, OUT_TRADE_CSV, build_daily_summary


def main():
    df = pd.read_csv(OUT_TRADE_CSV)
    cal = build_daily_summary(df, by_apex_session=False)
    apex = build_daily_summary(df, by_apex_session=True)
    cal.to_csv(OUT_CALENDAR_DAILY_CSV, index=False)
    apex.to_csv(OUT_APEX_SESSION_DAILY_CSV, index=False)
    print(cal.to_string(index=False))
    print(f"\nSaved: {OUT_CALENDAR_DAILY_CSV.resolve()}")
    print(apex.to_string(index=False))
    print(f"\nSaved: {OUT_APEX_SESSION_DAILY_CSV.resolve()}")

if __name__ == "__main__":
    main()
