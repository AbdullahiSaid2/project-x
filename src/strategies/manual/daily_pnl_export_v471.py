from v471_shared import (
    OUT_DAILY_APEX_CSV,
    OUT_DAILY_CALENDAR_CSV,
    build_apex_daily_summary,
    build_calendar_daily_summary,
    load_v471_trades,
)


def main() -> None:
    df = load_v471_trades()

    cal = build_calendar_daily_summary(df)
    apex = build_apex_daily_summary(df)

    cal.to_csv(OUT_DAILY_CALENDAR_CSV, index=False)
    apex.to_csv(OUT_DAILY_APEX_CSV, index=False)

    print(f"Saved: {OUT_DAILY_CALENDAR_CSV}")
    print(cal.to_string(index=False))
    print()
    print(f"Saved: {OUT_DAILY_APEX_CSV}")
    print(apex.to_string(index=False))


if __name__ == "__main__":
    main()
