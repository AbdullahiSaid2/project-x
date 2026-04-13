
from __future__ import annotations

from v460_shared import (
    OUT_DAILY_APEX_CSV,
    OUT_DAILY_CALENDAR_CSV,
    build_daily_apex_summary,
    build_daily_calendar_summary,
    load_v460_trades,
)

def main() -> None:
    df = load_v460_trades()

    cal = build_daily_calendar_summary(df)
    cal.to_csv(OUT_DAILY_CALENDAR_CSV, index=False)
    print(f"Saved: {OUT_DAILY_CALENDAR_CSV}")
    print(cal.to_string(index=False))

    print()
    apex = build_daily_apex_summary(df)
    apex.to_csv(OUT_DAILY_APEX_CSV, index=False)
    print(f"Saved: {OUT_DAILY_APEX_CSV}")
    print(apex.to_string(index=False))

if __name__ == "__main__":
    main()
