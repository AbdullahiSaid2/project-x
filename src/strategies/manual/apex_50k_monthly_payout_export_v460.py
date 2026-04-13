
from __future__ import annotations

from v460_shared import OUT_APEX_MONTHLY_CSV, build_apex_monthly_summary, load_v460_trades

def main() -> None:
    df = load_v460_trades()
    monthly = build_apex_monthly_summary(df)
    monthly.to_csv(OUT_APEX_MONTHLY_CSV, index=False)
    print(f"Saved: {OUT_APEX_MONTHLY_CSV}")
    print(monthly.to_string(index=False))

if __name__ == "__main__":
    main()
