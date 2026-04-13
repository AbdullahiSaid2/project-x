from __future__ import annotations

import pandas as pd


def apex_session_date(ts: pd.Timestamp):
    ts = pd.Timestamp(ts)
    if ts.hour >= 18:
        return ts.date()
    return (ts - pd.Timedelta(days=1)).date()


def main() -> None:
    df = pd.read_csv("v457_trade_log.csv")
    df["entry_time_et"] = pd.to_datetime(df["entry_time_et"])
    df["exit_time_et"] = pd.to_datetime(df["exit_time_et"])
    df["entry_session_date"] = df["entry_time_et"].apply(apex_session_date)
    df["exit_session_date"] = df["exit_time_et"].apply(apex_session_date)

    viol = df[df["entry_session_date"] != df["exit_session_date"]].copy()
    print(f"Total trades: {len(df)}")
    print(f"Apex session violations: {len(viol)}")
    if not viol.empty:
        cols = [
            "side", "setup_type", "bridge_type", "setup_tier", "realized_dollars_5_mnq",
            "entry_time_et", "exit_time_et", "entry_session_date", "exit_session_date"
        ]
        print("\n=== Apex session violations ===")
        print(viol[cols].to_string(index=False))
        viol.to_csv("v457_apex_session_violations.csv", index=False)
        print("\nSaved violations CSV: v457_apex_session_violations.csv")


if __name__ == "__main__":
    main()
