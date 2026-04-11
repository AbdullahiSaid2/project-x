import pandas as pd
from pathlib import Path

TRADE_LOG = "v454_trade_log.csv"
FORCE_FLAT_HOUR = 16
FORCE_FLAT_MINUTE = 59


def apex_session_date(ts: pd.Timestamp) -> pd.Timestamp.date:
    """
    Apex session:
    - starts at 18:00 ET
    - ends at 16:59 ET next day

    If time >= 18:00, the trade belongs to the next calendar date's session label.
    Otherwise, it belongs to the current calendar date's session label.
    """
    if ts.hour >= 18:
        return (ts + pd.Timedelta(days=1)).date()
    return ts.date()


def main() -> None:
    path = Path(TRADE_LOG)
    if not path.exists():
        raise FileNotFoundError(f"Could not find trade log: {path.resolve()}")

    df = pd.read_csv(path)

    required_cols = ["entry_time_et_naive", "exit_time_et_naive"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df["entry_time_et_naive"] = pd.to_datetime(df["entry_time_et_naive"])
    df["exit_time_et_naive"] = pd.to_datetime(df["exit_time_et_naive"])

    df["entry_session_date"] = df["entry_time_et_naive"].apply(apex_session_date)
    df["exit_session_date"] = df["exit_time_et_naive"].apply(apex_session_date)

    df["exit_after_1659"] = (
        (df["exit_time_et_naive"].dt.hour > FORCE_FLAT_HOUR)
        | (
            (df["exit_time_et_naive"].dt.hour == FORCE_FLAT_HOUR)
            & (df["exit_time_et_naive"].dt.minute > FORCE_FLAT_MINUTE)
        )
    )

    df["crossed_apex_session"] = df["entry_session_date"] != df["exit_session_date"]

    violations = df[df["crossed_apex_session"]].copy()

    print(f"Total trades: {len(df)}")
    print(f"Apex session violations: {len(violations)}")

    show_cols = [
        "side",
        "setup_type",
        "bridge_type",
        "setup_tier",
        "realized_dollars_5_mnq",
        "entry_time_et_naive",
        "exit_time_et_naive",
        "entry_session_date",
        "exit_session_date",
    ]

    optional_cols = [c for c in show_cols if c in df.columns]

    if len(violations) > 0:
        print("\n=== Apex session violations ===")
        print(violations[optional_cols].to_string(index=False))
    else:
        print("\nNo Apex session violations found.")

    late_same_session = df[
        (~df["crossed_apex_session"])
        & (df["exit_time_et_naive"].dt.hour == 16)
        & (df["exit_time_et_naive"].dt.minute >= 0)
    ].copy()

    if len(late_same_session) > 0:
        print("\n=== Trades exiting during 16:00 hour ET but still same Apex session ===")
        print(late_same_session[optional_cols].to_string(index=False))

    # Optional export
    violations_out = path.with_name("v454_apex_session_violations.csv")
    violations.to_csv(violations_out, index=False)
    print(f"\nSaved violations CSV: {violations_out.resolve()}")


if __name__ == "__main__":
    main()