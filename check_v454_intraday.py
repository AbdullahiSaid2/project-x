import pandas as pd

CSV_PATH = "v454_trade_log.csv"

df = pd.read_csv(CSV_PATH)

entry_col = "entry_time_et_naive"
exit_col = "exit_time_et_naive"

df[entry_col] = pd.to_datetime(df[entry_col])
df[exit_col] = pd.to_datetime(df[exit_col])

# 1) Trades that end on a different calendar day than they started
overnight = df[df[exit_col].dt.date != df[entry_col].dt.date].copy()

# 2) Trades that exit after 16:00 ET
after_1600 = df[
    (df[exit_col].dt.hour > 16)
    | ((df[exit_col].dt.hour == 16) & (df[exit_col].dt.minute > 0))
].copy()

# 3) Trades that exit after 16:30 ET
after_1630 = df[
    (df[exit_col].dt.hour > 16)
    | ((df[exit_col].dt.hour == 16) & (df[exit_col].dt.minute > 30))
].copy()

show_cols = [
    c for c in [
        "side",
        "setup_type",
        "bridge_type",
        "setup_tier",
        "pnl",
        "realized_dollars_5_mnq",
        entry_col,
        exit_col,
    ]
    if c in df.columns
]

print("Total trades:", len(df))
print("Overnight trades:", len(overnight))
print("Exit after 16:00 ET:", len(after_1600))
print("Exit after 16:30 ET:", len(after_1630))

print("\n=== Overnight trades ===")
if len(overnight):
    print(overnight[show_cols].to_string(index=False))
else:
    print("None")

print("\n=== Trades exiting after 16:00 ET ===")
if len(after_1600):
    print(after_1600[show_cols].to_string(index=False))
else:
    print("None")

print("\n=== Trades exiting after 16:30 ET ===")
if len(after_1630):
    print(after_1630[show_cols].to_string(index=False))
else:
    print("None")