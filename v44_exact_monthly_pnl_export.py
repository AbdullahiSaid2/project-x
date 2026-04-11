from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtesting import Backtest
from src.data.fetcher import get_ohlcv
from src.strategies.manual.ict_multi_setup_v44 import ICT_MULTI_SETUP_V44


symbol = "NQ"
timeframe = "1m"
days_back = 365
tail_rows = 180000
contracts = 5
dollars_per_point = contracts * 2.0  # MNQ = $2/point per contract

out_csv = Path("v44_exact_monthly_pnl_5_mnq.csv")
out_xlsx = Path("v44_exact_monthly_pnl_5_mnq.xlsx")


def to_et(ts):
    if pd.isna(ts):
        return ts
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("America/New_York")


def realized_points(row):
    if row["side"] == "LONG":
        return float(row["exit_price"]) - float(row["entry_price"])
    return float(row["entry_price"]) - float(row["exit_price"])


print("1) Loading data...")
df = get_ohlcv(symbol, exchange="tradovate", timeframe=timeframe, days_back=days_back)
df = df.tail(tail_rows)
print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()}")

print("2) Running V44 backtest...")
bt = Backtest(
    df,
    ICT_MULTI_SETUP_V44,
    cash=1_000_000,
    commission=0.0,
    exclusive_orders=True,
)
stats = bt.run()
print(stats)

print("3) Building exact realized trade log...")
meta = pd.DataFrame(ICT_MULTI_SETUP_V44.last_trade_log)
if meta.empty:
    raise SystemExit("No trade metadata found. Aborting.")

meta["entry_time"] = pd.to_datetime(meta["entry_time"], errors="coerce")
meta["exit_time"] = pd.to_datetime(meta["exit_time"], errors="coerce")
meta["entry_time_et"] = meta["entry_time"].apply(to_et)
meta["exit_time_et"] = meta["exit_time"].apply(to_et)

meta["realized_points"] = meta.apply(realized_points, axis=1)
meta["realized_dollars_5_mnq"] = meta["realized_points"] * dollars_per_point

# Make month buckets from timezone-naive ET timestamps to avoid warnings
meta["entry_month_et"] = (
    pd.to_datetime(meta["entry_time_et"], errors="coerce")
    .dt.tz_localize(None)
    .dt.to_period("M")
    .astype(str)
)
meta["exit_month_et"] = (
    pd.to_datetime(meta["exit_time_et"], errors="coerce")
    .dt.tz_localize(None)
    .dt.to_period("M")
    .astype(str)
)

monthly = (
    meta.groupby("exit_month_et", dropna=False)
    .agg(
        trades=("realized_dollars_5_mnq", "size"),
        gross_pnl_dollars_5_mnq=("realized_dollars_5_mnq", "sum"),
        avg_trade_dollars=("realized_dollars_5_mnq", "mean"),
        gross_points=("realized_points", "sum"),
        avg_points_per_trade=("realized_points", "mean"),
        win_rate_pct=("realized_dollars_5_mnq", lambda s: (s > 0).mean() * 100),
    )
    .reset_index()
    .sort_values("exit_month_et")
)

monthly["cumulative_pnl_dollars_5_mnq"] = monthly["gross_pnl_dollars_5_mnq"].cumsum()

by_setup_month = (
    meta.groupby(["exit_month_et", "setup_type"], dropna=False)
    .agg(
        trades=("realized_dollars_5_mnq", "size"),
        pnl_dollars_5_mnq=("realized_dollars_5_mnq", "sum"),
        avg_trade_dollars=("realized_dollars_5_mnq", "mean"),
    )
    .reset_index()
    .sort_values(["exit_month_et", "pnl_dollars_5_mnq"], ascending=[True, False])
)

trade_cols = [
    "side",
    "setup_type",
    "bridge_type",
    "entry_variant",
    "planned_entry_price",
    "planned_stop_price",
    "planned_target_price",
    "stop_points",
    "tp_points",
    "planned_rr",
    "entry_price",
    "exit_price",
    "realized_points",
    "realized_dollars_5_mnq",
    "return_pct",
    "entry_time_et",
    "exit_time_et",
    "exit_month_et",
]
trade_cols = [c for c in trade_cols if c in meta.columns]
trades = meta[trade_cols].copy()

# CSV output
monthly.to_csv(out_csv, index=False)

# Excel cannot store timezone-aware datetimes, so strip timezone for export
monthly_excel = monthly.copy()
by_setup_month_excel = by_setup_month.copy()
trades_excel = trades.copy()

for col in ["entry_time_et", "exit_time_et"]:
    if col in trades_excel.columns:
        trades_excel[col] = pd.to_datetime(trades_excel[col], errors="coerce").dt.tz_localize(None)

with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    monthly_excel.to_excel(writer, sheet_name="Monthly PnL", index=False)
    by_setup_month_excel.to_excel(writer, sheet_name="By Setup By Month", index=False)
    trades_excel.to_excel(writer, sheet_name="Trades", index=False)

print("\n4) Exact monthly table")
print(monthly.to_string(index=False))

print("\n5) By setup by month")
print(by_setup_month.to_string(index=False))

print(f"\nSaved: {out_csv.resolve()}")
print(f"Saved: {out_xlsx.resolve()}")