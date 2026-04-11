from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backtesting import Backtest
from src.data.fetcher import get_ohlcv
from src.strategies.manual.ict_multi_setup_v4535 import ICT_MULTI_SETUP_V4535


symbol = "NQ"
timeframe = "1m"
days_back = 365
tail_rows = 180000

# Position assumption for dollarizing trades
contracts = 5
dollars_per_point = contracts * 2.0  # MNQ = $2/point per contract

# Apex-style reference assumptions for a 50k account
apex_account_size = 50_000
starting_balance = 50_000
trailing_threshold_reference = 2_500  # simple planning reference, not broker-enforced here
daily_soft_loss_cap = 1_000           # planning/risk reference only


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

print("2) Running backtest...")
bt = Backtest(
    df,
    ICT_MULTI_SETUP_V4535,
    cash=1_000_000,
    commission=0.0,
    exclusive_orders=True,
)
stats = bt.run()

print("\n3) Headline stats")
print(stats)

meta = pd.DataFrame(ICT_MULTI_SETUP_V4535.last_trade_log)
if meta.empty:
    raise SystemExit("No trade metadata found. Nothing to export.")

meta["entry_time"] = pd.to_datetime(meta["entry_time"], errors="coerce")
meta["exit_time"] = pd.to_datetime(meta["exit_time"], errors="coerce")
meta["entry_time_et"] = meta["entry_time"].apply(to_et)
meta["exit_time_et"] = meta["exit_time"].apply(to_et)

meta["entry_time_et_naive"] = pd.to_datetime(meta["entry_time_et"], errors="coerce").dt.tz_localize(None)
meta["exit_time_et_naive"] = pd.to_datetime(meta["exit_time_et"], errors="coerce").dt.tz_localize(None)

meta["entry_month_et"] = meta["entry_time_et_naive"].dt.to_period("M").astype(str)
meta["exit_month_et"] = meta["exit_time_et_naive"].dt.to_period("M").astype(str)
meta["exit_date_et"] = meta["exit_time_et_naive"].dt.date

meta["realized_points"] = meta.apply(realized_points, axis=1)
meta["realized_dollars_5_mnq"] = meta["realized_points"] * dollars_per_point

# Daily summary
daily = (
    meta.groupby("exit_date_et", dropna=False)
    .agg(
        trades=("realized_dollars_5_mnq", "size"),
        pnl_dollars=("realized_dollars_5_mnq", "sum"),
        gross_points=("realized_points", "sum"),
        avg_trade_dollars=("realized_dollars_5_mnq", "mean"),
        win_rate_pct=("realized_dollars_5_mnq", lambda s: (s > 0).mean() * 100),
    )
    .reset_index()
    .sort_values("exit_date_et")
)

daily["month_et"] = pd.to_datetime(daily["exit_date_et"]).dt.to_period("M").astype(str)
daily["is_green_day"] = daily["pnl_dollars"] > 0
daily["is_red_day"] = daily["pnl_dollars"] < 0
daily["soft_loss_cap_breached"] = daily["pnl_dollars"] <= -daily_soft_loss_cap

# Monthly summary
monthly = (
    meta.groupby("exit_month_et", dropna=False)
    .agg(
        trades=("realized_dollars_5_mnq", "size"),
        gross_pnl_dollars=("realized_dollars_5_mnq", "sum"),
        gross_points=("realized_points", "sum"),
        avg_trade_dollars=("realized_dollars_5_mnq", "mean"),
        avg_points_per_trade=("realized_points", "mean"),
        win_rate_pct=("realized_dollars_5_mnq", lambda s: (s > 0).mean() * 100),
    )
    .reset_index()
    .sort_values("exit_month_et")
)

monthly_days = (
    daily.groupby("month_et", dropna=False)
    .agg(
        trading_days=("exit_date_et", "size"),
        green_days=("is_green_day", "sum"),
        red_days=("is_red_day", "sum"),
        worst_day_dollars=("pnl_dollars", "min"),
        best_day_dollars=("pnl_dollars", "max"),
        avg_day_dollars=("pnl_dollars", "mean"),
        soft_loss_cap_breach_days=("soft_loss_cap_breached", "sum"),
    )
    .reset_index()
)

monthly = monthly.merge(monthly_days, left_on="exit_month_et", right_on="month_et", how="left")
monthly = monthly.drop(columns=["month_et"])

monthly["cumulative_pnl_dollars"] = monthly["gross_pnl_dollars"].cumsum()
monthly["account_balance_est"] = starting_balance + monthly["cumulative_pnl_dollars"]
monthly["month_green"] = monthly["gross_pnl_dollars"] > 0

# Very simple payout-style planning fields
monthly["eligible_for_review_flag"] = monthly["gross_pnl_dollars"] > 0
monthly["strong_month_flag"] = monthly["gross_pnl_dollars"] >= 2_000
monthly["conservative_payout_view"] = monthly["gross_pnl_dollars"].clip(lower=0) * 0.80
monthly["retained_buffer_view"] = monthly["gross_pnl_dollars"].clip(lower=0) * 0.20

# Rolling drawdown-style planning view from month-end balances
monthly["equity_peak_est"] = monthly["account_balance_est"].cummax()
monthly["drawdown_from_peak_dollars"] = monthly["account_balance_est"] - monthly["equity_peak_est"]

# Setup/month breakdown
by_setup_month = (
    meta.groupby(["exit_month_et", "setup_type"], dropna=False)
    .agg(
        trades=("realized_dollars_5_mnq", "size"),
        pnl_dollars=("realized_dollars_5_mnq", "sum"),
        avg_trade_dollars=("realized_dollars_5_mnq", "mean"),
        gross_points=("realized_points", "sum"),
    )
    .reset_index()
    .sort_values(["exit_month_et", "pnl_dollars"], ascending=[True, False])
)

# Tier/month breakdown if available
by_tier_month = None
if "setup_tier" in meta.columns:
    by_tier_month = (
        meta.groupby(["exit_month_et", "setup_tier"], dropna=False)
        .agg(
            trades=("realized_dollars_5_mnq", "size"),
            pnl_dollars=("realized_dollars_5_mnq", "sum"),
            avg_trade_dollars=("realized_dollars_5_mnq", "mean"),
        )
        .reset_index()
        .sort_values(["exit_month_et", "pnl_dollars"], ascending=[True, False])
    )

# Trade export
trade_export_cols = [
    "side",
    "setup_type",
    "bridge_type",
    "setup_tier",
    "entry_variant",
    "planned_entry_price",
    "planned_stop_price",
    "planned_target_price",
    "partial_target_price",
    "runner_target_price",
    "stop_points",
    "tp_points",
    "planned_rr",
    "entry_price",
    "exit_price",
    "realized_points",
    "realized_dollars_5_mnq",
    "return_pct",
    "entry_time_et_naive",
    "exit_time_et_naive",
    "entry_month_et",
    "exit_month_et",
]
trade_export_cols = [c for c in trade_export_cols if c in meta.columns]
trades_export = meta[trade_export_cols].copy()

# Output files
output_xlsx = Path("v4535_apex_50k_monthly_payout_export.xlsx")
output_csv_monthly = Path("v4535_apex_50k_monthly_summary.csv")
output_csv_daily = Path("v4535_apex_50k_daily_summary.csv")
output_csv_trades = Path("v4535_apex_50k_trade_log.csv")

monthly.to_csv(output_csv_monthly, index=False)
daily.to_csv(output_csv_daily, index=False)
trades_export.to_csv(output_csv_trades, index=False)

with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
    monthly.to_excel(writer, sheet_name="MonthlySummary", index=False)
    daily.to_excel(writer, sheet_name="DailySummary", index=False)
    by_setup_month.to_excel(writer, sheet_name="BySetupMonth", index=False)
    if by_tier_month is not None:
        by_tier_month.to_excel(writer, sheet_name="ByTierMonth", index=False)
    trades_export.to_excel(writer, sheet_name="Trades", index=False)

print("\n4) Export complete")
print(f"Excel: {output_xlsx.resolve()}")
print(f"Monthly CSV: {output_csv_monthly.resolve()}")
print(f"Daily CSV: {output_csv_daily.resolve()}")
print(f"Trades CSV: {output_csv_trades.resolve()}")

print("\n5) Apex 50k monthly payout-style summary")
print(monthly.to_string(index=False))