from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backtesting import Backtest
from src.data.fetcher import get_ohlcv
from src.strategies.manual.ict_multi_setup_v454 import ICT_MULTI_SETUP_V454


symbol = "NQ"
timeframe = "1m"
days_back = 365
tail_rows = 180000

contracts = 5
dollars_per_point = contracts * 2.0  # MNQ = $2/point per contract


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


def ensure_rr_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "stop_points" not in out.columns:
        if {"planned_entry_price", "planned_stop_price"}.issubset(out.columns):
            out["stop_points"] = (
                pd.to_numeric(out["planned_entry_price"], errors="coerce")
                - pd.to_numeric(out["planned_stop_price"], errors="coerce")
            ).abs()
        else:
            out["stop_points"] = pd.NA

    if "tp_points" not in out.columns:
        if {"planned_entry_price", "planned_target_price"}.issubset(out.columns):
            out["tp_points"] = (
                pd.to_numeric(out["planned_target_price"], errors="coerce")
                - pd.to_numeric(out["planned_entry_price"], errors="coerce")
            ).abs()
        else:
            out["tp_points"] = pd.NA

    if "planned_rr" not in out.columns:
        stop_num = pd.to_numeric(out["stop_points"], errors="coerce")
        tp_num = pd.to_numeric(out["tp_points"], errors="coerce")
        out["planned_rr"] = tp_num / stop_num.replace(0, pd.NA)

    return out


print("1) Loading data...")
df = get_ohlcv(symbol, exchange="tradovate", timeframe=timeframe, days_back=days_back)
df = df.tail(tail_rows)
print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()}")

print("2) Running backtest...")
bt = Backtest(
    df,
    ICT_MULTI_SETUP_V454,
    cash=1_000_000,
    commission=0.0,
    exclusive_orders=True,
)
stats = bt.run()

print("\n3) Headline stats")
print(stats)

meta = pd.DataFrame(ICT_MULTI_SETUP_V454.last_trade_log)
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

meta["realized_points"] = meta.apply(realized_points, axis=1)
meta["realized_dollars_5_mnq"] = meta["realized_points"] * dollars_per_point

meta = ensure_rr_columns(meta)

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
        gross_points=("realized_points", "sum"),
    )
    .reset_index()
    .sort_values(["exit_month_et", "pnl_dollars_5_mnq"], ascending=[True, False])
)

by_bridge_month = (
    meta.groupby(["exit_month_et", "bridge_type"], dropna=False)
    .agg(
        trades=("realized_dollars_5_mnq", "size"),
        pnl_dollars_5_mnq=("realized_dollars_5_mnq", "sum"),
        avg_trade_dollars=("realized_dollars_5_mnq", "mean"),
    )
    .reset_index()
    .sort_values(["exit_month_et", "pnl_dollars_5_mnq"], ascending=[True, False])
)

by_tier_month = None
if "setup_tier" in meta.columns:
    by_tier_month = (
        meta.groupby(["exit_month_et", "setup_tier"], dropna=False)
        .agg(
            trades=("realized_dollars_5_mnq", "size"),
            pnl_dollars_5_mnq=("realized_dollars_5_mnq", "sum"),
            avg_trade_dollars=("realized_dollars_5_mnq", "mean"),
        )
        .reset_index()
        .sort_values(["exit_month_et", "pnl_dollars_5_mnq"], ascending=[True, False])
    )

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

output_xlsx = Path("v454_monthly_pnl_export.xlsx")
output_csv_monthly = Path("v454_monthly_pnl_summary.csv")
output_csv_trades = Path("v454_trade_log.csv")

monthly.to_csv(output_csv_monthly, index=False)
trades_export.to_csv(output_csv_trades, index=False)

with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
    monthly.to_excel(writer, sheet_name="MonthlySummary", index=False)
    by_setup_month.to_excel(writer, sheet_name="BySetupMonth", index=False)
    by_bridge_month.to_excel(writer, sheet_name="ByBridgeMonth", index=False)
    if by_tier_month is not None:
        by_tier_month.to_excel(writer, sheet_name="ByTierMonth", index=False)
    trades_export.to_excel(writer, sheet_name="Trades", index=False)

print("\n4) Export complete")
print(f"Excel: {output_xlsx.resolve()}")
print(f"Monthly CSV: {output_csv_monthly.resolve()}")
print(f"Trades CSV: {output_csv_trades.resolve()}")

print("\n5) Monthly summary")
print(monthly.to_string(index=False))