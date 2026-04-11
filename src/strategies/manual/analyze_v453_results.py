from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backtesting import Backtest
from src.data.fetcher import get_ohlcv
from src.strategies.manual.ict_multi_setup_v453 import ICT_MULTI_SETUP_V453


symbol = "NQ"
timeframe = "1m"
days_back = 365
tail_rows = 180000
contracts = 5
dollars_per_point = contracts * 2.0  # MNQ = $2 per point per contract


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


def summarize_group(df: pd.DataFrame, group_cols):
    if df.empty:
        return pd.DataFrame()

    aggregations = {
        "trades": ("realized_dollars_5_mnq", "size"),
        "win_rate_pct": ("realized_dollars_5_mnq", lambda s: (s > 0).mean() * 100),
        "avg_trade_dollars": ("realized_dollars_5_mnq", "mean"),
        "total_pnl_dollars": ("realized_dollars_5_mnq", "sum"),
        "avg_points": ("realized_points", "mean"),
        "total_points": ("realized_points", "sum"),
        "avg_return_pct": ("return_pct", "mean"),
    }

    if "stop_points" in df.columns:
        aggregations["avg_stop_points"] = ("stop_points", "mean")
    if "tp_points" in df.columns:
        aggregations["avg_tp_points"] = ("tp_points", "mean")
    if "planned_rr" in df.columns:
        aggregations["avg_planned_rr"] = ("planned_rr", "mean")

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(
            ["total_pnl_dollars", "avg_trade_dollars", "trades"],
            ascending=[False, False, False],
        )
    )
    return out


print("1) Loading data...")
df = get_ohlcv(symbol, exchange="tradovate", timeframe=timeframe, days_back=days_back)
df = df.tail(tail_rows)
print(f"Loaded {len(df)} rows | start={df.index.min()} end={df.index.max()}")

print("2) Running backtest...")
bt = Backtest(
    df,
    ICT_MULTI_SETUP_V453,
    cash=1_000_000,
    commission=0.0,
    exclusive_orders=True,
)
stats = bt.run()

print("\n3) Headline stats")
print(stats)

print("\n4) Metadata log")
meta = pd.DataFrame(ICT_MULTI_SETUP_V453.last_trade_log)

if meta.empty:
    print("No trade metadata found.")
    raise SystemExit(0)

meta["entry_time"] = pd.to_datetime(meta["entry_time"], errors="coerce")
meta["exit_time"] = pd.to_datetime(meta["exit_time"], errors="coerce")
meta["entry_time_et"] = meta["entry_time"].apply(to_et)
meta["exit_time_et"] = meta["exit_time"].apply(to_et)

meta["realized_points"] = meta.apply(realized_points, axis=1)
meta["realized_dollars_5_mnq"] = meta["realized_points"] * dollars_per_point

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

meta = ensure_rr_columns(meta)

display_cols = [
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
    "entry_time_et",
    "exit_time_et",
]
display_cols = [c for c in display_cols if c in meta.columns]
print(meta[display_cols].tail(30).to_string(index=False))

print("\n5) By setup_type")
print(summarize_group(meta, ["setup_type"]).to_string(index=False))

print("\n6) By bridge_type")
print(summarize_group(meta, ["bridge_type"]).to_string(index=False))

print("\n7) By setup_type + bridge_type")
print(summarize_group(meta, ["setup_type", "bridge_type"]).to_string(index=False))

if "setup_tier" in meta.columns:
    print("\n8) By setup_tier")
    print(summarize_group(meta, ["setup_tier"]).to_string(index=False))

    print("\n9) By setup_type + setup_tier")
    print(summarize_group(meta, ["setup_type", "setup_tier"]).to_string(index=False))
else:
    print("\n8) By setup_tier")
    print("setup_tier column not found")
    print("\n9) By setup_type + setup_tier")
    print("setup_tier column not found")

print("\n10) Direction split")
print(summarize_group(meta, ["side"]).to_string(index=False))

print("\n11) Setup + side")
print(summarize_group(meta, ["setup_type", "side"]).to_string(index=False))

print("\n12) Entry variant")
print(summarize_group(meta, ["entry_variant"]).to_string(index=False))

print("\n13) Monthly realized PnL at 5 MNQ")
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
print(monthly.to_string(index=False))

print("\n14) By month + setup_type")
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
print(by_setup_month.to_string(index=False))