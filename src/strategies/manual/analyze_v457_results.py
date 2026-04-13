from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from backtesting import Backtest

from src.strategies.manual.ict_multi_setup_v457 import ICT_MULTI_SETUP_V457
from src.strategies.manual.tmp_test_ict_multi_setup_v457 import load_nq_data


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            trades=("realized_dollars_5_mnq", "size"),
            win_rate_pct=("realized_dollars_5_mnq", lambda s: (s.gt(0).mean() * 100.0) if len(s) else 0.0),
            avg_trade_dollars=("realized_dollars_5_mnq", "mean"),
            total_pnl_dollars=("realized_dollars_5_mnq", "sum"),
            avg_points=("realized_points", "mean"),
            total_points=("realized_points", "sum"),
            avg_return_pct=("return_pct", "mean"),
            avg_stop_points=("stop_points", "mean"),
            avg_tp_points=("tp_points", "mean"),
            avg_planned_rr=("planned_rr", "mean"),
        )
        .reset_index()
    )
    return agg.sort_values("total_pnl_dollars", ascending=False)


def main() -> None:
    df = load_nq_data()
    print("2) Running backtest...\n")
    bt = Backtest(df, ICT_MULTI_SETUP_V457, cash=1_000_000, commission=0.0, exclusive_orders=False, trade_on_close=False)
    stats = bt.run()
    print("3) Headline stats")
    print(stats)

    meta = pd.DataFrame(ICT_MULTI_SETUP_V457.TRADE_METADATA_LOG)
    print("\n4) Metadata log")
    if meta.empty:
        print("No metadata log found.")
        return

    if "entry_time_et" in meta.columns:
        entry_ts = pd.to_datetime(meta["entry_time_et"])
        meta["entry_time_et_naive"] = entry_ts.dt.tz_localize(None)
        meta["entry_month_et"] = entry_ts.dt.to_period("M").astype(str)
    if "exit_time_et" in meta.columns:
        exit_ts = pd.to_datetime(meta["exit_time_et"])
        meta["exit_time_et_naive"] = exit_ts.dt.tz_localize(None)
        meta["exit_month_et"] = exit_ts.dt.to_period("M").astype(str)

    cols = [c for c in [
        "side","setup_type","bridge_type","setup_tier","entry_variant",
        "planned_entry_price","planned_stop_price","planned_target_price",
        "partial_target_price","runner_target_price","stop_points","tp_points","planned_rr",
        "entry_price","exit_price","realized_points","realized_dollars_5_mnq","return_pct",
        "entry_time_et","exit_time_et","entry_apex_session_date","exit_apex_session_date"
    ] if c in meta.columns]
    print(meta[cols].tail(40).to_string(index=False))

    print("\n5) By setup_type")
    print(summarize_group(meta, ["setup_type"]).to_string(index=False))

    print("\n6) By bridge_type")
    print(summarize_group(meta, ["bridge_type"]).to_string(index=False))

    if "setup_type" in meta.columns and "bridge_type" in meta.columns:
        print("\n7) By setup_type + bridge_type")
        print(summarize_group(meta, ["setup_type", "bridge_type"]).to_string(index=False))

    if "setup_tier" in meta.columns:
        print("\n8) By setup_tier")
        print(summarize_group(meta, ["setup_tier"]).to_string(index=False))

    print("\n9) Direction split")
    print(summarize_group(meta, ["side"]).to_string(index=False))

    print("\n10) Entry variant")
    print(summarize_group(meta, ["entry_variant"]).to_string(index=False))

    if "exit_month_et" in meta.columns:
        monthly = (
            meta.groupby("exit_month_et", dropna=False)
            .agg(
                trades=("realized_dollars_5_mnq", "size"),
                gross_pnl_dollars_5_mnq=("realized_dollars_5_mnq", "sum"),
                avg_trade_dollars=("realized_dollars_5_mnq", "mean"),
                gross_points=("realized_points", "sum"),
                avg_points_per_trade=("realized_points", "mean"),
                win_rate_pct=("realized_dollars_5_mnq", lambda s: (s.gt(0).mean() * 100.0) if len(s) else 0.0),
            )
            .reset_index()
            .sort_values("exit_month_et")
        )
        monthly["cumulative_pnl_dollars_5_mnq"] = monthly["gross_pnl_dollars_5_mnq"].cumsum()
        print("\n11) Monthly realized PnL at 5 MNQ")
        print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
