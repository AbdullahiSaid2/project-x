from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtesting import Backtest

from src.strategies.manual.ict_multi_setup_v461 import ICT_MULTI_SETUP_V461
from src.strategies.manual.reporting_v461 import load_strategy_meta
from src.strategies.manual.tmp_test_ict_multi_setup_v461 import load_nq_data


def main() -> None:
    df = load_nq_data()
    print("2) Running backtest...\n")
    bt = Backtest(df, ICT_MULTI_SETUP_V461, cash=1_000_000, commission=0.0, exclusive_orders=False, trade_on_close=False)
    stats = bt.run()
    print("3) Headline stats")
    print(stats)

    meta = load_strategy_meta(ICT_MULTI_SETUP_V461)
    if meta.empty:
        print("\n4) No metadata log found.")
        return

    print("\n4) Metadata log")
    print(meta.tail(40).to_string(index=False))

    def summarize(df_in, group_cols, value_col, title):
        grp = (
            df_in.groupby(group_cols, dropna=False)
            .agg(
                trades=(value_col, "size"),
                win_rate_pct=(value_col, lambda s: (s.gt(0).mean() * 100.0) if len(s) else 0.0),
                avg_trade_dollars=(value_col, "mean"),
                total_pnl_dollars=(value_col, "sum"),
                avg_points=("realized_points", "mean"),
                total_points=("realized_points", "sum"),
                avg_return_pct=("return_pct", "mean"),
                avg_stop_points=("stop_points", "mean"),
                avg_tp_points=("tp_points", "mean"),
                avg_planned_rr=("planned_rr", "mean"),
            )
            .reset_index()
            .sort_values("total_pnl_dollars", ascending=False)
        )
        print(f"\n{title}")
        print(grp.to_string(index=False))

    summarize(meta, ["setup_type"], "realized_dollars_dynamic_contracts", "5) By setup_type (dynamic 5-10 MNQ)")
    summarize(meta, ["bridge_type"], "realized_dollars_dynamic_contracts", "6) By bridge_type (dynamic 5-10 MNQ)")
    summarize(meta, ["setup_type", "bridge_type"], "realized_dollars_dynamic_contracts", "7) By setup_type + bridge_type (dynamic 5-10 MNQ)")
    summarize(meta, ["setup_tier"], "realized_dollars_dynamic_contracts", "8) By setup_tier (dynamic 5-10 MNQ)")
    summarize(meta, ["side"], "realized_dollars_dynamic_contracts", "9) Direction split (dynamic 5-10 MNQ)")
    summarize(meta, ["entry_variant"], "realized_dollars_dynamic_contracts", "10) Entry variant (dynamic 5-10 MNQ)")

    monthly = (
        meta.groupby("exit_month_et", dropna=False)
        .agg(
            trades=("realized_dollars_dynamic_contracts", "size"),
            gross_pnl_dollars_dynamic=("realized_dollars_dynamic_contracts", "sum"),
            gross_pnl_dollars_5_mnq=("realized_dollars_5_mnq", "sum"),
            gross_pnl_dollars_10_mnq=("realized_dollars_10_mnq", "sum"),
            avg_trade_dollars_dynamic=("realized_dollars_dynamic_contracts", "mean"),
            gross_points=("realized_points", "sum"),
            avg_points_per_trade=("realized_points", "mean"),
            win_rate_pct=("realized_dollars_dynamic_contracts", lambda s: (s.gt(0).mean() * 100.0) if len(s) else 0.0),
        )
        .reset_index()
        .sort_values("exit_month_et")
    )
    monthly["cumulative_pnl_dollars_dynamic"] = monthly["gross_pnl_dollars_dynamic"].cumsum()
    print("\n11) Monthly realized PnL (dynamic 5-10 MNQ)")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
