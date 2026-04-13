
from __future__ import annotations

from v460_shared import POLICY, SOURCE_TRADE_CSV, OUT_TRADE_CSV, apply_v460_policy, load_source_trades

def main() -> None:
    print("1) Loading source trades...")
    print(f"📦 Using local root CSV: {SOURCE_TRADE_CSV}")
    print(
        "✅ V460 policy: source=v452 | all sessions kept | tier-weighted dynamic sizing | "
        f"min RR {POLICY.min_planned_rr:g} | min target ${POLICY.min_planned_target_dollars_10_mnq:g} on 10 MNQ | "
        "long-size discount enabled | Apex 50k"
    )

    source_df = load_source_trades()
    trades = apply_v460_policy(source_df)
    trades.to_csv(OUT_TRADE_CSV, index=False)

    print(f"2) Saved filtered trade log: {OUT_TRADE_CSV}")
    print(f"3) Trade count: {len(trades)}")
    cols = [
        "setup_type",
        "setup_tier",
        "bridge_type",
        "side",
        "planned_rr",
        "planned_target_dollars_10_mnq",
        "report_contracts",
        "gross_pnl_dollars_dynamic",
        "entry_time_et",
        "exit_time_et",
    ]
    print(trades[cols].head(20).to_string(index=False))

    print("\nV460 headline")
    print(f"Trades: {len(trades)}")
    print(f"Total realized PnL: {trades['gross_pnl_dollars_dynamic'].sum():.2f}")

    by_setup = (
        trades.groupby("setup_type")
        .agg(
            trades=("gross_pnl_dollars_dynamic", "size"),
            total_pnl_dollars=("gross_pnl_dollars_dynamic", "sum"),
            avg_trade_dollars=("gross_pnl_dollars_dynamic", "mean"),
        )
        .reset_index()
        .sort_values("total_pnl_dollars", ascending=False)
    )
    print("\nBy setup_type")
    print(by_setup.to_string(index=False))

if __name__ == "__main__":
    main()
