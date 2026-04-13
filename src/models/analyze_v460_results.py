
from __future__ import annotations

from v460_shared import build_monthly_summary, load_v460_trades

def main() -> None:
    df = load_v460_trades()

    print("V460 headline")
    print(f"Trades: {len(df)}")
    print(f"Total realized PnL: {df['gross_pnl_dollars_dynamic'].sum():.2f}")

    for group_col in ["setup_type", "setup_tier", "bridge_type", "side"]:
        summary = (
            df.groupby(group_col)
            .agg(
                trades=("gross_pnl_dollars_dynamic", "size"),
                total_pnl_dollars=("gross_pnl_dollars_dynamic", "sum"),
                avg_trade_dollars=("gross_pnl_dollars_dynamic", "mean"),
            )
            .reset_index()
            .sort_values("total_pnl_dollars", ascending=False)
        )
        print(f"\nBy {group_col}")
        print(summary.to_string(index=False))

    monthly = build_monthly_summary(df)
    print("\nMonthly summary")
    print(monthly.to_string(index=False))

if __name__ == "__main__":
    main()
