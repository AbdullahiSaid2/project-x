from v471_shared import build_monthly_summary, load_v471_trades


def main() -> None:
    df = load_v471_trades()

    print("V471 headline")
    print(f"Trades: {len(df)}")
    if "gross_pnl_dollars_dynamic" in df.columns:
        print(f"Total realized PnL: {df['gross_pnl_dollars_dynamic'].sum():.2f}")

    if not df.empty:
        print("\nBy setup_type")
        if "setup_type" in df.columns:
            out = (
                df.groupby("setup_type")
                .agg(
                    trades=("gross_pnl_dollars_dynamic", "count"),
                    total_pnl_dollars=("gross_pnl_dollars_dynamic", "sum"),
                    avg_trade_dollars=("gross_pnl_dollars_dynamic", "mean"),
                )
                .reset_index()
            )
            print(out.to_string(index=False))

        print("\nMonthly summary")
        print(build_monthly_summary(df).to_string(index=False))


if __name__ == "__main__":
    main()
