import pandas as pd
from v472_shared import OUT_TRADE_CSV


def main():
    df = pd.read_csv(OUT_TRADE_CSV)
    print("V472 headline")
    print(f"Trades: {len(df)}")
    print(f"Total realized PnL: {df['gross_pnl_dollars_dynamic'].sum():.2f}")
    print("\nBy symbol")
    print(df.groupby("symbol").agg(trades=("gross_pnl_dollars_dynamic", "size"), total_pnl_dollars=("gross_pnl_dollars_dynamic", "sum"), avg_trade_dollars=("gross_pnl_dollars_dynamic", "mean")).reset_index().to_string(index=False))
    print("\nBy setup_type")
    print(df.groupby(["symbol", "setup_type"]).agg(trades=("gross_pnl_dollars_dynamic", "size"), total_pnl_dollars=("gross_pnl_dollars_dynamic", "sum"), avg_trade_dollars=("gross_pnl_dollars_dynamic", "mean")).reset_index().sort_values(["symbol", "total_pnl_dollars"], ascending=[True, False]).to_string(index=False))

if __name__ == "__main__":
    main()
