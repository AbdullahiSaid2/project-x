from pathlib import Path
import pandas as pd
from v473_shared import OUT_TRADE_CSV, run_all_symbols


def main():
    print("1) Running V473 multi-instrument build...")
    print("✅ Base logic: v452 preserved | symbols = NQ, MES, MYM, MGC | Apex 50k reporting layer | MCL removed | Apex hours filter enabled")
    df = run_all_symbols(["NQ", "MES", "MYM", "MGC"])
    if df.empty:
        raise SystemExit("No trades generated.")
    df.to_csv(OUT_TRADE_CSV, index=False)
    print(f"\n2) Saved trade log: {OUT_TRADE_CSV.resolve()}")
    print(f"3) Trade count: {len(df)}")
    show_cols = [c for c in ["symbol", "setup_type", "setup_tier", "bridge_type", "side", "planned_rr", "planned_target_dollars_dynamic", "report_contracts", "gross_pnl_dollars_dynamic", "entry_time_et", "exit_time_et"] if c in df.columns]
    print(df[show_cols].tail(30).to_string(index=False))
    print("\n4) Headline by symbol")
    summary = df.groupby("symbol", dropna=False).agg(trades=("gross_pnl_dollars_dynamic", "size"), total_realized_pnl=("gross_pnl_dollars_dynamic", "sum"), avg_trade=("gross_pnl_dollars_dynamic", "mean")).reset_index()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
