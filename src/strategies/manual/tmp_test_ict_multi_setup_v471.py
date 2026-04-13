from v471_shared import SOURCE_TRADE_CSV, build_v471_trade_log, save_v471_trade_log


def main() -> None:
    print("1) Loading source trades...")
    print(f"📦 Using local root CSV: {SOURCE_TRADE_CSV}")
    print("✅ V471 policy: source=v470 | A-tier only | London+NYPM only | fixed 10 MNQ | min RR 5 | min target $500 | keep losers | winners must realize >= $500")

    df = build_v471_trade_log()
    out = save_v471_trade_log(df)

    print(f"2) Saved filtered trade log: {out}")
    print(f"3) Trade count: {len(df)}")

    if not df.empty:
        cols = [c for c in [
            "setup_type", "setup_tier", "planned_rr", "planned_target_dollars_10_mnq",
            "gross_pnl_dollars_dynamic", "entry_time_et", "exit_time_et"
        ] if c in df.columns]
        print(df[cols].tail(20).to_string(index=False))
    else:
        print("No trades passed the V471 filter.")


if __name__ == "__main__":
    main()
