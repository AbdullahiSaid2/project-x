from v471_shared import OUT_MONTHLY_CSV, build_monthly_summary, load_v471_trades


def main() -> None:
    df = load_v471_trades()
    monthly = build_monthly_summary(df)
    monthly.to_csv(OUT_MONTHLY_CSV, index=False)
    print(f"Saved: {OUT_MONTHLY_CSV}")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
