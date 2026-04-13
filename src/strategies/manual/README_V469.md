# V469 build

This build implements the structural changes implied by your V464–V468 diagnostics:

- A-tier setups only
- London + NYPM continuations only
- Fixed 10 MNQ position sizing
- Minimum planned RR of 5.0
- No hard per-trade dollar floor filter
- Preserve runner logic instead of killing trades early

## What is included

- `src/strategies/generated/ict_multi_setup_v469.py`
- `src/strategies/manual/tmp_test_ict_multi_setup_v469.py`
- `src/strategies/manual/analyze_v469_results.py`
- `src/strategies/manual/monthly_pnl_export_v469.py`
- `src/strategies/manual/apex_50k_monthly_payout_export_v469.py`
- `src/strategies/manual/daily_pnl_export_v469.py`

## Important

I do **not** have your full repository in this environment, so these files are written to fit the structure strongly suggested by your logs.

The key assumptions are:

1. Your existing strategy engine already exposes setup metadata like:
   - `setup_type`
   - `bridge_type`
   - `setup_tier`
   - `planned_rr`
   - `entry_variant`
2. Your engine already supports partial + runner exits.
3. You already have local NQ 1m data cached and V468/V467-style export pipeline available.

## Intended logic change

The strategy should now only allow trades that satisfy all of:

- `setup_tier == "A"`
- `setup_type in {"LONDON_CONTINUATION", "NYPM_CONTINUATION"}`
- `planned_rr >= 5.0`
- fixed `10` MNQ contracts

## How to use

Copy these files into your repo, replacing/adapting your v468/v467 equivalents.

Then run:

```bash
python src/strategies/manual/tmp_test_ict_multi_setup_v469.py
python src/strategies/manual/analyze_v469_results.py
python src/strategies/manual/monthly_pnl_export_v469.py
python src/strategies/manual/apex_50k_monthly_payout_export_v469.py
python src/strategies/manual/daily_pnl_export_v469.py
```

## Why this build

Your logs showed:

- A-tier materially outperformed B-tier
- Asia continuation was the weakest concentration area
- Big PnL came from runner-style expansions, not from hard-capped exits
- Hard `$500 minimum target` filtering collapsed trade count too far

So V469 is a **quality concentration build**, not a harder target build.
