
# Algotec Payout Optimisation Runner

This adds a payout-first optimisation runner.

It tests combinations of:

```text
strategy
symbols
target-R
risk per trade
daily profit target
daily soft loss stop
max trades/day
pause after losses
lifecycle profile
```

Then it ranks each run by payout lifecycle performance.

## Install

From repo root:

```bash
cd /Users/Abdullahi/trading-project/trading_system

unzip /path/to/algotec_payout_optimisation_runner_package.zip -d /tmp/algotec_payout_opt

cp -R /tmp/algotec_payout_opt/src/* src/
```

## Start small

Do not run a huge grid first.

Start with ICT Fractal, 365 days, 6 configs:

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.payout_optimisation_runner \
  --strategies ict_fractal \
  --symbol-sets MNQ,MYM MNQ,MES,MYM \
  --target-r 3 4 5 \
  --risk-per-trade 75 100 \
  --daily-profit-target 350 500 \
  --daily-soft-loss-stop 250 300 \
  --max-trades-per-day 3 5 \
  --pause-after-consecutive-losses 1 2 \
  --lifecycle-profiles apex_50k_eod_lifecycle_safe apex_50k_eod_lifecycle_balanced \
  --days-back 365 \
  --max-runs 6
```

Output:

```text
src/strategies/manual/researched_prop_trend/payout_optimisation/payout_optimisation_results.csv
```

## Run more configs after smoke test

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.payout_optimisation_runner \
  --strategies ict_fractal \
  --symbol-sets MNQ MNQ,MYM MNQ,MES,MYM MNQ,MES,MYM,MGC \
  --target-r 2 3 4 5 \
  --risk-per-trade 75 100 125 150 \
  --daily-profit-target 350 500 650 \
  --daily-soft-loss-stop 250 300 350 \
  --max-trades-per-day 3 5 6 \
  --pause-after-consecutive-losses 1 2 \
  --lifecycle-profiles apex_50k_eod_lifecycle_safe apex_50k_eod_lifecycle_balanced apex_50k_eod_lifecycle_payout_fast \
  --days-back 365 \
  --max-runs 50
```

## Continue from later grid index

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.payout_optimisation_runner \
  --strategies ict_fractal \
  --symbol-sets MNQ MNQ,MYM MNQ,MES,MYM MNQ,MES,MYM,MGC \
  --target-r 2 3 4 5 \
  --risk-per-trade 75 100 125 150 \
  --daily-profit-target 350 500 650 \
  --daily-soft-loss-stop 250 300 350 \
  --max-trades-per-day 3 5 6 \
  --pause-after-consecutive-losses 1 2 \
  --lifecycle-profiles apex_50k_eod_lifecycle_safe apex_50k_eod_lifecycle_balanced apex_50k_eod_lifecycle_payout_fast \
  --days-back 365 \
  --start-index 50 \
  --max-runs 50
```

## Compare all three strategies

Only do this after ICT Fractal smoke tests work:

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.payout_optimisation_runner \
  --strategies ict_fractal researched_prop_trend top_bottom_ticking \
  --symbol-sets MNQ,MYM MNQ,MES,MYM \
  --target-r 3 4 5 \
  --risk-per-trade 75 100 125 \
  --daily-profit-target 350 500 650 \
  --daily-soft-loss-stop 250 300 350 \
  --max-trades-per-day 3 5 \
  --pause-after-consecutive-losses 1 2 \
  --lifecycle-profiles apex_50k_eod_lifecycle_safe apex_50k_eod_lifecycle_balanced \
  --days-back 365 \
  --max-runs 60
```

## Ranking score

The runner creates a `payout_score`:

```text
approved payout total
+ 50% of PA net + payouts
+ bonus for PA completed max payout cycle
- penalty for PA blown
- penalty for long first payout delay
- penalty for long payout gaps
```

This ranks payout production, not just normal backtest PnL.

## Most important columns

```text
approved_payout_count
approved_payout_total
pa_blown
median_days_to_first_payout
median_days_between_payouts
total_pa_net_plus_payouts
payout_score
event_net_pnl
event_trades
```

## Recommended workflow

1. Run small 365-day ICT Fractal grid.
2. Pick top 5 configs by `payout_score`.
3. Run those configs for 1825 days.
4. Only then compare top_bottom_ticking and researched_prop_trend.
