# Required Event Engine V2 Files

This package fixes the optimiser error:

```text
unrecognized arguments:
--risk-per-trade
--daily-profit-target
--daily-soft-loss-stop
--max-trades-per-day
--pause-after-consecutive-losses
```

## Files included

```text
src/backtesting/event_engine/run_event_backtest.py
src/strategies/manual/researched_prop_trend/prop_firm_lifecycle_profiles_balanced_patch.yaml
src/strategies/manual/researched_prop_trend/install_balanced_payout_profiles.py
```

## Install

From repo root:

```bash
cd /Users/Abdullahi/trading-project/trading_system

unzip /path/to/algotec_event_engine_v2_required_files.zip -d /tmp/algotec_event_v2

cp -R /tmp/algotec_event_v2/src/* src/
```

## Merge payout profiles

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.install_balanced_payout_profiles
```

## Verify the event engine is updated

```bash
PYTHONPATH=. python -m src.backtesting.event_engine.run_event_backtest --help | grep -E "risk-per-trade|daily-profit-target|daily-soft-loss-stop|max-trades|pause-after"
```

You should see:

```text
--risk-per-trade
--daily-profit-target
--daily-soft-loss-stop
--max-trades-per-day
--pause-after-consecutive-losses
```

## Rerun optimiser

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
