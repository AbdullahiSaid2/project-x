
# Prop Lifecycle Simulator

This adds the realistic prop-firm account flow:

```text
Eval pass
→ trade PA
→ if PA max loss is hit, PA is blown/stopped
→ start new Eval
→ if Eval passes, trade next PA
→ repeat
```

## Important rule logic

```text
Daily Loss Limit:
  pause/lock trading for that day only
  not account failure

Max Loss / EOD Drawdown:
  account failure
  Eval fails if hit before target
  PA is blown if hit after funded account starts
```

## Install

Copy:

```text
prop_lifecycle_simulator.py
```

into:

```text
src/strategies/manual/researched_prop_trend/
```

## Run

From repo root:

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.prop_lifecycle_simulator   --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv   --eval-profile apex_50k_eod_eval   --pa-profile apex_50k_pa
```

Optional 30-day eval expiry:

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.prop_lifecycle_simulator   --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv   --eval-profile apex_50k_eod_eval   --pa-profile apex_50k_pa   --eval-max-calendar-days 30
```

## Outputs

```text
prop_lifecycle_cycles.csv
prop_lifecycle_events.csv
prop_lifecycle_daily.csv
prop_lifecycle_summary.txt
```

## Interpretation

`prop_lifecycle_cycles.csv` has one row per eval phase and one row per PA phase.

Examples:

```text
phase=eval, status=EVAL_PASSED
phase=eval, status=EVAL_FAILED
phase=pa, status=PA_BLOWN
phase=pa, status=DATA_END
```

`prop_lifecycle_events.csv` logs exact event points:

```text
EVAL_PASSED
EVAL_FAILED_MAX_LOSS
PA_BLOWN_MAX_LOSS
DLL_PAUSE
```
