# Researched Prop Trend v9 Event Engine

This package adds the production-style validation layer we discussed.

Files:

```text
prop_event_engine_backtest.py        # true multi-symbol event-driven simulator
diagnostics_report.py                # diagnostics for any generated trade log
prop_profiles.yaml                   # 50K prop rules
symbol_specs.yaml                    # micro futures specs
```

The new engine fixes the main research limitations:

```text
Backtesting.py candle-fill assumptions: conservative stop-first collision handling
Post-processed portfolio guard: replaced with one live account event loop
Possible double daily-stop logic: removed; one account daily stop only
Same-bar exits: flagged and reported
Losses larger than planned risk: planned vs actual risk breach report
Not true event-driven simulator: replaced with multi-symbol timestamp loop
```

Recommended command:

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.prop_event_engine_backtest \
  --symbols MNQ MES MYM MGC \
  --prop-profile apex_50k_eod_eval \
  --days-back 1825 \
  --timeframe 1m \
  --no-tail \
  --commission-per-contract-side 2.0 \
  --min-trend-score 3 \
  --target-r 10.0 \
  --min-planned-target-dollars 500
```

Diagnostics:

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.diagnostics_report \
  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv
```


---

## v10 update: Apex 50K evaluation simulator

`apex_eval_simulator.py` reads the event-engine trade log and simulates prop-firm evaluation pass/fail rules.

It answers:

```text
Did the account pass the $3,000 profit target?
Did it fail the $2,000 drawdown first?
How many days to pass/fail?
Worst drawdown before pass/fail
Sequential attempt pass rate
```

### Single evaluation attempt

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.apex_eval_simulator \
  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv \
  --prop-profile apex_50k_eod_eval \
  --mode single
```

### Sequential 30-day attempts

This splits the 5-year trade log into repeated eval attempts. Each attempt starts where the prior attempt ended.

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.apex_eval_simulator \
  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv \
  --prop-profile apex_50k_eod_eval \
  --mode sequential \
  --max-calendar-days 30
```

### Outputs

```text
apex_eval_single_result.csv
apex_eval_attempts.csv
apex_eval_daily_curve.csv
apex_eval_summary.txt
```


---

## v11 update: Eval mode + Performance Account mode + news blackout

v11 splits the prop journey into two modes:

```text
1. Evaluation Mode
   Goal: pass the $3,000 target once.

2. Performance Account Mode
   Goal: trade after passing.
   No profit target.
   Same daily loss/drawdown rules.
   News blackout: 5 minutes before and 5 minutes after high-impact events.
```

### Profiles

```text
apex_50k_eod_eval
apex_50k_pa
```

### News file

```text
news_events.csv
```

Format:

```csv
event_time_et,event_name,currency,impact
2025-06-11 08:30:00,CPI,USD,high
2025-06-18 14:00:00,FOMC,USD,high
```

### Run eval event engine

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.prop_event_engine_backtest \
  --symbols MNQ MES MYM MGC \
  --prop-profile apex_50k_eod_eval \
  --days-back 1825 \
  --timeframe 1m \
  --no-tail \
  --commission-per-contract-side 2.0 \
  --min-trend-score 3 \
  --target-r 10.0 \
  --min-planned-target-dollars 500
```

### Run eval pass/fail

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.apex_eval_simulator \
  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv \
  --prop-profile apex_50k_eod_eval \
  --mode single
```

### Find eval-to-PA transition point

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.eval_to_pa_transition \
  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv
```

### Run PA event engine with news blackout

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.prop_event_engine_backtest \
  --symbols MNQ MES MYM MGC \
  --prop-profile apex_50k_pa \
  --days-back 1825 \
  --timeframe 1m \
  --no-tail \
  --commission-per-contract-side 2.0 \
  --min-trend-score 3 \
  --target-r 10.0 \
  --min-planned-target-dollars 500 \
  --news-events src/strategies/manual/researched_prop_trend/news_events.csv
```

### Run PA survival report

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.performance_account_simulator \
  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv \
  --prop-profile apex_50k_pa
```

### Run PA from the post-eval point

Use the `Next PA start index` from `eval_to_pa_transition.py`:

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.performance_account_simulator \
  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv \
  --prop-profile apex_50k_pa \
  --start-index <NEXT_PA_START_INDEX>
```

---

## v11b hotfix

Fixes profile loading when `apex_50k_pa` has:

```yaml
profit_target: null
```

The event engine can now load both eval and PA profiles without crashing.

---

## v11c hotfix: DLL pause, not failure

Apex EOD evaluation Daily Loss Limit is not an account-fail condition.

The eval simulator now treats DLL as:

```text
DLL hit:
  record DLL pause day
  trading should stop for that session
  continue evaluation next session
```

Fail conditions are now focused on:

```text
EOD drawdown threshold breach
time expiry / incomplete if max calendar days ends
```

The simulator output now includes:

```text
dll_hit_days
dll_hit_day_list
Attempts with DLL pause days
Total DLL pause days
```

---

## v11d hotfix: event engine news CLI enabled

This fixes `prop_event_engine_backtest.py --help` not showing `--news-events`.

Expected help args now include:

```text
--news-events
--enable-news-blackout
--disable-news-blackout
--news-minutes-before
--news-minutes-after
--flatten-before-news
--risk-per-trade
```

The included `news_events.csv` is the combined 2021-2026 Forex Factory file.

