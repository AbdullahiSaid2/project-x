# Top Bottom Ticking — Separated Normal and AXL Versions

This package separates the original `top_bottom_ticking` model from the AXL-enhanced version so you can run, compare, and event-replay them independently.

## Files

```text
src/strategies/manual/ict_top_bottom_ticking.py          # ORIGINAL strategy logic
src/strategies/manual/top_bottom_ticking_shared.py       # ORIGINAL runner/harness

src/strategies/manual/ict_top_bottom_ticking_axl.py      # AXL strategy logic
src/strategies/manual/top_bottom_ticking_shared_axl.py   # AXL runner/harness

src/strategies/manual/top_bottom_ticking_event_engine.py # Event-driven Eval/PA replay engine
```

The original files are kept under their existing names. The AXL files use `_axl` names so they do not overwrite or mix with the normal model.

## Install

From your repo root:

```bash
cd /Users/Abdullahi/trading-project/trading_system
unzip ~/Downloads/top_bottom_ticking_separated_normal_axl.zip -d .
```

Optional backup first:

```bash
cp src/strategies/manual/ict_top_bottom_ticking.py src/strategies/manual/ict_top_bottom_ticking.py.bak
cp src/strategies/manual/top_bottom_ticking_shared.py src/strategies/manual/top_bottom_ticking_shared.py.bak
```

## Run the normal/original model

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared \
  --prop-profile apex_50k_eval \
  --days-back 365 \
  --no-tail
```

Outputs use the original names, for example:

```text
src/strategies/manual/top_bottom_ticking_trade_log_apex_50k_eval_365d_notail.csv
src/strategies/manual/top_bottom_ticking_debug_counts_apex_50k_eval_365d_notail.csv
```

## Run the AXL model

Log-only mode, to tag trades without blocking:

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared_axl \
  --prop-profile apex_50k_eval \
  --days-back 365 \
  --no-tail \
  --axl-mode log_only
```

Filter mode, to only allow AXL-approved trades:

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared_axl \
  --prop-profile apex_50k_eval \
  --days-back 365 \
  --no-tail \
  --axl-mode filter
```

Strict filter mode:

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared_axl \
  --prop-profile apex_50k_eval \
  --days-back 365 \
  --no-tail \
  --axl-mode filter \
  --axl-require-internal \
  --axl-require-htf-anchor \
  --axl-require-mtf-poi
```

AXL outputs use separate names, for example:

```text
src/strategies/manual/top_bottom_ticking_axl_trade_log_apex_50k_eval_365d_notail_axlfilter.csv
src/strategies/manual/top_bottom_ticking_axl_debug_counts_apex_50k_eval_365d_notail_axlfilter.csv
src/strategies/manual/top_bottom_ticking_axl_grade_summary_apex_50k_eval_365d_notail_axlfilter.csv
```

## Run the event-driven Eval/PA engine on either version

Normal model:

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_event_engine \
  --input src/strategies/manual/top_bottom_ticking_trade_log_apex_50k_eval_365d_notail.csv \
  --preset apex_50k_eval_pa \
  --mode replay \
  --out-dir src/strategies/manual/event_reports_normal
```

AXL model:

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_event_engine \
  --input src/strategies/manual/top_bottom_ticking_axl_trade_log_apex_50k_eval_365d_notail_axlfilter.csv \
  --preset apex_50k_eval_pa \
  --mode replay \
  --out-dir src/strategies/manual/event_reports_axl
```

Monte Carlo AXL Eval/PA test:

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_event_engine \
  --input src/strategies/manual/top_bottom_ticking_axl_trade_log_apex_50k_eval_365d_notail_axlfilter.csv \
  --preset apex_50k_conservative \
  --mode monte_carlo \
  --iterations 500 \
  --seed 42 \
  --out-dir src/strategies/manual/event_reports_axl
```

## Why this separation matters

- The normal model remains untouched and can be used as the baseline.
- The AXL model can evolve separately without breaking the baseline.
- The event engine can replay either trade log as one portfolio account.
- Reports are separated so results do not get mixed.
