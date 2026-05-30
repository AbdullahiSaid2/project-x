# Top Bottom Ticking Event-Driven Eval/PA Engine

This package adds a portfolio-level event replay and Eval/PA simulator for the `top_bottom_ticking` model.

## Why this is needed

The existing `top_bottom_ticking_shared.py` runner uses Backtesting.py per symbol/variant, then combines the reports afterwards. That is useful for raw strategy testing, but it is not enough for prop-firm simulation because one Apex-style account must see all symbols and variants together.

This engine replays the combined trade log chronologically and applies account-level rules across the whole portfolio:

- daily loss lockout
- hard daily-loss failure
- max account loss failure
- evaluation profit target
- minimum trading days
- PA transition after eval pass
- PA max loss / daily loss failure
- max open trades
- max trades per day
- max daily open risk
- optional one-trade-per-bar rule
- Monte Carlo by day for robustness testing

## Install

From your repo root:

```bash
cd /Users/Abdullahi/trading-project/trading_system
unzip ~/Downloads/top_bottom_ticking_event_engine_package.zip -d .
```

The file should land here:

```text
src/strategies/manual/top_bottom_ticking_event_engine.py
```

## Step 1: Generate the AXL-filter trade log

Run the AXL-filter model first:

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared \
  --prop-profile apex_50k_eval \
  --days-back 365 \
  --no-tail \
  --axl-mode filter
```

This should create something like:

```text
src/strategies/manual/top_bottom_ticking_trade_log_apex_50k_eval_365d_notail_axlfilter.csv
```

## Step 2: Replay it through one combined Eval/PA account

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_event_engine \
  --input src/strategies/manual/top_bottom_ticking_trade_log_apex_50k_eval_365d_notail_axlfilter.csv \
  --preset apex_50k_eval_pa \
  --mode replay \
  --out-dir src/strategies/manual/event_reports
```

## Step 3: Conservative PA-style version

This is the command I would trust more for PA-style survival testing:

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_event_engine \
  --input src/strategies/manual/top_bottom_ticking_trade_log_apex_50k_eval_365d_notail_axlfilter.csv \
  --preset apex_50k_conservative \
  --mode replay \
  --max-open-trades 1 \
  --max-trades-per-day 4 \
  --max-daily-open-risk 600 \
  --daily-loss 900 \
  --out-dir src/strategies/manual/event_reports
```

## Step 4: Monte Carlo Eval/PA simulation

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_event_engine \
  --input src/strategies/manual/top_bottom_ticking_trade_log_apex_50k_eval_365d_notail_axlfilter.csv \
  --preset apex_50k_conservative \
  --mode monte_carlo \
  --iterations 500 \
  --seed 42 \
  --out-dir src/strategies/manual/event_reports
```

## Outputs

The engine writes:

```text
*_lifecycle_summary.json
*_accepted_entries.csv
*_rejected_entries.csv
*_exit_events_applied.csv
*_daily_summary.csv
*_event_audit.csv
*_monte_carlo.csv
*_monte_carlo_summary.json
```

## Important limitation

This is a trade-log replay engine. It is much better than simply summing CSV rows, because it replays entries/exits chronologically and applies portfolio-level account rules. However, it is not a true tick-level fill simulator. It cannot know exact intra-trade MAE unless the strategy outputs tick-level/1m path data or MAE/MFE fields.

So use it as the next serious layer after the AXL-filter backtest, but before live demo/PA deployment.
