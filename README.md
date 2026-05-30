# Prop Dual Mode Lifecycle Simulator V11 First Payout Buffer

Adds first-payout buffer mode:

- Build first PA payout to target balance, e.g. $53,000
- Take only first payout amount, e.g. $1,000
- Leaves retained cushion, e.g. $2,000
- Later payouts only protect the retained cushion, not rebuild the full first buffer

New flags:
--enable-first-payout-buffer-mode
--first-payout-build-balance 53000
--first-payout-amount 1000
--retained-cushion-after-payout 2000

Recommended test:
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.prop_dual_mode_lifecycle_simulator \
  --eval-trade-log src/backtesting/event_engine/outputs/ict_fractal_v2_eval_365_event_trade_log.csv \
  --pa-trade-log src/backtesting/event_engine/outputs/ict_fractal_v2_quality_be_partial_no_mym_long_event_trade_log.csv \
  --profile apex_50k_dual_mode \
  --eval-risk-multiplier 2.5 \
  --pa-risk-multiplier 3.0 \
  --target-payout-frequency-days 7 \
  --pa-min-payout-trading-days 5 \
  --pa-consistency-rule-pct 50 \
  --pa-min-payout-amount 250 \
  --pa-max-payout-amount 2000 \
  --enable-first-payout-buffer-mode \
  --first-payout-build-balance 53000 \
  --first-payout-amount 1000 \
  --retained-cushion-after-payout 2000 \
  --disable-consistency-repair \
  --disable-pa-volatility-smoother \
  --aggressive-eval-until-pass
