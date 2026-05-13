# Prop Lifecycle Payout Simulator

Adds a payout-aware lifecycle simulator for Algotec.

Flow:

```text
Eval pass
→ PA account starts
→ PA uses daily profit lock / daily soft loss controls
→ payout eligibility is checked at EOD
→ approved payout is withdrawn
→ PA blows only if max-loss/EOD drawdown is hit
→ after PA blow or max payout cycle, start new eval
```

Copy these into:

```text
src/strategies/manual/researched_prop_trend/
```

Files:

```text
prop_lifecycle_payout_simulator.py
prop_firm_lifecycle_profiles.yaml
```

Run default Apex 50K EOD lifecycle:

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.prop_lifecycle_payout_simulator \
  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv \
  --lifecycle-profile apex_50k_eod_lifecycle
```

Run safer PA mode:

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.prop_lifecycle_payout_simulator \
  --trade-log src/strategies/manual/researched_prop_trend/event_trade_log.csv \
  --lifecycle-profile apex_50k_eod_lifecycle_safe
```

Outputs:

```text
prop_lifecycle_payout_summary.txt
prop_lifecycle_payout_cycles.csv
prop_lifecycle_payout_events.csv
prop_lifecycle_payout_daily.csv
prop_lifecycle_payouts.csv
```

Edit `prop_firm_lifecycle_profiles.yaml` to test other prop firms or PA rules.
