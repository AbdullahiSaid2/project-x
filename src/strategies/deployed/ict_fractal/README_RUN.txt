ICT_FRACTAL EXACT V473 FORWARD-TEST RUNNER

What this is:
- one command
- exact V473 backtest model scan on every loop
- pulls fresh market data
- reruns the real V473 strategy logic
- extracts fresh entry signals from the newest bar window
- sends them to paper mode or PickMyTrade

Place these files here:
project-x/src/strategies/deployed/ict_fractal/

This runner expects these existing repo files to already exist:
- src/strategies/manual/v473_shared.py
- src/strategies/manual/ict_multi_setup_v452.py
- src/data/fetcher.py
- your normal data dependencies already used by backtests

Required .env values for live demo execution:
ICT_FRACTAL_EXECUTION_MODE=demo
PICKMYTRADE_TOKEN=your_token_here
PICKMYTRADE_ACCOUNT_ID=your_account_id_here
PICKMYTRADE_STRATEGY_ID=ict_fractal

Recommended:
ICT_FRACTAL_LOOP_SECONDS=20
ICT_FRACTAL_DEFAULT_QTY=1
ICT_FRACTAL_SYMBOLS=NQ,MES,MYM,MGC
ICT_FRACTAL_FORCE_FLAT_HOUR_ET=16
ICT_FRACTAL_FORCE_FLAT_MINUTE_ET=50
ICT_FRACTAL_SIGNAL_LOOKBACK_BARS=1

One command to run:
python src/strategies/deployed/ict_fractal/app.py

How this behaves:
- before every cycle it checks ET time
- after the force-flat cutoff it sends no new orders
- before cutoff it reruns V473 exactly from the manual strategy module
- new signals are deduped by signal_id so the same entry is not resent every cycle

Important:
- this is the exact V473 signal-generation path from your backtest module, not a simplified imitation model
- it is heavier than a lightweight live model because it reruns the strategy each cycle
- execution is still your PickMyTrade order adapter, so order placement is live/demo while signal generation is exact V473
