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
ICT_FRACTAL_GLOBEX_REOPEN_HOUR_ET=18
ICT_FRACTAL_GLOBEX_REOPEN_MINUTE_ET=0
ICT_FRACTAL_FORCE_FLAT_ENABLED=1
ICT_FRACTAL_SIGNAL_LOOKBACK_BARS=1

Optional for a real force-flat API call:
PICKMYTRADE_FORCE_FLAT_URL=/api/v1/positions/flatten

One command to run:
python src/strategies/deployed/ict_fractal/app.py

How this behaves now:
- the runner is session-aware, not calendar-day cutoff based
- entry allowed from 18:00 ET to 16:49:59 ET next day
- no new entries from 16:50 ET to 17:59:59 ET
- Friday after 16:50 ET stays blocked until Sunday 18:00 ET
- at the force-flat window the runner attempts one flatten action per session window
- new signals are deduped by signal_id so the same entry is not resent every cycle

Important:
- this is the exact V473 signal-generation path from your backtest module, not a simplified imitation model
- it is heavier than a lightweight live model because it reruns the strategy each cycle
- force-flat execution for demo/live requires a working flatten endpoint if you want the app to actually close positions via PickMyTrade
- if PICKMYTRADE_FORCE_FLAT_URL is not configured, the runner will still block entries correctly and log that force-flat was skipped
