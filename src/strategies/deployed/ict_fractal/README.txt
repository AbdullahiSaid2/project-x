
ICT_FRACTAL EXACT V473 EVENT-SURFACED DEPLOY

What this version does:
- keeps V473 as the only source of truth for entries, stops, targets, partials, and BE logic
- surfaces exact V473 runtime events from the manual strategy:
  - entry_opened
  - partial_taken
  - stop_moved_to_be
  - trade_closed
- forwards exact V473 entry events to PickMyTrade
- forwards exact V473 full trade_closed events as exit commands
- logs partial/BE events to monitoring without synthesizing new broker behavior

Important limitation:
- the current proven PickMyTrade transport path is confirmed for entries and exits
- it is NOT yet proven for true partial-reduce or stop-update broker actions
- therefore partial_taken and stop_moved_to_be are surfaced and monitored exactly, but not translated into broker-side actions unless/until you confirm a working PickMyTrade API for those actions

Files to replace:
- src/strategies/manual/ict_multi_setup_v452.py
- src/strategies/deployed/ict_fractal/app.py
- src/strategies/deployed/ict_fractal/live_model.py
- src/strategies/deployed/ict_fractal/execution.py
- src/strategies/deployed/ict_fractal/state.py
- src/strategies/deployed/ict_fractal/config.py
- src/strategies/deployed/ict_fractal/monitoring.py

Recommended .env:
ICT_FRACTAL_EXECUTION_MODE=demo
ICT_FRACTAL_USE_TRADINGVIEW_BARS=1
ICT_FRACTAL_TV_FALLBACK_TO_FETCHER=1
ICT_FRACTAL_TV_MIN_BARS=50
ICT_FRACTAL_TRADINGVIEW_SYMBOL_MAP=NQ:MNQ1!,MES:MES1!,MYM:MYM1!,MGC:MGC1!
ICT_FRACTAL_LOOP_SECONDS=20
ICT_FRACTAL_DEFAULT_QTY=5
ICT_FRACTAL_SYMBOLS=NQ,MES,MYM,MGC
ICT_FRACTAL_FORCE_FLAT_HOUR_ET=16
ICT_FRACTAL_FORCE_FLAT_MINUTE_ET=50
ICT_FRACTAL_GLOBEX_REOPEN_HOUR_ET=18
ICT_FRACTAL_GLOBEX_REOPEN_MINUTE_ET=0
ICT_FRACTAL_FORCE_FLAT_ENABLED=1
ICT_FRACTAL_SIGNAL_LOOKBACK_BARS=1

Logs:
- logs/execution_log.jsonl
- logs/monitor_log.jsonl
- state/runtime_state.json

Run:
python src/strategies/deployed/ict_fractal/app.py
