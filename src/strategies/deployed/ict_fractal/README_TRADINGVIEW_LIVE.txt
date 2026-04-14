ICT_FRACTAL TRADINGVIEW LIVE-CANDLE RUNNER

Replace these files in:
project-x/src/strategies/deployed/ict_fractal/
- app.py
- live_model.py
- config.py

These files can stay as-is, but are included for completeness:
- state.py
- execution.py

What this changes:
- Live signal generation now prefers TradingView candle files from:
  src/data/tradingview_bars/
- Default symbol map:
  NQ -> MNQ1!
  MES -> MES1!
  MYM -> MYM1!
  MGC -> MGC1!
- If TradingView candles are missing or too short, it can fall back to the normal fetcher.

Important .env values:
ICT_FRACTAL_EXECUTION_MODE=demo
ICT_FRACTAL_USE_TRADINGVIEW_BARS=1
ICT_FRACTAL_TV_FALLBACK_TO_FETCHER=1
ICT_FRACTAL_TV_MIN_BARS=200
ICT_FRACTAL_TRADINGVIEW_SYMBOL_MAP=NQ:MNQ1!,MES:MES1!,MYM:MYM1!,MGC:MGC1!
ICT_FRACTAL_LOOP_SECONDS=20
ICT_FRACTAL_DEFAULT_QTY=1
ICT_FRACTAL_SYMBOLS=NQ,MES,MYM,MGC
ICT_FRACTAL_FORCE_FLAT_HOUR_ET=16
ICT_FRACTAL_FORCE_FLAT_MINUTE_ET=50
ICT_FRACTAL_GLOBEX_REOPEN_HOUR_ET=18
ICT_FRACTAL_GLOBEX_REOPEN_MINUTE_ET=0
ICT_FRACTAL_FORCE_FLAT_ENABLED=1
ICT_FRACTAL_SIGNAL_LOOKBACK_BARS=1

Run:
python src/strategies/deployed/ict_fractal/app.py

Expected behavior:
- Webhook fills src/data/tradingview_bars/*.json
- Runner uses TradingView candles instead of Databento/fetcher by default
- It still remains session-aware for force-flat and reopen windows
- Signals are still deduped by signal_id in runtime_state.json
