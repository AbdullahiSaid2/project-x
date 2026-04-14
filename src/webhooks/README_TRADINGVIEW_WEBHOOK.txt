Files included:
- server_amended.py                -> replace repo root server.py
- tradingview_webhook_amended.py  -> replace src/webhooks/tradingview_webhook.py
- tradingview_alert_example.json  -> paste into TradingView alert message box

What this adds:
- Keeps your old TradingView BUY/SELL/CLOSE alerts working.
- Adds OHLCV bar ingestion from TradingView into src/data/tradingview_bars/*.json
- Adds dedupe so duplicate bar alerts do not get stored twice.
- Adds optional PickMyTrade execution route controlled by env vars.
- Adds /webhook/bars/<symbol>/<timeframe> so you can inspect buffered bars.

Recommended .env values for your current setup:
WEBHOOK_SECRET=your_secret_here
USE_PICKMYTRADE=1
PICKMYTRADE_TOKEN=your_pickmytrade_token_here
PICKMYTRADE_ACCOUNT_ID=your_account_id_if_needed
TRADINGVIEW_EXECUTE_ON_BAR_ALERT=0

Important:
- TRADINGVIEW_EXECUTE_ON_BAR_ALERT=0 means bar alerts are only stored, not traded automatically.
- Leave that at 0 first.
- Use the stored OHLCV bars to feed your model next.
- When ready, either:
  1) send direct action alerts from TradingView, or
  2) switch TRADINGVIEW_EXECUTE_ON_BAR_ALERT=1 and include an action field in the payload.

Useful endpoints:
- POST /webhook/tradingview
- GET  /webhook/status
- GET  /webhook/log
- GET  /webhook/bars/NQ/1

TradingView setup:
- Alert type: Once Per Bar Close
- Webhook URL: https://YOUR_DOMAIN/webhook/tradingview
- Message: contents of tradingview_alert_example.json

Note:
This does not yet run ict_fractal automatically from the incoming bars. It creates the correct webhook entrypoint and persistent OHLCV bar store so the next step is easy and clean.
