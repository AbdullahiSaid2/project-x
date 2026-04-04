#!/usr/bin/env python3
# ============================================================
# 🌙 Vault Forward Tester
#
# Runs your vaulted strategies on live market data and
# generates trade signals. Three modes:
#
#   notify  — prints signals, logs to CSV, no trades placed
#   manual  — shows on dashboard, you click Approve/Reject
#   auto    — executes trades automatically (use with care)
#
# USAGE:
#   python src/agents/vault_forward_test.py                  # notify mode
#   python src/agents/vault_forward_test.py --mode manual    # dashboard approval
#   python src/agents/vault_forward_test.py --mode auto      # auto execute
#   python src/agents/vault_forward_test.py --once           # single scan, no loop
#   python src/agents/vault_forward_test.py --market crypto  # crypto only
#   python src/agents/vault_forward_test.py --market futures # futures only
# ============================================================

import sys, json, time, csv, importlib.util, traceback, argparse
import numpy as np
import pandas as pd
import ta as ta_lib
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT   = Path(__file__).resolve().parents[2]
VAULT_DIR   = REPO_ROOT / "src" / "strategies" / "vault"
VAULT_INDEX = VAULT_DIR / "vault_index.json"
DATA_DIR    = REPO_ROOT / "src" / "data"
SIGNAL_LOG  = DATA_DIR / "forward_test_log.csv"
PENDING     = DATA_DIR / "pending_signals.json"

FUTURES_SYMS    = {"MES", "MNQ", "MYM"}    # Tradovate via Apex/PickMyTrade
RWA_PERP_SYMS   = {"SPX", "NDX", "OIL", "GOLD"}  # Hyperliquid via Trade[XYZ]
# SPX perp trades 24/7 on Hyperliquid — same strategies as MES but no prop rules

# Open positions tracker (in-memory for session)
OPEN_POSITIONS: list[dict] = []

# Scan interval per timeframe (how long to wait before next scan)
TF_INTERVALS = {
    "15m": 15 * 60,
    "1H":  60 * 60,
    "4H":  4  * 60 * 60,
    "1D":  24 * 60 * 60,
}

# ── Kill zone check ────────────────────────────────────────────
def in_kill_zone() -> tuple[bool, str]:
    """Returns (is_active, zone_name) based on current EST time."""
    from datetime import datetime
    import zoneinfo
    try:
        est = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    except Exception:
        est = datetime.utcnow()   # fallback
    hm = est.hour * 60 + est.minute

    zones = [
        (120,  300,  "London KZ"),
        (510,  660,  "NY AM"),
        (590,  610,  "Macro 09:50"),
        (650,  670,  "Macro 10:50"),
        (810,  960,  "NY PM"),
        (830,  850,  "Macro 13:50"),
        (1200, 1440, "Asian KZ"),
    ]
    for start, end, name in zones:
        if start <= hm < end:
            return True, name
    return False, ""


# ── Load vaulted strategies ─────────────────────────────────────
def load_vault(market: str = "all") -> list[dict]:
    if not VAULT_INDEX.exists():
        print("❌ No vault index found. Run the backtester first.")
        return []
    strategies = json.loads(VAULT_INDEX.read_text()).get("strategies", [])

    # Filter by market
    if market == "crypto":
        strategies = [s for s in strategies
                      if s["symbol"] not in FUTURES_SYMS
                      and s["symbol"] not in RWA_PERP_SYMS]
    elif market == "futures":
        strategies = [s for s in strategies if s["symbol"] in FUTURES_SYMS]

    # ── Sort by priority: best strategy executes first ─────────
    # Priority order:
    #   1. Expected value per trade (highest EV first)
    #      EV = (win_rate × risk × R:R) - ((1-win_rate) × risk)
    #   2. Sharpe ratio as tiebreaker
    #
    # Why this matters:
    #   • Risk manager allows MAX_CONCURRENT_FUTURES = 1
    #   • If two futures signals fire at the same time, only
    #     the FIRST one executes — so best strategy goes first
    #   • For crypto (MAX_CONCURRENT = 3), top 3 by EV execute
    def _priority_score(s: dict) -> float:
        wr   = s.get("win_rate", 50) / 100
        rr   = 2.0      # minimum R:R enforced by RBI
        # Relative risk (futures get higher risk per trade)
        risk = 500 if s.get("symbol") in FUTURES_SYMS else 10
        ev   = (wr * risk * rr) - ((1 - wr) * risk)
        # Combine EV with Sharpe (80% EV, 20% Sharpe)
        sharpe = s.get("sharpe", 1.0)
        return ev * 0.80 + sharpe * 100 * 0.20   # scale Sharpe to match EV

    strategies.sort(key=_priority_score, reverse=True)

    return strategies


# ── Load strategy class from vault file ────────────────────────
def load_strategy_class(vault_entry: dict):
    """
    Dynamically import VaultStrategy class from vault .py file.
    Returns (VaultStrategy class, error_string).
    """
    fname = vault_entry.get("vault_file", "")
    if not fname:
        return None, "No vault_file in index"

    fpath = VAULT_DIR / fname
    if not fpath.exists():
        return None, f"File not found: {fpath.name}"

    try:
        spec   = importlib.util.spec_from_file_location("vault_strat", fpath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Try VaultStrategy first (new format)
        cls = getattr(module, "VaultStrategy", None)

        # Fall back: find any Strategy subclass in the module
        if cls is None:
            from backtesting import Strategy
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                try:
                    if (isinstance(attr, type) and
                            issubclass(attr, Strategy) and
                            attr is not Strategy):
                        cls = attr
                        break
                except TypeError:
                    continue

        if cls is None:
            return None, "No Strategy subclass found in file"
        return cls, None
    except Exception as e:
        return None, str(e)[:100]


# ── Get live signal from strategy ──────────────────────────────
def get_signal(strategy: dict, verbose: bool = True) -> dict | None:
    """
    Runs the vault strategy on recent live data.
    Returns a signal dict if entry conditions are met, else None.
    """
    name      = strategy["name"]
    symbol    = strategy["symbol"]
    timeframe = strategy["timeframe"]

    if verbose:
        print(f"  Scanning {name} | {symbol} {timeframe}...", end=" ", flush=True)

    # Load data
    try:
        from src.data.fetcher import get_ohlcv
        from src.config import EXCHANGE
        exch = "hyperliquid" if symbol not in FUTURES_SYMS else "tradovate"
        df   = get_ohlcv(symbol, exchange=exch,
                         timeframe=timeframe, days_back=365)
        if df is None or len(df) < 50:
            if verbose: print("⚠️  Not enough data")
            return None
    except Exception as e:
        if verbose: print(f"⚠️  Data error: {str(e)[:60]}")
        return None

    # Format for backtesting.py
    ohlcv = pd.DataFrame({
        "Open":   df["Open"].astype(float).values,
        "High":   df["High"].astype(float).values,
        "Low":    df["Low"].astype(float).values,
        "Close":  df["Close"].astype(float).values,
        "Volume": df["Volume"].astype(float).values,
    }, index=df.index)

    # Load strategy class
    cls, err = load_strategy_class(strategy)
    if cls is None:
        if verbose: print(f"⚠️  {err}")
        return None

    # Run backtest on full data — get last trade signal
    try:
        from backtesting import Backtest
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bt    = Backtest(ohlcv, cls, cash=100_000,
                             commission=0.001, exclusive_orders=True)
            stats = bt.run()

        trades = stats._trades
        if trades is None or len(trades) == 0:
            if verbose: print("○  No recent signal")
            return None

        # Check if the most recent trade entry is on the LAST bar
        last_bar_time = ohlcv.index[-1]
        last_trade    = trades.iloc[-1]

        entry_time = pd.to_datetime(last_trade.get("EntryTime",
                                    last_trade.get("Entry Time", None)))
        if entry_time is None:
            if verbose: print("○  No signal")
            return None

        # Signal is valid if entry was on the last or second-to-last bar
        bars_ago = len(ohlcv) - ohlcv.index.get_loc(
            ohlcv.index[ohlcv.index <= entry_time][-1]
        ) - 1

        if bars_ago > 1:
            if verbose:
                # Convert bars to human time
                tf_mins = {"15m":15,"1H":60,"4H":240,"1D":1440}.get(timeframe,60)
                hours_ago = bars_ago * tf_mins / 60
                if hours_ago < 24:
                    time_str = f"{hours_ago:.0f}h ago"
                else:
                    time_str = f"{hours_ago/24:.1f}d ago"
                direction_hint = "LONG" if last_trade.get("Size",0)>0 else "SHORT"
                print(f"○  In {direction_hint} from {bars_ago} bars ago ({time_str}) — no new entry")
            return None

        # Build signal
        direction = "LONG"  if last_trade.get("Size", 0) > 0 else "SHORT"
        entry     = float(ohlcv["Close"].iloc[-1])
        sl        = float(last_trade.get("SL",  entry * (0.97 if direction=="LONG" else 1.03)))
        tp        = float(last_trade.get("TP",  entry * (1.06 if direction=="LONG" else 0.94)))
        rr        = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0

        in_kz, kz_name = in_kill_zone()

        sig = {
            "strategy":  name,
            "symbol":    symbol,
            "timeframe": timeframe,
            "direction": direction,
            "entry":     round(entry, 4),
            "sl":        round(sl, 4),
            "tp":        round(tp, 4),
            "rr":        round(rr, 2),
            "sharpe":    strategy.get("sharpe", 0),
            "win_rate":  strategy.get("win_rate", 0),
            "in_kill_zone": in_kz,
            "kill_zone": kz_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status":    "PENDING",
        }

        kz_tag = f" [{kz_name}]" if in_kz else ""
        if verbose:
            col = "🟢" if direction == "LONG" else "🔴"
            print(f"{col} {direction} @ {entry:.4f} | "
                  f"SL {sl:.4f} TP {tp:.4f} | R:R {rr:.1f}{kz_tag}")

        return sig

    except Exception as e:
        if verbose:
            print(f"⚠️  Error: {str(e)[:80]}")
        return None


# ── Log signal to CSV ───────────────────────────────────────────
def log_signal(sig: dict):
    exists = SIGNAL_LOG.exists()
    with open(SIGNAL_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp","strategy","symbol","timeframe","direction",
            "entry","sl","tp","rr","sharpe","win_rate",
            "in_kill_zone","kill_zone","status","outcome","pnl_pct"
        ])
        if not exists:
            writer.writeheader()
        writer.writerow({k: sig.get(k, "") for k in writer.fieldnames})


# ── Add to pending signals (for dashboard manual approval) ─────
def add_pending(sig: dict):
    import hashlib
    sig_id = hashlib.md5(
        f"{sig['strategy']}{sig['symbol']}{sig['timestamp']}".encode()
    ).hexdigest()[:8]
    sig["id"] = sig_id

    pending = []
    if PENDING.exists():
        try:
            pending = json.loads(PENDING.read_text())
        except Exception:
            pending = []

    # Remove expired (>30 min old)
    now     = datetime.now(timezone.utc)
    pending = [p for p in pending
               if (now - datetime.fromisoformat(p["timestamp"])
                   .replace(tzinfo=timezone.utc)).seconds < 1800]
    pending.append(sig)
    PENDING.write_text(json.dumps(pending, indent=2))
    return sig_id


# ── Print signal summary ────────────────────────────────────────
def print_signal(sig: dict):
    dir_icon = "🟢 LONG" if sig["direction"] == "LONG" else "🔴 SHORT"
    kz_line  = f"   Kill Zone : {sig['kill_zone']} ✅" if sig["in_kill_zone"] else \
                "   Kill Zone : Outside ⚠️"
    print(f"""
  ┌─────────────────────────────────────────────┐
  │  🔔 SIGNAL — {sig['strategy']} {sig['symbol']} {sig['timeframe']}
  │  Direction  : {dir_icon}
  │  Entry      : {sig['entry']}
  │  Stop Loss  : {sig['sl']}  ({abs(sig['sl']-sig['entry'])/sig['entry']*100:.1f}%)
  │  Take Profit: {sig['tp']}  (+{abs(sig['tp']-sig['entry'])/sig['entry']*100:.1f}%)
  │  R:R        : {sig['rr']:.1f}
  │  Sharpe     : {sig['sharpe']} | Win Rate: {sig['win_rate']}%
{kz_line}
  └─────────────────────────────────────────────┘""")


# ── Main scan loop ──────────────────────────────────────────────
def run(mode: str = "notify", market: str = "all",
        once: bool = False):
    strategies = load_vault(market)
    if not strategies:
        print("No strategies in vault. Run the backtester first.")
        return

    print(f"""
🌙 Vault Forward Tester
{'='*50}
  Mode       : {mode.upper()}
  Market     : {market}
  Strategies : {len(strategies)}
  Log file   : {SIGNAL_LOG.name}
{'='*50}
Strategies loaded:""")
    for s in strategies:
        print(f"  • {s['name']:<28} {s['symbol']} {s['timeframe']}"
              f"  Sharpe={s.get('sharpe',0):.2f}")

    scan_count = 0
    while True:
        scan_count += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'─'*50}")
        print(f"  Scan #{scan_count} — {now}")

        in_kz, kz_name = in_kill_zone()
        if in_kz:
            print(f"  ⏰ Active kill zone: {kz_name}")
        else:
            print(f"  ⏰ Outside kill zones (signals still tracked)")

        # Show daily P&L
        try:
            from src.agents.risk_manager import load_daily_pnl
            from src.config import TRADOVATE_SIM
            pnl = load_daily_pnl()
            exch_tag = "Tradovate SIM" if TRADOVATE_SIM else "Tradovate LIVE"
            print(f"  💰 Today: ${pnl['total_pnl']:+.2f} | "
                  f"{pnl['trades']} trades | "
                  f"{'🚫 KILL SWITCH' if pnl.get('kill_switch') else '✅ Trading OK'}")
        except Exception:
            pass

        # ── Scan strategies in parallel threads ─────────────
        # Each strategy runs independently — a slow data fetch
        # on one strategy doesn't delay others from firing.
        import concurrent.futures
        signals_found = 0
        raw_signals = {}

        def _scan_one(strat):
            try:
                return strat["name"], get_signal(strat, verbose=False)
            except Exception as e:
                return strat["name"], None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_scan_one, s): s for s in strategies}
            for future in concurrent.futures.as_completed(futures):
                name, sig = future.result()
                raw_signals[name] = sig
                if sig:
                    sym = sig.get("symbol","")
                    tf  = sig.get("timeframe","")
                    print(f"  🔔 Signal: {name} {sym} {tf}")

        # Process signals in vault order (deterministic)
        for strategy in strategies:
            sig = raw_signals.get(strategy["name"])
            if sig is None:
                continue

            signals_found += 1

            # ── Risk check + position sizing ──────────────
            try:
                from src.agents.risk_manager import (
                    check_risk, calculate_position_size,
                    print_risk_summary, log_risk_event
                )
                from src.config import (TRADOVATE_ACCOUNT_SIZE,
                                        HYPERLIQUID_ACCOUNT_SIZE)
                is_fut   = sig["symbol"] in FUTURES_SYMS
                account  = TRADOVATE_ACCOUNT_SIZE if is_fut else HYPERLIQUID_ACCOUNT_SIZE
                sizing   = calculate_position_size(sig, account)
                approved, reason = check_risk(sig, OPEN_POSITIONS)
                sig["size_usd"]   = sizing.get("size_usd", 0)
                sig["risk_usd"]   = sizing.get("risk_usd", 0)
                sig["risk_pct"]   = sizing.get("risk_pct", 0)
                sig["contracts"]  = sizing.get("contracts", None)
            except Exception as e:
                approved, reason = True, f"Risk check skipped: {e}"
                sizing = {}

            print_signal(sig)
            print_risk_summary(sig, sizing, (approved, reason))

            if not approved:
                log_risk_event("BLOCKED", reason, sig)
                sig["status"] = "BLOCKED"
                log_signal(sig)
                continue

            log_signal(sig)

            if mode == "notify":
                print("  📋 Logged — place trade manually if desired")

            elif mode == "manual":
                sig_id = add_pending(sig)
                print(f"  👆 Added to dashboard — approve/reject at http://algotectrading")
                print(f"     Signal ID: {sig_id}")

            elif mode == "auto":
                sym      = sig["symbol"].upper()
                is_fut   = sym in FUTURES_SYMS
                is_rwa   = sym in RWA_PERP_SYMS
                exchname = ("Tradovate SIM" if is_fut
                            else "Hyperliquid RWA (SPX)" if is_rwa
                            else "Hyperliquid")
                # Check caution list from weekly brief
                if not is_fut:
                    try:
                        from src.agents.weekly_briefing import get_caution_tokens
                        if sig["symbol"] in get_caution_tokens():
                            print(f"  ⚠️  {sig['symbol']} flagged cautious in weekly brief — skipping")
                            sig["status"] = "SKIPPED_CAUTION"
                            log_signal(sig)
                            continue
                    except Exception:
                        pass
                print(f"  ⚡ Auto mode — executing on {exchname}...")
                try:
                    from src.agents.apex_bridge import execute_signal
                    result = execute_signal(sig)
                    if result.get("error"):
                        print(f"  ❌ {result['error']}")
                        sig["status"] = "FAILED"
                    else:
                        print(f"  ✅ Executed via bridge")
                        sig["status"] = "EXECUTED"
                        OPEN_POSITIONS.append(sig)
                except Exception as e:
                    print(f"  ❌ Execution failed: {e}")
                    sig["status"] = "FAILED"
                log_signal(sig)

        if signals_found == 0:
            print("  ○  No signals this scan")

        if once:
            print(f"\n✅ Single scan complete. {signals_found} signal(s) found.")
            break

        # Auto-regenerate handoff every 6 scan cycles (~1.5 hrs)
        if scan_count % 6 == 0:
            try:
                import subprocess as _sp
                _sp.Popen(
                    ["python3", "src/agents/handoff_generator.py"],
                    cwd=str(Path(__file__).resolve().parents[2]),
                    stdout=_sp.DEVNULL, stderr=_sp.DEVNULL
                )
                print("  📄 Handoff auto-updated")
            except Exception:
                pass   # non-critical — never block trading for this

        # Wait for next scan — 15m candle = 15m wait
        tfs  = [s["timeframe"] for s in strategies]
        wait = min(TF_INTERVALS.get(tf, 900) for tf in tfs)
        wait = min(wait, 900)  # cap at 15 min

        print(f"\n  💤 Next scan in {wait//60}min — ", end="", flush=True)
        try:
            for remaining in range(wait, 0, -30):
                mins, secs = divmod(remaining, 60)
                print(f"\r  💤 Next scan in {mins}m{secs:02d}s — Ctrl+C to stop   ",
                      end="", flush=True)
                time.sleep(min(30, remaining))
            print()  # newline before next scan
        except KeyboardInterrupt:
            print("\n\n  ⏹  Stopped by user")
            break


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="🌙 Vault Forward Tester — live signals from vaulted strategies"
    )
    p.add_argument("--mode",   default="notify",
                   choices=["notify","manual","auto"],
                   help="notify=log only | manual=dashboard approval | auto=execute")
    p.add_argument("--market", default="all",
                   choices=["all","crypto","futures"],
                   help="Filter by market")
    p.add_argument("--once",   action="store_true",
                   help="Single scan then exit (good for cron)")
    args = p.parse_args()
    run(mode=args.mode, market=args.market, once=args.once)