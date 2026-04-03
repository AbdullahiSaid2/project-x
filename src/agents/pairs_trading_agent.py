#!/usr/bin/env python3
# ============================================================
# 🌙 Pairs Trading Agent — Statistical Arbitrage
#
# The core RenTec strategy: exploit temporary price divergences
# between correlated assets. When two historically correlated
# tokens diverge beyond 2 standard deviations, bet on reversion.
#
# STRATEGY:
#   1. Find cointegrated crypto pairs (move together long-term)
#   2. Calculate rolling Z-score of their price ratio
#   3. When Z-score > 2: short leader, long lagger
#   4. When Z-score < -2: long leader, short lagger
#   5. Close when Z-score returns to 0
#
# This is MARKET NEUTRAL — profits regardless of market direction.
# Works in bull, bear, and sideways markets.
#
# ⚠️  PROP FIRM NOTE:
#   Simultaneous long/short on correlated assets may violate
#   prop firm rules (especially "no hedging" clauses).
#   Use on PERSONAL Hyperliquid account only.
#   Set PAIRS_TRADING_ENABLED = False in config when on prop eval.
#
# HOW TO RUN:
#   python src/agents/pairs_trading_agent.py --scan    # find pairs
#   python src/agents/pairs_trading_agent.py --live    # run live
#   python src/agents/pairs_trading_agent.py --backtest  # backtest
# ============================================================

import sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "src" / "data"
PAIRS_FILE = DATA_DIR / "pairs_state.json"
LOG_FILE   = DATA_DIR / "pairs_log.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Your vault crypto tokens — find pairs within these ───────
UNIVERSE = [
    "BTC", "ETH", "SOL", "BNB", "AVAX",
    "TAO", "FET", "RNDR",          # AI tokens — highly correlated
    "PEPE", "WIF",                  # Meme tokens — correlated
    "ARB", "OP",                    # L2 tokens — correlated
    "LINK", "UNI",                  # DeFi tokens
]

# Known strong pairs (pre-seeded from domain knowledge)
KNOWN_PAIRS = [
    ("TAO",  "FET"),    # AI tokens
    ("TAO",  "RNDR"),   # AI tokens
    ("FET",  "RNDR"),   # AI tokens
    ("PEPE", "WIF"),    # Meme tokens
    ("SOL",  "AVAX"),   # L1 competitors
    ("ARB",  "OP"),     # L2 competitors
    ("BTC",  "ETH"),    # Core pair
    ("ETH",  "SOL"),    # Smart contract platforms
    ("LINK", "UNI"),    # DeFi
]

# Z-score thresholds
ENTRY_Z   = 2.0    # enter when spread exceeds 2 std devs
EXIT_Z    = 0.5    # close when spread returns within 0.5 std devs
STOP_Z    = 3.5    # stop loss if spread blows out to 3.5


# ══════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════

def fetch_prices(symbols: list[str], timeframe: str = "1H",
                 days: int = 90) -> pd.DataFrame:
    """Fetch OHLCV for multiple symbols, return close prices DataFrame."""
    from src.data.ccxt_fetcher import fetch_ohlcv_ccxt

    prices = {}
    for sym in symbols:
        try:
            df = fetch_ohlcv_ccxt(sym, timeframe=timeframe, days=days)
            if df is not None and len(df) > 50:
                prices[sym] = df["Close"]
        except Exception as e:
            print(f"  ⚠️  Could not fetch {sym}: {e}")

    if not prices:
        return pd.DataFrame()

    df = pd.DataFrame(prices)
    df = df.dropna()
    return df


# ══════════════════════════════════════════════════════════════
# COINTEGRATION TESTING
# ══════════════════════════════════════════════════════════════

def test_cointegration(series_a: pd.Series, series_b: pd.Series) -> dict:
    """
    Test if two price series are cointegrated using ADF test.
    Returns stats and whether pair is tradeable.
    """
    try:
        from statsmodels.tsa.stattools import coint
        import statsmodels.api as sm

        # Log prices
        log_a = np.log(series_a)
        log_b = np.log(series_b)

        # Cointegration test
        score, pvalue, _ = coint(log_a, log_b)

        # OLS regression to find hedge ratio
        X = sm.add_constant(log_b)
        model = sm.OLS(log_a, X).fit()
        hedge_ratio = model.params.iloc[1]
        spread = log_a - hedge_ratio * log_b

        # Spread stats
        spread_mean  = spread.mean()
        spread_std   = spread.std()
        z_current    = (spread.iloc[-1] - spread_mean) / spread_std

        return {
            "pvalue":       round(float(pvalue), 4),
            "cointegrated": pvalue < 0.05,
            "hedge_ratio":  round(float(hedge_ratio), 4),
            "spread_mean":  round(float(spread_mean), 6),
            "spread_std":   round(float(spread_std), 6),
            "z_current":    round(float(z_current), 3),
            "spread_last":  round(float(spread.iloc[-1]), 6),
        }
    except ImportError:
        print("  ⚠️  statsmodels not installed: pip install statsmodels")
        return {"cointegrated": False, "error": "statsmodels missing"}
    except Exception as e:
        return {"cointegrated": False, "error": str(e)[:80]}


def scan_pairs(prices: pd.DataFrame) -> list[dict]:
    """
    Test all pairs for cointegration and return tradeable pairs
    sorted by signal strength.
    """
    print(f"\n  🔍 Scanning {len(prices.columns)} tokens for cointegrated pairs...")
    results = []

    # Test known pairs first, then scan universe
    pairs_to_test = list(KNOWN_PAIRS)

    # Add additional pairs from universe
    cols = list(prices.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair = (cols[i], cols[j])
            if pair not in pairs_to_test and pair[::-1] not in pairs_to_test:
                pairs_to_test.append(pair)

    for sym_a, sym_b in pairs_to_test:
        if sym_a not in prices.columns or sym_b not in prices.columns:
            continue

        result = test_cointegration(prices[sym_a], prices[sym_b])
        if result.get("error"):
            continue

        results.append({
            "pair":        f"{sym_a}/{sym_b}",
            "sym_a":       sym_a,
            "sym_b":       sym_b,
            "pvalue":      result["pvalue"],
            "cointegrated": result["cointegrated"],
            "hedge_ratio": result["hedge_ratio"],
            "spread_mean": result["spread_mean"],
            "spread_std":  result["spread_std"],
            "z_current":   result["z_current"],
            "signal":      "LONG_A" if result["z_current"] < -ENTRY_Z
                           else "LONG_B" if result["z_current"] > ENTRY_Z
                           else "NEUTRAL",
            "abs_z":       abs(result["z_current"]),
        })

    # Sort by abs z-score for strongest current signals
    results.sort(key=lambda x: x["abs_z"], reverse=True)

    tradeable  = [r for r in results if r["cointegrated"]]
    signalling = [r for r in tradeable if r["signal"] != "NEUTRAL"]

    print(f"  ✅ {len(tradeable)} cointegrated pairs found")
    print(f"  🟡 {len(signalling)} pairs currently signalling")

    return results


# ══════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════

def generate_signals(prices: pd.DataFrame,
                     pairs: list[dict]) -> list[dict]:
    """
    Generate trade signals from active pair divergences.
    Each signal specifies both legs of the trade.
    """
    signals = []

    for pair in pairs:
        if not pair["cointegrated"]:
            continue
        if pair["signal"] == "NEUTRAL":
            continue

        sym_a = pair["sym_a"]
        sym_b = pair["sym_b"]
        z     = pair["z_current"]

        price_a = float(prices[sym_a].iloc[-1])
        price_b = float(prices[sym_b].iloc[-1])

        if z < -ENTRY_Z:
            # A is cheap relative to B — buy A, sell B
            direction_a = "LONG"
            direction_b = "SHORT"
            rationale   = f"{sym_a} cheap vs {sym_b} (Z={z:.2f})"
        elif z > ENTRY_Z:
            # A is expensive relative to B — sell A, buy B
            direction_a = "SHORT"
            direction_b = "LONG"
            rationale   = f"{sym_a} expensive vs {sym_b} (Z={z:.2f})"
        else:
            continue

        # Stop loss / take profit in Z-score terms
        # TP: z returns to EXIT_Z, SL: z blows to STOP_Z

        signals.append({
            "type":        "pairs",
            "pair":        pair["pair"],
            "sym_a":       sym_a,
            "sym_b":       sym_b,
            "direction_a": direction_a,
            "direction_b": direction_b,
            "price_a":     round(price_a, 6),
            "price_b":     round(price_b, 6),
            "hedge_ratio": pair["hedge_ratio"],
            "z_score":     round(z, 3),
            "z_entry":     ENTRY_Z,
            "z_exit":      EXIT_Z,
            "z_stop":      STOP_Z,
            "rationale":   rationale,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "status":      "PENDING",
        })

    return signals


# ══════════════════════════════════════════════════════════════
# EXECUTION
# ══════════════════════════════════════════════════════════════

def execute_pair(signal: dict, size_usd: float = 100) -> dict:
    """
    Execute both legs of a pairs trade on Hyperliquid.
    Both legs sized to be market-neutral (dollar-neutral).

    size_usd: total position size per leg in USD
    """
    sym_a = signal["sym_a"]
    sym_b = signal["sym_b"]

    print(f"\n  🔀 Pairs trade: {signal['pair']} Z={signal['z_score']:.2f}")
    print(f"     {signal['direction_a']:5} {sym_a} @ ${signal['price_a']:.4f}")
    print(f"     {signal['direction_b']:5} {sym_b} @ ${signal['price_b']:.4f}")
    print(f"     Size: ${size_usd:.0f} per leg (total ${size_usd*2:.0f} notional)")

    results = {}

    try:
        from src.exchanges.hyperliquid import market_buy, market_sell

        # Leg A
        if signal["direction_a"] == "LONG":
            results["leg_a"] = market_buy(sym_a, size_usd)
        else:
            results["leg_a"] = market_sell(sym_a, size_usd)

        time.sleep(0.5)  # Small delay between legs

        # Leg B
        if signal["direction_b"] == "LONG":
            results["leg_b"] = market_buy(sym_b, size_usd)
        else:
            results["leg_b"] = market_sell(sym_b, size_usd)

        print(f"  ✅ Both legs executed")
        results["status"] = "EXECUTED"

    except Exception as e:
        print(f"  ❌ Execution failed: {e}")
        results["status"] = "FAILED"
        results["error"]  = str(e)

    # Log
    _log_signal(signal, results)
    return results


def close_pair(signal: dict, size_usd: float = 100) -> dict:
    """Close both legs of an open pairs trade."""
    print(f"\n  ↩️  Closing pair: {signal['pair']}")

    try:
        from src.exchanges.hyperliquid import market_buy, market_sell
        # Reverse the original directions
        if signal["direction_a"] == "LONG":
            market_sell(signal["sym_a"], size_usd)
        else:
            market_buy(signal["sym_a"], size_usd)

        time.sleep(0.5)

        if signal["direction_b"] == "LONG":
            market_sell(signal["sym_b"], size_usd)
        else:
            market_buy(signal["sym_b"], size_usd)

        print(f"  ✅ Pair closed")
        return {"status": "CLOSED"}
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


# ══════════════════════════════════════════════════════════════
# BACKTESTING
# ══════════════════════════════════════════════════════════════

def backtest_pair(prices: pd.DataFrame, sym_a: str, sym_b: str,
                  lookback: int = 30, days: int = 180) -> dict:
    """
    Quick backtest of a single pair.
    Returns win rate, Sharpe, avg profit per trade.
    """
    if sym_a not in prices.columns or sym_b not in prices.columns:
        return {}

    log_a = np.log(prices[sym_a])
    log_b = np.log(prices[sym_b])

    # Rolling hedge ratio
    trades  = []
    in_trade = None

    for i in range(lookback, len(prices)):
        window_a = log_a.iloc[i-lookback:i]
        window_b = log_b.iloc[i-lookback:i]

        # Calculate spread and Z-score
        spread   = (window_a - window_b)
        mean     = spread.mean()
        std      = spread.std()
        if std == 0:
            continue

        z_current = (spread.iloc[-1] - mean) / std
        price_a   = prices[sym_a].iloc[i]
        price_b   = prices[sym_b].iloc[i]

        if in_trade is None:
            # Look for entry
            if z_current < -ENTRY_Z:
                in_trade = {"direction": "LONG_A", "z_entry": z_current,
                            "price_a": price_a, "price_b": price_b, "i": i}
            elif z_current > ENTRY_Z:
                in_trade = {"direction": "LONG_B", "z_entry": z_current,
                            "price_a": price_a, "price_b": price_b, "i": i}
        else:
            # Check exit conditions
            should_exit = (
                abs(z_current) < EXIT_Z or
                abs(z_current) > STOP_Z or
                i - in_trade["i"] > 72   # max hold 72 bars
            )

            if should_exit:
                # Calculate P&L
                if in_trade["direction"] == "LONG_A":
                    pnl_a = (price_a - in_trade["price_a"]) / in_trade["price_a"]
                    pnl_b = (in_trade["price_b"] - price_b) / in_trade["price_b"]
                else:
                    pnl_a = (in_trade["price_a"] - price_a) / in_trade["price_a"]
                    pnl_b = (price_b - in_trade["price_b"]) / in_trade["price_b"]

                total_pnl = (pnl_a + pnl_b) / 2
                trades.append({
                    "pnl": total_pnl,
                    "bars": i - in_trade["i"],
                    "win": total_pnl > 0,
                    "exit_reason": "converged" if abs(z_current) < EXIT_Z
                                   else "stop" if abs(z_current) > STOP_Z
                                   else "timeout"
                })
                in_trade = None

    if not trades:
        return {"trades": 0, "sharpe": 0}

    pnls     = [t["pnl"] for t in trades]
    wins     = [t for t in trades if t["win"]]
    win_rate = len(wins) / len(trades)
    avg_pnl  = np.mean(pnls)
    std_pnl  = np.std(pnls)
    sharpe   = (avg_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0

    return {
        "pair":        f"{sym_a}/{sym_b}",
        "trades":      len(trades),
        "win_rate":    round(win_rate * 100, 1),
        "avg_pnl_pct": round(avg_pnl * 100, 3),
        "sharpe":      round(sharpe, 2),
        "avg_bars":    round(np.mean([t["bars"] for t in trades]), 1),
    }


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════

def _log_signal(signal: dict, result: dict):
    import csv
    exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp","pair","z_score","direction_a","direction_b",
            "price_a","price_b","status"
        ])
        if not exists:
            w.writeheader()
        w.writerow({
            "timestamp":   signal.get("timestamp",""),
            "pair":        signal.get("pair",""),
            "z_score":     signal.get("z_score",""),
            "direction_a": signal.get("direction_a",""),
            "direction_b": signal.get("direction_b",""),
            "price_a":     signal.get("price_a",""),
            "price_b":     signal.get("price_b",""),
            "status":      result.get("status",""),
        })


# ══════════════════════════════════════════════════════════════
# LIVE SCANNER
# ══════════════════════════════════════════════════════════════

def run_live(mode: str = "notify", size_usd: float = 100,
             interval_min: int = 60):
    """
    Live pairs trading scanner.
    Runs every hour (pairs trade on 1H-4H timeframe).

    mode: "notify" (log only) | "auto" (execute on Hyperliquid)
    """
    print(f"\n🌙 Pairs Trading Agent — {mode.upper()} MODE")
    print("=" * 50)
    print(f"  Universe : {', '.join(UNIVERSE[:8])}...")
    print(f"  Timeframe: 1H | Entry Z: ±{ENTRY_Z} | Exit Z: ±{EXIT_Z}")
    print(f"  Mode     : {mode}")
    print(f"  Interval : {interval_min}min")
    print("=" * 50)

    scan_count = 0

    while True:
        scan_count += 1
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n{'─'*50}")
        print(f"  Pairs Scan #{scan_count} — {now}")

        try:
            # Fetch prices
            prices = fetch_prices(UNIVERSE, timeframe="1H", days=90)
            if prices.empty:
                print("  ❌ Could not fetch price data")
            else:
                print(f"  📊 {len(prices.columns)} tokens loaded, "
                      f"{len(prices)} bars each")

                # Scan for pairs
                all_pairs = scan_pairs(prices)
                signals   = generate_signals(prices, all_pairs)

                if not signals:
                    print(f"  ○  No pairs signalling (all within ±{ENTRY_Z} Z)")
                else:
                    print(f"\n  🟡 {len(signals)} pairs signalling:\n")
                    for sig in signals:
                        icon = "🟢" if "LONG_A" in sig["rationale"] else "🔴"
                        print(f"  {icon} {sig['pair']:12} Z={sig['z_score']:+.2f} "
                              f"| {sig['direction_a']} {sig['sym_a']} / "
                              f"{sig['direction_b']} {sig['sym_b']}")
                        print(f"     {sig['rationale']}")

                        if mode == "auto":
                            execute_pair(sig, size_usd)
                        else:
                            _log_signal(sig, {"status": "LOGGED"})

                # Show top cointegrated pairs for context
                top_pairs = [p for p in all_pairs if p["cointegrated"]][:5]
                if top_pairs:
                    print(f"\n  📋 Top cointegrated pairs:")
                    for p in top_pairs:
                        icon = "🔥" if p["abs_z"] >= ENTRY_Z else "  "
                        print(f"  {icon} {p['pair']:12} p={p['pvalue']:.3f} "
                              f"Z={p['z_current']:+.2f} "
                              f"→ {p['signal']}")

        except KeyboardInterrupt:
            print("\n\n  ⏹  Stopped by user")
            break
        except Exception as e:
            print(f"  ❌ Scan error: {e}")

        print(f"\n  💤 Next scan in {interval_min}min...")
        try:
            for remaining in range(interval_min * 60, 0, -30):
                mins, secs = divmod(remaining, 60)
                print(f"\r  💤 Next scan in {mins}m{secs:02d}s   ",
                      end="", flush=True)
                time.sleep(min(30, remaining))
            print()
        except KeyboardInterrupt:
            print("\n\n  ⏹  Stopped")
            break


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="🌙 Pairs Trading Agent")
    p.add_argument("--scan",      action="store_true",
                   help="Scan for cointegrated pairs once")
    p.add_argument("--live",      action="store_true",
                   help="Run live scanner (notify mode)")
    p.add_argument("--auto",      action="store_true",
                   help="Run live scanner with auto execution")
    p.add_argument("--backtest",  action="store_true",
                   help="Backtest all known pairs")
    p.add_argument("--size",      type=float, default=100,
                   help="Position size USD per leg (default $100)")
    p.add_argument("--interval",  type=int,   default=60,
                   help="Scan interval in minutes (default 60)")
    p.add_argument("--tf",        default="1H",
                   help="Timeframe: 5m 15m 1H 4H (default 1H)")
    p.add_argument("--days",      type=int, default=90)
    args = p.parse_args()

    if args.scan or (not any([args.scan, args.live, args.auto, args.backtest])):
        prices = fetch_prices(UNIVERSE, timeframe=args.tf, days=args.days)
        if not prices.empty:
            pairs   = scan_pairs(prices)
            signals = generate_signals(prices, pairs)
            print(f"\n  Current signals: {len(signals)}")
            for s in signals:
                print(f"  → {s['pair']} Z={s['z_score']:+.2f} "
                      f"{s['direction_a']} {s['sym_a']} / "
                      f"{s['direction_b']} {s['sym_b']}")

    elif args.backtest:
        print("\n🔬 Backtesting all pairs...")
        prices = fetch_prices(UNIVERSE, timeframe=args.tf, days=365)
        if not prices.empty:
            results = []
            for sym_a, sym_b in KNOWN_PAIRS:
                r = backtest_pair(prices, sym_a, sym_b)
                if r.get("trades", 0) > 5:
                    results.append(r)
                    print(f"  {r['pair']:<14} "
                          f"Trades={r['trades']:>4} "
                          f"WR={r['win_rate']:>5.1f}% "
                          f"Sharpe={r['sharpe']:>5.2f} "
                          f"AvgPnL={r['avg_pnl_pct']:>+6.3f}%")

    elif args.live:
        run_live(mode="notify", size_usd=args.size, interval_min=args.interval)

    elif args.auto:
        run_live(mode="auto", size_usd=args.size, interval_min=args.interval)
