# ============================================================
# 🌙 Monte Carlo Risk Simulation Agent
#
# Runs thousands of simulated trading scenarios based on your
# actual backtest results to answer key risk questions:
#
#   "What's the worst 3-month streak I should expect?"
#   "What's the probability of blowing up my account?"
#   "What position size keeps risk under control?"
#   "How long could a drawdown last?"
#
# HOW TO RUN:
#   python src/agents/monte_carlo_agent.py
#   python src/agents/monte_carlo_agent.py --trades 500
# ============================================================

import sys
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config               import BACKTEST_INITIAL_CASH
from src.models.llm_router import model

REPO_ROOT  = Path(__file__).resolve().parents[2]
MC_LOG     = REPO_ROOT / "src" / "data" / "monte_carlo_results.csv"
MC_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────
N_SIMULATIONS   = 1000     # number of Monte Carlo paths
N_TRADES        = 200      # trades per simulation path
RUIN_THRESHOLD  = 0.50     # account is "ruined" if it drops 50%
STARTING_CASH   = BACKTEST_INITIAL_CASH


def load_backtest_trades() -> list[dict]:
    """Load real trade results from backtest CSV files."""
    trades = []

    # Try RBI backtest results
    rbi_csv = REPO_ROOT / "src" / "data" / "rbi_results" / "backtest_stats.csv"
    if rbi_csv.exists():
        df = pd.read_csv(rbi_csv)
        for _, row in df.iterrows():
            ret = float(row.get("return_pct", 0) or 0)
            wr  = float(row.get("win_rate", 50) or 50) / 100
            n   = int(row.get("num_trades", 0) or 0)
            if n > 5 and ret != 0:
                trades.append({
                    "strategy":  row.get("strategy", ""),
                    "symbol":    row.get("symbol", ""),
                    "return":    ret,
                    "win_rate":  wr,
                    "trades":    n,
                    "sharpe":    float(row.get("sharpe", 0) or 0),
                })

    # Try ICT backtest results
    ict_csv = REPO_ROOT / "src" / "data" / "ict_backtest" / "ict_backtest_summary.csv"
    if ict_csv.exists():
        df = pd.read_csv(ict_csv)
        for _, row in df.iterrows():
            ret = float(row.get("return_pct", 0) or 0)
            if float(row.get("trades", 0) or 0) > 5:
                trades.append({
                    "strategy":  "ICT_" + str(row.get("symbol", "")),
                    "symbol":    row.get("symbol", ""),
                    "return":    ret,
                    "win_rate":  float(row.get("win_rate", 50) or 50) / 100,
                    "trades":    int(row.get("trades", 0) or 0),
                    "sharpe":    float(row.get("sharpe", 0) or 0),
                })

    return trades


def synthetic_trades_from_stats(win_rate: float, avg_win_pct: float,
                                  avg_loss_pct: float, n: int = 500) -> np.ndarray:
    """Generate synthetic trade P&L array from statistics."""
    outcomes = np.random.random(n)
    pnls     = np.where(
        outcomes < win_rate,
        np.random.normal(avg_win_pct, avg_win_pct * 0.3, n),    # wins
        -np.abs(np.random.normal(avg_loss_pct, avg_loss_pct * 0.3, n)),  # losses
    )
    return pnls


def run_monte_carlo(win_rate: float, avg_win_pct: float, avg_loss_pct: float,
                     n_sims: int = N_SIMULATIONS,
                     n_trades: int = N_TRADES,
                     starting_cash: float = STARTING_CASH) -> dict:
    """
    Run Monte Carlo simulation.
    Returns statistics across all simulation paths.
    """
    final_values   = []
    max_drawdowns  = []
    ruin_count     = 0
    equity_paths   = []

    for _ in range(n_sims):
        cash     = starting_cash
        peak     = cash
        max_dd   = 0.0
        path     = [cash]

        trade_pnls = synthetic_trades_from_stats(win_rate, avg_win_pct, avg_loss_pct, n_trades)

        for pnl_pct in trade_pnls:
            # Apply trade (use 10% of cash per trade)
            trade_size = cash * 0.10
            cash      += trade_size * (pnl_pct / 100)
            cash       = max(cash, 0)

            peak   = max(peak, cash)
            dd     = (peak - cash) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
            path.append(round(cash, 2))

            if cash <= starting_cash * (1 - RUIN_THRESHOLD):
                ruin_count += 1
                break

        final_values.append(cash)
        max_drawdowns.append(max_dd)
        if len(equity_paths) < 20:   # save 20 sample paths for charting
            equity_paths.append(path)

    final_arr = np.array(final_values)
    dd_arr    = np.array(max_drawdowns)

    return {
        "n_simulations":   n_sims,
        "n_trades":        n_trades,
        "starting_cash":   starting_cash,
        "win_rate":        round(win_rate * 100, 1),
        "avg_win_pct":     round(avg_win_pct, 2),
        "avg_loss_pct":    round(avg_loss_pct, 2),
        # Final value stats
        "median_final":    round(float(np.median(final_arr)), 2),
        "mean_final":      round(float(np.mean(final_arr)), 2),
        "p10_final":       round(float(np.percentile(final_arr, 10)), 2),
        "p90_final":       round(float(np.percentile(final_arr, 90)), 2),
        "worst_final":     round(float(np.min(final_arr)), 2),
        "best_final":      round(float(np.max(final_arr)), 2),
        # Returns
        "median_return":   round((float(np.median(final_arr)) / starting_cash - 1) * 100, 1),
        "p10_return":      round((float(np.percentile(final_arr, 10)) / starting_cash - 1) * 100, 1),
        "p90_return":      round((float(np.percentile(final_arr, 90)) / starting_cash - 1) * 100, 1),
        # Drawdown
        "median_max_dd":   round(float(np.median(dd_arr)), 1),
        "worst_max_dd":    round(float(np.max(dd_arr)), 1),
        "p90_max_dd":      round(float(np.percentile(dd_arr, 90)), 1),
        # Risk
        "ruin_probability":round(ruin_count / n_sims * 100, 1),
        "equity_paths":    equity_paths[:10],
    }


INTERPRETATION_PROMPT = """You are a quant risk analyst interpreting Monte Carlo simulation results.

SIMULATION DATA:
{data}

Give a clear, practical interpretation for a retail trader.
Focus on:
1. Is this strategy safe to trade with real money?
2. What position size is appropriate?
3. What's the realistic expected outcome after {n_trades} trades?
4. Any red flags?

Respond in plain text, 4 sentences max. Be direct and honest."""


def get_ai_interpretation(mc_results: dict, strategy_name: str) -> str:
    """Get AI plain-English interpretation of Monte Carlo results."""
    try:
        summary = {k: v for k, v in mc_results.items() if k != "equity_paths"}
        raw = model.chat(
            system_prompt="You are a quant risk analyst. Be direct and practical.",
            user_prompt=INTERPRETATION_PROMPT.format(
                data=json.dumps(summary, indent=2),
                n_trades=mc_results["n_trades"],
            ),
        )
        return raw.strip()
    except Exception as e:
        return f"AI interpretation unavailable: {e}"


def print_results(strategy: str, mc: dict, interpretation: str):
    print(f"\n  {'─'*58}")
    print(f"  🎲 {strategy}")
    print(f"  {'─'*58}")
    print(f"  Simulations : {mc['n_simulations']:,} paths × {mc['n_trades']} trades")
    print(f"  Win rate    : {mc['win_rate']}%")
    print(f"  Avg win/loss: +{mc['avg_win_pct']:.2f}% / -{mc['avg_loss_pct']:.2f}%")
    print()
    print(f"  📊 Final Account Value (${mc['starting_cash']:,} start):")
    print(f"     Median    : ${mc['median_final']:>10,.2f}  ({mc['median_return']:+.1f}%)")
    print(f"     Best 10%  : ${mc['p90_final']:>10,.2f}  ({mc['p90_return']:+.1f}%)")
    print(f"     Worst 10% : ${mc['p10_final']:>10,.2f}  ({mc['p10_return']:+.1f}%)")
    print(f"     Worst case: ${mc['worst_final']:>10,.2f}")
    print()
    print(f"  📉 Drawdown:")
    print(f"     Median max DD : {mc['median_max_dd']:.1f}%")
    print(f"     90th pct DD   : {mc['p90_max_dd']:.1f}%")
    print(f"     Worst DD seen : {mc['worst_max_dd']:.1f}%")
    print()
    ruin_col = "🔴" if mc['ruin_probability'] > 10 else "🟡" if mc['ruin_probability'] > 3 else "🟢"
    print(f"  {ruin_col} Ruin probability (>{RUIN_THRESHOLD*100:.0f}% loss): {mc['ruin_probability']}%")
    print()
    print(f"  🤖 Interpretation:")
    for line in interpretation.split(". "):
        if line.strip():
            print(f"     {line.strip()}.")


def log_mc_results(strategy: str, mc: dict):
    row = {
        "timestamp":       datetime.now().isoformat(),
        "strategy":        strategy,
        "n_sims":          mc["n_simulations"],
        "win_rate":        mc["win_rate"],
        "avg_win":         mc["avg_win_pct"],
        "avg_loss":        mc["avg_loss_pct"],
        "median_return":   mc["median_return"],
        "p10_return":      mc["p10_return"],
        "p90_return":      mc["p90_return"],
        "worst_max_dd":    mc["worst_max_dd"],
        "ruin_pct":        mc["ruin_probability"],
    }
    write_header = not MC_LOG.exists()
    with open(MC_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


class MonteCarloAgent:

    def run(self, n_trades: int = N_TRADES):
        print(f"\n{'═'*60}")
        print(f"🎲 Monte Carlo Risk Simulation")
        print(f"   {N_SIMULATIONS:,} simulations × {n_trades} trades each")
        print(f"   Starting cash: ${STARTING_CASH:,}")
        print(f"{'═'*60}")

        # Load real backtest data
        trades = load_backtest_trades()

        if not trades:
            print("\n  ⚠️  No backtest data found. Run the RBI backtester first.")
            print("      Using demo parameters instead...")
            trades = [
                {"strategy": "Demo_Strategy", "symbol": "ETH",
                 "win_rate": 0.55, "return": 12.0, "trades": 30, "sharpe": 1.5}
            ]

        # Run simulation for each strategy
        for t in trades[:5]:   # limit to top 5
            strategy = f"{t['strategy']} ({t['symbol']})"
            wr       = t.get("win_rate", 0.5)
            total_ret = t.get("return", 0)
            n        = t.get("trades", 30)

            if n < 5 or total_ret == 0:
                continue

            # Estimate avg win/loss from return and win rate
            # Simple approximation: if win_rate=0.55 and return=12%
            # avg_win ≈ return / (win_rate * n) * 100
            per_trade_ret = abs(total_ret) / n
            avg_win  = per_trade_ret / wr        if wr > 0 else 1.0
            avg_loss = per_trade_ret / (1 - wr)  if wr < 1 else 1.0

            print(f"\n  🔄 Simulating {strategy}...")
            mc = run_monte_carlo(
                win_rate=wr,
                avg_win_pct=avg_win,
                avg_loss_pct=avg_loss,
                n_trades=n_trades,
            )
            interp = get_ai_interpretation(mc, strategy)
            print_results(strategy, mc, interp)
            log_mc_results(strategy, mc)

        print(f"\n{'═'*60}")
        print(f"✅ Monte Carlo complete. Results saved to: {MC_LOG}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--trades", type=int, default=N_TRADES,
                   help="Number of trades per simulation path")
    args = p.parse_args()
    MonteCarloAgent().run(n_trades=args.trades)
