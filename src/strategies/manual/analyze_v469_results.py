from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TRADES_CSV = ROOT / "v469_trade_log.csv"


def safe_pct(x: float, y: float) -> float:
    return 0.0 if y == 0 else 100.0 * x / y


def main() -> None:
    if not TRADES_CSV.exists():
        print(f"Missing {TRADES_CSV}")
        return

    df = pd.read_csv(TRADES_CSV)
    print("Loaded", len(df), "trades from", TRADES_CSV)

    for col in ["setup_type", "bridge_type", "setup_tier", "side", "entry_variant"]:
        if col not in df.columns:
            print(f"Missing column: {col}")

    pnl_col = "realized_dollars_dynamic_contracts" if "realized_dollars_dynamic_contracts" in df.columns else "pnl"
    if pnl_col not in df.columns:
        raise ValueError("No usable pnl column found")

    if "setup_type" in df.columns:
        summary = (
            df.groupby("setup_type", dropna=False)
            .agg(
                trades=(pnl_col, "size"),
                win_rate_pct=(pnl_col, lambda s: safe_pct((s > 0).sum(), len(s))),
                total_pnl_dollars=(pnl_col, "sum"),
                avg_trade_dollars=(pnl_col, "mean"),
            )
            .reset_index()
            .sort_values("total_pnl_dollars", ascending=False)
        )
        print("\nBy setup_type")
        print(summary.to_string(index=False))

    if {"setup_type", "bridge_type"}.issubset(df.columns):
        combo = (
            df.groupby(["setup_type", "bridge_type"], dropna=False)
            .agg(
                trades=(pnl_col, "size"),
                win_rate_pct=(pnl_col, lambda s: safe_pct((s > 0).sum(), len(s))),
                total_pnl_dollars=(pnl_col, "sum"),
                avg_trade_dollars=(pnl_col, "mean"),
            )
            .reset_index()
            .sort_values("total_pnl_dollars", ascending=False)
        )
        print("\nBy setup_type + bridge_type")
        print(combo.to_string(index=False))


if __name__ == "__main__":
    main()
