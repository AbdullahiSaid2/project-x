"""
Audit an existing top_bottom_ticking trade log for partial-exit PnL errors.

Usage:
    python audit_top_bottom_trade_log_pnl.py --csv top_bottom_ticking_trade_log_apex_50k_eval_1825d_notail.csv

This does not rerun the strategy. It recalculates the correct futures-dollar PnL
from the existing trade log using:
    realized_points * dollars_per_point * actual_closed_contracts

If `closed_size_contracts` is missing, it infers actual closed size from:
    abs(pnl / realized_points)
where Backtesting.py `pnl` is points * actual_closed_size.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EPS = 1e-12


def realized_points(row: pd.Series) -> float:
    side = str(row.get("side", "")).upper()
    entry = float(row["entry_price"])
    exit_ = float(row["exit_price"])
    return exit_ - entry if side == "LONG" else entry - exit_


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    return float(gross_profit / gross_loss) if gross_loss > EPS else np.nan


def audit(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    required = {"side", "entry_price", "exit_price", "dollars_per_point"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "pnl" not in df.columns and "engine_pnl_points_contracts" not in df.columns:
        raise ValueError("Need either 'pnl' or 'engine_pnl_points_contracts' to infer actual closed size.")

    out = df.copy()
    out["realized_points_calc"] = out.apply(realized_points, axis=1)
    engine_col = "engine_pnl_points_contracts" if "engine_pnl_points_contracts" in out.columns else "pnl"
    out["engine_pnl_points_contracts"] = pd.to_numeric(out[engine_col], errors="coerce")

    if "closed_size_contracts" in out.columns:
        out["actual_closed_contracts"] = pd.to_numeric(out["closed_size_contracts"], errors="coerce").abs()
    else:
        out["actual_closed_contracts"] = np.nan

    inferred = np.where(
        out["realized_points_calc"].abs() > EPS,
        (out["engine_pnl_points_contracts"] / out["realized_points_calc"]).abs(),
        np.nan,
    )
    out["inferred_closed_contracts"] = inferred
    out["actual_closed_contracts"] = out["actual_closed_contracts"].fillna(out["inferred_closed_contracts"])
    out["correct_realized_pnl_dollars"] = out["realized_points_calc"] * out["dollars_per_point"] * out["actual_closed_contracts"]
    out["engine_dollarized_pnl_check"] = out["engine_pnl_points_contracts"] * out["dollars_per_point"]
    out["pnl_math_diff_dollars"] = out["correct_realized_pnl_dollars"] - out["engine_dollarized_pnl_check"]

    if "report_contracts" in out.columns:
        configured = pd.to_numeric(out["report_contracts"], errors="coerce")
    elif "configured_contracts" in out.columns:
        configured = pd.to_numeric(out["configured_contracts"], errors="coerce")
    else:
        configured = out["actual_closed_contracts"]
    out["legacy_full_size_pnl_dollars"] = out["realized_points_calc"] * out["dollars_per_point"] * configured
    if "gross_pnl_dollars_dynamic" in out.columns:
        out["old_reported_gross_pnl_dollars_dynamic"] = pd.to_numeric(out["gross_pnl_dollars_dynamic"], errors="coerce")
    else:
        out["old_reported_gross_pnl_dollars_dynamic"] = out["legacy_full_size_pnl_dollars"]
    out["legacy_overstatement_dollars"] = out["legacy_full_size_pnl_dollars"] - out["correct_realized_pnl_dollars"]

    pnl = out["correct_realized_pnl_dollars"].fillna(0)
    summary = pd.DataFrame([
        {
            "source_csv": str(path),
            "rows": len(out),
            "old_reported_gross_total": float(out["old_reported_gross_pnl_dollars_dynamic"].sum()),
            "correct_realized_pnl_dollars": float(pnl.sum()),
            "legacy_overstatement_dollars": float(out["legacy_overstatement_dollars"].sum()),
            "engine_dollarized_pnl_check": float(out["engine_dollarized_pnl_check"].sum()),
            "max_abs_math_diff_dollars": float(out["pnl_math_diff_dollars"].abs().max()),
            "win_rate_pct_exit_rows": float((pnl > 0).mean() * 100.0),
            "profit_factor_exit_rows": profit_factor(pnl),
            "best_exit_dollars": float(pnl.max()),
            "worst_exit_dollars": float(pnl.min()),
            "partial_rows_less_than_configured": int((out["actual_closed_contracts"] < configured).sum()),
        }
    ])
    return out, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to trade log CSV")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    src = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    audited, summary = audit(src)
    audit_path = out_dir / f"{src.stem}_corrected_pnl_audit_rows.csv"
    summary_path = out_dir / f"{src.stem}_corrected_pnl_audit_summary.csv"
    audited.to_csv(audit_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote rows -> {audit_path}")
    print(f"Wrote summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
