from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


TRADE_KEY_COLS = [
    "side",
    "entry_time_et_naive",
    "exit_time_et_naive",
    "entry_price",
    "exit_price",
    "realized_points",
]

META_PRIORITY_COLS = [
    "setup_type",
    "bridge_type",
    "setup_tier",
    "entry_variant",
    "planned_entry_price",
    "planned_stop_price",
    "planned_target_price",
    "partial_target_price",
    "runner_target_price",
    "stop_points",
    "tp_points",
    "planned_rr",
    "entry_apex_session_date",
    "exit_apex_session_date",
    "calendar_exit_date_et",
    "preferred_production_bar_size",
    "optional_fast_bar_size",
]

STRONG_SETUP_TYPES = {"ASIA_CONTINUATION", "LONDON_CONTINUATION", "NYAM_CONTINUATION", "NYPM_CONTINUATION"}
STRONG_BRIDGES = {"IFVG", "C2C3", "MSS", "CISD", "iFVG"}


def _score_row(row: pd.Series) -> tuple[int, int, int]:
    filled = 0
    priority = 0
    dynamic = 0
    for col in META_PRIORITY_COLS:
        val = row.get(col)
        if pd.notna(val) and val != "":
            filled += 1
            if col in {"setup_type", "bridge_type", "setup_tier", "planned_rr", "stop_points", "tp_points"}:
                priority += 1
    if pd.notna(row.get("planned_rr")):
        try:
            if float(row.get("planned_rr")) >= 2.0:
                dynamic += 1
        except Exception:
            pass
    if pd.notna(row.get("setup_tier")) and str(row.get("setup_tier")).upper() == "A":
        dynamic += 1
    return filled, priority, dynamic


def _ensure_datetime(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _add_naive_time_cols(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    for src, dest in (("entry_time_et", "entry_time_et_naive"), ("exit_time_et", "exit_time_et_naive")):
        if src in meta.columns:
            ts = pd.to_datetime(meta[src], errors="coerce")
            try:
                if getattr(ts.dt, "tz", None) is not None:
                    ts = ts.dt.tz_localize(None)
            except Exception:
                pass
            meta[dest] = ts
        elif dest not in meta.columns:
            meta[dest] = pd.NaT
    return meta


def _dedupe_trade_rows(meta: pd.DataFrame) -> pd.DataFrame:
    if meta.empty:
        return meta.copy()

    meta = meta.copy()
    meta = _ensure_datetime(meta, ["entry_time_et", "exit_time_et"])
    meta = _add_naive_time_cols(meta)

    for col in TRADE_KEY_COLS:
        if col not in meta.columns:
            meta[col] = pd.NA

    scores = meta.apply(_score_row, axis=1)
    meta["_score_filled"] = [s[0] for s in scores]
    meta["_score_priority"] = [s[1] for s in scores]
    meta["_score_dynamic"] = [s[2] for s in scores]
    meta["_orig_order"] = range(len(meta))

    meta = meta.sort_values(
        [*TRADE_KEY_COLS, "_score_filled", "_score_priority", "_score_dynamic", "_orig_order"],
        ascending=[True, True, True, True, True, True, False, False, False, True],
        na_position="last",
    )

    meta = meta.drop_duplicates(subset=TRADE_KEY_COLS, keep="first").copy()
    meta = meta.drop(columns=["_score_filled", "_score_priority", "_score_dynamic", "_orig_order"], errors="ignore")
    meta = meta.sort_values(["entry_time_et_naive", "exit_time_et_naive"], na_position="last").reset_index(drop=True)
    return meta


def _naive_excel_copy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64tz_dtype(out[col]):
            out[col] = out[col].dt.tz_localize(None)
    return out


def _safe_month(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_localize(None)
    except Exception:
        pass
    return ts.dt.to_period("M").astype(str)


def _dynamic_contracts(row: pd.Series) -> int:
    setup_type = str(row.get("setup_type") or "").upper()
    bridge = str(row.get("bridge_type") or "").upper()
    tier = str(row.get("setup_tier") or "").upper()

    stop_points = row.get("stop_points")
    planned_rr = row.get("planned_rr")
    try:
        stop_points = float(stop_points) if pd.notna(stop_points) else None
    except Exception:
        stop_points = None
    try:
        planned_rr = float(planned_rr) if pd.notna(planned_rr) else None
    except Exception:
        planned_rr = None

    if (
        tier == "A"
        and setup_type in STRONG_SETUP_TYPES
        and bridge in STRONG_BRIDGES
        and stop_points is not None
        and stop_points <= 40.0
        and planned_rr is not None
        and planned_rr >= 3.0
    ):
        return 10
    return 5


def prepare_trade_log(meta: pd.DataFrame) -> pd.DataFrame:
    meta = _dedupe_trade_rows(meta)
    if meta.empty:
        return meta

    meta = _ensure_datetime(meta, ["entry_time_et", "exit_time_et"])
    meta = _add_naive_time_cols(meta)

    if "calendar_exit_date_et" not in meta.columns or meta["calendar_exit_date_et"].isna().all():
        meta["calendar_exit_date_et"] = meta["exit_time_et_naive"].dt.date
    if "exit_apex_session_date" not in meta.columns or meta["exit_apex_session_date"].isna().all():
        meta["exit_apex_session_date"] = meta["exit_time_et_naive"].dt.date
    if "entry_apex_session_date" not in meta.columns or meta["entry_apex_session_date"].isna().all():
        meta["entry_apex_session_date"] = meta["entry_time_et_naive"].dt.date

    meta["entry_month_et"] = _safe_month(meta["entry_time_et_naive"])
    meta["exit_month_et"] = _safe_month(meta["exit_time_et_naive"])

    if "realized_points" not in meta.columns:
        meta["realized_points"] = pd.NA

    fallback_pnl = pd.to_numeric(meta.get("pnl"), errors="coerce")
    points_num = pd.to_numeric(meta.get("realized_points"), errors="coerce")

    meta["realized_dollars_5_mnq"] = (points_num * 10.0).fillna(fallback_pnl)
    meta["realized_dollars_10_mnq"] = (points_num * 20.0).fillna(fallback_pnl)

    if "report_contracts" not in meta.columns or meta["report_contracts"].isna().all():
        meta["report_contracts"] = meta.apply(_dynamic_contracts, axis=1)
    else:
        inferred = meta.apply(_dynamic_contracts, axis=1)
        meta["report_contracts"] = pd.to_numeric(meta["report_contracts"], errors="coerce").fillna(inferred).astype(int)

    # For real in-engine dynamic sizing builds, the actual backtested dollars are the `pnl`
    # column. Only fall back to synthetic points x contracts when actual PnL is unavailable.
    actual_engine_pnl = fallback_pnl.copy()
    synthetic_dynamic = points_num * (2.0 * pd.to_numeric(meta["report_contracts"], errors="coerce"))
    meta["realized_dollars_dynamic_contracts"] = actual_engine_pnl.fillna(synthetic_dynamic)
    return meta


def load_strategy_meta(strategy_cls) -> pd.DataFrame:
    raw = pd.DataFrame(strategy_cls.TRADE_METADATA_LOG)
    return prepare_trade_log(raw)


def save_excel_with_naive_datetimes(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            _naive_excel_copy(df).to_excel(writer, index=False, sheet_name=name)
