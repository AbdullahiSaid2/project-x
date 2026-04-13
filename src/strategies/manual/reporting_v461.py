from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


TRADE_KEY_COLS = [
    "side",
    "entry_time_et",
    "exit_time_et",
    "entry_price",
    "exit_price",
    "pnl",
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
]

STRONG_SETUP_TYPES = {"LONDON_CONTINUATION", "NYAM_CONTINUATION", "NYPM_CONTINUATION"}
STRONG_BRIDGES = {"IFVG", "C2C3", "MSS", "CISD"}


def _score_row(row: pd.Series) -> tuple[int, int]:
    filled = 0
    priority = 0
    for col in META_PRIORITY_COLS:
        val = row.get(col)
        if pd.notna(val) and val != "":
            filled += 1
            if col in {"setup_type", "bridge_type", "setup_tier", "planned_rr", "stop_points", "tp_points"}:
                priority += 1
    return filled, priority


def _ensure_datetime(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _dedupe_trade_rows(meta: pd.DataFrame) -> pd.DataFrame:
    if meta.empty:
        return meta.copy()

    meta = meta.copy()
    meta = _ensure_datetime(meta, ["entry_time_et", "exit_time_et"])

    for col in TRADE_KEY_COLS:
        if col not in meta.columns:
            meta[col] = pd.NA

    meta["_score_filled"] = meta.apply(lambda r: _score_row(r)[0], axis=1)
    meta["_score_priority"] = meta.apply(lambda r: _score_row(r)[1], axis=1)
    meta["_orig_order"] = range(len(meta))

    meta = meta.sort_values(
        [*TRADE_KEY_COLS, "_score_filled", "_score_priority", "_orig_order"],
        ascending=[True, True, True, True, True, True, False, False, True],
        na_position="last",
    )

    meta = meta.drop_duplicates(subset=TRADE_KEY_COLS, keep="first").copy()
    meta = meta.drop(columns=["_score_filled", "_score_priority", "_orig_order"], errors="ignore")
    meta = meta.sort_values(["entry_time_et", "exit_time_et"], na_position="last").reset_index(drop=True)
    return meta


def _naive_excel_copy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64tz_dtype(out[col]):
            out[col] = out[col].dt.tz_localize(None)
    return out


def _safe_month(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    return ts.dt.to_period("M").astype(str)


def _dynamic_contracts(row: pd.Series) -> int:
    setup_type = str(row.get("setup_type") or "").upper()
    bridge = str(row.get("bridge_type") or "").upper()
    tier = str(row.get("setup_tier") or "").upper()
    stop_points = row.get("stop_points")
    try:
        stop_points = float(stop_points) if pd.notna(stop_points) else None
    except Exception:
        stop_points = None

    if tier == "A" and setup_type in STRONG_SETUP_TYPES and bridge in STRONG_BRIDGES and stop_points is not None and stop_points <= 40:
        return 10
    return 5


def prepare_trade_log(meta: pd.DataFrame) -> pd.DataFrame:
    meta = _dedupe_trade_rows(meta)
    if meta.empty:
        return meta

    meta = _ensure_datetime(meta, ["entry_time_et", "exit_time_et"])

    if "calendar_exit_date_et" not in meta.columns or meta["calendar_exit_date_et"].isna().all():
        meta["calendar_exit_date_et"] = meta["exit_time_et"].dt.tz_localize(None).dt.date
    if "exit_apex_session_date" not in meta.columns or meta["exit_apex_session_date"].isna().all():
        meta["exit_apex_session_date"] = meta["exit_time_et"].dt.tz_localize(None).dt.date
    if "entry_apex_session_date" not in meta.columns or meta["entry_apex_session_date"].isna().all():
        meta["entry_apex_session_date"] = meta["entry_time_et"].dt.tz_localize(None).dt.date

    meta["entry_month_et"] = _safe_month(meta["entry_time_et"])
    meta["exit_month_et"] = _safe_month(meta["exit_time_et"])

    if "realized_points" not in meta.columns:
        meta["realized_points"] = pd.NA

    meta["realized_dollars_5_mnq"] = pd.to_numeric(meta.get("realized_points"), errors="coerce") * 10.0
    fallback_pnl = pd.to_numeric(meta.get("pnl"), errors="coerce")
    meta["realized_dollars_5_mnq"] = meta["realized_dollars_5_mnq"].fillna(fallback_pnl)
    meta["realized_dollars_10_mnq"] = pd.to_numeric(meta.get("realized_points"), errors="coerce") * 20.0
    meta["realized_dollars_10_mnq"] = meta["realized_dollars_10_mnq"].fillna(fallback_pnl)
    meta["report_contracts"] = meta.apply(_dynamic_contracts, axis=1)
    meta["realized_dollars_dynamic_contracts"] = pd.to_numeric(meta.get("realized_points"), errors="coerce") * (2.0 * meta["report_contracts"])
    meta["realized_dollars_dynamic_contracts"] = meta["realized_dollars_dynamic_contracts"].fillna(fallback_pnl)

    meta["entry_time_et_naive"] = meta["entry_time_et"].dt.tz_localize(None)
    meta["exit_time_et_naive"] = meta["exit_time_et"].dt.tz_localize(None)
    return meta


def load_strategy_meta(strategy_cls) -> pd.DataFrame:
    raw = pd.DataFrame(strategy_cls.TRADE_METADATA_LOG)
    return prepare_trade_log(raw)


def save_excel_with_naive_datetimes(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            _naive_excel_copy(df).to_excel(writer, index=False, sheet_name=name)
