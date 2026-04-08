from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "src" / "data" / "rbi_results"
OUTPUT_JSON = RESULTS_DIR / "batch_summary.json"
OUTPUT_CSV = RESULTS_DIR / "batch_summary_rows.csv"

CLASSIFICATION_ORDER = [
    "vault_candidate",
    "research_candidate",
    "watchlist_faithful",
    "watchlist_discovery",
    "reject",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        if x != x:
            return default
        return x
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    try:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no", ""}:
            return False
        return default
    except Exception:
        return default


def _load_result_files() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        [
            p
            for p in RESULTS_DIR.glob("*.json")
            if p.name not in {
                "batch_summary.json",
                "vault_index.json",
                "alpha_factory_report.json",
            }
        ]
    )


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _pick_best(summary: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    best = summary.get(key)
    return best if isinstance(best, dict) else None


def _idea_row(payload: Dict[str, Any], filename: str) -> Dict[str, Any]:
    idea = payload.get("idea", "")
    schema = payload.get("schema", {}) or {}
    summary = payload.get("summary", {}) or {}
    classification = payload.get("classification", "unclassified")
    idea_quality = payload.get("idea_quality", {}) or {}

    best_faithful = _pick_best(summary, "best_faithful")
    best_discovery = _pick_best(summary, "best_discovery")

    row = {
        "file": filename,
        "idea": idea,
        "strategy_name": schema.get("name", ""),
        "family": schema.get("family", ""),
        "classification": classification,
        "created_at_utc": payload.get("created_at_utc", ""),
        "idea_quality_score": _safe_int(idea_quality.get("score")),
        "idea_quality_pass": _safe_bool(idea_quality.get("pass")),
        "eligible_count": _safe_int(summary.get("eligible_count")),
        "valid_count": _safe_int(summary.get("valid_count")),
        "reliable_count": _safe_int(summary.get("reliable_count")),
        "overtraded_count": _safe_int(summary.get("overtraded_count")),
        "research_pass_faithful": _safe_bool(summary.get("research_pass_faithful")),
        "research_pass_discovery": _safe_bool(summary.get("research_pass_discovery")),
        "discovery_quality_pass": _safe_bool(summary.get("discovery_quality_pass")),
        "vault_ready_hint_faithful": _safe_bool(summary.get("vault_ready_hint_faithful")),
        "vault_ready_hint_discovery": _safe_bool(summary.get("vault_ready_hint_discovery")),
        "faithful_symbol": (best_faithful or {}).get("symbol", ""),
        "faithful_timeframe": (best_faithful or {}).get("timeframe", ""),
        "faithful_return_pct": _safe_float((best_faithful or {}).get("return_pct")),
        "faithful_sharpe": _safe_float((best_faithful or {}).get("sharpe")),
        "faithful_win_rate": _safe_float((best_faithful or {}).get("win_rate")),
        "faithful_num_trades": _safe_int((best_faithful or {}).get("num_trades")),
        "faithful_score": _safe_float((best_faithful or {}).get("ranking_score_faithful")),
        "faithful_expectancy_proxy": _safe_float((best_faithful or {}).get("expectancy_proxy")),
        "faithful_overtraded": _safe_bool((best_faithful or {}).get("overtraded")),
        "faithful_unreliable": _safe_bool((best_faithful or {}).get("unreliable_sample")),
        "discovery_symbol": (best_discovery or {}).get("symbol", ""),
        "discovery_timeframe": (best_discovery or {}).get("timeframe", ""),
        "discovery_return_pct": _safe_float((best_discovery or {}).get("return_pct")),
        "discovery_sharpe": _safe_float((best_discovery or {}).get("sharpe")),
        "discovery_win_rate": _safe_float((best_discovery or {}).get("win_rate")),
        "discovery_num_trades": _safe_int((best_discovery or {}).get("num_trades")),
        "discovery_score": _safe_float((best_discovery or {}).get("ranking_score_discovery")),
        "discovery_expectancy_proxy": _safe_float((best_discovery or {}).get("expectancy_proxy")),
        "discovery_overtraded": _safe_bool((best_discovery or {}).get("overtraded")),
        "discovery_unreliable": _safe_bool((best_discovery or {}).get("unreliable_sample")),
    }
    return row


def _sort_rows(rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode == "faithful":
        return sorted(
            rows,
            key=lambda r: (
                r["vault_ready_hint_faithful"],
                r["research_pass_faithful"],
                not r["faithful_overtraded"],
                not r["faithful_unreliable"],
                r["faithful_return_pct"],
                r["faithful_sharpe"],
                r["faithful_num_trades"],
                r["faithful_score"],
            ),
            reverse=True,
        )
    return sorted(
        rows,
        key=lambda r: (
            r["vault_ready_hint_discovery"],
            r["discovery_quality_pass"],
            r["research_pass_discovery"],
            not r["discovery_overtraded"],
            not r["discovery_unreliable"],
            r["discovery_return_pct"],
            r["discovery_sharpe"],
            r["discovery_num_trades"],
            r["discovery_score"],
        ),
        reverse=True,
    )


def _top_n(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return rows[:n]


def _serialize_top(rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if mode == "faithful":
            out.append(
                {
                    "idea": r["idea"],
                    "strategy_name": r["strategy_name"],
                    "family": r["family"],
                    "classification": r["classification"],
                    "symbol": r["faithful_symbol"],
                    "timeframe": r["faithful_timeframe"],
                    "return_pct": r["faithful_return_pct"],
                    "sharpe": r["faithful_sharpe"],
                    "win_rate": r["faithful_win_rate"],
                    "num_trades": r["faithful_num_trades"],
                    "score": r["faithful_score"],
                    "overtraded": r["faithful_overtraded"],
                    "unreliable": r["faithful_unreliable"],
                }
            )
        else:
            out.append(
                {
                    "idea": r["idea"],
                    "strategy_name": r["strategy_name"],
                    "family": r["family"],
                    "classification": r["classification"],
                    "symbol": r["discovery_symbol"],
                    "timeframe": r["discovery_timeframe"],
                    "return_pct": r["discovery_return_pct"],
                    "sharpe": r["discovery_sharpe"],
                    "win_rate": r["discovery_win_rate"],
                    "num_trades": r["discovery_num_trades"],
                    "score": r["discovery_score"],
                    "overtraded": r["discovery_overtraded"],
                    "unreliable": r["discovery_unreliable"],
                }
            )
    return out


def _write_csv(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        OUTPUT_CSV.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_batch_report() -> Dict[str, Any]:
    files = _load_result_files()
    rows: List[Dict[str, Any]] = []
    skipped_files: List[str] = []

    for path in files:
        payload = _read_json(path)
        if not payload:
            skipped_files.append(path.name)
            continue
        rows.append(_idea_row(payload, path.name))

    classification_counts = Counter(r["classification"] for r in rows)
    family_counts = Counter(r["family"] for r in rows)

    faithful_sorted = _sort_rows(rows, "faithful")
    discovery_sorted = _sort_rows(rows, "discovery")

    by_classification: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_classification[r["classification"]].append(
            {
                "idea": r["idea"],
                "strategy_name": r["strategy_name"],
                "family": r["family"],
                "faithful_symbol": r["faithful_symbol"],
                "faithful_timeframe": r["faithful_timeframe"],
                "faithful_return_pct": r["faithful_return_pct"],
                "faithful_sharpe": r["faithful_sharpe"],
                "discovery_symbol": r["discovery_symbol"],
                "discovery_timeframe": r["discovery_timeframe"],
                "discovery_return_pct": r["discovery_return_pct"],
                "discovery_sharpe": r["discovery_sharpe"],
            }
        )

    ordered_classification_counts = {
        key: classification_counts.get(key, 0) for key in CLASSIFICATION_ORDER
    }
    ordered_classification_counts.update(
        {
            key: value
            for key, value in classification_counts.items()
            if key not in ordered_classification_counts
        }
    )

    report = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "results_dir": str(RESULTS_DIR),
        "total_result_files": len(files),
        "total_loaded_results": len(rows),
        "skipped_files": skipped_files,
        "classification_counts": ordered_classification_counts,
        "family_counts": dict(sorted(family_counts.items(), key=lambda kv: kv[0])),
        "top_faithful": _serialize_top(_top_n(faithful_sorted, 10), "faithful"),
        "top_discovery": _serialize_top(_top_n(discovery_sorted, 10), "discovery"),
        "vault_candidates": by_classification.get("vault_candidate", []),
        "research_candidates": by_classification.get("research_candidate", []),
        "watchlist_faithful": by_classification.get("watchlist_faithful", []),
        "watchlist_discovery": by_classification.get("watchlist_discovery", []),
        "rejects": by_classification.get("reject", []),
    }
    return report, rows


def print_report(report: Dict[str, Any]) -> None:
    print("\n════════════════════════════════════════════════════════════")
    print("RBI BATCH SUMMARY")
    print("════════════════════════════════════════════════════════════")
    print(f"Generated: {report['generated_at_utc']}")
    print(f"Loaded results: {report['total_loaded_results']}")
    print()

    print("Classification counts:")
    for key, value in report["classification_counts"].items():
        print(f"  - {key}: {value}")

    print("\nTop faithful:")
    top_faithful = report.get("top_faithful", [])
    if not top_faithful:
        print("  (none)")
    else:
        for i, item in enumerate(top_faithful[:5], start=1):
            print(
                f"  {i}. {item['strategy_name']} | {item['symbol']} {item['timeframe']} | "
                f"Return {item['return_pct']:.1f}% | Sharpe {item['sharpe']:.2f} | "
                f"Trades {item['num_trades']} | {item['classification']}"
            )

    print("\nTop discovery:")
    top_discovery = report.get("top_discovery", [])
    if not top_discovery:
        print("  (none)")
    else:
        for i, item in enumerate(top_discovery[:5], start=1):
            print(
                f"  {i}. {item['strategy_name']} | {item['symbol']} {item['timeframe']} | "
                f"Return {item['return_pct']:.1f}% | Sharpe {item['sharpe']:.2f} | "
                f"Trades {item['num_trades']} | {item['classification']}"
            )

    print(f"\nSaved JSON: {OUTPUT_JSON}")
    print(f"Saved CSV:  {OUTPUT_CSV}")


def main() -> None:
    report, rows = build_batch_report()
    OUTPUT_JSON.write_text(json.dumps(report, indent=2))
    _write_csv(rows)
    print_report(report)


if __name__ == "__main__":
    main()