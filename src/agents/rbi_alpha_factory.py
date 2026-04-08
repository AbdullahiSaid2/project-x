from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "src" / "data" / "rbi_results"
OUTPUT_JSON = RESULTS_DIR / "alpha_factory_report.json"
OUTPUT_CSV = RESULTS_DIR / "alpha_factory_candidates.csv"

MIN_SHARPE = 1.0
MIN_RETURN = 1.0
MIN_TRADES = 50
MIN_EXPECTANCY = 0.02
MAX_CANDIDATES_PER_BUCKET = 2
MAX_FINAL_CANDIDATES = 12

FUTURES_SYMBOLS = {"MES", "MNQ", "MYM"}
CRYPTO_SYMBOLS = {
    "BTC", "ETH", "SOL", "BNB", "AVAX", "LINK", "ARB", "OP", "UNI",
    "PEPE", "WIF", "FET", "RNDR", "TAO"
}

FUTURES_TIMEFRAME_BONUS = {
    "15m": -1.25,
    "1H": 1.25,
    "4H": 1.75,
    "1D": 0.75,
    "5m": -2.0,
}
CRYPTO_TIMEFRAME_BONUS = {
    "15m": -2.0,
    "1H": 0.75,
    "4H": 1.0,
    "1D": 0.5,
    "5m": -3.0,
}

SYMBOL_PREFERENCE_BONUS = {
    "MNQ": 1.75,
    "MES": 1.50,
    "MYM": 0.75,
    "BTC": 0.25,
    "ETH": 0.15,
}

MAX_WEIGHT_PER_SYMBOL = 0.45
MAX_WEIGHT_PER_CANDIDATE = 0.25


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
            p for p in RESULTS_DIR.glob("*.json")
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


def _choose_candidate(summary: Dict[str, Any], classification: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    best_faithful = summary.get("best_faithful")
    best_discovery = summary.get("best_discovery")

    if classification == "watchlist_faithful" and isinstance(best_faithful, dict):
        return "faithful", best_faithful

    if classification == "watchlist_discovery" and isinstance(best_discovery, dict):
        return "discovery", best_discovery

    faithful = best_faithful if isinstance(best_faithful, dict) else None
    discovery = best_discovery if isinstance(best_discovery, dict) else None

    if faithful and discovery:
        faithful_score = _safe_float(faithful.get("ranking_score_faithful"))
        discovery_score = _safe_float(discovery.get("ranking_score_discovery"))
        if faithful_score >= discovery_score:
            return "faithful", faithful
        return "discovery", discovery

    if faithful:
        return "faithful", faithful
    if discovery:
        return "discovery", discovery
    return None


def _candidate_bucket(symbol: str, timeframe: str, family: str) -> str:
    return f"{symbol.upper()}::{timeframe}::{family}"


def _market_of_symbol(symbol: str) -> str:
    sym = symbol.upper()
    if sym in FUTURES_SYMBOLS:
        return "futures"
    if sym in CRYPTO_SYMBOLS:
        return "crypto"
    return "unknown"


@dataclass
class Candidate:
    file: str
    idea: str
    family: str
    strategy_name: str
    classification: str
    candidate_mode: str
    symbol: str
    timeframe: str
    market: str
    return_pct: float
    sharpe: float
    win_rate: float
    num_trades: int
    expectancy_proxy: float
    regime_score: float
    trend_regime: str
    trend_aligned: bool
    vol_expanding: bool
    overtraded: bool
    unreliable_sample: bool
    ranking_score: float
    portfolio_score: float = 0.0
    allocation_weight: float = 0.0
    allocation_pct: float = 0.0
    risk_budget_pct: float = 0.0
    deployment_tier: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "idea": self.idea,
            "family": self.family,
            "strategy_name": self.strategy_name,
            "classification": self.classification,
            "candidate_mode": self.candidate_mode,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "market": self.market,
            "return_pct": self.return_pct,
            "sharpe": self.sharpe,
            "win_rate": self.win_rate,
            "num_trades": self.num_trades,
            "expectancy_proxy": self.expectancy_proxy,
            "regime_score": self.regime_score,
            "trend_regime": self.trend_regime,
            "trend_aligned": self.trend_aligned,
            "vol_expanding": self.vol_expanding,
            "overtraded": self.overtraded,
            "unreliable_sample": self.unreliable_sample,
            "ranking_score": self.ranking_score,
            "portfolio_score": self.portfolio_score,
            "allocation_weight": self.allocation_weight,
            "allocation_pct": self.allocation_pct,
            "risk_budget_pct": self.risk_budget_pct,
            "deployment_tier": self.deployment_tier,
        }


def _base_portfolio_score(c: Candidate) -> float:
    score = 0.0
    score += c.sharpe * 3.0
    score += c.return_pct * 0.50
    score += c.expectancy_proxy * 35.0
    score += min(c.num_trades, 250) * 0.015
    score += c.regime_score * 0.8

    if c.classification == "vault_candidate":
        score += 5.0
    elif c.classification == "research_candidate":
        score += 3.0
    elif c.classification == "watchlist_faithful":
        score += 1.75
    elif c.classification == "watchlist_discovery":
        score += 0.75

    if c.candidate_mode == "faithful":
        score += 0.75

    if c.trend_aligned:
        score += 0.50
    if c.vol_expanding:
        score += 0.50

    # market / symbol preference: futures first
    if c.market == "futures":
        score += 2.25
        score += FUTURES_TIMEFRAME_BONUS.get(c.timeframe, 0.0)
    elif c.market == "crypto":
        score -= 0.25
        score += CRYPTO_TIMEFRAME_BONUS.get(c.timeframe, 0.0)

    score += SYMBOL_PREFERENCE_BONUS.get(c.symbol.upper(), 0.0)

    if c.overtraded:
        score -= 10.0
    if c.unreliable_sample:
        score -= 5.0

    return round(score, 6)


def _extract_candidates() -> List[Candidate]:
    candidates: List[Candidate] = []

    for path in _load_result_files():
        payload = _read_json(path)
        if not payload:
            continue

        idea = payload.get("idea", "")
        schema = payload.get("schema", {}) or {}
        summary = payload.get("summary", {}) or {}
        classification = str(payload.get("classification", "reject"))

        if classification not in {
            "vault_candidate",
            "research_candidate",
            "watchlist_faithful",
            "watchlist_discovery",
        }:
            continue

        chosen = _choose_candidate(summary, classification)
        if not chosen:
            continue

        mode, best = chosen

        sharpe = _safe_float(best.get("sharpe"))
        return_pct = _safe_float(best.get("return_pct"))
        num_trades = _safe_int(best.get("num_trades"))
        expectancy_proxy = _safe_float(best.get("expectancy_proxy"))
        overtraded = _safe_bool(best.get("overtraded"))
        unreliable = _safe_bool(best.get("unreliable_sample"))

        if sharpe < MIN_SHARPE:
            continue
        if return_pct < MIN_RETURN:
            continue
        if num_trades < MIN_TRADES:
            continue
        if expectancy_proxy < MIN_EXPECTANCY:
            continue
        if overtraded:
            continue
        if unreliable:
            continue

        ranking_score = _safe_float(
            best.get("ranking_score_faithful")
            if mode == "faithful"
            else best.get("ranking_score_discovery")
        )

        candidate = Candidate(
            file=path.name,
            idea=idea,
            family=str(schema.get("family", "")),
            strategy_name=str(schema.get("name", "")),
            classification=classification,
            candidate_mode=mode,
            symbol=str(best.get("symbol", "")),
            timeframe=str(best.get("timeframe", "")),
            market=_market_of_symbol(str(best.get("symbol", ""))),
            return_pct=return_pct,
            sharpe=sharpe,
            win_rate=_safe_float(best.get("win_rate")),
            num_trades=num_trades,
            expectancy_proxy=expectancy_proxy,
            regime_score=_safe_float(best.get("regime_score")),
            trend_regime=str(best.get("trend_regime", "")),
            trend_aligned=_safe_bool(best.get("trend_aligned")),
            vol_expanding=_safe_bool(best.get("vol_expanding")),
            overtraded=overtraded,
            unreliable_sample=unreliable,
            ranking_score=ranking_score,
        )
        candidate.portfolio_score = _base_portfolio_score(candidate)
        candidates.append(candidate)

    return candidates


def _diversify_candidates(candidates: List[Candidate]) -> List[Candidate]:
    by_bucket: Dict[str, List[Candidate]] = defaultdict(list)
    for c in sorted(candidates, key=lambda x: x.portfolio_score, reverse=True):
        by_bucket[_candidate_bucket(c.symbol, c.timeframe, c.family)].append(c)

    diversified: List[Candidate] = []
    for bucket in sorted(by_bucket.keys()):
        diversified.extend(by_bucket[bucket][:MAX_CANDIDATES_PER_BUCKET])

    diversified = sorted(diversified, key=lambda x: x.portfolio_score, reverse=True)

    final: List[Candidate] = []
    family_counts: Dict[str, int] = defaultdict(int)
    symbol_counts: Dict[str, int] = defaultdict(int)

    for c in diversified:
        if len(final) >= MAX_FINAL_CANDIDATES:
            break
        if family_counts[c.family] >= 4:
            continue
        if symbol_counts[c.symbol] >= 3:
            continue

        final.append(c)
        family_counts[c.family] += 1
        symbol_counts[c.symbol] += 1

    return final


def _normalize_allocations(selected: List[Candidate]) -> List[Candidate]:
    if not selected:
        return selected

    raw_scores = [max(c.portfolio_score, 0.01) for c in selected]
    total = sum(raw_scores)

    for c, score in zip(selected, raw_scores):
        c.allocation_weight = score / total

    # cap per candidate
    overflow = 0.0
    for c in selected:
        if c.allocation_weight > MAX_WEIGHT_PER_CANDIDATE:
            overflow += c.allocation_weight - MAX_WEIGHT_PER_CANDIDATE
            c.allocation_weight = MAX_WEIGHT_PER_CANDIDATE

    if overflow > 0:
        uncapped = [c for c in selected if c.allocation_weight < MAX_WEIGHT_PER_CANDIDATE]
        if uncapped:
            uncapped_total = sum(c.allocation_weight for c in uncapped)
            if uncapped_total > 0:
                for c in uncapped:
                    c.allocation_weight += overflow * (c.allocation_weight / uncapped_total)

    # cap per symbol
    symbol_groups: Dict[str, List[Candidate]] = defaultdict(list)
    for c in selected:
        symbol_groups[c.symbol].append(c)

    symbol_overflow = 0.0
    for symbol, group in symbol_groups.items():
        group_sum = sum(c.allocation_weight for c in group)
        if group_sum > MAX_WEIGHT_PER_SYMBOL:
            ratio = MAX_WEIGHT_PER_SYMBOL / group_sum
            reduced_sum = 0.0
            for c in group:
                c.allocation_weight *= ratio
                reduced_sum += c.allocation_weight
            symbol_overflow += group_sum - reduced_sum

    if symbol_overflow > 0:
        eligible = []
        for symbol, group in symbol_groups.items():
            if sum(c.allocation_weight for c in group) < MAX_WEIGHT_PER_SYMBOL:
                eligible.extend(group)

        eligible_total = sum(c.allocation_weight for c in eligible)
        if eligible_total > 0:
            for c in eligible:
                c.allocation_weight += symbol_overflow * (c.allocation_weight / eligible_total)

    # final normalize
    total_weight = sum(c.allocation_weight for c in selected)
    if total_weight > 0:
        for c in selected:
            c.allocation_weight /= total_weight
            c.allocation_pct = round(c.allocation_weight * 100.0, 2)

            # prop-firm-friendly risk budget
            # notional idea: small risk budgets for futures execution
            if c.market == "futures":
                c.risk_budget_pct = round(min(0.60, 0.20 + c.allocation_pct * 0.015), 2)
            else:
                c.risk_budget_pct = round(min(0.40, 0.10 + c.allocation_pct * 0.010), 2)

            if c.classification == "vault_candidate":
                c.deployment_tier = "A"
            elif c.classification == "research_candidate":
                c.deployment_tier = "B"
            else:
                c.deployment_tier = "C"

    return selected


def _write_csv(candidates: List[Candidate]) -> None:
    rows = [c.to_dict() for c in candidates]
    if not rows:
        OUTPUT_CSV.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_alpha_factory_report() -> Dict[str, Any]:
    raw_candidates = _extract_candidates()
    selected = _diversify_candidates(raw_candidates)
    selected = _normalize_allocations(selected)

    market_mix: Dict[str, int] = defaultdict(int)
    symbol_mix: Dict[str, int] = defaultdict(int)

    for c in selected:
        market_mix[c.market] += 1
        symbol_mix[c.symbol] += 1

    report = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "filters": {
            "min_sharpe": MIN_SHARPE,
            "min_return_pct": MIN_RETURN,
            "min_trades": MIN_TRADES,
            "min_expectancy_proxy": MIN_EXPECTANCY,
            "max_candidates_per_bucket": MAX_CANDIDATES_PER_BUCKET,
            "max_final_candidates": MAX_FINAL_CANDIDATES,
            "max_weight_per_symbol": MAX_WEIGHT_PER_SYMBOL,
            "max_weight_per_candidate": MAX_WEIGHT_PER_CANDIDATE,
        },
        "positioning": {
            "primary_market": "futures",
            "primary_symbols": ["MNQ", "MES", "MYM"],
            "primary_timeframes": ["1H", "4H"],
            "secondary_market": "crypto",
        },
        "raw_candidate_count": len(raw_candidates),
        "selected_candidate_count": len(selected),
        "market_mix": dict(market_mix),
        "symbol_mix": dict(symbol_mix),
        "selected_candidates": [c.to_dict() for c in selected],
    }
    return report, selected


def print_report(report: Dict[str, Any]) -> None:
    print("\n════════════════════════════════════════════════════════════")
    print("RBI ALPHA FACTORY")
    print("════════════════════════════════════════════════════════════")
    print(f"Generated: {report['generated_at_utc']}")
    print(f"Raw candidates: {report['raw_candidate_count']}")
    print(f"Selected candidates: {report['selected_candidate_count']}")
    print(f"Market mix: {report.get('market_mix', {})}")
    print()

    selected = report.get("selected_candidates", [])
    if not selected:
        print("No candidates passed the alpha factory filters.")
    else:
        print("Selected portfolio candidates:")
        for idx, row in enumerate(selected, start=1):
            print(
                f"  {idx}. {row['strategy_name']} | {row['symbol']} {row['timeframe']} | "
                f"Return {row['return_pct']:.1f}% | Sharpe {row['sharpe']:.2f} | "
                f"Trades {row['num_trades']} | {row['classification']} | "
                f"Alloc {row['allocation_pct']:.2f}% | Risk {row['risk_budget_pct']:.2f}% | Tier {row['deployment_tier']}"
            )

    print(f"\nSaved JSON: {OUTPUT_JSON}")
    print(f"Saved CSV:  {OUTPUT_CSV}")


def main() -> None:
    report, selected = build_alpha_factory_report()
    OUTPUT_JSON.write_text(json.dumps(report, indent=2))
    _write_csv(selected)
    print_report(report)


if __name__ == "__main__":
    main()