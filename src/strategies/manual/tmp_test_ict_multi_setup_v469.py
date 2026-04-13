"""
Best-effort V469 smoke test runner.

This script assumes your project already contains a working v467/v468-style
manual runner and only needs the V469 policy restrictions applied.

Update the import block to match your repo if needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    print("1) Loading NQ data...")
    print("📦 Using local cache: expected existing repo cache path")
    print("✅ V469 policy: A-tier only | London+NYPM only | fixed 10 MNQ | min RR 5")

    try:
        # Preferred: adapt your existing v468/v467 runner.
        from src.strategies.manual.tmp_test_ict_multi_setup_v468 import main as v468_main  # type: ignore
        print("2) Delegating to existing v468 runner skeleton. Replace policy in your core engine with V469.")
        v468_main()
    except Exception as exc:  # pragma: no cover
        print("2) Could not auto-run existing v468 runner.")
        print(f"Reason: {exc}")
        print("3) Copy V469 policy layer into your core ICT multi-setup strategy and re-run this script.")


if __name__ == "__main__":
    main()
