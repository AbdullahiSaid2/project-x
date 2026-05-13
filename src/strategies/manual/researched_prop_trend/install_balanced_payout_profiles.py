from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
profiles_path = ROOT / "prop_firm_lifecycle_profiles.yaml"
patch_path = ROOT / "prop_firm_lifecycle_profiles_balanced_patch.yaml"

existing = yaml.safe_load(profiles_path.read_text()) if profiles_path.exists() else {}
patch = yaml.safe_load(patch_path.read_text())

existing = existing or {}
existing.update(patch)

profiles_path.write_text(yaml.safe_dump(existing, sort_keys=False), encoding="utf-8")

print(f"Merged {len(patch)} profiles into {profiles_path}")
for name in patch:
    print(f"  - {name}")
