
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent
main=ROOT/'prop_firm_lifecycle_profiles.yaml'
patch=ROOT/'prop_firm_lifecycle_profiles_ict_payout_cycle_patch.yaml'
existing=yaml.safe_load(main.read_text()) if main.exists() else {}
addon=yaml.safe_load(patch.read_text()) or {}
existing=existing or {}; existing.update(addon)
main.write_text(yaml.safe_dump(existing, sort_keys=False))
print(f"Merged ICT Payout Cycle profile into {main}")
for k in addon: print(f"  - {k}")
