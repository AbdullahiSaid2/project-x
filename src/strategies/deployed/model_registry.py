from __future__ import annotations

from pathlib import Path
from typing import Dict, List

DEPLOYED_ROOT = Path(__file__).resolve().parent

DEPLOYED_MODELS: Dict[str, dict] = {
    "ict_fractal": {
        "app_path": DEPLOYED_ROOT / "ict_fractal" / "app.py",
        "description": "ICT fractal deployed loop",
        "supports_prop_guard": True,
    },
    "top_bottom_ticking": {
        "app_path": DEPLOYED_ROOT / "top_bottom_ticking" / "app.py",
        "description": "Top/bottom ticking deployed loop",
        "supports_prop_guard": True,
    },
}


def list_models() -> List[str]:
    return sorted(DEPLOYED_MODELS.keys())
