"""
How to run this
---------------
Place this file at:
    trading_system/src/strategies/deployed/model_registry.py

Purpose
-------
Central registry for deployed strategy apps. The multi-model runner imports this
file and uses it to find each model's app.py entrypoint.

Examples
--------
List models from Python:
    from src.strategies.deployed.model_registry import list_models
    print(list_models())

Add a new deployed model:
    DEPLOYED_MODELS["my_model"] = {
        "app_path": DEPLOYED_ROOT / "my_model" / "app.py",
        "description": "My deployed model",
        "supports_prop_guard": True,
    }
"""
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
    # Add more models here as you deploy them.
    # "v473": {
    #     "app_path": DEPLOYED_ROOT / "v473" / "app.py",
    #     "description": "V473 deployed loop",
    #     "supports_prop_guard": True,
    # },
}

def list_models() -> List[str]:
    return sorted(DEPLOYED_MODELS.keys())
