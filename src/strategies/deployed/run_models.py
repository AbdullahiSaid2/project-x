"""
How to run this
---------------
Place this file at:
    trading_system/src/strategies/deployed/run_models.py

This runner lets you:
- run one model
- run many models
- run all registered models
- attach a prop-firm profile
- switch mode (dry/demo/live)
- toggle live order sending
- pass environment variables into each model app
- log pre-entry allow/block reasons from the model process output

Typical commands
----------------
List available models:
    python src/strategies/deployed/run_models.py --list-models

Run one model in dry mode:
    python src/strategies/deployed/run_models.py --models ict_fractal --prop-profile apex_pa_50k --mode dry

Run one model in live mode, but do not allow live orders:
    python src/strategies/deployed/run_models.py --models ict_fractal --prop-profile apex_pa_50k --mode live --no-live-orders

Run one model in live mode with live orders enabled:
    python src/strategies/deployed/run_models.py --models ict_fractal --prop-profile apex_pa_50k --mode live --live-orders

Run multiple models:
    python src/strategies/deployed/run_models.py --models ict_fractal,v473 --prop-profile apex_pa_50k --mode live --live-orders

Run all registered models:
    python src/strategies/deployed/run_models.py --models all --prop-profile apex_pa_50k --mode live --live-orders

Run models in parallel:
    python src/strategies/deployed/run_models.py --models all --prop-profile none --mode demo --parallel

Important integration contract
------------------------------
This runner does not place orders by itself. It launches each model's app.py and
injects these environment variables:

    DEPLOY_MODE
    PROP_PROFILE
    LIVE_ORDERS
    MODEL_NAME

Your deployed model app should read them, for example:
    mode = os.getenv("DEPLOY_MODE", "dry")
    prop_profile = os.getenv("PROP_PROFILE", "none")
    live_orders = os.getenv("LIVE_ORDERS", "0") == "1"

If the model prints JSON or log lines containing:
    "pre_entry_allowed"
    "pre_entry_blocked"
    "blocked_reason"
the runner will surface them clearly in stdout.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Dict, Iterable, List

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]

# Make sibling and package imports both work, whether run as:
#   python src/strategies/deployed/run_models.py
# or:
#   python -m src.strategies.deployed.run_models
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from model_registry import DEPLOYED_MODELS, list_models
except ModuleNotFoundError:
    from src.strategies.deployed.model_registry import DEPLOYED_MODELS, list_models


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one or more deployed strategy models.")
    p.add_argument("--models", default="ict_fractal",
                   help="Comma-separated model names, or 'all'. Example: ict_fractal,v473 or all")
    p.add_argument("--prop-profile", default="none",
                   help="Prop-firm profile name to expose to model apps, e.g. apex_pa_50k")
    p.add_argument("--mode", default="dry", choices=["dry", "demo", "live"],
                   help="Deployment mode passed to the model app")
    p.add_argument("--live-orders", action="store_true",
                   help="Enable live order sending in the model app")
    p.add_argument("--no-live-orders", action="store_true",
                   help="Force-disable live order sending in the model app")
    p.add_argument("--parallel", action="store_true",
                   help="Run selected models in parallel subprocesses")
    p.add_argument("--list-models", action="store_true",
                   help="List registered deployed models and exit")
    return p.parse_args()


def resolve_models(models_arg: str) -> List[str]:
    if models_arg.strip().lower() == "all":
        return list_models()

    requested = [m.strip() for m in models_arg.split(",") if m.strip()]
    unknown = [m for m in requested if m not in DEPLOYED_MODELS]
    if unknown:
        raise SystemExit(
            f"Unknown model(s): {', '.join(unknown)}\nAvailable models: {', '.join(list_models())}"
        )
    return requested


def build_env(model_name: str, mode: str, prop_profile: str, live_orders: bool) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MODEL_NAME"] = model_name
    env["DEPLOY_MODE"] = mode
    env["PROP_PROFILE"] = prop_profile
    env["LIVE_ORDERS"] = "1" if live_orders else "0"
    return env


def emit_interpreted_line(model_name: str, line: str) -> None:
    text = line.rstrip("\n")
    stripped = text.strip()

    if not stripped:
        return

    # Try JSON first if model logs structured dict/json
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            if payload.get("pre_entry_allowed") is True:
                print(f"[{model_name}] PRE-ENTRY ALLOWED | {payload}")
                return
            if payload.get("pre_entry_blocked") is True or payload.get("blocked_reason"):
                reason = payload.get("blocked_reason", "unknown")
                print(f"[{model_name}] PRE-ENTRY BLOCKED | reason={reason} | {payload}")
                return
            print(f"[{model_name}] {payload}")
            return
    except Exception:
        pass

    # Fallback plain-text interpretation
    lower = stripped.lower()
    if "pre_entry_allowed" in lower:
        print(f"[{model_name}] PRE-ENTRY ALLOWED | {stripped}")
    elif "pre_entry_blocked" in lower or "blocked_reason" in lower:
        print(f"[{model_name}] PRE-ENTRY BLOCKED | {stripped}")
    else:
        print(f"[{model_name}] {stripped}")


def stream_subprocess_output(model_name: str, proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        emit_interpreted_line(model_name, line)


def run_one_model(model_name: str, mode: str, prop_profile: str, live_orders: bool) -> int:
    record = DEPLOYED_MODELS[model_name]
    app_path = Path(record["app_path"]).resolve()

    if not app_path.exists():
        print(f"[{model_name}] ERROR: app not found at {app_path}")
        return 1

    env = build_env(model_name, mode, prop_profile, live_orders)

    print(
        f"[{model_name}] launching | mode={mode} | prop_profile={prop_profile} | "
        f"live_orders={'on' if live_orders else 'off'} | app={app_path}"
    )

    proc = subprocess.Popen(
        [sys.executable, str(app_path)],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        stream_subprocess_output(model_name, proc)
    finally:
        return_code = proc.wait()

    print(f"[{model_name}] exited with code {return_code}")
    return return_code


def run_many_serial(models: Iterable[str], mode: str, prop_profile: str, live_orders: bool) -> int:
    worst = 0
    for model_name in models:
        rc = run_one_model(model_name, mode, prop_profile, live_orders)
        worst = max(worst, rc)
    return worst


def run_many_parallel(models: Iterable[str], mode: str, prop_profile: str, live_orders: bool) -> int:
    procs: Dict[str, subprocess.Popen] = {}
    threads: List[threading.Thread] = []

    for model_name in models:
        record = DEPLOYED_MODELS[model_name]
        app_path = Path(record["app_path"]).resolve()

        if not app_path.exists():
            print(f"[{model_name}] ERROR: app not found at {app_path}")
            return 1

        env = build_env(model_name, mode, prop_profile, live_orders)
        print(
            f"[{model_name}] launching | mode={mode} | prop_profile={prop_profile} | "
            f"live_orders={'on' if live_orders else 'off'} | app={app_path}"
        )

        proc = subprocess.Popen(
            [sys.executable, str(app_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs[model_name] = proc
        t = threading.Thread(target=stream_subprocess_output, args=(model_name, proc), daemon=True)
        t.start()
        threads.append(t)

    worst = 0
    for model_name, proc in procs.items():
        rc = proc.wait()
        print(f"[{model_name}] exited with code {rc}")
        worst = max(worst, rc)

    for t in threads:
        t.join(timeout=1.0)

    return worst


def main() -> int:
    args = parse_args()

    if args.list_models:
        print("Available deployed models:")
        for name in list_models():
            desc = DEPLOYED_MODELS[name].get("description", "")
            print(f"  - {name}: {desc}")
        return 0

    live_orders = args.live_orders and not args.no_live_orders
    models = resolve_models(args.models)

    print(
        f"Selected models: {', '.join(models)}\n"
        f"Mode: {args.mode}\n"
        f"Prop profile: {args.prop_profile}\n"
        f"Live orders: {'enabled' if live_orders else 'disabled'}\n"
        f"Parallel: {'yes' if args.parallel else 'no'}"
    )

    if args.parallel and len(models) > 1:
        return run_many_parallel(models, args.mode, args.prop_profile, live_orders)

    return run_many_serial(models, args.mode, args.prop_profile, live_orders)


if __name__ == "__main__":
    raise SystemExit(main())
