from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DEPLOY_ROOT = BASE_DIR / "src" / "strategies" / "deployed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one or more deployed models.")
    parser.add_argument("--models", required=True, help="Comma-separated model names, e.g. ict_fractal,top_bottom_ticking")
    parser.add_argument("--prop-profile", default="apex_pa_50k", help="Prop profile name")
    parser.add_argument("--mode", default="live", choices=["live", "demo", "paper"], help="Runtime mode")
    parser.add_argument("--no-live-orders", action="store_true", help="Disable real order routing")
    parser.add_argument("--parallel", action="store_true", help="Run models in parallel")
    return parser.parse_args()


def _model_deploy_dir(model: str) -> Path:
    return DEPLOY_ROOT / model


def _model_app_path(model: str) -> Path:
    return _model_deploy_dir(model) / "app.py"


def _runtime_stack_summary(model: str) -> str:
    deploy_dir = _model_deploy_dir(model)
    parts = []
    for name in ["app.py", "config.py", "execution.py", "state.py", "live_model.py"]:
        parts.append(name if (deploy_dir / name).exists() else f"{name} [missing]")
    return " + ".join(parts)


def print_global_summary(models: list[str], mode: str, prop_profile: str, live_orders: bool, parallel: bool) -> None:
    print(f"Selected models: {', '.join(models)}")
    print(f"Mode: {mode}")
    print(f"Prop profile: {prop_profile}")
    print(f"Live orders: {'enabled' if live_orders else 'disabled'}")
    print(f"Parallel: {'yes' if parallel else 'no'}")


def print_model_startup_summary(model: str, mode: str, prop_profile: str, live_orders: bool) -> None:
    deploy_dir = _model_deploy_dir(model)
    app_path = _model_app_path(model)
    print(f"[{model}] startup summary")
    print(f"[{model}]   deploy_dir: {deploy_dir}")
    print(f"[{model}]   app: {app_path}")
    print(f"[{model}]   mode: {mode}")
    print(f"[{model}]   prop_profile: {prop_profile}")
    print(f"[{model}]   live_orders: {'on' if live_orders else 'off'}")
    print(f"[{model}]   runtime_stack: {_runtime_stack_summary(model)}")


def launch_model(model: str, mode: str, prop_profile: str, live_orders: bool) -> subprocess.Popen:
    app_path = _model_app_path(model)
    if not app_path.exists():
        raise FileNotFoundError(f"Missing app.py for model '{model}' at {app_path}")

    print_model_startup_summary(model, mode, prop_profile, live_orders)
    print(f"[{model}] launching | mode={mode} | prop_profile={prop_profile} | live_orders={'on' if live_orders else 'off'} | app={app_path}")

    cmd = [sys.executable, str(app_path)]
    env = dict(**__import__("os").environ)
    env["MODEL_RUNTIME_MODE"] = mode
    env["PROP_PROFILE"] = prop_profile
    env["LIVE_ORDERS_ENABLED"] = "1" if live_orders else "0"

    return subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def stream_output(model: str, proc: subprocess.Popen) -> int:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{model}] {line}", end="")
    return proc.wait()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    live_orders = not args.no_live_orders

    print_global_summary(models, args.mode, args.prop_profile, live_orders, args.parallel)

    if args.parallel:
        procs = {}
        for model in models:
            procs[model] = launch_model(model, args.mode, args.prop_profile, live_orders)

        exit_code = 0
        for model, proc in procs.items():
            rc = stream_output(model, proc)
            if rc != 0:
                print(f"[{model}] exited with code {rc}")
                exit_code = rc
        raise SystemExit(exit_code)

    exit_code = 0
    for model in models:
        proc = launch_model(model, args.mode, args.prop_profile, live_orders)
        rc = stream_output(model, proc)
        if rc != 0:
            print(f"[{model}] exited with code {rc}")
            exit_code = rc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
