from __future__ import annotations

"""
connectivity_checks.py

What this file is about
-----------------------
This is a single preflight / connectivity check script for your deployed trading setup.

It is designed for your actual route:

    Python bot -> PickMyTrade -> Tradovate / Apex

So this script focuses on:
- environment loading
- deployed model registry
- deployed app file existence
- Databento client sanity
- PickMyTrade-related runtime sanity
- optional PickMyTrade host reachability
- syntax/import sanity for deployed model app files
- optional dual-model dry-run launcher check

Why this is one file
--------------------
You asked for a single script so you do not have to run multiple snippets or manage
multiple connectivity files. This file is meant to be your one-stop preflight check.

Do we need a Tradovate auth test here?
--------------------------------------
Usually: NO, not for your current setup.

Because your execution path is:

    bot -> PickMyTrade -> Tradovate

your bot is not directly responsible for placing orders into Tradovate.
So the most relevant checks are:
- PickMyTrade token/account/strategy config
- deployed execution path sanity
- data/feed sanity
- deployed model/runtime sanity

A direct Tradovate auth test is only useful if:
1. you still use direct Tradovate API anywhere else in the repo, or
2. you want a separate lower-level broker health test outside PickMyTrade.

What this script checks step by step
------------------------------------
STEP 1
- Checks whether the important environment variables are actually loaded into the shell.

STEP 2
- Prints the important runtime values used by the deployment runner.

STEP 3
- Imports the deployed model registry and confirms the expected models are registered.

STEP 4
- Confirms the deployed model app paths exist on disk.

STEP 5
- Performs basic Python syntax compilation checks on deployed app files.

STEP 6
- Checks Databento SDK import and Databento Historical client creation.

STEP 7
- Checks PickMyTrade-related env vars and bridge route sanity.

STEP 8
- Optionally checks whether the PickMyTrade host is reachable if a base URL is provided.

STEP 9
- Optionally runs a dual-model dry-run launcher check.
- This launches both deployed models together using:
      --models ict_fractal,top_bottom_ticking
      --mode live
      --no-live-orders
- It only runs for a short timeout and is meant to confirm that both processes can
  start together without immediate import/runtime failure.
- It does NOT send real orders.
- It does NOT stay running forever.
- It is just a startup sanity test.

What this script does NOT do
----------------------------
- It does not send real orders
- It does not call live PickMyTrade order endpoints
- It does not call live Tradovate login
- It does not perform a full broker-side order round trip

How to run it
-------------
From the project root:

    cd /Users/Abdullahi/trading-project/trading_system
    source venv/bin/activate
    set -a
    source .env
    set +a
    PYTHONPATH=. python src/strategies/deployed/connectivity_checks.py

Optional dry-run launcher test
------------------------------
By default, the launcher test is skipped.

To enable it, set either:

    CONNECTIVITY_RUN_DRY_LAUNCH=1

or:

    CONNECTIVITY_RUN_DRY_LAUNCH=true

Optional dry-run timeout
------------------------
Default timeout is 25 seconds.

You can override it with:

    CONNECTIVITY_DRY_LAUNCH_TIMEOUT=40

How to read the exit code
-------------------------
0 = all checks passed
1 = a structural / import / path / client / host reachability / dry-run startup check failed
2 = required environment variables are missing from the shell
"""

import os
import py_compile
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def env_status(name: str, value: str | None) -> None:
    print(f"{name}: {'SET' if value else 'MISSING'}")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == '':
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def reachable_host_from_url(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    if not host:
        return False, 'No hostname parsed from URL'
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f'TCP connect OK to {host}:{port}'
    except Exception as exc:
        return False, f'TCP connect FAILED to {host}:{port} -> {exc}'


def run_dual_model_dry_launch(timeout_seconds: int) -> tuple[bool, str]:
    """
    Short startup sanity test for both deployed models together.

    This runs the deployed launcher with:
    - ict_fractal
    - top_bottom_ticking
    - live mode
    - no live orders

    A timeout is treated as success if the process stayed alive and no immediate
    failure markers appeared, because the runner is designed to loop forever.
    """
    prop_profile = os.getenv('PROP_PROFILE', 'none').strip() or 'none'

    cmd = [
        sys.executable,
        'src/strategies/deployed/run_models.py',
        '--models',
        'ict_fractal,top_bottom_ticking',
        '--prop-profile',
        prop_profile,
        '--mode',
        'live',
        '--no-live-orders',
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = (result.stdout or '') + ('\n' + result.stderr if result.stderr else '')
        if result.returncode != 0:
            return False, (
                f'Dry launch FAILED with exit code {result.returncode}\n'
                f'--- output start ---\n{output[-4000:]}\n--- output end ---'
            )

        required_markers = ['Selected models:', 'ict_fractal', 'top_bottom_ticking']
        missing_markers = [m for m in required_markers if m not in output]
        if missing_markers:
            return False, (
                'Dry launch completed but expected startup markers were missing: '
                f'{missing_markers}\n--- output start ---\n{output[-4000:]}\n--- output end ---'
            )

        return True, 'Dry launch completed successfully.'
    except subprocess.TimeoutExpired as exc:
        partial = (exc.stdout or '') + ('\n' + exc.stderr if exc.stderr else '')
        failure_markers = ['Traceback', 'ImportError', 'ModuleNotFoundError', 'NameError', 'SyntaxError', 'exited with code']
        found_failures = [m for m in failure_markers if m in partial]
        if found_failures:
            return False, (
                'Dry launch timed out but output contained failure markers: '
                f'{found_failures}\n--- output start ---\n{partial[-4000:]}\n--- output end ---'
            )

        required_markers = ['Selected models:', 'ict_fractal', 'top_bottom_ticking']
        missing_markers = [m for m in required_markers if m not in partial]
        if missing_markers:
            return False, (
                'Dry launch timed out before confirming expected startup markers: '
                f'{missing_markers}\n--- output start ---\n{partial[-4000:]}\n--- output end ---'
            )

        return True, (
            f'Dry launch startup check PASSED. Runner stayed alive for {timeout_seconds}s '
            'without immediate failure. Timeout is expected because deployed runners loop continuously.'
        )
    except Exception as exc:
        return False, f'Dry launch check FAILED before completion -> {exc}'


def main() -> int:
    print_header('STEP 1 - Environment Variables Loaded Into Shell')
    required_env = [
        'DEPLOY_MODE',
        'PROP_PROFILE',
        'LIVE_ORDERS',
        'DATA_SOURCE',
        'DATABENTO_API_KEY',
        'PICKMYTRADE_TOKEN',
        'PICKMYTRADE_ACCOUNT_ID',
    ]
    optional_env = [
        'MODEL_NAME',
        'PICKMYTRADE_STRATEGY_ID',
        'PICKMYTRADE_BASE_URL',
        'ICT_FRACTAL_BRIDGE_ROUTE',
        'TOP_BOTTOM_TICKING_EXECUTION_MODE',
        'ICT_FRACTAL_EXECUTION_MODE',
        'CONNECTIVITY_RUN_DRY_LAUNCH',
        'CONNECTIVITY_DRY_LAUNCH_TIMEOUT',
    ]

    missing_required: list[str] = []
    for key in required_env:
        value = os.getenv(key)
        env_status(key, value)
        if not value:
            missing_required.append(key)

    print('\nOptional values:')
    for key in optional_env:
        env_status(key, os.getenv(key))

    print_header('STEP 2 - Important Runtime Values')
    print('PROJECT_ROOT:', PROJECT_ROOT)
    print('DEPLOY_MODE:', os.getenv('DEPLOY_MODE'))
    print('PROP_PROFILE:', os.getenv('PROP_PROFILE'))
    print('LIVE_ORDERS:', os.getenv('LIVE_ORDERS'))
    print('MODEL_NAME:', os.getenv('MODEL_NAME'))
    print('DATA_SOURCE:', os.getenv('DATA_SOURCE'))
    print('PICKMYTRADE_ACCOUNT_ID:', os.getenv('PICKMYTRADE_ACCOUNT_ID'))
    print('PICKMYTRADE_STRATEGY_ID:', os.getenv('PICKMYTRADE_STRATEGY_ID'))
    print('ICT_FRACTAL_BRIDGE_ROUTE:', os.getenv('ICT_FRACTAL_BRIDGE_ROUTE'))

    print_header('STEP 3 - Deployed Model Registry Import')
    try:
        from src.strategies.deployed.model_registry import DEPLOYED_MODELS
        print('Registry import: OK')
        print('Models found:', sorted(DEPLOYED_MODELS.keys()))
    except Exception as exc:
        print('Registry import FAILED:', exc)
        return 1

    expected_models = {'ict_fractal', 'top_bottom_ticking'}
    missing_models = expected_models - set(DEPLOYED_MODELS.keys())
    if missing_models:
        print('Missing expected deployed models from registry:', sorted(missing_models))
        return 1

    print_header('STEP 4 - Deployed Model App Paths')
    bad_paths: list[tuple[str, Path]] = []
    for name, cfg in DEPLOYED_MODELS.items():
        app_path = Path(cfg['app_path'])
        exists = app_path.exists()
        print(f"{name}: {'OK' if exists else 'MISSING'} -> {app_path}")
        if not exists:
            bad_paths.append((name, app_path))

    if bad_paths:
        print('\nMissing deployed app files:')
        for name, app_path in bad_paths:
            print(f'- {name}: {app_path}')
        return 1

    print_header('STEP 5 - Python Syntax Compile Check')
    try:
        for name, cfg in DEPLOYED_MODELS.items():
            app_path = Path(cfg['app_path'])
            py_compile.compile(str(app_path), doraise=True)
            print(f'{name}: syntax compile OK')
    except Exception as exc:
        print('Syntax compile FAILED:', exc)
        return 1

    print_header('STEP 6 - Databento SDK / Client Check')
    try:
        import databento as db  # type: ignore
        api_key = os.getenv('DATABENTO_API_KEY')
        if not api_key:
            print('Databento client check skipped: missing DATABENTO_API_KEY')
            return 2
        _ = db.Historical(api_key)
        print('Databento SDK import: OK')
        print('Databento Historical client: OK')
    except Exception as exc:
        print('Databento check FAILED:', exc)
        return 1

    print_header('STEP 7 - PickMyTrade Route Sanity')
    pmt_token = os.getenv('PICKMYTRADE_TOKEN')
    pmt_account = os.getenv('PICKMYTRADE_ACCOUNT_ID')
    pmt_strategy = os.getenv('PICKMYTRADE_STRATEGY_ID')
    bridge_route = os.getenv('ICT_FRACTAL_BRIDGE_ROUTE')

    print('PICKMYTRADE_TOKEN:', 'SET' if pmt_token else 'MISSING')
    print('PICKMYTRADE_ACCOUNT_ID:', pmt_account if pmt_account else 'MISSING')
    print('PICKMYTRADE_STRATEGY_ID:', pmt_strategy if pmt_strategy else 'MISSING')
    print('ICT_FRACTAL_BRIDGE_ROUTE:', bridge_route if bridge_route else 'MISSING')

    if bridge_route and bridge_route.lower() != 'pickmytrade':
        print("WARNING: ICT_FRACTAL_BRIDGE_ROUTE is not set to 'pickmytrade'")

    print_header('STEP 8 - Optional PickMyTrade Host Reachability')
    pmt_base_url = os.getenv('PICKMYTRADE_BASE_URL', '').strip()
    if pmt_base_url:
        ok, message = reachable_host_from_url(pmt_base_url)
        print('PICKMYTRADE_BASE_URL:', pmt_base_url)
        print(message)
        if not ok:
            return 1
    else:
        print('Skipped: PICKMYTRADE_BASE_URL not set.')
        print('If you want host reachability checked too, add for example:')
        print('  PICKMYTRADE_BASE_URL=https://<your-pickmytrade-host>')

    print_header('STEP 9 - Optional Dual-Model Dry-Run Launcher Check')
    run_dry_launch = env_bool('CONNECTIVITY_RUN_DRY_LAUNCH', False)
    dry_launch_timeout = env_int('CONNECTIVITY_DRY_LAUNCH_TIMEOUT', 25)

    print('CONNECTIVITY_RUN_DRY_LAUNCH:', run_dry_launch)
    print('CONNECTIVITY_DRY_LAUNCH_TIMEOUT:', dry_launch_timeout)

    if run_dry_launch:
        ok, message = run_dual_model_dry_launch(dry_launch_timeout)
        print(message)
        if not ok:
            return 1
    else:
        print('Skipped: dry-run launcher test disabled.')
        print('To enable it, set:')
        print('  CONNECTIVITY_RUN_DRY_LAUNCH=1')
        print('Optional:')
        print('  CONNECTIVITY_DRY_LAUNCH_TIMEOUT=25')

    print_header('FINAL SUMMARY')
    if missing_required:
        print('Some required environment variables are missing from the current shell.')
        print('Missing required values:')
        for key in missing_required:
            print(f'- {key}')
        print('\nThis usually means your `.env` file has not been loaded into the shell yet.')
        print('Run:')
        print('  set -a')
        print('  source .env')
        print('  set +a')
        return 2

    print('All single-file preflight checks passed.')
    print('This means:')
    print('- shell env is loaded')
    print('- deployed models are registered')
    print('- deployed app files exist')
    print('- app files compile')
    print('- Databento client can initialize')
    print('- PickMyTrade runtime config is present')
    if run_dry_launch:
        print('- both models can start together in dry-run launcher mode')
    print('\nFor your current architecture, this is the right main connectivity check path.')
    print('A direct Tradovate auth test is not required unless you want separate broker-layer validation.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
