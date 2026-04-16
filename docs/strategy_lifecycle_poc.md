# Strategy Lifecycle POC

This is a draft only.

It does not change the live production workflow.

The goal is to reshape the current strategy process into a clearer product-style lifecycle:

`research -> test -> dev -> prod`


## Why This Helps

Right now the intent is already mostly there:

- `src/strategies/manual` is the research and idea-lab area.
- `src/strategies/deployed` is the live deployment area.
- `src/strategies/vault` is acting like a curated middle layer.

The main gap is not capability. It is naming, stage boundaries, and promotion workflow.

That matters because over time:

- research code and live code start to feel too close together,
- version sprawl becomes harder to reason about,
- promotion criteria become implicit instead of obvious,
- rollback and audit trails get harder when a model moves from idea to production.


## Current State

Observed structure today:

- `src/strategies/manual`
  - manual ideas
  - backtest variants
  - reporting/export scripts
  - many versioned strategy files like `ict_multi_setup_v4xx.py`
- `src/strategies/vault`
  - curated/backtest-approved strategies
  - generated strategy files
  - `vault_index.json`
- `src/strategies/deployed`
  - production-capable runtime strategy packages like `ict_fractal`

That translates roughly to:

- `manual` ~= research
- `vault` ~= dev/candidate registry
- `deployed` ~= prod


## Proposed Lifecycle

### 1. Research

Purpose:

- create new ideas,
- iterate quickly,
- run manual experiments,
- produce candidate trade logs and reports.

Suggested home:

- `src/strategies/research`

What would live there:

- idea notebooks or scripts
- manual trade hypothesis files
- one-off experiment variants
- analysis/export helpers
- experiment notes and assumptions

Current mapping:

- mostly your existing `src/strategies/manual`

Suggested internal structure:

```text
src/strategies/research/
  ict/
    multi_setup/
      ideas/
      experiments/
      reports/
      archived/
```


### 2. Test

Purpose:

- formalize a research candidate,
- replay it on historical data,
- validate metrics,
- make the candidate reproducible.

Suggested home:

- `src/strategies/test`

What would live there:

- stable backtest harnesses
- scenario tests
- trade-log builders
- metric calculators
- promotion reports

Key difference from research:

- research is flexible and exploratory,
- test is repeatable and comparable.

Suggested internal structure:

```text
src/strategies/test/
  ict_fractal/
    backtests/
    fixtures/
    reports/
    promotion_checks.py
```


### 3. Dev

Purpose:

- package a candidate into a clean runnable model,
- connect it to shared interfaces,
- forward test it without treating it as live prod yet.

Suggested home:

- `src/strategies/dev`

What would live there:

- candidate runtime package
- config defaults for paper/forward-test mode
- signal normalization
- dry-run execution adapters
- monitoring hooks for validation

Current mapping:

- much of what `vault` is doing today,
- plus any forward-test-ready packaged strategy before production promotion.

Suggested internal structure:

```text
src/strategies/dev/
  ict_fractal/
    config.py
    live_model.py
    execution_sim.py
    monitoring.py
    README.md
```


### 4. Prod

Purpose:

- live execution only,
- minimal moving parts,
- locked-down configuration,
- clear rollback path.

Suggested home:

- `src/strategies/prod`

What would live there:

- live-only strategy package
- production config
- execution adapter
- state/logging files
- operational readme/runbook

Current mapping:

- your existing `src/strategies/deployed`

Suggested internal structure:

```text
src/strategies/prod/
  ict_fractal/
    app.py
    config.py
    execution.py
    live_model.py
    state.py
    monitoring.py
    README_RUN.md
```


## Promotion Workflow

Here is the clean workflow I would aim for:

```text
idea
  ->
research experiment
  ->
repeatable backtest candidate
  ->
promotion review
  ->
dev forward test
  ->
paper validation
  ->
prod deployment
```

Recommended gates between stages:

### Research -> Test

Require:

- clear strategy name
- fixed entry/exit rules
- fixed market and timeframe
- reproducible input dataset
- saved summary report

### Test -> Dev

Require:

- minimum trade count
- minimum Sharpe
- minimum return
- acceptable drawdown
- no broken assumptions in event generation
- signal schema defined

### Dev -> Prod

Require:

- forward test period completed
- execution mode validated in paper
- force-flat and state recovery checked
- logs and monitoring confirmed
- rollback plan documented


## Recommended Folder Model

This is the cleanest end-state without mixing intent:

```text
src/strategies/
  research/
  test/
  dev/
  prod/
  shared/
```

Where:

- `research` is fast iteration
- `test` is controlled validation
- `dev` is packaged pre-production
- `prod` is live runtime
- `shared` is reusable utilities, schemas, constants, reporting logic


## Low-Risk Migration Plan

Because the system is live, I would not rename everything immediately.

I would do this in phases:

### Phase 0: No behavior change

- keep all current folders as-is
- document the lifecycle
- start using stage labels in file names and READMEs

### Phase 1: Introduce aliases, not moves

- create new folders gradually:
  - `src/strategies/research`
  - `src/strategies/test`
  - `src/strategies/dev`
  - `src/strategies/prod`
- add README files describing what belongs in each
- keep current code paths untouched

### Phase 2: New work follows new structure

- new strategies start in `research`
- validated backtests move to `test`
- packaged forward-test versions move to `dev`
- only promoted models enter `prod`

### Phase 3: Migrate old strategy families slowly

- move one strategy family at a time
- leave compatibility imports if needed
- do not migrate live systems during active trading windows


## Suggested Naming Improvement

Current versioned file naming like `v452`, `v470`, `v473` is useful, but it gets hard to scan at scale.

I would keep version numbers, but add stage and model identity:

Examples:

- `research_ict_multi_setup_v474.py`
- `test_ict_multi_setup_v474_backtest.py`
- `dev_ict_fractal_v474/live_model.py`
- `prod_ict_fractal/live_model.py`

Or, better yet, use folders:

```text
src/strategies/research/ict_multi_setup/v474/
src/strategies/test/ict_multi_setup/v474/
src/strategies/dev/ict_fractal/v474/
src/strategies/prod/ict_fractal/
```

That keeps versioning visible without turning the root folder into a wall of files.


## Improvements I’d Recommend

These are the highest-value improvements with low operational risk:

### 1. Add stage-level READMEs

Each stage folder should answer:

- what belongs here,
- what does not belong here,
- how something gets promoted out,
- who or what consumes its outputs.

### 2. Separate shared logic from stage logic

Anything reused across research, test, dev, and prod should move into a shared area.

Example candidates:

- signal schemas
- instrument metadata
- reporting utilities
- event normalization
- common config parsing

This reduces copy-paste drift between manual and deployed versions.

### 3. Add promotion manifests

For each promoted strategy, save a small JSON or YAML file recording:

- source strategy/version
- test metrics
- promotion date
- approved stage
- dataset used
- operator notes

That creates an audit trail.

### 4. Keep prod minimal

Production should contain only:

- the model actually being run,
- the exact config it needs,
- execution code,
- monitoring,
- state management.

Prod should not be the place for idea iteration.

### 5. Add explicit paper/live split inside prod packages

Even in production-ready packages, make mode boundaries obvious:

- `paper`
- `live`

That helps prevent accidental live behavior when testing.

### 6. Add a runbook per production model

For each live model, document:

- start command
- stop command
- required env vars
- expected logs
- failure modes
- safe rollback steps

### 7. Add a promotion checklist script later

Not now, but eventually a small validator could confirm:

- required files exist,
- metrics pass thresholds,
- config variables are present,
- force-flat settings are enabled,
- logging paths are writable.


## POC Folder Sketch

If we were drafting the future structure without touching live code, I would sketch it like this:

```text
src/strategies/
  research/
    ict_multi_setup/
      v474/
        hypothesis.md
        strategy.py
        analyze.py
        exports.py

  test/
    ict_multi_setup/
      v474/
        backtest.py
        metrics.py
        promotion_report.md

  dev/
    ict_fractal/
      v474/
        config.py
        live_model.py
        paper_execution.py
        README.md

  prod/
    ict_fractal/
      app.py
      config.py
      execution.py
      live_model.py
      state.py
      monitoring.py
      README_RUN.md

  shared/
    schemas/
    reporting/
    market_data/
    risk/
```


## Best First Step

If we want to be very safe, the first implementation step should be:

1. keep `manual` and `deployed` untouched,
2. add lifecycle documentation,
3. add empty stage folders with READMEs only,
4. move only new strategy work into the new structure,
5. migrate old strategies later one family at a time.

That gives you the product workflow you want without introducing production risk.


## Short Mapping From Your Current Language

Your current mental model:

- `manual` = write ideas and backtest
- `deployed` = final model after backtest

Proposed product model:

- `research` = write ideas
- `test` = formal backtest and validation
- `dev` = forward-test package
- `prod` = live deployed model

This is a natural evolution of what you already have, not a totally new system.
