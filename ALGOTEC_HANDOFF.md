# ALGOTEC HANDOFF

## Overview

Algotec is a modular trading research and execution system focused primarily on futures, with crypto retained as a secondary research lane. The system has evolved from a simple idea-to-backtest loop into a more structured research platform with:

- market-specific idea files
- schema-driven strategy generation
- multi-dataset backtesting
- classification and filtering
- local historical futures caching
- alpha promotion / factory stage
- existing forward-testing execution already connected externally

The current direction is **futures-first**, especially index micros, with oil and gold added as additional futures lanes.

---

## Current State of Algotec

Algotec is now best described as a **research and strategy selection engine** rather than just a bot that converts text to trades.

### What it currently does well

- converts trading ideas into structured strategy schemas
- compiles schemas into executable strategy classes
- smoke tests generated strategies before wider testing
- runs multi-dataset parallel backtests
- scores results across multiple dimensions
- classifies strategies into candidate / reject buckets
- supports multiple market lanes:
  - index futures
  - oil
  - gold
  - crypto
- uses local Databento-backed futures cache for repeated backtests
- feeds stronger candidates toward batch reporting and alpha-factory review
- can hand off stronger ideas into an already-existing forward test / execution setup

### What is already working outside the research engine

Forward testing and execution wiring are already in place and should not be redesigned unless explicitly requested.

Existing confirmed execution stack:

- Apex
- Tradovate
- PickMyTrade bridge

So the main bottleneck is no longer execution plumbing. The bottleneck is:

- alpha quality
- idea quality
- research filtering quality

---

## Strategic Direction

### Primary lane

Futures are the primary focus, especially because of the user’s prop-firm context.

Current main instruments:

- `MNQ`
- `MES`
- `MYM`

### Added futures lanes

Additional futures instruments now supported:

- `MCL` — Micro Crude Oil
- `MGC` — Micro Gold

### Secondary lane

Crypto remains in the system but is currently treated as a side lane rather than the main deployment focus.

---

## Architecture Summary

## 1. Idea Layer

The idea layer has moved away from one generic `ideas.txt` file.

Current idea files are split by market lane:

- `src/data/ideas_indices.txt`
- `src/data/ideas_futures.txt`
- `src/data/ideas_oil.txt`
- `src/data/ideas_gold.txt`
- `src/data/ideas_crypto.txt`

### Why this split exists

Previously, mixing all ideas into one pool reduced signal quality and made research noisier.

Now ideas are separated by market behavior:

- indices behave differently from oil
- oil behaves differently from gold
- crypto behaves differently from futures

So idea separation is intentional and should be preserved.

### Practical meaning of each file

#### `ideas_indices.txt`
Focused only on index futures ideas:

- `MNQ`
- `MES`
- `MYM`

This is the cleanest primary research file for futures deployment.

#### `ideas_futures.txt`
Broader umbrella futures file. Can be used as a combined futures lane, but if the separate files exist, the split-lane approach is usually cleaner.

#### `ideas_oil.txt`
Oil-specific ideas, intended for:

- `MCL`

These ideas lean more toward:

- displacement
- expansion
- strong continuation
- raid/reclaim structures

#### `ideas_gold.txt`
Gold-specific ideas, intended for:

- `MGC`

These ideas lean more toward:

- level respect
- sweep + reclaim / reject
- pullback continuation
- cleaner structure-based setups

#### `ideas_crypto.txt`
Crypto ideas retained as a secondary research lane.

---

## 2. RBI Research Workflow

The main research runner is:

- `src/agents/rbi_parallel_v2.py`

This is the core idea-to-research engine.

### Current workflow

For each idea:

1. load idea from the selected ideas file
2. evaluate basic idea quality
3. build schema from idea
4. compile schema into executable strategy code
5. smoke test on a suitable dataset
6. route strategy to the correct dataset lane
7. run backtests across datasets
8. compute rankings for:
   - discovery
   - faithful
9. summarize results
10. classify the idea
11. optionally generate rewrite variants
12. optionally promote rewrite variants
13. save research artifacts to results

### Classification outcomes

The current system uses these classifications:

- `vault_candidate`
- `research_candidate`
- `watchlist_faithful`
- `watchlist_discovery`
- `reject`

### Important interpretation

The system rejecting many strategies is not necessarily failure. It often means the filter is doing its job. The bigger issue historically was poor / overly generic idea quality.

---

## 3. Batch Reporting

Batch summary stage:

- `src/agents/rbi_batch_report.py`

This aggregates the research output from RBI runs and produces summary views across all processed ideas.

Typical outputs include:

- classification counts
- top faithful ideas
- top discovery ideas
- batch summary json
- batch summary csv

Uploaded example files seen during this phase:

- `batch_summary_rows.csv`
- `backtest_stats_v2.csv`

---

## 4. Alpha Factory

Alpha-factory stage:

- `src/agents/rbi_alpha_factory.py`

This is the post-research promotion stage. It should be treated as the next gate after RBI batch summary, not as the first research filter.

### Recommended high-level order

1. run lane-specific RBI research
2. generate batch report
3. run alpha factory
4. review output
5. push best candidates into forward testing

---

## 5. Futures Market Routing

Futures are no longer treated as one homogeneous market.

They are split into sub-markets:

### Indices
- `MNQ`
- `MES`
- `MYM`

### Oil
- `MCL`

### Gold
- `MGC`

This routing matters because different markets suit different ideas and different behavior regimes.

### Why this matters

Historically one of the core issues was applying generic logic to the wrong market.

Examples:

- index futures: better for structured continuation and session-driven logic
- oil: stronger for displacement and expansion
- gold: better for level reactions, pullbacks, reclaim/reject structures

This separation should remain part of the system design.

---

## 6. ICT-Native Direction

Algotec is being moved toward an ICT-native and more institutional idea framework.

This does **not** mean vague discretionary ICT narration. It means systematic proxies for ICT concepts that can be backtested.

### New deterministic ICT feature layer

Supporting file:

- `src/strategies/families/ict_features.py`

This file introduces structured feature detection for concepts such as:

- FVG detection
- displacement candle detection
- swing point detection
- structure shift detection
- CISD proxy
- premium / discount context
- simplified PD array context

### New parsing path

Registry logic was updated so ideas mentioning ICT concepts can set flags.

Supporting file:

- `src/strategies/families/registry.py`

ICT-related terms in ideas are now parsed into flags such as:

- `uses_fvg`
- `uses_cisd`
- `uses_smt`
- `uses_pd_array`
- `uses_liquidity_sweep`
- `uses_displacement`
- `requires_reclaim`
- `requires_rejection`

### Compiler integration

Supporting file:

- `src/strategies/families/compiler.py`

Generated strategies now have access to the ICT feature frame, allowing entry logic to be gated by:

- FVG presence
- displacement
- CISD proxy
- premium / discount context
- reclaim / rejection confirmation
- structure shift filters

### Important limitation

The system is **not yet** fully implementing true institutional multi-asset SMT logic, explicit geometric FVG lifecycle handling, or fully sophisticated PD array reasoning. It currently uses deterministic proxies.

That is acceptable for this stage and is still a large step forward over generic breakout/retest logic.

---

## 7. Databento Historical Cache

Futures historical data is now designed to use a **local Databento parquet cache** first.

Main file:

- `src/data/databento_fetcher.py`

Main cache directory:

- `src/data/databento_cache/`

### Supported futures roots

Currently supported in Databento mapping:

- `MNQ`
- `MES`
- `MYM`
- `NQ`
- `ES`
- `YM`
- `MCL`
- `CL`
- `MGC`
- `GC`

### Cache design

The cache uses base timeframes:

- `1m`
- `1H`
- `1D`

Derived timeframes are resampled:

- `5m` from `1m`
- `15m` from `1m`
- `4H` from `1H`

### Why this design

This avoids duplicating storage unnecessarily and keeps refresh logic simpler.

### Cache behavior

Normal behavior:

- try local cache first
- if cache exists and covers the range, use it
- if cache missing or insufficient, call Databento API
- if successful, write back to local parquet cache

### Debug prints added

The fetcher now prints whether it is:

- missing cache
- using local cache
- fetching from Databento API
- retrying with an earlier end time
- writing local cache

This makes it easier to verify whether backtests are using local data or network data.

---

## 8. Databento Issues Solved

Several issues were addressed during setup.

### Environment variable issue
The Databento API key was not initially being found because the dotenv loading path needed to be explicit.

Current approach:

- explicit load of `.env` using repo-root path

### Historical end-time issue
Databento rejected requests when end time exceeded dataset available range.

Fix:

- safe historical lag window added
- retry logic added for earlier end times if Databento returns a boundary hint

### Recent-license / unavailable-range issue
Databento historical access near the most recent boundary can require different entitlements.

Fix:

- default historical lag increased
- retry parser added to respect Databento-provided safe boundary

### Result
`MCL` and `MGC` were successfully downloaded and cached for:

- `1m`
- `1H`
- `1D`

And successfully verified through resampling for:

- `5m`
- `15m`
- `4H`

This means those symbols are now usable in local backtests.

---

## 9. Current Data Status

### Confirmed working for oil and gold

Backtest data confirmed working locally for:

#### `MCL`
- `5m`
- `15m`
- `4H`

#### `MGC`
- `5m`
- `15m`
- `4H`

This confirms local resampling from cache is functioning.

### Important note on older index cache
Older `MNQ` / `MES` / `MYM` cache layout may not match the newer parquet base-cache format.

If cache is missing under the new layout, the system will fall back to Databento API. The correct long-term fix is to standardize all futures symbols into the same cache layout.

---

## 10. Idea Quality Shift

Historically many ideas were too generic, for example:

- simple breakout / retest
- generic prior bar break
- pattern-only logic without context

This produced weak backtests and noisy research output.

The newer idea files are intentionally more contextual and institutional-style.

### Themes now emphasized

- previous day high / low sweep
- opening range raid + reclaim / reject
- HTF bias alignment
- displacement after compression
- retrace into displacement range
- structure shift
- reclaim / rejection triggers
- premium / discount context
- ICT-native framing where possible

---

## 11. Execution Status

The execution stack is already connected and should be treated as existing infrastructure.

Confirmed by user:

- forward testing already exists
- connection to Apex via Tradovate already exists
- PickMyTrade bridge is already in place

### Implication
Do **not** spend time redesigning execution unless requested. The current system’s highest-value work is:

- better research candidates
- cleaner strategy logic
- stronger filtering
- better handoff into the existing forward-testing path

---

## 12. Recommended Current Run Order

### Preferred futures-first run order

```bash
python src/agents/rbi_parallel_v2.py --ideas-file src/data/ideas_indices.txt
python src/agents/rbi_parallel_v2.py --ideas-file src/data/ideas_oil.txt
python src/agents/rbi_parallel_v2.py --ideas-file src/data/ideas_gold.txt
python src/agents/rbi_parallel_v2.py --ideas-file src/data/ideas_crypto.txt
python src/agents/rbi_batch_report.py
python src/agents/rbi_alpha_factory.py