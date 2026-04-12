# ANCS-Core — Implementation Status & Architecture

## Overview

ANCS-Core is Morphon's native in-process memory substrate. It replaces the old
MORPHON→HTTP→Node.js→PostgreSQL architecture with a single Rust binary module.
All memory access is a pointer dereference; no serialization, no IPC.

**Biological mapping:**
- `InMemoryBackend` ↔ Hippocampus (episodic binding, consolidation, replay)
- AXION importance ↔ SM-2 forgetting curves (6-factor composite)
- TruthKeeper ↔ Epistemic immune system (cascading invalidation)
- `MemoryTier` ↔ Neocortical consolidation gradient (vivid → schematic)
- SOMNUS (deferred) ↔ Circadian consolidation (wake/sleep replay)

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | `MemoryBackend` trait + `MemoryItem` + `InMemoryBackend` + `SystemHeartbeat` | ✅ Done |
| 1 | VBC-lite tier classification + AXION 6-factor importance + F7 pressure modes | ✅ Done |
| 2 | TruthKeeper epistemic transitions + blast-radius cascade + reconsolidation dispatch | ✅ Done |
| 3 | Epistemic-filtered RRF retrieval (`retrieve_memory()` in `system.rs`) | ✅ Done |
| 4 | Forward-importance synapse pruning (`forward_importance` on `Synapse`) | ✅ Done (v4.7.0) |
| 5 | SOMNUS sleep/wake consolidation cycle | ⏸ Deferred — see below |

---

## Phase 4: Forward-Importance Synapse Pruning

**Files:** `src/morphon.rs`, `src/learning.rs`

### What it does

The pruning heuristic previously protected synapses via two backward signals:
`reward_correlation` (was this synapse rewarded?) and `usage_count` (how often was it used?).
Both are backward-looking — they can prune a synapse that rarely fires but is critical for
downstream reward.

`forward_importance` is a forward signal: it tracks how much reward-correlated credit flows
*through* a synapse to its post-synaptic targets. A synapse with high `forward_importance`
is on a reward-carrying path and must survive even if its weight is currently weak.

### Implementation

**`Synapse.forward_importance: f64`** (added to `src/morphon.rs`)
- EMA α=0.05 (faster than `reward_correlation`'s 0.01 — responds to recent reward flow)
- Updated in `apply_weight_update()` in `src/learning.rs`:
  ```rust
  synapse.forward_importance =
      synapse.forward_importance * 0.95 + synapse.eligibility.abs() * r.abs() * 0.05;
  ```

**`LearningParams.forward_importance_min: f64`** (default `0.01`)
- Added guard to `should_prune_with_cost()`:
  ```rust
  && synapse.forward_importance < params.forward_importance_min
  ```

### Why α=0.05 not α=0.01

`reward_correlation` uses α=0.01 (window ~100 medium ticks) to track lifetime
reward history. `forward_importance` uses α=0.05 (window ~20 medium ticks) to
track *recent* reward flow — a synapse that stopped carrying reward recently should
lose protection, not carry it indefinitely from an early lucky correlation.

---

## Phase 5: SOMNUS — Why Deferred

### The three reasons

**1. Incompatible with continuous control.**
SOMNUS requires pausing external input and injecting replay patterns through the input
ports as fake sensor data. For CartPole (continuous control), you cannot pause mid-episode
— the environment keeps running regardless. For MNIST (batch), sleep between images is
natural. This task dependency means SOMNUS needs configuration that can be turned wrong,
and adds a correctness burden we don't yet need.

**2. Current replay is working.**
`dream_cycle()` already replays top-5 ANCS items by importance every `memory_period`
steps. It's directed (importance-weighted), it runs continuously, and it doesn't require
pausing the environment. There is no evidence the current approach is the bottleneck for
CartPole or MNIST. SOMNUS would replace it with something more complex without a proven
payoff yet.

**3. Depends on Phase 4 being validated.**
SOMNUS's value is in consolidating high-importance, low-confidence items during sleep.
`forward_importance` (Phase 4) is what identifies synapses worth consolidating. Running
SOMNUS without Phase 4 means the sleep-phase consolidation has no way to preferentially
protect critical-path synapses. Phase 4 must be stable first.

### Prerequisites before implementing Phase 5

1. **Phase 4 validated** — `forward_importance` stable, CartPole ≥195.0, no regressions
2. **`SchedulerConfig` extension** — add `somnus_period: u64` and `somnus_duration: u64`
3. **`sleep_phase` toggle wired** — `SystemHeartbeat.sleep_phase: AtomicBool` stub already
   exists in `src/ancs.rs`; needs to be read in the fast path:
   ```rust
   if self.heartbeat.sleep_phase.load(Ordering::Relaxed) { skip_external_input(); }
   ```
4. **`dream_cycle_deep()`** — sleep-phase variant that:
   - Streams Contested + low-consolidation ANCS items (not just top-5 by importance)
   - Runs TruthKeeper full sweep
   - Injects replay patterns through input ports as if they were real sensor data
   - Recalculates importance scores for all items

---

## Integration with `system.rs`

### ANCS item encoding (medium path, ~line 2066)

On each episodic encoding event:
1. `ids` extracted from the active pattern
2. `working_overlap` computed (Jaccard against current WM items)
3. `classify_tier()` routes to Verbatim/Structural/Semantic/Procedural
4. `MemoryItem` built from pattern, reward, novelty, step_count
5. `self.ancs.store(ancs_item)` — stored in `InMemoryBackend`
6. `self.current_ancs_item` set to last stored ID
7. `self.memory.working.store(ids, wm_activation)` — WM now populated (fixed v4.7.0)

### Synapse→ANCS linkage (Phase 2d)

During weight updates (`system.rs` lines ~1433, ~1503), when `|delta_w| > 0.001`:
```rust
j.memory_item_ref = self.current_ancs_item;
```
This links the synapse's justification record to the ANCS item active at reinforcement
time. TruthKeeper uses these refs to find synapses encoding a Contested memory.

### Dream cycle (`system.rs` ~line 3339)

`dream_cycle()` retrieves top-5 ANCS items and replays them through the learning path.
Called from the slow path every `memory_period` steps and explicitly from
`System::report_episode_end()`.

---

## What Was Broken Before v4.7.0

### Working memory never populated
`System::step()` computed `working_overlap` from WM contents for tier classification,
but never called `self.memory.working.store()`. WM was perpetually empty.
**Fixed:** `self.memory.working.store(ids, wm_activation)` added after `ancs.store()`.
`wm_activation` = mean absolute membrane potential of pattern morphons, clamped [0,1].

### `forward_importance` missing from `Synapse`
Pruning could not protect synapses based on downstream reward flow — only backward
correlation and usage count. **Fixed:** Phase 4 adds the field and EMA update.

---

## File Index

| File | Contents |
|------|----------|
| `src/ancs.rs` | `MemoryBackend` trait, `MemoryItem`, `InMemoryBackend`, VBC-lite, AXION importance, TruthKeeper, `SystemHeartbeat`, Phases 0–3 |
| `src/morphon.rs` | `Synapse.forward_importance` (Phase 4) |
| `src/learning.rs` | `apply_weight_update()` EMA update, `LearningParams.forward_importance_min`, `should_prune_with_cost()` guard |
| `src/system.rs` | ANCS encoding path (~line 2066), `dream_cycle()` (~line 3339), `memory_item_ref` linkage (~lines 1433, 1503), WM population fix |
| `src/memory.rs` | `TripleMemory`, `WorkingMemory`, `EpisodicMemory`, RRF retrieval (`retrieve_memory()`) |
| `src/justification.rs` | `SynapticJustification.memory_item_ref` |
| `docs/plans/morphon-ancs.md` | Original design document with full phase specs |
