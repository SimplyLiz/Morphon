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

The pruning heuristic protects synapses via two backward signals: `reward_correlation`
(was this synapse rewarded?) and `usage_count` (how often was it used?). Both are
backward-looking — they could theoretically prune a synapse that is on a reward-carrying
path but hasn't yet accumulated signal.

Phase 4 adds a third protection channel: `tag_strength`. A synapse with `tag_strength ≥
forward_importance_min` (default 0.01) is in the "Hebbian coincidence awaiting capture"
state — pre and post fired together and the synapse is waiting for delayed reward to arrive.
Pruning it before capture happens would destroy a credit-assignment pathway.

### Implementation

**`Synapse.forward_importance: f64`** (added to `src/morphon.rs`)  
Maintained as an EMA of eligibility × modulation across all channels:
```rust
synapse.forward_importance =
    synapse.forward_importance * 0.995 + synapse.eligibility.abs() * m.abs() * 0.005;
```
This field is available for future use; the active pruning guard uses `tag_strength`.

**`LearningParams.forward_importance_min: f64`** (default `0.01`)  
Guard in `should_prune_with_cost()`:
```rust
// Phase 4: Protect tagged synapses awaiting capture
&& synapse.tag_strength < params.forward_importance_min
```

### Why tag_strength, not the EMA

The EMA-based `forward_importance` is structurally anti-correlated with the pruning
conditions. For a synapse to be prunable, it needs `weight < 0.001 && usage_count < 5`.
For `forward_importance` to be non-zero, the synapse must have `eligibility > 0` during
reward delivery — which also increments `usage_count` and drives `weight` up. The two
conditions cannot coexist in practice.

`tag_strength` is set when `eligibility > tag_threshold (0.3)` and is NOT continuously
decayed (it's a high-water mark). A synapse can have `tag_strength > 0.01` while being
otherwise weak — the tag was set on Hebbian coincidence, then the synapse went silent
before reward arrived.

### Diagnostic: `fwd_saved`

`src/morphogenesis.rs::pruning()` compares `should_prune_without_fwd()` vs
`should_prune_with_cost()` and increments `saved_by_fwd` for each synapse rescued
by the tag_strength guard. Propagated to `SystemStats.synapses_saved_fwd_recent`
and printed as `fwd_saved=N` in epoch logs when non-zero.

**Current observation:** `fwd_saved` does not appear in MNIST or CartPole runs.
This is expected — in healthy learning dynamics, a tagged synapse (eligibility > 0.3)
almost always also has `weight > weight_min` and `usage_count ≥ 5` because the same
activity that produces tagging also produces weight growth and usage increments.
The guard serves as an **alarm**: if `fwd_saved > 0` ever appears, it indicates
unusual learning dynamics (e.g., strong inhibitory reward driving a tagged synapse
below weight_min while the tag persists). That's worth investigating when it occurs.

---

## Phase 5: SOMNUS — Why Deferred

### Scope: what SOMNUS is and is not

SOMNUS is **retrospective** — it replays stored episodic memories through the learning
path to consolidate weights during an offline phase. The question it answers: *what
happened, and how do we strengthen those pathways?* Biologically this maps to slow-wave
sleep hippocampal replay and hippocampus → neocortex transfer.

It is **not** prospective simulation. Generating novel internal rollouts over the current
topology — imagining future states without environment interaction, analogous to the
default mode network or mental time travel — is a separate capability. Blumberg & Miller
(2025) call this "imaginative memory" in their cognitive architecture proposal and treat
it as a distinct memory area from episodic consolidation [1]. That distinction is correct
and applies here: if Morphon ever gains forward simulation (relevant first for the Drone3D
benchmark, where internal trajectory planning would reduce costly real rollouts), it would
sit alongside SOMNUS, not inside it.

[1] Blumberg, M. & Miller, M.S.P. (2025). Building Sentient Beings.
    DOI: 10.5281/zenodo.19234488

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

1. **Phase 4 deployed** — tag_strength guard and diagnostic counter in place (v4.7.0). No regressions vs CartPole ≥195.0, MNIST stateless ≥87.7%.
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

### Phase 4 guard missing from pruning
Pruning had no protection for synapses in the "tagged, awaiting capture" state. A
reward-path synapse that formed Hebbian coincidence but hadn't yet received the capture
signal could be pruned before consolidation. **Fixed:** Phase 4 adds `tag_strength`
guard + `forward_importance` EMA field (both in v4.7.0). See Phase 4 section for
the detailed analysis of why `fwd_saved` doesn't fire in current task profiles.

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
