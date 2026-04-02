# Plan: ANCS-Core — Native Rust Memory Substrate for Morphon

## The Idea

Rewrite ANCS as a native Rust module inside Morphon. No PostgreSQL, no HTTP, no separate process. The hypergraph lives in the same address space as the morphons. Memory access is a pointer dereference, not a database query. TruthKeeper is a trait constraint, not a service. AXION importance scoring runs in the same rayon thread pool as spike propagation.

```
Old world:  MORPHON (Rust) ──HTTP──→ ANCS (Node.js) ──SQL──→ PostgreSQL
New world:  MORPHON + ANCS-Core: one binary, shared memory, zero serialization
```

**Biological mapping:**
- MORPHON = Neocortex (processing, plasticity, structural growth)
- ANCS-Core Hypergraph = Hippocampus (episodic binding, consolidation, replay)
- AXION importance = Forgetting curves (SM-2 inspired, 6-factor)
- TruthKeeper = Immune system (detects and cascades knowledge invalidation)
- SOMNUS = Circadian consolidation (wake: process, sleep: replay + prune)

---

## What Already Exists (and maps to ANCS concepts)

Before building anything new, here's what Morphon already has that the ANCS notes propose:

| ANCS Proposal | Already Exists As | Gap |
|---|---|---|
| Epistemic states (6-state) | `epistemic.rs`: 4 states on clusters (Supported/Outdated/Contested/Hypothesis) | Missing: Stale, Suspect. States live on clusters, not on individual memory items |
| Justification records | `justification.rs`: `SynapticJustification` with FormationCause on every synapse | Missing: ULID refs to external store, bi-temporal tracking |
| Proprioceptive OS (PROTOS) | `endoquilibrium.rs`: senses VitalSigns (firing rates, eligibility density, weight entropy), feeds back via channel gains | Missing: feeding vitals as actual morphon input (currently modulates gains, not input) |
| AXION dendrite gating | `learning.rs`: receptor-gated modulation — morphons already filter by ReceptorSet (Motor: Reward+Arousal, Sensory: Novelty+Arousal, etc.) | Missing: learnable per-morphon mask (current receptors are fixed by cell type) |
| Catastrophic forgetting prevention | `homeostasis.rs`: checkpoint/rollback (reverts synapses when PE spikes); consolidation_level scaling (consolidated synapses get 10% of updates) | Missing: TruthKeeper-driven reconsolidation (current rollback is PE-triggered, not knowledge-validity-triggered) |
| Circadian scheduler (SOMNUS) | `scheduler.rs`: 4 timescales (fast/medium/slow/glacial); episodic replay already runs on memory_period=100 | Missing: explicit wake/sleep phase toggle, replay-driven consolidation during "sleep" |
| F7 pressure / energy budget | `morphon.rs`: MetabolicConfig with base_cost, firing_cost, utility_reward; apoptosis at energy < 0.1 | Missing: system-wide pressure modes (Normal/Pressure/Emergency/Critical), importance-driven demotion |
| Inter-cluster inhibition | `morphogenesis.rs`: creates inhibitory Modulatory morphons between correlated clusters | Fully implemented |
| Hyperbolic geometry | `types.rs`: Poincaré ball position, distance(), exp_map(), log_map() | Fully implemented |
| Serialization | `snapshot.rs`: full system state to/from JSON | Missing: bi-temporal snapshots, incremental persistence |

**Bottom line:** ~60% of what the notes propose as "new" already exists in some form. The real gaps are in the **memory system** (no backend trait, no importance scoring, no fused retrieval, no tier classification) and in the **knowledge-validity feedback loop** (TruthKeeper → synapse reconsolidation).

---

## What's Genuinely New

### 1. Memory Backend Trait + Unified Memory Item
The current `TripleMemory` has no trait, no retrieval interface, and three separate structs. This is the single biggest blocker.

### 2. AXION Importance Scoring
Current eviction: working memory evicts lowest-activation; episodic evicts least-consolidated. No composite importance score, no forgetting curves, no forward-reference density.

### 3. Tier Classification (VBC-lite)
Currently all episodic encoding uses the same `novelty > 0.3` gate. No distinction between "protect this verbatim" vs "compress this" vs "this is just a topology snapshot."

### 4. TruthKeeper → Synapse Reconsolidation Loop
Epistemic states exist on clusters but don't feed back into weight updates. When a cluster goes Contested, nothing happens to its synapses. The V6 contradiction-driven reconsolidation concept addresses exactly this.

### 5. RRF Fused Retrieval
Three memory stores queried independently. No rank fusion. Episodic replay selects by `reward + novelty - consolidation`, working memory refreshes by overlap. No cross-store signal.

### 6. Lock-Free Hypergraph for Knowledge Entities
The topology (petgraph) tracks morphon connectivity. There's no separate knowledge graph tracking entities, relations, or typed hyperedges. The justification records on synapses are the closest thing, but they're per-synapse, not a queryable graph.

### 7. Persistence Beyond JSON Snapshots
`snapshot.rs` does full-state JSON dumps. No incremental persistence, no mmap, no bi-temporal history.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MORPHON + ANCS-Core                         │
│                                                                  │
│  L4: Governance                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ TruthKeeper  │  │ Constitution │  │ Endoquilibrium       │   │
│  │ (validates   │  │ (hard limits)│  │ (predictive homeo)   │   │
│  │  knowledge)  │  │              │  │                      │   │
│  └──────┬───────┘  └──────────────┘  └──────────────────────┘   │
│         │ reconsolidation signal                                 │
│  L3: Processing                                                  │
│  ┌──────┴───────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Learning     │  │ Resonance│  │ Morpho-  │  │ Scheduler│   │
│  │ (3-factor    │  │ (spikes) │  │ genesis  │  │ (4-scale │   │
│  │  + capture)  │  │          │  │          │  │  + sleep) │   │
│  └──────┬───────┘  └──────────┘  └──────────┘  └──────────┘   │
│         │ justification write                                    │
│  L2: Memory Substrate (ANCS-Core)                                │
│  ┌──────┴──────────────────────────────────────────────────┐    │
│  │  trait MemoryBackend                                     │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐  │    │
│  │  │ T0      │  │ T1      │  │ T2      │  │ T3        │  │    │
│  │  │Verbatim │  │Structural│ │Semantic │  │Procedural │  │    │
│  │  │(immut.) │  │(working) │  │(episodic)│ │(topology) │  │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └───────────┘  │    │
│  │                                                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │    │
│  │  │ AXION        │  │ RRF Fused    │  │ Hypergraph    │  │    │
│  │  │ Importance   │  │ Retrieval    │  │ (knowledge    │  │    │
│  │  │ (6-factor)   │  │              │  │  entities)    │  │    │
│  │  └──────────────┘  └──────────────┘  └───────────────┘  │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  L1: Shared State                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  SystemHeartbeat (atomic reads, zero-cost cross-module)   │   │
│  │  global_arousal | global_novelty | energy_pressure | ...  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  L0: Data Structures                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ petgraph     │  │ DashMap      │  │ rkyv (optional,      │   │
│  │ (topology)   │  │ (hypergraph) │  │  persistence)        │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phases

### Phase 0: Backend Trait + SystemHeartbeat

**The foundation. Everything depends on this.**

#### 0a: `trait MemoryBackend`

```rust
pub trait MemoryBackend: Send + Sync {
    fn store(&mut self, item: MemoryItem, tier: MemoryTier);
    fn retrieve(&self, query: &RetrievalQuery, top_k: usize) -> Vec<(MemoryItem, f64)>;
    fn step(&mut self, dt: f64);  // decay, forgetting, reconsolidation
    fn record_access(&mut self, id: MemoryItemId);
    fn importance(&self, id: MemoryItemId) -> f64;
    fn epistemic_state(&self, id: MemoryItemId) -> EpistemicState;
    fn items_with_state(&self, state: EpistemicState) -> Vec<MemoryItemId>;
    fn mark_stale(&mut self, id: MemoryItemId);  // trigger cascade
    fn pressure_mode(&self) -> PressureMode;
}
```

#### 0b: Core enums

```rust
pub enum MemoryTier {
    Verbatim,    // T0 — immutable, hash-verified
    Structural,  // T1 — lossless active patterns (working memory)
    Semantic,    // T2 — relational, lossy (episodic)
    Procedural,  // T3 — compressed topology snapshots
}

// Extends existing epistemic.rs EpistemicState (currently 4 states on clusters)
// to 6 states on memory items
pub enum MemoryEpistemicState {
    Hypothesis,  // Just encoded, unverified
    Supported,   // Confirmed by reward correlation or replay
    Stale,       // Source pattern changed significantly (NEW)
    Suspect,     // Transitive dependent of stale item (NEW)
    Contested,   // Contradicted by newer evidence
    Outdated,    // Superseded, candidate for eviction
}

pub enum PressureMode {
    Normal,     // energy_usage < 0.70 — full 6-factor scoring
    Pressure,   // 0.70-0.85 — fast-path: only retrievability + pinned
    Emergency,  // 0.85-0.95 — only pinned survives
    Critical,   // >= 0.95 — safe mode
}
```

#### 0c: Unified `MemoryItem`

```rust
pub struct MemoryItem {
    pub id: MemoryItemId,          // ULID (compatible with existing MorphonId scheme)
    pub tier: MemoryTier,
    pub pattern: Vec<(MorphonId, f64)>,  // morphon activations at encoding
    pub reward: f64,
    pub novelty: f64,
    pub timestamp: u64,            // step_count at encoding
    pub epistemic: MemoryEpistemicState,
    pub consolidation: f64,        // 0.0 → 1.0
    pub importance: f64,           // AXION composite score
    pub stability: f64,            // SM-2 stability (increases with replay)
    pub replay_count: u32,
    pub access_count: u32,
    pub content_hash: Option<u64>, // T0: FxHash for verbatim verification
    pub justification_refs: Vec<JustificationId>,  // links to synapse justifications
}
```

#### 0d: Wrap existing `TripleMemory` as `InMemoryBackend`

Zero behavior change. The existing WorkingMemory, EpisodicMemory, ProceduralMemory continue to work exactly as they do. The trait just wraps them with a uniform interface. All 116 tests pass.

#### 0e: SystemHeartbeat

```rust
#[repr(C)]
pub struct SystemHeartbeat {
    // Neuromodulation snapshot (already computed in system.rs each step)
    pub global_arousal: AtomicU32,     // f32 bits via to_bits/from_bits
    pub global_novelty: AtomicU32,
    pub global_reward: AtomicU32,
    pub global_homeostasis: AtomicU32,
    pub plasticity_mult: AtomicU32,

    // ANCS-Core state
    pub energy_pressure: AtomicU32,    // F7 pressure level
    pub contradiction_count: AtomicU32, // TruthKeeper alarm
    pub stale_count: AtomicU32,        // items needing reconsolidation

    // SOMNUS
    pub sleep_phase: AtomicBool,       // consolidation mode active
}
```

This replaces scattered field reads with a single cache-line-sized struct. Every morphon reads it once per tick. Not technically necessary for correctness but eliminates the "pass 6 different references" pattern in `step()`.

**Files:** `src/memory.rs`, `src/system.rs`
**New deps:** None (AtomicU32 is std)
**Risk:** Medium — trait extraction touches step loop. But behavior is identical.
**Gate:** All 116 tests pass, CartPole avg >= 195.0

---

### Phase 1: VBC-lite Tier Classification + AXION Importance

**The core new behavior.**

#### 1a: VBC-lite — tier routing from neuromodulatory state

No external classifier. The system's own signals determine tier:

```rust
fn classify_tier(novelty: f64, reward: f64, working_overlap: f64) -> MemoryTier {
    if novelty > 0.7 && reward.abs() > 0.5 {
        MemoryTier::Verbatim    // T0: high-significance moment, protect verbatim
    } else if working_overlap > 0.5 {
        MemoryTier::Structural  // T1: reinforcing active pattern
    } else if novelty > 0.3 {
        MemoryTier::Semantic    // T2: standard episodic capture (current behavior)
    } else {
        MemoryTier::Procedural  // T3: background topology snapshot
    }
}
```

Content hashing for T0: FxHash of pattern vector, verified on retrieval.

#### 1b: AXION 6-factor importance

```rust
fn compute_importance(item: &MemoryItem, system: &System) -> f64 {
    let f1_retrievability = {
        let days = (system.step_count - item.timestamp) as f64;
        (1.0 + (19.0/81.0) * days / item.stability).powf(-0.5)
    };
    let f2_frequency = 1.0 - (-0.3 * item.access_count as f64).exp();
    let f3_centrality = system.topology.pagerank_score(item.pattern[0].0);  // approximate
    let f4_surprise = item.novelty;  // novelty at encoding time
    let f5_pinned = if item.consolidation > 0.8 { 1.0 } else { item.importance };
    let f6_relevance = task_relevance(item, &system.modulation);

    0.25*f1 + 0.15*f2 + 0.15*f3 + 0.15*f4 + 0.15*f5 + 0.15*f6
}
```

Stability increases with each successful replay (SM-2 model):
```rust
item.stability *= 1.0 + 0.1 * (item.replay_count as f64).ln().max(1.0);
```

#### 1c: F7 Pressure modes

Compute `energy_pressure` from total system energy:
```rust
let total_energy: f64 = system.morphons.values().map(|m| m.energy).sum();
let max_energy = system.morphons.len() as f64;  // each morphon max 1.0
let usage = 1.0 - (total_energy / max_energy);

let mode = match usage {
    u if u < 0.70 => PressureMode::Normal,
    u if u < 0.85 => PressureMode::Pressure,
    u if u < 0.95 => PressureMode::Emergency,
    _ => PressureMode::Critical,
};
```

Under Pressure: evict by importance, only `f1 + f5`.
Under Emergency: only pinned/consolidated items survive.
Under Critical: governor takes over (existing Constitutional constraints).

#### 1d: Tier demotion

Items whose importance drops below threshold get demoted:
- T0 → T2 (importance < 0.5, allow lossy representation)
- T2 → T3 (importance < 0.3, only topology trace)
- T3 → eviction (importance < 0.1)

**Files:** `src/memory.rs`
**Depends on:** Phase 0
**Risk:** Low — new behavior, old paths unchanged

---

### Phase 2: TruthKeeper Reconsolidation Loop

**The knowledge-validity feedback loop. This is where ANCS and Morphon genuinely converge.**

#### 2a: Epistemic state transitions on memory items

Extend the existing 4-state cluster model to 6 states on individual memory items:

- `Hypothesis → Supported`: replay produces positive reward correlation (consolidation > 0.5)
- `Supported → Stale`: source input pattern cosine distance > 0.5 from encoding
- `Stale → Suspect`: cascade to items in temporal window (± 50 steps of stale item)
- `Any → Contested`: newer item with opposing reward sign for same output morphons
- `Contested → Outdated`: N steps without re-support (default: 1000)

#### 2b: Blast radius cascade

When an item goes Stale:
1. Walk temporal neighborhood (items encoded within ± 50 steps)
2. Mark as Suspect
3. Count blast radius
4. If blast_radius > threshold (default 10), trigger reconsolidation burst

#### 2c: Reconsolidation — the missing link

Currently, `homeostasis.rs` has checkpoint/rollback for PE spikes. But there's no mechanism to re-open consolidated synapses when the *knowledge they represent* is invalidated. This is the gap.

```rust
// In the medium path (every 10 steps):
fn reconsolidate(memory: &MemoryBackend, morphons: &mut HashMap<MorphonId, Morphon>, topology: &mut Topology) {
    for item_id in memory.items_with_state(MemoryEpistemicState::Contested) {
        let item = memory.get(item_id);
        // Find synapses that were consolidated during this item's timestamp window
        for (pre, post, synapse) in topology.synapses_in_window(item.timestamp, 50) {
            if synapse.consolidated {
                synapse.consolidated = false;
                synapse.consolidation_level *= 0.7;  // partial reset
                synapse.tag = synapse.eligibility;    // re-open for learning
                synapse.tag_strength *= 0.5;          // weaken old tag
            }
        }
    }
}
```

This merges:
- V6 Contradiction-Driven Reconsolidation (from Ingestible notes)
- TruthKeeper cascade invalidation (from ANCS)
- Existing checkpoint/rollback mechanism (which handles PE spikes but not knowledge invalidation)

#### 2d: Semantic tagging — synapse → justification → memory item

Each synapse's `SynapticJustification` (already exists) gets a `memory_item_ref: Option<MemoryItemId>` linking it to the memory item that was active when the synapse was strengthened. When TruthKeeper marks that memory item Contested, the linked synapses are candidates for reconsolidation.

**Files:** `src/memory.rs`, `src/learning.rs`, `src/justification.rs`, `src/system.rs`
**Depends on:** Phase 1
**Risk:** Medium — modifies the learning path. Must validate CartPole doesn't regress.
**Expected improvement:** CartPole early-episode recovery. Bad policies from episodes 1-50 currently persist in consolidated weights. Reconsolidation explicitly un-freezes them.

---

### Phase 3: RRF Fused Retrieval (Ingestible)

#### 3a: Per-tier `rank()` methods

- Working (T1): rank by activation (recency)
- Episodic (T2): rank by cosine similarity of morphon activation vectors
- Procedural (T3): rank by Jaccard of active morphon sets

#### 3b: Reciprocal Rank Fusion

```rust
pub fn reciprocal_rank_fusion(
    rankings: &[Vec<(MemoryItemId, f64)>],
    k: usize,  // default 60
    top_k: usize,
) -> Vec<(MemoryItemId, f64)> {
    let mut scores: FxHashMap<MemoryItemId, f64> = FxHashMap::default();
    for ranking in rankings {
        for (rank, (id, _)) in ranking.iter().enumerate() {
            *scores.entry(*id).or_default() += 1.0 / (k + rank + 1) as f64;
        }
    }
    let mut results: Vec<_> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.truncate(top_k);
    results
}
```

#### 3c: Epistemic filter

Only `Supported` and `Hypothesis` items participate in standard retrieval. `Stale`/`Suspect`/`Contested` are excluded unless explicitly queried for reconsolidation replay.

#### 3d: Wire into episodic replay

Currently replay selects by `reward + novelty - consolidation`. Replace with RRF-fused ranking that considers recency (working), similarity (episodic), and structural overlap (procedural) together.

**Files:** `src/memory.rs`, `src/system.rs`
**Depends on:** Phase 0
**Risk:** Low — additive, doesn't modify existing paths

---

### Phase 4: Forward-Importance Synapse Pruning (Ingestible)

#### 4a: Per-synapse forward importance

Add `forward_importance: f64` to `Synapse` struct (currently 44 fields, this makes 45):

```rust
// In learning.rs, after weight update:
synapse.forward_importance = 0.95 * synapse.forward_importance
                           + 0.05 * (synapse.eligibility * reward_delta).abs();
```

#### 4b: Updated pruning heuristic

Current: `age > 100 AND weight.abs() < 0.001 AND usage_count < 5 AND NOT consolidated`

New:
```rust
fn should_prune(synapse: &Synapse) -> bool {
    if synapse.consolidated { return false; }
    if synapse.age < 100 { return false; }

    let backward = synapse.usage_count as f64 / synapse.age as f64;
    let forward = synapse.forward_importance;
    let combined = 0.5 * backward + 0.5 * forward;

    combined < 0.01 && synapse.weight.abs() < 0.01
}
```

Synapses with high `forward_importance` survive even if rarely used — they enable downstream reward.

**Files:** `src/morphon.rs`, `src/learning.rs`, `src/morphogenesis.rs`
**Depends on:** Phase 0
**Risk:** Low — one new field, modified threshold

---

### Phase 5: SOMNUS — Sleep/Wake Consolidation Cycle

**Formalizes what already happens implicitly during episodic replay.**

#### 5a: Sleep phase toggle

Add to scheduler: when `step_count % somnus_period == 0`, enter sleep phase for `somnus_duration` steps.

During sleep:
1. External input decoupled (sensory morphons receive replay, not environment)
2. ANCS-Core streams high-error items (items with low consolidation, high importance)
3. TruthKeeper runs full sweep (mark Contested items, trigger reconsolidation)
4. Importance scores recalculated for all items

During wake (default):
- Normal operation, ANCS-Core logs only
- Fast-path importance (no full recalculation)

#### 5b: Dream replay

During sleep, instead of random episodic replay, select items by:
1. High importance + low consolidation (need strengthening)
2. Contested state (need reconsolidation)
3. High forward_importance synapses with low usage (need reinforcement)

Inject replay patterns through input ports as if they were real sensor data. The learning system doesn't know the difference.

**Files:** `src/scheduler.rs`, `src/system.rs`, `src/memory.rs`
**Depends on:** Phase 2, Phase 3
**Risk:** Medium — changes step loop behavior during sleep phases
**Expected improvement:** Faster consolidation, cleaner forgetting

---

### Knowledge Hypergraph (separate spec)

Extracted to [`docs/knowledge-hypergraph-spec.md`](../knowledge-hypergraph-spec.md). Covers the persistent, queryable knowledge entity graph (DashMap + optional redb), bi-temporal queries, entity resolution, and the foundation for multi-instance sync. Build after Phases 0-4 are validated.

---

## Execution Order

```
Phase 0 ─── Backend Trait + Heartbeat ────────────────────── GATE
  │
  ├── Phase 1 ─── VBC-lite + AXION Importance + F7 Pressure
  │     │
  │     └── Phase 2 ─── TruthKeeper Reconsolidation Loop
  │           │
  │           └── Phase 5 ─── SOMNUS Sleep/Wake Cycle (deferred)
  │
  ├── Phase 3 ─── RRF Fused Retrieval (independent of 1-2)
  │
  └── Phase 4 ─── Forward-Importance Pruning (independent of 1-3)

After validation: Knowledge Hypergraph (separate spec)
```

**Phases 3 and 4 can run in parallel with the 1→2 chain.** Phase 5 and the hypergraph are deferred until Phases 0-4 prove value.

---

## What We're NOT Building

| Proposed | Why Not |
|---|---|
| PostgreSQL backend | The whole point is in-process. If persistence is needed, redb (Phase 6). |
| REST API / MCP tools | Morphon exposes via PyO3/WASM. ANCS-Core is internal. |
| Full 7-head VBC classifier | VBC-lite uses neuromodulatory signals. No ML model to train. |
| CRDT multi-instance sync | Single-instance first. Multi-instance is a separate project. |
| AXION wire protocol | No inter-process communication. Everything shares memory. |
| Custom allocator for F7 | Rust's allocator is fine. F7 is a logical eviction policy, not a malloc hook. |
| SIMD for vector comparison | Premature. Profile first. The bottleneck is topology traversal, not vector ops. |
| Lock-free everything | DashMap for the optional hypergraph (Phase 6). The core memory backend uses `&mut self` — no contention, no locks needed. System::step() is the sole writer. |
| `AtomicF32` | Doesn't exist in stable Rust. Use `AtomicU32` with `f32::to_bits/from_bits`. |
| mmap for zero-copy | Only relevant with persistence (Phase 6). redb already does this internally. |
| PROTOS as separate system | Endoquilibrium already does this. Extend it, don't duplicate it. |
| AXION dendrite gating | Receptor-gated learning already filters by cell type. A learnable `axion_mask` per morphon is interesting but adds a field to the hot path. Defer. |

---

## Validation

**Each phase must pass:**
1. All 116 existing tests (no regression)
2. `cargo test` clean
3. CartPole: avg >= 195.0 (current: 195.2)
4. MNIST: no regression

**Expected improvements by phase:**
- **Phase 1** (importance): smarter eviction → fewer valuable memories lost → steadier learning
- **Phase 2** (reconsolidation): early-episode bad policies don't persist → faster CartPole convergence, possibly fewer episodes to solve
- **Phase 3** (RRF): cross-store retrieval → MNIST benefits from episodic + structural overlap signals
- **Phase 4** (forward pruning): critical-path synapses survive → less structural damage from pruning
- **Phase 5** (SOMNUS): directed replay → consolidation of high-value, low-confidence items

---

## New Dependencies

**Phases 0-4 require zero new dependencies.** ULIDs for memory item IDs can be generated from the existing `rand` crate. FxHash is available via std.

The knowledge hypergraph (separate spec) adds `dashmap` and optionally `redb` — but those are feature-gated and don't affect the default build.

---

## Honest Assessment

### What's strong about this concept

**The reconsolidation loop (Phase 2) is the single most valuable idea here.** No neuromorphic system I'm aware of has a mechanism to re-open consolidated weights when the knowledge they encode is invalidated. Morphon already has checkpoint/rollback for PE spikes, but that's reactive — it catches destabilization after it happens. TruthKeeper-driven reconsolidation is *proactive* — it identifies which specific consolidated weights are backed by contested evidence and surgically reopens them. That's a genuine contribution.

**The importance scoring (Phase 1) solves a real problem.** The current pruning heuristic (`age > 100, weight < 0.001, usage < 5`) is dead simple. It works for CartPole (small network, short runs), but it will break at scale because it can't distinguish "rarely fires but critical for output" from "genuinely useless." Forward-reference density from Ingestible directly addresses this. The data already exists in the eligibility traces — it's just not being used for pruning decisions.

**The tier classification is biologically elegant.** Using the system's own neuromodulatory state to classify memory items into tiers — rather than training a separate VBC classifier — is exactly the kind of self-referential design that makes Morphon coherent. The system classifies its own experiences the same way it learns: through its internal signals.

### What I'd be cautious about

**The SystemHeartbeat is nice-to-have, not need-to-have.** The current code passes neuromodulation state through function arguments and that works fine. An atomic shared struct saves some parameter passing but introduces shared mutable state. It's a premature optimization for a system that runs single-threaded on the hot path (rayon parallelism is within each step, not across steps). I'd implement it only if profiling shows parameter-passing overhead matters. If you want it for cleanliness, fine, but don't let it gate Phase 1.

**SOMNUS (Phase 5) is the idea I'm least certain about.** The current episodic replay (3 items every 100 steps) is simple and works. Introducing a formal sleep phase changes the system's behavior fundamentally — during sleep, the system can't respond to its environment. For CartPole (continuous control), you can't sleep. For MNIST (batch classification), sleep between batches makes sense. This means SOMNUS is task-dependent, which means it needs careful configuration, which means it's a knob that can be turned wrong. I'd defer it until Phases 1-4 are validated and there's evidence that the current replay strategy is limiting.

**The knowledge hypergraph (Phase 6) is a different project.** It's valuable for the multi-instance, persistent, queryable vision — but Morphon's topology *is* already a graph. Adding a second graph structure (DashMap-based hypergraph) alongside petgraph creates a question: which is the source of truth? The topology or the knowledge graph? For single-instance benchmarks, the answer is clearly the topology. Phase 6 makes sense when you're building the persistent, multi-robot, auditable version. Not before.

**The notes have some ideas that sound exciting but are either already implemented or don't survive contact with the codebase:**
- "PROTOS" — Endoquilibrium already senses VitalSigns and adjusts channel gains. Adding it as a separate system would duplicate logic.
- "Predictive prefetching into L1 cache" — you can't control L1 cache placement from Rust userspace. What you *can* do is structure data for locality (which Morphon already does with the morphon HashMap).
- "Counterfactual learning via bitemporal rollback" — interesting but it requires running a second Morphon instance on historical state, which is a full simulation clone. The existing checkpoint/rollback in homeostasis.rs is the lightweight version of this. Full counterfactual simulation is a research project, not a feature.
- "Lock-free everything" — `System::step()` has exclusive `&mut self`. There's no contention. Lock-free structures add complexity for zero benefit when you have a single writer.

### The risk I'd watch

**Complexity budget.** Morphon is already a 6000+ line system with 4 timescales, 4 neuromodulation channels, 7 morphogenesis mechanisms, checkpoint/rollback, epistemic states, justification records, endoquilibrium, frustration, fields, k-WTA, DFA, critic/actor, anchor/sail plasticity, and a dual-clock scheduler. Every new system (TruthKeeper reconsolidation, AXION importance, VBC-lite, RRF retrieval, SOMNUS) adds interaction surface area. The question isn't "can each piece work in isolation" — it's "do they compose without unexpected emergent behavior?"

The plan's answer to that is the validation gates: 116 tests + CartPole >= 195.0 after every phase. If Phase 2 (reconsolidation) causes a regression, we know immediately. But the harder question is: does the system become impossible to debug? Right now, when CartPole fails, you can trace through diagnostics.rs and find the problem. With 6 new interacting systems, the diagnosis path gets exponentially longer.

**My recommendation: Phases 0, 1, 2, 4 in that order. That's the core value.** Phase 3 (RRF) is low-risk and can go in parallel. Phase 5 (SOMNUS) and Phase 6 (hypergraph) are for after the core is validated.
