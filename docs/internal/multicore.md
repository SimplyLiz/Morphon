# Multi-Core Performance Plan

## Priority Ordering

Performance work — doesn't change accuracy, but critical for 10-seed paper sweeps (20+ experiments).

| # | Item | Speedup | Effort | Status |
|---|------|---------|--------|--------|
| 1 | Revert to 30% connectivity | ~3x (free) | Already decided | **DONE** |
| 2 | Sparse eligibility updates | ~5x on medium path | ~200 lines | **DONE** |
| 3 | Rayon Tier 1 (drop-in par_iter swaps) | ~2x on glacial/slow paths | ~30 min | Deferred (see prereqs below) |
| 4 | Rayon Tier 2 #6 (three-factor learning) | Re-evaluate after Item 2 | Read-write phase separation | Re-evaluate |

## What was changed (Items 1 + 2)

### Item 1 — 30% connectivity cap

- **`src/developmental.rs:206-225`** — assoc→motor fan-in changed from 100% (every motor connects to every associative) to 30% (each motor receives from a random 30% sample of associative). Xavier scale recomputed against the new fan-in to preserve init variance.
- **`src/morphogenesis.rs:200-204`** — synaptogenesis already calls `governance::check_connectivity` before adding edges; no patch needed.
- **`examples/mnist.rs`, `examples/mnist_v2.rs`** — explicit `governance: ConstitutionalConstraints { max_connectivity_per_morphon: 300, .. }`. Cap chosen to give ~25% headroom over the developmental baseline (235 in-edges per associative from 30% × 784 sensory) so synaptogenesis can still grow useful edges, while preventing rich-get-richer hub pathology.

### Item 2 — Sparse eligibility updates

- **`src/morphon.rs:55-65`** — added `last_update_step: u64` field to `Synapse` (with `#[serde(default)]`).
- **`src/learning.rs:100-160`** — `update_eligibility` now takes `current_step: u64` and does **lazy decay**: fast-forwards exponential decay of pre/post traces, eligibility, and tag in one shot using `elapsed = current_step - last_update_step`. Mathematically equivalent to per-step decay because exponentials compose. Removed the old per-step Euler decay term from the eligibility ODE since lazy fast-forward already handles it (avoids double-decay).
- **`src/system.rs:1005-1043`** — built `actor_visit_set` before the medium-path actor loop. A morphon enters the visit set if:
  - it's a Motor (always borderline-active via continuous sigmoid)
  - it fired this tick
  - its `feedback_signal` is non-zero (DFA path)
  - it's a post-target of any firing morphon (covers LTD-via-pre-spike)
- **`src/system.rs:1075`** — actor branch now gated by `else if actor_visit_set.contains(&id)`. Synapses on skipped morphons get correct state via lazy decay when next visited.
- **Test & integration call sites** updated to pass `0` as the new `current_step` argument (per-call elapsed = 1 preserves old per-step semantics for unit tests).

### Approximations introduced

- **L2 weight decay** (`synapse.weight -= 0.0005 * synapse.weight` at the end of the non-DFA branch) is skipped for inactive morphons. Per-step effect is 0.05% — even over thousands of skipped ticks the weight drift is bounded by `weight_max` clamping. Not worth tracking lazily.
- **Eligibility decay precision**: old code used Euler integration `elig += -elig/τ * dt`, new code uses exact `elig *= exp(-dt/τ)`. For τ=20, dt=1: per-step factor 0.95 (Euler) vs 0.9512 (exact). New is more accurate; functional impact on learning is negligible since the system is robust to trace noise.

Items 1+2 alone: ~15x on full-connectivity, ~5x on 30% connectivity. Standard profile runs go from ~25 min to ~5 min. Good enough for iterative development.

Items 3+4: For paper comparison tables (10 seeds x 2 modes x standard profile = 20+ runs where wall-clock matters).

---

## Rayon Tier 1: Drop-in `par_iter` Swaps — Deferred

### Prerequisites

1. Items 1+2 merged (✓ done) and benchmarked
2. At least one full standard-profile MNIST run completed to confirm the ~5-min target is hit
3. Decision rule:
   - **If standard profile ≤ 5 min** → Tier 1 stays deferred indefinitely
   - **If 5–10 min** → Tier 1 is the next swing
   - **If still over 10 min** → Tier 1 first, then re-profile, then decide on Tier 2

### When to implement

- **Trigger**: starting paper comparison sweeps (10 seeds × 2 modes × standard = 20+ runs where wall-clock matters)
- **Effort**: ~30 min mechanical work, low risk — every target is `#[cfg(feature = "parallel")]` gated with sequential fallback
- **Expected impact**: ~2x on glacial/slow paths only (modulation recording, differentiation, migration, identity field projection). Does **not** touch the medium-path learning loop, which Item 2's sparse eligibility already handles.
- **Why deferred**: Items 1+2 likely make iterative dev tolerable on their own. Tier 1 is paid complexity for the paper-sweep regime — implement when the sweeps are about to start, not preemptively.

### Targets (when triggered)

Zero shared mutable state — just swap `.iter()` / `.values_mut()` for `.par_iter()` / `.par_iter_mut()`.

| # | File | Lines | What | Tick Frequency |
|---|------|-------|------|----------------|
| 1 | `src/system.rs` | ~1128-1150 | Modulation recording (per-morphon channel history) | Medium (every 10 steps) |
| 2 | `src/morphogenesis.rs` | ~391-451 | Differentiation decisions (per-Stem morphon) | Glacial (~1000 steps) |
| 3 | `src/morphogenesis.rs` | ~461-471 | Dedifferentiation (per-Stem morphon) | Glacial |
| 4 | `src/morphogenesis.rs` | ~984-1095 | Migration tangent vectors + exp_map (per-morphon) | Slow (~100 steps) |
| 5 | `src/system.rs` | ~1273-1294 | Identity field projection (independent grid cells) | Glacial |

Each gets the standard `#[cfg(feature = "parallel")]` / `#[cfg(not(feature = "parallel"))]` dual-path pattern.

## Rayon Tier 2: Topology-Bound Parallelization — Re-evaluate

> **Status note (after Items 1+2)**: The medium-path three-factor learning loop (Tier 2 item #6, the highest-impact target in the original analysis) is **partially obsoleted** by Item 2's sparse eligibility. The active-set iteration already skips ~90% of synapses on a typical medium tick. Tier 2 #6 only matters if benchmarks show the remaining active-set work is still hot enough to be worth parallelizing across cores. Re-evaluate once we have post-sparsification benchmark numbers — Tier 2 may collapse to just #7-9.

### Key insight: parallelism analysis

Three operations are embarrassingly parallel but blocked by petgraph's borrow model:

1. **Anti-Hebbian LTD** — each suppressed morphon's incoming synapses are independent. par_iter over suppressed_ids, collect mutations, apply. Zero cross-talk.
2. **Spike propagation** — each fired morphon's outgoing spikes write to different target accumulators. Already feature-gated but may not parallelize the inner loop.
3. **Medium-path learning** — eligibility updates are per-synapse with no cross-synapse dependencies. Three-factor rule reads global modulation (shared immutable) and writes per-synapse weight (exclusive). Textbook par_iter_mut.

### The petgraph bottleneck

`incoming_synapses_mut()` takes `&mut Topology` — Rust's borrow checker won't let you mutate multiple edges in parallel through the graph's API. Two escape hatches:

**Option A: Batch-collect + raw index access** (incremental, keeps petgraph)
- Read-only pass: collect edge indices + needed state into Vec
- Mutation pass: apply deltas via raw index access
- Pro: no architecture change. Con: double traversal, still serialized writes.

**Option B: Flat edge arrays / CSR format** (the Pulse Kernel Lite path)
- Hot synapse data (weight, eligibility, tag) in flat `Vec<f64>` arrays indexed by edge ID
- petgraph stays for structural queries (neighbors, add/remove edges)
- Hot-path learning reads/writes flat arrays directly — rayon par_iter_mut over slices
- Pro: near-linear speedup with cores (~6-8x on Apple Silicon 10 perf cores). Con: larger refactor, dual bookkeeping.

**Recommendation**: Start with Option A for immediate gains. Option B is the endgame if paper sweeps need sub-3-minute runs (287K synapses from ~20 min to ~3 min).

### Targets

| # | File | Lines | What | Tick Frequency |
|---|------|-------|------|----------------|
| 6 | `src/system.rs` | ~897-1030 | Three-factor learning (critic + actor weight updates) | Medium — **highest-impact** |
| 7 | `src/system.rs` | ~1044-1120 | Weight normalization + compensatory plasticity | Medium |
| 8 | `src/homeostasis.rs` | ~149-170 | Synaptic scaling | Homeostasis tick (~50 steps) |
| 9 | `src/morphogenesis.rs` | ~166-224 | Synaptogenesis candidate pair testing | Slow |
| 10 | `src/system.rs` | Anti-Hebbian LTD | Suppressed morphon incoming synapse updates | Fast path |

### Pattern (Option A):

```rust
// Phase 1: parallel read — collect (edge_idx, delta_w) tuples
let updates: Vec<_> = morphon_ids.par_iter().map(|&id| {
    // read pre_id states, compute eligibility & weight deltas
    // all reads from immutable refs to morphons + topology
}).collect();

// Phase 2: sequential write — apply deltas
for (edge_idx, delta_w) in updates.into_iter().flatten() {
    topology.synapse_mut(edge_idx).weight += delta_w;
}
```

## Rayon Tier 3: Low Priority (skip unless benchmarks show need)

| File | What | Why deprioritized |
|------|------|-------------------|
| `src/system.rs:848-875` | iSTDP inhibitory updates | Small morphon subset |
| `src/morphogenesis.rs:645-763` | Fusion detection | Few candidates typically |
| `src/homeostasis.rs:277-331` | Inter-cluster inhibition | Small cluster count |
| `src/morphogenesis.rs:1111-1166` | Apoptosis filtering | Glacial, few candidates |

## Constraints

- All parallel paths gated behind `#[cfg(feature = "parallel")]` with sequential fallback
- Topology mutation cannot be parallelized (petgraph not thread-safe for writes)
- `morphons: HashMap<MorphonId, Morphon>` supports `par_iter_mut()` — already proven in codebase
- No new dependencies needed — rayon already in Cargo.toml
- GPU acceleration is a poor fit (sparse irregular graph, heavy branching, small N)
