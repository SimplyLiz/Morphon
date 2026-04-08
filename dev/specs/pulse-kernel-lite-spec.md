# Pulse Kernel Lite
## Pragmatic Fast-Path Optimization for MORPHON
### Technical Specification v2.0 — TasteHub GmbH, April 2026

---

| | |
|---|---|
| **Replaces** | pulse-kernel-spec.md v1.0 (full SoA rewrite — deferred to Phase 2) |
| **Approach** | Hot-array extraction alongside existing AoS — not a rewrite |
| **Keeps** | petgraph topology, Morphon structs, all learning/lifecycle logic |
| **Changes** | 4 hot arrays extracted for cache-friendly inner loop |
| **Estimated Effort** | 6–8 hours |
| **When to Build** | After Endoquilibrium is validated on CartPole |

---

## 1. Why v2.0 Exists

The v1.0 Pulse Kernel spec proposed a full SoA rewrite with CSR/CSC synapse storage. Honest peer review identified five problems:

1. **It's a rewrite, not a refactor.** The current `system.rs` has 1500+ lines of interleaved fast/medium/slow logic. DFA feedback, k-WTA, H6 compensatory plasticity, and readout-coupled anchoring all cross the fast/medium boundary. The clean separation v1.0 assumed doesn't match the codebase.

2. **Performance isn't the bottleneck.** CartPole runs at ~0.01ms/tick with 300 morphons. The bottleneck is credit assignment and learning dynamics, not spike propagation.

3. **CSR fights structural plasticity.** A system that grows from 60K to 110K synapses during training can't afford O(S) rebuilds. petgraph gives O(1) edge insertion. CSR's faster iteration doesn't justify slower mutation for a system whose core feature is runtime rewiring.

4. **The LIF model was oversimplified.** The current `Morphon::step()` does ~15 computations per tick, many affecting same-tick firing decisions. The v1.0 spec moved only voltage integration to the fast path without specifying where everything else goes.

5. **TruthKeeper and Governor interfaces were speculative.** Designing around subsystems that don't exist yet risks wrong abstractions.

**v2.0 takes the pragmatic path:** extract the 4 hottest arrays for cache-friendly iteration, keep everything else unchanged. 80% of v1.0's cache benefit at 20% of the implementation cost.

---

## 2. What Changes

### 2.1 Four Hot Arrays

Extract four fields from the Morphon struct into parallel arrays that live on the `System` struct. These are the fields touched on every tick for every morphon in the inner loop:

```rust
/// Hot-path arrays for cache-friendly spike processing.
/// These MIRROR the corresponding fields in each Morphon struct.
/// The Morphon struct remains the source of truth for lifecycle/learning.
/// Sync happens once per medium-path tick.
pub struct HotArrays {
    /// Membrane potential. Written every tick by integrate().
    pub voltage: Vec<f32>,
    
    /// Adaptive firing threshold. Read every tick by threshold_check().
    /// Written by homeostatic regulation (medium path) and Endoquilibrium.
    pub threshold: Vec<f32>,
    
    /// Did this morphon fire this tick? Written by threshold_check().
    /// Read by spike delivery and eligibility computation.
    pub fired: BitVec,
    
    /// Previous tick's fires. For integration (which sources fired?).
    pub fired_prev: BitVec,
    
    /// Refractory countdown. Decremented every tick.
    pub refractory: Vec<u8>,
    
    /// Maps hot-array index → petgraph NodeIndex.
    /// Stable within a slow-path epoch. Rebuilt on structural changes.
    pub idx_to_node: Vec<petgraph::graph::NodeIndex>,
    
    /// Maps petgraph NodeIndex → hot-array index.
    /// Sparse — uses a HashMap since NodeIndex isn't contiguous after deletions.
    pub node_to_idx: HashMap<petgraph::graph::NodeIndex, usize>,
    
    /// How many active morphons are in the arrays.
    pub active_count: usize,
}
```

**Key design decision: the hot arrays are a cache-friendly VIEW, not the source of truth.** The Morphon structs in petgraph remain authoritative. The hot arrays are rebuilt from scratch on every slow-path structural change (birth, death, synaptogenesis). Within a slow-path epoch (hundreds of fast-path ticks), the hot arrays are updated in-place and synced back to structs on the medium path.

### 2.2 What Stays Exactly the Same

Everything else:

- **petgraph** for topology (nodes = morphons, edges = synapses)
- **Morphon struct** with all 25+ fields (lifecycle, learning, metabolic, position)
- **Synapse struct** on petgraph edges (weight, delay, eligibility, tag)
- **k-WTA** reading from hot arrays instead of struct fields
- **DFA feedback** injected into voltage via the hot arrays
- **H6 compensatory plasticity** reading fired from hot arrays
- **Developmental Engine** operating on petgraph (division, death, pruning, synaptogenesis)

---

## 3. The Sync Protocol

The hot arrays and the Morphon structs must stay consistent. The sync protocol is simple because the dual-clock architecture already separates fast and medium paths:

```
Fast Path (every tick):
  - READ: hot.voltage, hot.threshold, hot.refractory
  - WRITE: hot.voltage (integrate), hot.fired (threshold check),
           hot.refractory (decrement)
  - The Morphon structs are NOT touched

Medium Path (every N ticks):
  - SYNC DOWN: copy hot.voltage → morphon.potential (for learning rules that read it)
  - SYNC DOWN: copy hot.fired → morphon.fired_this_tick (for eligibility, STDP)
  - RUN: STDP, DFA weight updates, Endoquilibrium sensing, k-WTA
  - SYNC UP: copy morphon.threshold → hot.threshold (homeostatic regulation changed it)
  - SYNC UP: copy Endoquilibrium threshold_bias into hot arrays

Slow Path (every M medium ticks):
  - STRUCTURAL CHANGES: birth, death, synaptogenesis, pruning
  - REBUILD: hot arrays from scratch (reindex from petgraph)
  - Cost: O(N) where N = morphon count. At 300 morphons: ~microseconds.
          At 2000 morphons: ~0.1ms. Negligible on the slow path.
```

### 3.1 Sync Implementation

```rust
impl System {
    /// Called at the start of each medium-path tick.
    /// Copies fast-path results into Morphon structs for learning rules.
    fn sync_hot_to_structs(&mut self) {
        for (hot_idx, &node_idx) in self.hot.idx_to_node.iter().enumerate() {
            if let Some(morphon) = self.graph.node_weight_mut(node_idx) {
                morphon.potential = self.hot.voltage[hot_idx] as f64;
                morphon.fired_this_tick = self.hot.fired[hot_idx];
            }
        }
    }
    
    /// Called at the end of each medium-path tick.
    /// Copies updated thresholds back to hot arrays for fast-path use.
    fn sync_structs_to_hot(&mut self) {
        for (hot_idx, &node_idx) in self.hot.idx_to_node.iter().enumerate() {
            if let Some(morphon) = self.graph.node_weight(node_idx) {
                self.hot.threshold[hot_idx] = morphon.threshold as f32;
            }
        }
    }
    
    /// Called after any structural change (birth, death, edge add/remove).
    /// Rebuilds hot arrays from scratch.
    fn rebuild_hot_arrays(&mut self) {
        self.hot.active_count = 0;
        self.hot.node_to_idx.clear();
        
        for node_idx in self.graph.node_indices() {
            let morphon = &self.graph[node_idx];
            let hot_idx = self.hot.active_count;
            
            self.hot.voltage[hot_idx] = morphon.potential as f32;
            self.hot.threshold[hot_idx] = morphon.threshold as f32;
            self.hot.fired.set(hot_idx, false);
            self.hot.fired_prev.set(hot_idx, false);
            self.hot.refractory[hot_idx] = morphon.refractory_timer;
            self.hot.idx_to_node[hot_idx] = node_idx;
            self.hot.node_to_idx.insert(node_idx, hot_idx);
            
            self.hot.active_count += 1;
        }
    }
}
```

---

## 4. The Fast-Path Inner Loop

The optimized inner loop processes only the hot arrays. No Morphon struct access, no petgraph traversal, no HashMap lookups in the hot path.

### 4.1 Voltage Integration

```rust
impl System {
    /// Fast-path voltage integration.
    /// Iterates hot arrays only — no struct access.
    fn fast_integrate(&mut self, external_input: &[f32]) {
        let decay = (-1.0_f32 / self.config.tau_membrane as f32).exp();
        let gain = 1.0 - decay;
        
        for j in 0..self.hot.active_count {
            if self.hot.refractory[j] > 0 {
                continue;
            }
            
            // Sum incoming synaptic currents
            // Still uses petgraph for topology — but only accesses weight and source
            let node_idx = self.hot.idx_to_node[j];
            let mut i_syn: f32 = 0.0;
            
            for edge in self.graph.edges_directed(node_idx, petgraph::Incoming) {
                let source_node = edge.source();
                if let Some(&source_hot) = self.hot.node_to_idx.get(&source_node) {
                    if self.hot.fired_prev[source_hot] {
                        i_syn += edge.weight().weight as f32;
                    }
                }
            }
            
            // External input (for sensory morphons)
            if j < external_input.len() {
                i_syn += external_input[j];
            }
            
            // DFA feedback (precomputed on medium path, stored per morphon)
            // Read from struct since it's computed externally
            let feedback = self.graph[node_idx].feedback_signal as f32;
            i_syn += feedback;
            
            // LIF update
            self.hot.voltage[j] = self.hot.voltage[j] * decay + i_syn * gain;
        }
    }
}
```

**Note:** This still uses petgraph for edge iteration. The speedup comes from the hot arrays for voltage/fired/threshold (no Morphon struct pulled into cache), not from CSR. When petgraph iteration becomes the bottleneck (measurable via profiling at >2K morphons), that's when the full Pulse Kernel v1.0 with CSR becomes justified.

### 4.2 Threshold Check with Bit-Slicing

```rust
    /// Fast-path threshold comparison.
    /// Uses Endoquilibrium threshold_bias.
    fn fast_threshold_check(&mut self) {
        // Swap fired buffers
        std::mem::swap(&mut self.hot.fired, &mut self.hot.fired_prev);
        self.hot.fired.fill(false);
        
        let bias = self.endo_threshold_bias;
        
        for j in 0..self.hot.active_count {
            if self.hot.refractory[j] > 0 {
                continue;
            }
            if self.hot.voltage[j] >= self.hot.threshold[j] + bias {
                self.hot.fired.set(j, true);
            }
        }
    }
```

### 4.3 Reset

```rust
    /// Fast-path reset of fired morphons.
    fn fast_reset(&mut self) {
        for j in 0..self.hot.active_count {
            if self.hot.fired[j] {
                self.hot.voltage[j] = self.config.v_rest as f32;
                self.hot.refractory[j] = self.config.refractory_ticks;
            }
            if self.hot.refractory[j] > 0 {
                self.hot.refractory[j] -= 1;
            }
        }
    }
```

### 4.4 The Complete Fast-Path Step

```rust
    /// One fast-path step. Called N times per process() call
    /// (where N = config.internal_steps, typically 5).
    fn fast_step(&mut self, external_input: &[f32]) {
        self.fast_integrate(external_input);
        self.fast_threshold_check();
        self.fast_reset();
        self.tick_count += 1;
    }
```

No Phase 3 (emit) or Phase 5 (deliver-delayed) because we're still using petgraph for topology. The `fired_prev` bitfield handles one-tick propagation delay. For multi-tick delays, the existing Morphon-level delay mechanism continues to work through the struct sync.

---

## 5. Integration with Existing Subsystems

### 5.1 DFA Feedback

DFA computes `feedback_signal` per Associative morphon on the medium path. The fast path reads it from the Morphon struct (via `graph[node_idx].feedback_signal`). This is one struct field access per Associative morphon per fast-path step — acceptable because Associative morphons are ~60% of the population and the access is read-only.

**No change to DFA code required.** It writes to the Morphon struct. The fast path reads from the struct. The sync protocol doesn't need to copy feedback_signal because it's not in the hot arrays.

### 5.2 k-WTA

k-WTA currently reads `morphon.potential` and writes `morphon.fired`. With hot arrays:

```rust
// k-WTA reads from hot arrays instead of structs
fn k_wta(&mut self, cell_type: CellType, k: usize) {
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    
    for j in 0..self.hot.active_count {
        let node_idx = self.hot.idx_to_node[j];
        if self.graph[node_idx].cell_type == cell_type {
            candidates.push((j, self.hot.voltage[j]));
        }
    }
    
    // Sort by voltage, descending
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // Top-k fire, rest don't
    for (rank, &(j, _)) in candidates.iter().enumerate() {
        if rank < k {
            self.hot.fired.set(j, true);
        }
    }
}
```

### 5.3 H6 Compensatory Plasticity

H6 reads `fired` and `activity_history` to compute weight adjustments. With hot arrays, it reads `fired` from the hot arrays (synced to struct on medium path). No change to H6 logic — it operates on Morphon structs after sync_hot_to_structs().

### 5.4 Analog Readout

The analog readout reads `potential` from Motor morphons to compute continuous output. With hot arrays, it reads from `hot.voltage`:

```rust
fn read_analog_output(&self) -> Vec<f64> {
    let mut outputs = Vec::new();
    for j in 0..self.hot.active_count {
        let node_idx = self.hot.idx_to_node[j];
        if self.graph[node_idx].cell_type == CellType::Motor {
            outputs.push(self.hot.voltage[j] as f64);
        }
    }
    outputs
}
```

### 5.5 Endoquilibrium

Endoquilibrium reads firing rates per cell type. With hot arrays, this becomes a single pass over the `fired` bitfield cross-referenced with cell types:

```rust
fn compute_firing_rates(&self) -> [f32; 4] {
    let mut fire_counts = [0u32; 4];
    let mut total_counts = [0u32; 4];
    
    for j in 0..self.hot.active_count {
        let node_idx = self.hot.idx_to_node[j];
        let ct = self.graph[node_idx].cell_type as usize;
        total_counts[ct] += 1;
        if self.hot.fired[j] {
            fire_counts[ct] += 1;
        }
    }
    
    let mut rates = [0.0f32; 4];
    for i in 0..4 {
        if total_counts[i] > 0 {
            rates[i] = fire_counts[i] as f32 / total_counts[i] as f32;
        }
    }
    rates
}
```

---

## 6. The Revised process() Loop

```rust
impl System {
    pub fn process(&mut self, input: &[f64]) -> Vec<f64> {
        // Convert input to f32 for hot path
        let input_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();
        
        // === FAST PATH: N internal steps ===
        for step in 0..self.config.internal_steps {
            let ext = if step == 0 { &input_f32[..] } else { &[] };
            self.fast_step(ext);
        }
        
        // === MEDIUM PATH: learning + regulation ===
        self.medium_tick_counter += 1;
        if self.medium_tick_counter >= self.config.medium_path_interval {
            self.medium_tick_counter = 0;
            
            // Sync fast-path results to structs
            self.sync_hot_to_structs();
            
            // Endoquilibrium: sense, predict, regulate
            if let Some(ref mut endo) = self.endoquilibrium {
                let vitals = endo.sense(self);
                let adjustments = endo.regulate(&vitals);
                endo.apply(&adjustments);
                self.endo_threshold_bias = endo.channels.threshold_bias;
            }
            
            // Learning: STDP, DFA, analog readout updates
            self.apply_learning();
            
            // Sync updated thresholds back to hot arrays
            self.sync_structs_to_hot();
        }
        
        // === SLOW PATH: structural changes ===
        self.slow_tick_counter += 1;
        if self.slow_tick_counter >= self.config.slow_path_interval {
            self.slow_tick_counter = 0;
            
            // Developmental: division, death, migration, synaptogenesis
            self.developmental_step();
            
            // Rebuild hot arrays after structural changes
            self.rebuild_hot_arrays();
        }
        
        // Read output from hot arrays
        self.read_analog_output()
    }
}
```

---

## 7. Memory Layout and Cache Analysis

| Array | Type | Size at 300 morphons | Size at 2,000 morphons | Cache Lines |
|---|---|---|---|---|
| `voltage` | `Vec<f32>` | 1.2 KB | 8 KB | 2 / 125 |
| `threshold` | `Vec<f32>` | 1.2 KB | 8 KB | 2 / 125 |
| `fired` | `BitVec` | 38 bytes | 250 bytes | 1 / 4 |
| `fired_prev` | `BitVec` | 38 bytes | 250 bytes | 1 / 4 |
| `refractory` | `Vec<u8>` | 300 bytes | 2 KB | 1 / 32 |
| `idx_to_node` | `Vec<NodeIndex>` | 1.2 KB | 8 KB | 2 / 125 |
| **Total hot data** | | **~4 KB** | **~27 KB** | **fits L1** / **fits L1** |

At 300 morphons, the entire hot dataset fits in 4 KB — a single L1 cache page. At 2,000 morphons, it's 27 KB — still fits in the 32-48 KB L1 data cache of modern CPUs. The fast-path inner loop runs entirely from L1 cache, never touching main memory.

Compare to AoS: 300 Morphon structs at ~200 bytes each = 60 KB — spills L1, pollutes L2 with cold fields (position, lineage, epistemic state) that the fast path never reads.

---

## 8. What's Deferred to Pulse Kernel v3.0 (Full SoA)

When profiling shows petgraph edge iteration is the bottleneck (expected at >5K morphons):

| Feature | v2.0 (Lite, now) | v3.0 (Full SoA, later) |
|---|---|---|
| Morphon storage | AoS structs in petgraph + 4 hot arrays | Full SoA MorphonStore |
| Synapse storage | petgraph edges | CSR + CSC |
| Topology changes | O(1) via petgraph | O(S) CSR rebuild (batched) |
| Structural plasticity | Natural (petgraph add/remove) | Requires batch + rebuild cycle |
| Cache efficiency | Good (hot arrays in L1) | Excellent (everything in SoA) |
| Implementation cost | 6–8 hours | 30–40 hours (rewrite) |
| When to build | After Endoquilibrium validated | When hitting >5K morphons |

Also deferred: SIMD intrinsics (the compiler auto-vectorizes the f32 loops), Rayon parallelization (not needed under 5K), circular delay buffer (add when delay variation becomes measurable), NUMA awareness (irrelevant at current scale).

---

## 9. Implementation Plan

| Step | What | Hours | Test |
|---|---|---|---|
| 1 | Define `HotArrays` struct on `System` | 0.5 | Compiles |
| 2 | Implement `rebuild_hot_arrays()` from petgraph | 1 | Round-trip: rebuild → check values match structs |
| 3 | Implement `fast_integrate()`, `fast_threshold_check()`, `fast_reset()` | 2 | Unit test: single tick matches old `step()` output |
| 4 | Implement `sync_hot_to_structs()` and `sync_structs_to_hot()` | 1 | Round-trip: sync down → sync up → values unchanged |
| 5 | Rewire `process()` to use fast path + sync | 1.5 | CartPole benchmark: identical results to old code |
| 6 | Update k-WTA, analog readout to use hot arrays | 1 | Classification benchmark: identical results |
| 7 | Profile and measure speedup | 1 | `cargo bench` comparison |

**Total: 6–8 hours.** Each step is independently testable. The critical validation is Step 5: CartPole and classification benchmarks must produce statistically identical results before and after the change. Any deviation means a sync bug.

---

## 10. Relationship to Endoquilibrium

Endoquilibrium should be implemented BEFORE the Pulse Kernel Lite. Reason: Endoquilibrium needs to read firing rates, eligibility density, and weight entropy. These computations are currently scattered across the codebase. Implementing Endoquilibrium first forces you to centralize vital-sign computation, which naturally creates the `compute_firing_rates()` function that the hot arrays later optimize.

The implementation sequence:

```
Week 1:  Endoquilibrium (vitals sensing, predictor, 6 regulation rules)
         → Run CartPole, see if dynamic regulation breaks the plateau
         
Week 2:  Pulse Kernel Lite (hot arrays, fast path, sync protocol)
         → Run CartPole, verify identical results, measure speedup
         
Week 3+: Use the combined system to scale to 1000+ morphons on MNIST
         → Profile, identify next bottleneck
```

If Endoquilibrium breaks the plateau (Associative FR rises above 0%, CartPole avg climbs past 30), the Pulse Kernel Lite enables scaling to the morphon counts needed for MNIST. If Endoquilibrium doesn't break the plateau, the bottleneck is still in the learning dynamics and no amount of fast-path optimization helps.

---

## 11. Success Criteria

The Pulse Kernel Lite is successful if:

1. **Behavioral equivalence:** CartPole and classification benchmarks produce statistically identical results (within f32 rounding tolerance) before and after the change.
2. **Measurable speedup:** The fast-path inner loop is at least 2x faster than the current AoS version at 300 morphons (measured by `cargo bench`).
3. **No regression in structural plasticity:** Division, death, synaptogenesis, and pruning continue to work correctly, with hot arrays automatically rebuilt after each structural change.
4. **Clean integration:** Endoquilibrium's vital-sign sensing reads from hot arrays without performance regression.

If any criterion fails, the change is reverted and the current AoS approach is retained until the crossover point is reached empirically.

---

*Pulse Kernel Lite — fast enough for now, ready to scale when needed.*

*TasteHub GmbH, Wien, April 2026*
