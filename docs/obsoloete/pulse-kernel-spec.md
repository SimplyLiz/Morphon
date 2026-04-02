# Pulse Kernel
## The Spiking Heart of MORPHON
### Technical Specification v1.0 — TasteHub GmbH, April 2026

---

| | |
|---|---|
| **MORPHON Layer** | Runtime — the Fast Path (innermost loop) |
| **Clock** | ~1ms real-time target per tick at 2,000 morphons |
| **Language** | Rust (`morphon-core` crate, `#![no_std]`-compatible inner loop) |
| **Parallelism** | Rayon for CPU, future CUDA/Metal for GPU |
| **Data Layout** | Structure-of-Arrays (SoA) for cache-line efficiency |
| **Status** | Implementation Specification |

---

## 1. What the Pulse Kernel Is

The Pulse Kernel is the innermost processing loop of MORPHON. It handles spike propagation, voltage integration, threshold comparison, and spike delivery — the millisecond-by-millisecond physics of the network. Every other MORPHON subsystem (Endoquilibrium, Developmental Engine, TruthKeeper, LocalParams) operates on top of the state that the Pulse Kernel produces.

The biological analogy is the **cardiac conduction system** — the SA node that generates the heartbeat. It doesn't decide what the heart should do. It generates the electrical pulse that makes everything else possible. The Pulse Kernel generates the computational pulse that drives the morphon network.

### 1.1 Design Principles

**Speed above all.** The Pulse Kernel runs at the highest frequency of any MORPHON subsystem. At 2,000 morphons with ~100 synapses each (200K connections), it must complete a full tick in <1ms on a modern CPU. This means no allocations, no indirection, no hash lookups in the hot path. Everything is pre-allocated SoA with linear memory access.

**SoA, not AoS.** The current MORPHON prototype uses Array-of-Structs (AoS) — each Morphon is a struct with all its fields. This is intuitive but cache-hostile: iterating over all voltages touches every Morphon struct, pulling in threshold, eligibility, position, lineage, and dozens of other fields that aren't needed for the voltage update. SoA stores each field in its own contiguous array. Iterating over voltages touches only voltages — one cache line fills 8 f64 values instead of 1.

**Zero-cost death.** When a morphon dies (apoptosis), the Pulse Kernel doesn't deallocate memory. It marks the slot as `inactive` in a bitfield. The slot is recycled when a new morphon is born (mitosis). This makes cell death and division nearly free — no allocator calls, no fragmentation, no GC pauses.

**Deterministic within a tick.** The Pulse Kernel processes all morphons in a fixed order within each tick. Spikes generated in tick N are delivered in tick N+1 (one-tick propagation delay). This makes the system deterministic given the same initial state and inputs, which is critical for debugging and reproducibility.

---

## 2. Data Layout: The SoA Morphon Store

### 2.1 Core Arrays

The morphon population is stored as parallel arrays, indexed by `MorphonIdx` (a `u32` newtype). All arrays have the same length: `capacity` (pre-allocated maximum). Active morphons are tracked by a bitfield.

```rust
/// The core SoA storage for all morphon state.
/// Every field is a contiguous array indexed by MorphonIdx.
/// This is the ONLY data structure the Pulse Kernel touches in the hot path.
pub struct MorphonStore {
    pub capacity: u32,
    
    // === LIFECYCLE ===
    /// Bitfield: 1 = alive, 0 = dead/empty slot.
    /// Checked before every operation. Dead slots are skipped.
    pub alive: BitVec,
    
    /// Free list for slot recycling. Push on death, pop on birth.
    pub free_slots: Vec<MorphonIdx>,
    
    // === IDENTITY (cold — rarely accessed in hot path) ===
    pub id: Vec<MorphonId>,              // unique ID (for external reference)
    pub cell_type: Vec<CellType>,        // Sensory, Associative, Motor, Modulatory
    pub generation: Vec<u16>,            // lineage depth
    pub parent: Vec<Option<MorphonIdx>>, // parent slot (for lineage tracking)
    
    // === VOLTAGE & SPIKING (hot — accessed every tick) ===
    pub voltage: Vec<f64>,       // membrane potential
    pub threshold: Vec<f64>,     // adaptive firing threshold
    pub fired: BitVec,           // did this morphon fire THIS tick?
    pub refractory: Vec<u8>,     // refractory countdown (ticks remaining)
    
    // === MODULATION (warm — accessed every medium-path tick) ===
    pub reward_sensitivity: Vec<f32>,
    pub novelty_sensitivity: Vec<f32>,
    pub arousal_sensitivity: Vec<f32>,
    pub homeostasis_sensitivity: Vec<f32>,
    
    // === LEARNING (warm — accessed during STDP/DFA updates) ===
    pub local_params: Vec<LocalParams>,  // per-morphon learning rule params
    pub feedback_signal: Vec<f64>,       // DFA backprojection for this morphon
    pub param_fitness: Vec<f64>,         // energy-based fitness for LocalParams
    
    // === METABOLIC (cold — accessed on slow path) ===
    pub energy: Vec<f32>,
    pub age: Vec<u32>,                   // ticks since birth
    
    // === POSITION (cold — accessed during migration) ===
    pub position: Vec<[f64; 2]>,         // Poincaré disk coordinates
    pub cluster_id: Vec<Option<ClusterIdx>>,
    
    // === ANALOG READOUT (hot for motor morphons only) ===
    pub readout_weights: Vec<Vec<f64>>,  // only populated for Motor morphons
}
```

### 2.2 Synapse Storage: Compressed Sparse Row (CSR)

Synapses are the bottleneck. At 2,000 morphons × 100 average connections = 200K synapses. The Pulse Kernel needs to: (1) iterate over all incoming synapses of a morphon to compute input current, and (2) iterate over all outgoing synapses of a firing morphon to deliver spikes.

CSR (Compressed Sparse Row) format gives O(1) access to all incoming synapses per morphon and O(k) iteration where k is the number of connections:

```rust
/// Compressed Sparse Row format for synapse storage.
/// Optimized for "give me all inputs to morphon j" — the most common operation.
pub struct SynapseStore {
    /// For morphon j, its incoming synapses are at indices
    /// row_ptr[j]..row_ptr[j+1] in the data arrays.
    pub row_ptr: Vec<u32>,       // length = capacity + 1
    
    // === SYNAPSE DATA (parallel arrays, indexed by synapse_idx) ===
    pub source: Vec<MorphonIdx>,       // presynaptic morphon
    pub weight: Vec<f64>,              // synaptic weight
    pub delay: Vec<u8>,                // propagation delay (ticks, 0-255)
    pub eligibility: Vec<f64>,         // current eligibility trace value
    pub tag: Vec<f64>,                 // synaptic tag (for tag-and-capture)
    pub tag_strength: Vec<f64>,        // strength when tag was set
    
    // === REVERSE INDEX (for outgoing iteration) ===
    /// CSC (Compressed Sparse Column) for "give me all outputs of morphon i"
    pub col_ptr: Vec<u32>,
    pub col_indices: Vec<u32>,  // indices into the main arrays
    
    pub total_synapses: u32,
}
```

**Why CSR + CSC (dual index)?** The Pulse Kernel needs both directions:
- **Input current computation:** "What signals arrive at morphon j?" → CSR (row_ptr, iterate `row_ptr[j]..row_ptr[j+1]`)
- **Spike delivery:** "Where does morphon i's spike go?" → CSC (col_ptr, iterate `col_ptr[i]..col_ptr[i+1]`)

The CSC reverse index is rebuilt on the Slow Path when the topology changes (synaptogenesis, pruning). The Pulse Kernel never modifies the CSR/CSC structure — it only reads weights and writes eligibility. Structural changes are batched and applied between ticks by the Developmental Engine.

### 2.3 Memory Budget

| Component | Per Unit | At 2,000 morphons | At 10,000 morphons |
|---|---|---|---|
| MorphonStore (all arrays) | ~200 bytes | 400 KB | 2 MB |
| SynapseStore (100 avg connections) | ~40 bytes/synapse | 8 MB (200K syn) | 40 MB (1M syn) |
| Spike queue (ring buffer) | 4 bytes/event | ~50 KB | ~250 KB |
| **Total** | | **~8.5 MB** | **~42 MB** |

This fits in L3 cache on any modern CPU. At 10K morphons it spills to main memory but the SoA layout ensures sequential access patterns that the hardware prefetcher handles well.

---

## 3. The Tick: What Happens Each Millisecond

### 3.1 The Five Phases

Each tick of the Pulse Kernel executes five phases in strict order. No phase depends on the output of a later phase within the same tick (no circular dependencies). Spikes from tick N arrive at tick N+1.

```
Tick N:
  Phase 1: INTEGRATE    — sum incoming currents, update voltages
  Phase 2: THRESHOLD    — compare voltage to threshold, mark fires
  Phase 3: EMIT         — queue spikes from firing morphons
  Phase 4: RESET        — reset fired morphons, update refractory
  Phase 5: DELIVER      — deliver queued spikes from tick N-1
```

### 3.2 Phase 1: Integrate

For each alive, non-refractory morphon, sum all incoming synaptic currents and external input, then leak the voltage toward resting potential.

```rust
impl PulseKernel {
    /// Phase 1: Voltage integration.
    /// LIF dynamics: dV/dt = -(V - V_rest)/tau_m + I_syn/C_m
    /// Discretized: V[t+1] = V[t] * decay + I_total * gain
    fn integrate(
        &self,
        store: &mut MorphonStore,
        synapses: &SynapseStore,
        external_input: &[f64],  // from feed_input()
    ) {
        let decay = (-1.0 / self.config.tau_membrane).exp();  // precomputed
        let gain = 1.0 - decay;
        
        for j in 0..store.capacity {
            if !store.alive[j] || store.refractory[j] > 0 {
                continue;
            }
            
            // Sum incoming synaptic currents (CSR iteration)
            let mut i_syn: f64 = 0.0;
            let start = synapses.row_ptr[j] as usize;
            let end = synapses.row_ptr[j + 1] as usize;
            
            for syn_idx in start..end {
                let pre = synapses.source[syn_idx];
                // Only deliver if presynaptic morphon fired in PREVIOUS tick
                // (spike_buffer stores previous tick's fires)
                if self.spike_buffer.fired_last_tick(pre) {
                    i_syn += synapses.weight[syn_idx];
                }
            }
            
            // Add external input (for sensory morphons)
            if (j as usize) < external_input.len() {
                i_syn += external_input[j as usize];
            }
            
            // Add DFA feedback signal (for associative morphons)
            i_syn += store.feedback_signal[j];
            
            // LIF update
            store.voltage[j] = store.voltage[j] * decay + i_syn * gain;
        }
    }
}
```

**Performance note:** The inner loop (synapse iteration) is the hottest code in the entire system. At 200K synapses, this loop executes 200K times per tick. The CSR layout ensures sequential memory access through the `weight` and `source` arrays. The `spike_buffer.fired_last_tick()` check is a bitfield lookup (1 bit per morphon, entire population fits in one cache line at 2K morphons).

### 3.3 Phase 2: Threshold Comparison

```rust
    /// Phase 2: Threshold comparison.
    /// Marks morphons that fire this tick.
    fn threshold_check(&self, store: &mut MorphonStore) {
        store.fired.fill(false);  // clear previous tick
        
        for j in 0..store.capacity {
            if !store.alive[j] || store.refractory[j] > 0 {
                continue;
            }
            
            // Apply Endoquilibrium threshold bias
            let effective_threshold = store.threshold[j] + self.endo_threshold_bias;
            
            if store.voltage[j] >= effective_threshold {
                store.fired.set(j, true);
            }
        }
    }
```

### 3.4 Phase 3: Emit Spikes

```rust
    /// Phase 3: Queue spike events from firing morphons.
    /// Spikes are NOT delivered this tick — they go into the spike_buffer
    /// for delivery in Phase 5 of the NEXT tick.
    fn emit_spikes(&mut self, store: &MorphonStore) {
        // Swap buffers: current fires become "last tick's fires" for next tick
        self.spike_buffer.advance_tick();
        
        for j in 0..store.capacity {
            if store.fired[j] {
                self.spike_buffer.record_fire(j);
                self.stats.spikes_this_tick += 1;
            }
        }
    }
```

### 3.5 Phase 4: Reset and Refractory

```rust
    /// Phase 4: Reset fired morphons to resting potential.
    /// Start refractory period.
    fn reset_fired(&self, store: &mut MorphonStore) {
        for j in 0..store.capacity {
            if store.fired[j] {
                store.voltage[j] = self.config.v_rest;
                store.refractory[j] = self.config.refractory_ticks;
            }
            // Decrement refractory counters
            if store.refractory[j] > 0 {
                store.refractory[j] -= 1;
            }
        }
    }
```

### 3.6 Phase 5: Deliver Previous Spikes (with delay)

For synapses with nonzero delay, spikes are buffered and delivered after `delay` ticks. For zero-delay synapses, spikes from tick N-1 are delivered in tick N's Phase 1 (already handled above). This phase handles delayed spikes:

```rust
    /// Phase 5: Deliver delayed spikes.
    /// Spikes with delay > 1 tick are stored in a delay buffer
    /// and delivered when their delay expires.
    fn deliver_delayed(&mut self, store: &mut MorphonStore, synapses: &SynapseStore) {
        let due_spikes = self.delay_buffer.drain_due(self.current_tick);
        
        for spike in due_spikes {
            let target = spike.target;
            if store.alive[target] {
                store.voltage[target] += spike.weight;
            }
        }
    }
```

### 3.7 The Complete Tick

```rust
impl PulseKernel {
    /// Execute one complete tick of the Pulse Kernel.
    /// This is the function that runs at ~1ms intervals.
    pub fn tick(
        &mut self,
        store: &mut MorphonStore,
        synapses: &SynapseStore,
        external_input: &[f64],
    ) -> TickStats {
        self.stats.reset();
        
        // Phase 1: Integrate incoming currents
        self.integrate(store, synapses, external_input);
        
        // Phase 2: Check thresholds
        self.threshold_check(store);
        
        // Phase 3: Queue spikes
        self.emit_spikes(store);
        
        // Phase 4: Reset fired morphons
        self.reset_fired(store);
        
        // Phase 5: Deliver delayed spikes
        self.deliver_delayed(store, synapses);
        
        // Bookkeeping
        self.current_tick += 1;
        self.stats.alive_count = store.alive.count_ones() as u32;
        self.stats.clone()
    }
}

/// Statistics produced by each tick.
/// Fed to Endoquilibrium as part of vital signs sensing.
#[derive(Clone, Default)]
pub struct TickStats {
    pub spikes_this_tick: u32,
    pub alive_count: u32,
    pub tick_number: u64,
}
```

---

## 4. The Spike Buffer: Double-Buffered Bitfield

Spikes from tick N must be visible to tick N+1's integration phase. A double-buffered bitfield handles this with zero allocation:

```rust
/// Double-buffered spike recording.
/// Buffer A records fires from tick N.
/// Buffer B records fires from tick N-1 (used for integration in tick N).
/// Swap every tick.
pub struct SpikeBuffer {
    buffers: [BitVec; 2],
    current: usize,  // 0 or 1: which buffer is "this tick"
}

impl SpikeBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffers: [BitVec::repeat(false, capacity), BitVec::repeat(false, capacity)],
            current: 0,
        }
    }
    
    /// Called at the start of Phase 3 (emit).
    /// Swaps buffers so last tick's fires are available for next tick's integration.
    pub fn advance_tick(&mut self) {
        self.current = 1 - self.current;
        self.buffers[self.current].fill(false);  // clear the new "current" buffer
    }
    
    /// Record that morphon j fired this tick.
    pub fn record_fire(&mut self, j: u32) {
        self.buffers[self.current].set(j as usize, true);
    }
    
    /// Check if morphon j fired LAST tick (used in Phase 1 integration).
    #[inline(always)]
    pub fn fired_last_tick(&self, j: u32) -> bool {
        self.buffers[1 - self.current][j as usize]
    }
}
```

**Why bitfield, not a list of firing indices?** At 10% firing rate with 2,000 morphons, ~200 morphons fire per tick. A list of 200 indices requires searching or hashing during integration. A bitfield requires one bit test per synapse source lookup — and the entire 2,000-morphon bitfield fits in 250 bytes (4 cache lines). The bitfield approach is branchless and cache-friendly.

---

## 5. The Shared Heartbeat: Pulse Kernel's Contribution

The Shared Heartbeat is a global state struct that every subsystem can read. The Pulse Kernel writes to it after every tick:

```rust
/// Global state visible to all MORPHON subsystems.
/// The Pulse Kernel updates spiking fields every tick.
/// Endoquilibrium reads them on the Medium Path.
/// The Developmental Engine reads them on the Slow Path.
pub struct Heartbeat {
    // === WRITTEN BY PULSE KERNEL (every tick) ===
    pub current_tick: u64,
    pub spikes_this_tick: u32,
    pub alive_count: u32,
    pub firing_rate: f32,           // EMA of spikes/alive over recent ticks
    pub firing_rate_by_type: [f32; 4],  // per CellType EMA
    
    // === WRITTEN BY ENDOQUILIBRIUM (Medium Path) ===
    pub global_arousal: f32,        // current arousal level
    pub threshold_bias: f32,        // global threshold offset
    pub plasticity_mult: f32,       // global learning rate multiplier
    pub channel_gains: [f32; 4],    // R, N, A, H channel gains
    
    // === WRITTEN BY DEVELOPMENTAL ENGINE (Slow Path) ===
    pub population_pressure: f32,   // how close to capacity
    pub division_count_recent: u32, // divisions in last slow-path window
    pub apoptosis_count_recent: u32,
    
    // === WRITTEN BY TRUTHKEEPER ===
    pub contradiction_count: u32,
    pub safe_mode: bool,
}
```

The Heartbeat is `Arc<RwLock<Heartbeat>>` — the Pulse Kernel takes a write lock once per tick (to update spiking stats), which is fast because the struct is small. All other systems take read locks only. If contention becomes an issue at scale, the spiking stats can be moved to an atomic struct with relaxed ordering.

---

## 6. Interaction with the Developmental Engine

The Developmental Engine operates on the Slow Path (~1s intervals). It manages the population: births, deaths, and structural changes. The critical contract between Pulse Kernel and Developmental Engine is:

**The Pulse Kernel never modifies the CSR/CSC structure.** It reads weights and source indices, and writes voltage, eligibility, and fired state. All structural modifications (adding/removing morphons, adding/removing synapses) are performed by the Developmental Engine between ticks.

### 6.1 Cell Death (Apoptosis)

When the Developmental Engine decides a morphon should die:

```rust
impl DevelopmentalEngine {
    pub fn kill_morphon(&mut self, store: &mut MorphonStore, idx: MorphonIdx) {
        // 1. Mark as dead in the bitfield
        store.alive.set(idx as usize, false);
        
        // 2. Return energy to global pool
        self.heartbeat.write().energy_pool += store.energy[idx];
        store.energy[idx] = 0.0;
        
        // 3. Push slot to free list for recycling
        store.free_slots.push(idx);
        
        // 4. Mark synapses for cleanup (done in batch before next CSR rebuild)
        self.pending_synapse_removals.push(idx);
        
        // Note: voltage, threshold, etc. are NOT zeroed — the alive bitfield
        // ensures the Pulse Kernel skips this slot. Cleaning is deferred.
    }
}
```

**Cost: ~5 nanoseconds.** Set one bit, push one u32. The Pulse Kernel's `if !store.alive[j]` check in every phase skips the dead slot automatically. No reallocation, no defragmentation, no GC.

### 6.2 Cell Division (Mitosis)

When the Developmental Engine creates a new morphon:

```rust
impl DevelopmentalEngine {
    pub fn divide_morphon(
        &mut self,
        store: &mut MorphonStore,
        parent_idx: MorphonIdx,
        rng: &mut impl Rng,
    ) -> Option<MorphonIdx> {
        // 1. Get a free slot
        let child_idx = store.free_slots.pop()?;
        
        // 2. Mark as alive
        store.alive.set(child_idx as usize, true);
        
        // 3. Copy parent state with asymmetric division
        store.voltage[child_idx] = 0.0;  // start at resting potential
        store.threshold[child_idx] = store.threshold[parent_idx];
        store.cell_type[child_idx] = CellType::Stem;  // asymmetric: child starts undifferentiated
        store.generation[child_idx] = store.generation[parent_idx] + 1;
        store.parent[child_idx] = Some(parent_idx);
        store.energy[child_idx] = store.energy[parent_idx] * 0.4;  // child gets 40%
        store.energy[parent_idx] *= 0.6;  // parent keeps 60%
        store.refractory[child_idx] = 0;
        store.age[child_idx] = 0;
        
        // 4. Inherit LocalParams with mutation (from LocalParams spec)
        let mutation_rate = self.compute_mutation_rate(parent_idx, store);
        store.local_params[child_idx] = store.local_params[parent_idx]
            .inherit_with_mutation(rng, mutation_rate);
        
        // 5. Queue synapse creation (batch before next CSR rebuild)
        self.pending_synapse_additions.push(ChildSynapses {
            child: child_idx,
            parent: parent_idx,
            inherit_fraction: 0.5,  // inherit 50% of parent's connections
        });
        
        Some(child_idx)
    }
}
```

### 6.3 CSR Rebuild

After a batch of structural changes (births, deaths, new synapses, pruned synapses), the Developmental Engine rebuilds the CSR/CSC indices. This happens on the Slow Path, not during ticks:

```rust
impl DevelopmentalEngine {
    /// Rebuild CSR/CSC after structural changes.
    /// Called once per Slow Path cycle (not per tick).
    /// Cost: O(S) where S = total synapses. At 200K synapses, ~1ms.
    pub fn rebuild_topology(&mut self, synapses: &mut SynapseStore, store: &MorphonStore) {
        // 1. Remove synapses to/from dead morphons
        self.prune_dead_synapses(synapses, store);
        
        // 2. Add synapses for new morphons
        self.create_child_synapses(synapses, store);
        
        // 3. Rebuild CSR (row_ptr) from scratch
        synapses.rebuild_csr();
        
        // 4. Rebuild CSC (col_ptr) from scratch
        synapses.rebuild_csc();
        
        // Clear pending queues
        self.pending_synapse_removals.clear();
        self.pending_synapse_additions.clear();
    }
}
```

---

## 7. Interaction with TruthKeeper

TruthKeeper monitors network integrity on the Slow Path. It reads the Heartbeat and the Pulse Kernel's firing history to detect anomalies:

### 7.1 Contradiction Detection

```rust
impl TruthKeeper {
    /// Check if any morphon's firing contradicts a verified belief.
    /// Called on the Slow Path (~1s intervals).
    pub fn check_contradictions(
        &mut self,
        store: &MorphonStore,
        heartbeat: &Heartbeat,
    ) -> Vec<Contradiction> {
        let mut contradictions = Vec::new();
        
        // Check: is a motor morphon consistently firing for the wrong action?
        // (Requires external ground truth — only available in supervised/RL settings)
        
        // Check: is a cluster's output contradicting its own epistemic state?
        for cluster in &self.monitored_clusters {
            if cluster.epistemic_state == EpistemicState::Supported
                && cluster.recent_error_rate > 0.5
            {
                contradictions.push(Contradiction {
                    cluster: cluster.id,
                    reason: "Supported cluster has >50% error rate",
                });
                // Slam plasticity to zero for this cluster
                for morphon_idx in &cluster.members {
                    store.local_params[*morphon_idx].plasticity_rate = 0.0;
                }
            }
        }
        
        // Check: runaway division (digital cancer)
        if heartbeat.division_count_recent > self.config.max_divisions_per_window {
            contradictions.push(Contradiction {
                cluster: ClusterIdx::GLOBAL,
                reason: "Runaway division detected — triggering mass apoptosis",
            });
            // Set safe_mode to halt all structural changes
            self.set_safe_mode(true);
        }
        
        contradictions
    }
}
```

### 7.2 TruthKeeper → Developmental Engine Pipeline

```
TruthKeeper detects contradiction
    → marks cluster as CONTESTED (epistemic state change)
    → opens inhibitory boundary (permeability ↑)
    → if contradiction persists >N ticks:
        → marks morphons for apoptosis (via Developmental Engine)
        → returns energy to global pool
        → Endoquilibrium detects energy_pressure drop, triggers new growth elsewhere
```

---

## 8. SIMD and Parallelization Strategy

### 8.1 SIMD Vectorization

The Pulse Kernel's inner loops are perfect SIMD candidates — they perform the same operation on contiguous f64 arrays. With AVX-512, each instruction processes 8 f64 values simultaneously:

```rust
// Phase 1 voltage update — SIMD-friendly version:
// Process 8 morphons at a time using f64x8
for chunk in store.voltage.chunks_exact_mut(8) {
    let v = f64x8::from_slice(chunk);
    let decayed = v * f64x8::splat(decay);
    // ... add synaptic input (gathered from CSR, harder to vectorize)
    decayed.write_to_slice(chunk);
}
```

The voltage decay and reset operations vectorize trivially. The synaptic input summation is harder because it involves indirect access (gather from weight array based on source morphon firing). This is the main bottleneck — on AVX-512 hardware, `vpgatherdd` can help but with irregular access patterns the speedup is modest (~2x rather than 8x).

### 8.2 Rayon Parallelization

For >5,000 morphons, parallelize across morphons using Rayon. The five phases are embarrassingly parallel across morphons (each morphon's update depends only on the previous tick's state, not on other morphons' current-tick updates):

```rust
use rayon::prelude::*;

fn integrate_parallel(
    store: &mut MorphonStore,
    synapses: &SynapseStore,
    spike_buffer: &SpikeBuffer,
    config: &PulseConfig,
) {
    let decay = (-1.0 / config.tau_membrane).exp();
    let gain = 1.0 - decay;
    
    // Split into chunks, process in parallel
    store.voltage.par_chunks_mut(256)
        .enumerate()
        .for_each(|(chunk_idx, voltage_chunk)| {
            let base = chunk_idx * 256;
            for (local_j, v) in voltage_chunk.iter_mut().enumerate() {
                let j = base + local_j;
                if !store.alive[j] || store.refractory[j] > 0 {
                    continue;
                }
                let mut i_syn = 0.0;
                let start = synapses.row_ptr[j] as usize;
                let end = synapses.row_ptr[j + 1] as usize;
                for syn_idx in start..end {
                    if spike_buffer.fired_last_tick(synapses.source[syn_idx]) {
                        i_syn += synapses.weight[syn_idx];
                    }
                }
                *v = *v * decay + i_syn * gain;
            }
        });
}
```

**Chunk size 256** is chosen to balance parallelism overhead vs. cache locality. At 2,000 morphons, this gives ~8 chunks — enough to saturate 4–8 cores. At 10,000 morphons, ~40 chunks provide full utilization of modern multi-core CPUs.

---

## 9. Configuration

```rust
pub struct PulseConfig {
    // === Neuron model ===
    pub tau_membrane: f64,       // membrane time constant (ms), default: 20.0
    pub v_rest: f64,             // resting potential, default: 0.0
    pub v_reset: f64,            // post-spike reset voltage, default: 0.0
    pub refractory_ticks: u8,    // refractory period in ticks, default: 3
    
    // === Performance ===
    pub internal_steps: u32,     // process_steps per external call, default: 5
    pub parallel_threshold: u32, // morphon count above which to use Rayon, default: 5000
    pub chunk_size: usize,       // Rayon chunk size, default: 256
    
    // === From Endoquilibrium (updated dynamically) ===
    pub endo_threshold_bias: f64, // added to all thresholds, default: 0.0
}
```

---

## 10. Performance Targets

| Morphon Count | Synapses | Target Tick Time | Strategy |
|---|---|---|---|
| 300 (current) | 10K | <0.1ms | Sequential, no SIMD |
| 2,000 (Phase 1 target) | 200K | <1ms | Sequential + SIMD |
| 10,000 (Phase 2 target) | 1M | <5ms | Rayon parallel + SIMD |
| 100,000 (Phase 3 stretch) | 10M | <50ms | GPU (CUDA/Metal) |

These targets assume a modern x86-64 CPU (Intel i7 or AMD Ryzen, 4+ cores, AVX2). ARM targets (Raspberry Pi 4/5) will be ~3x slower due to lack of AVX, but NEON SIMD provides some acceleration.

---

## 11. Migration Path from Current Prototype

The current MORPHON prototype uses AoS (each morphon is a struct). Migration to the Pulse Kernel's SoA layout requires:

| Step | What | Effort | Risk |
|---|---|---|---|
| Step 1 | Define `MorphonStore` and `SynapseStore` structs | 2 hours | None — additive |
| Step 2 | Write `from_aos()` converter that populates SoA from existing structs | 2 hours | Low — validation by round-tripping |
| Step 3 | Implement `PulseKernel::tick()` with 5 phases | 4 hours | Medium — must match existing behavior |
| Step 4 | Replace `system.process()` inner loop with Pulse Kernel | 2 hours | Medium — regression testing |
| Step 5 | Implement CSR rebuild for Developmental Engine | 3 hours | Medium |
| Step 6 | Add SIMD intrinsics for Phase 1 integration | 3 hours | Low — optional optimization |
| Step 7 | Add Rayon parallelization | 2 hours | Low — optional for >5K morphons |

**Total: 14–18 hours.** Steps 1–4 give you a working Pulse Kernel that replaces the current inner loop. Steps 5–7 add structural change support and performance optimization.

**Critical testing:** After Step 4, run the CartPole and classification benchmarks and verify identical behavior (within floating-point tolerance) to the AoS version. Any deviation means a bug in the SoA translation.

---

## 12. Relationship to Other Subsystems

```
                    External Input
                         │
                         ▼
                  ┌──────────────┐
                  │ PULSE KERNEL │ ←── Endo threshold_bias
                  │  (Fast Path) │ ←── DFA feedback_signal
                  │  ~1ms/tick   │
                  └──────┬───────┘
                         │ TickStats (spikes, firing rates)
                         ▼
                  ┌──────────────┐
                  │ENDOQUILIBRIUM│ ←── Reads vitals from store + stats
                  │(Medium Path) │ ──→ Writes channel_gains, threshold_bias
                  │  ~10ms       │ ──→ Writes plasticity_mult
                  └──────┬───────┘
                         │ Regulation signals
                         ▼
               ┌──────────────────┐
               │LEARNING ENGINE   │ ←── Reads eligibility from SynapseStore
               │(Medium Path)     │ ──→ Writes weight updates to SynapseStore
               │ STDP + DFA +     │ ←── Reads LocalParams per morphon
               │ Analog Readout   │
               └──────────────────┘
                         │
                         ▼
               ┌──────────────────┐
               │DEVELOPMENTAL     │ ←── Reads Heartbeat + Endo vitals
               │ENGINE (Slow Path)│ ──→ Creates/kills morphons in MorphonStore
               │ Division, Death, │ ──→ Rebuilds CSR/CSC in SynapseStore
               │ Migration, Diff. │
               └──────┬───────────┘
                      │
                      ▼
               ┌──────────────────┐
               │  TRUTHKEEPER     │ ←── Reads Heartbeat
               │  (Slow Path)     │ ──→ Sets safe_mode, marks contradictions
               │  Integrity check │ ──→ Signals Developmental Engine (kill/freeze)
               └──────────────────┘
                      │
                      ▼
               ┌──────────────────┐
               │    GOVERNOR      │
               │  Constitutional  │ ──→ Hard limits on all subsystems
               │  Constraints     │
               └──────────────────┘
```

---

*Pulse Kernel — the heartbeat that makes the organism live.*

*TasteHub GmbH, Wien, April 2026*
