# MORPHON Core API Reference

## Quick Start

```rust
use morphon_core::system::{System, SystemConfig};

let config = SystemConfig::default();
let mut system = System::new(config);

// Feed input and get output (with continuous learning)
let output = system.process(&[1.0, 0.5, 0.3]);

// Inject neuromodulatory signals
system.inject_reward(0.8);
system.inject_novelty(0.6);

// Inspect the system's state
let stats = system.inspect();
println!("Morphons: {}", stats.total_morphons);
println!("Clusters: {}", stats.fused_clusters);
```

---

## `types.rs` — Core Types

### Type Aliases

| Alias | Type | Description |
|-------|------|-------------|
| `MorphonId` | `u64` | Unique Morphon identifier |
| `ClusterId` | `u64` | Unique cluster identifier |
| `SynapseId` | `u64` | Unique synapse identifier |
| `LineageId` | `u64` | Lineage tracking identifier |
| `Generation` | `u32` | Division count since seed |
| `ReceptorSet` | `HashSet<ModulatorType>` | Neuromodulator sensitivity set |
| `Position` | `HyperbolicPoint` | Alias — all positions are hyperbolic |

### `CellType` (enum)

| Variant | Description | Default Activation | Default Receptors |
|---------|-------------|-------------------|-------------------|
| `Stem` | Pluripotent, initial state | Sigmoid | All 4 |
| `Sensory` | Input processing | HardThreshold | Novelty, Arousal |
| `Associative` | Pattern recognition | LeakyIntegrator | Reward, Novelty |
| `Motor` | Output generation | Burst | Reward, Arousal |
| `Modulatory` | Internal regulation | Oscillatory | Homeostasis |
| `Fused` | Part of fused cluster | Sigmoid | Reward, Homeostasis |

### `ModulatorType` (enum)

| Variant | Biological Analog | Function |
|---------|-------------------|----------|
| `Reward` | Dopamine | Reinforces recently active eligibility traces |
| `Novelty` | Acetylcholine | Increases plasticity rate systemwide |
| `Arousal` | Noradrenaline | Increases threshold sensitivity |
| `Homeostasis` | Serotonin | Regulates baseline activity level |

### `ActivationFn` (enum)

| Variant | Formula | Used By |
|---------|---------|---------|
| `Sigmoid` | `1/(1+e^(-x))` | Stem, Fused |
| `HardThreshold` | `x > 0 ? 1 : 0` | Sensory |
| `LeakyIntegrator` | `max(x, 0.01x)` | Associative |
| `Burst` | tanh ramp above 0.5 | Motor |
| `Oscillatory` | `sin(πx)` | Modulatory |

Methods:
- `apply(x: f64) -> f64` — apply the function
- `max_output() -> f64` — maximum possible output value (used for threshold clamping)
- `for_cell_type(cell_type) -> Self` — get default for a cell type

### `DevelopmentalProgram` (enum)

`Cortical`, `Hippocampal`, `Cerebellar`, `Custom`.

### Free Functions

- `default_receptors(cell_type) -> ReceptorSet` — get default receptor set for a cell type

### `HyperbolicPoint` (struct)

Position in the Poincaré ball model of hyperbolic space.

| Field | Type | Description |
|-------|------|-------------|
| `coords` | `Vec<f64>` | Coordinates (||coords|| must be < 1) |
| `curvature` | `f64` | Learnable curvature (default 1.0) |

Methods:
- `origin(dimensions) -> Self` — center of the ball (most general)
- `random(dimensions, rng) -> Self` — random point inside ball (max radius 0.9)
- `distance(&self, other) -> f64` — hyperbolic distance
- `exp_map(&self, tangent) -> HyperbolicPoint` — project tangent vector onto manifold
- `log_map(&self, target) -> Vec<f64>` — compute tangent vector to target
- `specificity() -> f64` — distance from origin (0 = general, ~1 = specialized)

### `LifecycleConfig` (struct)

Boolean toggles for lifecycle features. All default to `true`.

| Field | Description |
|-------|-------------|
| `division` | Allow cell division (mitosis) |
| `differentiation` | Allow functional specialization |
| `fusion` | Allow cluster formation |
| `apoptosis` | Allow programmed cell death |
| `migration` | Allow migration in information space |

### `RingBuffer` (struct)

Fixed-capacity circular buffer for activity history.

Methods: `new(capacity)`, `push(value)`, `mean()`, `variance()`, `max()`, `len()`, `is_empty()`, `iter()`

---

## `morphon.rs` — Morphon, Synapse & MetabolicConfig

### `MetabolicConfig` (struct)

V3 Metabolic Budget — controls how morphons earn and spend energy.

| Field | Default | Description |
|-------|---------|-------------|
| `base_cost` | 0.001 | Energy cost per step for being alive |
| `synapse_cost` | 0.0001 | Additional cost per connection per step |
| `utility_reward` | 0.02 | Energy earned per unit of prediction error reduction |
| `basal_regen` | 0.003 | Small unconditional regen (prevents total starvation) |
| `firing_cost` | 0.002 | Extra cost when the morphon spikes |

Part of `SystemConfig`. All fields are tunable at system construction time.

### `Synapse` (struct)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `weight` | `f64` | (param) | Connection weight |
| `delay` | `f64` | 1.0 | Signal delay (learnable) |
| `eligibility` | `f64` | 0.0 | Fast eligibility trace (τ ~ 20 steps) |
| `pre_trace` | `f64` | 0.0 | Pre-synaptic spike trace for STDP (τ ~ 10 steps) |
| `post_trace` | `f64` | 0.0 | Post-synaptic spike trace for STDP (τ ~ 10 steps) |
| `tag` | `f64` | 0.0 | Slow synaptic tag (τ ~ 6000 steps) |
| `tag_strength` | `f64` | 0.0 | Hebbian strength at tagging time |
| `consolidated` | `bool` | false | Whether tag has been captured |
| `age` | `u64` | 0 | Age in simulation steps |
| `usage_count` | `u64` | 0 | How often activated |

Constructors: `new(weight)`, `.with_delay(delay)`

### `Morphon` (struct)

| Group | Field | Type | Default |
|-------|-------|------|---------|
| Identity | `id` | `MorphonId` | (param) |
| | `position` | `Position` | (param) |
| | `lineage` | `Option<MorphonId>` | None |
| | `generation` | `Generation` | 0 |
| Cell Type | `cell_type` | `CellType` | Stem |
| | `differentiation_level` | `f64` | 0.0 |
| | `activation_fn` | `ActivationFn` | Sigmoid |
| | `receptors` | `ReceptorSet` | all 4 |
| State | `potential` | `f64` | 0.0 |
| | `threshold` | `f64` | 0.3 |
| | `refractory_timer` | `f64` | 0.0 |
| | `prediction_error` | `f64` | 0.0 |
| | `desire` | `f64` | 0.0 |
| Learning | `activity_history` | `RingBuffer` | capacity 100 |
| | `fired` | `bool` | false |
| | `input_accumulator` | `f64` | 0.0 |
| | `prev_potential` | `f64` | 0.0 |
| Lifecycle | `age` | `u64` | 0 |
| | `energy` | `f64` | 1.0 |
| | `division_pressure` | `f64` | 0.0 |
| Fusion | `fused_with` | `Option<ClusterId>` | None |
| | `autonomy` | `f64` | 1.0 |
| Homeostasis | `migration_cooldown` | `f64` | 0.0 |
| | `feedback_signal` | `f64` | 0.0 |
| | `homeostatic_setpoint` | `f64` | 0.1 |

Methods:
- `new(id, position) -> Self` — create stem-cell Morphon
- `divide(child_id, rng) -> Self` — mitosis (exp_map offset, inherit ~50% state, stochastic mutation)
- `step(dt, synapse_count, metabolic)` — integrate input, fire/not-fire, update prediction error, desire, threshold, division pressure, metabolic budget (V3), cooldown
- `should_apoptose() -> bool` — age > 1000 && energy < 0.1 && activity < 0.01 && not fused
- `should_divide() -> bool` — division_pressure > 1.0 && energy > 0.3
- `differentiate(target) -> bool` — change cell type + activation function + receptors
- `dedifferentiate()` — reduce differentiation_level, revert to Stem if < 0.2

---

## `neuromodulation.rs` — Four Broadcast Channels

### `Neuromodulation` (struct)

| Field | Default | Decay Rate |
|-------|---------|------------|
| `reward` | 0.0 | 0.95 |
| `novelty` | 0.0 | 0.90 |
| `arousal` | 0.0 | 0.85 |
| `homeostasis` | 0.5 | 0.99 (towards 0.5 baseline) |
| `reward_baseline` | 0.0 | EMA of reward (α=0.01) — used for advantage computation |
| `prev_reward` | 0.0 | Previous step's reward level (for delta computation) |

Methods:
- `inject_reward(strength)`, `inject_novelty(strength)`, `inject_arousal(strength)`, `inject_homeostasis(strength)` — inject signal (clamped to 0.0..1.0)
- `reward_advantage() -> f64` — `(reward - reward_baseline).max(0.0)` — the advantage signal used by the learning rule
- `reward_delta() -> f64` — change in reward since last step
- `inject(channel, strength)` — generic injection
- `level(channel) -> f64` — read current level
- `combined_signal(αr, αn, αa, αh) -> f64` — `αr*R + αn*N + αa*A + αh*H`
- `default_signal() -> f64` — combined with weights (1.0, 0.5, 0.3, 0.1)
- `decay()` — decay all channels toward resting state
- `plasticity_rate() -> f64` — `0.01 + 0.09 * novelty` (range 0.01..0.10)

---

## `learning.rs` — Three-Factor Learning + Tag-and-Capture

### `LearningParams` (struct)

| Field | Default | Description |
|-------|---------|-------------|
| `tau_eligibility` | 20.0 | Fast trace time constant |
| `tau_trace` | 10.0 | STDP spike trace window (Lava/Loihi: 10) |
| `a_plus` | 1.0 | LTP magnitude (post fires, pre_trace > 0) |
| `a_minus` | -1.0 | LTD magnitude (pre fires, post_trace > 0) |
| `tau_tag` | 6000.0 | Slow tag time constant |
| `tag_threshold` | 0.3 | Eligibility threshold for tagging |
| `capture_threshold` | 0.5 | Reward threshold for capture |
| `capture_rate` | 0.1 | Learning rate at capture |
| `weight_max` | 5.0 | Weight clamp |
| `weight_min` | 0.001 | Pruning threshold |
| `alpha_reward` | 1.0 | Reward weight in M(t) |
| `alpha_novelty` | 0.5 | Novelty weight in M(t) |
| `alpha_arousal` | 0.3 | Arousal weight in M(t) |
| `alpha_homeostasis` | 0.1 | Homeostasis weight in M(t) |

### Functions

**`update_eligibility(synapse, pre_fired, post_activity, params, dt)`**

Weight-dependent trace-based STDP (Frémaux & Gerstner 2016, Gilson & Fukai 2011). Updates pre/post traces and eligibility:
1. Decay `pre_trace` and `post_trace` by `exp(-dt/τ_trace)`
2. Compute multiplicative bounds: `ltp_scale = (w_max - w) / w_max`, `ltd_scale = (w + w_max) / (2 * w_max)`
3. If pre fired: `stdp += a_minus * post_trace * ltd_scale` (LTD), then `pre_trace += 1.0`
4. If post active (>0.01): `stdp += a_plus * pre_trace * post_activity * ltp_scale` (LTP), then `post_trace += post_activity`
5. Eligibility: `e += (-e/τ_e + stdp) * dt`, clamped to [-1, 1]
6. Slow tag: if `eligibility > tag_threshold` and not consolidated, `tag = 1.0`, `tag_strength = eligibility`

`post_activity` is continuous: 1.0 for spiking morphons. For Motor morphons: `(sigmoid(potential) - 0.5) * 2 ∈ [-1, 1]` — centered at zero so only above-average potential generates positive post_trace.

**`apply_weight_update(synapse, modulation, params, plasticity_rate, post_receptors)`**

Receptor-gated: only modulation channels matching `post_receptors` are included. Reward channel uses **advantage** (`reward - baseline`, clamped ≥ 0) instead of raw reward.

Standard: `Δw = eligibility * M(t) * plasticity_rate`

Tag-and-Capture: if `tag > 0.1` AND `reward > capture_threshold` AND not consolidated:
`w += capture_rate * tag_strength * reward`, then `consolidated = true`, `tag = 0`.

**`should_prune(synapse, params) -> bool`**

True if: not consolidated AND age > 100 AND |weight| < weight_min AND usage_count < 5.

---

## `resonance.rs` — Signal Propagation

### `SpikeEvent` (struct)

| Field | Type | Description |
|-------|------|-------------|
| `source` | `MorphonId` | Firing Morphon |
| `target` | `MorphonId` | Destination Morphon |
| `strength` | `f64` | `synapse.weight` |
| `delay` | `f64` | Remaining delay |
| `initial_delay` | `f64` | Original delay at creation (for progress calculation) |

### `ResonanceEngine` (struct)

Methods:
- `new() -> Self`
- `propagate(morphons, topology)` — generate SpikeEvents for all firing Morphons' outgoing connections
- `deliver(morphons, dt) -> Vec<SpikeEvent>` — decrement delays, deliver expired spikes to targets' `input_accumulator`
- `last_delivered() -> &[SpikeEvent]` — spikes delivered in the most recent `deliver()` call
- `pending_count() -> usize`
- `pending_spikes() -> &VecDeque<SpikeEvent>` — all spikes still in transit
- `clear()`

Complexity: O(k*N) where k = average connectivity.

---

## `topology.rs` — Dynamic Connection Graph

Built on `petgraph::DiGraph<MorphonId, Synapse>` with a `HashMap<MorphonId, NodeIndex>` for O(1) lookups.

### `Topology` (struct)

Public fields:
- `graph: DiGraph<MorphonId, Synapse>` — the underlying petgraph (direct access for advanced operations)

Methods:
- `new()`, `add_morphon(id)`, `remove_morphon(id)`
- `add_synapse(from, to, synapse)`, `remove_synapse(edge)`
- `incoming(id)`, `outgoing(id)` — get connections with synapse data
- `incoming_synapses_mut(id)` — get edge indices for mutation
- `synapse_mut(edge)`, `synapse_between(from, to)`
- `has_connection(from, to)`, `degree(id)`, `node_index(id) -> Option<NodeIndex>`
- `morphon_count()`, `synapse_count()`
- `all_morphon_ids()`, `all_edges()`
- `duplicate_connections(parent_id, child_id, rng)` — copy ~50% of parent's connections to child with weight mutations (for mitosis)

---

## `morphogenesis.rs` — Runtime Structural Changes

Seven mechanisms operating at different timescales:

### Parameters (`MorphogenesisParams`)

| Field | Default | Description |
|-------|---------|-------------|
| `synaptogenesis_threshold` | 0.6 | Correlation for new connections |
| `pruning_min_age` | 100 | Min synapse age for pruning |
| `division_threshold` | 1.0 | Division pressure trigger |
| `division_min_energy` | 0.3 | Min energy to divide |
| `fusion_correlation_threshold` | 0.95 | Pearson r for fusion |
| `fusion_min_size` | 3 | Min group size for fusion |
| `migration_rate` | 0.05 | Migration step size |
| `apoptosis_min_age` | 1000 | Min age for death eligibility |
| `apoptosis_energy_threshold` | 0.1 | Energy threshold for death |
| `max_morphons` | 10,000 | Hard cap on growth |
| `min_morphons` | 10 | V3 Governor: apoptosis stops below this |
| `min_sensory_fraction` | 0.05 | V3 Governor: min Sensory fraction |
| `min_motor_fraction` | 0.02 | V3 Governor: min Motor fraction |
| `max_single_type_fraction` | 0.80 | V3 Governor: max any single type |
| `transdifferentiation_desire_threshold` | 0.5 | Min chronic desire to trigger A→B conversion |
| `transdifferentiation_min_age` | 500 | Min age before transdifferentiation eligible |

### Functions

| Function | Timescale | Description |
|----------|-----------|-------------|
| `synaptogenesis()` | Slow | New connections between correlated, proximate, unconnected pairs |
| `pruning()` | Slow | Remove synapses where `should_prune()` is true |
| `migration()` | Slow | Move Morphons via `log_map`/`exp_map` in hyperbolic space toward neighbors with lower PE. Respects cooldown and homeostasis-modulated rate. |
| `division()` | Glacial | Mitosis for Morphons where `should_divide()`. Creates child via `exp_map` offset, duplicates ~50% connections. |
| `differentiation()` | Glacial | Stem cells specialize based on activity patterns (high consistent→Associative, high variable→Associative) |
| `transdifferentiation()` | Glacial | Direct A→B cell type conversion for non-Stem, non-I/O morphons under chronic mismatch (high desire). Infers target type from incoming connectivity: >60% of inputs from one type implies a role change. Rate 0.01 (5× slower than normal differentiation). |
| `dedifferentiation()` | Glacial | High desire + high arousal → reduce differentiation_level |
| `fusion()` | Glacial | Groups of N>=3 correlated Morphons merge into a Cluster (shared threshold, reduced autonomy) |
| `defusion()` | Glacial | Clusters with high PE variance break apart |
| `apoptosis()` | Glacial | Remove Morphons: old, low energy, low activity, unfused, poorly connected |

Orchestration:
- `step_slow()` — runs synaptogenesis + pruning + migration
- `step_glacial()` — runs division + differentiation + transdifferentiation + dedifferentiation + fusion + defusion + apoptosis

### `Cluster` (struct)

| Field | Type |
|-------|------|
| `id` | `ClusterId` |
| `members` | `Vec<MorphonId>` |
| `shared_threshold` | `f64` |
| `inhibitory_morphons` | `Vec<MorphonId>` |

### `MorphogenesisReport` (struct)

Returned by `step()` — describes what changed during one simulation step.

| Field | Type |
|-------|------|
| `synapses_created` | `usize` |
| `synapses_pruned` | `usize` |
| `morphons_born` | `usize` |
| `morphons_died` | `usize` |
| `differentiations` | `usize` |
| `transdifferentiations` | `usize` |
| `fusions` | `usize` |
| `defusions` | `usize` |
| `migrations` | `usize` |

---

## `memory.rs` — Triple Memory System

### Working Memory

Capacity-limited attractor patterns (default capacity: 7, like Miller's 7 +/- 2).

- `store(pattern, activation)` — add or refresh a pattern (>50% overlap = refresh)
- `step(dt)` — decay all patterns (rate 0.05/step), remove below 0.01
- Eviction: when full, least-activated pattern is removed

### Episodic Memory

One-shot event storage with replay-based consolidation.

- `encode(pattern, reward, novelty, timestamp)` — store new episode
- `replay(count)` — replay top-priority episodes (high reward + novelty - consolidation), increasing their consolidation level
- Eviction: when full, least-consolidated episode is removed

### Procedural Memory

The topology itself IS the procedural memory. This module tracks `TopologySnapshot` history for analysis:

- `record(timestamp, morphon_count, synapse_count, cluster_count)`

### `TripleMemory`

Container: `working: WorkingMemory`, `episodic: EpisodicMemory`, `procedural: ProceduralMemory`.

---

## `homeostasis.rs` — Stability Mechanisms

### Parameters (`HomeostasisParams`)

| Field | Default | Description |
|-------|---------|-------------|
| `scaling_interval` | 50 | How often to run synaptic scaling |
| `inhibition_strength` | 0.3 | Inter-cluster inhibition strength |
| `inhibition_correlation_threshold` | 0.9 | Sync threshold for inhibition |
| `migration_cooldown_duration` | 20.0 | Steps of cooldown after migration |
| `rollback_pe_threshold` | 0.5 | PE increase that triggers rollback |

### Mechanisms

**A) Synaptic Scaling** — `synaptic_scaling(morphons, topology)`

For each Morphon: `scaling_factor = homeostatic_setpoint / actual_firing_rate`. All incoming synapse weights are multiplied by this factor (clamped to 0.5..2.0). Preserves relative weight ratios.

**B) Inter-Cluster Inhibition** — `inter_cluster_inhibition(morphons, clusters, params)`

For each pair of clusters: if both have mean activity > 0.3 and their synchrony > threshold, reduce the potential of all members by `inhibition_strength`.

**C) Migration Damping**

- `can_migrate(morphon) -> bool` — true if `migration_cooldown <= 0`
- `apply_migration_cooldown(morphon, params)` — set cooldown to `migration_cooldown_duration`
- `migration_rate_modifier(homeostasis_level, avg_pe) -> f64` — system-wide modifier: high homeostasis brakes migration, high PE allows it

**D) Checkpoint/Rollback**

### `LocalCheckpoint` (struct)

| Field | Type | Description |
|-------|------|-------------|
| `morphon_states` | `Vec<(MorphonId, f64, f64)>` | (id, prediction_error, potential) per morphon |
| `avg_prediction_error` | `f64` | Average PE at checkpoint time |
| `synapse_states` | `Vec<(MorphonId, MorphonId, f64)>` | (from, to, weight) for rollback |

- `create_checkpoint(morphon_ids, morphons, topology) -> LocalCheckpoint` — snapshot PE, potentials, synapse weights
- `should_rollback(checkpoint, morphon_ids, morphons, params) -> bool` — true if avg PE increased more than threshold
- `rollback_synapses(checkpoint, topology)` — revert synapse weights to checkpoint state

---

## `scheduler.rs` — Dual-Clock Architecture

### `SchedulerConfig` (struct)

| Field | Default | Processes |
|-------|---------|-----------|
| `medium_period` | 10 | Synaptic plasticity, eligibility traces |
| `slow_period` | 100 | Synaptogenesis, pruning, migration |
| `glacial_period` | 1000 | Division, differentiation, fusion, apoptosis |
| `homeostasis_period` | 50 | Synaptic scaling, inter-cluster inhibition |
| `memory_period` | 100 | Procedural topology snapshots |

### `SchedulerTick` (struct)

Result of `config.tick(step_number)`. Fields: `fast` (always true), `medium`, `slow`, `glacial`, `homeostasis`, `memory`.

---

## `developmental.rs` — Bootstrapping Programs

### `DevelopmentalConfig` (struct)

| Field | Type | Default |
|-------|------|---------|
| `program` | `DevelopmentalProgram` | Cortical |
| `seed_size` | `usize` | 100 |
| `dimensions` | `usize` | 8 |
| `initial_connectivity` | `f64` | 0.1 |
| `proliferation_rounds` | `usize` | 3 |
| `type_ratios` | `CellTypeRatios` | sensory 0.2, assoc 0.5, motor 0.2, mod 0.1 |
| `target_input_size` | `Option<usize>` | None |
| `target_output_size` | `Option<usize>` | None |

`target_input_size` / `target_output_size`: when set, developmental creates exactly this many Sensory/Motor morphons for precise I/O port matching.

### `CellTypeRatios` (struct)

| Field | Default |
|-------|---------|
| `sensory` | 0.2 |
| `associative` | 0.5 |
| `motor` | 0.2 |
| `modulatory` | 0.1 |

Presets: `cortical()`, `hippocampal()`, `cerebellar()`.

### Development Phases

`develop(config, rng) -> (HashMap<MorphonId, Morphon>, Topology, next_id)`

1. **Seed**: Create `seed_size` stem-cell Morphons at random hyperbolic positions
2. **Connectivity**: Random connections with probability `initial_connectivity`, weights in [-0.5, 0.5], delays in [1.0, 3.0]
3. **Proliferation**: `proliferation_rounds` rounds of 30% division probability per Morphon
4. **Differentiation**: Assign cell types based on position along primary axis (morphogen gradient)
5. **Pruning**: Remove synapses with |weight| < 0.05

---

## `system.rs` — Top-Level System

### `SystemConfig` (struct)

Aggregates all subsystem configs:
- `developmental`, `learning`, `morphogenesis`, `homeostasis`, `scheduler`, `lifecycle`, `metabolic` (V3)
- `working_memory_capacity` (default 7), `episodic_memory_capacity` (default 1000)
- `dt` (default 1.0)

Methods:
- `save_json(path) -> io::Result<()>` — serialize config to JSON file for reproducibility
- `load_json(path) -> io::Result<Self>` — load config from JSON file

### `System` (struct)

Public fields: `morphons`, `topology`, `resonance`, `modulation`, `clusters`, `memory`, `config`.

Internal fields (pub(crate)):
- `input_ports`, `output_ports` — stable I/O mappings (index → MorphonId)
- `critic_ports` — subset of Associative morphons predicting state value V(s)
- `prev_critic_value`, `last_td_error` — TD error computation state
- `feedback_weights: HashMap<MorphonId, Vec<(MorphonId, f64)>>` — fixed random DFA weights (assoc→motors), initialized once, never updated
- `readout_weights: Vec<HashMap<MorphonId, f64>>` — analog readout weights (output_index→assoc_id→weight), trained by delta rule
- `use_analog_readout: bool` — whether to use analog readout bypass

Key methods:
- `new(config) -> Self` — run developmental program, initialize DFA feedback weights, warm-up
- `step() -> MorphogenesisReport` — one simulation step (see Data Flow above)
- `feed_input(inputs: &[f64])` — distribute raw values to sensory Morphons (zero-bias, fan-out to port groups)
- `read_output() -> Vec<f64>` — read output; if `use_analog_readout`, computes `Σ readout_weights × sigmoid(potential)`, otherwise raw motor potentials
- `process(input) -> Vec<f64>` — feed_input + step + read_output
- `inject_reward(strength)`, `inject_novelty(strength)`, `inject_arousal(strength)`
- `inject_reward_at(output_index, strength)` — targeted two-hop reward at a specific output port
- `inject_inhibition_at(output_index, strength)` — targeted two-hop eligibility decay at a specific output port
- `reward_contrastive(correct_index, reward_strength, inhibit_strength)` — reward correct output, inhibit incorrect ones
- `teach_hidden(correct_index, strength)` — inject DFA feedback signal for a specific correct output
- `teach_supervised(correct_index, learning_rate)` — combined: train_readout + teach_hidden + reward_contrastive
- `teach_supervised_with_input(input, correct_index, learning_rate)` — feed_input + step + teach_supervised in one call
- `enable_analog_readout()` — initialize readout weights with Xavier scaling, enable analog bypass
- `train_readout(correct_index, learning_rate)` — delta rule update on readout weights + DFA backprojection + tag-and-capture on input synapses
- `critic_value() -> f64` — current state value estimate V(s) from critic morphons
- `critic_size() -> usize` — number of critic morphons
- `inject_td_error(reward, gamma) -> f64` — compute TD error (reward + γV(s') - V(s)), inject as reward, return the TD error
- `process_steps(input, n) -> Vec<f64>` — multi-step processing (feed input each step, read output after n steps)
- `inspect() -> SystemStats` — full system statistics
- `diagnostics() -> &Diagnostics` — current learning pipeline diagnostics
- `step_count() -> u64`

### `SystemStats` (struct)

| Field | Type |
|-------|------|
| `total_morphons` | `usize` |
| `total_synapses` | `usize` |
| `fused_clusters` | `usize` |
| `differentiation_map` | `HashMap<CellType, usize>` |
| `max_generation` | `Generation` |
| `avg_energy` | `f64` |
| `avg_prediction_error` | `f64` |
| `firing_rate` | `f64` |
| `working_memory_items` | `usize` |
| `episodic_memory_items` | `usize` |
| `step_count` | `u64` |
| `total_born` | `usize` |
| `total_died` | `usize` |
| `total_transdifferentiations` | `usize` |

---

## `diagnostics.rs` — Learning Pipeline Observability

### `Diagnostics` (struct, Serialize/Deserialize)

| Group | Field | Type | Description |
|-------|-------|------|-------------|
| Weights | `weight_mean` | `f64` | Mean synapse weight |
| | `weight_std` | `f64` | Std dev of synapse weights |
| | `weight_abs_max` | `f64` | Max absolute weight |
| | `total_synapses` | `usize` | Total synapse count |
| Eligibility | `eligibility_mean_abs` | `f64` | Mean absolute eligibility |
| | `eligibility_max_abs` | `f64` | Max absolute eligibility |
| | `eligibility_nonzero_count` | `usize` | Synapses with \|e\| > 0.01 |
| Tags | `active_tags` | `usize` | Synapses with tag > 0.1 |
| | `captures_this_step` | `u64` | Captures this step |
| | `total_captures` | `u64` | Cumulative captures |
| Firing | `firing_by_type` | `HashMap<CellType, (usize, usize)>` | (firing, total) per cell type |
| Spikes | `spikes_delivered_this_step` | `usize` | Delivered this step |
| | `spikes_pending` | `usize` | Still in transit |
| Structural | `rollback_triggered` | `bool` | Rollback this step |
| | `total_rollbacks` | `u64` | Cumulative rollbacks |

Methods:
- `snapshot(morphons, topology) -> Self` — compute diagnostics from current state
- `summary() -> String` — one-line formatted summary for logging
- `firing_summary() -> String` — per-type firing rate percentages

---

## `lineage.rs` — Lineage Tree Export

### `LineageNode` (struct, Serialize/Deserialize)

| Field | Type |
|-------|------|
| `morphon_id` | `MorphonId` |
| `parent_id` | `Option<MorphonId>` |
| `generation` | `Generation` |
| `cell_type` | `CellType` |
| `age` | `u64` |
| `energy` | `f64` |
| `position_specificity` | `f64` |

### `LineageTree` (struct, Serialize/Deserialize)

Built via `build_lineage_tree(morphons)` or `system.lineage_tree()`.

Methods:
- `to_json() -> String` — pretty-printed JSON
- `root_ids() -> Vec<MorphonId>` — morphons with no parent (seed cells)
- `children_of(id) -> Vec<MorphonId>` — direct children
- `max_depth() -> u32` — deepest generation in the tree

---

## `snapshot.rs` — Serialization

### `SystemSnapshot` (struct, Serialize/Deserialize)

Contains: `config`, `morphons`, `connections: Vec<(from, to, Synapse)>`, `clusters`, `modulation`, `next_morphon_id`, `next_cluster_id`, `step_count`, `input_ports`, `output_ports`.

### System Methods

- `snapshot() -> SystemSnapshot` — create serializable snapshot
- `from_snapshot(snapshot) -> System` — restore from snapshot
- `save_json() -> Result<String>` — serialize to JSON
- `save_json_pretty() -> Result<String>` — serialize to pretty JSON
- `load_json(json) -> Result<System>` — deserialize from JSON
- `lineage_tree() -> LineageTree` — build lineage tree from current morphons
- `input_size() -> usize` — number of input ports
- `output_size() -> usize` — number of output ports

---

## `python.rs` — Python Bindings (feature: `python`)

### `morphon.System`

```python
system = morphon.System(seed_size=100, growth_program="cortical", dimensions=8)
output = system.process([1.0, 0.5, 0.3])
system.inject_reward(0.8)
system.inject_novelty(0.6)
system.inject_arousal(0.9)
system.inject_reward_at(0, 0.5)
system.inject_inhibition_at(1, 0.3)
system.reward_contrastive(0, 0.8, 0.3)
system.teach_hidden(0, 0.5)
system.enable_analog_readout()
system.train_readout(0, 0.01)
stats = system.inspect()
json = system.save_json()
restored = morphon.System.load_json(json)
```

Methods: `process`, `feed_input`, `step`, `read_output`, `inject_reward`, `inject_novelty`, `inject_arousal`, `inject_reward_at`, `inject_inhibition_at`, `reward_contrastive`, `teach_hidden`, `enable_analog_readout`, `train_readout`, `inspect`, `save_json`, `load_json` (static), `step_count` (property).

### `morphon.SystemStats`

Read-only properties: `total_morphons`, `total_synapses`, `fused_clusters`, `max_generation`, `avg_energy`, `avg_prediction_error`, `firing_rate`, `working_memory_items`, `episodic_memory_items`, `step_count`, `total_born`, `total_died`, `total_transdifferentiations`, `differentiation_map` (dict of str→int).

---

## `wasm.rs` — WASM Bindings (feature: `wasm`)

### `WasmSystem`

```javascript
const system = new WasmSystem(100, "cortical", 8);
const output = system.process(new Float64Array([1.0, 0.5]));
system.inject_reward(0.8);
system.inject_reward_at(0, 0.5);
system.inject_inhibition_at(1, 0.3);
system.reward_contrastive(0, 0.8, 0.3);
system.teach_hidden(0, 0.5);
system.enable_analog_readout();
system.train_readout(0, 0.01);
const stats = system.inspect();  // JSON string
```

Methods: `process`, `step`, `feed_input`, `read_output`, `inject_reward`, `inject_novelty`, `inject_arousal`, `inject_reward_at`, `inject_inhibition_at`, `reward_contrastive`, `teach_hidden`, `enable_analog_readout`, `train_readout`, `inspect` (JSON), `lineage_json`, `save_json`, `loadJson` (static), `topology_json`, `modulation_json`.

Visualization helpers (for Three.js web viz):
- `fired_ids() -> Vec<u32>` — IDs of morphons that fired this step
- `pending_spikes_flat() -> Vec<f64>` — flat array of [source, target, progress, ...] for spike animation
- `delivered_target_ids() -> Vec<u32>` — target IDs of spikes delivered this step (for flash effect)

Powers the Three.js web visualizer in `web/`.
