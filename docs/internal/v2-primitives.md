# V2 Primitives — Phase 1

Three new primitives added in V2 Phase 1, implementing organizational principles from Levin's Multiscale Competency Architecture. These transform MORPHON from a purely reactive system into one that escapes local minima, communicates spatially, and maintains functional structure.

## 1. Frustration-Driven Stochastic Exploration

**Module**: `types.rs` (FrustrationState, FrustrationConfig), `morphon.rs` (step logic), `morphogenesis.rs` (migration gate), `system.rs` (weight perturbation)

### Biological Basis

Locus coeruleus noradrenaline burst under sustained prediction failure. Organisms increase behavioral variability when stuck — "frustrated exploration" rather than continued exploitation.

### How It Works

Each morphon tracks a `FrustrationState`:

```
stagnation_counter: u32    — consecutive fast ticks where |PE_delta| < 0.005
frustration_level: f64     — tanh(counter / saturation_steps * 3.0), in [0, 1]
noise_amplitude: f64       — 1.0 + frustration_level * (max_noise_multiplier - 1.0)
exploration_mode: bool     — true when frustration_level > 0.3
prev_pe: f64               — previous prediction error for delta computation
```

**Fast tick** (`Morphon::step()`): After PE/desire update, compute PE delta. If `|delta| < stagnation_threshold` AND `desire > 0.1` (the morphon is trying but failing), increment `stagnation_counter`. Otherwise decay it fast (saturating_sub(5) — 5:1 recovery ratio). The `noise_amplitude` scales the existing potential noise: motor morphons are exempt (noise_scale = 0.0 regardless).

**Medium tick** (`System::step()`): Morphons in `exploration_mode` get small random weight perturbations on their incoming synapses. Amplitude: `pseudo_random * weight_perturbation_scale * frustration_level * weight_max`. Consolidated synapses are protected — only unconsolidated weights are perturbed.

**Slow tick** (`migration()`): Frustrated morphons bypass the normal `desire >= 0.3` gate. When `frustration_level > random_migration_threshold` (0.6) and no lower-PE neighbors exist, a random tangent direction is generated. This still goes through radial stripping and exp_map — frustration doesn't break topological safety.

**Self-correcting**: When a random perturbation reduces PE, normal three-factor learning stabilizes it immediately. The stagnation counter decays 5x faster than it accumulates, so frustration dissipates quickly once progress resumes.

### Configuration

`FrustrationConfig` lives inside `MorphogenesisParams`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `enabled` | true | Master switch |
| `stagnation_threshold` | 0.005 | Min PE delta to consider non-stagnant |
| `saturation_steps` | 200 | Counter value at which frustration saturates |
| `exploration_threshold` | 0.3 | Frustration level to enter exploration_mode |
| `max_noise_multiplier` | 5.0 | Maximum noise amplification |
| `weight_perturbation_scale` | 0.01 | Perturbation amplitude (fraction of weight_max) |
| `frustration_migration` | true | Allow frustrated migration without desire gate |
| `random_migration_threshold` | 0.6 | Frustration level for random migration direction |

### Diagnostics

Three fields added to `Diagnostics`:
- `avg_frustration` — mean frustration level across all morphons
- `exploration_mode_count` — morphons currently in exploration mode
- `max_frustration` — peak frustration in the population

Shown in `summary()` as `frust=0.07(7)` (avg frustration 0.07, 7 in exploration mode).

---

## 2. Bioelektrisches Feld (Morphon-Field)

**Module**: `field.rs` (new), `system.rs` (orchestration), `morphogenesis.rs` (migration augmentation)

### Biological Basis

Cells communicate not only via synapses (direct) but via membrane potential fields (Vmem) — a slow, spatially-diffuse information medium. Levin describes this as "Wi-Fi between cell layers". It provides position information and morphogenetic targets that direct structure formation.

### How It Works

A 2D scalar field grid over the Poincare disk. Multiple named layers diffuse independently.

**Projection**: The first two coordinates of the N-dimensional Poincare ball (which lie in (-1, 1)) are mapped to grid indices: `gx = ((coord[0] + 1) * 0.5 * (resolution - 1)).round()`. Dimensions 2..N are intentionally ignored — the field is for coarse spatial communication, not precise addressing.

**Layers** (`FieldType` enum):
- `PredictionError` — where in the system is error high?
- `Energy` — where are metabolic resources?
- `Stress` — where is chronic frustration? (from V2 frustration state)
- `Novelty` — where is new activity?
- `Identity` — functional role targets (used by Target Morphology)

**Slow tick** update cycle:
1. **Write**: Clear all layers. Each morphon writes its PE, energy, frustration_level to the respective layer at its projected grid position (additive, O(N)).
2. **Diffuse**: Discrete 2D heat equation per layer — `next[x,y] = (center + diffusion_rate * laplacian) * (1 - decay_rate)`. O(resolution^2) per layer. At 32x32 = 1024 cells, this is negligible.

**Migration augmentation**: After the neighbor-based tangent vector is computed, the PE field gradient is blended in:
```
tangent[0] = tangent[0] * (1 - field_weight) + (-gradient_x) * field_weight * migration_rate
tangent[1] = tangent[1] * (1 - field_weight) + (-gradient_y) * field_weight * migration_rate
```
Negated because we move AWAY from high PE. Only modifies the first 2 dimensions.

The Identity field gradient is also added (positive direction — move TOWARD high identity) with a weaker weight of 0.1.

### Configuration

`FieldConfig` lives in `SystemConfig`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `enabled` | false | Master switch (opt-in, ~40KB memory) |
| `resolution` | 32 | Grid cells per axis |
| `diffusion_rate` | 0.1 | Fraction of Laplacian applied per slow tick |
| `decay_rate` | 0.05 | Exponential decay per slow tick |
| `active_layers` | [PE, Energy, Stress] | Which field types are active |
| `migration_field_weight` | 0.3 | Blend weight (0 = ignore field, 1 = field only) |

When `target_morphology` is configured, `Identity` is auto-added to `active_layers` during `System::new()`.

### Diagnostics

- `field_pe_max` — peak value in the PredictionError layer
- `field_pe_mean` — mean value in the PredictionError layer

---

## 3. Target Morphology

**Module**: `developmental.rs` (structs, factories, healing), `system.rs` (glacial tick orchestration), `morphogenesis.rs` (differentiation bias, identity migration)

### Biological Basis

Levin's key insight: cells have implicit knowledge of the target form. A salamander regenerates exactly one arm — not two, not zero. The bioelectric prepattern stores a "goal image" that the organism works toward. Defects can be corrected "in software" by inducing the right bioelectric pattern.

### How It Works

**TargetRegion** defines a functional zone in hyperbolic space:
```rust
struct TargetRegion {
    name: String,
    center: HyperbolicPoint,      // where in information space
    radius: f64,                   // hyperbolic distance
    target_cell_type: CellType,    // what should grow here
    target_density: usize,         // how many morphons
    target_connectivity: f64,      // desired avg degree
    identity_strength: f64,        // field broadcast strength
}
```

**TargetMorphology** bundles regions with healing config:
```rust
struct TargetMorphology {
    regions: Vec<TargetRegion>,
    self_healing: bool,            // default: true
    healing_threshold: f64,        // default: 0.5 (trigger at 50% of target)
}
```

**Factory methods**: `cortical(dims)`, `cerebellar(dims)`, `hippocampal(dims)` — each creates 3 regions (Sensory, Associative, Motor) at appropriate positions via `exp_map` from origin along different tangent directions.

**Glacial tick** integration (three mechanisms):

1. **Identity field write**: If field is enabled, the Identity layer is populated — each target region fills grid cells within its projected radius with `identity_strength`. This creates an attractive field for migration.

2. **Self-healing**: For each region, count morphons within `radius` of `center`. If `current / target_density < healing_threshold`:
   - Boost `division_pressure += 0.1` for existing in-region morphons
   - If the region is completely empty AND below `max_morphons`: seed one new morphon near the center with the target cell type

3. **Differentiation bias**: Stem cells inside a target region differentiate toward that region's `target_cell_type`, even if their activity signature would normally keep them as Stem. Region membership overrides activity-based differentiation.

### Configuration

`target_morphology: Option<TargetMorphology>` in `SystemConfig`. Default: `None` (opt-in).

### Diagnostics

`region_health: Vec<(usize, usize, usize)>` in both `Diagnostics` and `SystemStats` — `(region_index, current_count, target_density)` per region. Also available via `TargetMorphology::region_health(&morphons)`.

---

## Cross-Cutting Notes

### Serde Compatibility

All new fields use `#[serde(default)]` — existing configs and snapshots deserialize without breaking. The field system is not persisted in snapshots (ephemeral).

### Performance Impact

- **Frustration**: ~10 f64 ops per morphon per fast tick. With 10K morphons = 100K extra ops/tick. Negligible.
- **Field**: 32x32 x 5 layers diffused every 100 steps. Plus O(N) write/read. Negligible.
- **Target Morphology**: O(N x R) distance computations every 1000 steps (R = 3-5 regions). Negligible.

### `Morphon::step()` Signature

Changed from `step(dt, synapse_count, metabolic)` to `step(dt, synapse_count, metabolic, frustration_config)`. The `FrustrationConfig` reference is obtained from `self.config.morphogenesis.frustration` in the system step loop.

### Interaction Between Primitives

Frustration feeds into the field (Stress layer = frustration_level). The field feeds into migration (PE and Identity gradients). Target morphology feeds into the field (Identity layer) and differentiation. The three primitives form a coherent loop:

```
Frustration → Stress field → migration toward less-stressed regions
Target Morphology → Identity field → migration toward needed regions
Target Morphology → differentiation bias → correct cell types in regions
Target Morphology → self-healing → division pressure in underpopulated regions
```

---

# V2 Primitives — Phase 2

Three additional primitives completing the V2 spec. These address collective efficiency, meta-learning, and offline consolidation.

## 4. Sub-Morphon Plasticity (Adaptive Receptors)

**Module**: `types.rs` (default_receptor_sensitivity), `morphon.rs` (fields, lifecycle), `learning.rs` (apply_weight_update, adapt_receptor_sensitivity), `system.rs` (recording, adaptation call)

### Biological Basis

Receptor sensitivity is not fixed — neurons regulate how strongly they respond to each neurotransmitter. A neuron in a region where dopamine is irrelevant reduces its dopamine receptor density. This is meta-learning: the system learns how to learn.

### How It Works

Each morphon has `receptor_sensitivity: HashMap<ModulatorType, f64>` — continuous values in [0.01, 2.0] per modulator channel. Initialized from `default_receptor_sensitivity(cell_type)`: present receptors = 1.0, absent = 0.01 (never fully deaf).

**In `apply_weight_update`**: The binary gating (`if receptors.contains(channel)`) is replaced with continuous multiplication by the sensitivity value. Falls back to binary when the map is empty (backward compatible).

```
reward_signal = sensitivity[Reward] * alpha_reward * reward_delta * endo_gain
```

**Adaptation** (slow tick): For each morphon, correlate recent modulation signals with PE deltas. If a channel's signal preceded PE reduction → increase sensitivity. If it preceded PE increase → decrease. Rate: 0.001 (very conservative — meaningful over thousands of episodes, not within a single run).

```rust
correlation = sum(mod_signal_i * (-pe_delta_i)) / n
sensitivity[channel] += 0.001 * correlation
```

**Lifecycle**: `divide()` inherits parent sensitivities ± 0.05 mutation. `differentiate()` and `dedifferentiate()` reset to defaults for the new type.

### Current Status

Validated as non-harmful: CartPole solves identically with adaptive receptors active. Sensitivities remain near 1.0 after 1000 episodes at 0.001 adaptation rate. **Not yet validated as beneficial** — would need a task where fixed receptor assignments are suboptimal (e.g., a morphon that needs reward sensitivity despite being Sensory type).

### Configuration

No dedicated config struct — sensitivity is always active. The adaptation rate (0.001) is hardcoded in `system.rs` slow tick. Future: make it configurable or Endo-controlled.

---

## 5. Collective Compute Scaling

**Module**: `morphogenesis.rs` (Cluster struct, fusion/defusion), `morphon.rs` (MetabolicConfig, step()), `system.rs` (cluster energy loop)

### Biological Basis

Cells in a tissue delegate housekeeping to the collective (shared metabolism, shared homeostasis) to focus on their specialty. A muscle fiber doesn't manage its own pH — the tissue does. The cluster is more efficient than the sum of its parts.

### How It Works

**Fusion** pools resources:
- Each member contributes 30% of energy to `shared_energy_pool`
- `shared_homeostatic_setpoint` = average of member setpoints

**During operation** (every step, in system.rs cluster loop):
1. Each fused morphon draws from the shared pool (`cluster_energy_draw_per_tick`)
2. Members contribute PE-reduction utility back to pool
3. Cluster-level homeostasis: single setpoint computation applied to all members' thresholds
4. Fused morphons pay reduced base cost (`base_cost * (1 - cluster_base_cost_reduction)`)
5. Fused morphons skip individual homeostatic threshold regulation (deferred to cluster)

**Defusion** returns energy: `shared_energy_pool / member_count` distributed equally.

### Current Status

Structurally complete. **Not exercised by CartPole** (fusion disabled). Would activate on tasks with `lifecycle.fusion = true` and sufficient correlated firing to trigger fusion. Needs a growing-network benchmark to validate the efficiency claim.

### Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `metabolic.cluster_base_cost_reduction` | 0.4 | Base cost reduction for fused morphons (40%) |
| `metabolic.cluster_energy_draw_per_tick` | 0.0003 | Energy drawn from pool per member per tick |

---

## 6. Dreaming Engine

**Module**: `types.rs` (DreamConfig), `system.rs` (dream_cycle, trigger_dream)

### Biological Basis

Sleep consolidation: the hippocampus replays recent experiences during sleep, strengthening successful patterns and pruning unused connections. The system "exists" between inputs — it processes and consolidates even when idle.

### How It Works

`dream_cycle()` runs three phases:

**Phase 1 — Consolidation**: Find synapses with `tag_strength > dream_tag_threshold` and `consolidation_level < 1.0`. Increase `consolidation_level` by `dream_lr * tag_strength * 0.1 * consolidation_gain`. Set `consolidated = true` when level > 0.5. Consume tags (tag *= 0.7). Also replay top-5 episodic memories with reduced reward injection.

**Phase 2 — Self-optimization**: Find synapses with `age > stale_synapse_age`, `usage_count < stale_usage_threshold`, `!consolidated`, and `weight.abs() < 0.1`. Reset weight to small pseudo-random value, reset age/usage/traces to zero. This gives dead connections a second chance without pruning them.

**Phase 3 — Curiosity signal**: For each cluster, compute internal PE variance and external connectivity ratio. If `pe_variance > 0.1` AND `external_ratio < 0.3` (high internal disagreement, isolated from the network), inject `curiosity_signal_strength` as input and mild novelty. This pushes the system to investigate and connect its anomalous regions.

### When It Runs

- **Automatic**: Glacial tick, when Endo stage is Mature/Consolidating OR firing rate < 5%
- **Manual**: `system.trigger_dream()` — e.g., between RL episodes or during task transitions

Both paths are gated on `recent_performance > consolidation_gate` (no consolidation of random weights). All consolidation is scaled by Endo's `consolidation_gain` channel.

### Current Status

**Mechanically functional. Not yet validated as beneficial.**

The three phases address real problems, but CartPole doesn't exercise them:
- Phase 1 is redundant with `report_episode_end` (both consolidate tagged synapses)
- Phase 2 has no stale synapses (fixed topology, no structural changes create dead connections)
- Phase 3 has no clusters (fusion disabled)

**Where dreaming would matter** (untested):
- **Continual learning**: train task A → `trigger_dream()` → train task B → measure task A retention
- **Growing networks**: with division/fusion, stale synapses and isolated clusters are real problems
- **Long-running autonomous systems**: consolidation during idle periods

### Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `dream.enabled` | true | Master switch |
| `dream.dream_learning_rate` | 0.3 | Consolidation rate multiplier (30% of waking) |
| `dream.dream_tag_threshold` | 0.2 | Minimum tag strength for consolidation candidates |
| `dream.max_dream_synapses` | 50 | Max synapses processed per cycle |
| `dream.stale_synapse_age` | 5000 | Age threshold for self-optimization |
| `dream.stale_usage_threshold` | 3 | Usage count below which a stale synapse resets |
| `dream.reset_weight_scale` | 0.1 | Weight magnitude for refreshed synapses |
| `dream.curiosity_signal_strength` | 0.15 | Novelty injection for anomalous clusters |

---

## Phase 2 Cross-Cutting Notes

### Interaction with Consolidation_Gain

All three consolidation paths (DFA capture, episode-end, dreaming) are scaled by Endo's `consolidation_gain` channel. This ensures stage-appropriate consolidation:

| Stage | cg | Effect |
|-------|-----|--------|
| Proliferating | 2.5 | Aggressive capture — nothing to protect |
| Differentiating | 2.0 | Still aggressive — refining features |
| Consolidating | 1.0 | Normal selectivity |
| Mature | 0.5 | Selective — protect learned weights |
| Stressed | 0.5 | Don't lock in bad patterns during crisis |

### Validation Status

| Primitive | Structurally Sound | Exercised by CartPole | Proven Beneficial |
|-----------|-------------------|----------------------|-------------------|
| Adaptive Receptors | Yes | Partially (active, neutral effect) | No |
| Collective Compute | Yes | No (fusion disabled) | No |
| Dreaming Engine | Yes | Partially (phases 2+3 are no-ops) | No |

Proving benefit requires benchmarks that exercise these features: continual learning (dreaming), growing networks (collective compute, curiosity), and tasks with suboptimal receptor assignments (adaptive receptors).
