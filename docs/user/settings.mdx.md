# MORPHON Settings Reference

All settings are fields of `SystemConfig`. Serialize to JSON for reproducibility (`config.save_json()`).

---

## Developmental

| Setting | Type | Default | Description |
|---|---|---|---|
| `developmental.program` | `Cortical\|Hippocampal\|Cerebellar\|Custom` | `Cortical` | Bootstrap program — determines initial cell type ratios and connectivity patterns |
| `developmental.seed_size` | `usize` | `100` | Number of initial morphons. Auto-scaled up if `target_input_size + target_output_size` requires more |
| `developmental.dimensions` | `usize` | `8` | Dimensionality of the hyperbolic information space |
| `developmental.initial_connectivity` | `f64` | `0.1` | Connection probability between seed morphons (largely superseded by I/O pathway phase) |
| `developmental.proliferation_rounds` | `usize` | `3` | Number of mitosis rounds during development. Each round: 30% of morphons divide |
| `developmental.target_input_size` | `Option<usize>` | `None` | If set, creates exactly this many Sensory morphons |
| `developmental.target_output_size` | `Option<usize>` | `None` | If set, creates exactly this many Motor morphons |

## Learning

| Setting | Type | Default | Description |
|---|---|---|---|
| `learning.tau_eligibility` | `f64` | `20.0` | Fast eligibility trace decay (timesteps). Lower = more reactive |
| `learning.tau_trace` | `f64` | `10.0` | Pre/post-synaptic trace window (STDP timing) |
| `learning.a_plus` | `f64` | `1.0` | LTP magnitude (pre-before-post) |
| `learning.a_minus` | `f64` | `-1.0` | LTD magnitude (post-before-pre). Less negative = weaker depression |
| `learning.tau_tag` | `f64` | `6000.0` | Slow synaptic tag decay (~minutes). Tags bridge the gap between activity and delayed reward |
| `learning.tag_threshold` | `f64` | `0.3` | Eligibility above which a synaptic tag is set |
| `learning.capture_threshold` | `f64` | `0.5` | Tag strength above which consolidation can occur |
| `learning.capture_rate` | `f64` | `0.1` | Learning rate for tag-and-capture consolidation |
| `learning.weight_max` | `f64` | `5.0` | Maximum absolute synapse weight |
| `learning.weight_min` | `f64` | `0.001` | Below this, synapses are pruning candidates |
| `learning.alpha_reward` | `f64` | `1.0` | Reward channel weight in the three-factor modulation signal. CartPole uses 2.0 |
| `learning.alpha_novelty` | `f64` | `0.5` | Novelty channel weight |
| `learning.alpha_arousal` | `f64` | `0.3` | Arousal channel weight |
| `learning.alpha_homeostasis` | `f64` | `0.1` | Homeostasis channel weight |
| `learning.transmitter_potentiation` | `f64` | `0.001` | Anti-silence floor: small positive dw when pre fires but post is chronically quiet |
| `learning.heterosynaptic_depression` | `f64` | `0.002` | Anti-runaway: depress all incoming weights when post fires |

## Readout

| Setting | Type | Default | Description |
|---|---|---|---|
| `readout_mode` | `Supervised\|TDOnly\|Hybrid` | `Supervised` | How the analog readout layer is trained (see below) |

### ReadoutTrainingMode

**`Supervised`** — The caller provides the correct output index each step. The readout learns a direct mapping from MI network activity to the correct action. Fastest convergence, but requires domain knowledge (e.g., "push in the direction the pole leans"). Biologically: cerebellar climbing-fiber supervised error signal.

**`TDOnly`** — Train from TD error signal only. Reinforces the chosen action proportional to TD error magnitude. More general (works without domain knowledge), but convergence is slower and less reliable. The system must discover the correct mapping from scalar reward alone.

**`Hybrid`** — Starts in Supervised mode. Transitions to TDOnly when `recent_performance > consolidation_gate`. This is a curriculum: bootstrap the readout with supervision to establish basic competence, then switch to autonomous refinement. Endoquilibrium can influence this transition in the future.

Query the active mode at runtime via `system.readout_training_mode()`.

## Morphogenesis

| Setting | Type | Default | Description |
|---|---|---|---|
| `morphogenesis.synaptogenesis_threshold` | `f64` | `0.6` | Activity correlation threshold for growing new connections |
| `morphogenesis.pruning_min_age` | `u64` | `100` | Minimum synapse age before pruning eligibility |
| `morphogenesis.division_threshold` | `f64` | `0.5` | Division pressure threshold for mitosis |
| `morphogenesis.division_min_energy` | `f64` | `0.3` | Minimum energy to allow division |
| `morphogenesis.fusion_correlation_threshold` | `f64` | `0.75` | Correlation threshold for cluster formation |
| `morphogenesis.fusion_min_size` | `usize` | `3` | Minimum group size for fusion |
| `morphogenesis.migration_rate` | `f64` | `0.05` | Step size for migration in hyperbolic space |
| `morphogenesis.apoptosis_min_age` | `u64` | `1000` | Minimum age before apoptosis eligibility |
| `morphogenesis.apoptosis_energy_threshold` | `f64` | `0.1` | Energy below which death is possible |
| `morphogenesis.max_morphons` | `Option<usize>` | `None` | Hard cap on morphon count. `None` = auto-derive (see below) |
| `morphogenesis.min_morphons` | `usize` | `10` | V3 Governor: apoptosis stops below this |
| `morphogenesis.min_sensory_fraction` | `f64` | `0.05` | V3 Governor: protect Sensory morphons |
| `morphogenesis.min_motor_fraction` | `f64` | `0.02` | V3 Governor: protect Motor morphons |
| `morphogenesis.max_single_type_fraction` | `f64` | `0.80` | V3 Governor: prevent any cell type from dominating |

### Sizing `max_morphons`

The morphon cap controls how large the system can grow. When set to `None` (the default), the cap is **auto-derived** from the I/O dimensions at `System::new()` time:

```
effective_max = max(500, (target_input_size + target_output_size) * 3)
```

This ensures the system always has enough headroom for associative (hidden) morphons on top of the I/O layer. For example, MNIST (784 inputs + 10 outputs) auto-derives to `max(500, 794 * 3) = 2382`.

If no `target_input_size` / `target_output_size` are set, the fallback is 500.

**Query the resolved value** at runtime via `system.max_morphons()` or `system.inspect().max_morphons`.

**Manual override:** Set `max_morphons: Some(n)` to use an explicit cap. This takes priority over auto-derivation.

| Task profile | I/O size | Auto-derived cap | Manual override examples |
|---|---|---|---|
| Simple control (few inputs, 2-3 actions) | ~10 | 500 | CartPole uses `Some(300)` |
| Multi-class classification, small input | ~20-70 | 500 | 3-class classifier uses `Some(60)` |
| High-dimensional input | 500+ | I/O × 3 | MNIST uses `Some(2000)`, or just `None` (auto = 2382) |

**Note:** `max_morphons` only matters when `lifecycle.division` is enabled. With frozen lifecycle (CartPole, MNIST supervised phase), population is fixed at bootstrap — determined by `seed_size` + `proliferation_rounds`, not the cap. The cap becomes relevant when division is re-enabled (e.g., MNIST v2 recovery phase).

**When to override with a lower value:**
- Fixed-topology runs where `lifecycle.division` is disabled — the cap is never reached anyway, a lower value saves memory.
- Real-time or WASM contexts where per-step latency matters. Spike propagation is O(k·N), so morphon count directly impacts fast-tick cost.

**When to override with a higher value:**
- The system consistently grows to the cap (check `system.inspect()`) — it needs more room.
- The task requires composing multi-level features (edges → shapes → objects), which demands deeper associative layers.
- Classification with many similar classes that need fine-grained discrimination.

**How to tell if your cap is wrong:** Run a training session and check `system.inspect().max_morphons` vs `system.inspect().total_morphons`. If total plateaus well below the cap, the system doesn't need more. If it's pinned at the cap and performance is still improving, raise it.

### Frustration-Driven Exploration (V2)

| Setting | Type | Default | Description |
|---|---|---|---|
| `morphogenesis.frustration.enabled` | `bool` | `true` | Enable frustration-driven stochastic exploration |
| `morphogenesis.frustration.stagnation_threshold` | `f64` | `0.005` | Minimum PE change to count as non-stagnant |
| `morphogenesis.frustration.saturation_steps` | `u32` | `200` | Stagnation counter at which frustration saturates |
| `morphogenesis.frustration.exploration_threshold` | `f64` | `0.3` | Frustration level above which exploration mode activates |
| `morphogenesis.frustration.max_noise_multiplier` | `f64` | `5.0` | Maximum noise amplitude when fully frustrated |
| `morphogenesis.frustration.weight_perturbation_scale` | `f64` | `0.01` | Weight perturbation amplitude (fraction of weight_max) |
| `morphogenesis.frustration.frustration_migration` | `bool` | `true` | Allow frustrated morphons to migrate without the desire gate |
| `morphogenesis.frustration.random_migration_threshold` | `f64` | `0.6` | Frustration level above which migration uses random direction |

## Lifecycle

| Setting | Type | Default | Description |
|---|---|---|---|
| `lifecycle.division` | `bool` | `true` | Allow cell division |
| `lifecycle.differentiation` | `bool` | `true` | Allow functional specialization changes |
| `lifecycle.fusion` | `bool` | `true` | Allow cluster formation |
| `lifecycle.apoptosis` | `bool` | `true` | Allow programmed cell death |
| `lifecycle.migration` | `bool` | `true` | Allow migration in information space |

For fixed-topology tasks (e.g., CartPole), disable division/fusion/apoptosis/migration to prevent structural changes during training.

## Metabolic Budget

| Setting | Type | Default | Description |
|---|---|---|---|
| `metabolic.base_cost` | `f64` | `0.001` | Energy cost per step for being alive |
| `metabolic.synapse_cost` | `f64` | `0.0001` | Additional cost per outgoing synapse per step |
| `metabolic.utility_reward` | `f64` | `0.02` | Energy earned per unit of PE reduction |
| `metabolic.basal_regen` | `f64` | `0.005` | Unconditional energy trickle (prevents starvation) |
| `metabolic.firing_cost` | `f64` | `0.002` | Extra cost when a morphon fires |
| `metabolic.cluster_overhead_per_tick` | `f64` | `0.0005` | Extra cost for being in a fused cluster |
| `metabolic.cluster_base_cost_reduction` | `f64` | `0.4` | V2: Base cost reduction for fused morphons (40% cheaper) |
| `metabolic.cluster_energy_draw_per_tick` | `f64` | `0.0003` | V2: Energy drawn from cluster pool per tick |

## Scheduler (Dual-Clock)

| Setting | Type | Default | Description |
|---|---|---|---|
| `scheduler.medium_period` | `u64` | `10` | Steps between medium-tick operations (eligibility, weight updates) |
| `scheduler.slow_period` | `u64` | `100` | Steps between slow-tick operations (synaptogenesis, pruning, migration) |
| `scheduler.glacial_period` | `u64` | `1000` | Steps between glacial-tick operations (division, differentiation, fusion, apoptosis) |
| `scheduler.homeostasis_period` | `u64` | `10` | Steps between homeostatic regulation |
| `scheduler.memory_period` | `u64` | `25` | Steps between memory operations (episodic replay, procedural recording) |

## Homeostasis

| Setting | Type | Default | Description |
|---|---|---|---|
| `homeostasis.kwta_fraction` | `f64` | `0.1` | Fraction of Associative morphons allowed to fire per step (k-WTA) |
| `homeostasis.inhibition_strength` | `f64` | `0.3` | Inter-cluster inhibition drive strength |
| `homeostasis.inhibition_correlation_threshold` | `f64` | `0.9` | Synchrony threshold for inter-cluster inhibition |
| `homeostasis.migration_cooldown_duration` | `f64` | `10.0` | Steps of migration cooldown after a move |

## Bioelectric Field (V2)

| Setting | Type | Default | Description |
|---|---|---|---|
| `field.enabled` | `bool` | `false` | Enable the spatial field system |
| `field.resolution` | `usize` | `32` | Grid resolution per axis (32x32 = 1024 cells) |
| `field.diffusion_rate` | `f64` | `0.1` | Fraction of Laplacian applied per slow tick |
| `field.decay_rate` | `f64` | `0.05` | Exponential decay rate per slow tick |
| `field.active_layers` | `Vec<FieldType>` | `[PE, Energy, Stress]` | Which field layers are active |
| `field.migration_field_weight` | `f64` | `0.3` | Weight of field gradient in migration direction (0=ignore, 1=field only) |

## Target Morphology (V2)

| Setting | Type | Default | Description |
|---|---|---|---|
| `target_morphology` | `Option<TargetMorphology>` | `None` | Functional region targets with self-healing |
| `target_morphology.self_healing` | `bool` | `true` | Recruit morphons to underpopulated regions |
| `target_morphology.healing_threshold` | `f64` | `0.5` | Deficit ratio below which healing triggers |

Use `TargetMorphology::cortical(dim)`, `::cerebellar(dim)`, or `::hippocampal(dim)` for presets.

## Dreaming Engine (V2)

| Setting | Type | Default | Description |
|---|---|---|---|
| `dream.enabled` | `bool` | `true` | Enable offline consolidation on glacial tick |
| `dream.dream_learning_rate` | `f64` | `0.3` | Learning rate multiplier during dream mode |
| `dream.dream_tag_threshold` | `f64` | `0.2` | Minimum tag strength for dream consolidation candidates |
| `dream.max_dream_synapses` | `usize` | `50` | Maximum synapses processed per dream cycle |
| `dream.stale_synapse_age` | `u64` | `5000` | Age threshold for self-optimization reset candidates |
| `dream.stale_usage_threshold` | `u64` | `3` | Usage count below which stale synapses are reset candidates |
| `dream.reset_weight_scale` | `f64` | `0.1` | Weight magnitude for refreshed stale synapses |
| `dream.curiosity_signal_strength` | `f64` | `0.15` | Novelty injection strength for topology anomalies |

Dream consolidation is gated on `recent_performance > consolidation_gate` — it won't lock in weights until the system has demonstrated competence.

## Endoquilibrium

| Setting | Type | Default | Description |
|---|---|---|---|
| `endoquilibrium.enabled` | `bool` | `false` | Enable predictive neuroendocrine regulation |
| `endoquilibrium.smoothing_alpha` | `f32` | `0.1` | EMA smoothing for channel gain updates |
| `endoquilibrium.fast_tau` | `f32` | `50.0` | Fast EMA time constant (acute changes) |
| `endoquilibrium.slow_tau` | `f32` | `500.0` | Slow EMA time constant (developmental trajectory) |

When enabled, Endoquilibrium monitors firing rates, eligibility density, weight entropy, tag-capture health, and energy pressure. It outputs six control channels that modulate the learning pipeline:

| Channel | Range | Effect |
|---|---|---|
| `reward_gain` | [0.1, 3.0] | Multiplies reward signal in three-factor learning |
| `novelty_gain` | [0.0, 2.0] | Multiplies novelty signal |
| `arousal_gain` | [0.1, 2.0] | Multiplies arousal signal |
| `homeostasis_gain` | [0.3, 2.0] | Multiplies homeostasis signal |
| `threshold_bias` | [-0.3, 0.3] | Added to morphon firing threshold |
| `plasticity_mult` | [0.1, 5.0] | Scales all weight update magnitudes |
| `consolidation_gain` | [0.2, 3.0] | Scales tag-and-capture consolidation rate. Biology: PRP availability |

### Consolidation Gain

`consolidation_gain` (cg) controls how aggressively tagged synapses get consolidated. It modulates three consolidation paths consistently:

- **DFA capture** (`train_readout`): effective capture threshold = `capture_threshold / cg`. Higher cg = lower bar for capture.
- **Episode-end consolidation** (`report_episode_end`): `delta_level *= cg`. Higher cg = faster consolidation_level increase.
- **Dream consolidation** (`dream_cycle`): `delta_level *= cg`. Same PRP model during offline replay.

Endo sets cg based on developmental stage (detected from reward trend):

| Stage | cg | plasticity_mult | Rationale |
|---|---|---|---|
| Proliferating | 2.5 | 1.5 | Learn fast, capture everything — nothing to protect |
| Differentiating | 2.0 | 1.2 | Refining, still capturing aggressively |
| Consolidating | 1.0 | 0.8 | Slowing down, normal selectivity |
| Mature | 0.5 | 0.5 | Protect what works — very selective |
| Stressed | 0.5 | 1.5 | Explore alternatives, but don't lock in bad patterns |

**Caution**: Endo amplifies learning signals based on vital signs. On tasks with fixed topology and already-tuned `alpha_*` values, this can cause over-amplification. If learning is unstable with Endo enabled, either reduce `alpha_reward` or disable Endo.

## Governance (V3)

| Setting | Type | Default | Description |
|---|---|---|---|
| `governance.max_connectivity_per_morphon` | `usize` | `50` | Maximum connections per morphon |
| `governance.max_cluster_size_fraction` | `f64` | `0.3` | Maximum cluster size as fraction of total morphons |
| `governance.max_unverified_fraction` | `f64` | `0.5` | Maximum fraction of unverified (Hypothesis) clusters |
| `governance.energy_floor` | `f64` | `0.0` | Minimum energy level enforced constitutionally |

---

## Morphon-Controlled Settings

Some settings can be influenced by the MI system itself at runtime:

| Setting | Controller | Mechanism |
|---|---|---|
| Channel gains (reward/novelty/arousal/homeostasis) | Endoquilibrium | Six regulation rules based on vital signs |
| Consolidation gain (capture aggressiveness) | Endoquilibrium | Stage-dependent: Proliferating=2.5, Mature=0.5 |
| `plasticity_mult` | Endoquilibrium | Scales all weight updates. Reduced under energy pressure |
| `threshold_bias` | Endoquilibrium | Raises/lowers global firing threshold |
| Receptor sensitivity per channel | Sub-morphon plasticity (V2) | Correlation between modulation and PE reduction |
| Noise amplitude | Frustration (V2) | Stagnation counter drives exploration noise |
| Migration direction | Bioelectric field (V2) | Field gradient steers migration |
| `readout_mode` (Hybrid) | Performance gate | Supervised → TDOnly at consolidation_gate |
| Consolidation | Performance gate + dreaming | Tags accumulate; capture gated on performance |

### Future: Endo-Controlled Readout Mode

In Hybrid mode, readout training transitions from Supervised to TDOnly based on the consolidation gate. A planned extension would let Endoquilibrium influence this transition by monitoring readout convergence metrics (weight stability, output discrimination) rather than relying on a fixed performance threshold.

### Future: Agency-Driven Supervision Requests

The V2 Agency spec describes a system that can request richer training signals when stuck. Combined with ReadoutTrainingMode, this would allow the system to:
- Detect learning stall (frustration + low readout convergence)
- Request transition back from TDOnly to Supervised ("I need help")
- Report confidence in its readout mapping

This is distinct from Endoquilibrium (which adjusts internal parameters) — it's the system acting on its environment to improve its own learning conditions.
