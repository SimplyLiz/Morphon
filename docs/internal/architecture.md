# MORPHON Core Architecture

## Overview

`morphon-core` is a Rust library implementing the Morphogenic Intelligence (MI) engine. The system creates networks of autonomous compute units (Morphons) that grow, self-organize, and adapt their own architecture at runtime — without retraining, without fixed architecture, without cloud dependency.

## Module Map

```
morphon-core/
├── src/
│   ├── lib.rs              # Crate root, public re-exports
│   ├── types.rs            # Core types: CellType, ModulatorType, HyperbolicPoint, RingBuffer
│   ├── morphon.rs          # Morphon (compute unit) and Synapse structs
│   ├── topology.rs         # petgraph-backed dynamic connection graph
│   ├── neuromodulation.rs  # 4-channel broadcast (Reward/Novelty/Arousal/Homeostasis)
│   ├── learning.rs         # Three-factor learning + Tag-and-Capture
│   ├── resonance.rs        # Local spike propagation with delays (parallelized via rayon)
│   ├── morphogenesis.rs    # 7 structural change mechanisms
│   ├── memory.rs           # Triple memory (Working/Episodic/Procedural)
│   ├── homeostasis.rs      # Stability protections (scaling, inhibition, rollback)
│   ├── scheduler.rs        # Dual-clock architecture (fast/medium/slow/glacial)
│   ├── developmental.rs    # Bootstrapping programs with I/O pathway guarantees
│   ├── lineage.rs          # Lineage tree export for visualization
│   ├── diagnostics.rs      # Learning pipeline observability (weights, eligibility, firing, captures)
│   ├── snapshot.rs         # Serde serialization (save/load JSON)
│   ├── system.rs           # Top-level System orchestrating everything
│   ├── python.rs           # PyO3 bindings (behind `python` feature flag)
│   └── wasm.rs             # wasm-bindgen bindings (behind `wasm` feature flag)
├── tests/
│   └── integration_test.rs # 18 integration tests
├── examples/
│   ├── cartpole.rs         # CartPole RL control task
│   ├── anomaly.rs          # Sensor anomaly detection benchmark
│   ├── mnist.rs            # MNIST digit classification (full 784px)
│   └── classify_tiny.rs    # Minimal classification sanity check
├── benches/
│   └── benchmarks.rs       # Criterion benchmarks
├── pyproject.toml          # Maturin config for Python wheel builds
└── Cargo.toml              # Dependencies: petgraph, rayon, rand, serde, pyo3 (optional)
```

## Spontaneous Developmental Activity (Warm-Up)

On construction, `System::new()` runs 100 warm-up steps before the system is exposed to the caller. Analogous to retinal waves and cortical spontaneous bursting in utero:
- Random noise input is fed to sensory ports each step
- Neuromodulation (reward/novelty/arousal at 0.3) is injected each step
- All lifecycle events (division, differentiation, fusion, apoptosis, migration) are **disabled** during warm-up — only learning correlations, not structural changes
- After warm-up, modulation is reset to default, step counter is zeroed, and spike queue is cleared

## Data Flow Per Step

The simulation loop in `System::step()` follows the dual-clock architecture:

```
Step N
│
├─ FAST (every step) — parallelized via rayon
│  ├─ resonance.propagate()     → generate SpikeEvents from firing Morphons (par_iter)
│  ├─ resonance.deliver()       → deliver spikes that reached their target
│  ├─ compute degree_map        → pre-compute synapse count per morphon for metabolic cost
│  ├─ morphons.par_iter_mut()   → integrate input, fire/not-fire, metabolic budget, update state
│  ├─ k-WTA lateral inhibition  → top 5% associative morphons survive, rest suppressed + threshold boost
│  └─ DFA feedback injection    → project output error to associative layer via fixed random weights
│
├─ MEDIUM (every 10 steps)
│  ├─ update_eligibility()      → fast eligibility traces + slow synaptic tags
│  ├─ apply_weight_update()     → three-factor rule + tag-and-capture
│  ├─ DFA climbing-fiber rule   → Δw = pre_trace × feedback_signal × lr - L2 decay
│  └─ weight normalization      → per-neuron L1 norm on Associative incoming weights (Diehl & Cook 2015)
│
├─ SLOW (every 100 steps)
│  ├─ synaptogenesis()          → grow new connections between correlated pairs
│  ├─ pruning()                 → remove weak/unused synapses
│  ├─ migration()               → move Morphons in hyperbolic space (with damping)
│  └─ curvature_learning()      → adjust local curvature based on desire
│
├─ GLACIAL (every 1000 steps) — wrapped in checkpoint/rollback
│  ├─ create_checkpoint()       → snapshot local state
│  ├─ division()                → mitosis for overloaded Morphons
│  ├─ differentiation()         → stem cells specialize
│  ├─ transdifferentiation()    → direct A→B cell type conversion (chronic mismatch)
│  ├─ dedifferentiation()       → stressed cells return to flexibility
│  ├─ fusion()                  → correlated groups merge into clusters
│  ├─ defusion()                → stressed clusters break apart
│  ├─ apoptosis()               → remove useless Morphons
│  └─ should_rollback()         → revert if PE spiked
│
├─ HOMEOSTASIS (every 50 steps)
│  ├─ synaptic_scaling()        → normalize weights to preserve learned ratios
│  └─ inter_cluster_inhibition()→ prevent over-synchronization
│
├─ MEMORY (every 100 steps)
│  ├─ procedural.record()       → topology snapshot
│  └─ episodic.replay(3)        → reactivate high-value episodes + re-inject reward
│
└─ ALWAYS
   ├─ modulation.decay()        → decay all 4 neuromodulatory channels
   ├─ working_memory.step()     → decay active patterns
   ├─ auto-detect novelty       → high avg prediction error → inject novelty
   └─ encode episodes           → when novelty > 0.3, store firing patterns
```

## Key Design Decisions

### Hyperbolic Space (Poincare Ball)

All Morphon positions live in a hyperbolic space rather than Euclidean. Points near the origin are "general" (stem-like), points near the boundary are "specialized". The curvature parameter is learnable per-point — regions with high desire (chronic prediction error) get stronger curvature at runtime. Migration uses `log_map` to compute tangent gradients, then `exp_map` to project back onto the manifold.

### Dual-Clock Architecture

The scheduler separates processes into four temporal scales to prevent structural changes from destabilizing fast inference. All periods are configurable via `SchedulerConfig`.

### Weight-Dependent STDP + Advantage Modulation

Learning uses trace-based STDP (Frémaux & Gerstner 2016) with **multiplicative (weight-dependent) bounds** (Gilson & Fukai 2011). Each synapse carries `pre_trace` and `post_trace` — decaying memory of recent spikes (τ=10). When pre fires, `post_trace` determines LTD; when post fires, `pre_trace` determines LTP. This widens the effective STDP window from 1 timestep to ~10 steps.

Weight-dependent scaling prevents bimodal weight collapse:
- **LTP** scales as `(w_max - w) / w_max` — easy to strengthen weak synapses, hard to over-strengthen strong ones
- **LTD** scales as `(w + w_max) / (2 * w_max)` — easy to weaken strong synapses, protects weak ones
- Produces stable long-tail weight distributions instead of all-or-nothing

**Motor readout**: Motor morphons use centered sigmoid-normalized membrane potential as `post_activity`: `(sigmoid(potential) - 0.5) * 2 ∈ [-1, 1]`. Only above-average potential generates positive post_trace, providing input discrimination. Non-spiking readout matches the DSQN/SpikeGym pattern.

The reward channel uses **advantage modulation** (reward - baseline EMA) instead of raw reward. This eliminates the unsupervised bias where `mean(reward) × mean(eligibility)` causes systematic weight drift (Frémaux et al. 2010). Advantage is clamped to non-negative: absence of reward is sufficient signal, active depression from negative advantage kills activity in sparse-reward environments.

### Three-Factor Learning + Tag-and-Capture

Two-timescale learning: fast eligibility traces (tau=20) for immediate credit assignment, slow synaptic tags (tau=6000) for delayed reward. Tags are "captured" into permanent weight changes when strong reward arrives, solving the credit assignment problem without backpropagation. Consolidated synapses are protected from pruning.

Tag-and-capture fires in two locations:
1. **DFA path** (medium learning): when `|pre_trace × feedback_signal| > 0.1`, tags the synapse; captures on strong reward (> 0.3).
2. **Readout training path**: during classification error backprop on sensory→associative synapses. Tags accumulate aggressively (`tag += fb.abs() * 0.5`), capture threshold lowered to 0.1. This is where most consolidations occur because fresh error signals are available.

### Direct Feedback Alignment (DFA)

Hidden layer credit assignment without backpropagation (Lillicrap et al. 2016). Each associative morphon gets a fixed set of random weights projecting from all motor morphons. These weights are initialized once during `System::new()` and **never updated**.

When output error occurs, it's projected backward: `feedback = Σ(fixed_weight_j × error_j)`. This signal is injected into associative morphons as both input current (`input_accumulator += 0.5 × feedback`) and a stored modulation signal (`feedback_signal`).

**Climbing-fiber rule**: The DFA learning update on incoming synapses uses `Δw = pre_trace × feedback_signal × 0.02 - 0.001 × w` (L2 decay). The key choice is `pre_trace` over binary `pre_fired` (too sparse at 5% firing rate) or `eligibility` (STDP-gated, attenuates signal). `pre_trace` is the Goldilocks gate — recent firing memory that persists ~10 steps.

### Analog Readout Bypass (Purkinje-style)

A parallel output pathway that bypasses spike propagation entirely. Enabled via `enable_analog_readout()` for classification tasks.

`readout_weights[output_j][assoc_i]` maps each associative morphon's potential to each output. The motor output becomes: `V_j = Σ_i readout_weights[j][i] × sigmoid(potential_i)`. Sensory morphons are also included for fast direct shortcuts.

Weights are initialized with Xavier scaling (`1/√n_assoc`) and trained with the **delta rule**: `Δw = lr × sigmoid(P_i) × (target_j - output_j) - L2_decay`. Targets are one-hot encoded (1.0 for correct class, 0.0 otherwise).

Output errors are backprojected through the same fixed DFA weights (not the readout weights) as "dendritic injection" to the hidden layer. This creates a loop: readout trains the output, DFA feedback trains the hidden layer.

### k-Winner-Take-All Lateral Inhibition

Associative (and Stem) morphons compete via k-WTA each fast step (Diehl & Cook 2015). Only the top k most active morphons survive firing; the rest are suppressed.

- **k = 5% of population** (minimum 3) — allows sparse distributed representations
- **Suppression is mild**: non-winners have `fired` reset and potential clamped to `threshold × 0.5`, but potential isn't zeroed — preserving information for the next step
- **Adaptive threshold boost**: winners get `threshold += 0.02` each time they win. This prevents any single neuron from dominating all inputs. Homeostatic threshold regulation (in `Morphon::step()`) decays threshold back over time.

Combined with weight normalization, this forces different associative morphons to specialize on different input patterns.

### Per-Neuron Weight Normalization

L1 normalization on incoming positive weights for Associative/Stem morphons, run every medium tick. Target norm = `n_incoming × 0.3`. Strengthening some inputs forces weakening of others — creates synaptic competition that produces specialized feature detectors.

Only positive weights are normalized (inhibitory connections are left untouched). This preserves the sign structure of the network while ensuring excitatory drive stays bounded.

### Motor Drift Prevention

Motor morphons have three special treatments to prevent potential saturation:
1. **Full leak**: `leak_rate = 1.0` (vs 0.1 for other types) — potential resets each step, reflecting only current input with no accumulation.
2. **Zero noise**: `noise_scale = 0.0` (vs 0.1) — prevents noise accumulation that drives saturation over hundreds of steps.
3. **Potential clamp**: `[-10, 10]` — prevents overflow from dense connectivity (300+ simultaneous inputs).

This makes motor morphons memoryless frame-to-frame — they track current input without historical drift.

### Sparse Zero-Bias Input Encoding

External inputs are fed through `feed_input()` as raw values with no bias transformation. For tasks like CartPole, each observation dimension is split into positive/negative channels (e.g., `max(0, velocity)` and `max(0, -velocity)`), giving 8 inputs for 4 observations.

Previous approach used `sigmoid(input + 2.0)` which squashed dynamic range to 0.11 (sigmoid(2) ≈ 0.88). Zero-bias gives 0.49 range — a 4.5× improvement. Homeostatic threshold regulation handles firing rate instead of input bias.

### V3 Constitutional Guards

Apoptosis is gated by cell-type diversity constraints in `MorphogenesisParams`:
- `min_morphons` (10) — apoptosis stops below this total count
- `min_sensory_fraction` (0.05) — at least 5% must be Sensory
- `min_motor_fraction` (0.02) — at least 2% must be Motor
- `max_single_type_fraction` (0.80) — no single type exceeds 80%

These prevent I/O starvation, output death, modulatory explosion, and total system collapse.

### Homeostatic Protection

Four mechanisms prevent the "stable-dynamic paradox":
- **Synaptic Scaling**: proportional weight normalization preserving learned ratios
- **Inter-Cluster Inhibition**: prevents runaway synchronization between clusters
- **Migration Damping**: per-morphon cooldown + system-wide rate modifier
- **Checkpoint/Rollback**: local state snapshots around glacial changes with automatic revert

### I/O Pathway Guarantees

The developmental program (Phase 4) creates guaranteed feedforward connections: Sensory → Associative → Motor, plus direct Sensory → Motor shortcuts. This ensures signal flow from input to output regardless of random connectivity. Developmental differentiation sets `differentiation_level = 0.6` so I/O morphons resist dedifferentiation under stress.

Feedforward Sensory→Associative connections use **excitatory-only initialization** (weights in [0.3, 0.8]) instead of Xavier-style mixed signs. Mixed-sign feedforward creates dead pathways that never activate — signal propagation needs positive weights above the firing threshold.

### V3 Metabolic Budget

Morphon energy is governed by a metabolic budget system (`MetabolicConfig` in `morphon.rs`) rather than flat regeneration. The core principle: **energy is earned through utility, not given unconditionally**.

Each step, a morphon pays:
- `base_cost` (0.001) — the cost of being alive
- `synapse_cost × degree` (0.0001 per connection) — maintaining connections is expensive
- `firing_cost` (0.002) — extra cost when the morphon spikes

And earns:
- `utility_reward × PE_reduction` (0.02 per unit) — reducing prediction error earns energy
- `basal_regen` (0.003) — a small unconditional trickle prevents total starvation of quiet morphons

**Economics of a typical morphon** (10 synapses):
- Outflow: `0.001 + 10×0.0001 = 0.002/step` (plus 0.002 per spike)
- Inflow: `0.003 + utility`
- Net: quiet morphons with few connections break even on basal regen; heavily connected or frequently firing morphons need PE reduction to stay solvent

This creates selection pressure: morphons that don't contribute to reducing prediction error drain to zero energy and become apoptosis candidates (energy < 0.1). Morphons with many connections pay proportionally more, preventing topological bloat. The result is a system that self-organizes toward minimal topology at maximal performance — ideal for edge hardware.

The `MetabolicConfig` is part of `SystemConfig` and all parameters are tunable. The degree map is pre-computed per step and passed into `Morphon::step()` to avoid topology access during parallel iteration.

### Parallelization

Rayon is used for:
- Morphon state updates (`par_iter_mut` on the HashMap)
- Spike generation in resonance (`par_iter` over firing morphons)
- Curvature learning (`par_iter_mut`)

### Contrastive Reward (Two-Hop Credit Assignment)

`reward_contrastive(correct_index, reward_strength, inhibit_strength)` provides output-specific credit assignment for classification tasks:
- **Correct output**: `inject_reward_at()` boosts eligibility on the motor morphon's incoming synapses (hop 0), then propagates backward to the associative layer (hop 1, attenuated by 0.5 × |weight|). A weak global reward (10%) covers deeper paths.
- **Incorrect outputs**: `inject_inhibition_at()` decays eligibility toward zero (doesn't drive negative). Two-hop backward propagation with the same structure.

This is the key mechanism for breaking mode collapse in classification tasks — without it, all motor morphons converge to the same output.

### Stable I/O Ports

The System maintains sorted `input_ports` and `output_ports` vectors that map external indices to specific Morphon IDs. `feed_input()` fans out each input to multiple sensory morphons. These port mappings survive serialization.
