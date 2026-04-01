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
│  └─ morphons.par_iter_mut()   → integrate input, fire/not-fire, update state
│
├─ MEDIUM (every 10 steps)
│  ├─ update_eligibility()      → fast eligibility traces + slow synaptic tags
│  └─ apply_weight_update()     → three-factor rule + tag-and-capture
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

### Trace-Based STDP + Advantage Modulation

Learning uses trace-based STDP (Frémaux & Gerstner 2016) instead of binary coincidence detection. Each synapse carries `pre_trace` and `post_trace` — decaying memory of recent spikes (τ=10). When pre fires, `post_trace` determines LTD; when post fires, `pre_trace` determines LTP. This widens the effective STDP window from 1 timestep to ~10 steps, solving the co-firing problem caused by refractory periods and spike delays.

The reward channel uses **advantage modulation** (reward - baseline EMA) instead of raw reward. This eliminates the unsupervised bias where `mean(reward) × mean(eligibility)` causes systematic weight drift (Frémaux et al. 2010). Advantage is clamped to non-negative: absence of reward is sufficient signal, active depression from negative advantage kills activity in sparse-reward environments.

### Three-Factor Learning + Tag-and-Capture

Two-timescale learning: fast eligibility traces (tau=20) for immediate credit assignment, slow synaptic tags (tau=6000) for delayed reward. Tags are "captured" into permanent weight changes when strong reward arrives, solving the credit assignment problem without backpropagation. Consolidated synapses are protected from pruning.

### Homeostatic Protection

Four mechanisms prevent the "stable-dynamic paradox":
- **Synaptic Scaling**: proportional weight normalization preserving learned ratios
- **Inter-Cluster Inhibition**: prevents runaway synchronization between clusters
- **Migration Damping**: per-morphon cooldown + system-wide rate modifier
- **Checkpoint/Rollback**: local state snapshots around glacial changes with automatic revert

### I/O Pathway Guarantees

The developmental program (Phase 4) creates guaranteed feedforward connections: Sensory → Associative → Motor, plus direct Sensory → Motor shortcuts. This ensures signal flow from input to output regardless of random connectivity. Developmental differentiation sets `differentiation_level = 0.6` so I/O morphons resist dedifferentiation under stress.

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
