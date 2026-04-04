# Morphon-Core

**Morphogenic Intelligence Engine — adaptive AI systems that grow, learn, and self-organize at runtime**

Morphon-Core is a biological-inspired, adaptive intelligence engine that implements Morphogenic Intelligence: systems that grow, self-organize, and learn at runtime without backpropagation. 

## Key Features

- **Hyperbolic Geometry**: Morphons live in Poincaré ball space with learnable curvature per-point
- **No Backpropagation**: Credit assignment via eligibility traces + neuromodulatory broadcast + tag-and-capture
- **Multi-Temporal Processing**: Four temporal scales (fast/medium/slow/glacial) via dual-clock scheduler
- **Structural Plasticity**: Runtime synaptogenesis, pruning, migration, division, differentiation, fusion, apoptosis
- **Neuromodulation**: Four broadcast channels (Reward, Novelty, Arousal, Homeostasis)
- **Developmental Programs**: Bootstrap cortical/hippocampal/cerebellar architectures with guaranteed I/O pathways
- **Triple Memory System**: Working (persistent activity), episodic (one-shot), procedural (topology snapshots)
- **Bindings**: Python (via PyO3/maturin) and WebAssembly (via wasm-bindgen) support
- **Parallel Processing**: Rayon-based parallelization on fast path (feature-gated)

## Project Structure

```
.
├── src/                 # Library source code
│   ├── system.rs        # Top-level orchestrator
│   ├── morphon.rs       # Morphon and Synapse structs
│   ├── topology.rs      # Petgraph-backed directed graph
│   ├── learning.rs      # Three-factor learning rule
│   ├── resonance.rs     # Spike propagation with delays
│   ├── morphogenesis.rs # Structural plasticity operations
│   ├── neuromodulation.rs # Four broadcast channels
│   ├── developmental.rs # Bootstrap programs
│   ├── homeostasis.rs   # Stability mechanisms
│   ├── memory.rs        # Triple memory system
│   ├── diagnostics.rs   # Learning pipeline observability
│   ├── snapshot.rs      # System state serialization
│   ├── python.rs        # PyO3 bindings (feature: python)
│   └── wasm.rs          # WASM bindings (feature: wasm)
├── examples/            # Runnable examples (cartpole, anomaly, mnist)
├── benches/             # Criterion benchmarks
├── tests/               # Unit and integration tests
├── web/                 # Three.js web visualizer
├── data/                # Data directory (MNIST files for examples)
├── docs/                # Documentation and benchmark results
└── scripts/             # Utility scripts
```

## Build & Test Commands

```bash
# Build optimized
cargo build --release

# All tests (116: 97 unit + 18 integration + 1 doctest)
cargo test

# Single test
cargo test <name>

# Show stdout during tests
cargo test -- --nocapture

# Criterion benchmarks
cargo bench

# Examples with run profiles (quick is default)
cargo run --example cartpole --release              # quick
cargo run --example cartpole --release -- --standard
cargo run --example cartpole --release -- --extended
# Same for: anomaly, mnist (mnist requires ./data/ with MNIST files)

# Python bindings
maturin develop --features python

# WASM build + serve
wasm-pack build --target web --features wasm --no-default-features
cd web && python3 -m http.server 8080
```

## Architecture Overview

### Core Loop (`System::step()`)
Four temporal scales via dual-clock scheduler:

| Scale | Default Period | Operations |
|-------|---------------|------------|
| **Fast** | 1 | Spike propagation (resonance), morphon firing, input integration |
| **Medium** | 10 | Eligibility traces, three-factor weight updates, tag-and-capture |
| **Slow** | 100 | Synaptogenesis, pruning, migration in hyperbolic space |
| **Glacial** | 1000 | Division, differentiation, fusion, apoptosis (with checkpoint/rollback) |

Plus homeostasis (synaptic scaling, inter-cluster inhibition) and memory recording at their own periods.

### Key Systems

#### Endoquilibrium — Predictive Neuroendocrine Regulation Engine
Endoquilibrium maintains network health by sensing vital signs, predicting healthy state via dual-timescale EMAs, and adjusting neuromodulatory channels through proportional control. Biological analogy: the endocrine system (allostasis), not the nervous system (homeostasis).

Runs on the medium path tick. Never modifies weights or topology directly — it modulates the environment in which the Builder operates.

#### Epistemic Model — Four-State Knowledge Tracking with Scarring
Every cluster has an epistemic state that reflects the system's confidence in the knowledge encoded by that cluster's synaptic topology. State transitions are driven by justification records on member synapses.

Features **Epistemic Scarring**: clusters that have been burned (repeatedly Outdated or Contested) develop higher skepticism thresholds, requiring stronger evidence before accepting new beliefs.

States:
- **Supported**: Verified against current evidence — cluster is protected and stable
- **Outdated**: Evidence has gone stale — needs re-verification  
- **Contested**: Conflicting evidence from multiple pathways
- **Hypothesis**: Newly formed, not yet verified

#### Governance Layer — Constitutional Constraints
Hard invariants that lie **outside the learning loop** and cannot be modified by the system itself. Only a human oracle (or explicit API call) can amend them. Biological analogy: DNA-coded checkpoint programs that epigenetic modification cannot alter.

Enforced at every structural decision point (synaptogenesis, division, fusion, apoptosis) and override any learned behavior.

### How the Systems Work Together

Morphon-Core consists of several interconnected biological-inspired systems that operate on different temporal scales:

#### 1. Endoquilibrium — Predictive Neuroendocrine Regulation Engine
Maintains network health by sensing vital signs, predicting healthy state via dual-timescale EMAs, and adjusting neuromodulatory channels through proportional control. Biological analogy: the endocrine system (allostasis), not the nervous system (homeostasis).

Runs on the medium path tick. Never modifies weights or topology directly — it modulates the environment in which the Builder operates.

**Key Functions**:
- Senses vital signs (firing rates, eligibility density, weight entropy, etc.)
- Uses dual-timescale EMAs (fast τ=50, slow τ=500) to track acute changes vs developmental trajectory
- Applies 17 regulation rules to adjust 10 neuromodulatory channels (reward_gain, novelty_gain, etc.)
- Detects developmental stage (Proliferating, Differentiating, Consolidating, Mature, Stressed)
- Provides health score (0-1) indicating how well vitals match setpoints

#### 2. Epistemic Model — Four-State Knowledge Tracking with Scarring
Every cluster has an epistemic state reflecting confidence in knowledge encoded by synaptic topology. State transitions driven by justification records on member synapses.

Features **Epistemic Scarring**: clusters repeatedly Outdated or Contested develop higher skepticism thresholds, requiring stronger evidence for new beliefs.

**States**:
- **Supported**: Verified against current evidence — cluster protected and stable
- **Outdated**: Evidence gone stale — needs re-verification (>5000 steps without reinforcement)
- **Contested**: Conflicting evidence from multiple pathways (>25% minority evidence)
- **Hypothesis**: Newly formed, not yet verified

**Effects**:
- Hypothesis: Boost plasticity (1.5x) for member morphons
- Outdated: Unconsolidate stale synapses to allow relearning
- Contested: Increase arousal (1.2x) for members to drive re-evaluation
- Supported: Reward members with energy incentive for verified knowledge

#### 3. Governance Layer — Constitutional Constraints
Hard invariants **outside the learning loop** that cannot be modified by the system itself. Only a human oracle can amend them. Biological analogy: DNA-coded checkpoint programs epigenetic modification cannot alter.

Enforced at structural decision points (synaptogenesis, division, fusion, apoptosis) overriding learned behavior.

**Constraints**:
- Max connectivity per morphon (prevents superhub pathology)
- Max cluster size fraction (prevents domination by single cluster)
- Max unverified fraction (Hypothesis state limit)
- Mandatory justification for certain cell types (Motor by default)
- Cascade depth limit for invalidation traversal
- Max fusion rate per epoch
- Max structural changes per epoch (division+fusion+apoptosis)
- Energy floor (prevents total starvation)
- Max morphon population cap (auto-derived or explicit)

#### 4. Core Learning Systems
Operating on different temporal scales via dual-clock scheduler:

**Fast Path (every step)**:
- Spike propagation (resonance) - O(k·N) not O(N²)
- Morphon state updates (integrate input, fire/not-fire)
- Input integration and firing decisions

**Medium Path (every 10 steps)**:
- Eligibility traces (pre-synaptic × post-synaptic activity)
- Three-factor weight updates: Δw = eligibility × modulation
- Tag-and-capture for delayed reward (no backprop)
- Homeostatics: synaptic scaling, inter-cluster inhibition
- Endoquilibrium regulation
- Astrocytic gate updates
- Inhibitory spike-timing dependent plasticity (iSTDP)

**Slow Path (every 100 steps)**:
- Synaptogenesis: form new connections based on activity correlation
- Pruning: remove weak connections
- Migration in hyperbolic space (Poincaré ball)
- Structural plasticity preparation

**Glacial Path (every 1000 steps)**:
- Division: morphon splitting with inheritance
- Differentiation: change cell type based on activity
- Fusion: merge morphons into clusters
- Apoptosis: programmed death with checkpoint/rollback
- Epistemic state evaluation and effects
- Memory consolidation (working → episodic → procedural)

#### 5. Memory Systems
Triple memory system operating at different timescales:

**Working Memory**: Persistent activity patterns (fast timescale)
- Limited capacity (default: 7 items)
- Maintained through recurrent activity

**Episodic Memory**: One-shot learning of specific events (medium timescale)
- Larger capacity (default: 1000 items)
- Stores consolidated patterns for later replay

**Procedural Memory**: Topology snapshots representing learned procedures (slow timescale)
- Encodes structural knowledge
- Enables skill-like behavior

#### 6. Developmental Programs
Bootstrap architectures creating guaranteed I/O pathways:

**Cortical**: Six-layer structure for sensory processing
**Hippocampal**: Memory formation and spatial navigation
**Cerebellar**: Motor coordination and timing

Creates exact I/O matching via `target_input_size`/`target_output_size` parameters ensuring the system has precisely the required number of sensory and motor morphons for interfacing with external systems.

#### 7. Additional Systems
- **Neuromodulation**: Four broadcast channels (Reward/dopamine, Novelty/ACh, Arousal/noradrenaline, Homeostasis/serotonin)
- **Resonance Engine**: Spike propagation with delays using efficient O(k·N) algorithm
- **Homeostasis**: Stability mechanisms (synaptic scaling, inter-cluster inhibition, migration damping)
- **Diagnostics**: Learning pipeline observability (weight stats, eligibility, tags, firing rates)
- **Snapshot**: Full system state serialization to JSON
- **Field** (V2): Bioelektrisches Feld for indirect morphon communication
- **Justification**: Tracks reinforcement history on synapses for epistemic evaluation
- **Lineage**: Tracks morphon genealogy for inheritance during division

### Key Design Decisions
1. **Hyperbolic Geometry**: Morphons live in Poincaré ball. Origin = general/stem, boundary = specialized. Curvature is learnable per-point.
2. **No Backpropagation**: Credit assignment via eligibility traces + neuromodulatory broadcast + tag-and-capture consolidation.
3. **Contrastive Reward**: Targeted reward/inhibition at specific output ports breaks mode collapse in classification tasks.
4. **Parallelization**: Rayon on fast path (morphon updates, spike generation). Feature-gated behind `parallel`.

### Bindings
- **Python**: PyO3 bindings via `maturin develop --features python`
- **WASM**: wasm-bindgen powers the Three.js web visualizer in `web/`

## License

Apache-2.0

## Version

2.4.0 (see Cargo.toml)