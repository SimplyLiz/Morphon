# Concept-to-Code Mapping

How the concept documents (MORPHON-product-concept.md, morphogenic-intelligence-concept.md) map to the implementation.

## Section 3.2: The Morphon

| Concept Field | Implemented In | Status |
|---------------|---------------|--------|
| `id: UniqueID` | `morphon.rs: Morphon.id` | Done |
| `position: Vector[N]` | `types.rs: HyperbolicPoint` | Done (upgraded to hyperbolic) |
| `lineage: LineageID` | `morphon.rs: Morphon.lineage` | Done |
| `generation: Int` | `morphon.rs: Morphon.generation` | Done |
| `cell_type: CellType` | `morphon.rs: Morphon.cell_type` | Done |
| `differentiation_level: Float` | `morphon.rs: Morphon.differentiation_level` | Done |
| `activation_function: Fn` | `morphon.rs: Morphon.activation_fn` | Done (5 variants) |
| `receptors: Set<ModulatorType>` | `morphon.rs: Morphon.receptors` | Done |
| `potential: Float` | `morphon.rs: Morphon.potential` | Done |
| `threshold: Float` | `morphon.rs: Morphon.threshold` | Done (homeostatically regulated) |
| `refractory_timer: Float` | `morphon.rs: Morphon.refractory_timer` | Done |
| `prediction_error: Float` | `morphon.rs: Morphon.prediction_error` | Done |
| `desire: Float` | `morphon.rs: Morphon.desire` | Done (EMA of PE) |
| `incoming/outgoing` | `topology.rs: Topology` | Done (petgraph DiGraph) |
| `eligibility_traces` | `morphon.rs: Synapse.eligibility` | Done |
| `pre_trace` / `post_trace` | `morphon.rs: Synapse.pre_trace/post_trace` | Done (STDP spike traces, τ=10) |
| `activity_history: RingBuffer` | `morphon.rs: Morphon.activity_history` | Done (100-step window) |
| `age: Int` | `morphon.rs: Morphon.age` | Done |
| `energy: Float` | `morphon.rs: Morphon.energy` | Done |
| `division_pressure: Float` | `morphon.rs: Morphon.division_pressure` | Done |
| `fused_with: Option<ClusterID>` | `morphon.rs: Morphon.fused_with` | Done |
| `autonomy: Float` | `morphon.rs: Morphon.autonomy` | Done |

## Synapse (concept Section 3.2)

| Concept Field | Implemented In | Status |
|---------------|---------------|--------|
| `weight: Float` | `morphon.rs: Synapse.weight` | Done |
| `delay: Float` | `morphon.rs: Synapse.delay` | Done |
| `eligibility: Float` | `morphon.rs: Synapse.eligibility` | Done (fast, τ=20) |
| `age: Int` | `morphon.rs: Synapse.age` | Done |
| `usage_count: Int` | `morphon.rs: Synapse.usage_count` | Done |
| `tag: Float` (3.7C) | `morphon.rs: Synapse.tag` | Done (slow, τ=6000) |
| `tag_strength: Float` (3.7C) | `morphon.rs: Synapse.tag_strength` | Done |
| `consolidated: Bool` (3.7C) | `morphon.rs: Synapse.consolidated` | Done |

## Section 3.3: Resonance

| Concept | Implemented In | Status |
|---------|---------------|--------|
| Local signal propagation | `resonance.rs: ResonanceEngine` | Done |
| Weighted + delayed signals | `SpikeEvent.strength` / `.delay` | Done |
| O(k*N) complexity | Topology-based propagation | Done |
| Selective long-range connections | Emerges from synaptogenesis | Done |

## Section 3.4: Morphogenesis (7 Mechanisms)

| Mechanism | Concept Section | Implemented In | Status |
|-----------|----------------|---------------|--------|
| A) Synaptic Plasticity | 3.4A | `learning.rs` | Done (trace-based STDP + three-factor rule + advantage modulation) |
| B) Synaptogenesis/Pruning | 3.4B | `morphogenesis.rs: synaptogenesis(), pruning()` | Done |
| C) Cell Division (Mitosis) | 3.4C | `morphogenesis.rs: division()`, `morphon.rs: divide()` | Done (inheritance + mutation + lineage tracking) |
| D) Differentiation | 3.4D | `morphogenesis.rs: differentiation(), dedifferentiation()` | Done (diff, dediff, transdiff) |
| E) Fusion / Autonomy Loss | 3.4E | `morphogenesis.rs: fusion(), defusion()` | Done (Cluster struct) |
| F) Migration | 3.4F | `morphogenesis.rs: migration()` | Done (hyperbolic log/exp maps + damping) |
| G) Apoptosis | 3.4G | `morphogenesis.rs: apoptosis()` | Done |

## Section 3.5: Neuromodulation

| Concept | Implemented In | Status |
|---------|---------------|--------|
| Reward (Dopamine) | `neuromodulation.rs: reward` | Done |
| Novelty (Acetylcholine) | `neuromodulation.rs: novelty` | Done |
| Arousal (Noradrenaline) | `neuromodulation.rs: arousal` | Done |
| Homeostasis (Serotonin) | `neuromodulation.rs: homeostasis` | Done |
| Formula: ẇ = e * (αr*R + αn*N + αa*A + αh*H) | `learning.rs: apply_weight_update()` | Done (receptor-gated, reward uses advantage) |
| Reward advantage (reward - baseline) | `neuromodulation.rs: reward_advantage()` | Done (EMA baseline, clamped ≥ 0) |
| Global broadcast | Single struct, all Morphons read | Done |

## Section 3.6: Memory Architecture

| System | Concept | Implemented In | Status |
|--------|---------|---------------|--------|
| Working Memory | Persistent activity patterns (attractors) | `memory.rs: WorkingMemory` | Done (capacity 7, decay 0.05) |
| Episodic Memory | Fast synaptic changes, consolidation via replay | `memory.rs: EpisodicMemory` | Done (encode + replay) |
| Procedural Memory | Topology IS memory | `memory.rs: ProceduralMemory` | Done (topology snapshots) |

## Section 3.7: Homeostatic Protection (NEW)

| Mechanism | Concept Section | Implemented In | Status |
|-----------|----------------|---------------|--------|
| A) Synaptic Scaling | 3.7A | `homeostasis.rs: synaptic_scaling()` | Done |
| B) Inhibitory Inter-Cluster | 3.7B | `homeostasis.rs: inter_cluster_inhibition()` | Done |
| C) Tag-and-Capture | 3.7C | `learning.rs: update_eligibility(), apply_weight_update()` | Done |
| D) Migration Damping | 3.7D | `homeostasis.rs: can_migrate(), migration_rate_modifier()` | Done |
| E) Checkpoint/Rollback | 3.7E | `homeostasis.rs: create_checkpoint(), should_rollback(), rollback_synapses()` | Done (API ready, not yet auto-triggered in system loop) |

## Section 3.8: Dual-Clock Architecture (NEW)

| Path | Concept Period | Implemented Period | Status |
|------|---------------|-------------------|--------|
| Fast | μs-ms (continuous) | Every step | Done |
| Medium | ms-s (~10ms) | Every 10 steps | Done |
| Slow | s-min (~100ms-1s) | Every 100 steps | Done |
| Glacial | min-h (~10s-60s) | Every 1000 steps | Done |
| Homeostasis | — | Every 50 steps | Done |

## Section 3.9: Hyperbolic Information Space (NEW)

| Concept | Implemented In | Status |
|---------|---------------|--------|
| Poincaré Ball model | `types.rs: HyperbolicPoint` | Done |
| Hyperbolic distance | `HyperbolicPoint::distance()` | Done |
| Exponential map (migration) | `HyperbolicPoint::exp_map()` | Done |
| Logarithmic map (gradients) | `HyperbolicPoint::log_map()` | Done |
| Möbius addition | `HyperbolicPoint::mobius_add()` | Done |
| Learnable curvature | `HyperbolicPoint.curvature` | Done (field exists, learned at runtime in slow path) |
| Specificity = distance from origin | `HyperbolicPoint::specificity()` | Done |

## Section 4.1: Developmental Programs

| Phase | Implemented In | Status |
|-------|---------------|--------|
| Seed phase | `developmental.rs: develop()` phase 1 | Done |
| Proliferation | `developmental.rs: develop()` phase 2 | Done (3 rounds, 30% division prob) |
| Differentiation | `developmental.rs: develop()` phase 3 | Done (positional gradient) |
| Pruning | `developmental.rs: develop()` phase 4 | Done (|w| < 0.05 removed) |
| Presets: cortical, hippocampal, cerebellar | `DevelopmentalConfig::cortical()` etc. | Done |

## SDK API (Product Concept Section 2.2)

| Python API | Rust Equivalent | Status |
|-----------|----------------|--------|
| `morphon.System(...)` | `System::new(SystemConfig)` | Done |
| `system.develop(...)` | Runs automatically in `System::new()` | Done |
| `system.process(input)` | `System::process(&[f64])` | Done |
| `system.inject_reward(0.8)` | `System::inject_reward(0.8)` | Done |
| `system.inject_novelty(0.6)` | `System::inject_novelty(0.6)` | Done |
| `system.inject_arousal(0.9)` | `System::inject_arousal(0.9)` | Done |
| `system.inspect()` | `System::inspect() -> SystemStats` | Done |
| `stats.total_morphons` | `SystemStats.total_morphons` | Done |
| `stats.fused_clusters` | `SystemStats.fused_clusters` | Done |
| `stats.differentiation_map` | `SystemStats.differentiation_map` | Done |
| `stats.max_generation` | `SystemStats.max_generation` | Done |

## Recently Completed

| Feature | Concept Section | Implemented In |
|---------|----------------|---------------|
| Checkpoint auto-trigger in system loop | 3.7E | `system.rs`: glacial path wraps changes in checkpoint/rollback |
| Runtime curvature learning | 3.9 | `system.rs`: slow path adjusts curvature based on desire |
| Episodic memory replay in step() | 3.6 | `system.rs`: memory path replays 3 top episodes, re-injects reward context |
| Serialization (checkpoint export/import) | Product 2.2 | `snapshot.rs`: `save_json()` / `load_json()` roundtrip |
| CartPole benchmark | 6.2 | `examples/cartpole.rs`: full environment + MI agent + exploration |
| Python bindings (PyO3) | Product 2.2 | `python.rs`: `morphon.System` class via maturin |
| Rayon parallelization | 6.1 | `system.rs` + `resonance.rs`: par_iter_mut for morphon steps + spike gen |
| Lineage tree export | Visualization | `lineage.rs`: LineageTree with JSON export, root/child/depth queries |
| Stable I/O port mapping | Product 2.2 | `system.rs`: sorted input_ports/output_ports with fan-out routing |
| I/O pathway guarantees | Developmental | `developmental.rs`: Phase 4 creates Sensory→Assoc→Motor + shortcuts |
| Inhibitory inter-cluster Morphons | 3.7B | `morphogenesis.rs`: Cluster.inhibitory_morphons + auto-generation |
| Learning diagnostics | Observability | `diagnostics.rs`: weight/eligibility stats, firing by type, capture events |
| Spike-timing eligibility | 3.5 | `system.rs`: eligibility updated at spike delivery time (STDP-like precision) |
| Contrastive reward API | Credit assignment | `system.rs`: `reward_contrastive()`, `inject_reward_at()`, `inject_inhibition_at()` |
| Anomaly detection benchmark | 6.2 | `examples/anomaly.rs`: sensor anomaly detection demo |
| MNIST full 784px | 6.2 | `examples/mnist.rs`: full resolution, no downsampling, auto-scaled seed |
| WASM runtime | Product 2.3 | `src/wasm.rs` + `web/index.html`. 350KB binary, Poincare disk viz. |
| Trace-based STDP | 3.4A | `learning.rs`: pre/post traces (τ=10), widens STDP window from 1 to ~10 steps (Frémaux & Gerstner 2016) |
| Advantage modulation | 3.5 | `neuromodulation.rs`: reward_advantage() = (reward - baseline EMA).max(0). Eliminates unsupervised weight drift |
| Spontaneous developmental activity | 4.1 | `system.rs`: 100-step warm-up with noise input + modulation, lifecycle disabled |
| Growth cap enforcement | 3.4C | `morphogenesis.rs`: hard cap on morphon count respected during division |
| Lifecycle disable during warm-up | 4.1 | `system.rs`: structural plasticity disabled during spontaneous activity phase |
| Contrastive reward (two-hop) | Credit assignment | `system.rs`: inject_reward_at() with backward hop to associative layer |
| Total born/died tracking | Observability | `system.rs`: SystemStats.total_born/total_died |
| classify_tiny example | 6.2 | `examples/classify_tiny.rs`: minimal classification sanity check |
| Run profiles | Usability | Examples support --quick/--standard/--extended CLI flags |
| Weight-dependent STDP | 3.4A | `learning.rs`: multiplicative LTP/LTD scaling (Gilson & Fukai 2011). Prevents bimodal weight collapse |
| Centered motor readout | 3.4A | `system.rs`: (sigmoid(potential) - 0.5) * 2 ∈ [-1,1] for input discrimination |
| Potential-based post_activity | 3.4A | `system.rs`/`learning.rs`: Motor morphons use graded potential, not binary firing |
| Excitatory-only feedforward init | 4.1 | `developmental.rs`: Sensory→Associative weights [0.3, 0.8], not mixed-sign Xavier |

## What's Not Yet Implemented

| Feature | Concept Section | Priority | Notes |
|---------|----------------|----------|-------|
| Eligibility propagation (multi-hop credit) | Credit assignment | High | Contrastive signal only reaches direct motor synapses (~5 each). Need backward propagation of eligibility boost through Assoc layer. |
| CartPole convergence | 6.2 | High | best=65, avg~16. Global reward + tag-and-capture. Needs eligibility propagation or temporal difference. |
| MNIST convergence | 6.2 | High | Contrastive breaks mode collapse (3 classes recognized vs 1). Test acc ~9%. Needs deeper credit. |
| GPU acceleration | Product 2.3 | Low | Phase 3 |
| Curriculum learning / physics simulator integration | 4.2 | Medium | Requires external environment interface |
| Population-based evolutionary meta-learning | 4.3 | Medium | CMA-ES/PBT over learning params. Would accelerate convergence research. |
