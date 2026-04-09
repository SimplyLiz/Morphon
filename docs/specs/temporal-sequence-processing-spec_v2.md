# Temporal Sequence Processing
## First Capability Extension for MORPHON
### Technical Specification v3.0 — TasteHub GmbH, April 2026

---

| | |
|---|---|
| **Status** | Design (simplified from v2) |
| **Author** | Lisa + Claude |
| **Depends on** | Prerequisite 1: CartPole v2 solved (avg ≥ 195), Prerequisite 2: MNIST accuracy ≥ 85% |
| **Interacts with** | Pulse Kernel Lite (hot arrays), Endoquilibrium V2 (trace regulation), ANCS-Core (episodic replay) |
| **Supersedes** | v2.0 (April 2026) — simplified architecture, removed redundant components |

---

## 0. Research Findings & Dead Paths

### 0.1 Dead Path: Hardwired XOR Circuits

**Status:** Investigated, tested extensively, abandoned.

We attempted to solve the NLP Tier 3 compositionality problem (XOR-like token combination) by hardwiring a dedicated XOR circuit with pre-tuned weights into the main topology. After extensive testing:

- **Consistently failed at ~50% accuracy** (chance level)
- **Root cause:** Circuit morphons added to main topology receive signals from ALL ~46 firing morphons. The intended +2.0/-2.0 XOR signals were completely drowned out by cumulative noise.
- **Debug evidence:** Both cores clamped at -10.0 for ALL patterns — they never fired.
- **Research confirmation:** Moser & Lunglmayr (2024) show SNNs solve XOR with temporal encoding (spike frequencies), not static voltage levels. Even with optimal conditions, only 3-7% of random weight initializations succeed.

**What was tried:**
1. Pre-wired XOR circuit with fixed weights (+2.0/-2.0)
2. Multiple weight configurations and threshold values
3. With and without interneuron
4. Pretrained vs learned weights
5. ID collision fix (critical bug: circuit morphon IDs started at 1, overwriting existing morphons)

**Lesson:** Compositionality cannot be hardwired into a spiking network that shares topology with other processing. It must emerge from the network's dynamics through training.

**The `xor_circuit.rs` module exists but is non-functional.** It should be either removed entirely or repurposed as part of the DeMorphon lifecycle (where internal wiring IS isolated from the main topology).

### 0.2 What Research Shows About SNNs and Temporal Processing

**Echo State Networks / Liquid State Machines** (Gaurav et al., 2023; Soures & Kudithipudi, 2019): Random recurrent connectivity + simple linear readout solves time series classification. The reservoir's state IS the working memory. No separate context feedback needed for sequences of length ≤ 15.

**Nam et al. (2023 — arXiv:2310.01807):** Discrete, compositional representations emerge from attractor dynamics without explicit symbolic structure. Input → dynamics evolve state → trajectory converges to attractor basin. Combined inputs converge to attractors that are NOT simple averages of individual attractors.

**NeuronSpark (Tang, 2026 — arXiv:2603.16148):** LIF neuron dynamics are structurally identical to Mamba's selective state space model. SNNs CAN learn language from scratch, but at enormous scale with backprop.

**Key insight for Morphons:** The recurrent reservoir's state trajectory IS the temporal representation. No separate context encoding, no hash-based projection, no dedicated context ports. The reservoir does all the work.

### 0.3 Temporal Capability Assessment

| Component | Status | Notes |
|-----------|--------|-------|
| Dual-clock scheduler | ✅ Fully implemented | Excellent |
| Hyperbolic geometry | ✅ Fully implemented | Natural hierarchy |
| Neuromodulation (4 channels) | ✅ Fully implemented | Reward/novelty/arousal |
| Analog readout (softmax + DFA) | ✅ Fully implemented | Classification works |
| Working memory (capacity 7) | ✅ Functional | Short sequences |
| Three-factor learning | ✅ Stable mechanisms | Credit assignment |
| **Recurrent connections** | ❌ Missing | Critical gap |
| **Temporal encoding** | ❌ Missing | Batch, not sequential |

**Bottom line:** We need recurrent connections + sequential input + longer eligibility traces. That's it. The rest is scaffolding we don't need yet.

### 0.4 Compositional Sequence Processing

The compositional task: given two tokens presented sequentially, are they same-group (VV/CC) or different-group (VC/CV)?

1. Feed character A at step 1 → reservoir state encodes A
2. Feed character B at step 2 → reservoir state evolves to encode (A+B)
3. Final attractor state represents the composition
4. Readout classifies: same-group vs different-group

The reservoir's recurrent dynamics naturally integrate sequential inputs. No context feedback, no hash encoding, no dedicated ports.

---

## 1. Motivation

Morphon currently processes independent frames. Each `process()` call receives a `Vec<f64>`, produces a `Vec<f64>`, and retains no temporal context between calls except through the slow drift of synaptic weights and the passive decay of membrane potentials.

This is the single biggest capability gap. Every interesting real-world task involves temporal structure:
- **Language** is sequences of tokens
- **Audio** is sequences of samples
- **Control** with partial observability needs memory over time
- **Time series** (anomaly detection, forecasting) is fundamentally temporal

Spiking networks *should* excel at temporal processing — spike timing is native to the compute model. Transformers bolt on positional encoding because they have no intrinsic notion of time. Morphon has delays, refractory periods, eligibility traces, and persistent activity. The temporal substrate exists. What's missing is the wiring, the credit assignment window, and the feedback loop that turns stored state into usable context.

### Why sequences first (not images, not generation)

1. **Smallest architectural delta.** We already have spike delays, eligibility traces, working memory, and episodic memory. Sequences require wiring changes and parameter adjustments, not building new subsystems.
2. **Largest capability unlock.** Text, audio, control-with-memory, time-series — all are sequences.
3. **Biologically correct.** The cortex is fundamentally a sequence prediction engine (Hawkins, Friston).
4. **Differentiating.** "No-backprop sequence processing" is a strong story.

### Why NOT the XOR circuit approach

See §0.1. The hardwired XOR circuit approach was tested and failed consistently. Compositionality must emerge from the network's dynamics through training, not from pre-wired circuits.

---

## 2. Prerequisites

These are hard gates. Do not begin implementation until both are met.

### Prerequisite 1: CartPole v2 Solved

| Metric | Target |
|---|---|
| `avg_last_100` | ≥ 195.0 |
| `solved` | true |
| Profile | `quick` (200 episodes, 300 max steps) |

### Prerequisite 2: MNIST ≥ 85% Accuracy

| Metric | Target |
|---|---|
| Test accuracy | ≥ 85% |
| Input dimensionality | 784 (28×28 pixels, no PCA) |
| Output classes | 10 |
| Profile | `standard` or `extended` |

---

## 3. What Exists Today (and What It Can't Do)

### 3.1 Temporal Infrastructure Already in Place

| Component | Location | What it does | Temporal gap |
|---|---|---|---|
| **Spike delays** | `morphon.rs:Synapse::delay` + `resonance.rs` | Signals arrive at targets after configurable delay (default 1.0 timestep) | Delays are per-synapse but not learned for temporal patterns. |
| **Eligibility traces** | `learning.rs:update_eligibility()` | STDP-like trace accumulates pre/post correlations, decays with `tau_eligibility=20` | τ=20 timesteps is ~4 process() calls. Too short for sequences. |
| **Tag-and-capture** | `learning.rs:tag_and_capture()` | Strong eligibility → synaptic tag → waits for modulatory signal → weight update | Tag decay (`tau_tag=500`) is long enough. |
| **Working memory** | `memory.rs:WorkingMemory` | Stores active morphon patterns with decay. Capacity ~7. | Stored patterns are not fed back as input. |
| **Refractory period** | `morphon.rs:Morphon::refractory_timer` | Morphon can't fire for N ticks after firing | Creates intrinsic temporal patterning but is static. |
| **Internal steps** | `system.rs:internal_steps` (typically 5) | Multiple sub-ticks per process() call | Sub-tick resolution within a frame, not across frames. |

### 3.2 What's Missing

**M1: Recurrent connectivity.** The developmental programs create feedforward I/O pathways: Sensory → Associative → Motor. No recurrent connections exist within the Associative population. Without recurrence, there's no mechanism for the network's current state to influence its next state — each frame is processed independently.

**M2: Extended credit assignment.** Eligibility traces decay with τ=20 timesteps. At 5 internal steps per process() call, that's ~4 frames of credit assignment window. A 10-element sequence where reward arrives at the end requires credit assignment spanning 10+ frames — the traces are cold long before reward arrives.

**M3: Temporal benchmarks.** No example or benchmark exercises temporal dependencies. CartPole is Markovian. MNIST is static. Without a temporal benchmark, there's no feedback signal.

**M4: Sequence boundary signaling.** The system has no concept of "sequence start" or "sequence end." In RL, episode resets serve this function. For supervised sequence tasks, there's no equivalent.

**Note:** The v2 spec identified M2 as "Context feedback loop." Research on Echo State Networks and Liquid State Machines shows this is unnecessary for sequences of length ≤ 15. The recurrent reservoir's state IS the working memory. Context feedback is deferred to Phase 5 (only if benchmarks fail at longer sequences).

---

## 4. Architecture

### 4.1 Overview

```
                          ┌──────────────────────────────┐
                          │         System::process()     │
                          │                               │
  input(t) ──────────────►│  Sensory ──► Associative ──► Motor ──────► output(t)
                          │               ▲      │        │
                          │               │      │        │
                          │               │      ▼        │
                          │          ┌─────────────┐      │
                          │          │  Recurrent   │      │
                          │          │  Reservoir   │      │
                          │          └─────────────┘      │
                          │                               │
                          │                               │
                          │     Sequence boundary signal  │
  seq_reset() ────────────┘     (resets working memory,   │
                                boosts novelty)           │
```

**Simplified from v2:** Removed the "Context Injection" block. The recurrent reservoir's state trajectory IS the temporal representation. No separate context encoding needed.

### 4.2 Design Principles

1. **No new subsystems.** Temporal processing emerges from wiring changes and parameter adjustments to existing modules.
2. **Biological plausibility.** Recurrence, persistent activity, and temporal credit assignment all have direct biological analogs.
3. **Backward compatible.** Existing examples (CartPole, MNIST) must work identically. Temporal features are opt-in via configuration.
4. **Incremental.** Each component (recurrence, extended traces) is independently testable and useful.
5. **Attractor-based compositionality.** Compositionality emerges from attractor dynamics in the recurrent reservoir, not from hardwired circuits.
6. **Reservoir IS memory.** The recurrent reservoir's state trajectory encodes temporal context. No separate context feedback loop needed for sequences ≤ 15 steps.

---

## 5. Component Design

### 5.1 Recurrent Reservoir (M1)

**What:** A population of Associative morphons with recurrent connections — each morphon can synapse onto other Associative morphons in the same cluster, including itself (autapses).

**Biological analog:** Cortical recurrent circuits. ~80% of excitatory synapses in cortex are recurrent (Douglas & Martin, 2004). This is the most prominent wiring pattern in the brain, and we have zero of it.

**Research basis:** Echo State Networks (Jaeger, 2001) and Liquid State Machines (Maass et al., 2002) prove that random recurrent connectivity + simple readout solves temporal tasks. The reservoir's state trajectory naturally encodes temporal context.

**Implementation:** Extend the developmental bootstrap to create recurrent connections within Associative clusters.

```rust
/// Configuration for recurrent connectivity within Associative populations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrentConfig {
    /// Whether to create recurrent connections during development.
    pub enabled: bool,

    /// Fraction of possible Associative→Associative connections to create.
    /// Biology: ~0.1-0.2 (sparse recurrence). Start conservative.
    pub recurrent_connectivity: f64,

    /// Initial weight for recurrent synapses.
    /// Should be weaker than feedforward to avoid runaway excitation.
    /// Biology: recurrent synapses are ~40-60% of feedforward strength.
    pub recurrent_weight_scale: f64,

    /// Delay range for recurrent synapses (in timesteps).
    /// Longer delays = longer temporal memory within a single process() call.
    pub delay_range: (f64, f64),

    /// Whether recurrent synapses undergo STDP.
    /// If true: the network learns temporal patterns in the recurrence.
    /// If false: recurrence provides fixed-timescale echoes.
    pub plastic: bool,

    /// Allow self-connections (autapses).
    /// Biology: autapses exist and serve self-excitation / self-inhibition.
    /// Useful for persistent activity (a morphon that keeps itself firing).
    pub allow_autapses: bool,
}

impl Default for RecurrentConfig {
    fn default() -> Self {
        Self {
            enabled: false,  // opt-in — existing examples unaffected
            recurrent_connectivity: 0.15,
            recurrent_weight_scale: 0.5,
            delay_range: (1.0, 3.0),
            plastic: true,
            allow_autapses: false,
        }
    }
}
```

**Where it lives:** Field on `DevelopmentalConfig`. The developmental engine creates recurrent connections during the initial bootstrap (`proliferate()`), after feedforward I/O pathways are established. Recurrent connections participate in all existing learning rules — they get eligibility traces, tags, and modulated weight updates like any other synapse.

**Interaction with synaptogenesis:** The existing synaptogenesis in `morphogenesis.rs` already creates new connections based on proximity and activity correlation. With recurrent wiring as a seed, synaptogenesis will naturally extend the recurrent connectivity as the network grows.

**Interaction with inhibitory competition:** Recurrent excitation without inhibitory balance causes runaway activity. This is exactly why local inhibitory competition (Endo V2 prerequisite) matters — the inhibitory interneurons within each cluster naturally counterbalance recurrent excitation. If local inhibitory competition is not yet implemented, the existing global k-WTA provides a less elegant but functional safety net.

**Risk: runaway excitation.** Recurrent connections create positive feedback loops. Mitigation:
- `recurrent_weight_scale` starts at 0.5× feedforward strength
- Homeostatic synaptic scaling (already in `homeostasis.rs`) normalizes total input per morphon
- Endoquilibrium threshold_bias responds to elevated firing rates
- Refractory periods prevent sustained firing
- Start with `recurrent_connectivity=0.15` — sparse, not dense

**Dimensionality requirement:** For compositionality tasks (NLP Tier 3), the reservoir needs sufficient "folding capacity" to separate attractor basins. The current `seed_size=30` may be too small. **Recommendation:** Use `seed_size ≥ 100` for temporal tasks. This ensures the reservoir has enough dimensions to encode distinct trajectories for different input sequences.

**Reservoir density:** For sequence classification (Benchmark B), increase `initial_connectivity` from 0.2 to **0.4** within the reservoir. More recurrent connections = more "echoes" to sustain the signal across multiple timesteps. The risk of runaway excitation is mitigated by the sparse weight scale (0.5×) and homeostatic scaling.

### 5.2 Extended Credit Assignment (M2)

**What:** Increase the eligibility trace time constant to support multi-step temporal dependencies, with **heterogeneous time constants** across morphons.

**Simplified from v2:** The v2 spec proposed multi-timescale eligibility traces with `eligibility_slow` field on every Synapse, leak rates, decay constants, etc. For our target sequences (length 2-8), this is overkill. Instead of a dual-trace system, we use a **distribution of time constants** across morphons.

**Current state:**
- `tau_eligibility = 20` timesteps → ~4 process() calls of credit assignment window
- `tau_trace = 50` timesteps → pre/post spike trace for STDP

**Change:** Make `tau_eligibility` configurable with higher defaults, and add **heterogeneous time constants**.

```rust
pub struct LearningParams {
    // ... existing fields ...

    /// Base time constant for eligibility trace decay.
    /// Default: 20. For temporal tasks: 50-100.
    /// Higher = longer credit assignment window, but noisier.
    pub tau_eligibility: f64,

    /// Time constant for STDP pre/post trace decay.
    /// Default: 50. For temporal tasks: 100-200.
    /// Should be ≥ tau_eligibility.
    pub tau_trace: f64,

    /// Enable heterogeneous time constants across morphons.
    /// When true, each morphon's eligibility trace decays at a slightly
    /// different rate: tau_i = tau_base * (0.5 + random() * 1.5).
    /// This naturally creates "fast" and "slow" memory lanes without
    /// the architectural bloat of a dual-trace system.
    /// Biology: different neuron types have different membrane time constants.
    pub heterogeneous_tau: bool,

    /// Range of tau variation when heterogeneous_tau is enabled.
    /// Default: (0.5, 1.5) means tau ranges from 0.5× to 1.5× the base.
    pub tau_range: (f64, f64),
}
```

**No Synapse struct change needed.** The existing `eligibility: f64` field is sufficient. Each morphon gets a `tau_eligibility_local: f64` assigned at creation time (sampled from the distribution). The decay rate uses the morphon's local tau, not the global one.

**Why heterogeneous taus:** A single tau creates a tradeoff — too short and the reservoir forgets the beginning of the sequence, too long and the system becomes "mushy" and cannot distinguish between `A → B` and `B → A`. Heterogeneous taus naturally create "fast" and "slow" memory lanes:
- Fast morphons (τ ≈ 25): precise timing, distinguish order
- Slow morphons (τ ≈ 100): maintain context across the full sequence
- The readout learns to combine both

This is biologically grounded — different neuron types have different membrane time constants. And it's computationally equivalent to multi-timescale traces without the architectural complexity.

**When to add multi-timescale traces:** If Benchmark C (next-element prediction) fails to converge for period-5 sequences even with heterogeneous taus, add the slow trace as a Phase 5 extension.

### 5.3 Sequence Boundary Signaling (M3)

**What:** An explicit API for signaling sequence boundaries, analogous to episode resets in RL but for supervised/unsupervised temporal tasks.

```rust
impl System {
    /// Signal the start of a new sequence.
    ///
    /// Effects:
    /// 1. Clears working memory (new context, old patterns irrelevant)
    /// 2. Resets eligibility traces to zero (credit from previous sequence
    ///    should not leak into this one)
    /// 3. Injects a Novelty burst (sequence boundaries are inherently novel —
    ///    the system should be maximally attentive at the start of a new sequence)
    /// 4. Does NOT reset membrane potentials (the network's dynamic state
    ///    carries over — only the learned temporal context resets)
    ///
    /// For RL tasks, episode reset already serves this function via inject_reward()
    /// + environment reset. This method is for supervised sequence tasks where
    /// there's no natural episode boundary.
    pub fn sequence_reset(&mut self) {
        // 1. Clear working memory
        self.memory.working.clear();

        // 2. Reset eligibility traces
        for edge_idx in self.topology.graph.edge_indices() {
            if let Some(synapse) = self.topology.graph.edge_weight_mut(edge_idx) {
                synapse.eligibility = 0.0;
            }
        }

        // 3. Novelty burst
        self.neuromodulation.inject(
            crate::neuromodulation::Channel::Novelty,
            0.8,
        );
    }
}
```

**Why not reset everything?** A full state reset (membrane potentials, firing history) destroys the network's learned dynamics. Only the *temporal context* should reset. The *learned representations* persist.

**Future: Novelty-Induced Reset (autonomous).** Manual resets work for benchmarks but are unrealistic for a "live" brain. A future enhancement: if the sensory input changes drastically (high novelty signal) OR a specific "Global Inhibition" spike occurs, the eligibility traces should decay rapidly. This makes the system autonomous — it resets its temporal context when the world changes, not when the test harness tells it to.

```rust
// Future: autonomous reset triggered by novelty surge
if self.neuromodulation.channels[Novelty] > 0.9 {
    // Rapid eligibility decay — "this is a new situation, forget the past"
    for synapse in &mut self.topology.synapses {
        synapse.eligibility *= 0.1;  // rapid decay, not full reset
    }
}
```

This would be added after the benchmarks pass, as a Phase 5 enhancement.

### 5.4 Temporal Benchmarks (M4)

Three benchmarks, ordered by difficulty. Each isolates a specific temporal capability.

#### Benchmark A: Delayed Match-to-Sample (DMS)

**Task:** See a pattern, wait N steps, see a probe, classify whether it matches.

```
Step 1: Input = pattern A (e.g., [1, 0, 0, 1])
Step 2-4: Input = zeros (delay period)
Step 5: Input = pattern A or B (probe)
Output: 1 if probe matches sample, 0 if not
```

**What it tests:** Recurrent reservoir memory. The network must maintain a representation of pattern A across the delay period through its recurrent dynamics.

**Configuration:**
- `target_input_size`: 4-8 (pattern dimensionality)
- `target_output_size`: 1 (match / no-match, via analog readout)
- `RecurrentConfig::enabled`: true
- `LearningParams::tau_eligibility`: 50
- Delay period: 3 steps (start easy), scale to 10+
- Pattern count: 4-8 distinct patterns
- Reward: contrastive on correct/incorrect classification

**Success criterion:** >80% accuracy at delay=3, >70% at delay=5.

**Why this first:** It's the simplest temporal task. Only recurrence + extended traces need to work. If DMS fails, the recurrent reservoir is broken.

#### Benchmark B: Sequence Classification

**Task:** Classify a sequence by its temporal pattern, not its individual elements.

```
Sequence type "rising":  [0.2, 0.4, 0.6, 0.8]  → class 0
Sequence type "falling": [0.8, 0.6, 0.4, 0.2]  → class 1
Sequence type "peak":    [0.2, 0.8, 0.8, 0.2]  → class 2
Sequence type "valley":  [0.8, 0.2, 0.2, 0.8]  → class 3
```

**What it tests:** Recurrent reservoir integration. Each element in isolation is ambiguous (0.8 appears in all classes). The network must integrate information across the full sequence to classify.

**Configuration:**
- `target_input_size`: 1-2 (scalar or small vector per timestep)
- `target_output_size`: 4 (4 classes)
- `RecurrentConfig::enabled`: true
- `LearningParams::tau_eligibility`: 80
- Sequence length: 4 (start), scale to 8-16
- Reward: contrastive at end of sequence

**Success criterion:** >85% accuracy at length=4, >70% at length=8.

**Why second:** Requires the reservoir to integrate information across multiple timesteps. If DMS works but this doesn't, the integration dynamics are wrong.

#### Benchmark C: Next-Element Prediction

**Task:** Given a deterministic sequence, predict the next element.

```
Repeating: [A, B, C, A, B, C, A, B, C, ...] → predict next
Sine wave: [sin(0), sin(0.1), sin(0.2), ...] → predict next
```

**What it tests:** Full temporal processing pipeline: recurrence for temporal dynamics, extended traces for credit assignment.

**Configuration:**
- `target_input_size`: 1 (scalar)
- `target_output_size`: 1 (predicted next value, analog)
- Full temporal config enabled
- Reward: `-|prediction - actual|` (continuous reward signal, per-step)
- Sequence: infinite (no episode boundaries)

**Success criterion:** Prediction error converges below 0.1 for repeating sequences of period ≤ 5.

**Why third:** Hardest. No episode boundaries, continuous prediction, requires the network to discover periodicity from reward signal alone.

#### Benchmark D: NLP Tier 3 — Compositionality

**Task:** Given two characters presented sequentially, classify whether they're same-group (both vowels or both consonants) or different-group (one vowel, one consonant).

```
Step 1: Input = char A (27-dim one-hot)
Step 2: Input = char B (27-dim one-hot)
Output: 0 if same-group (VV or CC), 1 if different-group (VC or CV)
```

**What it tests:** The recurrent reservoir applied to a compositional task. The network must encode character A, then integrate character B with the reservoir state encoding A, and converge to an attractor that represents the composition.

**Configuration:**
- `target_input_size`: 27 (one-hot character encoding)
- `target_output_size`: 2 (same-group vs different-group)
- `RecurrentConfig::enabled`: true
- `LearningParams::tau_eligibility`: 50
- Sequence length: 2 (one char per step)
- Reward: contrastive on correct/incorrect classification

**Success criterion:** >60% accuracy on compositional XOR.

**Why this matters:** This is the gate for temporal sequence processing. It proves the system can combine token meanings via recurrent dynamics — the core capability needed for any sequential task.

---

## 6. Integration Points

### 6.1 Changes to Existing Modules

| Module | Change | Size |
|---|---|---|
| `developmental.rs` | Add `RecurrentConfig` to `DevelopmentalConfig`. In `proliferate()`, after creating feedforward I/O pathways, create recurrent connections within Associative clusters. | ~50 lines |
| `system.rs` | Add `sequence_reset()` method. | ~20 lines |
| `learning.rs` | Make `tau_eligibility` configurable with higher defaults for temporal tasks. No struct changes. | ~5 lines |
| `types.rs` | Add `RecurrentConfig` struct. | ~30 lines |

**Total: ~105 lines of changes across 4 files.** No new modules. No new dependencies. No Synapse struct changes.

### 6.2 What Doesn't Change

- Spike propagation (`resonance.rs`) — recurrent synapses are just synapses
- Neuromodulation — all four channels work the same
- Morphogenesis — synaptogenesis/pruning treats recurrent synapses like any other
- Homeostasis — synaptic scaling normalizes total input including recurrent input
- Working memory — still stores patterns, but the reservoir provides the temporal memory
- Snapshot — serialization unchanged (no new Synapse fields)
- Python/WASM bindings — process() API unchanged, sequence_reset() exposed

### 6.3 Configuration Surface

```rust
pub struct SystemConfig {
    pub developmental: DevelopmentalConfig,    // existing (extended with RecurrentConfig)
    pub learning: LearningParams,             // existing (tau_eligibility configurable)
    // ... rest unchanged
}
```

For non-temporal tasks (CartPole, MNIST), `RecurrentConfig::enabled` defaults to `false`. Zero behavioral change.

---

## 7. Interaction with Other Specs

### 7.1 Pulse Kernel Lite

No conflict. Recurrent synapses are petgraph edges like any other — PKL's hot arrays and sync protocol handle them without modification.

One consideration: recurrent connections increase edge count. At `recurrent_connectivity=0.15` with 100 Associative morphons, that's ~0.15 × 100 × 99 ≈ 1485 new edges. PKL's petgraph edge iteration becomes the hot path sooner. This strengthens the case for PKL, not against it.

### 7.2 Endoquilibrium V2

Recurrence creates new failure modes that Endo should regulate:
- **Runaway excitation:** Endo's existing `threshold_bias` and `arousal_gain` respond to elevated firing rates. No new rule needed.
- **tau_eligibility regulation:** Phase C candidate in Endo V2 spec. The optimal tau depends on sequence length.

### 7.3 ANCS-Core

Temporal processing generates richer episodic memories (sequences of events, not individual frames). ANCS-Core's design should account for temporal episodes.

### 7.4 DeMorphon Spec

The DeMorphon spec describes composite organisms with internal body plans that naturally solve composition. There is significant overlap:

| Aspect | Temporal Processing Spec | DeMorphon Spec |
|--------|------------------------|----------------|
| Compositionality | Attractor dynamics in recurrent reservoir | Internal body plans (Input→Core→Output) |
| Memory | Reservoir state trajectory | Internal body plan state |
| Timescales | Extended eligibility traces | Dual-clock scheduler (already shared) |

**Relationship:** Temporal processing is the **foundation** that DeMorphons build on. The recurrent reservoir creates the temporal substrate. DeMorphons then organize this substrate into specialized composite organisms. Implement temporal processing first; DeMorphons are a later phase.

---

## 8. Implementation Plan

### Phase 0: Prerequisites (must be complete before starting)

| Step | What | Done when |
|---|---|---|
| 0a | Diagnose and fix CartPole v2 regression | `avg_last_100 ≥ 195`, `solved = true` on quick profile |
| 0b | Get MNIST to reportable accuracy | Test accuracy ≥ 85% on standard or extended profile |

### Phase 1: Recurrent Reservoir (M1) + DMS Benchmark

| Step | What | Depends on | Test |
|---|---|---|---|
| 1a | Add `RecurrentConfig` struct to types.rs | Phase 0 | Compiles |
| 1b | Implement recurrent wiring in developmental.rs `proliferate()` | 1a | System with `recurrent.enabled=true` has Assoc→Assoc edges |
| 1c | Write DMS benchmark example | 1a | Runs (accuracy irrelevant initially) |
| 1d | Validate CartPole with `recurrent.enabled=false` | 1b | No regression |
| 1e | Validate CartPole with `recurrent.enabled=true` | 1b | No catastrophic failure |
| 1f | Tune recurrent params on DMS | 1b, 1c | DMS accuracy >50% (above chance) |

### Phase 2: Extended Traces (M2) + Sequence Boundary (M3) + Full Benchmarks

| Step | What | Depends on | Test |
|---|---|---|---|
| 2a | Make `tau_eligibility` configurable with higher defaults | Phase 1 | Compiles |
| 2b | Implement `System::sequence_reset()` | 2a | Working memory clears, eligibility resets, novelty burst |
| 2c | Write sequence classification benchmark (Benchmark B) | 2a, 2b | Runs |
| 2d | Tune on sequence classification | 2c | >85% accuracy at length=4 |
| 2e | Write next-element prediction benchmark (Benchmark C) | 2a | Runs |
| 2f | Tune on next-element prediction | 2e | Error <0.1 for period-3 repeating sequences |
| 2g | Full regression: CartPole + MNIST unchanged | 2a | No regression |

### Phase 3: Compositionality Benchmark

| Step | What | Depends on | Test |
|---|---|---|---|
| 3a | Write compositionality benchmark (sequential token input, XOR task) | Phase 2 | Runs |
| 3b | Train on same-group vs different-group token pairs | 3a | >60% accuracy |
| 3c | Validate compositional reasoning end-to-end | 3b | >60% accuracy stable across seeds |

### Phase 5: Add Context Feedback If Needed (Deferred)

**Only if benchmarks fail at sequences > 10 steps.**

If DMS fails at delay > 7, or sequence classification fails at length > 8, add the context feedback loop from the v2 spec:
- Working memory → hash encoding → context vector → sensory injection
- `ContextFeedbackConfig` struct
- `WorkingMemory::top_n()` and `WorkingMemory::clear()` methods
- Context injection in `process()`

This adds ~60 lines. Don't build it until we need it.

---

## 9. Parameter Tuning

**The spec provides reasonable defaults that work without external optimization.** No custom optimizer needed.

### Recommended Defaults (work out of the box)

| Parameter | Default | Notes |
|---|---|---|
| `recurrent_connectivity` | 0.15 | Sparse recurrence, safe from runaway |
| `recurrent_weight_scale` | 0.5 | Half of feedforward strength |
| `recurrent_delay_range` | (1.0, 3.0) | Short delays for short sequences |
| `tau_eligibility` | 50 | ~10 frames of credit assignment |
| `heterogeneous_tau` | true | Natural multi-scale memory |
| `tau_range` | (0.5, 1.5) | 25-75 timesteps spread |
| `seed_size` | ≥ 100 | Enough dimensions for attractor separation |

### Optional: CMA-ES for Fine-Tuning

The existing `cmaes` crate (`examples/cma_optimize.rs`) can search this parameter space for marginal improvements. It's an **external meta-optimizer**, not part of the core system. Use it if you need to squeeze out the last few percentage points.

**Don't build a custom optimizer.** CMA-ES is well-established and already integrated. Endo handles runtime regulation (neuromodulatory gains). CMA-ES handles offline parameter search. They serve different purposes and complement each other.

| | Endoquilibrium | CMA-ES |
|---|---|---|
| **What** | Runtime self-regulation | Offline parameter search |
| **When** | Every tick | Before/during training |
| **Tunes** | Neuromodulatory channel gains | Structural parameters (tau, connectivity) |
| **Analogy** | Autonomic nervous system | Evolution |

---

## 10. Success Criteria

The temporal sequence processing feature is successful if:

1. **DMS solved:** >80% accuracy at delay=3, >70% at delay=5. This proves recurrent reservoir memory works.
2. **Sequence classification works:** >85% accuracy on 4-class temporal pattern classification at length=4. This proves recurrence + credit assignment work together.
3. **Prediction converges:** Next-element prediction error <0.1 for deterministic repeating sequences of period ≤ 5. This proves the full temporal pipeline handles continuous prediction.
4. **NLP Tier 3 passed:** >60% accuracy on vowel/consonant composition task. This proves compositional reasoning.
5. **No regression:** CartPole (avg ≥ 195) and MNIST (≥ 85%) unchanged when temporal features are disabled.
6. **Opt-in:** All temporal features are behind `enabled: false` defaults. Existing code paths are untouched when not enabled.

### What constitutes failure

- If DMS accuracy doesn't exceed chance (50%) after tuning: recurrent reservoir is not maintaining state. Check that Assoc→Assoc edges exist and have non-zero weights.
- If sequence classification stalls below 60%: recurrence is either too weak (no temporal dynamics) or too strong (runaway). Check firing rates and homeostatic response.
- If CartPole or MNIST regress with temporal features disabled: a code change broke the non-temporal path. Revert and isolate.
- If prediction error doesn't converge for period-2 sequences: credit assignment can't span even 2 steps. Check that `tau_eligibility` is high enough and eligibility traces actually accumulate.
- If NLP Tier 3 stays below 60%: the attractor basins for same-group vs different-group aren't separating. Check that sequential input is actually creating different reservoir states.

---

## 11. What This Does NOT Cover

- **Variable-length input/output:** All benchmarks use fixed-size I/O per timestep.
- **Generation / autoregressive decoding:** The system predicts the next element but doesn't have a mechanism for feeding its output back as input for multi-step generation.
- **Text / tokenization:** Temporal processing operates on numeric vectors. Text requires tokenization + embedding, which is a separate concern.
- **Transformer-competitive performance:** The goal is not to match transformer accuracy. The goal is to prove that a morphogenic system can process temporal dependencies at all, without backpropagation.
- **Hardwired XOR circuits:** This approach was investigated and abandoned. See §0.1.
- **Context feedback loop:** Deferred to Phase 5. Only needed if benchmarks fail at sequences > 10 steps.

---

## 12. Open Questions

1. **Should recurrent connections be delay-learned?** Synapse delays are already a field (`Synapse::delay`) and marked as "learnable" in the doc, but no delay-learning rule exists. **Recommendation:** Ship with fixed random delays from `delay_range`. Add delay learning as a follow-up if benchmark performance plateaus.

2. **Working memory capacity for temporal tasks.** Current capacity is ~7 (Miller's 7±2). For sequences of length 10+, the working memory fills up and starts evicting. **Recommendation:** Start with capacity=7. If DMS fails at delay>7, increase capacity and measure.

3. **Interaction between recurrence and spike delays in resonance.** Recurrent connections have delays. These interact with the resonance engine's spike delivery. A recurrent spike with delay=3 that triggers another recurrent spike with delay=2 creates an echo at t+5. **Recommendation:** This is a feature, not a bug. The resonance cascades are temporal dynamics. But monitor for oscillations in the DMS benchmark.

4. **Should sequence_reset() be called automatically between episodes in RL?** **Recommendation:** Keep it manual initially. The RL examples call `sequence_reset()` explicitly when temporal features are enabled. Future: novelty-induced autonomous reset (§5.3).

5. **Attractor basin separation for compositionality.** How do we ensure that same-group (VV/CC) and different-group (VC/CV) pairs converge to different attractor basins? **Recommendation:** Start with the sequential input approach (§5.4, Benchmark D). If attractor basins don't separate after reasonable training, increase recurrent connectivity and `tau_eligibility`.

6. **Network entropy monitoring.** How do we know if the reservoir is "frozen" (all morphons silent) or "evaporated" (all morphons firing)? **Recommendation:** Add a diagnostic metric that tracks the fraction of morphons firing per step. The "edge of chaos" is where ~10-30% of morphons fire. If firing rate drops below 5% or exceeds 50%, the reservoir is in a bad regime — adjust `recurrent_weight_scale` or `recurrent_connectivity`.

---

## 13. Reservoir Health Monitoring (v3.1 Addition)

Based on feedback, we need to monitor the reservoir's "liquid state" to ensure it's operating at the edge of chaos.

### Network Entropy Metric

Add to `diagnostics.rs`:

```rust
pub struct ReservoirStats {
    /// Fraction of morphons that fired in the last step.
    pub firing_rate: f64,
    
    /// Standard deviation of membrane potentials.
    /// Low = frozen (all similar), High = evaporated (all different).
    pub potential_variance: f64,
    
    /// Number of distinct firing patterns in recent history.
    /// Low = repetitive dynamics, High = rich dynamics.
    pub pattern_diversity: usize,
}
```

**Healthy reservoir indicators:**
- Firing rate: 10-30% (not frozen, not evaporated)
- Potential variance: moderate (not all morphons at same potential)
- Pattern diversity: increases with training (learning new patterns)

**Unhealthy reservoir indicators:**
- Firing rate < 5%: frozen — increase `recurrent_weight_scale` or `recurrent_connectivity`
- Firing rate > 50%: evaporated — decrease `recurrent_weight_scale` or increase inhibitory competition
- Pattern diversity = 1: stuck in limit cycle — increase `tau_eligibility` or add noise

This diagnostic is critical for debugging. Without it, we can't tell if a failing benchmark is due to wrong architecture or wrong parameters.

---

## Changelog

### v3.1 — Feedback Integration (April 2026)

**Changes based on review feedback:**

1. **Added heterogeneous time constants** (§5.2) — Instead of a single `tau_eligibility`, each morphon gets a slightly different decay rate sampled from a distribution. This naturally creates "fast" and "slow" memory lanes without the architectural bloat of v2.0's dual-trace system. Added `heterogeneous_tau: bool` and `tau_range: (f64, f64)` to `LearningParams`.

2. **Added dimensionality requirement** (§5.1) — The reservoir needs `seed_size ≥ 100` for compositionality tasks. The current `seed_size=30` is too small to ensure the reservoir has enough "folding capacity" to separate attractor basins.

3. **Added reservoir density recommendation** (§5.1) — For sequence classification, increase `initial_connectivity` from 0.2 to 0.4 within the reservoir. More recurrent connections = more "echoes" to sustain the signal across multiple timesteps.

4. **Added novelty-induced autonomous reset** (§5.3) — Manual `sequence_reset()` works for benchmarks but is unrealistic for a "live" brain. Future enhancement: if novelty signal exceeds threshold, eligibility traces decay rapidly. This makes the system autonomous.

5. **Added reservoir health monitoring** (§13) — New section on network entropy metrics: firing rate, potential variance, pattern diversity. The "edge of chaos" is where 10-30% of morphons fire. Added `ReservoirStats` struct for diagnostics.

6. **Added CMA-ES parameters** (§9) — `heterogeneous_tau`, `tau_range_min`, `tau_range_max` added to searchable parameters.

### v3.0 — Simplified Architecture (April 2026)

**Major changes from v2:**

1. **Removed M2: Context Feedback Loop** — The v2 spec proposed a complex context feedback system: Working memory → hash encoding → context vector → dedicated sensory ports → context injection. Research on Echo State Networks and Liquid State Machines shows this is unnecessary for sequences of length ≤ 15. The recurrent reservoir's state trajectory IS the temporal representation. The reservoir naturally integrates sequential inputs through its recurrent dynamics. No separate context encoding needed.

   **Why:** The context feedback loop added ~60 lines of code (ContextFeedbackConfig, encode_context, context ports, injection logic) for a mechanism that doesn't improve performance on our target benchmarks. DMS at delay=3, sequence classification at length=4-8, and NLP Tier 3 at length=2 are all well within the reservoir's memory capacity. Deferred to Phase 5 (only if benchmarks fail at sequences > 10 steps).

2. **Simplified M3: Extended Credit Assignment** — The v2 spec proposed multi-timescale eligibility traces with `eligibility_slow` field on every Synapse, leak rates, decay constants, and a separate slow trace update rule. For our target sequences (length 2-8), this is overkill. Just increasing `tau_eligibility` from 20 to 50-100 is sufficient.

   **Why:** The multi-timescale approach added ~25 lines of code plus a new field on every Synapse (8 bytes × 110K synapses = 880 KB). For sequences of length 2-8, a single eligibility trace with τ=50-100 provides 10-20 frames of credit assignment window — more than enough. Multi-timescale traces only become necessary for sequences > 15 steps.

3. **Reduced implementation scope from ~210 lines to ~105 lines** — By removing context feedback and multi-timescale traces, the implementation is halved. Changes now touch 4 files instead of 6, with no Synapse struct changes.

4. **Updated architecture diagram** — Removed the "Context Injection" block. The diagram now shows the simpler flow: input → sensory → associative (with recurrent reservoir) → motor → output.

5. **Updated benchmarks** — DMS now tests recurrent reservoir memory (not context feedback in isolation). Sequence classification tests recurrence + extended traces (not recurrence + context + extended traces).

6. **Added Phase 5: Context Feedback If Needed** — Explicitly deferred. Only implement if benchmarks fail at sequences > 10 steps. This ensures we don't build complexity we don't need.

7. **Added this changelog** — For tracking spec evolution.

**What didn't change:** Recurrent reservoir (M1), sequence boundary signaling (M5), temporal benchmarks (M4), NLP Tier 3 compositionality benchmark, implementation plan phases 0-3, interaction with other specs.

### v2.0 — Research Findings & Dead Paths (April 2026)

- Added §0: Research findings (NeuronSpark, Nam et al., Zheng et al.) and dead path documentation (XOR circuits)
- Added temporal capability assessment (§0.3)
- Added Benchmark D: Compositionality
- Added Phase 3: Compositionality benchmark implementation
- Added interaction with DeMorphon spec
- Added design principle #5: Attractor-based compositionality

### v1.0 — Initial Spec (April 2026)

- Initial temporal sequence processing specification
- Identified 5 missing components (M1-M5)
- Defined 3 benchmarks (DMS, sequence classification, next-element prediction)
- Defined 3-phase implementation plan

---

*Temporal Sequence Processing — because a brain that can't remember yesterday isn't a brain.*

*TasteHub GmbH, Wien, April 2026*
