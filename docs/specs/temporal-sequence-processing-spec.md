# Temporal Sequence Processing
## First Capability Extension for MORPHON
### Technical Specification v1.0 — TasteHub GmbH, April 2026

---

| | |
|---|---|
| **Status** | Design |
| **Author** | Lisa + Claude |
| **Depends on** | Prerequisite 1: CartPole v2 solved (avg ≥ 195), Prerequisite 2: MNIST accuracy ≥ 85% |
| **Interacts with** | Pulse Kernel Lite (hot arrays), Endoquilibrium V2 (trace regulation), ANCS-Core (episodic replay) |

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

1. **Smallest architectural delta.** We already have spike delays, eligibility traces, working memory, and episodic memory. Sequences require wiring these together, not building new subsystems.
2. **Largest capability unlock.** Text, audio, control-with-memory, time-series — all are sequences. Image classification (MNIST) is already partially working. Generation requires a decoder architecture that doesn't exist.
3. **Biologically correct.** The cortex is fundamentally a sequence prediction engine (Hawkins, Friston). A morphogenic system that can't process sequences is missing its most natural capability.
4. **Differentiating.** "No-backprop sequence processing" is a strong story. "No-backprop image classification at 85% on MNIST" is interesting but not compelling — a 2-layer perceptron does that.

---

## 2. Prerequisites

These are hard gates. Do not begin implementation until both are met.

### Prerequisite 1: CartPole v2 Solved

| Metric | Target |
|---|---|
| `avg_last_100` | ≥ 195.0 |
| `solved` | true |
| Profile | `quick` (200 episodes, 300 max steps) |

**Why this matters:** CartPole regression from v0.5.0 (195.2) to v2.0.0 (156.1) signals that the V2 primitives (adaptive receptors, collective compute, dreaming, frustration, field, self-healing) or V3 additions (governance, epistemic states, justification) broke the learning pipeline. Building temporal processing on a regressed foundation will compound the instability. The learning pipeline must be whole before we extend it.

**What "solved" proves:** The three-factor learning rule (eligibility × modulation), tag-and-capture consolidation, contrastive reward, Endoquilibrium regulation, and the developmental bootstrap all work together correctly in the v2 codebase. This is the foundation temporal processing builds on.

### Prerequisite 2: MNIST ≥ 85% Accuracy

| Metric | Target |
|---|---|
| Test accuracy | ≥ 85% |
| Input dimensionality | 784 (28×28 pixels, no PCA) |
| Output classes | 10 |
| Profile | `standard` or `extended` |

**Why this matters:** MNIST proves the system scales beyond toy RL. 784 inputs → 10 outputs requires hundreds of morphons with working credit assignment across multiple layers of associative processing. If this doesn't work, the system can't handle the state sizes that temporal tasks demand (a sequence of length T with input dimension D means the system must maintain context across T×D total input values).

**What 85% proves:** Contrastive reward works for multi-class discrimination. The developmental bootstrap creates viable I/O pathways at scale. Synaptic plasticity converges on stable feature representations. 85% is not state-of-the-art — it's the minimum bar that says "the basic machinery works at non-trivial scale."

---

## 3. What Exists Today (and What It Can't Do)

### 3.1 Temporal Infrastructure Already in Place

| Component | Location | What it does | Temporal gap |
|---|---|---|---|
| **Spike delays** | `morphon.rs:Synapse::delay` + `resonance.rs` | Signals arrive at targets after configurable delay (default 1.0 timestep) | Delays are per-synapse but not learned for temporal patterns. Range is too narrow for multi-step dependencies. |
| **Eligibility traces** | `learning.rs:update_eligibility()` | STDP-like trace accumulates pre/post correlations, decays with `tau_eligibility=20` | τ=20 timesteps is ~4 process() calls at internal_steps=5. Too short for sequences longer than a few elements. |
| **Tag-and-capture** | `learning.rs:tag_and_capture()` | Strong eligibility → synaptic tag → waits for modulatory signal → weight update | Tag decay (`tau_tag=500`) is long enough, but tags are set by eligibility magnitude, not by temporal patterns. |
| **Working memory** | `memory.rs:WorkingMemory` | Stores active morphon patterns with decay. Capacity ~7 (Miller's 7±2). | Stored patterns are not fed back as input. Working memory is write-only from the processing loop's perspective. |
| **Episodic memory** | `memory.rs:EpisodicMemory` | Records significant events (input, output, reward, novelty). Supports replay. | Replay exists but doesn't drive learning — it's a passive record, not an active consolidation mechanism. |
| **Refractory period** | `morphon.rs:Morphon::refractory_timer` | Morphon can't fire for N ticks after firing | Creates intrinsic temporal patterning but is static per morphon. |
| **Internal steps** | `system.rs:internal_steps` (typically 5) | Multiple sub-ticks per process() call | Provides sub-tick temporal resolution but only within a single frame, not across frames. |
| **Neuromodulation** | `neuromodulation.rs` | Four broadcast channels: Reward, Novelty, Arousal, Homeostasis | Modulation is instantaneous broadcast — no temporal modulation profile (e.g., ramping novelty at sequence boundaries). |

### 3.2 What's Missing

**M1: Recurrent connectivity.** The developmental programs (`DevelopmentalConfig::cortical/hippocampal/cerebellar`) create feedforward I/O pathways: Sensory → Associative → Motor. No recurrent connections exist within the Associative population. Without recurrence, there's no mechanism for the network's current state to influence its next state — each frame is processed independently.

**M2: Context feedback loop.** Working memory stores patterns but never re-injects them. The `WorkingMemory::retrieve()` method returns stored patterns, but nothing in `System::step()` or `System::process()` calls it and feeds the result back as additional input. Working memory is a black box that the processing loop doesn't read from.

**M3: Extended credit assignment.** Eligibility traces decay with τ=20 timesteps. At 5 internal steps per process() call, that's ~4 frames of credit assignment window. A 10-element sequence where reward arrives at the end requires credit assignment spanning 10+ frames — the traces are cold long before reward arrives.

**M4: Temporal benchmarks.** No example or benchmark exercises temporal dependencies. CartPole is Markovian (full state observable each frame). MNIST is static (one image, one classification). Without a temporal benchmark, there's no feedback signal for whether temporal processing works.

**M5: Sequence boundary signaling.** The system has no concept of "sequence start" or "sequence end." In RL, episode resets serve this function via `inject_reward()`. For supervised sequence tasks, there's no equivalent — no way to signal "this is a new sequence, reset your temporal context."

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
                    ┌─────┤          │  Reservoir   │      │
                    │     │          └─────────────┘      │
                    │     │               ▲               │
                    │     │               │               │
                    │     │    ┌──────────┴──────────┐    │
                    │     │    │  Context Injection   │    │
                    │     │    │  (Working Memory     │    │
                    │     │    │   → Sensory input)   │    │
                    │     │    └──────────────────────┘    │
                    │     │                               │
                    │     └───────────────────────────────┘
                    │
                    │     Sequence boundary signal
                    │     (resets working memory,
  seq_reset() ─────┘      boosts novelty)
```

### 4.2 Design Principles

1. **No new subsystems.** Temporal processing emerges from wiring changes and parameter adjustments to existing modules. No "sequence processor" module.
2. **Biological plausibility.** Recurrence, persistent activity, and temporal credit assignment all have direct biological analogs. No attention mechanisms, no positional encoding.
3. **Backward compatible.** Existing examples (CartPole, MNIST) must work identically. Temporal features are opt-in via configuration.
4. **Incremental.** Each component (recurrence, context feedback, extended traces) is independently testable and useful.

---

## 5. Component Design

### 5.1 Recurrent Reservoir (M1)

**What:** A population of Associative morphons with recurrent connections — each morphon can synapse onto other Associative morphons in the same cluster, including itself (autapses).

**Biological analog:** Cortical recurrent circuits. ~80% of excitatory synapses in cortex are recurrent (Douglas & Martin, 2004). This is the most prominent wiring pattern in the brain, and we have zero of it.

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
    /// Biology: recurrent delays are 1-5ms (1-5 timesteps at our resolution).
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

**Interaction with synaptogenesis:** The existing synaptogenesis in `morphogenesis.rs` already creates new connections based on proximity and activity correlation. With recurrent wiring as a seed, synaptogenesis will naturally extend the recurrent connectivity as the network grows. No changes needed — the proximity-based rule doesn't distinguish feedforward from recurrent.

**Interaction with inhibitory competition:** Recurrent excitation without inhibitory balance causes runaway activity. This is exactly why local inhibitory competition (Endo V2 prerequisite, §3 of endoquilibrium_v2.md) matters — the inhibitory interneurons within each cluster naturally counterbalance recurrent excitation. If local inhibitory competition is not yet implemented when this ships, the existing global k-WTA provides a less elegant but functional safety net.

**Risk: runaway excitation.** Recurrent connections create positive feedback loops. Mitigation:
- `recurrent_weight_scale` starts at 0.5× feedforward strength
- Homeostatic synaptic scaling (already in `homeostasis.rs`) normalizes total input per morphon
- Endoquilibrium threshold_bias responds to elevated firing rates
- Refractory periods prevent sustained firing
- Start with `recurrent_connectivity=0.15` — sparse, not dense

### 5.2 Context Feedback Loop (M2)

**What:** Working memory patterns are re-injected as additional input to Sensory morphons at the start of each `process()` call. The network sees both the current input AND the echo of its recent past.

**Biological analog:** Prefrontal cortex → sensory cortex top-down projections. PFC maintains task-relevant information in persistent activity and feeds it back to sensory areas, biasing perception toward task-relevant features. This is one of the best-established feedback circuits in neuroscience.

**Implementation:**

```rust
/// Configuration for context feedback from working memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFeedbackConfig {
    /// Whether to inject working memory as additional sensory input.
    pub enabled: bool,

    /// Strength of context injection relative to external input.
    /// 1.0 = same strength as external input.
    /// 0.1-0.3 = mild bias (context nudges, doesn't override).
    pub feedback_strength: f64,

    /// How many working memory items to inject (top-N by activation).
    /// More items = richer context but noisier input.
    pub max_context_items: usize,

    /// How context patterns map to sensory morphons.
    /// "overlap": context pattern IDs that happen to be Sensory get direct injection.
    /// "dedicated": reserve N extra Sensory morphons as "context ports."
    pub injection_mode: ContextInjectionMode,

    /// Number of dedicated context input ports (only used if injection_mode = Dedicated).
    /// These are additional Sensory morphons beyond target_input_size.
    pub context_port_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextInjectionMode {
    /// Working memory patterns that reference Sensory morphon IDs
    /// get injected as additional activation on those morphons.
    /// Simple but couples context to input topology.
    Overlap,

    /// Dedicated Sensory morphons receive a compressed representation
    /// of working memory state. Decouples context from input topology.
    Dedicated,
}

impl Default for ContextFeedbackConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            feedback_strength: 0.2,
            max_context_items: 3,
            injection_mode: ContextInjectionMode::Dedicated,
            context_port_count: 8,
        }
    }
}
```

**Where context injection happens in `process()`:**

```
process(input):
    1. Retrieve top-N working memory items by activation
    2. Encode working memory state into context_port_count dimensional vector
       (hash morphon IDs → fixed-size representation, scale by activation)
    3. Concatenate: extended_input = [input..., context_encoding...]
    4. Inject extended_input into Sensory morphons (original + context ports)
    5. Run internal_steps as normal
    6. After processing, store current Associative firing pattern in working memory
```

**Encoding working memory → context vector:** The working memory stores patterns as `Vec<MorphonId>`. To inject this into fixed-size context ports, use a simple hash-based projection:

```rust
fn encode_context(
    working_memory: &WorkingMemory,
    max_items: usize,
    port_count: usize,
) -> Vec<f64> {
    let mut context = vec![0.0; port_count];
    for item in working_memory.top_n(max_items) {
        for &morphon_id in &item.pattern {
            // Deterministic hash → port index
            let port = (morphon_id as usize) % port_count;
            context[port] += item.activation / item.pattern.len() as f64;
        }
    }
    // Normalize to [0, 1]
    let max_val = context.iter().cloned().fold(0.0_f64, f64::max);
    if max_val > 0.0 {
        for v in &mut context {
            *v /= max_val;
        }
    }
    context
}
```

This is a bloom-filter-like encoding — lossy but fixed-size and deterministic. The network learns to interpret this encoding through normal synaptic plasticity on the context port → Associative connections.

**Impact on target_input_size:** When `ContextFeedbackConfig::enabled` with `Dedicated` mode, the developmental bootstrap creates `target_input_size + context_port_count` Sensory morphons. The first `target_input_size` receive external input; the remaining `context_port_count` receive context encoding. The user doesn't need to change their `target_input_size`.

**Risk: feedback instability.** Context re-injection creates a loop: output → working memory → context → processing → output. Mitigation:
- `feedback_strength=0.2` means context is a weak bias, not a primary driver
- Working memory decays naturally (existing `decay_rate=0.05`)
- Context ports are separate from input ports — they can't override external input
- Sequence boundary reset clears working memory (§5.4)

### 5.3 Extended Credit Assignment (M3)

**What:** Configurable eligibility trace time constants that support multi-step temporal dependencies. Not a mechanism change — a parameter surface expansion with Endoquilibrium regulation.

**Biological analog:** Synaptic tagging and capture (Frey & Morris, 1997). The two-timescale model: fast eligibility trace (seconds) for correlation detection, slow synaptic tag (minutes-hours) for consolidation. We already implement both; the issue is that the fast trace decays too quickly for sequences.

**Current state:**
- `tau_eligibility = 20` timesteps → ~4 process() calls of credit assignment window
- `tau_trace = 50` timesteps → pre/post spike trace for STDP
- `tau_tag = 500` timesteps → synaptic tag persistence (already long enough)

**Change:**

```rust
pub struct LearningParams {
    // ... existing fields ...

    /// Time constant for fast eligibility trace decay.
    /// Default: 20. For temporal tasks: 50-200.
    /// Higher = longer credit assignment window, but noisier (more spurious associations).
    pub tau_eligibility: f64,

    /// Time constant for STDP pre/post trace decay.
    /// Default: 50. For temporal tasks: 100-300.
    /// Must be ≥ tau_eligibility.
    pub tau_trace: f64,

    // NEW FIELDS:

    /// Enable multi-timescale eligibility traces.
    /// When true, each synapse maintains two eligibility traces:
    /// - Fast trace (tau_eligibility): standard STDP-driven, for within-frame correlations
    /// - Slow trace (tau_eligibility_slow): accumulates fast trace, for cross-frame dependencies
    /// The modulated weight update uses the sum: Δw = (e_fast + e_slow) × modulation
    pub multi_timescale_traces: bool,

    /// Time constant for slow eligibility trace.
    /// Only used when multi_timescale_traces = true.
    /// Default: 200 timesteps (~40 process() calls at internal_steps=5).
    pub tau_eligibility_slow: f64,

    /// Decay rate of slow trace per fast-trace update.
    /// Controls how much fast trace "leaks" into slow trace.
    /// 0.0 = no leak (slow trace unused). 1.0 = full copy.
    /// Biology: ~0.1-0.3 (slow trace is a filtered echo of fast trace).
    pub slow_trace_leak: f64,
}
```

**Synapse struct change:**

```rust
pub struct Synapse {
    // ... existing fields ...

    /// Slow eligibility trace for multi-timescale credit assignment.
    /// Accumulates filtered fast eligibility over time.
    /// Only active when LearningParams::multi_timescale_traces = true.
    pub eligibility_slow: f64,
}
```

**Update rule for slow trace:**

```rust
// In update_eligibility(), after computing e_delta for the fast trace:
if params.multi_timescale_traces {
    let slow_decay = (-dt / params.tau_eligibility_slow).exp();
    synapse.eligibility_slow *= slow_decay;
    synapse.eligibility_slow += params.slow_trace_leak * synapse.eligibility.abs();
    synapse.eligibility_slow = synapse.eligibility_slow.clamp(-1.0, 1.0);
}

// In apply_three_factor(), the modulated update becomes:
let total_eligibility = synapse.eligibility
    + if params.multi_timescale_traces { synapse.eligibility_slow } else { 0.0 };
let delta_w = total_eligibility * modulation * params.learning_rate;
```

**Why two traces instead of just increasing tau_eligibility?** A single long trace creates noise — correlations from 40 frames ago are as strong as correlations from 2 frames ago. Two timescales give precision on recent events (fast trace) and fading memory of older events (slow trace, attenuated by `slow_trace_leak`). This matches the biology: fast eligibility (seconds) encodes precise timing, slow tag (minutes) encodes "something happened in this general timeframe."

**Endo interaction:** `tau_eligibility` and `tau_eligibility_slow` are candidates for Endo V2 Phase C regulation (§4.3 of endoquilibrium_v2.md). Short-term: leave them as static config params searchable by CMA-ES. The slow trace leak rate is the most likely future Endo lever — during Proliferating stage, high leak (broad credit assignment, explore). During Mature stage, low leak (precise credit assignment, exploit).

### 5.4 Sequence Boundary Signaling (M5)

**What:** An explicit API for signaling sequence boundaries, analogous to episode resets in RL but for supervised/unsupervised temporal tasks.

```rust
impl System {
    /// Signal the start of a new sequence.
    ///
    /// Effects:
    /// 1. Clears working memory (new context, old patterns irrelevant)
    /// 2. Resets slow eligibility traces to zero (credit from previous sequence
    ///    should not leak into this one)
    /// 3. Injects a Novelty burst (sequence boundaries are inherently novel —
    ///    the system should be maximally attentive at the start of a new sequence)
    /// 4. Does NOT reset membrane potentials or fast traces (the network's
    ///    dynamic state carries over — only the learned temporal context resets)
    ///
    /// For RL tasks, episode reset already serves this function via inject_reward()
    /// + environment reset. This method is for supervised sequence tasks where
    /// there's no natural episode boundary.
    pub fn sequence_reset(&mut self) {
        // 1. Clear working memory
        self.memory.working.clear();

        // 2. Reset slow eligibility traces
        if self.config.learning.multi_timescale_traces {
            for edge_idx in self.topology.graph.edge_indices() {
                if let Some(synapse) = self.topology.graph.edge_weight_mut(edge_idx) {
                    synapse.eligibility_slow = 0.0;
                }
            }
        }

        // 3. Novelty burst
        self.neuromodulation.inject(
            crate::neuromodulation::Channel::Novelty,
            0.8,  // strong but not maximal
        );
    }
}
```

**Why not reset everything?** A full state reset (membrane potentials, firing history, fast traces) destroys the network's learned dynamics. The morphons have learned firing patterns through weight changes — resetting voltages disrupts those patterns. Only the *temporal context* should reset (working memory = "what was I just processing?", slow traces = "what credit should I assign from the previous sequence?"). The *learned representations* persist.

### 5.5 Temporal Benchmarks (M4)

Three benchmarks, ordered by difficulty. Each isolates a specific temporal capability.

#### Benchmark A: Delayed Match-to-Sample (DMS)

**Task:** See a pattern, wait N steps, see two patterns, classify which one matches the original.

```
Step 1: Input = pattern A (e.g., [1, 0, 0, 1])
Step 2-4: Input = zeros (delay period)
Step 5: Input = pattern A or B (probe)
Output: 1 if probe matches sample, 0 if not
```

**What it tests:** Working memory. The network must maintain a representation of pattern A across the delay period and compare it to the probe. No recurrence needed — this is pure context feedback.

**Configuration:**
- `target_input_size`: 4-8 (pattern dimensionality)
- `target_output_size`: 1 (match / no-match, via analog readout)
- `ContextFeedbackConfig::enabled`: true
- `RecurrentConfig::enabled`: false (test context feedback in isolation)
- Delay period: 3 steps (start easy), scale to 10+
- Pattern count: 4-8 distinct patterns
- Reward: contrastive on correct/incorrect classification

**Success criterion:** >80% accuracy at delay=3, >70% at delay=5.

**Why this first:** It's the simplest temporal task. Only one component (context feedback) needs to work. If DMS fails, context feedback is broken and nothing else will work either.

#### Benchmark B: Sequence Classification

**Task:** Classify a sequence by its temporal pattern, not its individual elements.

```
Sequence type "rising":  [0.2, 0.4, 0.6, 0.8]  → class 0
Sequence type "falling": [0.8, 0.6, 0.4, 0.2]  → class 1
Sequence type "peak":    [0.2, 0.8, 0.8, 0.2]  → class 2
Sequence type "valley":  [0.8, 0.2, 0.2, 0.8]  → class 3
```

**What it tests:** Recurrence + context feedback together. Each element in isolation is ambiguous (0.8 appears in all classes). The network must integrate information across the full sequence to classify.

**Configuration:**
- `target_input_size`: 1-2 (scalar or small vector per timestep)
- `target_output_size`: 4 (4 classes)
- `RecurrentConfig::enabled`: true
- `ContextFeedbackConfig::enabled`: true
- `LearningParams::multi_timescale_traces`: true
- Sequence length: 4 (start), scale to 8-16
- Reward: contrastive at end of sequence

**Success criterion:** >85% accuracy at length=4, >70% at length=8.

**Why second:** Requires both recurrence and context feedback. If DMS works but this doesn't, recurrence is the issue.

#### Benchmark C: Next-Element Prediction

**Task:** Given a deterministic sequence, predict the next element. Reward is proportional to prediction accuracy.

```
Repeating: [A, B, C, A, B, C, A, B, C, ...] → predict next
Fibonacci-like: [1, 1, 2, 3, 5, 8, ...] → predict next (normalized)
Sine wave: [sin(0), sin(0.1), sin(0.2), ...] → predict next
```

**What it tests:** Full temporal processing pipeline: recurrence for temporal dynamics, context for longer-term patterns, extended traces for credit assignment (reward is per-step but improvement requires learning the pattern, which spans the full sequence).

**Configuration:**
- `target_input_size`: 1 (scalar)
- `target_output_size`: 1 (predicted next value, analog)
- Full temporal config enabled
- Reward: `-|prediction - actual|` (continuous reward signal, per-step)
- Sequence: infinite (no episode boundaries)

**Success criterion:** Prediction error converges below 0.1 for repeating sequences of period ≤ 5. Sine wave prediction achieves error < 0.15.

**Why third:** Hardest. No episode boundaries, continuous prediction, requires the network to discover periodicity from reward signal alone.

---

## 6. Integration Points

### 6.1 Changes to Existing Modules

| Module | Change | Size |
|---|---|---|
| `developmental.rs` | Add `RecurrentConfig` to `DevelopmentalConfig`. In `proliferate()`, after creating feedforward I/O pathways, create recurrent connections within Associative clusters. | ~50 lines |
| `system.rs` | In `process()`, add context encoding step before input injection. Add `sequence_reset()` method. | ~40 lines |
| `memory.rs` | Add `WorkingMemory::top_n()` and `WorkingMemory::clear()` methods. Add `Associative firing pattern → working memory store` at end of process(). | ~30 lines |
| `learning.rs` | Add `eligibility_slow` field to `Synapse`. Add slow trace update in `update_eligibility()`. Modify `apply_three_factor()` to sum both traces. | ~25 lines |
| `morphon.rs` | Add `eligibility_slow: f64` to `Synapse` struct with default 0.0. | ~3 lines |
| `types.rs` | Add `RecurrentConfig`, `ContextFeedbackConfig`, `ContextInjectionMode` structs. | ~60 lines |

**Total: ~210 lines of changes across 6 files.** No new modules. No new dependencies.

### 6.2 What Doesn't Change

- Spike propagation (`resonance.rs`) — recurrent synapses are just synapses, they propagate through the same engine
- Neuromodulation — all four channels work the same
- Morphogenesis — synaptogenesis/pruning treats recurrent synapses like any other
- Homeostasis — synaptic scaling normalizes total input including recurrent input
- Diagnostics — weight stats include recurrent synapses
- Snapshot — serialization picks up new Synapse field automatically (serde)
- Python/WASM bindings — process() API unchanged, sequence_reset() exposed

### 6.3 Configuration Surface

```rust
pub struct SystemConfig {
    pub developmental: DevelopmentalConfig,    // existing
    pub learning: LearningParams,             // existing (extended)
    pub recurrent: RecurrentConfig,           // NEW
    pub context_feedback: ContextFeedbackConfig, // NEW
    // ... rest unchanged
}
```

For non-temporal tasks (CartPole, MNIST), both new configs default to `enabled: false`. Zero behavioral change.

---

## 7. Interaction with Other Specs

### 7.1 Pulse Kernel Lite

No conflict. Recurrent synapses are petgraph edges like any other — PKL's hot arrays and sync protocol handle them without modification. Context injection happens before the fast path (it's additional input to Sensory morphons), so PKL's `fast_integrate()` receives the extended input naturally.

One consideration: recurrent connections increase edge count. At `recurrent_connectivity=0.15` with 100 Associative morphons, that's ~0.15 × 100 × 99 ≈ 1485 new edges. PKL's petgraph edge iteration becomes the hot path sooner. This strengthens the case for PKL, not against it.

### 7.2 Endoquilibrium V2

Recurrence creates new failure modes that Endo should regulate:
- **Runaway excitation:** Endo's existing `threshold_bias` and `arousal_gain` respond to elevated firing rates. No new rule needed, but the firing rate setpoints may need to be higher for temporal tasks (recurrence naturally increases average firing rate).
- **tau_eligibility regulation:** Phase C candidate in Endo V2 spec. If temporal processing ships first, this becomes more urgent — the optimal tau depends on sequence length, which may change during training.
- **Slow trace leak regulation:** Not in current Endo V2 spec. Should be added as a Phase C candidate once temporal processing is validated.

### 7.3 ANCS-Core

Temporal processing generates richer episodic memories (sequences of events, not individual frames). ANCS-Core's memory backend trait and AXION importance scoring become more valuable when episodes have temporal structure — replay of a sequence is more than replaying individual frames, it's replaying the temporal context in which they occurred. This spec does not depend on ANCS-Core, but ANCS-Core's design should account for temporal episodes.

---

## 8. Implementation Plan

### Phase 0: Prerequisites (must be complete before starting)

| Step | What | Done when |
|---|---|---|
| 0a | Diagnose and fix CartPole v2 regression | `avg_last_100 ≥ 195`, `solved = true` on quick profile |
| 0b | Get MNIST to reportable accuracy | Test accuracy ≥ 85% on standard or extended profile |

### Phase 1: Recurrent Reservoir (M1) + DMS Benchmark (M4-A)

| Step | What | Depends on | Test |
|---|---|---|---|
| 1a | Add `RecurrentConfig` struct to types.rs | Phase 0 | Compiles |
| 1b | Implement recurrent wiring in developmental.rs `proliferate()` | 1a | System with `recurrent.enabled=true` has Assoc→Assoc edges |
| 1c | Write DMS benchmark example | 1a | Runs (accuracy irrelevant initially) |
| 1d | Validate CartPole with `recurrent.enabled=false` | 1b | No regression (identical results) |
| 1e | Validate CartPole with `recurrent.enabled=true` | 1b | No catastrophic failure (may differ slightly due to extra synapses) |
| 1f | Tune recurrent params on DMS | 1b, 1c | DMS accuracy >50% (above chance for 2-class) |

### Phase 2: Context Feedback (M2) + DMS Validation

| Step | What | Depends on | Test |
|---|---|---|---|
| 2a | Add `ContextFeedbackConfig` struct, `ContextInjectionMode` to types.rs | Phase 1 | Compiles |
| 2b | Add `WorkingMemory::top_n()`, `WorkingMemory::clear()` | 2a | Unit tests pass |
| 2c | Implement context encoding in `encode_context()` | 2a | Unit test: deterministic encoding for fixed input |
| 2d | Add context injection to `System::process()` | 2b, 2c | System creates extra context Sensory ports |
| 2e | Add working memory store at end of `process()` (Associative firing pattern) | 2d | Working memory fills during processing |
| 2f | DMS benchmark with context feedback | 2e | DMS accuracy >80% at delay=3 |
| 2g | Validate CartPole with `context_feedback.enabled=false` | 2d | No regression |

### Phase 3: Extended Traces (M3) + Sequence Boundary (M5) + Full Benchmarks

| Step | What | Depends on | Test |
|---|---|---|---|
| 3a | Add `eligibility_slow` to Synapse struct | Phase 2 | Compiles, serde round-trips |
| 3b | Implement multi-timescale trace update in learning.rs | 3a | Unit test: slow trace accumulates, decays with tau_eligibility_slow |
| 3c | Implement `System::sequence_reset()` | 3a | Working memory clears, slow traces reset, novelty burst |
| 3d | Write sequence classification benchmark (Benchmark B) | 3b, 3c | Runs |
| 3e | Tune on sequence classification | 3d | >85% accuracy at length=4 |
| 3f | Write next-element prediction benchmark (Benchmark C) | 3b | Runs |
| 3g | Tune on next-element prediction | 3f | Error <0.1 for period-3 repeating sequences |
| 3h | Full regression: CartPole + MNIST unchanged | 3b | No regression |

---

## 9. CMA-ES Searchable Parameters

Temporal processing adds these to the CMA-ES search space:

| Parameter | Range | Default | Notes |
|---|---|---|---|
| `recurrent_connectivity` | [0.05, 0.3] | 0.15 | Higher = more temporal memory, more runaway risk |
| `recurrent_weight_scale` | [0.2, 0.8] | 0.5 | Relative to feedforward weight |
| `recurrent_delay_min` | [0.5, 2.0] | 1.0 | Minimum recurrent delay |
| `recurrent_delay_max` | [2.0, 5.0] | 3.0 | Maximum recurrent delay |
| `feedback_strength` | [0.05, 0.5] | 0.2 | Context injection strength |
| `context_port_count` | [4, 16] | 8 | Dimensionality of context encoding |
| `tau_eligibility_slow` | [100, 500] | 200 | Slow trace time constant |
| `slow_trace_leak` | [0.05, 0.5] | 0.15 | Fast→slow trace transfer rate |

These interact with existing params (`tau_eligibility`, `learning_rate`, `internal_steps`). The CMA-ES examples (`cma_cartpole`, `cma_endo`) should be extended with a temporal variant (`cma_sequence`) that searches this space on the sequence classification benchmark.

---

## 10. Success Criteria

The temporal sequence processing feature is successful if:

1. **DMS solved:** >80% accuracy at delay=3, >70% at delay=5, on the Delayed Match-to-Sample benchmark. This proves context feedback works.
2. **Sequence classification works:** >85% accuracy on 4-class temporal pattern classification at length=4. This proves recurrence + context + credit assignment work together.
3. **Prediction converges:** Next-element prediction error <0.1 for deterministic repeating sequences of period ≤ 5. This proves the full temporal pipeline handles continuous prediction.
4. **No regression:** CartPole (avg ≥ 195) and MNIST (≥ 85%) unchanged when temporal features are disabled.
5. **Opt-in:** All temporal features are behind `enabled: false` defaults. Existing code paths are untouched when not enabled.

### What constitutes failure

- If DMS accuracy doesn't exceed chance (50%) after tuning: context feedback mechanism is wrong. Re-evaluate the encoding or injection point.
- If sequence classification stalls below 60%: recurrence is either too weak (no temporal dynamics) or too strong (runaway). Check firing rates and homeostatic response.
- If CartPole or MNIST regress with temporal features disabled: a code change broke the non-temporal path. Revert and isolate.
- If prediction error doesn't converge for period-2 sequences: credit assignment can't span even 2 steps. The slow trace mechanism isn't working — check that eligibility_slow actually accumulates and contributes to weight updates.

---

## 11. What This Does NOT Cover

- **Variable-length input/output:** All benchmarks use fixed-size I/O per timestep. Variable-length sequences (e.g., natural language with variable token count) require an attention-like mechanism or dynamic I/O sizing that's out of scope.
- **Generation / autoregressive decoding:** The system predicts the next element but doesn't have a mechanism for feeding its output back as input for multi-step generation. That's the next capability after temporal processing.
- **Text / tokenization:** Temporal processing operates on numeric vectors. Text requires tokenization + embedding, which is a separate concern. Once temporal processing works on numeric sequences, text becomes "temporal processing + tokenizer."
- **Transformer-competitive performance:** The goal is not to match transformer accuracy on sequence tasks. The goal is to prove that a morphogenic system can process temporal dependencies at all, without backpropagation, using biologically plausible mechanisms. Performance optimization comes later.
- **Online learning during inference:** The system already learns online (weights update every medium-path tick). This spec doesn't change that — it extends the temporal window over which online learning operates. Explicit "train vs. inference" mode separation is not addressed.

---

## 12. Open Questions

1. **Should recurrent connections be delay-learned?** Synapse delays are already a field (`Synapse::delay`) and marked as "learnable" in the doc, but no delay-learning rule exists. For temporal processing, delay learning is powerful — the network could learn to create specific temporal patterns by adjusting delays. But delay learning interacts with STDP in complex ways (the temporal window shifts as delays change). **Recommendation:** Ship with fixed random delays from `delay_range`. Add delay learning as a follow-up if benchmark performance plateaus.

2. **Working memory capacity for temporal tasks.** Current capacity is ~7 (Miller's 7±2). For sequences of length 10+, the working memory fills up and starts evicting. Is that correct behavior (limited temporal context, like humans) or a bottleneck? **Recommendation:** Start with capacity=7. If DMS fails at delay>7, increase capacity and measure. The capacity limit is biologically motivated — don't remove it, but it may need to be a CMA-ES searchable param.

3. **Context encoding quality.** The hash-based encoding (§5.2) is lossy. Two different working memory states might hash to the same context vector. At `context_port_count=8`, the collision probability is low for small patterns but increases with network size. **Recommendation:** Start with the hash encoding. If context feedback fails on DMS (which is the cleanest test), try a learned encoding via a small dedicated readout from working memory to context ports.

4. **Interaction between recurrence and spike delays in resonance.** Recurrent connections have delays (§5.1, `delay_range`). These interact with the resonance engine's spike delivery. A recurrent spike with delay=3 that triggers another recurrent spike with delay=2 creates an echo at t+5 — which might interfere with the next process() call's input. **Recommendation:** This is a feature, not a bug. The resonance cascades are temporal dynamics. But monitor for oscillations in the DMS benchmark.

5. **Should sequence_reset() be called automatically between episodes in RL?** Currently, RL examples manually call `inject_reward()` and reset the environment between episodes. Adding an automatic `sequence_reset()` at episode boundaries would couple the RL loop to the temporal system. **Recommendation:** Keep it manual. The RL examples call `sequence_reset()` explicitly when temporal features are enabled. This avoids hidden coupling.

6. **Multi-timescale traces and memory cost.** Adding `eligibility_slow: f64` to every Synapse adds 8 bytes per synapse. At 110K synapses (observed in CartPole runs), that's ~880 KB. Not a problem for desktop but worth noting for WASM builds where memory is constrained. **Recommendation:** Skip the field entirely when `multi_timescale_traces = false` (use a separate `HashMap<EdgeIndex, f64>` or accept the 8 bytes and zero it).

---

*Temporal Sequence Processing — because a brain that can't remember yesterday isn't a brain.*

*TasteHub GmbH, Wien, April 2026*
