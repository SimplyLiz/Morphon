# Coupled Oscillatory Dynamics — Phase-Coupled Spike Timing
## The Huygens Synchrony Implementation for Morphogenic Intelligence
### Technical Specification v1.0 — TasteHub GmbH, April 2026

---

| | |
|---|---|
| **System name** | Coupled Oscillatory Dynamics (COD) |
| **Not part of** | Endoquilibrium (Endo is regulation; COD is computation) |
| **Lives in** | Resonance Engine (`resonance.rs`) — extended from spike delivery to include phase dynamics |
| **Depends on** | Local inhibitory competition (Phase 1), Endoquilibrium V1 (implemented) |
| **Recommended after** | Metabolic selection fix (validated), Endo V2 Phase A (astrocytic gate) |
| **Biological basis** | Kuramoto coupled oscillators, gamma/theta neural oscillations, Huygens synchrony (Copenhagen 2024) |
| **Effort** | 4–6 hours core implementation, 2–3 hours validation |
| **Priority** | Phase 2C (after Endo V2 Phase A+/B, before Phase 3 MNIST benchmark) |

---

## 1. What This Is and What It Isn't

### 1.1 What It Is

Every neuron in the brain is an oscillator. Between spikes, the membrane potential rises and falls rhythmically, driven by intrinsic ion channel dynamics and synaptic input. When neurons are coupled through synapses, their oscillations can synchronize — locking into phase relationships where they fire together (in-phase) or in alternation (anti-phase). This synchronization is not a side effect. It is a computational mechanism: the brain uses oscillatory phase to bind distributed features into coherent percepts (Singer & Gray, 1995), gate information flow between regions (Fries, 2005), and coordinate memory encoding and retrieval (Buzsáki, 2006).

In MORPHON, morphons currently fire when voltage exceeds threshold — a point process with no oscillatory structure. The phase variable adds a continuous oscillatory state to each morphon, creating temporal windows of high and low excitability. Morphons that synchronize their phases fire together, reinforcing each other. Morphons that desynchronize fire at different times, effectively ignoring each other. This turns timing into a computational resource.

### 1.2 What It Isn't

This is **not** Endoquilibrium. Endoquilibrium is a slow, systemic, regulatory controller — it senses aggregate vitals and adjusts global gains. COD is fast, local, and computational — it determines *when* individual morphons are receptive to input and *which* morphons fire together.

The hierarchy:

| System | Biological analog | What it controls | Timescale |
|---|---|---|---|
| **Electrome / Resonance Engine** | Action potentials, synaptic transmission | Spike propagation, voltage integration | Fastest (1 tick) |
| **COD (this spec)** | Neural oscillations (gamma, theta) | Phase alignment, temporal binding, firing windows | Fast-medium (5–50 ticks) |
| **iSTDP** | GABAergic synaptic plasticity | Inhibitory synapse strength | Medium (per spike event) |
| **Astrocytic gate (Endo V2)** | Tripartite synapse | Per-morphon plasticity gating | Slow (500–1000 ticks) |
| **Endoquilibrium** | Endocrine system | Systemic gain modulation, stage detection | Medium-slow (50–500 ticks) |
| **Governor** | Constitutional constraints | Hard limits | Static |

COD sits between the Electrome (spike dynamics) and the learning/regulation layers. It modulates *when* spikes happen, not *whether* learning occurs or *how* the system is regulated.

This is also **not frequency-encoded classification** ("the network resonates at the 3-frequency"). Neural oscillations don't encode class identity as frequency. They bind features through phase synchronization at a common frequency. A gamma-frequency cluster for "vertical edge" and a gamma-frequency cluster for "horizontal edge" synchronize when they both belong to the same digit — binding by synchrony. The frequency is the carrier, not the signal. The phase relationships between clusters are the signal.

---

## 2. The Biology

### 2.1 Huygens Synchrony (Copenhagen, 2024)

Huygens observed in 1655 that two pendulum clocks on the same wall synchronize their swings through mechanical coupling via the wall. The Copenhagen group (Heltberg, Jensen, Nielsen) demonstrated that human cells synchronize their metabolic and transport rhythms through the same mechanism — coupling through a shared medium (extracellular matrix, gap junctions, chemical signaling).

Key properties of Huygens synchrony:

1. **No central controller.** Synchronization emerges from pairwise coupling, not from a master clock.
2. **Coupling medium matters.** The "wall" (shared substrate) determines coupling strength. Stiffer wall = stronger coupling = faster synchronization.
3. **Frequency matching.** Oscillators with similar natural frequencies synchronize easily. Distant frequencies require stronger coupling.
4. **Phase is the information carrier.** Two clocks at the same frequency but different phases are out of sync. Phase alignment is the synchronization condition.

### 2.2 Neural Oscillations

In the cortex, neural oscillations serve specific computational functions:

| Band | Frequency | Function | MORPHON analog |
|---|---|---|---|
| **Gamma** | 30–80 Hz | Feature binding within a region, attention | Intra-cluster synchronization |
| **Beta** | 13–30 Hz | Sensorimotor integration, status quo maintenance | Inter-cluster communication |
| **Theta** | 4–8 Hz | Memory encoding, sequence processing | Working memory gating |
| **Alpha** | 8–13 Hz | Idle/inhibition, attentional suppression | Inactive cluster state |

MORPHON's tick rate determines which biological frequencies map to which computational timescales. With `internal_steps=5` and a medium_period of 10 ticks, a gamma-like cycle (4–8 ticks) fits naturally within the fast path.

### 2.3 The EI-Kuramoto Model

Iwase et al. (2025, Neural Computation) introduced the EI-Kuramoto model: a Kuramoto model where oscillators are split into excitatory and inhibitory groups with four interaction types (E→E, E→I, I→E, I→I). This produces three dynamic states — synchronized, bistable, and desynchronized — controlled by the E/I balance. MORPHON already has excitatory and inhibitory morphons through the local inhibition system (Endo V2 Section 3). The EI-Kuramoto model maps directly onto this existing architecture.

### 2.4 Phase Oscillators with Structural Plasticity

Tass et al. (2022, Scientific Reports) studied Kuramoto oscillators with both STDP and structural plasticity — a network that changes its weights AND its topology based on oscillatory dynamics. They found that STDP alone creates synchronized clusters, but structural plasticity (adding/removing connections) enables desynchronization through rewiring. This is exactly MORPHON's architecture: STDP for weight learning, morphogenesis for structural plasticity. Adding phase dynamics to this existing combination is natural, not forced.

---

## 3. The Design

### 3.1 Per-Morphon Phase State

One new field on the `Morphon` struct:

```rust
/// Oscillatory phase state ∈ [0.0, 1.0).
/// 0.0 = just fired (post-spike reset), 0.5 = maximally hyperpolarized,
/// ~1.0 = approaching threshold (most excitable).
/// Phase advances continuously; spike resets it to 0.0.
#[serde(default = "default_phase")]
pub phase: f64,

/// Natural frequency — intrinsic oscillation speed.
/// Determined by threshold and membrane dynamics.
/// Higher frequency = faster cycling = more spikes per unit time.
#[serde(default = "default_natural_frequency")]
pub natural_frequency: f64,

fn default_phase() -> f64 { 0.0 }
fn default_natural_frequency() -> f64 { 0.05 }  // ~20 ticks per cycle
```

The phase is a **linear accumulator** — it advances by `natural_frequency` per tick and wraps at 1.0. This is simpler than computing sine/cosine every tick and produces equivalent dynamics for coupled oscillators.

### 3.2 Phase Update Rule (Extended Kuramoto)

Updated on the fast path, every tick, as part of the Resonance Engine's spike propagation:

```rust
impl Morphon {
    /// Update oscillatory phase. Called every fast-path tick.
    fn update_phase(
        &mut self,
        neighbor_phases: &[(f64, f64)],  // (neighbor_phase, coupling_weight)
        config: &OscillatorConfig,
    ) {
        // === Natural frequency advance ===
        self.phase += self.natural_frequency;
        
        // === Kuramoto coupling: dθ/dt += (K/N) Σ sin(2π(θ_j - θ_i)) ===
        if !neighbor_phases.is_empty() {
            let n = neighbor_phases.len() as f64;
            let mut coupling_sum = 0.0;
            for &(neighbor_phase, weight) in neighbor_phases {
                let phase_diff = neighbor_phase - self.phase;
                // sin(2π·Δφ) approximated by sin_fast for performance
                coupling_sum += weight * sin_2pi(phase_diff);
            }
            self.phase += config.coupling_strength * coupling_sum / n;
        }
        
        // === Phase-dependent threshold modulation ===
        // Excitability peaks at phase ≈ 1.0 (about to fire)
        // Excitability lowest at phase ≈ 0.5 (maximally hyperpolarized)
        let phase_modulation = cos_2pi(self.phase) * config.threshold_modulation_depth;
        self.effective_threshold_offset = -phase_modulation;
        // Negative offset at phase≈1.0 = lower threshold = easier to fire
        // Positive offset at phase≈0.5 = higher threshold = harder to fire
        
        // === Spike reset ===
        if self.fired {
            self.phase = 0.0;  // Reset phase on spike (phase-resetting)
        }
        
        // === Wrap phase ===
        self.phase = self.phase.rem_euclid(1.0);
    }
}

/// Fast approximation of sin(2π·x) for x ∈ [-0.5, 0.5].
/// Uses the Bhaskara I approximation: accurate to <0.2% error.
fn sin_2pi(x: f64) -> f64 {
    let x = x.rem_euclid(1.0);
    let x = if x > 0.5 { x - 1.0 } else { x };
    // Bhaskara I: sin(π·x) ≈ 16x(1-x) / (5 - x(1-x)) for x ∈ [0,1]
    // We need sin(2π·x) = sin(π·2x)
    let t = 2.0 * x;
    let t_abs = t.abs();
    let approx = 16.0 * t_abs * (1.0 - t_abs) / (5.0 - t_abs * (1.0 - t_abs));
    if t < 0.0 { -approx } else { approx }
}

fn cos_2pi(x: f64) -> f64 {
    sin_2pi(x + 0.25)
}
```

### 3.3 Configuration

```rust
/// Coupled oscillatory dynamics configuration.
/// All defaults are disabled (0.0) — opt-in, zero behavioral change for existing tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatorConfig {
    /// Master switch — if false, skip all phase updates.
    #[serde(default)]
    pub enabled: bool,
    
    /// Global coupling strength (K in Kuramoto model).
    /// Higher = faster synchronization, tighter phase locking.
    /// Too high = all morphons lock into one beat (trivial synchrony).
    /// Too low = no synchronization (independent oscillators).
    /// Typical range: 0.01–0.1.
    #[serde(default)]
    pub coupling_strength: f64,
    
    /// How much the oscillatory phase modulates the firing threshold.
    /// 0.0 = phase has no effect on firing (oscillations are decorative).
    /// 0.3 = threshold varies ±30% with phase (strong temporal gating).
    /// Typical range: 0.1–0.3.
    #[serde(default)]
    pub threshold_modulation_depth: f64,
    
    /// Base natural frequency for new morphons.
    /// Actual frequency varies per morphon (see Section 3.4).
    /// Typical: 0.05 (20 ticks per cycle, ~gamma range at 1ms tick).
    #[serde(default = "default_nat_freq")]
    pub base_natural_frequency: f64,
    
    /// Frequency spread — how much natural frequencies vary across morphons.
    /// Higher spread = harder to synchronize = more diverse rhythms.
    /// Typical: 0.01 (±20% around base frequency).
    #[serde(default)]
    pub frequency_spread: f64,
    
    /// Whether to use synaptic weights as coupling weights.
    /// If true: coupling weight = synapse.weight (strong connections couple more).
    /// If false: coupling weight = 1.0 for all neighbors (uniform coupling).
    #[serde(default = "default_true")]
    pub weight_proportional_coupling: bool,
    
    /// Whether excitatory and inhibitory morphons couple differently.
    /// If true: inhibitory synapses produce anti-phase coupling (repulsive).
    /// This implements the EI-Kuramoto model (Iwase et al. 2025).
    #[serde(default = "default_true")]
    pub ei_kuramoto: bool,
}

fn default_nat_freq() -> f64 { 0.05 }
fn default_true() -> bool { true }

impl Default for OscillatorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            coupling_strength: 0.0,
            threshold_modulation_depth: 0.0,
            base_natural_frequency: 0.05,
            frequency_spread: 0.01,
            weight_proportional_coupling: true,
            ei_kuramoto: true,
        }
    }
}
```

### 3.4 Natural Frequency Assignment

Each morphon gets a natural frequency at birth:

```rust
fn assign_natural_frequency(config: &OscillatorConfig, rng: &mut impl Rng) -> f64 {
    let base = config.base_natural_frequency;
    let spread = config.frequency_spread;
    // Gaussian around base, clamped to positive
    let freq = base + rng.sample::<f64, _>(StandardNormal) * spread;
    freq.max(0.01)  // minimum frequency — never static
}
```

Natural frequency is **heritable** — when the MorphonGenome is implemented (Phase 6), `natural_frequency` becomes a genome field that mutates at division. Different lineages can evolve different oscillation speeds, creating frequency-based specialization (fast oscillators for temporal precision, slow oscillators for integration).

### 3.5 Phase Coupling Sources

The `neighbor_phases` input comes from two sources:

**Synaptic coupling (primary):** For each incoming synapse to morphon i, the presynaptic morphon's phase contributes proportional to the synapse weight:

```rust
fn collect_phase_neighbors(
    morphon_id: MorphonId,
    morphons: &HashMap<MorphonId, Morphon>,
    topology: &Topology,
    config: &OscillatorConfig,
) -> Vec<(f64, f64)> {
    topology.incoming(morphon_id)
        .iter()
        .filter_map(|(source_id, synapse)| {
            let source = morphons.get(source_id)?;
            let weight = if config.weight_proportional_coupling {
                if config.ei_kuramoto && synapse.weight < 0.0 {
                    synapse.weight  // Inhibitory: negative weight → repulsive coupling
                } else {
                    synapse.weight.abs()  // Excitatory: positive → attractive coupling
                }
            } else {
                if config.ei_kuramoto && synapse.weight < 0.0 { -1.0 } else { 1.0 }
            };
            Some((source.phase, weight))
        })
        .collect()
}
```

With `ei_kuramoto: true`, inhibitory synapses produce **repulsive** coupling — the inhibitory morphon pushes its target toward anti-phase (opposite timing). This creates the E/I oscillatory balance: excitatory morphons in a cluster synchronize (in-phase), while the inhibitory interneuron oscillates in anti-phase, creating the alternating excitation/inhibition windows that define gamma oscillations.

**Field coupling (optional, future):** The Poincaré ball field could provide a second coupling channel — morphons close in hyperbolic space experience each other's phase through the bioelectric field, independent of synaptic connections. This is the "wall" in Huygens' analogy. Deferred to a future version — synaptic coupling is sufficient and well-grounded.

---

## 4. Interactions with Existing Systems

### 4.1 Spike Generation

Currently: morphon fires when `voltage > threshold`.
With COD: morphon fires when `voltage > threshold + effective_threshold_offset`.

At phase ≈ 1.0 (peak excitability), `effective_threshold_offset` is negative, lowering the effective threshold. The morphon is "ready to fire" — a modest input pushes it over. At phase ≈ 0.5 (trough), the offset is positive, raising the threshold. Even strong input may not trigger a spike.

This creates **temporal windows of opportunity.** A morphon is receptive during its excitable phase and refractory (beyond the normal refractory period) during its inhibited phase. Morphons that synchronize their phases are simultaneously receptive — they can exchange information. Desynchronized morphons are receptive at different times — they effectively ignore each other.

### 4.2 Local Inhibitory Competition (iSTDP)

Local inhibition determines *which* morphons fire in a neighborhood. Phase coupling determines *when* they fire. These are complementary:

Without COD: local competition resolves in 2–3 sub-ticks. The first morphon to reach threshold wins, inhibitory interneurons suppress the rest. Winner selection depends on input strength.

With COD: local competition resolves within a phase-defined window. Morphons whose phase is near 1.0 (excitable) compete. Morphons whose phase is near 0.5 are not competing — they're in their refractory phase. This means the *same* input can produce *different* winners at different times in the oscillatory cycle, increasing representational diversity over time.

The inhibitory interneuron oscillates in anti-phase to excitatory morphons (through repulsive coupling). When excitatory morphons are at their peak, the interneuron is at its trough — minimal inhibition, the window is open. As the excitatory morphons fire, the interneuron approaches its peak — maximal inhibition, the window closes. This is the biological gamma cycle: excitation → inhibition → excitation → inhibition, at ~40Hz.

### 4.3 Myelination (Axonal Properties)

Myelinated synapses have shorter effective delays. In oscillatory networks, delay determines the phase relationship between coupled oscillators. A myelinated connection between morphon A and morphon B couples them at a shorter phase lag — they synchronize more tightly. An unmyelinated connection couples them at a longer phase lag — they synchronize loosely or not at all.

This means myelination doesn't just make pathways faster — it makes them more phase-coherent. Consolidated, myelinated pathways carry phase-locked signals. New, unmyelinated pathways carry phase-jittered signals. The phase coherence of a pathway becomes a measure of its maturity and reliability.

### 4.4 Endoquilibrium

Endo does not control phase dynamics directly. It modulates the *conditions* under which phase dynamics operate:

| Endo lever | Effect on COD |
|---|---|
| `arousal_gain` | Modulates `coupling_strength` globally. High arousal → strong coupling → tight synchronization → precise timing. Low arousal → weak coupling → loose synchrony → broader integration windows. |
| `threshold_bias` | Adds to the effective threshold, shifting the excitability profile. High bias → harder to fire at any phase → sparser activity. Low bias → easier to fire → denser activity. |
| `plasticity_mult` | No direct effect on COD (phase coupling is not a learning mechanism). |
| Developmental stage | Proliferating: low coupling (diverse, unsynchronized exploration). Mature: high coupling (coordinated, synchronized computation). Stressed: coupling disrupted (forced desynchronization for exploration). |

The mapping to neuromodulatory control of oscillations is biologically direct: acetylcholine enhances gamma synchronization (→ arousal_gain modulates coupling_strength), noradrenaline desynchronizes for exploration (→ Stressed stage reduces coupling), dopamine gates reward-related synchronization (→ reward_gain affects which clusters synchronize during reward delivery).

### 4.5 Astrocytic Gate

The astrocytic gate modulates plasticity per-morphon. Phase coupling is not a plasticity mechanism — it's a dynamic computational mechanism. The two are orthogonal: the astrocytic gate says "this morphon is allowed to learn." Phase coupling says "this morphon is ready to fire right now."

However, there's an indirect interaction: morphons that are phase-synchronized fire together, which means they trigger STDP together (correlated pre/post timing), which means the astrocytic gate's plasticity modulation affects groups of synchronized morphons collectively. A cluster of phase-locked morphons acts as a learning unit — the astrocytic gate opens or closes for the whole synchronized ensemble.

### 4.6 DeMorphon

DeMorphon internal morphons should synchronize strongly (strong internal coupling). The DeMorphon's Output cell fires at a specific phase relative to the internal oscillation — it produces a phase-locked output signal. Different DeMorphons that process related features synchronize their output phases — binding by synchrony. DeMorphons that process unrelated features desynchronize.

This is the "binding problem" solution: how does the system know that the "vertical edge" detected by DeMorphon A and the "horizontal edge" detected by DeMorphon B belong to the same digit? Answer: their output phases are synchronized. The readout can detect co-firing (phase-locked outputs) vs. independent firing (phase-unlocked outputs) as a binding signal.

### 4.7 Working Memory (Persistent Activity)

The bistable Memory cell pair in a DeMorphon maintains persistent activity through phase-locked oscillation. Both cells oscillate in anti-phase, keeping each other alive: when Cell 1 is at its peak, it excites Cell 2 (which is at its trough), and vice versa. This reciprocal phase-locked oscillation sustains activity indefinitely without external input.

Breaking the phase lock (through a strong inhibitory pulse that resets both cells to the same phase) collapses the bistable state — clearing the working memory. This is the "reset" mechanism for working memory, implemented through phase manipulation rather than direct voltage manipulation.

### 4.8 MorphonGenome

When implemented, `natural_frequency` becomes a heritable genome field:

```rust
pub struct MorphonGenome {
    // ... existing fields ...
    
    /// Natural oscillation frequency — heritable, mutable at division.
    /// Different lineages evolve different rhythms.
    pub natural_frequency: f64,
    
    /// Coupling strength preference — heritable.
    /// High: gregarious oscillator, synchronizes easily.
    /// Low: independent oscillator, maintains own rhythm.
    pub coupling_preference: f64,
}
```

This enables evolutionary frequency specialization: some lineages evolve fast oscillations (temporal precision, good for spike-timing-dependent tasks), others evolve slow oscillations (temporal integration, good for accumulating evidence over time). Natural selection through the metabolic system favors the frequencies that produce reward-correlated firing patterns.

---

## 5. Emergent Computational Capabilities

### 5.1 Temporal Binding (Feature Integration)

**Problem:** How does the system know that two features belong to the same object?

**Solution without COD:** It doesn't. Each morphon fires independently. The readout must learn all pairwise combinations explicitly.

**Solution with COD:** Features that co-occur synchronize their phases. The readout detects synchronized firing as "these features belong together." A morphon for "vertical edge at position (3,5)" and a morphon for "horizontal edge at position (3,7)" synchronize when they're both active for the same digit — their phase-locked co-firing tells the readout they're part of the same spatial structure.

### 5.2 Temporal Attention (Information Gating)

**Problem:** How does the system focus on relevant input and ignore distracting input?

**Solution without COD:** Global k-WTA or local inhibition suppresses losers. But this is all-or-nothing — you fire or you don't.

**Solution with COD:** Phase-dependent threshold modulation creates graded attention. Morphons whose phases are aligned with the current "attention rhythm" (set by the most active cluster) are more receptive. Morphons out of phase are less receptive. Attention is a phase alignment — attended features synchronize, unattended features desynchronize. This is the "Communication Through Coherence" (CTC) hypothesis (Fries, 2005).

### 5.3 Sequence Discrimination

**Problem:** How does the system distinguish "A then B" from "B then A"?

**Solution without COD:** Requires explicit delay chains (the DeMorphon temporal pattern detection mechanism).

**Solution with COD:** The phase at which a stimulus arrives determines its effect. Input A arriving at phase 0.2 produces a different phase trajectory than input A arriving at phase 0.8. Two sequential inputs create a phase-coded temporal pattern that differs depending on the order. The system doesn't need explicit delay chains — the oscillatory dynamics naturally encode temporal order through phase relationships.

### 5.4 Noise Filtering (Oscillatory Filtering)

**Problem:** How does the system ignore random noise in the input?

**Solution without COD:** Threshold-based filtering — noise below threshold is ignored.

**Solution with COD:** Phase-coherent input (signal) reinforces the oscillatory rhythm. Phase-incoherent input (noise) cancels out across the oscillatory cycle. This is oscillatory filtering — the system naturally bandpass-filters its input around the resonant frequency. Random noise that doesn't have temporal structure is attenuated. Temporally structured input that matches the oscillation frequency is amplified.

---

## 6. Observability and Diagnostics

### 6.1 Kuramoto Order Parameter

The standard measure of synchronization in coupled oscillator networks:

```rust
/// Compute the Kuramoto order parameter for a set of morphons.
/// r = |1/N Σ exp(2πi·φ_j)| ∈ [0, 1]
/// r = 0: completely desynchronized (random phases)
/// r = 1: perfectly synchronized (all phases identical)
fn kuramoto_order_parameter(morphons: &[&Morphon]) -> f64 {
    if morphons.is_empty() { return 0.0; }
    let n = morphons.len() as f64;
    let (sum_cos, sum_sin) = morphons.iter()
        .fold((0.0, 0.0), |(c, s), m| {
            (c + cos_2pi(m.phase), s + sin_2pi(m.phase))
        });
    ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt()
}
```

Track per cluster and globally. Report in diagnostics alongside existing metrics.

### 6.2 Phase Locking Value (PLV)

Pairwise synchronization between two morphons over a time window:

```rust
/// Phase Locking Value between two morphons over their recent history.
/// PLV ∈ [0, 1]: 0 = no phase relationship, 1 = perfectly locked.
fn phase_locking_value(
    phase_history_a: &[f64],
    phase_history_b: &[f64],
) -> f64 {
    let n = phase_history_a.len().min(phase_history_b.len()) as f64;
    if n < 2.0 { return 0.0; }
    let (sum_cos, sum_sin) = phase_history_a.iter()
        .zip(phase_history_b.iter())
        .fold((0.0, 0.0), |(c, s), (a, b)| {
            let diff = a - b;
            (c + cos_2pi(diff), s + sin_2pi(diff))
        });
    ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt()
}
```

### 6.3 Diagnostic Output

Extend the benchmark JSON:

```json
{
  "oscillatory": {
    "enabled": true,
    "global_order_parameter": 0.45,
    "per_cluster_order_parameter": [0.82, 0.91, 0.34, 0.78, 0.15],
    "mean_frequency": 0.048,
    "frequency_std": 0.012,
    "mean_plv_within_cluster": 0.73,
    "mean_plv_between_cluster": 0.21,
    "phase_reset_count": 1247
  }
}
```

**Key diagnostic:** `mean_plv_within_cluster` should be high (morphons in a cluster synchronize). `mean_plv_between_cluster` should be low (different clusters oscillate independently). If both are high, coupling is too strong (global synchrony — useless). If both are low, coupling is too weak (no synchronization — oscillations are decorative).

---

## 7. Performance Considerations

### 7.1 Cost Per Tick

Phase update is O(E) where E is the number of incoming synapses per morphon (same as voltage integration). The Kuramoto coupling sum iterates over the same edges that spike delivery iterates over — it can be computed alongside spike integration with negligible additional cost.

The sin/cos computation uses the Bhaskara I approximation (two multiplies, one divide) instead of libm sin/cos (Taylor series, ~50ns). At 2000 morphons × 5 internal steps × 10 medium ticks, that's 100K phase updates per medium tick — roughly 0.5ms with the fast approximation.

### 7.2 Memory Cost

One f64 per morphon (`phase`): 8 bytes × 2000 = 16 KB.
One f64 per morphon (`natural_frequency`): 8 bytes × 2000 = 16 KB.
Optional phase history ring buffer (for PLV): 100 × 8 bytes × 2000 = 1.6 MB (only for diagnostics, not required for core computation).

### 7.3 Pulse Kernel Lite Integration

If PKL is implemented, `phase` and `natural_frequency` become hot array fields alongside voltage and threshold:

```rust
struct HotArrays {
    voltage: Vec<f64>,
    threshold: Vec<f64>,
    fired: Vec<bool>,
    refractory: Vec<f64>,
    phase: Vec<f64>,           // NEW
    natural_freq: Vec<f64>,    // NEW
    threshold_offset: Vec<f64>, // NEW (phase-dependent modulation)
}
```

Cache-friendly layout. Phase update is SIMD-friendly (uniform operation across all morphons).

---

## 8. Implementation Plan

| Step | What | Lines | Depends on |
|---|---|---|---|
| 1 | `OscillatorConfig` struct with all defaults 0/false | 30 | Nothing |
| 2 | Add `phase`, `natural_frequency` to Morphon struct | 5 | Step 1 |
| 3 | `assign_natural_frequency()` called in Morphon::new() | 5 | Step 1 |
| 4 | `sin_2pi()` / `cos_2pi()` fast approximation functions | 15 | Nothing |
| 5 | `collect_phase_neighbors()` — gather phases from incoming synapses | 15 | Step 2 |
| 6 | `update_phase()` — Kuramoto update + threshold modulation + spike reset | 20 | Steps 2, 4, 5 |
| 7 | Integrate into ResonanceEngine: call `update_phase()` before spike delivery | 10 | Step 6 |
| 8 | Modify threshold check: `threshold + effective_threshold_offset` | 3 | Step 6 |
| 9 | `kuramoto_order_parameter()` diagnostic | 10 | Step 2 |
| 10 | Add OscillatorConfig to SystemConfig, wire defaults | 5 | Step 1 |
| 11 | Validate: CartPole unchanged (oscillators disabled by default) | Run | Step 10 |
| 12 | Test: enable oscillators on CartPole, verify no regression | Run | Step 10 |
| 13 | Test: enable oscillators on MNIST, measure order parameter and accuracy | Run | Step 10 |

**Total: ~120 lines of Rust. 4–6 hours including testing.**

---

## 9. Validation Plan

### 9.1 No-Regression Test

With `enabled: false` (default), all existing benchmarks produce identical results. The phase fields exist but are never updated. Zero computational overhead when disabled.

### 9.2 CartPole with Oscillations

Enable oscillations with moderate coupling:

```rust
oscillator: OscillatorConfig {
    enabled: true,
    coupling_strength: 0.05,
    threshold_modulation_depth: 0.1,
    base_natural_frequency: 0.05,
    frequency_spread: 0.01,
    ..Default::default()
}
```

**Expected:** CartPole still solves (avg ≥ 195). Order parameter should be moderate (0.3–0.7) within clusters, low (0.1–0.3) between clusters. If CartPole regresses, reduce `threshold_modulation_depth` first — the phase should modulate timing, not prevent firing.

### 9.3 MNIST with Oscillations

Same config as CartPole. Run after metabolic pressure fix is validated.

**Expected:** If local inhibition + metabolic pressure have created diverse features, oscillations should improve binding — the order parameter within feature clusters should correlate with per-class accuracy. Classes where morphons synchronize strongly should have higher accuracy.

### 9.4 Synthetic Temporal Task

A simple task that COD should solve that the current system cannot: **temporal order detection.**

Present two stimuli A and B separated by 5 ticks. Task: detect "A then B" vs "B then A." Without oscillations, the LIF integration loses the order information (both produce similar voltage trajectories). With oscillations, the phase at which B arrives depends on whether A came first (A resets the phase, so B arrives at a specific phase) or B came first (different phase trajectory).

This is the temporal processing capability that the temporal sequence spec targets — but achievable through oscillatory dynamics without explicit delay chains or working memory.

---

## 10. Relationship to Other Planned Features

| Feature | Interaction | When |
|---|---|---|
| **Local inhibition (Phase 1)** | Prerequisite. Inhibitory interneurons provide the anti-phase coupling that creates gamma-like oscillation cycles. | Before COD |
| **Metabolic pressure** | Independent. Metabolic selection kills hubs; oscillations bind survivors. | Before COD |
| **Astrocytic gate (Phase 2A+)** | Orthogonal. Gate modulates plasticity; oscillations modulate timing. | Can be parallel |
| **Temporal processing** | COD provides the temporal substrate. Context feedback and multi-timescale traces build on top of phase-coded temporal information. | After COD |
| **DeMorphon (Phase 7)** | Internal DeMorphon synchronization uses strong phase coupling. Body plan includes phase relationships between roles. | After COD |
| **MorphonGenome (Phase 6)** | `natural_frequency` becomes a heritable genome field. | After COD |
| **Multi-instance (Phase 11)** | Inter-system social learning transmits phase patterns (resonance frequencies) rather than weights. | Long-term |
| **Hardware (Phase 12)** | Oscillatory dynamics are naturally parallel — each morphon updates its phase independently. FPGA implementation is straightforward. | Long-term |

---

## 11. For the Paper

### 11.1 Current Paper (v1/v2)

Mention in future work: "The Resonance Engine's spike delivery infrastructure naturally supports Kuramoto-type phase coupling between morphons, enabling binding-by-synchrony for temporal feature integration. We plan to implement this following validation of the metabolic selection and local inhibition systems."

### 11.2 Future Paper

If oscillations measurably improve MNIST accuracy or enable temporal tasks, it becomes a section in the Endoquilibrium paper (Frontiers / Neural Computation): "Coupled Oscillatory Dynamics: Phase-Coupled Spike Timing in Morphogenic Intelligence — integrating the EI-Kuramoto model (Iwase et al. 2025) with structural plasticity and neuroendocrine regulation."

---

## 12. References

Buzsáki, G. (2006). Rhythms of the Brain. Oxford University Press.

Fries, P. (2005). A mechanism for cognitive dynamics: neuronal communication through neuronal coherence. Trends in Cognitive Sciences, 9(10), 474–480.

Heltberg, M., Jensen, M. H., & Nielsen, M. S. (2024). Huygens synchrony in human cells. University of Copenhagen / Niels Bohr Institute.

Iwase, M. et al. (2025). Excitation–Inhibition Balance Controls Synchronization in a Simple Model of Coupled Phase Oscillators. Neural Computation, 37(7), 1353–1385.

Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence. Springer.

Kusch, L. et al. (2025). Synchronization in spiking neural networks with short and long connections and time delays. Chaos, 35(1), 013161.

Singer, W. & Gray, C. M. (1995). Visual Feature Integration and the Temporal Correlation Hypothesis. Annual Review of Neuroscience, 18, 555–586.

Tass, P. A. et al. (2022). Dynamics of phase oscillator networks with synaptic weight and structural plasticity. Scientific Reports, 12, 15003.

---

*Coupled Oscillatory Dynamics — when morphons learn to swing together, intelligence finds its rhythm.*

*TasteHub GmbH, Wien, April 2026*
