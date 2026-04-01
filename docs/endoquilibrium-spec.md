# Endoquilibrium
## Predictive Neuroendocrine Regulation Engine for MORPHON
### Technical Specification v1.0 — TasteHub GmbH, April 2026

---

| | |
|---|---|
| **MORPHON Layer** | Runtime — sits between Builder and Governor |
| **Language** | Rust (morphon-core crate) |
| **Status** | Implementation Specification |

---

## 1. What Endoquilibrium Is

Endoquilibrium is a predictive regulation engine for MORPHON that maintains network health by sensing vital signs, learning what "normal" looks like for this specific network at this developmental stage, and proactively adjusting neuromodulatory channels before instabilities occur.

The biological analogy is the **endocrine system** — not the nervous system. The nervous system reacts fast to specific stimuli. The endocrine system maintains systemic equilibrium through slow, predictive, hormone-mediated regulation. It learns circadian rhythms. It anticipates stress. It adjusts cortisol before you wake up, not after. Endoquilibrium does the same for MORPHON: it watches aggregate network state, learns the rhythms of healthy morphogenesis, and adjusts modulation channels *before* the network destabilizes.

The technical term for this is **allostasis** — predictive regulation that maintains stability through change, as opposed to homeostasis which is reactive regulation toward a fixed setpoint. MORPHON's network is constantly changing (growing, pruning, migrating, fusing). A fixed setpoint doesn't work. Endoquilibrium learns the *trajectory* of healthy development and regulates toward that trajectory.

### 1.1 The Problem It Solves

Every failure mode discovered during MORPHON development traces back to unregulated dynamics:

- **Modulatory Explosion:** One cell type dominates because division is unchecked — no hormonal signal says "stop dividing, you're 49% of the population."
- **Eligibility Starvation:** Firing rate drops to 0% because thresholds are too high — no system lowers them proactively when activity drops.
- **Weight Saturation:** STDP parameters are too aggressive — no system detects the weight distribution narrowing and dampens plasticity.
- **LTD Vicious Cycle:** Weak synapses get weaker until they die — no system detects the collapse and intervenes.
- **CMA-ES Dependence:** Static hyperparameters can't adapt to changing network state — what's optimal at episode 10 is wrong at episode 500.

Endoquilibrium replaces *static hyperparameters* with *dynamic regulation*. The learning rate isn't 0.05 — it's "whatever keeps the weight distribution entropy above 2.0 and below 4.0." The firing threshold isn't 0.5 — it's "whatever maintains 8–12% aggregate firing rate for this cell type at this developmental stage."

---

## 2. Architecture

### 2.1 Where It Sits

Endoquilibrium sits between the Builder (the morphon network itself) and the Governor (V3 constitutional constraints). It's a continuous control loop that runs on the Medium Path of the Dual-Clock architecture (~every 10ms, same as synaptic plasticity):

```
┌───────────────────────────────────────────┐
│              GOVERNOR (V3)                │
│  Constitutional constraints, hard limits  │
├───────────────────────────────────────────┤
│          ENDOQUILIBRIUM (NEW)             │
│  Predictive regulation, dynamic setpoints │
│  Senses vitals → Predicts → Adjusts      │
├───────────────────────────────────────────┤
│           BUILDER (V1/V2)                 │
│  Morphon network, learning, morphogenesis │
└───────────────────────────────────────────┘
```

The Governor sets hard limits ("never exceed 10,000 morphons"). Endoquilibrium sets soft targets ("maintain 300–400 morphons at current task complexity") and adjusts the four modulation channels to stay on target. The Governor can override Endoquilibrium but never the reverse.

### 2.2 Core Struct

```rust
pub struct Endoquilibrium {
    // === VITAL SIGNS (sensed every tick) ===
    vitals: VitalSigns,
    vitals_history: RingBuffer<VitalSigns, 1000>,

    // === PREDICTIVE MODEL ===
    predictor: AllostasisPredictor,

    // === ACTUATORS (modulation channels) ===
    channels: ChannelState,

    // === LEARNED SETPOINTS ===
    setpoints: DevelopmentalSetpoints,

    // === CONFIG ===
    config: EndoConfig,
}
```

---

## 3. Vital Signs — What Endoquilibrium Senses

Every Medium Path tick, Endoquilibrium reads seven vital signs from the network. These are aggregate statistics — cheap to compute (O(N) scan of the morphon array) and sufficient for systemic regulation.

### 3.1 The Seven Vitals

| Vital | What It Measures | Healthy Range | Failure If Out |
|---|---|---|---|
| `firing_rate_by_type` | % of morphons that fired, per CellType | 8–15% per type | 0% = dead layer, >30% = epilepsy |
| `eligibility_density` | % of synapses with \|eligibility\| > 0.01 | 20–60% | <5% = learning stalled |
| `weight_entropy` | Shannon entropy of weight distribution | 2.0–4.5 bits | <1.5 = collapsed, >5.0 = noise |
| `cell_type_balance` | Fraction per CellType (S/A/M/Mod) | Per config targets | Any type >50% = explosion |
| `energy_budget` | Total energy pool utilization (0–1) | 0.3–0.7 | >0.85 = pressure, <0.1 = waste |
| `tag_capture_rate` | Captures per 100 tags | >2% | 0% = consolidation broken |
| `prediction_error_mean` | Avg PE across all morphons | Task-dependent | Rising trend = destabilization |

### 3.2 Vital Signs Struct

```rust
pub struct VitalSigns {
    pub timestamp: u64,

    // Per-type firing rates
    pub fr_sensory: f32,
    pub fr_associative: f32,
    pub fr_motor: f32,
    pub fr_modulatory: f32,

    // Learning pipeline health
    pub eligibility_density: f32,   // fraction with |e| > 0.01
    pub weight_entropy: f32,        // Shannon entropy of |w| distribution
    pub tag_count: u32,
    pub capture_count: u32,

    // Structural health
    pub cell_type_fractions: [f32; 4],  // S, A, M, Mod
    pub total_morphons: u32,
    pub total_synapses: u32,
    pub cluster_count: u32,

    // Metabolic health
    pub energy_utilization: f32,

    // Task performance
    pub prediction_error_mean: f32,
    pub prediction_error_trend: f32,  // slope over last 100 ticks
    pub reward_rate: f32,             // avg reward per episode
}
```

Computing vitals is cheap: a single O(N) pass over the morphon array plus an O(S) pass over the synapse array, where N≈300 and S≈10,000. At the Medium Path tick rate (~every 10ms), this adds <0.1ms of overhead.

---

## 4. The Allostasis Predictor

The predictor is the core intelligence of Endoquilibrium. It learns what "healthy" looks like for this specific network at this developmental stage, and predicts what vitals should be N ticks from now. Deviations between predicted and actual vitals drive regulation.

### 4.1 Design Choice: Exponential Moving Averages, Not Neural Networks

The predictor must be simple, fast, and stable. It cannot itself be a learning system that destabilizes (no turtles all the way down). The implementation uses exponential moving averages (EMAs) at two timescales:

- **Fast EMA (τ = 50 ticks):** Tracks recent network state. Responds to acute changes (sudden firing rate drop, weight saturation event).
- **Slow EMA (τ = 500 ticks):** Tracks developmental trajectory. Represents the network's "personality" — what normal looks like for this organism after growing for 500 ticks.

The predicted vital for the next tick is the slow EMA. The actual vital is the current measurement. The regulation error is the difference, weighted by urgency:

```
regulation_error = (actual - slow_ema) * urgency_weight

// urgency_weight is higher when fast_ema is diverging
// from slow_ema (acute crisis vs normal fluctuation)
urgency = |fast_ema - slow_ema| / slow_ema
```

### 4.2 Predictor Struct

```rust
pub struct AllostasisPredictor {
    // Dual-timescale EMAs for each vital
    fast_emas: VitalSigns,  // tau = 50
    slow_emas: VitalSigns,  // tau = 500

    // Learned urgency weights per vital
    urgency_weights: [f32; 7],

    // Developmental stage detection
    stage: DevelopmentalStage,
    stage_transition_history: Vec<(u64, DevelopmentalStage)>,
}

pub enum DevelopmentalStage {
    Proliferating,   // Network is growing (morphon count rising)
    Differentiating, // Cell types are specializing
    Consolidating,   // Network is pruning and stabilizing
    Mature,          // Network has reached steady state
    Stressed,        // PE rising, structure destabilizing
}
```

Stage detection is simple: track the first derivative of morphon count, cluster count, and PE trend. Proliferating = morphon count rising >1% per 100 ticks. Consolidating = morphon count falling, cluster count stable. Mature = all derivatives near zero. Stressed = PE trend positive for >200 ticks.

### 4.3 Setpoints Adapt by Stage

The healthy ranges aren't fixed — they shift with developmental stage. A proliferating network should have high firing rates (lots of new morphons exploring). A consolidating network should have low plasticity (protecting what's been learned). Endoquilibrium adjusts its setpoints accordingly:

| Vital | Proliferating | Differentiating | Consolidating | Mature |
|---|---|---|---|---|
| Firing rate target | 12–18% | 10–15% | 8–12% | 8–12% |
| Eligibility density | 40–70% | 30–60% | 20–40% | 15–35% |
| Weight entropy | 3.0–4.5 | 2.5–4.0 | 2.0–3.5 | 2.0–3.5 |
| Novelty channel | High (0.6–0.8) | Medium (0.3–0.5) | Low (0.1–0.3) | Low (0.1–0.2) |
| Homeostasis channel | Low (0.2–0.4) | Medium (0.4–0.6) | High (0.6–0.8) | High (0.7–0.9) |

---

## 5. Actuators — How Endoquilibrium Acts

Endoquilibrium acts exclusively through the four existing neuromodulatory channels plus two new meta-parameters. It never modifies weights, morphons, or synapses directly — it modulates the environment in which the Builder operates.

### 5.1 The Six Levers

| Lever | What It Adjusts | Biological Analog |
|---|---|---|
| Reward channel gain | Amplitude of reward modulation signal | Dopamine tonic level |
| Novelty channel gain | Amplitude of novelty/plasticity signal | Acetylcholine tonic level |
| Arousal channel gain | Amplitude of arousal/sensitivity signal | Noradrenaline tonic level |
| Homeostasis channel gain | Amplitude of stability signal | Serotonin tonic level |
| Global threshold bias | Offset added to all firing thresholds | Cortisol (raises/lowers excitability) |
| Plasticity rate multiplier | Scales all learning rates | BDNF (brain-derived neurotrophic factor) |

```rust
pub struct ChannelState {
    pub reward_gain: f32,      // default 1.0, range [0.1, 3.0]
    pub novelty_gain: f32,     // default 1.0, range [0.0, 2.0]
    pub arousal_gain: f32,     // default 1.0, range [0.1, 2.0]
    pub homeostasis_gain: f32, // default 1.0, range [0.3, 2.0]
    pub threshold_bias: f32,   // default 0.0, range [-0.3, 0.3]
    pub plasticity_mult: f32,  // default 1.0, range [0.1, 5.0]
}
```

### 5.2 Regulation Rules

Each vital sign maps to one or more levers through simple proportional control with clamping. The rules are explicit and deterministic — no neural network, no learned mapping, no ambiguity:

#### Rule 1: Firing Rate Regulation

```rust
// If Associative firing rate is too low:
if vitals.fr_associative < setpoints.fr_assoc_min {
    let deficit = setpoints.fr_assoc_min - vitals.fr_associative;
    channels.threshold_bias -= deficit * 0.5;     // lower thresholds
    channels.arousal_gain += deficit * 0.3;       // increase sensitivity
    channels.novelty_gain += deficit * 0.2;       // increase exploration
}
// If firing rate is too high (pre-epileptic):
if vitals.fr_associative > setpoints.fr_assoc_max {
    let excess = vitals.fr_associative - setpoints.fr_assoc_max;
    channels.threshold_bias += excess * 0.8;      // raise thresholds fast
    channels.homeostasis_gain += excess * 0.5;    // dampen activity
}
```

#### Rule 2: Eligibility Density Regulation

```rust
// If eligibility is too low (learning stalled):
if vitals.eligibility_density < setpoints.elig_min {
    channels.novelty_gain += 0.2;     // boost plasticity
    channels.plasticity_mult *= 1.2;  // amplify learning rates
}
// If eligibility is too high (everything changing at once):
if vitals.eligibility_density > setpoints.elig_max {
    channels.homeostasis_gain += 0.3;  // stabilize
    channels.plasticity_mult *= 0.8;   // dampen learning
}
```

#### Rule 3: Weight Distribution Health

```rust
// If weight entropy is collapsing (all weights converging):
if vitals.weight_entropy < setpoints.entropy_min {
    channels.novelty_gain += 0.4;       // inject diversity
    channels.plasticity_mult *= 1.5;    // amplify exploration
    // This is the LTD vicious cycle detector
}
// If weight entropy is exploding (weights are random noise):
if vitals.weight_entropy > setpoints.entropy_max {
    channels.plasticity_mult *= 0.5;    // dampen all learning
    channels.homeostasis_gain += 0.4;   // heavy stabilization
}
```

#### Rule 4: Cell Type Balance

```rust
// If any cell type exceeds its target fraction + 15%:
for (cell_type, fraction) in vitals.cell_type_fractions {
    let target = setpoints.type_targets[cell_type];
    if fraction > target + 0.15 {
        // Inject differentiation pressure on overrepresented type
        system.inject_differentiation_pressure(
            cell_type,
            strength: (fraction - target) * 2.0
        );
        // This prevents Modulatory Explosion
    }
}
```

#### Rule 5: Tag-and-Capture Health

```rust
// If tags are forming but no captures happening:
if vitals.tag_count > 100 && vitals.capture_count == 0 {
    // The reward signal isn't reaching tagged synapses in time
    channels.reward_gain *= 1.5;   // amplify reward signal
    // Also check: is the capture threshold too high?
    if self.ticks_since_last_capture > 500 {
        config.capture_threshold *= 0.8;  // lower the bar
    }
}
```

#### Rule 6: Energy Pressure (from ANCS F7)

```rust
match vitals.energy_utilization {
    e if e > 0.95 => {  // Critical: Safe Mode
        channels.plasticity_mult = 0.0;    // freeze all learning
        channels.novelty_gain = 0.0;        // no exploration
        channels.homeostasis_gain = 2.0;    // maximum stability
    },
    e if e > 0.85 => {  // Emergency
        channels.plasticity_mult *= 0.3;
        channels.novelty_gain *= 0.2;
    },
    e if e > 0.70 => {  // Pressure
        channels.plasticity_mult *= 0.7;
    },
    _ => {}  // Normal: no adjustment
}
```

### 5.3 Channel Smoothing and Clamping

All channel adjustments are smoothed through an EMA to prevent oscillation, and hard-clamped to prevent extreme values:

```rust
impl ChannelState {
    fn apply_and_smooth(&mut self, adjustments: &ChannelState) {
        let alpha = 0.1;  // smoothing factor
        self.reward_gain = (self.reward_gain * (1.0 - alpha)
            + adjustments.reward_gain * alpha)
            .clamp(0.1, 3.0);
        self.novelty_gain = (self.novelty_gain * (1.0 - alpha)
            + adjustments.novelty_gain * alpha)
            .clamp(0.0, 2.0);
        // ... same for all channels
    }
}
```

The smoothing factor α=0.1 means adjustments take ~10 ticks to fully apply. This prevents the regulation from oscillating: a sudden firing rate drop doesn't slam all thresholds down instantly, it gradually lowers them over 100ms. The biological endocrine system operates on similar timescales — cortisol takes minutes to peak after a stress signal.

---

## 6. Integration with Existing MORPHON Systems

### 6.1 Integration Points

Endoquilibrium connects to the existing MORPHON architecture at four points:

- **Neuromodulation API (V1):** Endoquilibrium sets the gain on each of the four modulation channels. The existing `inject_reward()`, `inject_novelty()`, `inject_arousal()` functions are multiplied by the Endoquilibrium gain before broadcasting. Zero code change to the modulation system — just a multiplier.
- **Morphon threshold (V1):** The `threshold_bias` is added to each morphon's adaptive threshold after the morphon's own homeostatic adjustment. Endoquilibrium provides a global offset; individual morphons still do local adaptation.
- **Learning rate (V1):** The `plasticity_mult` scales all weight update deltas before they're applied. The existing `apply_weight_update()` function multiplies Δw by this factor. One-line change.
- **Constitutional Guards (V3):** Endoquilibrium reads the Governor's constraints and ensures its adjustments never violate them. If the Governor says `min_morphons=100`, Endoquilibrium won't boost apoptosis when morphon count is at 105.

### 6.2 Where It Runs in the Processing Loop

```rust
// In system.process() — the main per-step function:

fn process(&mut self, input: &[f64]) -> Vec<f64> {
    // 1. Fast Path: spike propagation (existing)
    self.feed_input(input);
    for _ in 0..self.config.internal_steps {
        self.propagate();
        self.deliver();
        self.step_morphons();
    }
    let output = self.read_output();

    // 2. Medium Path: learning + ENDOQUILIBRIUM (every N steps)
    if self.tick % self.config.medium_path_interval == 0 {
        // 2a. Sense vitals
        let vitals = self.endo.sense(&self);

        // 2b. Predict and compute regulation error
        let errors = self.endo.predict_and_compare(&vitals);

        // 2c. Compute channel adjustments
        let adjustments = self.endo.regulate(&errors);

        // 2d. Apply (smoothed, clamped, governor-checked)
        self.endo.apply(&adjustments, &self.governor);

        // 2e. Apply existing learning rules with Endo gains
        self.apply_stdp(self.endo.channels.plasticity_mult);
        self.apply_dfa(self.endo.channels.plasticity_mult);
    }

    // 3. Slow Path: morphogenesis (existing, less frequent)
    // ...

    output
}
```

---

## 7. Replacing Static Hyperparameters

Endoquilibrium makes several CMA-ES parameters unnecessary. Instead of searching for optimal fixed values, the regulation engine adapts them in real time:

| CMA-ES Parameter | Endoquilibrium Replacement | Adaptation Mechanism |
|---|---|---|
| `learning_rate` | `plasticity_mult × base_rate` | Scales based on weight entropy and eligibility density |
| `a_plus / a_minus` | `plasticity_mult × base_stdp` | Same multiplier, prevents weight saturation |
| `threshold` | `morphon.threshold + threshold_bias` | Regulated by per-type firing rate targets |
| `novelty_strength` | `novelty_gain × base_novelty` | High during proliferation, low during consolidation |
| `reward_strength` | `reward_gain × base_reward` | Boosted when tags exist but captures don't |

CMA-ES is still useful for finding the *base values* and the *regulation rule parameters* (proportional gains, smoothing factors, setpoint ranges). But the search space shrinks from 15+ dimensions to ~6 meta-parameters, and the fitness landscape becomes smoother because Endoquilibrium compensates for parameter sensitivity.

---

## 8. Diagnostics Dashboard

Endoquilibrium exposes its state through a diagnostics API that feeds into MORPHON Studio and CLI logging:

```rust
pub struct EndoDiagnostics {
    pub vitals: VitalSigns,
    pub stage: DevelopmentalStage,
    pub channels: ChannelState,
    pub regulation_errors: [f32; 7],  // per vital
    pub interventions_this_tick: Vec<Intervention>,
    pub health_score: f32,  // 0–1, composite network health
}

pub struct Intervention {
    pub rule: &'static str,  // "firing_rate_low"
    pub vital: &'static str, // "fr_associative"
    pub actual: f32,
    pub setpoint: f32,
    pub lever: &'static str, // "threshold_bias"
    pub adjustment: f32,     // -0.05
}
```

Every intervention is logged with what triggered it, what lever was pulled, and by how much. This is the "endocrinology report" — you can look at the log and see: *"At tick 4500, fr_associative was 2.1% (target 8–12%), so threshold_bias was lowered by 0.03 and arousal_gain was increased by 0.006."* Full explainability for every regulation decision.

---

## 9. Implementation Plan

| Phase | What | Effort | Depends On |
|---|---|---|---|
| Phase 1 | VitalSigns struct + sensing (O(N) scan per tick) | 2–3 hours | Nothing — start here |
| Phase 2 | AllostasisPredictor with dual EMAs | 2–3 hours | Phase 1 |
| Phase 3 | ChannelState + 6 regulation rules | 3–4 hours | Phase 2 |
| Phase 4 | Integration into process() loop | 1–2 hours | Phase 3 |
| Phase 5 | DevelopmentalStage detection | 2–3 hours | Phase 2 |
| Phase 6 | Diagnostics API + logging | 2–3 hours | Phase 4 |
| Phase 7 | CMA-ES re-run with Endo in loop | Compute time | Phase 4 |

**Total estimated implementation: 12–18 hours of Rust work.** Each phase is independently testable. Phase 1–4 gives you a working Endoquilibrium with immediate impact on training stability. Phases 5–7 add sophistication.

---

## 10. Expected Impact on Current Bottlenecks

| Current Problem | How Endoquilibrium Fixes It | Mechanism |
|---|---|---|
| Associative FR = 0% | Detects within 50 ticks, lowers thresholds until firing starts | Rule 1: threshold_bias |
| Eligibility too low (0.002–0.009) | Detects stalled learning, boosts plasticity multiplier | Rule 2: plasticity_mult |
| Weight saturation (±max) | Detects entropy collapse, dampens learning rate | Rule 3: plasticity_mult |
| Modulatory explosion (49%) | Detects imbalance, injects differentiation pressure | Rule 4: cell type balance |
| Zero captures | Detects tag/capture mismatch, boosts reward gain | Rule 5: reward_gain |
| CMA-ES sensitivity | Reduces parameter sensitivity by adapting in real time | All rules together |
| Extended run plateau | Adapts regulation as network matures (stage detection) | Stage-dependent setpoints |

The single biggest impact is on the **Associative firing rate = 0%** problem. Endoquilibrium would detect this within the first 50 ticks of training and progressively lower thresholds until the hidden layer starts firing. This alone could break the deadlock that has limited CartPole to avg=17.6.

---

## 11. What Endoquilibrium Is Not

- **Not a second learning system.** It doesn't learn task-specific representations. It maintains the conditions under which the Builder can learn. It's a thermostat, not a brain.
- **Not a replacement for the Governor.** The Governor enforces hard constitutional limits. Endoquilibrium provides soft dynamic regulation within those limits. If Endoquilibrium says "lower thresholds" but the Governor says "minimum threshold is 0.1," the Governor wins.
- **Not a neural network.** It's proportional control with EMAs. Deliberately simple, deliberately deterministic, deliberately debuggable. The complexity lives in the Builder, not in the regulator.
- **Not optional for production.** Running MORPHON without Endoquilibrium is like running a body without an endocrine system. It might work briefly with manually tuned hormones (static hyperparameters), but it can't adapt to changing conditions or recover from perturbations.

---

*Endoquilibrium — the system that keeps the organism in balance.*
*Not by reacting to crises, but by predicting them.*

*TasteHub GmbH, Wien, April 2026*
