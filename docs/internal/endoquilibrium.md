# Endoquilibrium

Predictive neuroendocrine regulation for MORPHON. Maintains network health by sensing vital signs, learning normal trajectories via dual-timescale EMAs, and adjusting 6 levers through proportional control. Biological analogy: the endocrine system (allostasis), not the nervous system (homeostasis).

**Module**: `src/endoquilibrium.rs`
**Runs on**: Medium path tick (same as learning)
**Never modifies**: weights, synapses, topology (only modulates the environment)

---

## The Problem It Solves

Every failure mode discovered during MORPHON development traces to unregulated dynamics:

| Failure Mode | What Happens | Static Param That Failed |
|---|---|---|
| FR deadlock | Associative morphons stop firing (0%) | Threshold too high for this network |
| LTD vicious cycle | Weak synapses get weaker until death | STDP `a_minus` too aggressive |
| Weight saturation | All weights hit `weight_max` | Learning rate too high |
| Modulatory explosion | One cell type dominates (49%+) | No feedback on differentiation |
| Eligibility starvation | Learning stalls, no active traces | Novelty channel insufficient |

Static hyperparameters can't adapt. What's optimal at episode 10 is wrong at episode 500. Endoquilibrium replaces static values with dynamic regulation: the learning rate isn't `0.05` -- it's "whatever keeps weight entropy between 2.0 and 3.5."

---

## Architecture

```
              GOVERNOR (V3)
      hard limits, constitutional constraints
   ─────────────────────────────────────────
           ENDOQUILIBRIUM (this)
   sense vitals → predict → regulate channels
   ─────────────────────────────────────────
              BUILDER (V1/V2)
       morphon network, learning, morphogenesis
```

The Governor sets hard limits ("never exceed 300 morphons"). Endoquilibrium sets soft targets ("maintain 8-12% firing rate") and adjusts modulation channels to stay on target. The Governor can override Endoquilibrium but never the reverse.

---

## The 7 Vital Signs

Sensed every medium-path tick via `sense_vitals()` -- a free function taking immutable refs to avoid borrow conflicts with `&mut self.endo` in `System::step()`.

| Vital | Source | Healthy Range |
|---|---|---|
| `fr_sensory/associative/motor/modulatory` | `Diagnostics::firing_by_type` | 8-15% per type |
| `eligibility_density` | `diag.eligibility_nonzero_count / total_synapses` | 20-60% |
| `weight_entropy` | Shannon entropy of weight distribution (20 bins) | 2.0-4.5 bits |
| `cell_type_fractions` | Count per CellType / total | Per config targets |
| `energy_utilization` | `1.0 - avg_energy` | 0.3-0.7 |
| `tag_count / capture_count` | `diag.active_tags`, `diag.captures_this_step` | captures > 2% of tags |
| `prediction_error_mean` | Mean PE across all morphons | Task-dependent |

Weight entropy computation is O(S) -- two passes over topology edges (find range, then bin). Everything else reuses values already in `Diagnostics`.

---

## The Allostasis Predictor

Dual-timescale EMAs that learn what "normal" looks like for this network:

- **Fast EMA** (tau=50 ticks): Tracks acute changes. Responds to sudden FR drops.
- **Slow EMA** (tau=500 ticks): Tracks developmental trajectory. The network's "personality."

Regulation error = `(fast_ema - setpoint)`. The fast EMA is used for reactive regulation: it responds to what's happening now, while the slow EMA informs stage detection.

### Developmental Stage Detection

Based on first derivatives of morphon count and PE trend:

| Stage | Condition | Effect |
|---|---|---|
| Proliferating | morphon count rising >1% per window | Higher FR targets, more novelty |
| Differentiating | some structural change, count stable | Medium FR targets |
| Consolidating | morphon count falling | Lower FR targets, higher homeostasis |
| Mature | all derivatives near zero | Lowest setpoints, stability mode |
| Stressed | PE trend positive for 100+ ticks | Wider ranges, damage control |

Setpoints shift with stage (table in `DevelopmentalSetpoints::for_stage()`). This prevents consolidation-phase regulation from fighting proliferation-phase behavior.

---

## The 6 Levers

Endoquilibrium acts through these actuators. All are smoothed via EMA (alpha=0.1) and hard-clamped to prevent extremes.

| Lever | Range | Default | Biological Analog |
|---|---|---|---|
| `reward_gain` | [0.1, 3.0] | 1.0 | Dopamine tonic level |
| `novelty_gain` | [0.0, 2.0] | 1.0 | Acetylcholine tonic level |
| `arousal_gain` | [0.1, 2.0] | 1.0 | Noradrenaline tonic level |
| `homeostasis_gain` | [0.3, 2.0] | 1.0 | Serotonin tonic level |
| `threshold_bias` | [-0.3, 0.3] | 0.0 | Cortisol (excitability) |
| `plasticity_mult` | [0.1, 5.0] | 1.0 | BDNF (learning rate scaling) |

---

## The 6 Regulation Rules

Each rule is proportional control with configurable coefficients (in `EndoConfig`).

### Rule 1: Firing Rate (most critical)

```
if fr_associative < setpoint.min:
    threshold_bias  -= deficit * 0.5   // lower thresholds
    arousal_gain    += deficit * 0.3   // increase sensitivity
    novelty_gain    += deficit * 0.2   // increase exploration

if fr_associative > setpoint.max:
    threshold_bias  += excess * 0.8   // raise thresholds fast
    homeostasis_gain += excess * 0.5   // dampen activity
```

This is the rule that breaks the FR=0% deadlock. At the default fast_tau of 50, Endoquilibrium detects a dead layer within ~50 ticks and progressively lowers thresholds until firing restarts.

### Rule 2: Eligibility Density

Low eligibility = stalled learning. Boosts novelty and plasticity_mult. High eligibility = everything changing at once. Boosts homeostasis, dampens plasticity.

### Rule 3: Weight Distribution Health

Monitors Shannon entropy of the weight distribution. Collapse (entropy < min) triggers the LTD vicious cycle detector: boosts novelty and plasticity to inject diversity. Explosion (entropy > max) dampens learning.

### Rule 4: Cell Type Balance

Logs when any cell type exceeds its target fraction + 15%. Currently logging-only (no automatic morphogenesis pressure).

### Rule 5: Tag-and-Capture Health

If tags are forming (>100) but no captures happening, the reward signal isn't reaching tagged synapses. Boosts `reward_gain` by 1.5x.

### Rule 6: Energy Pressure (from ANCS F7)

Three-tier escalation:

| Level | Threshold | Response |
|---|---|---|
| Pressure | energy > 0.70 | `plasticity_mult *= 0.7` |
| Emergency | energy > 0.85 | `plasticity_mult *= 0.3`, `novelty *= 0.2` |
| Critical | energy > 0.95 | `plasticity_mult = 0`, `novelty = 0`, `homeostasis = 2.0` |

---

## Integration Points

### Where gains are applied (read site, not injection site)

The `inject_*` methods on `Neuromodulation` are unchanged. Gains multiply at the point where modulation is consumed:

1. **Channel gains in learning** (`learning.rs:apply_weight_update`):
   ```rust
   // Receptor-gated signal scaled by Endo gain
   let r = alpha_reward * modulation.reward_delta() * channel_gains[0];
   let n = alpha_novelty * modulation.novelty * channel_gains[1];
   // ...
   ```
   Passed as `[f64; 4]` from `system.rs`.

2. **Plasticity multiplier** (`system.rs`, two sites):
   ```rust
   // Three-factor path
   let plasticity = modulation.plasticity_rate() * plast_rate * endo.channels.plasticity_mult;
   // DFA path
   let dfa_lr = 0.02 * plast_rate * endo.channels.plasticity_mult;
   ```

3. **Threshold bias** (`morphon.rs:Morphon::step`):
   ```rust
   self.fired = activation > (self.threshold + threshold_bias) && self.energy > 0.0;
   ```
   Passed as `f64` parameter. Keeps `morphon.threshold` clean for its own homeostatic regulation. Bias is transient -- disable Endo and it vanishes.

### Processing order in `System::step()`

```
FAST PATH
├── spike propagation
├── k-WTA
├── morphon.step(threshold_bias)      ← bias applied here
├── DFA feedback injection
│
MEDIUM PATH
├── Endoquilibrium.tick(vitals)       ← sense, predict, regulate
├── STDP + weight updates             ← plasticity_mult + channel_gains applied here
├── weight normalization
├── compensatory plasticity
```

Endoquilibrium runs **before** the learning loop so gains are current when weight updates happen.

---

## Configuration

```rust
SystemConfig {
    endoquilibrium: EndoConfig {
        enabled: true,          // master switch (default: false)
        fast_tau: 50.0,         // fast EMA time constant
        slow_tau: 500.0,        // slow EMA time constant
        smoothing_alpha: 0.1,   // channel adjustment smoothing
        // Rule coefficients (all configurable, see EndoConfig for full list)
        ..Default::default()
    },
    ..Default::default()
}
```

When `enabled: false` (default), all channels stay at neutral values (`gain=1.0`, `bias=0.0`, `mult=1.0`). Existing behavior is identical. All `EndoConfig` fields use `#[serde(default)]` for backward-compatible deserialization.

---

## Diagnostics

`system.endo.summary()` returns a one-line log string:

```
endo: stage=Mature rg=1.50 ng=1.40 ag=1.00 hg=1.32 tb=0.039 pm=1.20 hp=0.72
```

`system.endo.last_diag` contains:
- `stage`: Current developmental stage
- `channels`: Current channel state
- `interventions`: Vec of every rule that fired this tick (rule name, vital, actual/setpoint, lever, adjustment)
- `health_score`: Composite 0-1 score (1.0 = all vitals within setpoints)

---

## Snapshot Serialization

Endoquilibrium state is included in `SystemSnapshot` with `#[serde(default)]`. Old snapshots without the `endo` field deserialize with `Endoquilibrium::default()` (disabled, neutral channels).

---

## Measured Impact

CartPole benchmark with Endoquilibrium enabled vs disabled (v0.5.0):

| Metric | Without Endo | With Endo |
|---|---|---|
| avg(100) | ~10 steps | ~21 steps |
| best | 57 steps | 78 steps |
| Firing rate | 0% - 10% (unstable) | ~16% (stable) |
| Eligibility density | variable | 30-40% (stable) |

The primary improvement is breaking the FR=0% deadlock. The system is alive and learning; the remaining performance gap to the 195-step target is a credit assignment problem (see `findings.md`), not a regulation problem.

---

## What Endoquilibrium Is Not

- **Not a second learning system.** It doesn't learn task-specific representations. It maintains conditions under which the Builder can learn. It's a thermostat, not a brain.
- **Not a replacement for the Governor.** Governor = hard limits. Endoquilibrium = soft dynamic regulation within those limits.
- **Not a neural network.** Proportional control with EMAs. Deliberately simple, deterministic, debuggable. The complexity lives in the Builder, not in the regulator.
