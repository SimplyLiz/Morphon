# Metabolic Pruning — Experimental Findings

Two findings from the metabolic pressure experiments (2026-04-03/04).
Branch: `feature/local-inhibitory-competition`

---

## 1. Finding: MI Substrate Creates Discriminative Representations

### Setup

NLP Readiness Benchmark: 4-tier synthetic text classification benchmark measuring language-processing capability. Each tier tests a specific NLP prerequisite:

| Tier | Task | Input | Chance |
|------|------|-------|--------|
| 0: Bag-of-Chars | Vowel-heavy vs consonant-heavy words | 27-dim frequency | 50% |
| 1: One-Hot Scale | Same task at one-hot encoding scale | 135-dim (5 chars x 27) | 50% |
| 2: Memory | Classify by first char after 3-char sequence | 27-dim per step | 50% |
| 3: Composition | Token-pair XOR (same-group vs cross-group) | 54-dim | 50% |

Two output pathways tested:
- **Spike-based**: `teach_supervised()` / `teach_supervised_with_input()` — motor potential read through spike propagation
- **Analog readout**: `train_readout()` — direct weighted sum of sigmoid(potential), bypassing spikes

### Results

| Tier | Spike-based | Analog readout |
|------|------------|---------------|
| 0: Bag-of-Chars | 46% (chance) | **99%** |
| 1: One-Hot Scale | 51% (chance) | **69%** |
| 2: Memory | 48% (chance) | **88%** |
| 3: Composition | 50% (chance) | 40% (fails — XOR needs nonlinear hidden features) |

### Interpretation

The morphon substrate **does** form discriminative representations during spike-based processing. The information is present in morphon potentials — the analog readout proves it by reading those potentials directly.

The bottleneck is the spike-based output pathway: signals are distorted by delays, leaky integration, multi-hop propagation, and noise. Motor potential != W*x because the forward pass goes through spike conversion → propagation → integration, losing the discriminative signal at each stage.

This is the same pattern observed in MNIST: the hidden layer develops features (proven by post-hoc labeling showing class-selective neurons), but the output pathway can't exploit them reliably.

**Tier 2 at 88% is particularly significant**: the system retains information about the first character through 3 sequential `process_steps()` calls. Temporal memory exists in the substrate through residual potentials and decaying traces — it just can't be accessed through the spike pipeline. This suggests the temporal sequence processing spec (recurrent connections, context feedback) is building on a foundation that already works at the potential level.

### Implication for Architecture

The long-term fix is not to abandon spikes but to improve spike-to-output fidelity. The analog readout (Purkinje-style bypass) is biologically motivated — cerebellar Purkinje cells perform analog dendritic integration for motor output while the rest of the circuit uses spikes. The dual-speed architecture (spike-based hidden layer + analog readout) is the proven classification path.

---

## 2. Finding: Endo Premature-Mature Bug on Classification Tasks

### Symptom

MNIST accuracy plateaus at ~20-26% despite having 388 associative morphons with class-selective firing patterns (proven by post-hoc labeling showing some neurons respond selectively to specific digits).

### Root Cause

Endoquilibrium's developmental stage detection (`detect_stage()` in `endoquilibrium.rs:506`) uses reward-based relative thresholds calibrated for RL tasks (CartPole). The Mature stage check (line 533):

```rust
if reward_slow > 0.05 && reward_cv < 0.15 && reward_trend.abs() < 0.005 * slow_abs
    && self.reward_history.len() >= 100
{
    return DevelopmentalStage::Mature;
}
```

For CartPole, reward is sparse and variable — it correlates with actual performance (episode length). The system only reaches Mature when it's genuinely performing well.

For MNIST with `reward_contrastive()`, reward is injected on **every sample** at constant strength (0.5) regardless of whether the classification was correct. The reward signal is:
- `reward_slow` > 0.05 within ~20 samples (constant injection → fast EMA climbs quickly)
- `reward_cv` ≈ 0 (constant signal → near-zero coefficient of variation)
- `reward_trend` ≈ 0 (flat constant → no trend)

All three conditions satisfied within the first 100 samples. Endo declares **Mature** before the network has learned anything. In Mature stage:
- `plasticity_mult` drops (learning rate effectively reduced)
- `novelty_gain` drops (exploration suppressed)
- `winner_adaptation_mult` drops (k-WTA winners stop rotating)

The system chokes its own learning at ~20% accuracy because it thinks it's already converged.

### Fix

Two options:
1. **Gate reward injection on correctness**: Only call `inject_reward()` when the classification is actually correct. This makes the reward signal correlate with performance, restoring the RL-like dynamics Endo expects.
2. **Add task-type awareness to stage detection**: Use a different stage detection strategy for supervised classification vs RL. For classification, use accuracy (which the system can compute from the readout error) rather than reward magnitude.

Option 1 is simpler and was applied in the Phase 1.5 implementation: `reward_contrastive()` now broadcasts reward into the modulation channel, and the metabolic system uses it for energy allocation. The stage detection issue is addressed by running Phase 1 (unsupervised, no reward) without Endo metabolic regulation, then Phase 1.5 (supervised, reward flowing) with metabolic pressure but shorter duration so Endo doesn't have time to reach Mature prematurely.

The proper fix (option 2) belongs in the Limbic Circuit module (`docs/specs/limbic-circuit-spec.md`), where the Motivational Drive component tracks reward prediction error (RPE) rather than raw reward. RPE-based stage detection would naturally handle both RL and classification tasks: RPE is high when the system is learning (surprising outcomes) and low when performance is stable (expected outcomes).

### Impact

With the premature-Mature bug avoided (Phase 1.5 approach):
- MNIST analog readout: **45.5%** (up from 12.5% without metabolic pruning, up from ~26% pre-fix baseline)
- 6 of 10 digit classes above 20% accuracy
- Classes 0, 7, 8 above 90%
- 10 morphons pruned via reward-guided apoptosis during Phase 1.5

Standard profile results pending.

---

## 3. Phase 1.5: Metabolic Pruning Protocol

### Architecture

```
Phase 1: Unsupervised STDP + k-WTA
  - Metabolic pressure OFF (no reward signal to distinguish hubs from specialists)
  - Features form through competitive STDP
  - 0 morphon deaths (apoptosis disabled)
      ↓
Phase 1.5: Supervised Metabolic Pruning
  - Enable: reward_energy_coefficient=0.01, superlinear_firing_factor=2.0, apoptosis=true
  - Enable: analog readout + train_readout() for supervised signal
  - reward_contrastive() drives reward into modulation channel
  - Specialists fire for correct class → earn energy → survive
  - Hubs fire for all classes → ±symmetric reward → zero net income → starve → die
  - Duration: half of Phase 1 sample budget
      ↓
Phase 2: Post-hoc labeling + evaluation
  - Metabolic pressure OFF (apoptosis disabled during eval)
  - Evaluate via post-hoc neuron labeling + analog readout
```

### Why Phase 1.5 is a Hand-Coded Limbic Function

The Phase 1 → 1.5 transition is manually orchestrated. In the planned Limbic Circuit architecture, this transition would be handled by the Motivational Drive (nucleus accumbens analog):
- During unsupervised learning: no reward signal → low RPE → low metabolic pressure
- When supervised signal appears: RPE spikes → metabolic pressure activates
- As performance stabilizes: RPE drops → metabolic pressure moderates

The manual Phase 1.5 validates the metabolic selection concept. The Limbic Circuit will make it self-regulating.

---

*Experimental record from NLP readiness + MNIST metabolic experiments.*
*TasteHub GmbH, Wien, April 2026*
