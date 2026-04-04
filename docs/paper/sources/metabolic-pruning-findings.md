# Metabolic Pruning & Endo Calibration — Paper Results

Experimental findings from the learning pipeline investigation (2026-04-03/04).
Branch: `feature/local-inhibitory-competition`

---

## Paper-Ready Numbers

| Benchmark | Result | Condition |
|-----------|--------|-----------|
| CartPole | **SOLVED** avg=195.2 | v0.5.0, episode 895 |
| MNIST baseline | **31.0%** | Endo Mature gate=2000, Consolidating=500, quick profile |
| MNIST post-recovery | **52.5%** | 30% neuron destruction + regrowth, same Endo config |
| Self-healing delta | **+21.5pp** | System improves after damage, doesn't just recover |
| NLP Tier 0 (analog readout) | **99%** | Bag-of-characters, proves MI substrate is discriminative |
| NLP Tier 2 (analog readout) | **88%** | Sequential memory — temporal information survives in potentials |

---

## Finding 1: MI Substrate Creates Discriminative Representations

### Experiment

NLP Readiness Benchmark: 4-tier synthetic text classification. Two output pathways tested — spike-based (teach_supervised) and analog readout (train_readout, Purkinje-style bypass).

### Results

| Tier | Spike-based | Analog readout |
|------|------------|---------------|
| 0: Bag-of-Chars (27-dim) | 46% (chance) | **99%** |
| 1: One-Hot Scale (135-dim) | 51% (chance) | **69%** |
| 2: Sequential Memory (27-dim x 3 steps) | 48% (chance) | **88%** |
| 3: Compositional XOR (54-dim) | 50% (chance) | 40% (needs nonlinear hidden features) |

### Interpretation

The morphon substrate creates discriminative representations during spike-based processing. The analog readout proves it by reading the same potentials that spikes are generated from. The bottleneck is spike-to-output fidelity: signals distorted by delays, leaky integration, and multi-hop propagation.

Tier 2 at 88% is significant: temporal memory exists in residual potentials across sequential `process()` calls. The substrate for temporal sequence processing already works at the potential level.

---

## Finding 2: Endo Premature Mature Bug

### Symptom

MNIST accuracy plateaus at ~26% despite having ~380 associative morphons with class-selective firing patterns.

### Root Cause

Endo's developmental stage detection uses reward-based thresholds calibrated for RL. For MNIST classification:

1. `reward_contrastive()` injects reward (0.2) on every sample regardless of accuracy
2. `report_performance()` feeds `recent_performance` into Endo's `reward_avg`
3. Early windowed accuracy inflates `recent_performance` before converging
4. `reward_slow` reaches 0.58-0.68 within 500 ticks
5. Mature triggers (`reward_slow > 0.3 && reward_cv < 0.15`)
6. `plasticity_mult` drops to 0.60 — learning effectively stops at ~26%

### Evidence

| pm level | Stage | Accuracy |
|----------|-------|----------|
| 0.60 | Mature (premature) | 27% |
| 0.96-1.77 | Differentiating/Consolidating oscillation | **31%** |
| 1.80 | Differentiating (constant, no consolidation) | 25% |
| 2.16 | Differentiating (post-damage recovery) | **52.5%** |

### Fix

Raise Mature history requirement from 100 to 2000 ticks. Keep Consolidating at 500.

This creates a natural explore/exploit oscillation: Differentiating (pm~1.77, explore) / Consolidating (pm~0.96, stabilize). The oscillation IS the correct dynamic — cortical learning alternates theta bursts (high plasticity) with sharp-wave ripples (consolidation). Mature at 2000 prevents permanent throttling.

### Key Insight: Oscillation Is the Feature

Pure high plasticity (pm=1.80 constant) produced WORSE results (25%) than oscillating (31%). The system needs periodic consolidation to lock in gains. The bug was premature Mature killing plasticity permanently, not the existence of plasticity modulation.

---

## Finding 3: Self-Healing Exceeds Intact Performance

### Experiment

MNIST V2 quick profile (3 epochs x 3000 samples, seed=42):
1. Train intact system -> baseline accuracy
2. Kill 30% of associative morphons randomly
3. Enable division + differentiation + migration
4. Retrain on 500 recovery samples

### Results

| Phase | Morphons | Synapses | Accuracy |
|-------|----------|----------|----------|
| Intact (trained) | 1253 | ~78K | 31.0% |
| Post-damage | 1138 | -- | 30.0% |
| Post-recovery | 2382 | -- | **52.5%** |

The system didn't just recover — it improved by 21.5 percentage points over its intact performance. The damage/regrowth cycle removed entrenched hubs and replaced them with fresh morphons that learned better features under high plasticity (pm=2.16 during recovery, Endo in Differentiating stage because damage reset the performance signal).

### Why Self-Healing Works Better Than Normal Training

During normal training, Endo oscillates between Differentiating and Consolidating, averaging pm~1.4. During recovery, the damage resets Endo to Differentiating (pm=2.16) because the performance signal drops. The fresh morphons learn under higher effective plasticity than the originals ever had.

The 31% -> 52.5% gap represents what the Limbic Circuit's RPE-scaled metabolic pressure will eventually provide during normal training: continuous hub pruning without requiring manual damage.

---

## Finding 4: Sparse Self-Pruning Produces Better Features

### Observation

The best pre-fix MNIST result (32.5%) came from a network that self-pruned from ~83K to 7377 synapses during training — a 91% reduction. The surviving synapses were the ones STDP found useful.

| Synapses | Accuracy | Source |
|----------|----------|--------|
| 7,377 (self-pruned) | 32.5% | Extended training with slow_period=5000 |
| ~80K (30% connectivity) | 28% | Standard training |
| ~287K (full connectivity) | worse | Full S->A connections |

Sparser networks produce more diverse receptive fields. At 6 synapses per morphon, each morphon sees a unique tiny subset of pixels — maximally diverse starting conditions for STDP.

---

## Endo Configuration for Classification Tasks

```rust
// In endoquilibrium.rs detect_stage():

// Mature: require 2000+ ticks of reward history
// Prevents premature throttling during critical learning phase
if reward_slow > 0.3 && reward_cv < 0.15 && ...
    && self.reward_history.len() >= 2000

// Consolidating: require 500+ ticks
// Allows natural Differentiating<->Consolidating oscillation
if reward_slow > 0.05 && reward_fast > reward_slow * 0.95 && ...
    && self.reward_history.len() >= 500
```

The long-term fix is the Limbic Circuit's Motivational Drive, which replaces raw reward with RPE (reward prediction error) for stage detection. RPE naturally handles both RL and classification: high RPE = still learning, low RPE = converged. This makes the history gate unnecessary.

---

## Reproduction

```bash
# Best baseline (31% + 52.5% recovery):
cargo run --example mnist_v2 --release
# Uses: Mature gate=2000, Consolidating gate=500, seed=42, quick profile

# NLP readiness benchmark:
cargo run --example nlp_readiness --release
# Uses: analog readout, LocalInhibition, 4 tiers
```

---

*Experimental record from NLP readiness + MNIST Endo calibration experiments.*
*TasteHub GmbH, Wien, April 2026*
