# Endoquilibrium — Experimental Findings

Detailed record of the Endoquilibrium regulation engine's behavior during CartPole training.

---

## 1. What Endoquilibrium Regulates

Endoquilibrium is a predictive neuroendocrine regulation engine. It maintains network health by sensing vital signs, predicting normal trajectories, and adjusting modulation channels through proportional control. It never modifies weights or topology directly.

### 1.1 The FR=0% Deadlock (Primary Contribution)

Before Endoquilibrium, Associative morphons stopped firing within the first 50 ticks of training. With 0% firing rate, no eligibility traces accumulate, no weight updates happen, FR stays at 0%. This is a positive feedback deadlock — the system's most critical failure mode.

Endoquilibrium detects this via Rule 1 (firing rate regulation):

```
if fr_associative < setpoint.min (8%):
    threshold_bias -= deficit * 0.5    // lower thresholds
    arousal_gain   += deficit * 0.3    // increase sensitivity
    novelty_gain   += deficit * 0.2    // increase exploration
```

With `fast_tau=50`, the fast EMA responds within ~50 ticks. The threshold_bias accumulates to ~-0.05 to -0.1, enough to push Associative morphons above threshold. The feedback is smooth — EMA smoothing (alpha=0.1) prevents oscillation.

### 1.2 Observed Channel Values During Training

Typical values after convergence (ep 100+):

| Channel | Value | Default | What it means |
|---------|-------|---------|---------------|
| reward_gain | 1.50 | 1.0 | Rule 5 boosting — tags forming but no captures |
| novelty_gain | 1.40 | 1.0 | Rule 2 boosting — eligibility density below setpoint |
| arousal_gain | 1.00 | 1.0 | No Rule 1 excess — FR in healthy range |
| homeostasis_gain | 1.32 | 1.0 | Mild Rule 2 response — eligibility slightly high |
| threshold_bias | 0.02-0.05 | 0.0 | Small positive — FR slightly above minimum |
| plasticity_mult | 1.20 | 1.0 | Rule 2 boost — eligibility needs help |
| health_score | 0.72-0.80 | 1.0 | Good but not perfect |

### 1.3 Developmental Stage Detection

The system correctly transitions through stages:

| Period | Stage | Trigger |
|--------|-------|---------|
| ep 1-100 | Stressed | PE trend positive, network unstable |
| ep 100-300 | Mature | All derivatives near zero |
| ep 300+ | Mature (stable) | Network settled |

The stage detection uses morphon count trend and PE trend. In CartPole with `lifecycle.division=false`, morphon count is constant, so stage depends primarily on PE trend. Stressed → Mature transition happens when PE stabilizes.

---

## 2. Endoquilibrium Alone Is Not Sufficient

Endoquilibrium doubles the baseline (10 → 21) by maintaining healthy firing rates. But it cannot fix:

- **The capture mechanism**: per-tick capture broken for RL (separate fix needed)
- **Activity instability**: noise-driven pattern variability (separate fix needed)
- **Readout discrimination**: centered sigmoid, bias, weight decay (separate fix needed)

Endoquilibrium provides the foundation — a living, regulated network — on which the other fixes build. Without it, the network is dead (FR=0%) and nothing else matters.

---

## 3. Interaction with Other Systems

### 3.1 Endoquilibrium + Consolidation

`plasticity_mult` interacts with `consolidation_level`:

```
effective_delta = delta_w * plasticity_mult * (1.0 - consolidation_level * 0.9)
```

When Endo boosts `plasticity_mult` to 1.2, consolidated synapses still only get `1.2 * 0.1 = 0.12` of normal update. The consolidation protection is preserved even under Endo amplification.

### 3.2 Endoquilibrium + Weight Normalization

The gentle weight normalization (0.9-1.1 clamping) operates independently of Endo. Endo's `plasticity_mult` scales the learning rule deltas, normalization scales the post-update weights. They don't conflict — Endo controls learning rate, normalization controls weight sum.

### 3.3 Endoquilibrium + Readout

Endo's channel gains multiply the receptor-gated modulation signal in `apply_weight_update`. The readout (`train_readout`) uses a separate softmax cross-entropy gradient that is NOT scaled by Endo gains. This is correct — the readout is a supervised pathway that should not be modulated by the RL-oriented neuromodulatory channels.

---

## 4. What CMA-ES Found (and Didn't)

CMA-ES searched 10 dimensions: 5 Endo regulation gains + 5 learning parameters. After 28 generations (560 evaluations), best avg was 21.2.

The fitness landscape was flat around avg=20 — no parameter combination improved beyond the Endo baseline because the bottleneck was in the readout architecture, not the regulation parameters.

**Implication for the paper:** Endoquilibrium's default parameters (hand-set from the spec) are reasonable. CMA-ES optimization over regulation gains is a tuning step, not a breakthrough. The architecture matters more than the coefficients.

---

## 5. Ablation: Endoquilibrium Disabled

| Metric | Endo enabled | Endo disabled |
|--------|-------------|--------------|
| avg (with all other fixes) | 195.2 | not tested in isolation |
| FR stability | 16% (constant) | 0-10% (unstable, deadlocks) |
| Eligibility density | 30-40% | variable, often near 0% |

Endoquilibrium is load-bearing. Without it, the FR deadlock prevents the other fixes from having any effect — there's nothing to learn from if the hidden layer is silent.
