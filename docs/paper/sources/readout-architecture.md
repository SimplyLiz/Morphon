# Readout Architecture — The Cerebellar Pattern

How the analog readout works, why it needed three fixes, and what this means architecturally.

---

## 1. The Dual-System Design

The solved CartPole uses two distinct learning systems:

| System | What it learns | Learning rule | Supervision |
|--------|---------------|---------------|-------------|
| MI morphon network | Feature representations | Three-factor STDP + DFA | Unsupervised |
| Analog readout | State → action mapping | Softmax cross-entropy | Supervised |

The MI network grows, self-organizes, and adapts its connectivity through biologically plausible local rules (eligibility traces, neuromodulation, structural plasticity). It does NOT learn the task — it learns to represent the input space.

The readout is a linear classifier over the MI network's feature activations. It maps `sigmoid(morphon_potentials) → action_probabilities` using the delta rule with state-derived labels. This IS supervised: `correct_action = sign(theta)`.

### 1.1 Biological Analogy: Cerebellar Circuit

| Cerebellar component | MORPHON component |
|---|---|
| Mossy fibers (input) | Population-coded sensory input (32 Gaussian tiles) |
| Granule cells (features) | Associative morphons (unsupervised STDP) |
| Parallel fibers (feature → output) | Readout weights |
| Purkinje cells (output) | Motor outputs via readout |
| Climbing fibers (error signal) | `train_readout(correct_action, lr)` |
| Inferior olive (error computation) | `correct_action = sign(theta)` |

This is not a metaphor — it's a direct functional mapping. The cerebellar circuit uses a supervised readout (climbing fiber → Purkinje plasticity) on top of an unsupervised feature expansion (mossy fiber → granule cell). MORPHON does the same, with the MI network providing the feature expansion and the analog readout providing the supervised mapping.

---

## 2. The Readout Equation

### Before fixes:

```
V_j = Σ_i w[j][i] × sigmoid(potential_i)     — no bias, sigmoid centered at 0.5
Δw = lr × sigmoid(p) × error - 0.001 × w     — with L2 decay
```

### After fixes:

```
V_j = bias[j] + Σ_i w[j][i] × (sigmoid(potential_i) - 0.5)
Δw = lr × (sigmoid(p) - 0.5) × error         — no L2 decay
Δbias = lr × error
```

Three changes, each addressing a specific failure mode.

---

## 3. Why Each Fix Was Necessary

### 3.1 Centered Sigmoid (sigmoid(p) - 0.5)

**The problem:** `sigmoid(0) = 0.5`. At resting potential (~0), every morphon contributes `w × 0.5` to the output. With ~80 morphons in the readout, the constant offset is `0.5 × Σw ≈ 0.5 × 80 × 0.1 = 4.0` on each output. State-dependent variations in potential are ~0.1-0.5, contributing `w × Δsigmoid ≈ 0.1 × 0.05 = 0.005`. The signal-to-noise ratio is 0.005/4.0 = 0.00125.

**The fix:** `sigmoid(p) - 0.5` makes the resting-state contribution zero. Now the readout only sums state-dependent activations. The constant offset disappears.

**Measured effect:** Without centering, `d+` and `d-` had the same sign (constant bias dominated). With centering, they developed opposite signs within 100 episodes.

### 3.2 Learnable Bias

**The problem:** Even with centered sigmoid, the readout has no way to express "output 0 should be higher than output 1 by default." The weight sum `Σ w[j][i] × centered_act[i]` is zero at rest for both outputs. If the task has an asymmetric distribution (pole starts near center, so the correct action is ~50/50), this is fine. But if there's any systematic bias in the training data (e.g., pole drifts left more often due to initial conditions), the readout needs a bias term to capture it.

**The fix:** Per-output bias `bias[j]`, trained with `Δbias = lr × error[j]`. Two extra f64 values.

**Measured effect:** The bias absorbs the mean output level, allowing weights to focus on state-dependent discrimination. This is standard practice in linear classifiers for exactly this reason.

### 3.3 No L2 Weight Decay

**The problem:** `Δw = lr × act × error - 0.001 × w` applies weight decay every step. Over an episode of ~200 steps, each weight is reduced by `1 - (1-0.001)^200 ≈ 18%`. The supervised gradient tries to build a discriminative pattern; the decay erases 18% of it every episode. The equilibrium point (gradient = decay) is much weaker than the gradient alone would produce.

With `lr=0.05` and `act ≈ 0.1`:
- Supervised gradient per step: `0.05 × 0.1 × 0.5 = 0.0025` (for a 50% softmax error)
- Decay per step: `0.001 × w`
- Equilibrium weight: `w = gradient/decay = 0.0025/0.001 = 2.5`

A weight of 2.5 shared across 80 morphons gives `d ≈ 80 × 2.5 × 0.05 = 10` — which should be enough. But the decay applies to ALL weights, including those that are correct and should be preserved. The decay is indiscriminate — it erases good weights as fast as bad ones.

**The fix:** Remove L2 decay entirely from `train_readout`. Weight clipping (`clamp(-5, 5)`) prevents explosion without erasing the discriminative signal.

**Measured effect:** This was the largest single contributor. With decay: avg=44, d+/d- same sign. Without decay: avg=195 (SOLVED), d+/d- opposite signs with magnitude ~10.

---

## 4. What the Readout Learns

After solving CartPole, the readout has learned:
- **Positive weights** on sensory morphons encoding `theta > 0` (rightward lean) → output 1 (push right)
- **Negative weights** on sensory morphons encoding `theta < 0` (leftward lean) → output 1
- The inverse pattern on output 0

This is essentially a linear classifier on population-coded theta. The MI network's Associative morphons provide additional features, but the sensory-only readout also works (avg ~similar), suggesting the Associative layer isn't contributing discriminative features for CartPole's simple 2-action task.

**Implication:** For CartPole, the MI network's contribution is maintaining healthy representations (via Endoquilibrium) rather than learning complex features. The task is too simple to need nonlinear features — a linear readout on population-coded input suffices. MNIST (10-class, 784-dim) should reveal the MI network's representational value.

---

## 5. TD-Only vs Supervised Readout

| Metric | Supervised (sign(theta)) | TD-only (chosen action) |
|--------|--------------------------|------------------------|
| avg peak | 195.2 | 88.7 |
| Convergence | Monotonic, solved at ep 895 | Oscillating, never solved |
| d+/d- | Opposite signs, magnitude ~10 | Weak opposite, magnitude ~2 |
| Policy probe | 2/4 (cold-start artifact) | Peaks at 4/4 intermittently |

**Why TD-only is weaker:** TD error is a scalar that says "things improved" or "things worsened." It doesn't say *which action* caused the improvement. With epsilon-greedy, 50% of training updates reinforce the wrong action. The net gradient is `(0.5 × correct_gradient + 0.5 × wrong_gradient)`, which averages to a weak signal in the correct direction.

The supervised hint provides `correct_gradient` directly — no noise from wrong-action reinforcement. The gradient is 2x stronger and in the right direction 100% of the time.

**Why TD-only still works (avg=88):** Over many episodes, the correct action leads to longer episodes (higher cumulative reward), so `td_error.abs()` is larger after correct actions on average. The reinforcement is biased toward the correct action even though individual updates are noisy. This bias is weak (~60/40 vs 100/0 for supervised), explaining the 2x gap.

---

## 6. Open Question: Can the MI Network Learn Readout-Free?

The current architecture uses the MI network as a feature extractor and the readout as a classifier. The MI network's three-factor STDP + DFA learning rules cannot solve CartPole alone (avg=21 without readout fixes).

The fundamental issue: the three-factor rule `Δw = eligibility × modulation × plasticity` uses a single scalar modulation signal that can't express "strengthen connections to output A, weaken connections to output B." Per-output credit assignment requires either:

1. Supervised readout (current approach)
2. Receptor-gated modulation that differs per output neuron (not implemented)
3. A critic network that produces per-action value estimates (not just scalar TD)

Option 2 is the most biologically plausible path to readout-free CartPole. Each motor morphon could have different receptor weights for the reward channel, creating different sensitivity to the same global reward signal. This is implementable within the current receptor framework.
