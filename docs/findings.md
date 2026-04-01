# MORPHON Learning Investigation — Complete Findings

## Status: April 2026

The MI engine is architecturally complete. All 6 biological principles are implemented and tested. The learning pipeline does NOT yet converge on classification or RL benchmarks. This document captures everything we know about why, what was tried, and what remains.

---

## What Works

### Architecture
- Full cell lifecycle: division, differentiation, fusion, defusion, migration, apoptosis
- Hyperbolic information space (Poincare ball) with learnable curvature
- Four neuromodulation channels with receptor-gated modulation
- Dual-clock scheduler (fast/medium/slow/glacial paths)
- Tag-and-capture for delayed reward (Frey & Morris 1997)
- Homeostatic protection: synaptic scaling, inter-cluster inhibition, checkpoint/rollback
- Dense I/O pathways with Xavier-scaled initialization
- Trace-based STDP with weight-dependent bounds (Gilson & Fukai 2011)
- Advantage modulation (reward - baseline EMA)
- Spike-timing eligibility updates at spike delivery
- 23+ tests, 3 build targets (native, WASM, Python), serialization, diagnostics

### Learning (proven)
- **2-class classification at 62% accuracy** (12% above 50% chance) — learn_compare example
- External logistic regression on the same MI infrastructure: **100%** accuracy
- Motor morphons fire and produce class-differentiated outputs
- teach_supervised_with_input produces correct weight gradients
- CMA-ES meta-learning found 48% on 3-class with biased encoding
- 3-class: all three per-class accuracies simultaneously above chance (epochs 30, 70, 150, 200 across runs)

---

## What Doesn't Work

### 1. Three-Factor Learning Alone Cannot Classify (Priority: Critical)

**Symptom:** Three-factor rule (eligibility × modulation) stays at random chance on all classification tasks.

**Root Cause:** The global modulation signal M(t) = αr×R + αn×N + αa×A + αh×H applies the SAME scalar to ALL synapses with non-zero eligibility. For classification, different output paths need different weight changes (class 0 path strengthened, class 1 path weakened). A single global scalar can't express "strengthen this path, weaken that one."

**What was tried:**
- Receptor-gated modulation (Motor→Reward, Sensory→Novelty) — helps differentiate WHICH morphon types learn, but within the Motor type, all motors still get the same reward signal
- Contrastive reward (reward correct output, inhibit incorrect) — inhibition drove weights to -5.0, silencing motors permanently
- Eligibility-decay inhibition (decay toward zero instead of negative push) — prevented silencing but didn't create class discrimination
- teach_hidden with ACTUAL motor activity — zero signal when all motors have similar potential (chicken-and-egg)
- teach_hidden with TARGET signal — correct conceptually but the three-factor multiplication with M(t) dilutes it
- CMA-ES over 15 parameters — found that novelty (not reward) drives learning, but best result was 48% (stochastic, not stable)

**Verdict:** The three-factor rule is correct for RL (temporal credit via tag-and-capture), but insufficient for classification (spatial credit across output classes). The concept doc's four channels cannot express "this output is correct, that one isn't" as a single global broadcast.

### 2. Motor Morphon Potential Drift (Priority: High)

**Symptom:** Over hundreds of epochs, 2 of 3 motor morphon potentials drift to the negative clamp (-5 or -10), leaving only 1 motor alive. Mode collapse.

**Root Cause:** Multiple contributing factors:
- **Noise accumulation:** Spontaneous noise (±0.05 per step) integrated over 5 propagation steps × 500 samples × 300 epochs = 750K noise samples. Even with zero mean, random walk variance grows as sqrt(N).
- **Leaky integration memory:** Standard leak rate 0.1 means potential = 0.9 × old + input. Motor morphons remember past state, so one negative excursion creates a bias for future steps.
- **Xavier-scaled mixed-sign weights:** Assoc→motor weights initialized in [-1/sqrt(N), +1/sqrt(N)] sum to near zero for each motor. The net signal is dominated by noise.
- **Warm-up residuals:** 100-step warm-up creates initial asymmetry that becomes permanent winner bias.

**What was tried:**
- Potential clamp at [-5, 5] then [-10, 10] — motors still hit the wall
- Motor weight reset after warm-up — caused mode collapse (all motors identical → first class wins)
- Weight decay (L2 regularization at 0.001 and 0.01) — slowed drift but didn't prevent it
- Full leak for motor morphons (leak_rate=1.0, memoryless) — improved stability, best results
- Zero noise for motor morphons — eliminated noise-driven drift, best combined with full leak
- Xavier init → zero init → Xavier (cycled) — zero causes mode collapse, Xavier causes one-motor dominance

**Best configuration found:** Motor morphons with leak_rate=1.0 + zero noise + weight decay 0.01. This keeps 2 of 3 motors alive for ~200 epochs. The third still drifts to -10 eventually.

**Verdict:** The MI propagation pipeline introduces uncontrollable noise into motor potentials. The delta rule can correct the direct weights, but the indirect signal through the spike pipeline (which the delta rule doesn't control) drifts the potential.

### 3. Input Bias Masks Class Signal (Priority: Fixed)

**Symptom:** All approaches stuck at random chance regardless of learning rule.

**Root Cause:** Input encoding used bias=2.0 to "keep the network active." With bias 2.0 and class signal 3.0, all sensory morphons had potential 2.0-5.0 → sigmoid ≈ 0.88-0.99. The 12% difference between active/inactive inputs was invisible to the learning rule.

**Fix:** Zero-bias encoding. Inactive channels stay at 0.0, active channels at 3.0. sigmoid(0)=0.5, sigmoid(3)=0.95 — clear discrimination.

**Impact:** Enabled the 62% 2-class result and all subsequent progress.

### 4. Burst Activation Dead Zone (Priority: Fixed)

**Symptom:** Motor morphons at 0% firing rate throughout training.

**Root Cause:** Old Burst activation: `if x > 0.5 { tanh... } else { 0.1 * x }`. For input 0.35 (typical motor input), output = 0.035 < threshold 0.05 → permanently silent.

**Fix:** New Burst: sigmoid baseline + tanh amplification above 0.5. Responsive at all input levels.

### 5. LTD Vicious Cycle (Priority: Fixed)

**Symptom:** Motor weights drift negative over time, motor potential → -5.0.

**Root Cause:** At 0% motor firing, the Hebbian coincidence function returned -0.06 for every spike delivery (pre fired, post didn't). Over thousands of steps, eligibility went permanently negative → weights decreased → motors got less input → still 0% firing → vicious cycle.

**Fix:** Protect cold morphons: skip LTD when post-synaptic activity < 2%. Only apply LTP on coincidence.

### 6. Modulatory Explosion (Priority: Fixed)

**Symptom:** Network grows to 2000+ morphons, 95%+ are Modulatory type.

**Root Cause:** Divided morphons inherited parent cell type. Modulatory parents produced Modulatory children, which divided again. Differentiation heuristic classified low-activity morphons as Modulatory.

**Fix:** Asymmetric division: children start as Stem. Developmental differentiation at level 0.6 resists dedifferentiation.

---

## The Fundamental Gap

External logistic regression (same data, same learning rate, same delta rule formula) achieves **100%** accuracy on the 3-class task in 10 epochs. The MI system with teach_supervised_with_input achieves **~38%** at best (41.5% peak).

The gap is entirely in how the MI propagation pipeline distorts the forward pass:

1. **Multi-step propagation:** `process_steps(input, 5)` runs 5 simulation steps. The motor potential at step 5 reflects a mixture of current input + decayed past inputs + accumulated noise. The delta rule assumes the output is a pure function of the current input.

2. **Spike-based communication:** Information travels as discrete spikes through delayed channels. The motor morphon's potential at read time reflects spikes that arrived on different steps, some from the current input and some from previous inputs' echoes.

3. **Leaky integration memory:** Each morphon's potential carries state from previous steps. Motor potential on sample N is contaminated by residuals from samples N-1, N-2, etc.

4. **Noise injection:** Spontaneous noise on non-motor morphons propagates through the network and arrives at motors as uncontrollable input.

In a standard feedforward network: output = f(Wx). The gradient dL/dW = dL/df × df/dW = error × input. This is exact.

In MI: output = f(spike_propagation(input, noise, previous_state, delays)). The gradient is not computable locally because the output depends on the entire propagation history, not just the current input.

---

## What We Recommend Next

### Short Term (prove convergence)
1. **Dedicated output layer bypass:** Motor morphons read directly from a weighted sum of sensory/associative potentials (no spike propagation). This makes `output = f(Wx)` exact, and the delta rule will work at 100%.
2. **Freeze MI propagation for classification:** Use MI only for the hidden layer's structural adaptation (growth, migration, fusion). The output layer uses standard gradient-free classification.

### Medium Term (restore MI learning)
3. **Temporal averaging:** Read motor output as the average potential over the last N steps, not a single timestep. This averages out noise.
4. **Population coding:** Use the MEAN of multiple motor morphons per class instead of a single motor. Noise cancels across the population.
5. **Re-run CMA-ES** with zero-bias encoding + motor fixes. The previous CMA-ES used biased encoding — the search space is fundamentally different now.

### Long Term (research)
6. **Implement proper SADP:** The full Cohen's kappa metric with spike-train agreement, not the simplified version. SADP achieves 99.1% MNIST without backprop.
7. **Hybrid architecture:** MI for structure (morphogenesis), gradient-free supervised for classification, three-factor for RL. Different tasks use different learning rules — biologically valid (cerebellum ≠ hippocampus ≠ cortex).
8. **TD-error critic for RL:** Already partially implemented. A linear value function providing δ = R + γV(s') - V(s) gives directional signal for temporal credit.

---

## Configuration Reference

### Best 2-class result: 62% (learn_compare Option A)
```
encoding: zero bias, scale=3.0
learning: default LearningParams
scheduler: medium_period=1
teach_hidden: strength=1.92
reward_contrastive: reward=0.1, inhibit=0.0
inject_novelty: 0.3
process_steps: 5
```

### Best 3-class result: 41.5% (classify_3class Epoch 100)
```
encoding: zero bias, scale=3.0, 9 inputs (3 per class, non-overlapping)
learning: default LearningParams
scheduler: medium_period=99999 (three-factor OFF)
teach_supervised_with_input: lr=0.02
motor: leak_rate=1.0, noise=0.0
weight_decay: 0.01
process_steps: 5
```

### CMA-ES optimal parameters (48% 3-class with biased encoding)
```json
{
  "tau_eligibility": 1.2,
  "tau_trace": 3.07,
  "a_plus": 4.99,
  "a_minus": -4.94,
  "alpha_reward": 0.5,
  "alpha_novelty": 3.0,
  "alpha_arousal": 0.0,
  "tag_threshold": 0.8,
  "capture_rate": 1.0,
  "weight_max": 2.07,
  "teach_strength": 1.92,
  "reward_strength": 0.1,
  "inhibit_strength": 0.0,
  "input_bias": 1.99,
  "input_scale": 3.86
}
```

---

## Files

| File | Description |
|------|-------------|
| `examples/learn_compare.rs` | Head-to-head: three-factor vs supervised delta vs minimal. Proved 62% on 2-class. |
| `examples/classify_3class.rs` | 3-class with external logistic regression control (100% ext, ~38% MI). |
| `examples/classify_tiny.rs` | CMA-ES parameter testing, 3-class with various configs. |
| `examples/cma_optimize.rs` | CMA-ES meta-learning over 15 parameters. |
| `examples/cartpole.rs` | CartPole RL with TD-error critic, sparse encoding. |
| `examples/mnist.rs` | Full 784px MNIST, zero-bias encoding. |
| `docs/benchmark_results/cma_best_params.json` | CMA-ES optimal parameter set. |
