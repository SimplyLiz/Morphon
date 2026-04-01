# CartPole Representational Drift — Research Log

**Date:** 2026-04-01
**Status:** Active investigation
**Severity:** Blocking — prevents reliable CartPole convergence

---

## 1. Problem Statement

CartPole averages ~15 steps at peak (ep 100-200), then degrades to ~12 (random baseline ~10) by ep 1000. Best single episodes reach 95-106, proving the system CAN learn, but it can't hold the learned policy. This is **representational drift** — the hidden layer's features shift under the readout's feet.

## 2. What We Fixed (2026-04-01)

### 2.1 Dead DFA Pathway
`cartpole.rs` never called `inject_td_error()`, so `last_td_error = 0.0` for the entire run. The DFA mechanism — the hidden layer's only source of neuron-specific credit — was multiplying by zero every step. Fixed by calling `system.inject_td_error(reward, GAMMA)` directly.

### 2.2 Saturated Reward Channel
`inject_reward()` is additive with 0.95 decay. The old scaling `(td_error * 0.3 + 0.5)` injected ~0.5 every step, pegging the channel at 1.0 permanently. `reward_delta()` ≈ 0.0 — the three-factor rule received zero reward signal. Fixed to only inject on positive TD, letting natural decay create directional signal.

### 2.3 Population Coding (Lisa)
Replaced binary pos/neg sparse encoding (8 channels) with 32-channel Gaussian population coding (4 obs x 8 tiles). Each state region now produces a distinct activation pattern. This is what enabled best=106 — the readout finally had features to discriminate on.

### 2.4 Learning Parameter Tuning
- Terminal reward: 0.0 → -1.0 (stronger failure signal)
- tau_eligibility: 15 → 3 (shorter, reactive)
- a_minus: -0.8 → -0.5 (weaker LTD, prevents inhibitory drift)
- weight_max: 5.0 → 3.0 (tighter bounds)
- capture_threshold: 0.3 → 10.0 (disable consolidation)
- Internal steps: 8 → 4 (reactive task)
- DFA feedback_strength: 0.5 → 1.0
- Internal critic td_lr: 0.01 → 0.03
- Both DFA consolidation paths gated on capture_threshold

### 2.5 Results After Fixes

| Metric | Before | After |
|--------|--------|-------|
| Best episode | 59 | 106 |
| Avg (ep 100) | 11 | 18 |
| Firing rate | 7.3% | 13% |
| DFA active | No | Yes |
| Tags | 146 | 1000+ |

## 3. Root Cause: Representational Drift

The hidden layer representations shift continuously while the readout tries to learn on top of them. Three interacting mechanisms create this instability:

### 3.1 Over-Plasticity
STDP with three-factor modulation keeps modifying hidden-layer weights every step. Even with mild alpha_reward (0.5-2.0), the cumulative effect over 1000 episodes is large. The readout learns to interpret feature pattern A, but by ep 500, the hidden layer has drifted to pattern B.

**Evidence:** Setting alpha_reward=0 (reservoir mode) produces identical degradation — the drift comes from the DFA weight updates, not just three-factor learning. The DFA climbing-fiber rule (`Δw = pre_trace × feedback_signal × lr`) continuously modifies associative→motor weights.

### 3.2 Homeostatic Seesaw
When a morphon becomes "important" (fires frequently, drives correct actions), homeostatic threshold regulation pushes its threshold up, reducing its firing rate. The readout depended on that morphon firing — now it gets weaker signal. The readout adapts, but then the morphon's threshold drops again (firing rate fell below setpoint), and the cycle repeats.

**Evidence:** Spike counts decline monotonically across all runs: 450 → 150 over 1000 episodes. This happens even with frozen three-factor learning (alpha=0).

### 3.3 Lack of Anchoring
In biology, "Pioneer Neurons" and strongly consolidated synapses provide stable reference points. MORPHON's consolidation (tag-and-capture) was locking in random early weights (before useful features emerged), so we disabled it. But without ANY consolidation, nothing is stable.

**Evidence:** With consolidation enabled (capture_threshold=0.3), con=1400/2000 by ep 100 — locking in random associations. With consolidation disabled (capture_threshold=10), con=0 — fully plastic, nothing anchored.

## 4. Experimental Observations

### 4.1 Δout (Output Differentiation)
The readout consistently produces differentiated outputs (Δout = 3-7). The outputs ARE different — but not consistently mapped to the right actions. The readout oscillates between configurations.

### 4.2 Seed Sensitivity
Across 10+ runs with identical params: best episodes range 36-106. The initial random topology determines whether the system gets lucky with representations that happen to be informative and stable.

### 4.3 Consolidation Catch-22
- Consolidate early → locks in garbage (random early-training weights)
- Never consolidate → nothing stable, drift continues
- Performance-gated consolidation (consolidation_gate=15) → flickers on/off around the gate threshold, inconsistent

### 4.4 Three-Factor vs. Reservoir Mode
Setting alpha_reward/novelty/arousal all to 0.0 (pure reservoir + readout) produces the SAME degradation pattern. This rules out three-factor learning as the primary drift source. The DFA weight updates and homeostatic threshold regulation are sufficient to cause drift.

## 5. Hypotheses

### H1: "Anchor & Sail" Heterogeneous Plasticity (HIGH PRIORITY)
Introduce two populations of morphons with different plasticity rates:
- **Anchor morphons (30%):** Very slow STDP rate (alpha_reward × 0.1), long eligibility traces, high consolidation priority. The readout learns to depend primarily on these stable features.
- **Sail morphons (70%):** Normal plasticity, fast exploration. They capture new patterns but the readout doesn't over-rely on them.

**Biological parallel:** Parvalbumin+ fast-spiking interneurons (stable) vs. pyramidal neurons (plastic) in cortex. PV+ neurons provide a stable temporal framework while pyramidal cells learn new associations.

**Implementation options:**
- A) Cell-type gating: Associative morphons with `differentiation_level > 0.8` get reduced plasticity rates. Already partially supported by the receptor gating system.
- B) New cell type attribute: `plasticity_rate: f64` on Morphon struct, initialized based on position in Poincare ball (origin = anchor, boundary = sail).
- C) Readout-coupled anchoring: Morphons that contribute strongly to readout predictions get their plasticity reduced (feedback from readout weights).

### H2: Readout-Coupled Consolidation
Instead of performance-gated consolidation, gate consolidation on readout weight magnitude. If a readout weight to morphon M is large (|w| > threshold), then M's incoming synapses get consolidated. The readout "votes" on which hidden features to stabilize.

**Mechanism:** After each `train_readout()` call, identify morphons with high readout weight magnitude. Tag their incoming synapses for consolidation. This creates a virtuous cycle: useful features get stabilized → readout can learn more reliably → more features get stabilized.

### H3: Plasticity Annealing
Reduce global plasticity rate over time. High plasticity early for exploration, low plasticity late for exploitation. Similar to learning rate schedules in SGD.

**Problem:** This is fundamentally un-biological and removes the continuous adaptation that makes MI interesting. Should be a last resort.

### H4: CMA-ES Meta-Optimization of Plasticity Rates
Use the existing CMA-ES infrastructure to search for plasticity parameters that produce stable representations. The fitness function would measure not just CartPole steps but also representation stability (e.g., readout weight change rate, spike pattern consistency).

### H5: Multi-Timescale Readout
Instead of a single readout learning rate, use a slow readout (EMA of weights) for action selection and a fast readout for error computation. The slow readout smooths over transient representation changes.

## 6. Experimental Priorities

| # | Hypothesis | Expected Impact | Effort | Biological Basis |
|---|-----------|----------------|--------|-----------------|
| 1 | H1: Anchor & Sail | High — directly addresses drift | Medium — new plasticity_rate field + gating | PV+ interneurons, Pioneer neurons |
| 2 | H2: Readout-coupled consolidation | High — creates virtuous stability cycle | Low — modification to train_readout() | Hebbian consolidation, synaptic tagging |
| 3 | H4: CMA-ES meta-opt | Medium — finds stable regimes automatically | Low — infrastructure exists | Evolutionary search |
| 4 | H5: Multi-timescale readout | Medium — smooths over drift | Low — EMA of readout weights | Fast/slow learning systems |
| 5 | H3: Plasticity annealing | Low — removes continuous adaptation | Trivial | Not biological |

## 7. Key Metrics for Future Experiments

- **avg(100):** Primary performance metric. Target: 100+ sustained.
- **Δout:** Output differentiation. Should stay > 2.0 throughout training.
- **spk:** Total spike count. Should not decline monotonically.
- **con/total:** Consolidation ratio. Should increase gradually, not spike early.
- **weight_std:** Should stabilize, not grow or shrink monotonically.
- **Readout weight variance:** (not yet tracked) Should stabilize after initial learning.
- **Representation stability:** (not yet tracked) Cosine similarity of hidden layer activations for the same input across episodes. Should increase over time.

## 8. Literature Pointers

- **Lillicrap et al. 2016:** Random synaptic feedback weights support error backpropagation for deep learning. (DFA foundation)
- **Diehl & Cook 2015:** Unsupervised learning of digit recognition using STDP. (k-WTA + weight normalization)
- **Gilson & Fukai 2011:** Stability versus neuronal specialization for STDP. (Multiplicative weight-dependent STDP)
- **Frey & Morris 1997:** Synaptic tagging and long-term potentiation. (Tag-and-capture)
- **Zenke, Agnes & Gerstner 2015:** Diverse synaptic plasticity mechanisms orchestrated to form and retrieve memories in spiking neural networks. (Heterogeneous plasticity timescales — directly relevant to H1)
- **Maryada et al. 2025:** Stable recurrent dynamics in heterogeneous neuromorphic systems using excitatory and inhibitory plasticity. Nature Communications. (Heterogeneous E/I plasticity for stability — already in our reference list)

---

*Research by Lisa (TasteHub GmbH) + Claude, 2026-04-01*
