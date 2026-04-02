# Diagnostic Instruments

Probes developed during CartPole optimization. Each section describes what to measure, how to interpret it, and what it predicts about system behavior.

---

## 1. Activity Stability (Jaccard Similarity)

**What it measures:** Consistency of the Associative layer's firing pattern for identical inputs.

**Protocol:**
```rust
for _ in 0..5 {
    system.reset_voltages();
    system.process_steps(&test_input, INTERNAL_STEPS);
    let fired: Vec<MorphonId> = system.morphons.values()
        .filter(|m| m.fired && m.cell_type == CellType::Associative)
        .map(|m| m.id).collect();
    patterns.push(fired);
}
// Jaccard(A, B) = |A ∩ B| / |A ∪ B| for consecutive pairs
```

**Interpretation:**

| Jaccard | Regime | Effect on readout |
|---------|--------|-------------------|
| 0.9-1.0 | Deterministic | Readout can converge (necessary condition) |
| 0.6-0.9 | Noisy-stable | Readout converges slowly, may oscillate |
| 0.3-0.6 | Semi-chaotic | Readout averages to noise, can't learn |
| 0.0-0.3 | Chaotic | Readout gradient averages to zero |

**Predictive value:** Jaccard < 0.7 → readout won't converge regardless of learning rule. Fix the stability first.

**Caveats:**
- `fired per trial: [0, 0, 0, 0, 0]` gives Jaccard=1.0 trivially (both sets empty). Check the fired count alongside Jaccard.
- The test input matters. Use a state the system will actually encounter during training, not an extreme value.

---

## 2. Output Discrimination (d+ / d-)

**What it measures:** Whether the readout maps opposite states to opposite actions.

**Protocol:**
```rust
system.reset_voltages();
let out_right = system.process_steps(&CartPole{theta: +0.1, ...}.observe(), INTERNAL_STEPS);
system.reset_voltages();
let out_left = system.process_steps(&CartPole{theta: -0.1, ...}.observe(), INTERNAL_STEPS);
let d_plus = out_right[1] - out_right[0];   // should be positive (push right)
let d_minus = out_left[1] - out_left[0];    // should be negative (push left)
```

**Interpretation:**

| d+ / d- | Meaning | Expected avg |
|---------|---------|-------------|
| Opposite signs, |d| > 5 | Strong discrimination | > 100 |
| Opposite signs, |d| = 1-5 | Weak discrimination | 40-100 |
| Same sign, different magnitude | Bias dominates | 20-40 |
| Same sign, similar magnitude | No discrimination | ~random |

**Predictive value:** This is the strongest single predictor of CartPole performance. When d+ and d- flip to opposite signs, avg starts climbing within 100 episodes.

**Tracking over time:** Log d+/d- every 100 episodes to monitor readout learning. A healthy system shows:
1. Random d+/d- early (ep 0-50)
2. Same-sign with growing difference (ep 50-100)
3. Sign flip (the critical transition)
4. Magnitude growth to ±10 (ep 100-500)

If d+/d- stays same-sign for 500+ episodes, the readout is stuck. Check: centered sigmoid? Bias term? Weight decay?

---

## 3. Policy Probe

**What it measures:** Policy accuracy on 4 test states with known correct actions.

**Protocol:**
```rust
let test_states = [
    (theta=+0.10, theta_dot= 0.0),  // should push right
    (theta=-0.10, theta_dot= 0.0),  // should push left
    (theta=+0.05, theta_dot=+0.5),  // should push right
    (theta=-0.05, theta_dot=-0.5),  // should push left
];
// For each: present input, check if argmax(output) matches correct action
```

**Interpretation:**

| Score | Meaning |
|-------|---------|
| 4/4 | Perfect policy on test states |
| 3/4 | Almost correct (one boundary case wrong) |
| 2/4 | Coin flip — no policy learned |
| 1/4 | Inverted policy (consistently wrong) |
| 0/4 | Fully inverted |

**Caveats:** The probe tests from cold voltage reset. If morphons need accumulated state to produce meaningful potentials (as in the current system where `fired per trial: [0,0,0,0,0]` from cold start), the probe underestimates policy quality. The d+/d- metric is more reliable.

---

## 4. Endoquilibrium Diagnostics

**What it measures:** Regulation state — what rules are active, what levers are pulled.

**Protocol:**
```rust
println!("{}", system.endo.summary());
// Output: endo: stage=Mature rg=1.50 ng=1.40 ag=1.00 hg=1.32 tb=0.039 pm=1.20 hp=0.72
```

**Interpretation:**

| Channel | If high | If low | Healthy |
|---------|---------|--------|---------|
| rg (reward_gain) | Rule 5: tags without captures | Normal | 1.0-1.5 |
| ng (novelty_gain) | Rule 2: eligibility low | Normal | 1.0-1.4 |
| ag (arousal_gain) | Rule 1: FR low | Normal | 1.0 |
| hg (homeostasis_gain) | Rule 1/2: FR/elig high | Normal | 1.0-1.3 |
| tb (threshold_bias) | FR slightly high | FR slightly low | -0.1 to 0.1 |
| pm (plasticity_mult) | Rule 2: elig low | Rule 6: energy pressure | 0.8-1.5 |
| hp (health_score) | n/a | Multiple vitals off | 0.7-1.0 |

**Warning signs:**
- `tb < -0.2`: FR deadlock — thresholds being aggressively lowered
- `pm < 0.3`: Energy pressure — system under stress
- `hp < 0.5`: Multiple systems failing simultaneously
- `rg = 3.0` (clamped at max): Captures completely stalled, Rule 5 maxed out

---

## 5. S→A Weight Distribution

**What it measures:** Diversity of sensory-to-associative feedforward weights.

**Protocol:**
```rust
let sa_weights: Vec<f64> = topology.all_edges().iter()
    .filter(|(from, to, _)| sensory_ids.contains(from) && assoc_ids.contains(to))
    .map(|(_, _, ei)| topology.edge_weight(ei).weight)
    .collect();
let std = statistical_std(&sa_weights);
```

**Interpretation:**

| weight_std | Meaning | Effect |
|-----------|---------|--------|
| > 0.3 | Good diversity | Different morphons respond to different inputs |
| 0.1-0.3 | Moderate | Some differentiation |
| < 0.1 | Near-uniform | All morphons respond identically (feature collapse) |

**Predictive value:** weight_std < 0.1 → the Associative layer provides no discriminative features to the readout. The readout must rely on sensory morphons for discrimination.

---

## 6. Consolidation Dynamics

**What it measures:** How many synapses are consolidated and at what level.

**From diagnostics:**
```
tags=350 con=440/556
```

**Interpretation:**
- `tags`: Number of synapses with active tags (candidates for capture)
- `con`: Number consolidated / total synapses

| Pattern | Meaning |
|---------|---------|
| tags=500 con=0 | Tags forming, no captures (capture threshold too high or gate not met) |
| tags=0 con=555 | Everything consolidated (premature — check capture mechanism) |
| tags=300 con=200 | Healthy: partial consolidation, ongoing exploration |
| Fluctuating con | Deconsolidation + recapture cycle (healthy dynamics) |

---

## 7. Recurrent Connectivity

**What it measures:** Proportion of Associative→Associative connections.

**Protocol:**
```rust
let recurrent = topology.all_edges().iter()
    .filter(|(from, to, _)| assoc_ids.contains(from) && assoc_ids.contains(to))
    .count();
```

**Relevance:** High recurrence (>20%) can cause chaotic dynamics where small perturbations are amplified. The cerebellar developmental program produces 0% recurrence (feedforward only). If recurrence develops through synaptogenesis, monitor Jaccard for stability degradation.

---

## 8. Recommended Diagnostic Suite

For CartPole optimization, log these every 100 episodes:

```
Ep  100 | steps  44 | avg(100)   39.1 | best 111 | policy=2/4 | d+=4.5 d-=-3.7
       endo: stage=Mature rg=1.50 ng=1.40 ag=1.00 hg=1.32 tb=0.039 pm=1.20 hp=0.72
       w=-0.97±1.71 e=0.20(154) tags=0 con=461/514 E=1.00 spk=490
```

This gives: performance (avg, best), readout quality (policy, d+/d-), regulation state (endo), learning pipeline health (weights, eligibility, tags, consolidation, energy, spikes).

The single most actionable signal is `d+/d-`. If d+ and d- have the same sign after 200 episodes, investigate the readout. If they have opposite signs and avg isn't climbing, investigate the capture/consolidation mechanism.
