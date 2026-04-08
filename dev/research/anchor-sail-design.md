# Anchor & Sail — Heterogeneous Plasticity Design

**Date:** 2026-04-01
**Status:** Design phase
**Depends on:** [CartPole Representational Drift Analysis](cartpole-representational-drift.md)
**Papers:** [Reference list](../paper/references-representational-drift.md)

---

## 1. The Problem (recap)

The hidden layer's representations drift under STDP + DFA, degrading the linear readout's learned mapping. Avg performance declines from ~18 to ~12 over 1000 episodes despite the system demonstrating it CAN learn (best single episodes: 95-106).

## 2. The Solution: Functional Heterogeneity

Not all morphons should be equally plastic. Introduce two populations:

### Anchor Morphons (~20-30% of Associative population)
- **Role:** Stable reference frame for the readout
- **Plasticity:** 10x slower STDP rate (`effective_lr = basal_plasticity * dynamic_plasticity`)
- **Membrane tau:** Longer (50-100 steps) — integrates over more context
- **STDP window:** Narrower — only responds to tight temporal coincidence
- **Protection:** Higher energy regen, resistant to pruning/apoptosis
- **Selection:** Top morphons by sustained `|feedback_signal|` or readout weight magnitude

### Sail Morphons (~70-80% of Associative population)
- **Role:** Explore state space, capture novel patterns
- **Plasticity:** Normal or elevated STDP rate
- **Membrane tau:** Shorter (5-20 steps) — fast transient detection
- **STDP window:** Normal width
- **Protection:** Normal metabolic budget, subject to standard lifecycle

### Biological Parallels
- **PV+ fast-spiking interneurons** (anchor): narrow STDP window, perisomatic inhibition, stabilize cortical dynamics
- **Pyramidal neurons** (sail): broad STDP, dendritic plasticity, learn new associations
- **Perez-Nieves 2021:** Optimal excitatory/inhibitory tau ratio ~3:1
- **Zenke 2015:** Three timescales (ms, seconds, minutes) necessary for stable memory formation

## 3. Implementation: Two Approaches

### Approach A: Basal + Dynamic Plasticity (Combined H1 + H2)

Add two fields to `Morphon`:

```
basal_plasticity: f64    // Set at birth, log-normal distribution [0.1, 2.0]
dynamic_plasticity: f64  // Modulated by readout importance, range [0.1, 1.0]
```

**Effective learning rate:** `basal_plasticity * dynamic_plasticity * alpha_reward * eligibility * modulation`

**Basal plasticity assignment:**
- At creation: draw from log-normal(mu=-0.5, sigma=0.8)
- This gives ~20% of morphons with plasticity < 0.3 (anchors)
- ~60% with plasticity 0.3-1.5 (normal)
- ~20% with plasticity > 1.5 (fast explorers)

**Dynamic plasticity modulation (from readout-coupled consolidation, H2):**
- After each `train_readout()`, compute importance per hidden morphon:
  `importance_i = max_j(|readout_weights[j][i]|) / max_readout_weight`
- Update dynamic_plasticity:
  `dynamic_plasticity *= (1 - 0.01) + 0.01 * (1 - importance)`
  (EMA that pushes high-importance morphons toward lower plasticity)
- Integration tau: ~1000 steps (glacial timescale, per iTDS paper: 100x learning rate)

**Effect:** Morphons that the readout finds useful get progressively stabilized. Morphons that are unused stay plastic and keep exploring.

### Approach B: AGMP-Style Stability Gate (Fourth Factor)

Add per-synapse stability gate (inspired by astrocyte-gated plasticity):

```
stability_gate: f64  // Slow EMA of |eligibility|, tau ~200 steps
```

**Modified three-factor rule:**
```
dw = eligibility * modulation * (1 - stability_gate)
```

When a synapse has been consistently active (high eligibility over time), the stability gate rises, reducing further plasticity. When the input distribution changes (eligibility drops), the gate decays, reopening plasticity.

**Numbers from AGMP paper:** Stability gate tau = 10x eligibility tau. If `tau_eligibility = 8`, then `tau_stability = 80`.

### Recommendation: Start with Approach A

Approach A is:
- Simpler to implement (two floats on Morphon)
- More aligned with the "Anchor & Sail" metaphor
- Readout-coupled (creates virtuous stability cycle)
- Compatible with existing three-factor rule (just scales the learning rate)

Approach B is more elegant (per-synapse, automatic) but harder to debug and tune.

## 4. Concrete Parameter Targets

From the literature:

| Parameter | Anchor (20%) | Normal (60%) | Sail (20%) |
|-----------|-------------|-------------|-----------|
| basal_plasticity | 0.1-0.3 | 0.3-1.5 | 1.5-2.0 |
| tau_membrane (if added) | 50-100 | 15-30 | 5-15 |
| STDP a_plus | 0.5 | 1.0 | 1.5 |
| Consolidation priority | High | Medium | Low |
| Apoptosis protection | High | Normal | None |

## 5. Expected Outcomes

- **Avg(100) stabilization:** Should plateau at 15-20+ instead of declining
- **Best episodes:** Should reach 100+ more reliably (not just lucky seeds)
- **Spike count stability:** Should not decline monotonically (anchors maintain baseline activity)
- **Readout weight stability:** Anchors provide stable features; readout weights to anchors should converge

## 6. Metrics to Track

- `anchor_count`: How many morphons are classified as anchors (dynamic_plasticity < 0.3)
- `anchor_firing_rate`: Firing rate of anchor vs. sail morphons
- `readout_weight_var`: Variance of readout weights over last 100 episodes (should decrease)
- `representation_cosine`: Cosine similarity of hidden activations for same input across time (should increase for anchors)

## 7. Missing Mechanisms (from Research, 2026-04-01)

The research revealed that Anchor & Sail alone may not be sufficient. Zenke 2015/2017 showed that **four** plasticity mechanisms must co-exist for stable memory formation. MORPHON has two of four:

| Mechanism | Status | What it does |
|-----------|--------|-------------|
| Triplet STDP (Hebbian) | **Have** — three-factor rule | Main learning |
| Slow consolidation | **Have** — tag-and-capture | Permanence |
| Transmitter-induced potentiation | **Missing** | Floor on weight change at low rates. Prevents silent death. |
| Heterosynaptic depression | **Missing** | When a morphon fires at high rate, ALL its incoming synapses get depressed. Prevents runaway. |

Critically, (3) and (4) must operate on the **same timescale as STDP** — not slower. Our homeostatic synaptic scaling operates on the slow timescale (every 50 steps), which is too late.

### H6: Fast Non-Hebbian Compensatory Plasticity (NEW, HIGH PRIORITY)
- **Transmitter-induced floor:** If pre-synaptic morphon is active but post rate is low, apply small positive `dw` regardless of STDP. Prevents morphons from going permanently silent.
- **Heterosynaptic depression:** When a morphon fires, depress ALL its incoming synapses by a small fraction, not just the active ones. This is independent of pre-synaptic activity — it normalizes total input.
- Both run on the MEDIUM path (every step), not the slow homeostasis path.

### H7: Multi-Level Weight Cascade (Benna-Fusi, MEDIUM PRIORITY)
Replace single `weight: f64` with three coupled variables:
```
w_fast: f64    // effective weight, changes via three-factor rule (tau ~ 1)
w_medium: f64  // slow shadow (tau ~ 100)
w_slow: f64    // glacial shadow (tau ~ 10000)
```
New memories written to w_fast. w_medium slowly tracks w_fast. w_slow slowly tracks w_medium. Old memories in w_slow resist overwriting because coupling is weak. Memory capacity: O(N) vs O(sqrt(N)).

### H8: Cross-Homeostatic Plasticity (Soldado-Magraner, MEDIUM PRIORITY)
Replace per-morphon homeostasis with crossed error signals: weights onto excitatory morphons use inhibitory population's error, and vice versa. Produces inhibition-stabilized networks with emergent soft-WTA. Requires tracking population-level firing rates per cell type.

## 8. Revised Implementation Priority

| # | What | Expected Impact | Effort |
|---|------|----------------|--------|
| 1 | H6: Fast non-Hebbian compensatory plasticity | **High** — prevents both silence and runaway | Low — two rules in medium path |
| 2 | H1+H2: Anchor & Sail + readout-coupled consolidation | **High** — stabilizes readout | Medium — two fields on Morphon |
| 3 | H7: Benna-Fusi weight cascade | **Medium** — protects old memories | Medium — 2 extra fields on Synapse |
| 4 | H8: Cross-homeostatic plasticity | **Medium** — better E/I balance | Medium — population-level tracking |
| 5 | Heterogeneous tau_membrane (Perez-Nieves) | **Medium** — 5-15% accuracy gain | Medium — per-morphon field |

**Recommended sprint:** H6 first (prevents the vicious silence cycle), then H1+H2 (stabilizes what's learned).

## 9. Open Questions

- Should anchors be selected at birth (fixed basal_plasticity) or earned through experience (dynamic_plasticity only)?
- How does anchor/sail interact with cell type differentiation? Should only Associative morphons be anchors, or also Stem/Modulatory?
- What happens to anchors during morphogenesis (division, migration)? Should anchors be immobile?
- Is 20% the right anchor fraction? Perez-Nieves says 10-20% form stable subspace. The drift literature says low-dimensional stable subspaces are sufficient.
- Should the Benna-Fusi cascade replace tag-and-capture, or complement it?
- Can cross-homeostatic plasticity be implemented via the existing neuromodulation broadcast channels?
