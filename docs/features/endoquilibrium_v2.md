# Endoquilibrium V2 — Widening the Regulatory Surface
## Feature Specification — TasteHub GmbH, April 2026

---

| | |
|---|---|
| **Depends on** | Endoquilibrium V1 (implemented) |
| **Prerequisite** | Local Inhibitory Competition (replaces global k-WTA — see Section 3) |
| **Interacts with** | Pulse Kernel Lite (docs/specs/pulse-kernel-lite-spec.md) |
| **Status** | Design / Discussion |
| **Author** | Lisa + Claude |

---

## 1. Motivation

Endoquilibrium V1 controls 6 levers: reward/novelty/arousal/homeostasis channel gains, global threshold bias, and plasticity rate multiplier. These cover the *learning dynamics* — firing rates, eligibility health, weight distribution, energy pressure.

But the system has dozens more static hyperparameters that CMA-ES currently has to search over as fixed values. Many of these *should* be dynamic — what's optimal at episode 10 is wrong at episode 500. The question is: should Endo control them, or does something else need to?

More fundamentally, some of the mechanisms Endo would need to regulate are themselves non-biological. The global k-WTA sort is the clearest example: no biological system uses a central authority to rank every neuron and decide who fires. Before Endo can regulate competition properly, competition itself needs to work like biology — local, emergent, based on real inhibitory signaling between cells that know their neighbors, not a god's-eye-view algorithm.

This document maps out:
1. The architectural prerequisite: replacing global k-WTA with local inhibitory competition
2. What's currently unregulated and should be dynamic
3. What belongs in Endo V2 vs. elsewhere

---

## 2. Current State: What Endo V1 Controls

| Lever | Actuator | Driven by |
|---|---|---|
| `reward_gain` | Scales reward broadcast amplitude | Rule 5: tag/capture health |
| `novelty_gain` | Scales novelty/plasticity signal | Rules 1, 2, 3: FR, eligibility, entropy |
| `arousal_gain` | Scales arousal/sensitivity signal | Rule 1: firing rate deficit |
| `homeostasis_gain` | Scales stability signal | Rules 1, 2, 3, 6: FR excess, eligibility excess, entropy explosion, energy |
| `threshold_bias` | Global offset on firing thresholds | Rule 1: firing rate regulation |
| `plasticity_mult` | Scales all weight update deltas | Rules 2, 3, 6: eligibility, entropy, energy |

These 6 levers are all **gain/bias modifiers** on the learning pipeline. Endo V1 does not touch structural plasticity, temporal dynamics, or competition mechanics.

---

## 3. Prerequisite: Local Inhibitory Competition (Replacing Global k-WTA)

### 3.1 What's Wrong with Global k-WTA

The current implementation (`system.rs:456-487`) does this:

```
1. Collect ALL associative/stem morphons
2. Sort globally by input_accumulator
3. Pick top-k (k = ceil(count * kwta_fraction))
4. Zero everyone else's input
5. Winners get hardcoded threshold += 0.02
```

This is a central planning algorithm, not lateral inhibition. Problems:

- **No locality.** Morphons in cluster A compete with morphons in cluster B, even if they represent completely different features. Biology doesn't work this way — competition is local.
- **The "k" is a parameter, not an emergent property.** In biology, the number of winners emerges from the balance of excitation and inhibition. Here it's set by a config value. This makes it a fake knob that Endo would be regulating artificially.
- **No communication.** Morphons don't know they're in competition. They don't send inhibitory signals. The system just silently zeros their input from above. Cells should suppress each other through real synaptic signaling.
- **It's instantaneous.** Real competition takes time — whoever fires first suppresses the rest. The global sort has no temporal dynamics.

### 3.2 How Biology Does It

In cortical circuits, competition works through **recurrent inhibition**:

1. Excitatory neurons receive input and begin depolarizing
2. They have local connections to nearby inhibitory interneurons
3. The fastest-depolarizing excitatory neuron fires first
4. Its connected inhibitory interneurons fire in response
5. Those interneurons suppress other excitatory neurons in the local circuit
6. The competition resolves in 2-3 ms — a few fast-path ticks

Key properties:
- **Local:** Competition happens within a circuit/cluster, not globally
- **Emergent:** The number of "winners" depends on inhibition strength, not a parameter
- **Temporal:** It's a race, not a ranking
- **Bidirectional:** Cells actively suppress each other through real synapses
- **Self-regulating:** Stronger input → faster firing → more inhibition → stable sparsity

### 3.3 What We Already Have

We're not starting from zero. The infrastructure for local inhibition exists:

- **Inter-cluster inhibitory morphons** (`morphogenesis.rs:645-755`): When clusters form, Modulatory morphons with negative-weight synapses are created between cluster pairs. These already suppress cross-cluster activity proportional to synchrony.
- **Inhibitory synapse weights** are already -0.3 by default, propagated through the normal spike delivery path.
- **`internal_steps`** (typically 5 sub-ticks per `process()` call) provides the multi-tick window needed for competition to resolve.
- **Cluster membership** (`morphogenesis.rs:Cluster::members`) defines the local neighborhoods.

What's missing: **intra-cluster inhibitory interneurons.** The existing inhibitory morphons sit *between* clusters. Competition *within* a cluster still uses the global sort.

### 3.4 The Design: Intra-Cluster Inhibitory Interneurons with iSTDP

Each cluster gets one or more local inhibitory interneurons. These are the same kind of Modulatory morphon that already exists for inter-cluster inhibition, but wired differently — and critically, their synapses are **plastic**.

**Wiring pattern:**
```
Excitatory member ──(+)──→ Local inhibitory interneuron ──(-)──→ All excitatory members
                                     │
                                     └── iSTDP adjusts (-) weights to maintain target FR
```

Every excitatory member of the cluster sends a positive-weight synapse to the local inhibitory interneuron. The interneuron sends negative-weight synapses back to all excitatory members (including the ones that activated it — this creates the negative feedback loop).

**How competition resolves across fast-path sub-ticks:**
```
Sub-tick 1: External input arrives. Strongest morphons accumulate most input.
Sub-tick 2: Strongest morphons fire. Their spikes reach local inhibitory interneurons.
Sub-tick 3: Inhibitory interneurons fire. Negative-weight spikes suppress remaining
            excitatory morphons that haven't fired yet.
Sub-tick 4-5: Suppressed morphons can't reach threshold. Competition resolved.
```

**Number of winners is emergent:**
- Strong inhibition → fewer winners → sparser representation
- Weak inhibition → more winners → denser representation
- The "sparsity" of the network is a function of inhibitory synapse strength, not a `kwta_fraction` parameter

**Inhibitory STDP (iSTDP) — the self-tuning mechanism:**

The key insight from Vogels et al. (2011, Science): inhibitory synapses don't need external tuning. They tune themselves via a local plasticity rule that maintains a target firing rate on the postsynaptic excitatory neuron:

```rust
// iSTDP rule (Vogels et al. 2011):
// On each postsynaptic spike: strengthen inhibitory synapse (too much firing)
// On each presynaptic spike without postsynaptic: weaken it (not enough firing)
//
// Δw_inh = η_inh * (post_trace - target_rate)
//
// If postsynaptic neuron fires too often → post_trace > target → Δw > 0 → more inhibition
// If postsynaptic neuron fires too rarely → post_trace < target → Δw < 0 → less inhibition
```

This is the mechanism that makes local competition self-regulating without Endo having to scale inhibitory weights directly. The `target_rate` parameter is what Endo controls — it sets the systemic target via `fr_assoc_min`/`fr_assoc_max`, and iSTDP does the local work of adjusting each inhibitory synapse to achieve that target for its specific postsynaptic morphon.

The biological analogy is exact: the endocrine system sets metabolic targets (cortisol level, blood glucose range), and local cellular mechanisms achieve them. The endocrine system doesn't reach into individual cells to turn dials.

**Why iSTDP, not static inhibitory weights:**
- Static weights require CMA-ES to find the right value and Endo to modulate it
- iSTDP finds the right value automatically per synapse per morphon
- Different morphons in the same cluster can have different effective inhibition (a morphon that receives strong input needs more inhibition than one that receives weak input)
- Adapts to structural changes (new members joining a cluster) without re-tuning

**Creation:** Extend `create_inhibitory_morphons_for_cluster()` (or add a parallel function) to also create intra-cluster inhibitory interneurons when a cluster forms. The number of inhibitory interneurons per cluster scales with cluster size: 1 per ~10-15 excitatory members. Brunel (2000) established ~20% inhibitory as the biological ratio; we start conservative and let iSTDP compensate.

**Position in Poincare ball:** Near the centroid of the cluster. Unlike Diehl & Cook (2015) where each inhibitory neuron connects to ALL excitatory neurons (global all-to-all inhibition implemented through synapses), Morphon uses spatially local inhibition where competition radius is determined by cluster membership in hyperbolic space. This is a key architectural distinction — see Section 3.11.

**Inhibitory synapse strength parameters:**
- Initial weight: -0.3 (same as existing inter-cluster inhibitory synapses, validated)
- iSTDP learning rate (η_inh): separate from excitatory STDP rate, typically 10x slower
- Target firing rate: set per developmental stage via Endo's `fr_assoc_min`/`fr_assoc_max`
- The initial weight is the starting point; iSTDP adjusts from there within bounds

### 3.5 What Gets Removed

- The global sort in `system.rs:456-487` — deleted entirely
- `HomeostasisParams::kwta_fraction` — no longer meaningful; there's no global "k" to set
- The `kwta_winners` vec on System — winners are identified by who actually fired after inhibition resolved, not by a pre-computed list
- The hardcoded `threshold += 0.02` for winners at `system.rs:593` — replaced by activity-dependent threshold adaptation (see 3.6)

### 3.6 What Stays (Reframed)

**Activity-dependent threshold adaptation (Diehl & Cook):** The biological principle is correct — morphons that fire frequently should become harder to fire, preventing winner-take-all monopolies. But instead of boosting the threshold of "winners" identified by a global sort, it applies to any morphon that fires:

```rust
// In Morphon::step(), after firing:
if self.fired_this_tick {
    self.threshold += self.winner_boost;  // base value, was hardcoded 0.02
}
// Slow decay back toward base threshold (already exists as homeostatic_setpoint)
self.threshold += (self.base_threshold - self.threshold) * threshold_decay_rate;
```

This is self-regulating: morphons that fire a lot get high thresholds, morphons that fire rarely get low thresholds. No external system needs to track "winners."

**`winner_boost` becomes a configurable base value** on `HomeostasisParams` (was hardcoded 0.02), searchable by CMA-ES. Endo V2 can then modulate it.

### 3.7 How Endo Regulates Local Competition (via iSTDP Targets)

With iSTDP-based local competition, Endo's role becomes cleaner than originally planned. It doesn't need a direct `inhibition_strength_mult` lever at all. Instead, it operates through a two-level control hierarchy:

**Level 1 — Endo (systemic, slow):** Sets target firing rates via `fr_assoc_min`/`fr_assoc_max` setpoints, which already adapt by developmental stage (Section 4.1 of the Endo V1 spec). These setpoints are the "metabolic targets" of the endocrine system.

**Level 2 — iSTDP (local, fast):** Each inhibitory synapse adjusts itself to achieve the target firing rate on its specific postsynaptic morphon. If a morphon fires above target, its incoming inhibitory synapses strengthen. If below, they weaken. No global coordination needed.

| Control layer | What it adjusts | Timescale | Biological analog |
|---|---|---|---|
| **Endo** `fr_assoc_min`/`fr_assoc_max` | iSTDP target rate | Medium path (every ~10 ticks) | Hypothalamic setpoints |
| **Endo** `threshold_bias` | Global excitability | Medium path | Cortisol |
| **Endo** `arousal_gain` | Sensitivity to all input | Medium path | Noradrenaline |
| **Endo** `winner_adaptation_mult` | How fast winners rotate | Medium path | Homeostatic plasticity rate |
| **iSTDP** | Individual inhibitory synapse weights | Every spike event | GABAergic synaptic plasticity |

The "number of winners" is now an **observable** that Endo senses (via `fr_associative`), not a parameter it sets. If firing rates drift outside the setpoint range, Endo adjusts the setpoints or threshold_bias, and iSTDP does the local work. This is a proper two-level feedback loop — systemic regulation sets targets, local mechanisms achieve them.

**Key consequence for Endo V2 lever inventory:** The `inhibition_strength_mult` lever proposed in the earlier draft is **no longer needed**. iSTDP replaces it with a more biologically correct mechanism. This simplifies the Endo V2 design — fewer new levers, cleaner separation of concerns.

### 3.8 Impact on Pulse Kernel Lite

PKL's k-WTA section (§5.2) assumed a global sort on `hot.voltage`. With local inhibitory competition:

- **The k-WTA function in PKL §5.2 is deleted.** No global sort needed.
- **Inhibitory interneurons are just morphons.** They participate in the normal fast-path integration and threshold check. PKL processes them the same as any other morphon — their negative-weight synapses propagate through the same edge iteration.
- **No special-case code.** The competition emerges from the same `fast_integrate()` → `fast_threshold_check()` → `fast_reset()` loop that processes everything else. This is simpler than the current PKL spec, not more complex.
- **Sync protocol unchanged.** Inhibitory interneurons are in petgraph like every other morphon. `rebuild_hot_arrays()` picks them up automatically.

The one addition: PKL's `compute_firing_rates()` (§5.5) should distinguish between excitatory and inhibitory firing rates when sensing vitals for Endo. Inhibitory interneuron firing is expected and healthy — it's not the same signal as "associative firing rate is too high."

### 3.9 Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Emergent "k" is unstable — oscillates between too-sparse and too-dense | High | iSTDP is specifically designed to stabilize this (Vogels et al. 2011 proved it). Endo's threshold_bias provides a second control surface. Start with strong clamping on inhibitory weight range. |
| iSTDP learning rate (η_inh) is a new hyperparameter | Medium | Vogels et al. found it's robust over a wide range. Start at 10x slower than excitatory STDP rate. Can be CMA-ES searched. |
| iSTDP + excitatory STDP interaction | Medium | Zenke et al. (2015) showed both are needed together for stable learning. Our existing excitatory STDP is three-factor (eligibility + modulation), which is more selective than vanilla STDP — should interact well with iSTDP. Validate empirically. |
| More morphons = more compute (inhibitory interneurons add ~10-20% to morphon count) | Low | At 300 morphons, 30-60 more is negligible. At scale, PKL hot arrays handle it. |
| More synapses = more memory (~6000 new edges for local inhibition at 300 morphons) | Low | Topology already handles 60K+ synapses growing to 110K. 6K more is within noise. |
| Competition takes 2-3 sub-ticks instead of being instantaneous | Medium | We already have `internal_steps = 5`. The temporal dynamics are a feature — they make competition sensitive to spike timing, which is biologically correct. |
| Harder to debug than global sort | Medium | Endo diagnostics already log interventions. Add `winners_per_cluster` and sparsity metrics (Section 11) for observability. |
| CartPole regression | High | A/B test with CompetitionMode enum (Section 11). Validate on CartPole first with 10 seeds before touching MNIST. If regression, tune iSTDP parameters — don't revert architecture. |

### 3.10 Implementation Sketch

| Step | What | Depends on |
|---|---|---|
| 1 | Add `winner_boost` to `HomeostasisParams` (extract hardcoded 0.02) | Nothing |
| 2 | Add `CompetitionMode` enum to `SystemConfig` (GlobalKWTA / LocalInhibition) — both modes coexist for A/B testing | Nothing |
| 3 | Implement iSTDP rule in `learning.rs` — `Δw_inh = η_inh * (post_trace - target_rate)` on inhibitory synapses only | Nothing |
| 4 | Create `create_local_inhibitory_interneurons()` — wires within-cluster inhibition during cluster formation, marks synapses for iSTDP | Nothing |
| 5 | Activity-dependent threshold adaptation in `Morphon::step()` (replaces winner-list-based boost) | Step 1 |
| 6 | Add `LocalInhibition` branch to step() — skips global sort, relies on inhibitory spike propagation + iSTDP | Steps 3, 4, 5 |
| 7 | Add validation metrics: population sparsity, lifetime sparsity, winner diversity entropy (Section 11) | Nothing |
| 8 | A/B benchmark: GlobalKWTA vs LocalInhibition, 10 seeds, CartPole quick profile | Steps 6, 7 |
| 9 | If CartPole validates: delete GlobalKWTA code path, `kwta_fraction`, `kwta_winners` | Step 8 |
| 10 | A/B benchmark: 10 seeds, MNIST standard profile | Step 9 |

**This is a prerequisite for Endo V2 Phase A**, not part of it. The local competition mechanism must be stable and validated before Endo starts modulating it.

### 3.11 Publishable Distinction: Local Hyperbolic Inhibition

Unlike Diehl & Cook (2015), who use global all-to-all inhibition (each inhibitory neuron connects to ALL excitatory neurons), Morphon uses spatially local inhibition where competition scope is determined by cluster membership in hyperbolic space. This is a meaningful architectural distinction:

- **Diehl & Cook:** 1 inhibitory neuron per excitatory neuron, each I→all-E connections. This is global WTA implemented through synapses — biologically implausible, functionally identical to a global sort.
- **Morphon:** Inhibitory interneurons within clusters, connections only to cluster members. Competition radius is an emergent property of the Poincare ball geometry — morphons near the origin (general) compete broadly, morphons near the boundary (specialized) compete locally within their cluster.
- **Hazan et al. (2018):** Distance-dependent inhibition on a 2D grid. Morphon extends this to hyperbolic space where distance has semantic meaning (generality ↔ specialization).

Combined with iSTDP (Vogels et al. 2011) for self-tuning inhibition strength, per-morphon astrocytic gating (AGMP-inspired, Section 4.1a), and Endo for systemic target-rate regulation, this gives a four-level control hierarchy that maps directly to biological timescales:

```
Governor (constitutional, static)          — hard limits, never violated
    └→ Endoquilibrium (systemic, medium-slow) — target rates, channel gains, stage detection
        └→ Astrocytic gate (per-morphon, slow)    — plasticity gating based on local activity
            └→ iSTDP + excitatory STDP (synaptic, fast) — weight adaptation per synapse
                └→ Spike dynamics (neural, fastest)       — competition via local inhibition
```

**Paper framing:** "Unlike Diehl & Cook (2015) who use global all-to-all inhibition implemented through synapses, MORPHON uses spatially local inhibition in hyperbolic space where competition radius is an emergent property of the Poincare ball geometry, self-tuned via inhibitory STDP (Vogels et al. 2011). Plasticity is further gated per-morphon by an astrocytic slow state (inspired by Dong & He, 2025), creating a four-level regulatory hierarchy from constitutional constraints to spike dynamics."

---

## 4. What's Static Today and Shouldn't Be

### 4.1 High-Value Targets (extend Endo — Phase A)

These params directly map to vitals Endo already senses. The regulation rules follow naturally from existing rule structure.

#### iSTDP target firing rate (replaces `kwta_fraction` and `inhibition_strength_mult`)
- **Currently (post-Section 3):** iSTDP on inhibitory synapses self-tunes inhibition strength to maintain a target firing rate per postsynaptic morphon.
- **How Endo controls it:** Endo already has `fr_assoc_min`/`fr_assoc_max` setpoints that adapt by developmental stage (Rule 1 in Endo V1). These setpoints become the iSTDP target rate. No new lever needed — Endo's existing firing rate regulation rule naturally drives the iSTDP targets.
- **What changed:** The earlier draft proposed `inhibition_strength_mult` as a new Endo lever. With iSTDP, this lever is unnecessary. iSTDP adjusts each inhibitory synapse individually to achieve the target rate. Endo sets the target systemically. This is the cleaner separation: endocrine = slow systemic targets, synaptic plasticity = fast local adaptation.
- **Stage-dependent behavior (already in Endo V1):** Proliferating: `fr_assoc_min`=0.12 (more winners, exploration). Mature: `fr_assoc_min`=0.08 (fewer winners, discrimination). iSTDP automatically adjusts inhibition to match.

#### Activity-dependent threshold adaptation rate
- **Currently (post-Section 3):** `winner_boost` base value in `HomeostasisParams`.
- **Problem:** During Proliferating stage, winner rotation should be fast — high adaptation rate means today's winners become harder to fire tomorrow, giving other morphons a chance. During Mature stage, winner stability matters more — low adaptation rate lets established feature detectors keep winning.
- **Endo regulation:** New lever `winner_adaptation_mult` (multiplier on `winner_boost`). Proliferating: high (1.5-2.0). Mature: low (0.5-0.8).
- **Vital already sensed:** developmental stage, `fr_associative`.

#### `LearningParams::capture_threshold`
- **Currently:** Static value.
- **Problem:** If the threshold is too high, tags expire without capture and learning stalls. Endo Rule 5 already detects this (`tag_count > 100 && capture_count == 0`) but only boosts `reward_gain` — it doesn't touch the capture threshold itself.
- **Endo regulation:** If `ticks_since_last_capture` exceeds threshold, progressively lower `capture_threshold` (with a floor). Reset when captures resume. This closes the loop that Rule 5 currently leaves half-open.
- **Vital already sensed:** `tag_count`, `capture_count`, `ticks_since_last_capture`.

#### `HomeostasisParams::rollback_pe_threshold`
- **Currently:** Static 0.2.
- **Problem:** During Proliferating stage, PE naturally fluctuates more — a fixed 0.2 threshold triggers false rollbacks that undo healthy structural exploration. During Mature stage, even small PE increases are suspicious.
- **Endo regulation:** Stage-dependent. Proliferating: relax to 0.4. Mature: tighten to 0.1. Stressed: tighten further to 0.05.
- **Vital already sensed:** developmental stage, `prediction_error_mean`.

### 4.1a Per-Morphon Astrocytic Gate (Phase A+ — after iSTDP validates)

The AGMP paper (Dong & He, Frontiers in Neuroscience, Dec 2025) introduces a **four-factor learning rule**: `eligibility × modulation × astrocytic_gate × stabilization`. We already have three of the four factors (eligibility, modulation, and Endo's `plasticity_mult` as global stabilization). The missing piece is the **per-morphon slow state variable** — an astrocytic gate that suppresses plasticity on well-established morphons while allowing it on morphons that are still adapting.

#### The mechanism

Each morphon gets a slow state variable `astrocytic_state` (one new `f64` field) that integrates recent activity over a long timescale (τ_a = 500-1000 ticks, much slower than the neuronal membrane τ_m = 20 ticks):

```rust
// Updated every medium tick, in Morphon::step() or a dedicated function:
// a_i(t+1) = (1 - 1/τ_a) * a_i(t) + (1/τ_a) * (η_v * V + η_s * I_syn + η_f * fired)
self.astrocytic_state = self.astrocytic_state * (1.0 - 1.0 / tau_a)
    + (1.0 / tau_a) * (
        eta_v * self.potential
        + eta_s * self.input_accumulator
        + eta_f * if self.fired_this_tick { 1.0 } else { 0.0 }
    );

// Gate: sigmoid maps astrocytic_state to [0, 1]
// High activity → high gate → plasticity allowed
// Silent morphon → low gate → plasticity suppressed (don't waste learning on non-participants)
let gate = 1.0 / (1.0 + (-self.astrocytic_state + threshold_a).exp());

// Applied to weight updates:
// Δw_ij = eligibility_ij × modulation × gate_i × plasticity_mult
```

#### Why this matters

- **Continual learning / catastrophic forgetting:** When MNIST moves to sequential digit classes, well-established feature detectors (high `astrocytic_state`, high gate) continue to learn and adapt. Silent morphons (low gate) don't get spurious weight updates from noise. This is the per-morphon analog of what Endo's `plasticity_mult` does globally.
- **Interaction with iSTDP is clean:** iSTDP adjusts inhibitory synapse weights (who competes with whom). The astrocytic gate modulates excitatory synapse updates (who learns what). Different synapses, different rules, orthogonal concerns.
- **Interaction with Endo is hierarchical:** Endo sets `plasticity_mult` (systemic). The astrocytic gate further modulates per-morphon. Final weight update = `eligibility × modulation × gate_i × plasticity_mult`. Endo can't see individual gates, and doesn't need to — it trusts the local mechanism.

#### Competitive positioning for the paper

AGMP uses a single slow astrocytic timescale. Endoquilibrium uses dual-timescale EMAs (fast τ=50, slow τ=500) with developmental stage detection. AGMP gates plasticity per-neuron; Morphon combines per-morphon gating (astrocytic state) with systemic regulation (Endo channel gains). The hierarchy is:

| Level | Timescale | Scope | Mechanism |
|---|---|---|---|
| Spike dynamics | ~1 tick | Per-synapse | Local inhibition, threshold check |
| Synaptic plasticity | ~10 ticks | Per-synapse | iSTDP (inhibitory), three-factor STDP (excitatory) |
| Astrocytic gate | ~500-1000 ticks | Per-morphon | Slow activity integration, sigmoid gating |
| Endoquilibrium | ~50-500 ticks | Systemic | Dual-EMA predictor, 6+ channel gains |
| Governor | Static | Constitutional | Hard limits |

#### Implementation

- **1 new field on Morphon:** `astrocytic_state: f64` (default 0.5)
- **3 new config params:** `tau_a`, `eta_v/eta_s/eta_f` coefficients, `threshold_a`
- **~15 lines in learning.rs:** Update astrocytic state on medium tick, multiply into weight update delta
- **CMA-ES searchable:** `tau_a`, `threshold_a`, and the η coefficients. The sigmoid shape determines how sharply the gate transitions from "no plasticity" to "full plasticity."

### 4.2 Medium-Risk Targets (Phase B — extend Endo with care)

These affect structural plasticity. Misregulation is harder to recover from than a bad plasticity_mult — you can't un-divide a morphon.

#### `MorphogenesisParams` — division/pruning/synaptogenesis thresholds
- **Currently:** Static thresholds for when morphons divide, synapses get pruned, new synapses form.
- **Problem:** During Proliferating stage, division should be easier and pruning harder. During Consolidating, the inverse. Currently CMA-ES finds a compromise that's optimal for neither phase.
- **Endo regulation:** Rule 4 (cell type balance) currently logs imbalances but takes no action (`lever: "logged_only"`). V2 should make this real:
  - Overrepresented type -> raise division threshold for that type, lower pruning threshold
  - Underrepresented type -> lower division threshold
  - Stage-dependent base rates: Proliferating favors division, Consolidating favors pruning
- **Risk:** Structural changes are irreversible within a checkpoint window. The checkpoint/rollback mechanism (homeostasis.rs D) is the safety net, but Endo's rollback_pe_threshold regulation (4.1 above) needs to land first.
- **Vital needed:** Add `division_rate` and `pruning_rate` to VitalSigns (events per 100 ticks).

#### `FrustrationConfig` — thresholds and response strengths
- **Currently:** Static frustration sensitivity.
- **Problem:** A system in Proliferating stage should have higher frustration tolerance (it's exploring, failures are expected). A Mature system should be more sensitive to frustration (something broke).
- **Endo regulation:** Scale frustration threshold with developmental stage. Proliferating: high tolerance. Mature: low tolerance.
- **Vital needed:** Add `frustration_level` to VitalSigns (already tracked per-morphon, just needs aggregation).

### 4.3 Needs Thought (Phase C — narrow band only)

These touch fundamental dynamics. Getting them wrong doesn't just cause suboptimal performance — it can destroy the learning pipeline entirely.

#### `LearningParams::a_plus / a_minus` ratio (STDP shape)
- **Currently:** Static STDP window shape.
- **Problem:** The LTP/LTD balance determines whether the network trends toward strengthening or weakening. A fixed ratio can cause weight saturation (too much LTP) or weight collapse (too much LTD).
- **Endo regulation candidate:** Endo already senses `weight_entropy`. If entropy is collapsing (all weights -> max), shift a_plus/a_minus ratio toward LTD. If entropy is exploding, shift toward LTP. But: the STDP shape interacts with everything else. A narrow band (base_ratio +/- 20%) is safer than full control.
- **Open question:** Is this better handled by the existing `plasticity_mult` lever, which already dampens/boosts all learning? Adding ratio control on top of rate control might create conflicting signals. Need to think about whether the failure modes that `plasticity_mult` can't fix actually exist in practice.

#### `LearningParams::tau_e` (eligibility trace time constant)
- **Currently:** Static trace decay.
- **Problem:** tau_e determines how long the credit assignment window stays open. Too short: reward arrives too late, no learning. Too long: spurious associations.
- **Endo regulation candidate:** If captures are happening quickly (reward latency is low), tau_e could shorten for precision. If captures are slow/absent, tau_e could lengthen to keep the window open.
- **Risk:** tau_e interacts with the tag-and-capture mechanism in subtle ways. A longer tau_e means more tags compete for the same capture event, which can dilute credit assignment. This needs simulation data before committing to a regulation rule.

#### `SchedulerConfig` — tick periods
- **Currently:** Fixed periods for fast/medium/slow/glacial paths.
- **Idea:** During Stressed stage, throttle the glacial path (pause morphogenesis). During Mature, lengthen the medium path interval (less frequent learning updates save compute).
- **Risk:** Changing temporal dynamics at runtime is inherently destabilizing. The dual-clock architecture assumes fixed ratios between paths. Changing the slow path period from 100 to 200 means morphogenesis now runs at half frequency, which changes pruning/division balance in non-obvious ways.
- **Recommendation:** Don't regulate tick periods. If throttling is needed, use the existing actuators: boost `homeostasis_gain` to dampen activity, or drop `plasticity_mult` to zero (which Rule 6 already does under energy pressure). The effect is similar without changing temporal structure.

---

## 5. What Does NOT Belong in Endo

Some things need dynamic control but should not be Endo levers. The principle: **Endo modulates the environment in which the Builder operates.** It adjusts gains, thresholds, rates — it never makes discrete structural decisions or controls external interfaces.

### 5.1 Governor / Constitutional Layer

The Governor (V3, not yet implemented) sets hard limits: max morphon count, min/max cluster count, forbidden topology patterns, resource budgets. Endo must operate *within* Governor constraints, never override them.

**What the Governor would control:**
- Hard morphon count ceiling/floor
- Maximum synapse fan-in/fan-out per morphon
- Cell type ratio hard limits (vs. Endo's soft targets)
- Energy budget allocation between subsystems
- Morphogenesis rate limits (max divisions per glacial tick)

**Why not Endo:** These are *constitutional* constraints, not *regulatory* targets. Endo optimizes within a range. The Governor defines the range. Mixing them in one system means a regulation rule could accidentally override a safety constraint.

### 5.2 Task Adaptation / Curriculum Controller

**What it would control:**
- Input encoding strategy (rate coding vs. population coding)
- Output decoding strategy (WTA vs. softmax vs. readout)
- Reward shaping parameters (contrastive reward strength, reward delay)
- Episode boundaries and reset behavior

**Why not Endo:** These are *task-level* decisions, not *organism-level* regulation. Endo doesn't know what task the system is solving — it only knows whether the organism is healthy. A curriculum controller would sit *above* Endo and the Governor, deciding what challenges to present and how to frame them.

### 5.3 Dream/Replay Orchestration

`DreamConfig` has params for replay rate, consolidation strength, memory selection strategy. These are currently static.

**Why separate from Endo:** Dreaming is a discrete *mode* the system enters, not a continuous gain adjustment. The decision "should we dream now?" is closer to a scheduler decision than a regulation decision. Endo could provide *input* to that decision (health score, developmental stage), but the orchestration logic — when to dream, how long, what to replay — belongs in a sleep/wake controller that coordinates with the main processing loop.

That said: once dreaming is active, Endo *should* regulate the consolidation strength and replay rate the same way it regulates learning during wake — via `plasticity_mult` and `novelty_gain`. The point is that Endo doesn't decide *when* to dream, but it does regulate *how intensely*.

### 5.4 External Interface Adaptation

- Python/WASM binding configuration
- Diagnostic verbosity levels
- Snapshot frequency

These are operational concerns, not biological regulation. They belong in a runtime configuration layer, not in a neuroendocrine model.

---

## 6. Proposed Endo V2 Lever Inventory

### New levers (Phase A — high value, low risk)

Note: `inhibition_strength_mult` was proposed in an earlier draft but is no longer needed. iSTDP on inhibitory synapses (Section 3.4) self-tunes inhibition strength using Endo's existing `fr_assoc_min`/`fr_assoc_max` setpoints as targets. No new Endo lever required for competition regulation.

| Lever | Type | Range | Default | Regulation Rule |
|---|---|---|---|---|
| `winner_adaptation_mult` | Multiplier on activity-dependent threshold boost (`winner_boost`) | [0.3, 2.5] | 1.0 | Stage-dependent: high during Proliferating (rotate winners), low during Mature (stable features) |
| `capture_threshold_mult` | Multiplier on base capture_threshold | [0.5, 1.5] | 1.0 | Tag/capture health (Rule 5 extension) |
| `rollback_pe_threshold_mult` | Multiplier on base rollback_pe_threshold | [0.25, 2.0] | 1.0 | Stage-dependent |

### New levers (Phase B — medium risk, after Phase A validates)

| Lever | Type | Range | Default | Regulation Rule |
|---|---|---|---|---|
| `division_threshold_mult` | Multiplier on morphogenesis division threshold | [0.5, 2.0] | 1.0 | Stage + cell type balance |
| `pruning_threshold_mult` | Multiplier on morphogenesis pruning threshold | [0.5, 2.0] | 1.0 | Stage + cell type balance |
| `frustration_sensitivity_mult` | Multiplier on frustration thresholds | [0.5, 2.0] | 1.0 | Stage-dependent |

### Deferred (Phase C — needs simulation data)

| Lever | Concern |
|---|---|
| `stdp_ratio_bias` | May conflict with `plasticity_mult`; need evidence of failure modes it uniquely fixes |
| `tau_e_mult` | Interacts with tag-and-capture in non-obvious ways; needs controlled experiments |
| Nonlinear neuromodulation (NACA-inspired) | See below |

#### Nonlinear neuromodulation — polarity reversal at high modulation

NACA (Science Advances, 2023) shows that the biological relationship between neuromodulator levels and plasticity is nonlinear with an inversion point — at very high dopamine levels, LTP becomes LTD. Our four-channel neuromodulation is currently linear (`gain × base_signal`). This matters for the "always Stressed" problem: at high modulation levels, the system should *reverse* plasticity direction, not just amplify it.

The investigation needed: how does a nonlinear modulation function (e.g., inverted-U or bell curve) interact with our existing three-factor rule + astrocytic gate? The risk is that adding nonlinearity to modulation while the astrocytic gate is also modulating plasticity creates interactions that are hard to predict analytically. Needs controlled simulation with the A/B benchmarking framework (Section 11) before committing.

### Explicitly excluded from Endo

| Parameter | Belongs in |
|---|---|
| Hard morphon/synapse limits | Governor (V3) |
| Input/output encoding strategy | Task/Curriculum Controller |
| Dream scheduling (when, how long) | Sleep/Wake Orchestrator |
| Replay intensity during dreams | Endo (via existing `plasticity_mult` / `novelty_gain`) |
| Diagnostic/snapshot config | Runtime Configuration |

---

## 7. Architecture: One Controller, Wider Surface

The recommendation is to **extend Endo, not add a second regulation service.** Two parallel controllers adjusting overlapping state will oscillate against each other — that's the exact instability Endo exists to prevent.

The new levers follow the same pattern as V1:
- All actuators are **multipliers on base values**, not absolute overrides
- Base values come from CMA-ES optimization or manual tuning
- Endo narrows the CMA-ES search space by compensating for parameter sensitivity
- All adjustments are **smoothed** (EMA) and **clamped** (hard bounds)
- All interventions are **logged** with rule, vital, actual, setpoint, lever, adjustment
- Governor can override any Endo output

The `ChannelState` struct grows from 6 to 12 fields (Phase A+B). Each new lever needs:
1. A field in `ChannelState` with default 1.0, a clamp range, and smoothing
2. A regulation rule in `regulate()` that maps vitals to adjustments
3. An integration point where the multiplier is applied (in the consuming module)
4. A test that the rule fires when the triggering condition is met

---

## 8. New Vitals Needed

Phase A adds one new vital:

| Vital | Source | Cost |
|---|---|---|
| `winners_per_cluster` | Count morphons that fired per cluster after inhibition resolved | O(N) scan, piggybacks on existing FR sensing |

Phase B adds three:

| Vital | Source | Cost |
|---|---|---|
| `division_rate` | Count division events per 100 glacial ticks | Cheap — increment counter in morphogenesis |
| `pruning_rate` | Count pruning events per 100 glacial ticks | Same |
| `frustration_mean` | Average frustration level across morphons | O(N) scan, already done in vitals sensing |

---

## 9. Interaction with Pulse Kernel Lite

The Pulse Kernel Lite spec (docs/specs/pulse-kernel-lite-spec.md) extracts hot arrays for cache-friendly fast-path processing. With local inhibitory competition (Section 3), the interaction becomes simpler than the original PKL spec assumed.

### What changes in PKL

**PKL §5.2 (k-WTA) is deleted entirely.** There's no global sort to optimize. Local inhibitory interneurons are just morphons — they participate in the same `fast_integrate()` → `fast_threshold_check()` → `fast_reset()` loop as everything else. Their negative-weight synapses propagate through the same petgraph edge iteration. No special-case code.

### What stays the same

| Concern | Owned by | Timescale |
|---|---|---|
| Spike integration, threshold check, reset (including inhibitory interneurons) | **PKL** (hot arrays) | Fast path (every tick) |
| iSTDP on inhibitory synapses | **Learning** (runs on medium path like excitatory STDP) | Medium path |
| Astrocytic gate update + gated excitatory STDP | **Learning** (per-morphon slow state, gates Δw) | Medium path |
| Firing rate targets that drive iSTDP | **Endo V2** (existing `fr_assoc_min`/`fr_assoc_max` setpoints) | Medium path |
| Firing rate sensing, eligibility density, weight entropy | **Endo V2** (reads from synced structs or hot arrays) | Medium path |
| Threshold adjustments (homeostatic, Endo bias, adaptation boost) | **Endo V2** writes to structs, **PKL** syncs to `hot.threshold` | Medium path sync-up |
| Structural changes (division, pruning, migration, new inhibitory interneurons) | **Morphogenesis** (existing), rates modulated by **Endo V2** | Slow path |
| Hot array rebuild after structural changes | **PKL** (`rebuild_hot_arrays()`) | Slow path |

### How the four-level hierarchy flows through PKL

```
Medium path (Endo tick):
  1. Endo senses vitals (FR per cluster, stage, winners_per_cluster)
  2. Endo adjusts fr_assoc_min/fr_assoc_max setpoints by developmental stage
  3. Endo computes winner_adaptation_mult, plasticity_mult, writes to ChannelState

Medium path (learning):
  4. Update per-morphon astrocytic_state (slow EMA of activity)
  5. Compute astrocytic gate: gate_i = sigmoid(astrocytic_state_i - threshold_a)
  6. iSTDP runs on inhibitory synapses: Δw = η_inh * (post_trace - target_rate)
     target_rate = Endo's current fr_assoc setpoint
  7. Excitatory STDP: Δw = eligibility × modulation × gate_i × plasticity_mult
  8. Activity-dependent threshold: boost = winner_boost * endo.winner_adaptation_mult

Medium path (sync up):
  9. Updated thresholds (including adaptation boost) synced to hot.threshold
  10. Updated synapse weights (from iSTDP + gated excitatory STDP) already in petgraph

Fast path (PKL):
  9. Normal integration — inhibitory synapses deliver current from their iSTDP-adjusted weights
  10. Normal threshold check — morphons that reach threshold fire
  11. Competition emerges from the dynamics, no special code
```

### Implementation order

1. **Endo V1** — implemented, validate on CartPole
2. **Local Inhibitory Competition** (Section 3) — prerequisite: intra-cluster interneurons + iSTDP + CompetitionMode A/B framework
3. **Endo V2 Phase A** — winner_adaptation_mult, capture_threshold_mult, rollback_pe_threshold_mult
4. **Endo V2 Phase A+** — per-morphon astrocytic gate (Section 4.1a) — 1 field, ~15 lines, high paper value
5. **PKL** — extract hot arrays; k-WTA section deleted, inhibitory interneurons + astrocytic state handled by normal paths
6. **Endo V2 Phase B** — morphogenesis rate control (division/pruning/frustration levers)

---

## 10. Open Questions

1. **How many inhibitory interneurons per cluster?** Brunel (2000) established ~20% inhibitory as the cortical ratio. At 300 morphons with ~5 clusters of 30, that's 6 inhibitory interneurons per cluster — 30 total new morphons. The ratio needs to be a morphogenesis parameter, not hardcoded. Starting point: 1 per 10 excitatory members, minimum 1 per cluster. iSTDP compensates if the ratio is imperfect.

2. **iSTDP learning rate (η_inh) relative to excitatory STDP.** Vogels et al. (2011) used a separate learning rate for inhibitory plasticity, typically slower than excitatory. Our existing three-factor STDP already has a learning rate modulated by `plasticity_mult`. Should iSTDP use the same base rate scaled by a fixed ratio (e.g., 0.1x)? Or should it have its own independent rate? The latter is more flexible but adds a CMA-ES parameter.

3. **iSTDP + three-factor interaction.** Our excitatory STDP is three-factor (eligibility * modulation), not vanilla STDP. Zenke et al. (2015) showed excitatory STDP + iSTDP + homeostatic scaling work together. But their excitatory STDP was vanilla, not three-factor. Need to verify empirically that iSTDP's rate-targeting doesn't fight our three-factor credit assignment. The A/B benchmark (Section 11) is the test.

4. **Rule 4 (cell type balance) currently takes no action** — it logs but has `lever: "logged_only"`. For Phase B, it needs to actually inject differentiation pressure. The mechanism for this (`system.inject_differentiation_pressure()`) doesn't exist yet. What should it do? Options: (a) raise division threshold for overrepresented type, (b) bias differentiation toward underrepresented types during existing division events, (c) trigger targeted apoptosis. Option (b) is least invasive.

5. **Should the Governor exist before Endo V2 Phase B ships?** Phase B (morphogenesis control) is more dangerous without Governor hard limits. If Endo bugs out and sets `division_threshold_mult = 0.5` indefinitely, the network could double in size with no backstop. The clamp range [0.5, 2.0] limits this, but a Governor ceiling would be safer.

6. **Is the multiplier pattern always right?** For `capture_threshold`, lowering it by 50% (mult=0.5) might not be enough if the base value is already too high. Should some levers be additive biases instead of multipliers? The risk: additive biases don't scale with the base value, so they interact differently with CMA-ES optimization.

7. **The `internal_steps` window.** Competition needs 2-3 sub-ticks to resolve. Currently `internal_steps = 5`, which should be enough. But if a task needs fewer internal steps (for throughput), competition might not fully resolve. Should we guarantee a minimum `internal_steps` when local competition is active? Or is incomplete competition acceptable (some steps have more winners, some fewer — averaging out)?

---

## 11. Validation and Benchmarking Plan

### 11.1 The A/B Testing Framework

A `CompetitionMode` enum allows both mechanisms to coexist for direct comparison:

```rust
pub enum CompetitionMode {
    /// Current implementation: global sort, top-k, zero losers.
    GlobalKWTA { fraction: f64 },
    /// New: intra-cluster inhibitory interneurons with iSTDP.
    LocalInhibition {
        interneuron_ratio: f64,      // inhibitory per excitatory (default 0.1)
        istdp_rate: f64,             // iSTDP learning rate (default 0.001)
        initial_inh_weight: f64,     // starting inhibitory weight (default -0.3)
    },
}
```

Both modes share everything else: same developmental bootstrap, same excitatory STDP, same Endo regulation, same reward pipeline. The only difference is how competition works.

### 11.2 Test Protocol

**Phase 1: CartPole regression check (blocks further work)**

| Parameter | Value |
|---|---|
| Task | CartPole |
| Profile | quick (200 episodes) |
| Seeds | 10 per mode |
| Pass criterion | LocalInhibition avg_last_100 not significantly worse than GlobalKWTA (Welch's t-test, p < 0.05) |
| Stretch goal | avg_last_100 >= 195.0 (solved) for at least 5/10 seeds |

**Phase 2: MNIST accuracy comparison**

| Parameter | Value |
|---|---|
| Task | MNIST (mnist_v2) |
| Profile | standard |
| Seeds | 10 per mode |
| Metrics | test_accuracy, per-class accuracy, convergence speed (episodes to 80%) |
| Key question | Does local inhibition improve class discrimination? Winner diversity entropy is the diagnostic. |

**Phase 3: Diagnostic deep-dive (both phases)**

Run both CartPole and MNIST with full diagnostic logging, compare:
- Population sparsity, lifetime sparsity, winner diversity entropy
- iSTDP convergence (do inhibitory weights stabilize?)
- Endo intervention frequency (does Endo need to intervene more or less?)
- Compute overhead (wall-clock per step)

### 11.3 Three New Metrics

These metrics are cheap to compute from data already collected and directly measure competition quality.

#### Population Sparsity (Treves-Rolls, per step)

Measures "how many fire per stimulus" — the instantaneous sparseness of the population response.

```rust
/// Treves-Rolls population sparsity. Returns 0.0 (dense) to 1.0 (maximally sparse).
/// Computed over associative morphon activations for the current step.
fn population_sparsity(activations: &[f64]) -> f64 {
    let n = activations.len() as f64;
    if n <= 1.0 { return 0.0; }
    let mean = activations.iter().sum::<f64>() / n;
    let mean_sq = activations.iter().map(|a| a * a).sum::<f64>() / n;
    if mean_sq < 1e-10 { return 0.0; }
    (1.0 - (mean * mean) / mean_sq) / (1.0 - 1.0 / n)
}
```

Computed per step. Track as running mean + variance in diagnostics. Source: existing `assoc_activity_min/max/mean` in `diagnostics.rs` — extend to full Treves-Rolls.

#### Lifetime Sparsity (Treves-Rolls, per morphon)

Measures "how selective is each neuron" — does it fire for everything (low sparsity) or only specific inputs (high sparsity)?

Same formula, but computed per morphon across its `activity_history` ring buffer (already exists in `types.rs`). Report as distribution statistics (mean, min, max, std across all associative morphons).

**Why both matter:** High population sparsity alone can mean a few morphons dominate. High lifetime sparsity means each morphon is selective. You want both high — that's the hallmark of good competitive learning.

#### Winner Diversity Entropy

Measures "do different inputs activate different subsets" — the key diagnostic for competition health.

```rust
/// Shannon entropy of the winner frequency distribution.
/// Uniform = all morphons contribute equally = max entropy.
/// One morphon dominates = near-zero entropy.
fn winner_diversity_entropy(winner_counts: &[u32]) -> f64 {
    let total: f64 = winner_counts.iter().map(|&c| c as f64).sum();
    if total < 1.0 { return 0.0; }
    let mut entropy = 0.0;
    for &count in winner_counts {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.ln();
        }
    }
    entropy
}
```

Track per-morphon firing counts over an episode, compute at episode end. Report in benchmark JSON alongside existing metrics.

**The paper figure:** If winner diversity entropy is near zero with GlobalKWTA and significantly positive with LocalInhibition, that's the headline result — local competition produces more diverse representations.

### 11.4 Supplementary Metrics

| Metric | What it measures | When to use |
|---|---|---|
| Winner count variance | Competition stability — does "k" oscillate? | Debugging, not headline |
| Neuron utilization | Fraction of morphons that ever win (target: close to 1.0) | Diehl & Cook diagnostic |
| Settling time | Sub-ticks until winner set stabilizes after input | Unique to LocalInhibition mode |
| iSTDP weight distribution | Are inhibitory weights converging or diverging? | Debugging iSTDP health |
| Step wall-clock time | Compute overhead of local inhibition | Performance regression check |

### 11.5 Result Format

Extend existing benchmark JSON with competition metrics:

```json
{
  "competition": {
    "mode": "LocalInhibition",
    "population_sparsity_mean": 0.85,
    "population_sparsity_std": 0.03,
    "lifetime_sparsity_mean": 0.72,
    "lifetime_sparsity_std": 0.08,
    "winner_diversity_entropy": 3.41,
    "neuron_utilization": 0.94,
    "avg_winners_per_step": 12.3,
    "winner_count_cv": 0.15,
    "inh_weight_mean": -0.28,
    "inh_weight_std": 0.09,
    "avg_settling_ticks": 2.4,
    "step_time_ms": 0.015
  }
}
```

### 11.6 Statistical Comparison

For each metric, across N=10 seeds per mode:

```
Welch's t-test on accuracy distributions → p-value + effect size (Cohen's d)
Mann-Whitney U on sparsity distributions (non-parametric fallback)
```

Report in a comparison summary JSON alongside individual run results:

```json
{
  "comparison": {
    "task": "cartpole",
    "n_seeds": 10,
    "global_kwta_avg": 195.2,
    "local_inhibition_avg": 198.4,
    "p_value": 0.23,
    "cohens_d": 0.31,
    "regression": false,
    "winner_entropy_improvement": 2.8
  }
}
```

---

## 12. References

### Competition and Inhibition

| Paper | Key finding for Morphon |
|---|---|
| **Vogels et al. (2011)** — "Inhibitory Plasticity Balances Excitation and Inhibition in Sensory Pathways and Memory Networks" (Science) | iSTDP: inhibitory synapses self-tune to maintain target firing rate. The mechanism that makes local competition self-regulating. |
| **Diehl & Cook (2015)** — "Unsupervised learning of digit recognition using STDP" (Frontiers Comp. Neuro.) | Global WTA baseline. Their inhibition was all-to-all — Morphon's local inhibition is a publishable distinction. |
| **Brunel (2000)** — "Dynamics of Sparsely Connected Networks of E-I Spiking Neurons" (J. Comp. Neuro.) | E:I ratio ~4:1, inhibitory weight 4-5x excitatory for stable sparse activity. Our starting parameters. |
| **Hazan et al. (2018)** — "Unsupervised Learning with Self-Organizing SNNs" (IJCNN) | Distance-dependent lateral inhibition. Comparable to global WTA on MNIST. Maps to Poincare ball distances. |
| **Zenke et al. (2015)** — "Diverse synaptic plasticity mechanisms orchestrated to form and retrieve memories in SNNs" (Nature Comms.) | Combined excitatory STDP + iSTDP + homeostatic scaling. All three needed for stable learning. |
| **Oster et al. (2009)** — "Computation with Spikes in a Winner-Take-All Network" | WTA emerges reliably when inhibitory synapses are 3-5x excitatory strength. |

### Astrocytic Gating and Neuromodulation

| Paper | Key finding for Morphon |
|---|---|
| **Dong & He (2025)** — "AGMP: Astrocyte-Guided Modulated Plasticity for SNNs" (Frontiers in Neuroscience, Dec 2025) | Four-factor learning rule: eligibility × modulation × astrocytic_gate × stabilization. Per-neuron slow state variable gates plasticity. Directly inspires Morphon's astrocytic gate (Section 4.1a). Morphon extends: dual-timescale EMAs vs. single slow timescale, systemic Endo + per-morphon gate vs. per-neuron only. |
| **NACA (2023)** — Nonlinear neuromodulation with expectation-based signals (Science Advances) | Neuromodulator-plasticity relationship is nonlinear with polarity reversal at high modulation levels. Our linear gain model is a simplification. Noted for Phase C investigation (Section 6). |
| **Patterns/Cell Press (Nov 2025)** — Three-factor learning rules overview incorporating neuromodulatory signals | Comprehensive review covering dopamine (reward), acetylcholine (attention/learning rate), noradrenaline (arousal), serotonin (homeostasis). Notes that interplay between multiple neuromodulators and homeostatic mechanisms is underexplored — which is exactly what Endoquilibrium provides. Our four-channel neuromodulation directly implements this frontier. |

### Homeostatic Validation

| Paper | Key finding for Morphon |
|---|---|
| **eLife (July 2025)** — Interplay between homeostatic synaptic scaling and homeostatic structural plasticity | Both synaptic scaling and structural plasticity are needed for robust firing rate maintenance; they interact nonlinearly. Validates Morphon's dual-mechanism approach: synaptic scaling (`homeostasis.rs` A) + structural plasticity (morphogenesis.rs) operating together. Empirical support for our architecture. |

---

*Endoquilibrium V2 — the system regulates itself because the cells regulate each other.*

*TasteHub GmbH, Wien, April 2026*
