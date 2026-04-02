# Axonal Properties — Activity-Dependent Signal Transmission
## Post-Endo-V2 Enhancement Plan — TasteHub GmbH, April 2026

---

| | |
|---|---|
| **Depends on** | Endo V2 (iSTDP + local inhibition must be working first) |
| **Lives in** | Existing systems — extensions to `Synapse`, `Morphon`, and `MetabolicConfig` |
| **NOT** | A new module. Axonal properties are distributed across existing structures. |
| **Priority** | After paper + FFG submission. Phase 2 funded work. |

---

## 1. Why Axonal Properties Matter for MORPHON

In the current system, all synapses transmit at the same speed. A freshly formed synapse from a just-divided stem morphon delivers spikes with the same delay as a consolidated, high-importance synapse that's been carrying reward-correlated signal for 1000 episodes.

In biology, this is wrong. Frequently used axonal pathways get **myelinated** — wrapped in insulating sheaths that increase conduction velocity by 10-100x. This is activity-dependent: use it more, it gets faster. The result is that established, important pathways have a temporal advantage over new, exploratory ones.

With local inhibitory competition (Endo V2 prerequisite), timing matters. The first morphon to fire suppresses its neighbors through inhibitory interneurons. If a consolidated pathway delivers its spike 2 ticks faster than a new pathway, the consolidated morphon wins the competition. This creates a natural stability mechanism: proven pathways are literally faster, so they win local competition, so they get more reward, so they stay consolidated. New pathways have to overcome both a weight disadvantage AND a speed disadvantage — which is exactly how biology prevents catastrophic forgetting.

Without local inhibitory competition, conduction velocity doesn't matter — global k-WTA is instantaneous and ignores timing. That's why this feature depends on Endo V2.

---

## 2. What Already Exists

| Concept | Current Implementation | Gap |
|---|---|---|
| Signal transmission | Synapse `weight` × spike → target morphon | No delay variation |
| Propagation delay | Synapse `delay` field (f64, ticks) | Static — set at creation, never changes |
| Conduction speed | Implicit in `internal_steps` (all synapses same speed) | No per-synapse velocity |
| Metabolic cost of connections | `MetabolicConfig.firing_cost` per spike | No distance-dependent maintenance cost |
| Consolidation level | `Synapse.consolidation_level` (0.0–1.0) | Not linked to transmission speed |
| Morphon excitability | `Morphon.threshold` (homeostatic) | No AIS-like independent threshold mechanism |
| Output connectivity | petgraph outgoing edges | No branching complexity or propagation failure |

---

## 3. Three Features, Ordered by Impact

### 3.1 Feature 1: Activity-Dependent Myelination (HIGH impact)

**What:** Consolidated, frequently-used synapses get faster signal delivery (lower effective delay).

**Biological basis:** Oligodendrocytes wrap active axons in myelin sheaths proportional to usage. Conduction velocity scales with myelination level. Activity-dependent myelination is well-documented (Fields, 2015; de Faria et al., 2019).

**Implementation:** Added a `myelination` field (f64, 0.0–1.0) to the Synapse struct. Myelination increases toward 1.0 proportional to usage and consolidation. Effective delay decreases with myelination.

```rust
// New field on Synapse
pub myelination: f64,  // 0.0 = unmyelinated, 1.0 = fully myelinated

// Update rule (slow path):
fn update_myelination(&mut self, dt: f64) {
    let activity = if self.usage_count > 0 && self.eligibility.abs() > 0.01 { 1.0 } else { 0.0 };
    let target = self.consolidation_level * activity;
    let tau_myelin = 5000.0;  // very slow timescale
    self.myelination += (target - self.myelination) * (dt / tau_myelin);
    self.myelination = self.myelination.clamp(0.0, 1.0);
}

// Effective delay (used in spike delivery):
fn effective_delay(&self) -> f64 {
    let speed_factor = 1.0 + self.myelination * 4.0;  // up to 5x faster
    (self.delay / speed_factor).max(0.5)  // minimum 0.5 tick delay
}
```

**Interaction with local inhibition:** With iSTDP-based competition, the morphon whose input arrives first has a firing advantage. Myelinated pathways deliver input 2-5x faster, so established feature detectors win local competition over newly formed ones. This is the "fast lane" effect — it stabilizes learned representations without freezing them (because myelination decays if usage drops).

**Interaction with Endoquilibrium:** Endo doesn't directly regulate myelination. Instead, it regulates the conditions under which myelination occurs:
- **Consolidating/Mature stage:** High consolidation_level → myelination increases on proven pathways
- **Stressed stage:** Low consolidation_gain → fewer synapses consolidate → less new myelination, but existing myelination persists (myelin doesn't degrade in a crisis)
- **Proliferating stage:** Most synapses are new and unconsolidated → minimal myelination → all pathways compete on equal temporal footing

**Cost:** One new f64 per synapse (8 bytes × 100K synapses = 800 KB). One O(S) pass on the slow path. Negligible.

**Where it lives:** `Synapse` struct in `morphon.rs`, myelination update in `system.rs` (slow path), effective delay in spike generation (`resonance.rs`).

### 3.2 Feature 2: Distance-Dependent Metabolic Cost (MEDIUM impact)

**What:** Maintaining a synapse costs energy proportional to the hyperbolic distance between source and target morphons in the Poincare ball.

**Biological basis:** Axonal transport (moving proteins, mitochondria, vesicles along the axon) is one of the largest metabolic costs in the brain. Longer axons cost more. This is why most neural connections are local — long-range projections are expensive and only maintained if they're important.

**Implementation:** Added a distance-dependent maintenance cost to the existing metabolic system. Precomputed per-morphon in the system step.

```rust
// Per-synapse maintenance cost:
let distance_factor = 1.0 + hyperbolic_distance * 0.5;  // 50% more per unit distance
let myelination_cost = synapse.myelination * 0.002;       // myelin maintenance
total_cost = synapse_cost_base * distance_factor + myelination_cost;
```

**Interaction with morphogenesis:** This naturally encourages local connectivity. When the system is under energy pressure (Endo Rule 6), long-distance low-importance synapses get pruned first because they're the most expensive to maintain. Short-distance synapses within clusters are cheap. This reinforces cluster formation without explicit clustering rules.

**Interaction with migration:** When a morphon migrates in the Poincare ball, the cost of its existing connections changes. Moving closer to your frequent communication partners reduces metabolic cost. Moving away increases it. This creates an energy gradient that migration can follow — morphons naturally drift toward the morphons they talk to most.

**Pruning heuristic:** The `should_prune_with_cost()` function raises the effective weight threshold for expensive synapses — long-distance weak synapses are pruned more aggressively than local weak synapses.

**Where it lives:** Precomputed in `system.rs` (step function), distance-aware pruning in `learning.rs`, passed through `morphogenesis.rs`.

### 3.3 Feature 3: Propagation Failure at Branch Points (LOW impact, deferred)

**What:** Spikes don't always reach every target. At axonal branch points, a spike can fail to propagate into one branch while succeeding in another, depending on the branch's diameter and recent activity.

**Biological basis:** Axonal propagation failure is documented at branch points where impedance mismatch causes spike attenuation. Failure probability depends on branch geometry and recent firing history (Bhatt et al., 2007).

**Why deferred:** This adds stochasticity to spike delivery, which is biologically correct but computationally messy. It makes the system non-deterministic, complicating debugging and benchmarking. If the system needs noise injection for exploration, frustration-driven noise (already implemented) is a more controlled mechanism.

---

## 4. What We're NOT Building

| Concept | Why not |
|---|---|
| Explicit axon objects | Axonal properties live on synapses and morphons. A separate Axon struct would duplicate state and create sync issues with the existing petgraph edges. |
| Axon initial segment (AIS) | The existing per-morphon threshold with homeostatic regulation covers excitability tuning. AIS is a refinement that doesn't add functional capability. |
| Saltatory conduction simulation | Simulating actual node-of-Ranvier physics is computationally expensive and functionally equivalent to reducing the delay parameter. The `effective_delay` formula captures the outcome without simulating the mechanism. |
| Axon guidance during development | Synaptogenesis already handles connection formation. Axon guidance cues (chemoattraction/repulsion) are implicit in the Poincare distance-based connectivity probability. |
| Demyelination diseases | Fun to model but zero benchmark impact. Defer to "MORPHON as neuroscience simulation tool" use case. |

---

## 5. Implementation Order

```
Phase 1: Myelination (Feature 3.1) — IMPLEMENTED
  - Added myelination field to Synapse
  - Update rule on slow path (system.rs)
  - effective_delay() in spike generation (resonance.rs)
  
Phase 2: Distance-dependent cost (Feature 3.2) — IMPLEMENTED
  - Distance factor in metabolic budget (system.rs precomputation)
  - Updated pruning heuristic with should_prune_with_cost() (learning.rs)
  - Propagated synapse_cost through morphogenesis pipeline
  
Phase 3 (deferred): Propagation failure (Feature 3.3)
  - Only if benchmarks show the system needs more stochastic exploration
  - Or if we're modeling biological phenomena specifically
```

---

## 6. Expected Impact on Benchmarks

| Feature | CartPole | MNIST | Self-healing |
|---|---|---|---|
| Myelination | Minimal — CartPole is already solved. Slightly faster convergence if established pathways win competition sooner. | Medium — established digit-specific pathways get temporal advantage, potentially improving accuracy. | High — after damage, myelinated pathways survive (myelination persists even if some synapses are lost). Recovery is faster because remaining pathways are fast-tracked. |
| Distance cost | None — CartPole network is tiny (100 morphons), distances are irrelevant. | Low-medium — encourages local cluster formation which may improve feature specialization. | Medium — surviving clusters have low-cost local connections, making them metabolically cheaper to maintain during recovery. |
| Propagation failure | Negative risk — adds noise to a task that needs precision. | Low — may help prevent mode collapse through stochastic diversity. | Unclear — failure in damaged pathways could help or hurt. |

---

## 7. Relationship to Other Planned Features

| System | Interaction with Axonal Properties |
|---|---|
| **Endoquilibrium** | Endo regulates conditions (consolidation, stage) that drive myelination. Doesn't directly control myelination rate. |
| **Local inhibition (iSTDP)** | Myelination gives temporal advantage in local competition. This is the primary functional impact — without local inhibition, myelination has no effect on competition outcomes. |
| **LocalParams** | Per-morphon meta-plasticity could include a `myelination_rate` parameter that evolves through inheritance — some lineages myelinate faster than others. Defer to LocalParams Phase 2. |
| **ANCS-Core** | Myelination level could be a factor in AXION importance scoring (f3_centrality or a new f7_conduction_speed). Well-myelinated pathways are more "important" by definition. |
| **Pulse Kernel Lite** | effective_delay() needs to be computed in the fast path. With PKL, the delay could be precomputed during medium-path sync and stored in a hot array for cache-friendly access. |

---

## 8. For the Paper

**Current paper (v1):** Mention axonal properties as future work in the Discussion section. One paragraph: "Activity-dependent myelination would create temporal advantages for consolidated pathways in local inhibitory competition, providing an additional stability mechanism beyond weight consolidation. Distance-dependent metabolic cost would create natural pressure toward local connectivity, reinforcing cluster-based feature specialization."

**Future paper (v2, post-FFG):** If myelination measurably improves MNIST accuracy or self-healing speed, it becomes a results section with comparison tables (with/without myelination, 10 seeds, Welch's t-test).

---

*Axonal properties — the system that makes proven pathways faster, and expensive connections prove their worth.*

*TasteHub GmbH, Wien, April 2026*
