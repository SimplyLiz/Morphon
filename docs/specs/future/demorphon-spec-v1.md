# DeMorphon — The Multicellular Transition
## From Single Cells to Composite Organisms in Morphogenic Intelligence
### Technical & Theoretical Specification v1.0 — TasteHub GmbH, April 2026

---

| | |
|---|---|
| **Authors** | Lisa Welsch, TasteHub GmbH |
| **Status** | Theoretical Specification / Paper Draft Foundation |
| **Depends on** | Endoquilibrium V2 (local inhibitory competition + iSTDP) |
| **Related work** | MORPHON V1–V3 concept papers, Endoquilibrium V1 spec, LocalParams spec |
| **Target venues** | ALIFE 2027, GECCO 2027, or Artificial Life journal |

---

## Abstract

We present DeMorphon, a computational framework for modeling the transition from unicellular to multicellular organization in morphogenic intelligence systems. In the MORPHON architecture, individual compute units (Morphons) are analogous to single-celled organisms — each senses, processes, learns, reproduces, and dies independently. DeMorphons extend this by modeling the evolutionary transition to multicellularity: groups of Morphons that sacrifice individual autonomy to form composite organisms with emergent computational capabilities beyond any constituent cell. This transition is driven by the same mechanism that drove biological multicellularity — the fitness advantage of compartmentalizing incompatible processes into specialized cells within a cooperative aggregate. We formalize the conditions under which DeMorphon formation is energetically favorable, define the internal body plan that emerges from cell specialization, and demonstrate that DeMorphons can perform computations (temporal pattern detection, multi-input coincidence, persistent state) that individual Morphons cannot. This work bridges evolutionary developmental biology, theoretical neuroscience, and neuromorphic computing by implementing the unicellular-to-multicellular major evolutionary transition as a runtime mechanism in an adaptive computational substrate.

---

## 1. Introduction: The Major Transition Problem

### 1.1 Biological Context

The evolution of multicellularity is one of the major evolutionary transitions — events that fundamentally reorganized how biological information is stored, transmitted, and processed (Maynard Smith & Szathmáry, 1995). Multicellularity evolved independently at least 25 times across the tree of life, suggesting it is not a historical accident but a convergent solution to a universal problem: how to build computational systems more complex than a single cell can be.

The key insight from evolutionary theory is that multicellularity doesn't require a top-down design. It emerges from the bottom up when three conditions are met (Rueffler et al., 2012; Ispolatov et al., 2012):

1. **Adhesion:** Cells can stick together, forming aggregates.
2. **Incompatible processes:** Individual cells face tradeoffs — optimizing for one function (e.g., motility) degrades another (e.g., reproduction). A single cell cannot be simultaneously optimized for both.
3. **Division of labor:** In a group, different cells can specialize for different functions, achieving a collective fitness higher than the sum of individual fitnesses.

Critically, this specialization can occur spontaneously among genetically identical cells through regulatory mechanisms already present in the unicellular ancestor (Ispolatov et al., 2012). No genetic predisposition for a particular role is required. The symmetry breaks at the regulatory level — one cell upregulates process A and downregulates process B, its neighbor does the opposite, and the aggregate achieves what neither could alone.

### 1.2 The Computational Analog

In MORPHON, individual Morphons face an analogous tradeoff. A single Morphon must simultaneously:

- **Sense** (respond to input features) — requires low threshold, broad receptive field
- **Integrate** (combine multiple signals over time) — requires high threshold, temporal summation, recurrent connections
- **Remember** (maintain persistent activity) — requires strong recurrent excitation, which conflicts with sensitivity to new input
- **Discriminate** (respond selectively to specific patterns) — requires sharp tuning, which conflicts with broad sensing
- **Explore** (try new connection patterns) — requires high plasticity, which conflicts with stability of learned representations

No single Morphon can optimize all five simultaneously. This is the same "incompatible processes" condition that drove biological multicellularity. A DeMorphon — a composite organism of specialized Morphons — can assign each function to different internal cells, achieving collective computation that no individual Morphon can.

### 1.3 What This Paper Contributes

1. A formal model of the unicellular-to-multicellular transition as a runtime mechanism in a neuromorphic system, including the fitness conditions under which DeMorphon formation is energetically favorable.
2. A specification of DeMorphon internal body plans — how specialized internal cells (Input, Core, Memory, Output) wire together to produce emergent computational capabilities.
3. Three emergent computational capabilities that DeMorphons provide and individual Morphons cannot: temporal pattern detection, multi-input coincidence gating, and persistent state (working memory).
4. A demonstration that DeMorphons can form, compete, reproduce, and dissolve through the same morphogenetic mechanisms (division, differentiation, fusion, apoptosis) that govern individual Morphons — no new primitives required.
5. A connection to the broader MORPHON architecture (Endoquilibrium regulation, local inhibitory competition, metabolic selection) showing how DeMorphons integrate with existing systems.

---

## 2. Background: What MORPHON Already Has

### 2.1 The Morphon as a Unicellular Organism

A Morphon is a complete computational unit with the following capabilities:

| Capability | Implementation |
|---|---|
| Sensation | Feed-input from external sensors; sensory CellType |
| Processing | Leaky integrate-and-fire dynamics; voltage, threshold, refractory |
| Learning | Three-factor STDP (eligibility × modulation); DFA feedback alignment |
| Reproduction | Asymmetric division with inheritance of weights and LocalParams |
| Death | Energy-based apoptosis; metabolic budget with utility reward |
| Migration | Movement in Poincaré ball based on prediction error gradients |
| Differentiation | CellType transitions (Stem → Sensory/Associative/Motor/Modulatory) |
| Communication | Spike-based signaling through weighted synapses |
| Homeostasis | Adaptive threshold; receptor-gated neuromodulation |

Each Morphon is a self-contained agent. It can survive and function independently. This is the unicellular state.

### 2.2 Existing Fusion Mechanism

MORPHON V1 already includes a Fusion primitive: two Morphons can merge into one with combined properties. Fusion is triggered by high correlation between adjacent Morphons (they fire together consistently) and results in a single Morphon with averaged weights and combined connectivity.

Fusion is the MORPHON analog of cell adhesion — the first step toward multicellularity. But current Fusion produces a single, larger Morphon, not a composite organism with internal structure. It's a merge, not a multicellular transition.

### 2.3 Existing Cluster Mechanism

Clusters are groups of Morphons identified by spatial proximity and correlated activity in the Poincaré ball. Clusters have:

- A `Cluster` struct with member list, centroid, and inhibitory morphons
- Inter-cluster inhibitory connections (preventing synchrony between clusters)
- Cluster-level epistemic states (Supported, Contested, etc.)

Clusters are the natural precursor to DeMorphons — they identify groups of Morphons that already function as a unit. The transition from Cluster to DeMorphon is: internal cells specialize, external competition identity shifts from individual to collective, and the body plan becomes fixed.

---

## 3. The DeMorphon: Definition and Formal Model

### 3.1 Definition

A **DeMorphon** is a composite computational organism consisting of N Morphons (N ≥ 3) that have:

1. **Committed** to collective identity — they no longer compete individually in local inhibitory competition; the DeMorphon competes as a unit.
2. **Specialized** into distinct internal roles — Input cells, Core cells, Memory cells, Output cells — each sacrificing capabilities irrelevant to their role.
3. **Developed a fixed internal body plan** — internal wiring is stable (low plasticity), while external connections remain plastic.
4. **Merged their energy budgets** — metabolic success/failure is collective, not individual.
5. **Gained emergent computational capabilities** — the collective can perform computations (temporal patterns, coincidence detection, persistent state) that no constituent Morphon could perform alone.

### 3.2 The Fitness Advantage of DeMorphon Formation

Following Ispolatov et al. (2012), we model the fitness advantage of multicellularity through incompatible process tradeoffs. Define two essential processes for a compute unit:

- **Process A: Sensitivity** — the ability to respond to weak input signals (requires low threshold, broad tuning)
- **Process B: Selectivity** — the ability to discriminate between similar input patterns (requires high threshold, sharp tuning)

A single Morphon optimizes a tradeoff parameter x ∈ [0, 1], where x = 0 is maximally sensitive and x = 1 is maximally selective. Its fitness for process A is fA(x) and for process B is fB(x), where fA is decreasing in x and fB is increasing in x.

The unicellular fitness is:

```
F_unicellular(x) = fA(x) · fB(x)
```

The optimal unicellular x* is a compromise — neither maximally sensitive nor maximally selective. The fitness at x* is a saddle point in the two-process space.

For a two-cell aggregate where cell 1 specializes in A (x₁ → 0) and cell 2 specializes in B (x₂ → 1):

```
F_multicellular(x₁, x₂) = fA(x₁) · fB(x₂)
```

If the fitness landscape has the property that fA(0) · fB(1) > fA(x*) · fB(x*), then the multicellular form achieves higher fitness than the unicellular form. This is the condition for DeMorphon formation to be energetically favorable.

**In MORPHON terms:** A DeMorphon with a dedicated Input cell (low threshold, broad receptive field) and a dedicated Discriminator cell (high threshold, narrow tuning) outperforms a single Morphon trying to do both simultaneously. The Input cell catches weak signals; the Discriminator cell filters them. Neither can do both well alone.

### 3.3 The Three Conditions for DeMorphon Formation

A DeMorphon forms when all three conditions are met simultaneously:

**Condition 1: Adhesion (Cluster Coherence)**

The candidate group of Morphons must already function as a coherent unit:

```rust
fn adhesion_condition(cluster: &Cluster, history_window: u64) -> bool {
    // Mutual information between member activities over recent history
    let mutual_info = cluster.pairwise_mutual_information(history_window);
    // Activity correlation: do members fire in coordinated patterns?
    let mean_correlation = cluster.mean_activity_correlation();
    // Stability: has the cluster membership been stable?
    let membership_stable = cluster.membership_unchanged_for > MIN_STABILITY_TICKS;
    
    mutual_info > ADHESION_MI_THRESHOLD
        && mean_correlation > ADHESION_CORRELATION_THRESHOLD
        && membership_stable
}
```

**Condition 2: Incompatible Processes (Functional Tradeoff)**

The individual members must be facing a tradeoff that specialization would resolve:

```rust
fn tradeoff_condition(cluster: &Cluster, vitals: &VitalSigns) -> bool {
    // Check if members have divergent optimal parameters
    let param_variance = cluster.local_params_variance();
    // High variance in local_params means members are "pulling in different directions"
    // — some want high threshold, some want low
    let has_tradeoff = param_variance.threshold > TRADEOFF_VARIANCE_MIN
        || param_variance.a_plus > TRADEOFF_VARIANCE_MIN;
    
    // Check if cluster performance is limited by the compromise
    let cluster_pe = cluster.mean_prediction_error();
    let pe_would_improve = cluster_pe > vitals.prediction_error_mean * 1.2;
    // Cluster PE is 20% above system average — it's struggling with the compromise
    
    has_tradeoff && pe_would_improve
}
```

**Condition 3: Division of Labor (Energy Advantage)**

Specialization must produce a measurable fitness improvement:

```rust
fn division_of_labor_condition(cluster: &Cluster) -> bool {
    // Estimate: if top-performing member specialized fully,
    // would the cluster earn more energy?
    let current_energy_rate = cluster.total_energy_rate();
    let best_member = cluster.highest_fitness_member();
    let estimated_specialized_rate = best_member.param_fitness * cluster.members.len() as f64 * 0.8;
    // 80% of best member's rate × group size > current collective rate
    
    estimated_specialized_rate > current_energy_rate * 1.1  // 10% improvement threshold
}
```

When all three conditions are true simultaneously on the glacial timescale, the Developmental Engine initiates DeMorphon formation.

---

## 4. Internal Body Plan: Cell Roles and Wiring

### 4.1 The Four Internal Cell Roles

When a DeMorphon forms, constituent Morphons differentiate into one of four internal roles. This specialization is not random — it's driven by the existing properties of each Morphon (what it was already good at as a unicellular unit):

#### Input Cells (Dendrites)

**Role:** Receive external synapses, convert to internal signals.
**Properties:** Low threshold (sensitive), broad receptive field, high fan-in from external morphons, low fan-out internally (feeds into Core cells).
**Sacrifice:** Gives up selectivity, recurrent connections, and output capability. Cannot fire externally.
**Selection:** Morphons with the lowest threshold and highest incoming synapse count become Input cells.

#### Core Cells (Soma)

**Role:** Integrate signals from Input cells, apply nonlinear processing, generate the DeMorphon's computational response.
**Properties:** Medium-high threshold (selective), recurrent connections to other Core cells, receives from Input cells and Memory cells, projects to Output cells.
**Sacrifice:** Gives up direct external connections (only receives from Input cells), gives up individual reproduction (the DeMorphon reproduces as a unit).
**Selection:** Morphons with the highest firing selectivity (high lifetime sparsity) become Core cells.

#### Memory Cells (Engram)

**Role:** Maintain persistent activity through mutual excitation, providing working memory within the DeMorphon.
**Properties:** Strong reciprocal excitatory connections (bistable pair), moderate threshold, slow time constant (high tau_membrane), resistant to inhibition.
**Sacrifice:** Gives up sensitivity to external input (only receives from Core cells), gives up plasticity (consolidation_level → 1.0 on internal connections).
**Selection:** Morphons with the highest recurrent connection strength and lowest firing rate variance become Memory cells. They were already functioning as "stable state holders."

#### Output Cells (Axon Terminal)

**Role:** Produce the DeMorphon's external output signal. Interface with the rest of the network.
**Properties:** High threshold (only fires when Core computation is complete), outgoing connections to external morphons, participates in local inhibitory competition on behalf of the entire DeMorphon.
**Sacrifice:** Gives up sensing capability, gives up internal processing (passively driven by Core cells).
**Selection:** Morphons with the highest outgoing synapse count and strongest connection to motor/output pathways become Output cells.

### 4.2 Internal Wiring Pattern (Body Plan)

```
External input
    │
    ▼
┌──────────────────────────────────────────┐
│ DeMorphon                                │
│                                          │
│  Input₁ ──┐                             │
│  Input₂ ──┼──→ Core₁ ←──→ Core₂ ──→ Output │──→ External targets
│  Input₃ ──┘     ↑   ↕          │        │
│                  │   ↕          │        │
│                  └── Memory ←───┘        │
│                   (bistable              │
│                    pair)                 │
└──────────────────────────────────────────┘
```

**Internal wiring rules:**

- Input → Core: excitatory, plastic (learns which inputs matter)
- Core ↔ Core: excitatory recurrent, plastic (learns temporal integration)
- Core → Memory: excitatory, one-directional (writes state)
- Memory → Core: excitatory, one-directional (reads state)
- Memory ↔ Memory: strong excitatory reciprocal, consolidated (maintains state)
- Core → Output: excitatory, moderate plasticity (learns output mapping)
- All internal inhibition: handled by a single local inhibitory interneuron within the DeMorphon (recycled from the cluster's existing inhibitory morphon)

**External wiring rules:**

- Input cells receive external synapses — these remain fully plastic
- Output cells project external synapses — these remain fully plastic
- Core and Memory cells have NO external connections — they are "interior" cells
- The DeMorphon's position in the Poincaré ball is its centroid — external distance calculations use this position

### 4.3 Specialization as Parameter Clamping

Differentiation into a role means clamping the Morphon's LocalParams to extreme values appropriate for that role:

```rust
fn specialize_for_role(morphon: &mut Morphon, role: DeMorphonRole) {
    match role {
        DeMorphonRole::Input => {
            morphon.threshold = morphon.base_threshold * 0.5;  // very sensitive
            morphon.local_params.a_plus *= 1.5;  // learn fast from external input
            morphon.local_params.feedback_sensitivity = 0.0;  // no DFA (not processing)
            morphon.can_divide = false;  // no individual reproduction
            morphon.can_migrate = false;  // position fixed within DeMorphon
        },
        DeMorphonRole::Core => {
            morphon.threshold = morphon.base_threshold * 1.2;  // selective
            morphon.local_params.tau_eligibility *= 2.0;  // long integration window
            morphon.local_params.feedback_sensitivity = 1.5;  // high DFA sensitivity
            morphon.can_divide = false;
            morphon.can_migrate = false;
        },
        DeMorphonRole::Memory => {
            morphon.threshold = morphon.base_threshold * 0.8;  // moderate
            morphon.local_params.a_plus = 0.0;  // no learning on internal synapses
            morphon.local_params.a_minus = 0.0;  // fully consolidated
            morphon.local_params.tau_trace *= 5.0;  // very slow dynamics
            morphon.can_divide = false;
            morphon.can_migrate = false;
        },
        DeMorphonRole::Output => {
            morphon.threshold = morphon.base_threshold * 1.5;  // high — only fires on strong Core signal
            morphon.local_params.a_plus *= 0.5;  // slow external learning
            morphon.can_divide = false;
            morphon.can_migrate = false;
        },
    }
}
```

**Key insight from biology:** Specialization is irreversible under normal conditions but reversible under stress. If the DeMorphon dissolves (fission), the Morphons regain their original LocalParams from a stored snapshot. This mirrors biological dedifferentiation — mature cells can revert to stem-cell-like states under injury or stress (as observed in planaria and some cnidarians).

---

## 5. Emergent Computational Capabilities

### 5.1 Temporal Pattern Detection

**What it is:** Detecting that input A occurred followed by input B within a specific time window.

**Why a single Morphon can't do it:** A single LIF neuron integrates input over one τ_membrane (~20ms). It can detect coincidence (A and B at the same time) but not sequence (A then B with a 50ms gap). The voltage from input A has decayed below threshold by the time input B arrives.

**How a DeMorphon does it:**

```
Input₁ receives A → fires → activates Core₁ (with delay d₁)
Input₂ receives B → fires → activates Core₂ (with delay d₂)

If d₁ is tuned so that Core₁'s residual voltage from A
coincides with Core₂'s activation from B:
  → Core₁ + Core₂ → Output fires (sequence detected)

If B arrives first: Core₂ fires, but Core₁ has no residual
  → insufficient input → Output does NOT fire
```

The internal delay chain (Input₁ → Core₁ with specific delay, Input₂ → Core₂ with different delay) implements a matched filter for the temporal pattern "A then B." The delays are learned through the internal Input→Core plasticity.

**Biological analog:** Cortical minicolumns detect temporal sequences across their vertical layer structure. Each layer adds a processing delay, and the column fires only when the sequence matches its tuned pattern.

### 5.2 Multi-Input Coincidence Gating (XOR/AND/OR)

**What it is:** Detecting specific combinations of inputs, including XOR (A or B but not both).

**Why a single Morphon can't do it:** A single LIF neuron is a linear threshold unit — it computes a weighted sum of inputs and fires if the sum exceeds threshold. Linear threshold units cannot compute XOR (this is the classic Minsky & Papert result). They can compute AND and OR but not their combination.

**How a DeMorphon does it:**

```
For XOR(A, B):

Input₁ receives A → Core₁ (excitatory) + Core₂ (inhibitory via interneuron)
Input₂ receives B → Core₂ (excitatory) + Core₁ (inhibitory via interneuron)

Core₁ fires only if A is active AND B is inactive
Core₂ fires only if B is active AND A is inactive

Either Core₁ OR Core₂ firing → Output fires

Result: Output fires for A-only or B-only, not for both or neither = XOR
```

The internal inhibitory interneuron (already present from the cluster's iSTDP-based competition) provides the inhibitory cross-connection that enables XOR. The DeMorphon learns the XOR pattern by adjusting the Input→Core and Interneuron→Core weights.

**Biological analog:** Dendritic computation in pyramidal neurons. Different dendritic branches perform local computations (nonlinear thresholding), and the soma integrates the branch outputs. A single neuron with active dendrites can compute XOR — the DeMorphon makes this explicit with separate Input cells acting as dendritic branches.

### 5.3 Persistent State (Working Memory)

**What it is:** Maintaining an active representation of a stimulus after the stimulus is removed.

**Why a single Morphon can't do it:** When input stops, voltage decays to resting potential within a few τ_membrane. A single Morphon cannot maintain activity without continuous input (unless it has a very long time constant, which makes it insensitive to new input — the incompatible processes tradeoff).

**How a DeMorphon does it:**

```
Core₁ activates Memory₁ → Memory₁ excites Memory₂ → Memory₂ excites Memory₁
→ bistable loop sustains activity indefinitely

To "write": Core₁ fires strongly → pushes Memory pair above threshold → loop activates
To "read": Memory pair's activity feeds back to Core₂ → Core₂ fires → Output reflects stored state
To "clear": Inhibitory interneuron fires strongly → suppresses Memory pair below threshold → loop deactivates
```

The Memory cell pair forms a bistable attractor — once activated, mutual excitation maintains the state. The consolidated weights on Memory↔Memory connections (consolidation_level = 1.0) prevent learning from disrupting the bistable dynamics.

**Biological analog:** Persistent activity in prefrontal cortex during working memory tasks. Recurrently connected excitatory neurons sustain firing for seconds after the stimulus is removed. The famous "delay period activity" in monkey PFC (Funahashi et al., 1989) is exactly this mechanism.

---

## 6. DeMorphon Lifecycle

### 6.1 Formation (Multicellular Transition)

```rust
impl DevelopmentalEngine {
    fn attempt_demorphon_formation(&mut self, cluster: &Cluster) -> Option<DeMorphonId> {
        // Check all three conditions
        if !adhesion_condition(cluster, HISTORY_WINDOW) { return None; }
        if !tradeoff_condition(cluster, &self.vitals) { return None; }
        if !division_of_labor_condition(cluster) { return None; }
        
        // Form the DeMorphon
        let demorphon = DeMorphon {
            id: DeMorphonId::new(),
            members: cluster.members.clone(),
            body_plan: assign_roles(cluster),
            energy_pool: cluster.total_energy(),
            position: cluster.centroid(),
            formation_tick: self.current_tick,
            parent_cluster: cluster.id,
            generation: 0,
            dissolution_snapshot: store_params_snapshot(cluster),
        };
        
        // Wire internal body plan
        self.wire_internal_body_plan(&demorphon);
        
        // Remove members from individual competition
        for &member in &demorphon.members {
            self.morphons[member].demorphon_id = Some(demorphon.id);
            self.morphons[member].participates_in_competition = false;
        }
        
        // Register the DeMorphon as a collective competitor
        self.register_demorphon_for_competition(demorphon.id);
        
        Some(demorphon.id)
    }
}
```

### 6.2 Competition (As a Unit)

In local inhibitory competition, DeMorphons compete through their Output cells. When a DeMorphon's Output cell fires, it inhibits neighboring morphons and other DeMorphons' Output cells through the same iSTDP-based mechanism that governs individual competition. From the competition layer's perspective, a DeMorphon is just another competitor — the fact that it's internally composed of multiple cells is invisible.

The DeMorphon has a competitive advantage: its Output cell fires only after internal processing (Input → Core → Output), which means it fires with higher selectivity — only for inputs that match its internal pattern detector. This precision wins competitions against individual Morphons that fire for any sufficiently strong input.

### 6.3 Reproduction (DeMorphon Division)

A DeMorphon can reproduce as a unit when it has sufficient energy and the system needs more capacity:

```rust
fn demorphon_division(&mut self, parent: &DeMorphon) -> Option<DeMorphonId> {
    if parent.energy_pool < DIVISION_ENERGY_THRESHOLD { return None; }
    
    // Create child DeMorphon with mutated body plan
    let child_members = Vec::new();
    for &parent_member in &parent.members {
        let child_morphon = self.morphons[parent_member].divide_with_mutation();
        child_members.push(child_morphon.id);
    }
    
    let child = DeMorphon {
        id: DeMorphonId::new(),
        members: child_members,
        body_plan: parent.body_plan.mutate_slightly(),  // body plan inherits with small mutations
        energy_pool: parent.energy_pool * 0.4,
        position: parent.position.offset_random(0.2),  // child appears nearby in Poincaré ball
        generation: parent.generation + 1,
        ..Default::default()
    };
    
    parent.energy_pool *= 0.6;  // parent keeps 60%
    self.wire_internal_body_plan(&child);
    Some(child.id)
}
```

**Key:** The body plan is inherited with small mutations. A DeMorphon that detects "digit 3" might produce offspring that detect "digit 3 variants" — similar but not identical pattern detectors. This is how DeMorphon populations can cover a feature space through evolutionary diversification.

### 6.4 Fission (Dissolution Back to Unicellular)

A DeMorphon dissolves when it's no longer earning sufficient energy or when the environment changes so dramatically that its specialized body plan is maladapted:

```rust
fn demorphon_fission(&mut self, demorphon: &DeMorphon) {
    // Restore individual morphon capabilities
    for (member_idx, &member_id) in demorphon.members.iter().enumerate() {
        let morphon = &mut self.morphons[member_id];
        
        // Restore from dissolution snapshot
        morphon.local_params = demorphon.dissolution_snapshot[member_idx].clone();
        morphon.demorphon_id = None;
        morphon.participates_in_competition = true;
        morphon.can_divide = true;
        morphon.can_migrate = true;
    }
    
    // Remove internal fixed wiring
    self.remove_demorphon_body_plan_synapses(demorphon.id);
    
    // De-register from collective competition
    self.deregister_demorphon(demorphon.id);
}
```

**Biological analog:** Dictyostelium discoideum (slime mold). Exists as individual amoebae when food is plentiful. When starved, cells aggregate into a multicellular "slug" that migrates collectively. When conditions improve, the slug disaggregates back into individual cells. The transition is fully reversible.

**The trigger for fission:**

```rust
fn should_dissolve(demorphon: &DeMorphon) -> bool {
    // Energy below survival threshold for sustained period
    let energy_crisis = demorphon.energy_pool < FISSION_ENERGY_THRESHOLD
        && demorphon.ticks_below_threshold > FISSION_PATIENCE;
    
    // Internal coherence lost (members' activity is uncorrelated)
    let coherence_lost = demorphon.internal_correlation() < FISSION_CORRELATION_MIN;
    
    // System is in Stressed state and needs individual diversity
    let stress_dissolution = demorphon.endo_stage == DevelopmentalStage::Stressed
        && demorphon.age > MIN_DEMORPHON_AGE;
    
    energy_crisis || coherence_lost || stress_dissolution
}
```

### 6.5 Death (DeMorphon Apoptosis)

A DeMorphon dies as a unit when its energy pool hits zero. All constituent Morphons die simultaneously — they cannot survive individually because they have surrendered their individual metabolic independence. This is the cost of multicellularity: if the organism dies, all cells die.

```rust
fn demorphon_apoptosis(&mut self, demorphon: &DeMorphon) {
    for &member_id in &demorphon.members {
        self.kill_morphon(member_id);  // existing apoptosis mechanism
    }
    self.deregister_demorphon(demorphon.id);
    // Energy NOT returned to pool — the organism failed, its energy is lost
    // (Unlike fission, where members return to the pool with their energy)
}
```

---

## 7. The DeMorphon Struct

```rust
/// A composite organism of specialized Morphons.
/// Competes as a single unit. Has emergent computational capabilities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeMorphon {
    pub id: DeMorphonId,
    
    // === COMPOSITION ===
    pub members: Vec<MorphonId>,
    pub body_plan: BodyPlan,
    pub roles: HashMap<MorphonId, DeMorphonRole>,
    
    // === IDENTITY ===
    pub position: [f64; 2],          // centroid in Poincaré ball
    pub generation: u16,             // lineage depth (0 = first DeMorphon)
    pub parent: Option<DeMorphonId>,
    pub formation_tick: u64,
    pub age: u64,
    
    // === METABOLIC ===
    pub energy_pool: f64,            // shared energy budget
    pub energy_rate: f64,            // recent energy earnings rate (EMA)
    
    // === COMPETITION ===
    pub output_cells: Vec<MorphonId>,  // cells that participate in external competition
    pub participates_in_competition: bool,
    
    // === REVERSIBILITY ===
    pub dissolution_snapshot: Vec<LocalParams>,  // stored individual params for fission
    
    // === DIAGNOSTICS ===
    pub internal_correlation: f64,    // mean pairwise activity correlation
    pub computation_type: Option<EmergentComputation>,  // what this DeMorphon computes (if identified)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DeMorphonRole {
    Input,      // receives external synapses
    Core,       // internal processing
    Memory,     // persistent state (bistable pair)
    Output,     // external output + competition
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyPlan {
    pub input_count: usize,
    pub core_count: usize,
    pub memory_count: usize,  // 0 or 2 (bistable pair)
    pub output_count: usize,
    pub internal_wiring: Vec<(DeMorphonRole, DeMorphonRole, f64)>,  // (from, to, initial_weight)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EmergentComputation {
    TemporalPattern { delay_ms: f64 },
    CoincidenceGate { gate_type: GateType },
    PersistentState,
    Unknown,
}
```

---

## 8. Interaction with Existing MORPHON Systems

### 8.1 Endoquilibrium

Endoquilibrium regulates the *conditions* for DeMorphon formation, not the formation itself:

| Endo lever | Effect on DeMorphon lifecycle |
|---|---|
| Developmental stage | Proliferating: discourage formation (need individual diversity). Consolidating/Mature: encourage formation (lock in proven functional groups). Stressed: may trigger fission (need individual diversity to explore). |
| `fr_assoc_min` / `fr_assoc_max` | Sets iSTDP targets for DeMorphon's internal inhibitory interneuron |
| `consolidation_gain` | High: DeMorphon internal connections consolidate faster (body plan stabilizes). Low: internal connections remain plastic (body plan can still adapt). |
| `plasticity_mult` | Affects external connections of Input/Output cells. Internal connections of Memory cells are always consolidated (exempt from plasticity_mult). |

### 8.2 Local Inhibitory Competition (iSTDP)

DeMorphons participate in local competition through their Output cells. The Output cell has incoming iSTDP-regulated inhibitory synapses from local inhibitory interneurons (the same ones that regulate individual Morphons). From the competition system's perspective, a DeMorphon is just another competitor with a very selective firing pattern.

**The competitive advantage:** Because the DeMorphon's Output only fires after internal multi-step processing (Input → Core → Memory → Core → Output), it fires later than individual Morphons but with higher precision. In a competition based on first-to-fire (as in biological lateral inhibition), DeMorphons don't win by speed — they win by precision. Their Output fires less often but is more reliably correlated with reward, which means their energy earnings per firing event are higher.

### 8.3 Metabolic System

The DeMorphon's energy pool is the sum of its members' individual budgets at formation. Energy is earned collectively: when the DeMorphon's Output fires and the system receives reward, the energy goes to the DeMorphon pool, not to individual members. Energy is spent collectively: each member's metabolic cost (base_cost, firing_cost) is deducted from the pool.

If the DeMorphon's energy rate exceeds the sum of what its members would earn individually, the multicellular form is favored. If it falls below, fission is favored. This creates a natural selection pressure: only DeMorphons that compute something useful survive.

### 8.4 Developmental Engine

DeMorphon lifecycle events (formation, division, fission, death) are managed by the Developmental Engine on the glacial timescale (~60s). This is slower than individual Morphon lifecycle events (slow path, ~1s) because DeMorphon formation requires sustained evidence of cluster coherence, not just a single-tick measurement.

### 8.5 TruthKeeper

Epistemic states can be assigned to DeMorphons as well as individual Morphons. A DeMorphon whose output contradicts verified beliefs is marked Contested. If the contestation persists, its internal connections are re-opened for reconsolidation (consolidation_level reduced). If reconsolidation fails to resolve the contradiction, the DeMorphon is dissolved (fission).

### 8.6 Pulse Kernel Lite

DeMorphon members are still individual Morphons in the hot arrays. The Pulse Kernel processes them identically — it doesn't know about DeMorphon boundaries. The internal body plan wiring is just regular synapses in petgraph. The only Pulse Kernel interaction: the `participates_in_competition` flag on individual members means the old global k-WTA (if still present as fallback) skips them. With local inhibitory competition (Endo V2), this flag isn't even needed — the competition is handled by real inhibitory synapses.

---

## 9. Scaling: From Individual to Population to Ecology

### 9.1 DeMorphon Populations

Multiple DeMorphons can coexist in the same network, each detecting different features or patterns. In an MNIST classification system:

- DeMorphon_A detects "horizontal strokes" (active for digits 2, 5, 7)
- DeMorphon_B detects "closed loops" (active for digits 0, 6, 8, 9)
- DeMorphon_C detects "vertical strokes" (active for digits 1, 4, 7)
- DeMorphon_D detects "crossing strokes" (active for digits 4, 8)

Each DeMorphon's Output cell projects to the supervised readout. The readout learns to map DeMorphon outputs to digit labels. This is more powerful than mapping individual Morphon activity because DeMorphons produce high-level, reliable features (always fires for "closed loop," never fires for "no loop") rather than noisy, low-level pixel responses.

### 9.2 DeMorphon-of-DeMorphons (Recursive Multicellularity)

In biology, multicellular organisms can form higher-order collectives: colonies, societies, superorganisms. In MORPHON, DeMorphons could theoretically aggregate into higher-order composites — a DeMorphon whose "cells" are themselves DeMorphons. This is the recursive self-organization property of Levin's "multi-scale competency architecture."

**We explicitly defer this.** The first-order DeMorphon (Morphons → DeMorphon) must be validated before considering second-order composition (DeMorphons → SuperDeMorphon). The conditions and body plans for higher-order composition are an open research question.

### 9.3 Evolutionary Dynamics

With DeMorphon reproduction, mutation, and selection, the DeMorphon population undergoes evolution. Body plans that produce useful features reproduce (their DeMorphons divide). Body plans that don't earn energy die. Over many generations, the population evolves toward body plans optimized for the current task — without any external optimizer or architect defining what "useful" means.

This is **neuroevolution within the network at runtime** — not a training loop, not a hyperparameter search, but a genuine evolutionary process happening alongside and entangled with the learning dynamics.

---

## 10. Implementation Plan

### 10.1 Prerequisites

| Prerequisite | Why needed | Status |
|---|---|---|
| Endoquilibrium V1 | Stage-dependent regulation controls formation conditions | Implemented |
| Local inhibitory competition (iSTDP) | DeMorphons compete through Output cells using real inhibition | Endo V2, in progress |
| Cluster formation | Clusters are the precursors to DeMorphons | Implemented but dormant in CartPole |
| Episode-gated capture | Internal consolidated connections must be selective | Implemented |

### 10.2 Implementation Phases

| Phase | What | Effort | Depends on |
|---|---|---|---|
| Phase 0 | `DeMorphon` and `BodyPlan` structs, `DeMorphonRole` enum | 2 hours | Nothing |
| Phase 1 | Formation conditions (adhesion, tradeoff, division-of-labor checks) | 4 hours | Phase 0, cluster formation working |
| Phase 2 | Internal body plan wiring (specialize + wire Input/Core/Memory/Output) | 4 hours | Phase 1 |
| Phase 3 | Competition integration (Output cells compete on behalf of DeMorphon) | 2 hours | Phase 2, local inhibition working |
| Phase 4 | Fission (dissolution back to individual Morphons with snapshot restore) | 2 hours | Phase 2 |
| Phase 5 | DeMorphon reproduction (divide as unit with body plan inheritance) | 3 hours | Phase 2 |
| Phase 6 | Metabolic integration (shared energy pool, collective earning/spending) | 2 hours | Phase 2 |
| Phase 7 | Validation: demonstrate temporal pattern detection on synthetic task | 4 hours | Phase 3 |
| Phase 8 | Validation: demonstrate working memory on delayed-match-to-sample task | 4 hours | Phase 3 |
| Phase 9 | MNIST integration: do DeMorphons improve classification accuracy? | 4 hours | Phase 7 |

**Total: ~31 hours.** Phases 0–3 give a working DeMorphon that can form and compete. Phase 4–6 add lifecycle. Phase 7–9 validate emergent capabilities.

---

## 11. Experimental Validation Plan

### 11.1 Synthetic Task: Temporal Pattern Detection

**Task:** Detect the sequence "A then B" (with 50ms gap) vs. "B then A" or "A and B simultaneous."

**Protocol:**
- Create a network with 50 sensory + 100 associative + 10 motor morphons
- Present patterns: A→B (positive), B→A (negative), A+B simultaneous (negative)
- Measure: Can individual Morphons learn this discrimination? Can DeMorphons?
- **Expected result:** Individual Morphons achieve ~60% (near chance for temporal discrimination). DeMorphons with Input→Core delay chain achieve >90%.

### 11.2 Synthetic Task: XOR

**Task:** Compute XOR(A, B) — output 1 if exactly one of two inputs is active.

**Protocol:**
- 2 input morphons, network develops freely
- 4 patterns: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
- **Expected result:** Individual Morphon readout cannot learn XOR (it's a linear threshold unit). DeMorphon with internal inhibitory cross-connection can.

### 11.3 Synthetic Task: Delayed Match-to-Sample

**Task:** See stimulus A, wait 500ms with no input, then see stimulus B. Report whether A == B.

**Protocol:**
- Present stimulus A for 100ms, blank for 500ms, stimulus B for 100ms
- Network must maintain representation of A during the blank period
- **Expected result:** Individual Morphons lose the representation during the blank (voltage decays). DeMorphons with Memory cells maintain persistent activity and can compare.

### 11.4 MNIST with DeMorphons

**Question:** Do DeMorphons, formed through self-organization, improve MNIST classification over individual Morphons?

**Protocol:**
- Run MNIST with DeMorphon formation enabled (formation conditions active during development, frozen during readout training)
- Compare accuracy with and without DeMorphons (A/B test, 10 seeds)
- **Expected result:** DeMorphons produce higher-level features (edge detectors, loop detectors) that improve readout accuracy vs. raw morphon activity

---

## 12. Relationship to Prior Work

| Work | Relationship to DeMorphon |
|---|---|
| **Ispolatov et al. (2012)** — Division of labor and evolution of multicellularity. Proc. R. Soc. B. | Direct theoretical foundation. Our formation conditions implement their fitness saddle-point model computationally. |
| **Colizzi et al. (2020)** — Evolution of multicellularity by collective integration of spatial information. eLife. | Demonstrates that multicellularity evolves for collective chemotaxis advantage. DeMorphons evolve for collective computation advantage. Same mechanism, different domain. |
| **Staps et al. (2023)** — Evolution of selfish multicellularity. BMC Ecology & Evolution. | Shows that multicellularity can evolve from selfish individual behavior via regulatory mechanisms. DeMorphon formation requires no cooperative pre-disposition — it emerges from energy-based selection. |
| **Levin (2023)** — Bioelectric networks as cognitive glue. Animal Cognition. | Multi-scale competency architecture — intelligence at every scale. DeMorphons are the first computational implementation of Levin's "nested intelligence" concept. |
| **Diehl & Cook (2015)** — Unsupervised learning of digit recognition. Frontiers. | MNIST baseline. DeMorphons extend their architecture from flat neuron populations to hierarchical multicellular organisms. |
| **Najarro et al. (2023)** — Neural developmental programs. ALIFE 2023. | Demonstrates learned body plans in neural networks. DeMorphon body plans are inherited and evolved, not learned through backpropagation. |
| **Vogels et al. (2011)** — Inhibitory plasticity. Science. | iSTDP enables the local competition within which DeMorphons compete as units. |
| **Maynard Smith & Szathmáry (1995)** — The major transitions in evolution. | DeMorphons implement the unicellular→multicellular major transition as a computational mechanism. |

---

## 13. Discussion: What DeMorphons Mean for MORPHON

### 13.1 The Scaling Argument

The conventional approach to scaling neuromorphic systems is "more neurons, more synapses." Go from 300 morphons to 30,000 morphons for harder tasks. This has diminishing returns — beyond a certain point, coordination overhead dominates and individual neurons can't represent complex features.

DeMorphons offer a different scaling path: **organizational complexity, not just size.** Instead of 30,000 individual morphons, you have 300 morphons organized into 30 DeMorphons of 10 cells each. Each DeMorphon is a specialized pattern detector with temporal, combinatorial, or state-maintenance capabilities that no individual morphon has. The system scales in what it can *compute*, not just in how many neurons it has.

### 13.2 The Biological Fidelity Argument

No existing neuromorphic system models the unicellular-to-multicellular transition. MORPHON with DeMorphons implements all five major features of biological multicellularity:

1. **Cell adhesion** → Cluster formation through spatial proximity and correlated activity
2. **Division of labor** → Internal specialization into Input/Core/Memory/Output roles
3. **Cell-cell communication** → Internal body plan wiring with distinct signaling patterns
4. **Collective reproduction** → DeMorphon division with body plan inheritance
5. **Programmed cell death of the whole** → DeMorphon apoptosis when collective fitness drops

This is the deepest implementation of Levin's multi-scale competency architecture in a computational system: individual cells that are themselves intelligent agents, organizing into multicellular organisms that are themselves intelligent agents, potentially organizing into higher-order collectives.

### 13.3 The Paper Contribution

**For ALIFE / GECCO:** "We present the first neuromorphic system that implements the unicellular-to-multicellular major evolutionary transition as a runtime mechanism, demonstrating that composite organisms with emergent computational capabilities form spontaneously from individual compute units under energy-based selection pressure."

**For the arXiv MORPHON paper:** DeMorphons become a future work section with the formal model and the three synthetic task predictions. If the validation results are strong enough, it becomes a second paper.

---

## 14. Open Questions

1. **Minimum DeMorphon size.** Is 3 cells enough for useful computation? The formal model suggests 3 (Input + Core + Output), but temporal detection may need 5+ (multiple Input cells with different delays). The answer is task-dependent.

2. **Body plan discovery.** The current spec assigns roles based on existing Morphon properties (lowest threshold → Input, highest selectivity → Core). Should the body plan be discovered through internal evolution instead? Each DeMorphon could try different role assignments and keep the one that earns the most energy. This adds complexity but increases flexibility.

3. **Internal learning rate.** Should internal connections learn at all after body plan formation? The spec says Input→Core stays plastic, Memory↔Memory is consolidated. But what about Core↔Core? If it's too plastic, the internal computation drifts. If it's too rigid, the DeMorphon can't adapt to slowly changing inputs.

4. **DeMorphon-DeMorphon communication.** Can DeMorphons exchange signals in ways more sophisticated than spike-based synapses? Biological multicellular organisms use gap junctions (direct cytoplasmic connections) between cells of the same organism. A DeMorphon "gap junction" could be a high-bandwidth, low-latency internal communication channel distinct from the sparse spike-based external communication.

5. **Reversibility gradient.** The spec models fission as complete dissolution. But biology has partial dedifferentiation — a liver can regenerate by mature cells reverting partially. Should DeMorphons support partial fission (lose one cell, recruit a replacement, re-specialize)?

6. **The formation timing problem.** DeMorphon formation on the glacial timescale (~60s) means the system needs sustained evidence of cluster coherence for a full minute. For fast-changing tasks (CartPole episodes last <10s), this may be too slow. Should formation be triggered by accumulated evidence across episodes rather than within-episode measurement?

---

## 15. References

Colizzi, F., Vroomans, R.M.A., & Merks, R.M.H. (2020). Evolution of multicellularity by collective integration of spatial information. eLife, 9, e56349.

Diehl, P.U. & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. Frontiers in Computational Neuroscience, 9, 99.

Funahashi, S., Bruce, C.J., & Goldman-Rakic, P.S. (1989). Mnemonic coding of visual space in the monkey's dorsolateral prefrontal cortex. Journal of Neurophysiology, 61(2), 331–349.

Ispolatov, I., Ackermann, M., & Doebeli, M. (2012). Division of labour and the evolution of multicellularity. Proceedings of the Royal Society B, 279(1734), 1768–1776.

Levin, M. (2023). Bioelectric networks as cognitive glue for multi-scale morphogenesis. Animal Cognition, 26, 1865–1891.

Maynard Smith, J. & Szathmáry, E. (1995). The Major Transitions in Evolution. Oxford University Press.

Minsky, M. & Papert, S. (1969). Perceptrons: An Introduction to Computational Geometry. MIT Press.

Najarro, E., Sudhakaran, S., Glanois, C., & Risi, S. (2023). Neural Developmental Programs. ALIFE 2023.

Rueffler, C., Hermisson, J., & Wagner, G.P. (2012). Evolution of functional specialization and division of labor. PNAS, 109(6), E326–E335.

Staps, M., Nair, A., & Kolchinsky, A. (2023). Evolution of selfish multicellularity. BMC Ecology & Evolution, 23, 33.

Vogels, T.P., Sprekeler, H., Zenke, F., Clopath, C., & Gerstner, W. (2011). Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks. Science, 334(6062), 1569–1573.

---

*DeMorphon — when cells discover that together, they can think thoughts that alone, they never could.*

*TasteHub GmbH, Wien, April 2026*
