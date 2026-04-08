# MorphonGenome — The Heritable Blueprint
## A Unified Genetic Representation for Morphogenic Intelligence
### Technical & Theoretical Specification v1.0 — TasteHub GmbH, April 2026

---

| | |
|---|---|
| **Authors** | Lisa Welsch, TasteHub GmbH |
| **Status** | Theoretical Specification / Paper Foundation |
| **Implements before** | DeMorphon (provides the inheritance mechanism DeMorphon needs) |
| **Depends on** | LocalParams (per-morphon meta-plasticity parameters), Endoquilibrium V1 |
| **Replaces** | Ad-hoc field copying in `divide_morphon()`, scattered inheritance logic |

---

## 1. What DNA Does in Biology — and What's Missing in MORPHON

### 1.1 The Four Functions of DNA

DNA serves four distinct functions in biological organisms:

| Function | What it does | Timescale |
|---|---|---|
| **Blueprint** | Encodes how to build the organism — which proteins, in what order, under what conditions | Developmental (hours–days) |
| **Heredity** | Faithful transmission from parent to offspring with small mutations | Generational (division events) |
| **Regulation** | Context-dependent gene expression — same genome, different cell types (epigenetics) | Continuous (responds to environment) |
| **Integrity** | Error detection and repair; triggers apoptosis when damage is irreparable | Continuous (every replication) |

### 1.2 What MORPHON Currently Has (Scattered)

Each DNA function already exists in MORPHON, but scattered across different structs and modules with no unifying abstraction:

| DNA Function | Current MORPHON Implementation | Where it lives | Problem |
|---|---|---|---|
| Blueprint | SystemConfig + DevelopmentalConfig + LearningParams | `config.rs`, `types.rs` | System-level only. No per-morphon blueprint. Cannot rebuild a single morphon from its specification. |
| Heredity | `divide_morphon()` copies fields ad-hoc: plasticity_rate, threshold, cell_type. LocalParams inherits with mutation. | `developmental.rs`, `types.rs` | Incomplete — some fields are inherited, some are reset to defaults, some are averaged. No single "what gets inherited" definition. |
| Regulation | CellType differentiation + receptor gating. Morphons start as Stem, differentiate based on position/activity. | `morphon.rs`, `learning.rs` | Works but isn't framed as gene expression. No concept of "this morphon has the potential for X but is currently expressing Y." |
| Integrity | TruthKeeper (epistemic validation) + checkpoint/rollback (PE-triggered) + apoptosis (energy < threshold) | `homeostasis.rs`, `epistemic.rs` | No genome-level integrity check. A morphon with corrupted LocalParams (from mutation gone wrong) isn't detected — it just performs badly and eventually dies. |

### 1.3 The Core Problem

When a Morphon divides, the inheritance logic is:

```rust
// Current (ad-hoc):
child.plasticity_rate = parent.plasticity_rate;  // inherited
child.threshold = parent.threshold;               // inherited
child.cell_type = CellType::Stem;                 // reset
child.potential = 0.0;                            // reset
child.local_params = parent.local_params.inherit_with_mutation(rng, rate);  // inherited + mutated
child.energy = parent.energy * 0.4;              // split
// child.position = ???  inherited? random? offset?
// child.receptors = ???  inherited? reset to Stem defaults?
// child.feedback_sensitivity = ???  part of LocalParams? separate?
```

Some fields are copied, some are mutated, some are reset. There's no principled definition of "what is heritable" vs. "what is developmental state." This becomes critical for DeMorphons: when a DeMorphon reproduces, what exactly constitutes its genome? The BodyPlan? The LocalParams of each member? The internal wiring weights? The role assignments?

**The MorphonGenome solves this** by creating a single, explicit data structure that contains everything heritable — and nothing that isn't. Developmental state (voltage, refractory timer, current firing status) is explicitly excluded. The genome is what you'd need to build an identical organism from scratch in a new environment.

---

## 2. The MorphonGenome

### 2.1 Design Principle: Genotype vs. Phenotype

Biology distinguishes between genotype (the DNA sequence — what's inherited) and phenotype (the expressed organism — what exists at runtime). The genotype is stable, serializable, and transmissible. The phenotype is dynamic, environment-dependent, and ephemeral.

In MORPHON:

- **Genotype (MorphonGenome):** The heritable specification. Fixed at birth (set during division), mutated only at reproduction. Can be serialized, compared, transmitted between systems.
- **Phenotype (Morphon struct):** The living organism. Changes continuously through learning, homeostasis, neuromodulation, energy dynamics. Cannot be meaningfully transmitted (it's a snapshot of a dynamic process, not a specification).

**The rule:** Everything in the Genome can be used to build a new Morphon via `express()`. Nothing in the Genome changes after the Morphon is born (except through mutation at the next reproduction event). The Genome is read-only during the Morphon's lifetime.

### 2.2 The Genome Struct

```rust
/// The heritable blueprint of a Morphon.
/// Contains everything needed to build a new Morphon from scratch.
/// Immutable during the Morphon's lifetime — only changes through
/// mutation at reproduction.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MorphonGenome {
    // === IDENTITY GENES ===
    /// Unique genome ID (not the same as MorphonId — multiple morphons
    /// can share a genome if cloned without mutation)
    pub genome_id: GenomeId,
    
    /// Lineage tracking — who are my ancestors?
    pub parent_genome: Option<GenomeId>,
    pub generation: u16,
    pub lineage_hash: u64,  // hash of ancestor chain for fast lineage comparison
    
    // === STRUCTURAL GENES (body plan) ===
    /// Base firing threshold before homeostatic adjustment
    pub base_threshold: f64,
    
    /// Preferred cell type — what this morphon differentiates into
    /// when conditions allow. Stem = no preference (totipotent).
    pub cell_type_affinity: CellTypeAffinity,
    
    /// Receptor configuration — which neuromodulation channels affect this morphon
    pub receptor_blueprint: ReceptorBlueprint,
    
    /// Preferred position bias in Poincaré ball (not absolute position —
    /// a tendency toward center/boundary, encoded as a radial preference)
    pub radial_preference: f64,  // 0.0 = center (generalist), 1.0 = boundary (specialist)

    // === LEARNING GENES (meta-plasticity) ===
    /// The LocalParams that govern how this morphon learns.
    /// This IS the morphon's "learning DNA" — inherited with mutation.
    pub learning_params: LocalParams,
    
    /// Base plasticity rate (how fast this morphon's weights change)
    pub base_plasticity_rate: f64,
    
    // === METABOLIC GENES ===
    /// Base energy efficiency — how much energy this morphon extracts from activity
    pub metabolic_efficiency: f64,  // default 1.0, range [0.5, 2.0]
    
    /// Firing cost multiplier — some morphons are cheap to fire, others expensive
    pub firing_cost_factor: f64,  // default 1.0, range [0.5, 2.0]
    
    /// Apoptosis resistance — how low energy can drop before triggering death
    pub apoptosis_threshold: f64,  // default 0.1, range [0.01, 0.3]
    
    // === REPRODUCTIVE GENES ===
    /// Division threshold — how much energy/fitness needed to trigger reproduction
    pub division_threshold: f64,
    
    /// Mutation rate for offspring — how much the genome changes at division
    /// This is the "mutation rate gene" — it can itself mutate, creating
    /// lineages with high or low evolvability.
    pub mutation_rate: f64,  // default 0.05, range [0.001, 0.3]
    
    // === SOCIAL GENES (DeMorphon-relevant) ===
    /// Adhesion propensity — how likely this morphon is to join/form a DeMorphon
    /// High values = gregarious (prefers collective). Low = solitary.
    pub adhesion_propensity: f64,  // default 0.5, range [0.0, 1.0]
    
    /// Role flexibility — can this morphon specialize into any DeMorphon role,
    /// or does it have a strong preference?
    pub role_flexibility: f64,  // 1.0 = totipotent, 0.0 = locked to cell_type_affinity
    
    /// Altruism coefficient — willingness to sacrifice individual fitness
    /// for collective benefit. Relevant for DeMorphon energy sharing.
    pub altruism: f64,  // default 0.5, range [0.0, 1.0]
}

/// What cell type does this genome prefer to differentiate into?
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CellTypeAffinity {
    Totipotent,                    // no preference (Stem)
    Biased(CellType, f64),         // prefers this type with strength 0-1
    Committed(CellType),           // always differentiates to this type
}

/// Which receptors does this morphon express?
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReceptorBlueprint {
    pub reward_sensitivity: f64,      // 0.0 = deaf to reward, 1.0 = maximum
    pub novelty_sensitivity: f64,
    pub arousal_sensitivity: f64,
    pub homeostasis_sensitivity: f64,
}
```

### 2.3 What's Explicitly NOT in the Genome

These are phenotypic (runtime) state — they change continuously during the Morphon's life and are not inherited:

```rust
// NOT heritable — these are phenotype, not genotype:
// voltage (membrane potential) — transient electrical state
// fired / refractory_timer — instantaneous spiking state
// potential / prev_potential — dynamic processing state
// energy (current level) — metabolic state (divided at birth, not inherited)
// position (exact coordinates) — developmental placement
// age — lifetime counter
// feedback_signal — DFA injection (computed externally)
// param_fitness — accumulated fitness score (earned, not inherited)
// param_age — ticks since last param change (runtime tracking)
// consolidation_level on synapses — earned through capture, not inherited
// synapse weights — learned, not inherited (but see Section 4.3)
// cluster_id — emergent group membership
// demorphon_id — collective membership
// epistemic_state — knowledge validity assessment
```

**The principle:** If it would be meaningful for a newborn Morphon in a completely different part of the network, it's genotype. If it only makes sense in the context of this specific Morphon's current location and history, it's phenotype.

---

## 3. Genome Operations

### 3.1 Snapshot: Living Morphon → Genome

Extract the genome from a living Morphon. Used at system initialization to record the "founding genome" and at any point for analysis.

```rust
impl MorphonGenome {
    /// Extract the genome from a living Morphon.
    /// This captures the heritable state, ignoring all phenotypic state.
    pub fn snapshot(morphon: &Morphon) -> Self {
        Self {
            genome_id: GenomeId::new(),
            parent_genome: morphon.genome_id,
            generation: morphon.generation,
            lineage_hash: morphon.lineage_hash,
            
            base_threshold: morphon.base_threshold,
            cell_type_affinity: match morphon.cell_type {
                CellType::Stem => CellTypeAffinity::Totipotent,
                ct => CellTypeAffinity::Biased(ct, 0.7),
            },
            receptor_blueprint: ReceptorBlueprint {
                reward_sensitivity: morphon.reward_sensitivity,
                novelty_sensitivity: morphon.novelty_sensitivity,
                arousal_sensitivity: morphon.arousal_sensitivity,
                homeostasis_sensitivity: morphon.homeostasis_sensitivity,
            },
            radial_preference: poincare_radius(morphon.position),
            
            learning_params: morphon.local_params.clone(),
            base_plasticity_rate: morphon.plasticity_rate,
            
            metabolic_efficiency: morphon.metabolic_efficiency,
            firing_cost_factor: morphon.firing_cost_factor,
            apoptosis_threshold: morphon.apoptosis_threshold,
            
            division_threshold: morphon.division_threshold,
            mutation_rate: morphon.mutation_rate,
            
            adhesion_propensity: morphon.adhesion_propensity,
            role_flexibility: morphon.role_flexibility,
            altruism: morphon.altruism,
        }
    }
}
```

### 3.2 Express: Genome → Living Morphon

Build a new Morphon from a genome. This is "gene expression" — the genome specifies the initial conditions, but the environment (position, neighbors, available energy) determines the final phenotype.

```rust
impl MorphonGenome {
    /// Build a new Morphon from this genome.
    /// Position and initial energy are provided by the environment (Developmental Engine).
    pub fn express(
        &self,
        id: MorphonId,
        position: [f64; 2],
        initial_energy: f64,
    ) -> Morphon {
        Morphon {
            id,
            genome_id: Some(self.genome_id),
            generation: self.generation,
            
            // Structural expression
            base_threshold: self.base_threshold,
            threshold: self.base_threshold,  // starts at base, homeostasis adjusts
            cell_type: match &self.cell_type_affinity {
                CellTypeAffinity::Committed(ct) => *ct,
                _ => CellType::Stem,  // starts as Stem, differentiates later
            },
            
            // Receptor expression
            reward_sensitivity: self.receptor_blueprint.reward_sensitivity,
            novelty_sensitivity: self.receptor_blueprint.novelty_sensitivity,
            arousal_sensitivity: self.receptor_blueprint.arousal_sensitivity,
            homeostasis_sensitivity: self.receptor_blueprint.homeostasis_sensitivity,
            
            // Learning expression
            local_params: self.learning_params.clone(),
            plasticity_rate: self.base_plasticity_rate,
            
            // Metabolic expression
            energy: initial_energy,
            metabolic_efficiency: self.metabolic_efficiency,
            firing_cost_factor: self.firing_cost_factor,
            apoptosis_threshold: self.apoptosis_threshold,
            
            // Reproductive expression
            division_threshold: self.division_threshold,
            mutation_rate: self.mutation_rate,
            
            // Social expression
            adhesion_propensity: self.adhesion_propensity,
            role_flexibility: self.role_flexibility,
            altruism: self.altruism,
            
            // Phenotypic initial state (NOT from genome)
            position,
            potential: 0.0,
            prev_potential: 0.0,
            fired: false,
            refractory_timer: 0.0,
            input_accumulator: 0.0,
            age: 0,
            param_fitness: 0.0,
            param_age: 0,
            feedback_signal: 0.0,
            demorphon_id: None,
            cluster_id: None,
            participates_in_competition: true,
            can_divide: true,
            can_migrate: true,
            
            ..Default::default()
        }
    }
}
```

### 3.3 Mutate: Genome → Mutated Genome

Create a child genome from a parent genome with small mutations. This is the core evolutionary mechanism.

```rust
impl MorphonGenome {
    /// Create a mutated copy of this genome.
    /// mutation_strength is scaled by Endo's novelty_gain:
    /// high novelty = more exploration of genetic space.
    pub fn mutate(&self, rng: &mut impl Rng, mutation_strength: f64) -> Self {
        let rate = self.mutation_rate * mutation_strength;
        
        let mutate_f64 = |value: f64, min: f64, max: f64| -> f64 {
            let noise = rng.sample::<f64, _>(StandardNormal) * rate * value.abs().max(0.01);
            (value + noise).clamp(min, max)
        };
        
        let mutate_f64_small = |value: f64, min: f64, max: f64| -> f64 {
            let noise = rng.sample::<f64, _>(StandardNormal) * rate * 0.5 * value.abs().max(0.01);
            (value + noise).clamp(min, max)
        };
        
        Self {
            genome_id: GenomeId::new(),
            parent_genome: Some(self.genome_id),
            generation: self.generation + 1,
            lineage_hash: hash_combine(self.lineage_hash, self.genome_id),
            
            // Structural genes — moderate mutation
            base_threshold: mutate_f64(self.base_threshold, 0.1, 5.0),
            cell_type_affinity: self.cell_type_affinity.clone(),  // usually stable
            receptor_blueprint: ReceptorBlueprint {
                reward_sensitivity: mutate_f64_small(
                    self.receptor_blueprint.reward_sensitivity, 0.0, 2.0),
                novelty_sensitivity: mutate_f64_small(
                    self.receptor_blueprint.novelty_sensitivity, 0.0, 2.0),
                arousal_sensitivity: mutate_f64_small(
                    self.receptor_blueprint.arousal_sensitivity, 0.0, 2.0),
                homeostasis_sensitivity: mutate_f64_small(
                    self.receptor_blueprint.homeostasis_sensitivity, 0.0, 2.0),
            },
            radial_preference: mutate_f64_small(self.radial_preference, 0.0, 1.0),
            
            // Learning genes — these mutate through LocalParams.inherit_with_mutation
            learning_params: self.learning_params.inherit_with_mutation(rng, rate),
            base_plasticity_rate: mutate_f64(self.base_plasticity_rate, 0.01, 3.0),
            
            // Metabolic genes — small mutation (survival-critical)
            metabolic_efficiency: mutate_f64_small(self.metabolic_efficiency, 0.5, 2.0),
            firing_cost_factor: mutate_f64_small(self.firing_cost_factor, 0.5, 2.0),
            apoptosis_threshold: mutate_f64_small(self.apoptosis_threshold, 0.01, 0.3),
            
            // Reproductive genes
            division_threshold: mutate_f64(self.division_threshold, 0.1, 5.0),
            // The mutation rate itself can mutate — enabling evolution of evolvability
            mutation_rate: mutate_f64_small(self.mutation_rate, 0.001, 0.3),
            
            // Social genes
            adhesion_propensity: mutate_f64(self.adhesion_propensity, 0.0, 1.0),
            role_flexibility: mutate_f64_small(self.role_flexibility, 0.0, 1.0),
            altruism: mutate_f64(self.altruism, 0.0, 1.0),
        }
    }
}
```

**Key design: the mutation rate is itself heritable and mutable.** This enables the evolution of evolvability — lineages can evolve to have high or low mutation rates depending on environmental stability. In stable environments, low-mutation lineages dominate (they preserve good genomes). In changing environments, high-mutation lineages explore faster. This is a well-studied phenomenon in evolutionary biology (Taddei et al., 1997) and emerges naturally from the self-referential mutation rate gene.

### 3.4 Crossover: Two Genomes → Child Genome

Sexual recombination of two parent genomes. Relevant when two DeMorphons merge or when two high-fitness Morphons produce an offspring that combines their strengths.

```rust
impl MorphonGenome {
    /// Create a child genome by recombining two parent genomes.
    /// Each gene is taken from one parent (uniform crossover)
    /// plus small mutation on the result.
    pub fn crossover(
        parent_a: &MorphonGenome,
        parent_b: &MorphonGenome,
        rng: &mut impl Rng,
        mutation_strength: f64,
    ) -> Self {
        let pick = |a: f64, b: f64| -> f64 {
            if rng.gen::<bool>() { a } else { b }
        };
        
        let child = Self {
            genome_id: GenomeId::new(),
            parent_genome: Some(parent_a.genome_id),  // primary parent
            generation: parent_a.generation.max(parent_b.generation) + 1,
            lineage_hash: hash_combine(parent_a.lineage_hash, parent_b.lineage_hash),
            
            base_threshold: pick(parent_a.base_threshold, parent_b.base_threshold),
            cell_type_affinity: if rng.gen::<bool>() {
                parent_a.cell_type_affinity.clone()
            } else {
                parent_b.cell_type_affinity.clone()
            },
            receptor_blueprint: ReceptorBlueprint {
                reward_sensitivity: pick(
                    parent_a.receptor_blueprint.reward_sensitivity,
                    parent_b.receptor_blueprint.reward_sensitivity),
                novelty_sensitivity: pick(
                    parent_a.receptor_blueprint.novelty_sensitivity,
                    parent_b.receptor_blueprint.novelty_sensitivity),
                arousal_sensitivity: pick(
                    parent_a.receptor_blueprint.arousal_sensitivity,
                    parent_b.receptor_blueprint.arousal_sensitivity),
                homeostasis_sensitivity: pick(
                    parent_a.receptor_blueprint.homeostasis_sensitivity,
                    parent_b.receptor_blueprint.homeostasis_sensitivity),
            },
            radial_preference: pick(parent_a.radial_preference, parent_b.radial_preference),
            
            learning_params: LocalParams::crossover(
                &parent_a.learning_params, &parent_b.learning_params, rng),
            base_plasticity_rate: pick(
                parent_a.base_plasticity_rate, parent_b.base_plasticity_rate),
            
            metabolic_efficiency: pick(
                parent_a.metabolic_efficiency, parent_b.metabolic_efficiency),
            firing_cost_factor: pick(
                parent_a.firing_cost_factor, parent_b.firing_cost_factor),
            apoptosis_threshold: pick(
                parent_a.apoptosis_threshold, parent_b.apoptosis_threshold),
            
            division_threshold: pick(
                parent_a.division_threshold, parent_b.division_threshold),
            mutation_rate: pick(parent_a.mutation_rate, parent_b.mutation_rate),
            
            adhesion_propensity: pick(
                parent_a.adhesion_propensity, parent_b.adhesion_propensity),
            role_flexibility: pick(
                parent_a.role_flexibility, parent_b.role_flexibility),
            altruism: pick(parent_a.altruism, parent_b.altruism),
        };
        
        // Apply small mutation on top of crossover
        child.mutate(rng, mutation_strength * 0.5)  // half-strength mutation after crossover
    }
}
```

### 3.5 Distance: Genome × Genome → Genetic Distance

How similar are two genomes? Used for: lineage analysis, speciation detection (are two populations diverging?), and mate selection in crossover (prefer genetically dissimilar mates for hybrid vigor).

```rust
impl MorphonGenome {
    /// Compute normalized genetic distance between two genomes.
    /// 0.0 = identical, 1.0 = maximally different.
    pub fn distance(&self, other: &MorphonGenome) -> f64 {
        let mut diffs = Vec::new();
        
        // Structural genes
        diffs.push(normalized_diff(self.base_threshold, other.base_threshold, 0.1, 5.0));
        diffs.push(normalized_diff(self.radial_preference, other.radial_preference, 0.0, 1.0));
        
        // Receptor genes
        diffs.push(normalized_diff(
            self.receptor_blueprint.reward_sensitivity,
            other.receptor_blueprint.reward_sensitivity, 0.0, 2.0));
        diffs.push(normalized_diff(
            self.receptor_blueprint.novelty_sensitivity,
            other.receptor_blueprint.novelty_sensitivity, 0.0, 2.0));
        
        // Learning genes
        diffs.push(normalized_diff(
            self.learning_params.a_plus, other.learning_params.a_plus, 0.01, 2.0));
        diffs.push(normalized_diff(
            self.learning_params.tau_eligibility,
            other.learning_params.tau_eligibility, 5.0, 50.0));
        
        // Social genes
        diffs.push(normalized_diff(
            self.adhesion_propensity, other.adhesion_propensity, 0.0, 1.0));
        diffs.push(normalized_diff(self.altruism, other.altruism, 0.0, 1.0));
        
        // Metabolic genes
        diffs.push(normalized_diff(
            self.metabolic_efficiency, other.metabolic_efficiency, 0.5, 2.0));
        
        diffs.iter().sum::<f64>() / diffs.len() as f64
    }
}

fn normalized_diff(a: f64, b: f64, min: f64, max: f64) -> f64 {
    ((a - b).abs() / (max - min)).min(1.0)
}
```

---

## 4. How the Genome Changes MORPHON

### 4.1 Division Refactored

Current division copies fields ad-hoc. With MorphonGenome, division becomes:

```rust
// BEFORE (ad-hoc):
fn divide_morphon(&mut self, parent_id: MorphonId) -> MorphonId {
    let parent = &self.morphons[parent_id];
    let mut child = Morphon::default();
    child.plasticity_rate = parent.plasticity_rate;
    child.threshold = parent.threshold;
    child.cell_type = CellType::Stem;
    child.local_params = parent.local_params.inherit_with_mutation(&mut rng, rate);
    child.energy = parent.energy * 0.4;
    // ... 15 more lines of field copying
    self.add_morphon(child)
}

// AFTER (genome-based):
fn divide_morphon(&mut self, parent_id: MorphonId) -> MorphonId {
    let parent = &self.morphons[parent_id];
    let parent_genome = MorphonGenome::snapshot(parent);
    
    // Mutation strength gated by Endoquilibrium novelty
    let mutation_strength = self.endo.channels.novelty_gain as f64;
    let child_genome = parent_genome.mutate(&mut self.rng, mutation_strength);
    
    // Express child in a position near parent
    let child_position = offset_position(parent.position, 0.1, &mut self.rng);
    let child_energy = parent.energy * 0.4;
    parent.energy *= 0.6;
    
    let child = child_genome.express(MorphonId::new(), child_position, child_energy);
    self.add_morphon(child)
}
```

**Benefits:** Every heritable property is now in one place (the genome). Adding a new heritable field means adding it to MorphonGenome — the division logic doesn't change. The mutation logic is centralized and testable.

### 4.2 System Bootstrap Refactored

Currently, `System::new()` creates morphons by calling `Morphon::new()` with various config values. With the genome:

```rust
fn bootstrap_system(config: &SystemConfig) -> System {
    let founder_genome = MorphonGenome::from_config(config);
    
    let mut morphons = Vec::new();
    for i in 0..config.developmental.seed_size {
        // Each seed morphon gets a slightly mutated copy of the founder genome
        let genome = founder_genome.mutate(&mut rng, 0.02);  // low initial diversity
        let position = random_poincare_position(&mut rng);
        let morphon = genome.express(MorphonId::new(), position, 1.0);
        morphons.push(morphon);
    }
    
    System { morphons, founder_genome, .. }
}
```

The **founder genome** is stored on the System — it's the "species definition" from which all morphons descend. This is useful for analysis: you can measure genetic drift (how far the current population's genomes have drifted from the founder) and for reset (revert to the founder genome if the population has evolved into a dead end).

### 4.3 Synapse Inheritance: Weights Are NOT Genome

Synapse weights are learned, not inherited. This is biologically correct — you don't inherit your parent's memories. However, the *propensity* for certain connection patterns IS genetic:

- `radial_preference` (center vs. boundary) determines where the morphon positions itself, which determines its neighbors, which determines its likely connections
- `base_threshold` determines how responsive the morphon is, which affects which input patterns drive STDP
- `receptor_blueprint` determines which modulation channels affect learning, which shapes what the morphon learns

So the genome doesn't encode weights directly, but it creates the conditions under which specific weight patterns emerge through development. This is the genotype→phenotype mapping: the genome specifies the developmental program, the environment + learning rules produce the actual synaptic weights.

### 4.4 Epigenetics: Runtime Gene Expression Changes

In biology, epigenetic modifications (methylation, histone changes) alter gene expression without changing the DNA sequence. A stressed cell expresses different genes than a relaxed one, even though the DNA is identical.

In MORPHON, epigenetic regulation already exists — it's what Endoquilibrium does. The `plasticity_mult`, `threshold_bias`, and channel gains modify how the genome's base values are expressed at runtime:

```
effective_threshold = genome.base_threshold + endo.threshold_bias + homeostatic_adjustment
effective_plasticity = genome.base_plasticity_rate * endo.plasticity_mult
effective_sensitivity = genome.receptor_blueprint.reward * endo.reward_gain
```

The genome sets the baseline. Endo provides the epigenetic modification. The phenotype is the combination.

**This means DeMorphon role specialization is epigenetic, not genetic.** When a Morphon becomes an Input cell in a DeMorphon, its genome doesn't change. Instead, the DeMorphon applies an "epigenetic override" that clamps its expressed parameters to role-appropriate values. If the DeMorphon dissolves (fission), the override is removed and the Morphon reverts to expressing its original genome — dedifferentiation.

```rust
/// Epigenetic override for DeMorphon role specialization.
/// Applied on top of genome expression. Removed on fission.
pub struct EpigeneticOverride {
    pub threshold_multiplier: f64,      // 0.5 for Input, 1.2 for Core, etc.
    pub plasticity_clamp: Option<f64>,  // Some(0.0) for Memory (frozen)
    pub suppress_division: bool,         // true for all DeMorphon members
    pub suppress_migration: bool,        // true for all DeMorphon members
    pub feedback_sensitivity_override: Option<f64>,
}
```

---

## 5. The DeMorphon Genome

### 5.1 Composite Genome

A DeMorphon's genome is not just a collection of individual Morphon genomes — it's a higher-order genome that specifies the body plan (how cells are organized) in addition to the cell specifications:

```rust
/// The heritable blueprint of a DeMorphon.
/// Contains the body plan + per-role genome templates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeMorphonGenome {
    pub genome_id: GenomeId,
    pub parent_genome: Option<GenomeId>,
    pub generation: u16,
    
    // === BODY PLAN GENES ===
    /// How many cells of each role
    pub body_plan: BodyPlan,
    
    /// Internal wiring specification (from_role, to_role, weight_range)
    pub wiring_spec: Vec<WiringGene>,
    
    // === PER-ROLE CELL GENOMES ===
    /// Genome template for each role — expressed with epigenetic overrides
    pub input_genome: MorphonGenome,
    pub core_genome: MorphonGenome,
    pub memory_genome: Option<MorphonGenome>,  // None if no memory cells
    pub output_genome: MorphonGenome,
    
    // === COLLECTIVE GENES ===
    /// How strongly the DeMorphon resists fission under stress
    pub cohesion_strength: f64,
    
    /// Energy sharing ratio between roles (e.g., Core gets more than Input)
    pub energy_allocation: [f64; 4],  // Input, Core, Memory, Output fractions
    
    /// Threshold for collective reproduction
    pub collective_division_threshold: f64,
}

/// A gene that specifies one internal wiring connection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WiringGene {
    pub from_role: DeMorphonRole,
    pub to_role: DeMorphonRole,
    pub weight_mean: f64,
    pub weight_std: f64,     // variance in initial weight (expressed with noise)
    pub probability: f64,     // connection probability (not all-to-all)
    pub is_inhibitory: bool,
}
```

### 5.2 DeMorphon Expression

Building a DeMorphon from its genome:

```rust
impl DeMorphonGenome {
    pub fn express(
        &self,
        position: [f64; 2],
        available_energy: f64,
        rng: &mut impl Rng,
    ) -> DeMorphon {
        let mut members = Vec::new();
        
        // Express Input cells
        for i in 0..self.body_plan.input_count {
            let genome = self.input_genome.mutate(rng, 0.01);  // tiny intra-organism variation
            let pos = offset_position(position, 0.05, rng);
            let morphon = genome.express(MorphonId::new(), pos, 
                available_energy * self.energy_allocation[0] / self.body_plan.input_count as f64);
            members.push((morphon, DeMorphonRole::Input));
        }
        
        // Express Core cells
        for i in 0..self.body_plan.core_count {
            let genome = self.core_genome.mutate(rng, 0.01);
            let pos = offset_position(position, 0.03, rng);  // Core cells are central
            let morphon = genome.express(MorphonId::new(), pos,
                available_energy * self.energy_allocation[1] / self.body_plan.core_count as f64);
            members.push((morphon, DeMorphonRole::Core));
        }
        
        // Express Memory cells (if present)
        if let Some(ref mem_genome) = self.memory_genome {
            for i in 0..self.body_plan.memory_count {
                let genome = mem_genome.mutate(rng, 0.005);  // very low mutation — memory must be stable
                let pos = offset_position(position, 0.02, rng);
                let morphon = genome.express(MorphonId::new(), pos,
                    available_energy * self.energy_allocation[2] / self.body_plan.memory_count.max(1) as f64);
                members.push((morphon, DeMorphonRole::Memory));
            }
        }
        
        // Express Output cells
        for i in 0..self.body_plan.output_count {
            let genome = self.output_genome.mutate(rng, 0.01);
            let pos = offset_position(position, 0.05, rng);
            let morphon = genome.express(MorphonId::new(), pos,
                available_energy * self.energy_allocation[3] / self.body_plan.output_count as f64);
            members.push((morphon, DeMorphonRole::Output));
        }
        
        // Wire internal connections from wiring_spec
        let demorphon = DeMorphon::from_members(members, self.body_plan.clone());
        demorphon.wire_internal(&self.wiring_spec, rng);
        
        // Apply epigenetic overrides for role specialization
        demorphon.apply_role_overrides();
        
        demorphon
    }
}
```

### 5.3 DeMorphon Reproduction

When a DeMorphon reproduces, its genome mutates at two levels simultaneously:

```rust
impl DeMorphonGenome {
    pub fn mutate(&self, rng: &mut impl Rng, strength: f64) -> Self {
        Self {
            genome_id: GenomeId::new(),
            parent_genome: Some(self.genome_id),
            generation: self.generation + 1,
            
            // Body plan mutations (rare, high impact)
            body_plan: self.body_plan.mutate(rng, strength * 0.3),  // slow body plan evolution
            wiring_spec: self.wiring_spec.iter()
                .map(|w| w.mutate(rng, strength * 0.5))
                .collect(),
            
            // Per-role genome mutations (frequent, lower impact)
            input_genome: self.input_genome.mutate(rng, strength),
            core_genome: self.core_genome.mutate(rng, strength),
            memory_genome: self.memory_genome.as_ref().map(|g| g.mutate(rng, strength * 0.5)),
            output_genome: self.output_genome.mutate(rng, strength),
            
            // Collective gene mutations
            cohesion_strength: mutate_f64(self.cohesion_strength, 0.0, 2.0, rng, strength),
            energy_allocation: mutate_allocation(self.energy_allocation, rng, strength),
            collective_division_threshold: mutate_f64(
                self.collective_division_threshold, 0.1, 10.0, rng, strength),
        }
    }
}
```

**Two-level mutation is biologically correct.** In biology, multicellular organisms evolve at two scales: the individual cell genome mutates at division (somatic mutation), and the organism's germline genome mutates at reproduction (heritable mutation). DeMorphon genome mutation captures both: per-role genomes mutate (cell-level), and the body plan mutates (organism-level).

---

## 6. Genome as Communication Protocol

### 6.1 Inter-System Genome Transfer

When two MORPHON instances share knowledge (the robot-to-robot scenario from the ANCS integration spec), genomes are the natural unit of transfer. Instead of sharing weights (environment-specific, non-portable), systems share genomes (universal blueprints that can be expressed in any environment):

```rust
// Robot A discovers a useful DeMorphon for "detecting slippery surfaces"
let genome = robot_a.export_genome(demorphon_id);

// Robot B imports the genome and expresses it in its own network
let new_demorphon = genome.express(
    available_position,
    available_energy,
    &mut robot_b.rng,
);
robot_b.add_demorphon(new_demorphon);
```

The imported DeMorphon starts with the same body plan but develops its own weights through learning in Robot B's environment. It's not a copy of Robot A's learned behavior — it's a copy of Robot A's *potential* for that behavior. Robot B's version might develop differently because its environment is different. This is exactly how biological organisms work: you inherit the genome, not the memories.

### 6.2 Genome as AXION-Compatible Payload

The genome serializes naturally to a compact representation suitable for AXION encoding:

```
AX:1.0 @demorphon_slippery_surface ⏳2026-04-02T10:00:00Z
[GENOME:v1] body={I:3,C:2,M:2,O:1} 
  input={th:0.3,rw:0.8,nv:0.5,ap:0.1,tp:0.03}
  core={th:0.7,rw:0.5,nv:0.3,ap:0.2,tp:0.08}
  memory={th:0.5,rw:0.1,nv:0.1,ap:0.0,tp:0.001}
  output={th:1.2,rw:0.9,nv:0.2,ap:0.3,tp:0.05}
  wiring=[I→C:0.5±0.1,C↔C:0.3±0.05,C→M:0.4,M↔M:0.8,C→O:0.6]
  social={coh:0.7,alt:0.6,div:3.5}
!HIGH_CONFIDENCE #sensor_torque_experiment_42
```

This is ~200 bytes — compact enough for inter-system broadcast, rich enough to express a complete organism.

---

## 7. Interaction with Existing Systems

### 7.1 Endoquilibrium

Endo doesn't modify genomes. It provides the epigenetic modulation that affects how genomes are expressed at runtime. The mutation rate at division IS influenced by Endo (novelty_gain scales mutation_strength), but the genome itself is only written at reproduction events, never during the Morphon's lifetime.

### 7.2 LocalParams

LocalParams becomes a component of MorphonGenome — the `learning_params` field. The existing `inherit_with_mutation()` on LocalParams is called by `MorphonGenome::mutate()`. No duplication — the genome wraps LocalParams, not replaces it.

### 7.3 TruthKeeper

TruthKeeper can flag a genome as "produces contested output" — if multiple Morphons with the same genome consistently generate contradicted beliefs, the genome itself is suspect. This is analogous to a genetic disease: the blueprint, not just the individual, is flawed. The Developmental Engine can then avoid expressing that genome in future divisions.

### 7.4 DeMorphon

The DeMorphon spec's `dissolution_snapshot` is replaced by `DeMorphonGenome` + `EpigeneticOverride`. Formation stores the genome. Fission removes the override and re-expresses the original individual genomes. Reproduction mutates the DeMorphon genome and expresses a new organism. Clean, principled, no ad-hoc snapshots.

---

## 8. Implementation Plan

| Phase | What | Effort | Depends on |
|---|---|---|---|
| Phase 0 | `MorphonGenome` struct + `GenomeId` type | 1 hour | Nothing |
| Phase 1 | `snapshot()` — extract genome from living Morphon | 1 hour | Phase 0 |
| Phase 2 | `express()` — build Morphon from genome | 2 hours | Phase 0 |
| Phase 3 | `mutate()` — mutation with configurable strength | 1 hour | Phase 0 |
| Phase 4 | Refactor `divide_morphon()` to use genome pipeline | 2 hours | Phases 1–3 |
| Phase 5 | Refactor `System::new()` to use founder genome | 1 hour | Phase 2 |
| Phase 6 | `distance()` — genetic distance metric | 1 hour | Phase 0 |
| Phase 7 | `crossover()` — two-parent reproduction | 1 hour | Phase 3 |
| Phase 8 | `DeMorphonGenome` struct + composite expression | 3 hours | Phase 2, DeMorphon spec |
| Phase 9 | `EpigeneticOverride` for DeMorphon role specialization | 2 hours | Phase 8 |
| Phase 10 | Genome serialization (serde + AXION-compatible format) | 2 hours | Phase 0 |
| Phase 11 | Validation: CartPole still solves with genome-based division | 1 hour | Phase 4 |
| Phase 12 | Analytics: lineage trees, genetic drift, speciation detection | 3 hours | Phase 6 |

**Total: ~21 hours.** Phases 0–5 give a working genome-based system with no behavior change (pure refactor). Phase 6–7 add evolutionary analysis capabilities. Phase 8–9 provide the DeMorphon inheritance foundation. Phase 10–12 add communication and analytics.

**Implementation order relative to DeMorphon:**

```
Week 1: MorphonGenome Phases 0–5 (genome foundation, refactor division)
Week 1: Validate CartPole still solves (Phase 11)
Week 2: DeMorphon Phases 0–3 (formation + body plan, USING genome)
Week 2: MorphonGenome Phase 8–9 (DeMorphonGenome + epigenetic overrides)
Week 3: DeMorphon Phases 4–6 (fission, reproduction, metabolics — all genome-based)
Week 3: MorphonGenome Phase 12 (lineage analytics for paper)
```

The genome is the foundation that DeMorphon builds on. They're implemented together but the genome comes first within each week.

---

## 9. Observability: The Genetic Observatory

### 9.1 Lineage Trees

Every genome has a `parent_genome` and `lineage_hash`. This creates a complete family tree of every Morphon and DeMorphon ever created. Visualization: a tree where each node is a genome, edges connect parent to child, and color indicates cell type or DeMorphon role.

### 9.2 Genetic Drift Measurement

```rust
fn genetic_drift(population: &[MorphonGenome], founder: &MorphonGenome) -> f64 {
    let distances: Vec<f64> = population.iter()
        .map(|g| g.distance(founder))
        .collect();
    distances.iter().sum::<f64>() / distances.len() as f64
}
```

Track over time: is the population diverging from the founder? High drift + high fitness = successful adaptation. High drift + low fitness = genetic degradation. Low drift = the founder genome was near-optimal (or mutation rate is too low).

### 9.3 Speciation Detection

If two subpopulations of Morphons have high within-group similarity but low between-group similarity (genetic distance between groups > 2× within-group distance), they're speciating — evolving into distinct "species" that could form different types of DeMorphons. This is an emergent phenomenon worth tracking for the paper.

### 9.4 Genome Fitness Correlation

Which genome fields correlate with high param_fitness? Run a correlation analysis across the population:

```rust
fn genome_fitness_correlations(population: &[(MorphonGenome, f64)]) -> HashMap<String, f64> {
    // For each genome field, compute Pearson correlation with fitness
    // High correlation = this gene matters for survival
    // Near-zero = this gene is neutral (evolutionary drift)
    // Negative = this gene is maladaptive
}
```

This tells you which parts of the genome are under selection pressure and which are drifting neutrally — directly informative for the paper's analysis of what the system has learned about itself.

---

## 10. For the Paper

The MorphonGenome provides two distinct paper contributions:

**For the MORPHON architecture paper (arXiv):** "We introduce a genotype-phenotype separation in morphogenic intelligence, where each compute unit carries an immutable heritable genome that specifies its developmental potential, while its runtime behavior (phenotype) emerges from gene expression modulated by the neuroendocrine regulation system (Endoquilibrium). Division operates through genome mutation and expression, implementing Darwinian evolution within the network at runtime."

**For the DeMorphon paper (ALIFE/GECCO):** "DeMorphon reproduction is genome-based: the composite organism's genome specifies both the body plan (cell roles and wiring) and per-role cell genomes. Mutations operate at two scales — body plan evolution (rare, high-impact) and cell-level evolution (frequent, low-impact) — mirroring the distinction between regulatory and structural mutations in biological evolution. The self-referential mutation rate gene enables the evolution of evolvability."

---

## 11. References

Ispolatov, I., Ackermann, M., & Doebeli, M. (2012). Division of labour and the evolution of multicellularity. Proc. R. Soc. B, 279(1734), 1768–1776.

Taddei, F., Radman, M., Maynard Smith, J., Toupance, B., Gouyon, P.H., & Godelle, B. (1997). Role of mutator alleles in adaptive evolution. Nature, 387(6634), 700–702.

Maynard Smith, J. & Szathmáry, E. (1995). The Major Transitions in Evolution. Oxford University Press.

Levin, M. (2023). Bioelectric networks as cognitive glue. Animal Cognition, 26, 1865–1891.

Fields, R.D. (2015). A new mechanism of nervous system plasticity: activity-dependent myelination. Nature Reviews Neuroscience, 16(12), 756–767.

---

*MorphonGenome — because every organism begins as a sentence written in the language of possibility.*

*TasteHub GmbH, Wien, April 2026*
