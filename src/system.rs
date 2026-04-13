//! The MI System — top-level orchestration of the Morphogenic Intelligence engine.
//!
//! Uses the Dual-Clock Architecture (Section 3.8) to separate fast inference
//! from slow morphogenesis, with homeostatic protection mechanisms throughout.

/// V6: Graceful Degradation — energy thresholds for three-mode operation.
/// Full mode (> DEGRADED): three-factor learning with all neuromodulation channels.
/// Degraded mode (SURVIVAL..DEGRADED]: simplified learning — only homeostasis channel.
/// Survival mode (≤ SURVIVAL): no firing, no weight updates — energy regeneration only.
/// Thresholds are near-depletion (0.05/0.1), not mid-range — synapse maintenance costs
/// drive morphon energy well below 0.5 during normal training, so 0.5 would disable
/// three-factor learning for the vast majority of training steps.
const ENERGY_DEGRADED_THRESHOLD: f64 = 0.1;
const ENERGY_SURVIVAL_THRESHOLD: f64 = 0.05;

use crate::ancs::{AncsConfig, InMemoryBackend, MemoryBackend, MemoryItem, SystemHeartbeat, classify_tier};
use crate::developmental::{self, DevelopmentalConfig, TargetMorphology};
use crate::diagnostics::Diagnostics;
use crate::field::{FieldConfig, MorphonField};
use crate::homeostasis::{self, HomeostasisParams};
use crate::learning::{self, LearningParams};
use crate::lineage::{self, LineageTree};
use crate::memory::TripleMemory;
use crate::morphogenesis::{self, Cluster, MorphogenesisParams, MorphogenesisReport};
use crate::morphon::{MetabolicConfig, Morphon};
use crate::neuromodulation::Neuromodulation;
use crate::resonance::ResonanceEngine;
use crate::scheduler::SchedulerConfig;
use crate::topology::Topology;
use crate::types::*;
use rand::Rng;
use rand::SeedableRng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for creating a new MI System.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub developmental: DevelopmentalConfig,
    pub learning: LearningParams,
    pub morphogenesis: MorphogenesisParams,
    pub homeostasis: HomeostasisParams,
    pub scheduler: SchedulerConfig,
    pub lifecycle: LifecycleConfig,
    /// V3 Metabolic Budget parameters.
    pub metabolic: MetabolicConfig,
    pub working_memory_capacity: usize,
    pub episodic_memory_capacity: usize,
    /// Timestep size for simulation.
    pub dt: f64,
    /// V2: Bioelektrisches Feld configuration.
    #[serde(default)]
    pub field: FieldConfig,
    /// V2: Target Morphology — functional region targets with self-healing.
    #[serde(default)]
    pub target_morphology: Option<TargetMorphology>,

    /// V3: Constitutional constraints — hard governance invariants.
    #[serde(default)]
    pub governance: crate::governance::ConstitutionalConstraints,

    /// Endoquilibrium: predictive neuroendocrine regulation.
    #[serde(default)]
    pub endoquilibrium: crate::endoquilibrium::EndoConfig,

    /// V2: Dreaming engine — offline consolidation and self-optimization.
    #[serde(default)]
    pub dream: DreamConfig,

    /// How the analog readout layer should be trained.
    /// Supervised = caller provides correct action (fastest, needs domain knowledge).
    /// TDOnly = learn from TD error signal only (general, slower).
    /// Hybrid = supervised until consolidation_gate, then TD-only.
    #[serde(default)]
    pub readout_mode: ReadoutTrainingMode,

    /// Optional RNG seed for fully deterministic initialization and training.
    /// When set, System::new(), enable_analog_readout(), and step() all use
    /// a seeded SmallRng — two runs with the same config + seed produce identical results.
    /// When None (default), seeded from OS entropy (non-deterministic).
    #[serde(default)]
    pub rng_seed: Option<u64>,

    /// ANCS-Core memory substrate configuration.
    #[serde(default)]
    pub ancs: AncsConfig,
}

impl SystemConfig {
    /// Save this configuration to a JSON file for reproducibility and sweeps.
    pub fn save_json(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load a configuration from a JSON file.
    pub fn load_json(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            developmental: DevelopmentalConfig::default(),
            learning: LearningParams::default(),
            morphogenesis: MorphogenesisParams::default(),
            homeostasis: HomeostasisParams::default(),
            scheduler: SchedulerConfig::default(),
            lifecycle: LifecycleConfig::default(),
            metabolic: MetabolicConfig::default(),
            working_memory_capacity: 7,
            episodic_memory_capacity: 1000,
            dt: 1.0,
            field: FieldConfig::default(),
            target_morphology: None,
            governance: crate::governance::ConstitutionalConstraints::default(),
            endoquilibrium: crate::endoquilibrium::EndoConfig::default(),
            dream: DreamConfig::default(),
            readout_mode: ReadoutTrainingMode::default(),
            rng_seed: None,
            ancs: AncsConfig::default(),
        }
    }
}

/// Inspection results for the system state.
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemStats {
    /// Effective morphon cap (auto-derived or explicitly set).
    pub max_morphons: usize,
    /// Whether the system has reached its morphon cap (division blocked).
    pub at_morphon_cap: bool,
    pub total_morphons: usize,
    pub total_synapses: usize,
    pub fused_clusters: usize,
    pub differentiation_map: HashMap<CellType, usize>,
    pub max_generation: Generation,
    pub avg_energy: f64,
    pub avg_prediction_error: f64,
    pub firing_rate: f64,
    pub working_memory_items: usize,
    pub episodic_memory_items: usize,
    pub step_count: u64,
    pub total_born: usize,
    pub total_died: usize,
    pub total_transdifferentiations: usize,
    // Apoptosis eligibility (from diagnostics)
    pub apoptosis_age_eligible: usize,
    pub apoptosis_silent: usize,
    pub apoptosis_energy_low: usize,
    pub assoc_activity_min: f64,
    pub assoc_activity_max: f64,
    pub assoc_activity_mean: f64,
    /// V2: Target morphology region health: (region_idx, current, target).
    pub region_health: Vec<(usize, usize, usize)>,
    /// Bioelectric field: peak prediction-error value (0.0 if field disabled).
    pub field_pe_max: f64,
    /// Bioelectric field: mean prediction-error value (0.0 if field disabled).
    pub field_pe_mean: f64,
    /// Axonal properties: mean myelination level across all synapses.
    #[serde(default)]
    pub avg_myelination: f64,
    /// Axonal properties: max myelination level across all synapses.
    #[serde(default)]
    pub max_myelination: f64,
    /// Axonal properties: number of synapses with myelination > 0.01.
    #[serde(default)]
    pub myelinated_synapses: usize,
    /// Axonal properties: mean effective delay across all synapses.
    #[serde(default)]
    pub avg_effective_delay: f64,
    /// Axonal properties: minimum base delay across all synapses (before myelination).
    #[serde(default)]
    pub min_base_delay: f64,
    /// Axonal properties: maximum base delay across all synapses (before myelination).
    #[serde(default)]
    pub max_base_delay: f64,

    // ── ANCS-Core ─────────────────────────────────────────────────────────────
    /// Total memory items across all tiers in the ANCS-Core store.
    #[serde(default)]
    pub ancs_items: usize,
    /// Items currently in Contested epistemic state (pending reconsolidation).
    #[serde(default)]
    pub ancs_contested: usize,
    /// Items currently in Supported epistemic state.
    #[serde(default)]
    pub ancs_supported: usize,
    /// Current F7 energy pressure mode (Normal / Pressure / Emergency / Critical).
    #[serde(default)]
    pub ancs_pressure: String,

    // ── Phase 4 diagnostic ────────────────────────────────────────────────────
    /// Synapses rescued by the forward_importance guard in the last slow tick.
    /// Non-zero confirms Phase 4 is actively protecting reward-path synapses.
    #[serde(default)]
    pub synapses_saved_fwd_recent: u32,
}

/// The Morphogenic Intelligence System.
pub struct System {
    /// All Morphons indexed by ID.
    pub morphons: HashMap<MorphonId, Morphon>,
    /// The connection topology.
    pub topology: Topology,
    /// The resonance (signal propagation) engine.
    pub resonance: ResonanceEngine,
    /// Global neuromodulation state.
    pub modulation: Neuromodulation,
    /// Fused clusters.
    pub clusters: HashMap<ClusterId, Cluster>,
    /// Triple memory system.
    pub memory: TripleMemory,
    /// ANCS-Core in-process memory substrate (Phases 0–3).
    pub ancs: InMemoryBackend,
    /// Cache-line-sized atomic snapshot of global system state.
    pub heartbeat: SystemHeartbeat,
    /// ID of the most recently stored ANCS memory item.
    /// Written on every episodic encode, read by justification records
    /// to link synapses → the ANCS item active during their reinforcement.
    pub(crate) current_ancs_item: Option<u64>,

    /// Configuration.
    pub config: SystemConfig,

    /// Stable input port mapping: index → MorphonId.
    /// These are the sensory Morphons that receive external input.
    pub(crate) input_ports: Vec<MorphonId>,
    /// Stable output port mapping: index → MorphonId.
    /// These are the motor Morphons that produce external output.
    pub(crate) output_ports: Vec<MorphonId>,
    /// Critic morphon IDs — a subset of Associative morphons that predict
    /// state value V(s). Their aggregate membrane potential is the value estimate.
    /// Biologically: ventral striatum (critic) vs dorsal striatum (actor).
    pub(crate) critic_ports: Vec<MorphonId>,
    /// Previous critic value — for TD error computation.
    pub(crate) prev_critic_value: f64,
    /// Last computed TD error — used by critic morphons' TD-LTP learning rule.
    pub(crate) last_td_error: f64,
    /// DFA feedback weights: for each associative morphon, a Vec of (motor_id, weight).
    /// Fixed random weights initialized during development. NEVER updated.
    /// Projects output error to hidden layer for credit assignment.
    /// (Lillicrap et al. 2016, GLSNN: global feedback + local STDP)
    pub(crate) feedback_weights: HashMap<MorphonId, Vec<(MorphonId, f64)>>,

    /// Analog readout weights: [output_index][assoc_morphon_id] → weight.
    /// These bypass the spike propagation pipeline entirely.
    /// V_motor_j = Σ_i readout_weights[j][i] × sigmoid(assoc_i.potential)
    /// Updated by the delta rule with exact gradients.
    /// Biologically: Purkinje cell dendritic integration (analog, not spike-based).
    pub(crate) readout_weights: Vec<HashMap<MorphonId, f64>>,
    /// Per-output bias for analog readout. Absorbs the constant offset from
    /// sigmoid(resting_potential) ≈ 0.5, freeing weights to learn state differences.
    pub(crate) readout_bias: Vec<f64>,
    /// Whether to use analog readout (true) or spike-based motor potential (false).
    pub(crate) use_analog_readout: bool,

    /// Next available Morphon ID.
    pub(crate) next_morphon_id: MorphonId,
    /// Next available Cluster ID.
    pub(crate) next_cluster_id: ClusterId,
    /// Current simulation step.
    pub(crate) step_count: u64,

    /// Learning pipeline diagnostics (updated on homeostasis ticks, not every step).
    pub(crate) diag: Diagnostics,

    /// Cumulative morphon birth count.
    pub(crate) total_born: usize,
    /// Cumulative morphon death count.
    pub(crate) total_died: usize,
    /// Per-morphon firing counts for winner diversity entropy. Reset per episode.
    pub(crate) episode_fire_counts: HashMap<MorphonId, u64>,
    /// Cumulative transdifferentiation count.
    pub(crate) total_transdifferentiations: usize,

    /// Running average of recent episode performance (set by caller via report_performance).
    /// Used to gate consolidation: no captures until performance exceeds a threshold.
    pub(crate) recent_performance: f64,
    /// Performance threshold for enabling consolidation.
    /// Below this, the system stays fully plastic. Above, it starts consolidating.
    pub(crate) consolidation_gate: f64,
    /// Peak performance seen so far — used for deconsolidation trigger.
    pub(crate) peak_performance: f64,

    /// Running average of episode steps — for episode-relative capture decisions.
    pub(crate) running_avg_steps: f64,

    /// V2: Bioelektrisches Feld — spatial field for indirect morphon communication.
    pub field: Option<MorphonField>,

    /// Endoquilibrium: predictive neuroendocrine regulation engine.
    pub endo: crate::endoquilibrium::Endoquilibrium,

    /// Resolved morphon cap (from config or auto-derived from I/O dimensions).
    pub(crate) effective_max_morphons: usize,

    /// Cached per-morphon synapse maintenance costs. Recomputed on slow ticks
    /// (when topology changes via synaptogenesis/pruning) instead of every step.
    pub(crate) cached_maintenance_costs: HashMap<MorphonId, f64>,

    /// Hot-path arrays for cache-friendly spike processing.
    /// Rebuilt on slow/glacial path, synced to structs on medium path.
    pub(crate) hot: crate::hot_arrays::HotArrays,

    /// Endoquilibrium threshold bias cached as f32 for hot-path use.
    /// Updated on medium tick from endo.channels.threshold_bias.
    pub(crate) endo_threshold_bias: f32,

    /// V6: Auto-Merge candidate buffer — tracks co-activating proximate morphon groups.
    pub(crate) auto_merge_candidates: morphogenesis::AutoMergeCandidates,

    /// V6: Confirmed auto-merge groups waiting to be processed by the next glacial fusion pass.
    /// Populated on slow ticks, drained on glacial ticks.
    pub(crate) pending_merge_groups: Vec<Vec<crate::types::MorphonId>>,

    /// Seeded RNG for deterministic runs. Initialized from config.rng_seed if set,
    /// otherwise from OS entropy. All rand calls in System use this — never rand::rng().
    pub(crate) rng: rand::rngs::SmallRng,

    /// Limbic Circuit — per-stimulus salience detection, RPE computation, episodic tagging.
    /// Disabled by default (`limbic.enabled = false`). Enable per-example to activate.
    pub limbic: crate::limbic::LimbicCircuit,
}

impl System {
    /// Create a new MI System by running its developmental program.
    pub fn new(config: SystemConfig) -> Self {
        let mut rng = match config.rng_seed {
            Some(seed) => rand::rngs::SmallRng::seed_from_u64(seed),
            None => rand::rngs::SmallRng::from_rng(&mut rand::rng()),
        };

        // Resolve effective morphon cap: governance takes precedence over morphogenesis.
        // The governance cap is a constitutional constraint the system cannot override.
        let effective_max_morphons = config.governance.max_morphons.unwrap_or_else(|| {
            config.morphogenesis.resolve_max_morphons(
                config.developmental.target_input_size,
                config.developmental.target_output_size,
            )
        });

        let (mut morphons, topology, next_id) =
            developmental::develop(&config.developmental, &config.homeostasis, &mut rng);

        // Build stable I/O port mappings from the developmental result
        let mut sensory_ids: Vec<MorphonId> = morphons
            .values()
            .filter(|m| m.cell_type == CellType::Sensory)
            .map(|m| m.id)
            .collect();
        sensory_ids.sort();

        let mut motor_ids: Vec<MorphonId> = morphons
            .values()
            .filter(|m| m.cell_type == CellType::Motor)
            .map(|m| m.id)
            .collect();
        motor_ids.sort();

        // Designate ~15% of Associative morphons as critic morphons.
        // They receive sensory input like regular Associatives but their
        // aggregate potential encodes V(state) for TD error computation.
        let mut assoc_ids: Vec<MorphonId> = morphons
            .values()
            .filter(|m| m.cell_type == CellType::Associative)
            .map(|m| m.id)
            .collect();
        assoc_ids.sort();
        let n_critics = (assoc_ids.len() as f64 * 0.15).ceil() as usize;
        let critic_ids: Vec<MorphonId> = assoc_ids.into_iter().take(n_critics).collect();

        // Set critic morphons' receptors to {Reward, Homeostasis} only.
        for &cid in &critic_ids {
            if let Some(m) = morphons.get_mut(&cid) {
                let mut receptors = std::collections::HashSet::new();
                receptors.insert(ModulatorType::Reward);
                receptors.insert(ModulatorType::Homeostasis);
                m.receptors = receptors;
            }
        }

        // Initialize DFA feedback weights: Motor → Associative (fixed random).
        // Each Associative morphon gets a random projection weight from each Motor.
        // These weights are NEVER updated — the forward weights align to them.
        let all_assoc_ids: Vec<MorphonId> = morphons
            .values()
            .filter(|m| m.cell_type == CellType::Associative)
            .map(|m| m.id)
            .collect();
        let mut feedback_weights: HashMap<MorphonId, Vec<(MorphonId, f64)>> = HashMap::new();
        for (i, &aid) in all_assoc_ids.iter().enumerate() {
            let motor_weights: Vec<(MorphonId, f64)> = motor_ids
                .iter()
                .enumerate()
                .map(|(j, &mid)| {
                    // Deterministic pseudo-random weight from morphon IDs
                    let hash = (aid.wrapping_mul(7919) ^ mid.wrapping_mul(104729))
                        .wrapping_add(i as u64 * 31 + j as u64 * 97);
                    let w = (hash % 10000) as f64 / 5000.0 - 1.0; // [-1, 1]
                    (mid, w)
                })
                .collect();
            feedback_weights.insert(aid, motor_weights);
        }

        let endo = crate::endoquilibrium::Endoquilibrium::new(config.endoquilibrium.clone());
        let ancs_config = config.ancs.clone();
        let limbic_input_size = config.developmental.target_input_size.unwrap_or(16);
        let mut system = System {
            morphons,
            topology,
            resonance: ResonanceEngine::new(),
            modulation: Neuromodulation::default(),
            clusters: HashMap::new(),
            memory: TripleMemory::new(
                config.working_memory_capacity,
                config.episodic_memory_capacity,
            ),
            config,
            input_ports: sensory_ids,
            output_ports: motor_ids,
            critic_ports: critic_ids,
            prev_critic_value: 0.0,
            last_td_error: 0.0,
            feedback_weights,
            readout_weights: Vec::new(), // initialized below after struct creation
            readout_bias: Vec::new(),
            use_analog_readout: false, // off by default, enabled per-task
            next_morphon_id: next_id,
            next_cluster_id: 0,
            step_count: 0,
            diag: Diagnostics::default(),
            total_born: 0,
            total_died: 0,
            episode_fire_counts: HashMap::new(),
            total_transdifferentiations: 0,
            recent_performance: 0.0,
            consolidation_gate: 10.0, // consolidate once above random baseline
            peak_performance: 0.0,
            running_avg_steps: 9.0, // initialize to random baseline
            field: None,            // initialized below
            endo,
            effective_max_morphons,
            cached_maintenance_costs: HashMap::new(),
            hot: crate::hot_arrays::HotArrays::new(),
            endo_threshold_bias: 0.0,
            auto_merge_candidates: morphogenesis::AutoMergeCandidates::default(),
            pending_merge_groups: Vec::new(),
            rng,
            limbic: crate::limbic::LimbicCircuit::new(limbic_input_size),
            ancs: InMemoryBackend::new(ancs_config),
            heartbeat: SystemHeartbeat::default(),
            current_ancs_item: None,
        };

        // V2: Initialize bioelectric field if enabled
        if system.config.field.enabled {
            // Auto-add Identity layer if target morphology is configured
            if system.config.target_morphology.is_some()
                && !system
                    .config
                    .field
                    .active_layers
                    .contains(&crate::field::FieldType::Identity)
            {
                system
                    .config
                    .field
                    .active_layers
                    .push(crate::field::FieldType::Identity);
            }
            system.field = Some(MorphonField::new(system.config.field.clone()));
        }

        // === Spontaneous developmental activity ===
        // Analogous to retinal waves and cortical spontaneous bursting in utero.
        // Drives noise through the network with sustained modulation to establish
        // initial firing correlations. Lifecycle events (division, fusion, etc.) are
        // disabled — the warm-up is for learning correlations, not structural changes.
        let saved_lifecycle = system.config.lifecycle.clone();
        system.config.lifecycle = LifecycleConfig {
            division: false,
            differentiation: false,
            fusion: false,
            apoptosis: false,
            migration: false,
            synaptogenesis: false, // warm-up builds correlations, not new structure
        };

        for step in 0..100_u64 {
            let n_inputs = system.input_ports.len();
            if n_inputs > 0 {
                let noise_input: Vec<f64> = (0..n_inputs)
                    .map(|i| {
                        let v = ((i as u64).wrapping_mul(step.wrapping_add(997)) % 1000) as f64
                            / 1000.0;
                        0.5 + v * 1.0
                    })
                    .collect();
                system.feed_input(&noise_input);
            }
            system.modulation.inject_reward(0.3);
            system.modulation.inject_novelty(0.3);
            system.modulation.inject_arousal(0.3);
            system.step();
        }

        // Reset to clean state — system appears freshly created to the caller.
        system.config.lifecycle = saved_lifecycle;
        system.modulation = Neuromodulation::default();
        system.step_count = 0;
        system.resonance.clear();
        for m in system.morphons.values_mut() {
            m.division_pressure = 0.0;
            m.potential = 0.0;
            m.prev_potential = 0.0;
            m.prediction_error = 0.0;
            m.desire = 0.0;
            m.frustration = FrustrationState::default();
            m.fired = false;
            m.input_accumulator = 0.0;
            m.refractory_timer = 0.0;
            m.energy = 1.0; // restore full energy after warm-up
                            // Lower Associative threshold to 60% of default — ensures 5-10% firing rate
                            // so the readout has diverse activity patterns to learn from.
            if m.cell_type == CellType::Associative || m.cell_type == CellType::Stem {
                m.threshold *= 0.6;
            }
        }

        // Reset motor synapse TRACES (not weights) — warm-up builds eligibility
        // that would bias the first few learning updates. Keep Xavier-scaled weights
        // from developmental program for symmetry-breaking.
        // Reset ALL synapse learning state — warm-up creates tags and captures
        // that would prematurely consolidate before real training starts.
        for ei in system.topology.graph.edge_indices() {
            if let Some(syn) = system.topology.graph.edge_weight_mut(ei) {
                syn.eligibility = 0.0;
                syn.pre_trace = 0.0;
                syn.post_trace = 0.0;
                syn.activity_trace = 0.0;
                syn.myelination = 0.0;
                syn.tag = 0.0;
                syn.consolidated = false; // critical: undo warm-up captures
            }
        }
        system.diag.total_captures = 0;

        // === ANCHOR & SAIL: Assign heterogeneous plasticity rates ===
        // Log-normal distribution: ~20% anchors (rate < 0.3), ~60% normal, ~20% sails (rate > 1.5)
        // Anchors provide stable features for the readout; sails explore the state space.
        // Only Associative and Stem morphons get heterogeneous rates — Sensory/Motor stay at 1.0.
        // (Perez-Nieves et al. 2021, Zenke et al. 2015)
        for m in system.morphons.values_mut() {
            if m.cell_type == CellType::Associative || m.cell_type == CellType::Stem {
                // Log-normal(mu=-0.3, sigma=0.7) → median ~0.74, ~20% below 0.3, ~20% above 1.5
                let u1: f64 = system.rng.random_range(0.001..1.0_f64);
                let u2: f64 = system.rng.random_range(0.0..std::f64::consts::TAU);
                let normal = (-2.0 * u1.ln()).sqrt() * u2.cos(); // Box-Muller
                let log_normal = (-0.3 + 0.7 * normal).exp();
                m.plasticity_rate = log_normal.clamp(0.1, 2.5);
            }
            // Sensory, Motor, Modulatory keep plasticity_rate = 1.0
        }

        // Build initial hot arrays from developmental state
        system.rebuild_hot_arrays();

        system
    }

    // =========================================================================
    // Hot-array methods (Pulse Kernel Lite)
    // =========================================================================

    /// Rebuild hot arrays from scratch after any structural change.
    /// Called after slow/glacial path or on initialization.
    fn rebuild_hot_arrays(&mut self) {
        let n = self.morphons.len();
        self.hot.allocate(n);
        self.hot.id_to_idx.clear();
        self.hot.active_count = 0;

        for (&id, morphon) in &self.morphons {
            let j = self.hot.active_count;
            self.hot.voltage[j] = morphon.potential as f32;
            self.hot.threshold[j] = morphon.threshold as f32;
            self.hot.input_current[j] = 0.0;
            self.hot.fired[j] = morphon.fired;
            self.hot.fired_prev[j] = false;
            self.hot.refractory[j] = morphon.refractory_timer as f32;
            self.hot.leak_decay[j] = if morphon.cell_type == CellType::Motor { 0.0 } else { 0.9 };
            self.hot.energy[j] = morphon.energy as f32;
            self.hot.idx_to_id[j] = id;
            self.hot.id_to_idx.insert(id, j);
            self.hot.active_count += 1;
        }
    }

    /// Sync hot-path results into Morphon structs for medium-path learning rules.
    /// Sets prev_potential BEFORE overwriting potential (for PE computation in medium_update).
    fn sync_hot_to_structs(&mut self) {
        for j in 0..self.hot.active_count {
            let id = self.hot.idx_to_id[j];
            if let Some(morphon) = self.morphons.get_mut(&id) {
                morphon.prev_potential = morphon.potential; // save old before overwrite
                morphon.potential = self.hot.voltage[j] as f64;
                morphon.fired = self.hot.fired[j];
                morphon.refractory_timer = self.hot.refractory[j] as f64;
                morphon.energy = self.hot.energy[j] as f64;
            }
        }
    }

    /// Sync updated thresholds from Morphon structs back to hot arrays.
    /// Called at end of medium-path tick after medium_update() ran.
    fn sync_structs_to_hot(&mut self) {
        let bias = self.endo.channels.threshold_bias;
        self.endo_threshold_bias = bias as f32;
        for j in 0..self.hot.active_count {
            let id = self.hot.idx_to_id[j];
            if let Some(morphon) = self.morphons.get(&id) {
                self.hot.threshold[j] = morphon.threshold as f32;
                self.hot.energy[j] = morphon.energy as f32;
                // Also update leak_decay in case cell type changed
                self.hot.leak_decay[j] = if morphon.cell_type == CellType::Motor { 0.0 } else { 0.9 };
            }
        }
    }

    /// Fast-path step: runs hot-array integration + threshold + reset.
    /// Replaces per-morphon morphon.step() calls for the fast path.
    /// Structs are accessed minimally: input_accumulator extract, age++, activity_history, energy.
    fn fast_step(&mut self, dt: f32) {
        let bias = self.endo_threshold_bias;
        let n = self.hot.active_count;

        // 1. Swap fired buffers
        std::mem::swap(&mut self.hot.fired, &mut self.hot.fired_prev);

        // 2. Add non-spike input accumulated in morphon structs since last fast tick:
        //    DFA feedback, external injection (feed_input), homeostasis drive, curiosity.
        //    Spike currents were already written into hot.input_current by resonance.deliver().
        //    += not = because spikes are already there; zero the struct accumulator after.
        for j in 0..n {
            let id = self.hot.idx_to_id[j];
            if let Some(m) = self.morphons.get_mut(&id) {
                self.hot.input_current[j] += m.input_accumulator as f32;
                m.input_accumulator = 0.0;
                m.age += 1;
            }
        }

        // 3. Fast integrate (pure hot array — no struct access)
        for j in 0..n {
            if self.hot.refractory[j] > 0.0 {
                // Decrement refractory, skip integration
                self.hot.refractory[j] = (self.hot.refractory[j] - dt).max(0.0);
                self.hot.fired[j] = false;
                continue;
            }
            // Leaky integration: V = V * decay + I
            let v = self.hot.voltage[j] * self.hot.leak_decay[j] + self.hot.input_current[j];
            self.hot.voltage[j] = v.clamp(-10.0, 10.0);

            // Threshold check: voltage + energy gate.
            // V6 Graceful Degradation: survival mode (energy ≤ 0.2) silences the morphon
            // without killing it — energy can regenerate and firing resumes when restored.
            let fires = self.hot.voltage[j] >= self.hot.threshold[j] + bias
                && self.hot.energy[j] > ENERGY_SURVIVAL_THRESHOLD as f32;
            self.hot.fired[j] = fires;

            if fires {
                // Deduct firing cost from hot.energy (no struct access needed).
                // Clamped to 0.0 — prevents negative energy runaway between medium ticks.
                let firing_cost = self.config.metabolic.firing_cost as f32;
                self.hot.energy[j] = (self.hot.energy[j] - firing_cost).max(0.0);
                // Set refractory (will be checked next tick)
                self.hot.refractory[j] = dt; // 1 tick of refractory
            }
        }

        // 4. Sync fired back to structs (resonance.propagate reads morphon.fired)
        //    Also push to activity_history and write energy back to struct.
        for j in 0..n {
            let id = self.hot.idx_to_id[j];
            if let Some(m) = self.morphons.get_mut(&id) {
                m.fired = self.hot.fired[j];
                m.refractory_timer = self.hot.refractory[j] as f64;
                m.energy = self.hot.energy[j] as f64;
                if self.hot.fired[j] {
                    m.activity_history.push(1.0);
                } else if self.hot.refractory[j] <= 0.0 {
                    // Only push 0.0 if not in refractory (matches original behavior)
                    m.activity_history.push(0.0);
                }
            }
        }
    }

    // =========================================================================
    // End of hot-array methods
    // =========================================================================

    /// Run one simulation step using the dual-clock architecture.
    ///
    /// The step cycle respects the scheduler's timing:
    /// - Fast (every step): spike propagation, resonance, Morphon state updates
    /// - Medium: eligibility traces, weight updates, synaptic scaling
    /// - Slow: synaptogenesis, pruning, migration
    /// - Glacial: division, differentiation, fusion, apoptosis
    /// - Homeostasis: synaptic scaling, inter-cluster inhibition
    pub fn step(&mut self) -> MorphogenesisReport {
        let dt = self.config.dt;
        self.step_count += 1;

        let tick = self.config.scheduler.tick(self.step_count);
        let mut report = MorphogenesisReport::default();

        // === FAST PATH (every step) ===
        // 1. Propagate spikes from currently firing Morphons
        self.resonance.propagate(&self.morphons, &self.topology);

        // 2. Deliver spikes that have reached their targets.
        //    Spike currents land directly in hot.input_current (Vec index lookup)
        //    instead of morphon.input_accumulator (HashMap lookup). hot.input_current
        //    is cleared here so last tick's spike contributions don't carry over.
        self.hot.input_current.fill(0.0);
        let delivered = self.resonance.deliver(&mut self.hot.input_current, &self.hot.id_to_idx, dt);

        let spikes_delivered = delivered.len();

        // 3. Lateral inhibition BEFORE firing decision.
        //    Inhibitory interneurons suppress neighbors through spike delivery (step 2).
        //    iSTDP tunes their synapse weights on the medium path to maintain target firing rate.

        // 4. Update all Morphon states (integrate input, fire/not-fire).
        let metabolic = &self.config.metabolic;
        // Apply Endo Phase B lever: frustration_sensitivity_mult scales stagnation_threshold.
        // Higher mult → higher threshold → more tolerant (Proliferating).
        // Lower mult → lower threshold → more sensitive (Mature).
        // Recompute maintenance costs only on slow ticks (topology changes)
        // or on the very first step (cache empty). Avoids O(N×M) hyperbolic
        // distance calculations every step.
        if tick.slow || self.cached_maintenance_costs.is_empty() {
            let synapse_cost_base = metabolic.synapse_cost;
            self.cached_maintenance_costs = self
                .morphons
                .keys()
                .map(|&id| {
                    let total_cost: f64 = self
                        .topology
                        .outgoing(id)
                        .into_iter()
                        .map(|(target_id, synapse)| {
                            let distance_factor = self
                                .morphons
                                .get(&target_id)
                                .map(|target| {
                                    let dist =
                                        self.morphons[&id].position.distance(&target.position);
                                    1.0 + dist * 0.5
                                })
                                .unwrap_or(1.0);
                            let myelination_cost = synapse.myelination * 0.002;
                            synapse_cost_base * distance_factor + myelination_cost
                        })
                        .sum();
                    (id, total_cost)
                })
                .collect();
        }
        // Hot-path fast step: integrates voltage, checks threshold, handles refractory.
        // Replaces per-morphon morphon.step() in the hot path.
        // metabolic/PE/frustration/threshold regulation run on medium path via medium_update().
        self.fast_step(dt as f32);

        // Extract iSTDP params once — used in both the post-spike loop below
        // and the pre-spike loop inside the spike-delivery section further down.
        let istdp_params: (f64, f64) = {
            use crate::homeostasis::CompetitionMode;
            let CompetitionMode::LocalInhibition { istdp_rate, target_rate, .. } =
                &self.config.homeostasis.competition_mode;
            let eta = *istdp_rate;
            let rho0 = target_rate.unwrap_or_else(|| self.endo.target_assoc_firing_rate());
            let tau = self.config.learning.tau_trace;
            (eta, 2.0 * rho0 * tau)
        };

        // iSTDP post-spike (Vogels 2011): for each fired excitatory morphon, nudge
        // its incoming inhibitory synapses. Δw = -η × (pre_trace - α).
        // Runs per fast tick — O(fired_exc × avg_inh_in) ≈ very small.
        {
            let (eta, alpha) = istdp_params;
            let fired_exc_ids: Vec<MorphonId> = self
                .morphons
                .values()
                .filter(|m| {
                    m.fired
                        && matches!(
                            m.cell_type,
                            CellType::Associative | CellType::Stem | CellType::Motor
                        )
                })
                .map(|m| m.id)
                .collect();

            for id in fired_exc_ids {
                // incoming_synapses_mut returns Vec — mutable borrow of topology ends here.
                let incoming = self.topology.incoming_synapses_mut(id);
                let inh_edges: Vec<petgraph::graph::EdgeIndex> = incoming
                    .into_iter()
                    .filter_map(|(src_id, ei)| {
                        if self
                            .morphons
                            .get(&src_id)
                            .map(|m| m.cell_type == CellType::InhibitoryInterneuron)
                            .unwrap_or(false)
                        {
                            Some(ei)
                        } else {
                            None
                        }
                    })
                    .collect();
                for ei in inh_edges {
                    if let Some(syn) = self.topology.synapse_mut(ei) {
                        crate::learning::istdp_post(syn, eta, alpha);
                    }
                }
            }
        }

        // Reward-correlated energy: morphons that fire when reward arrives earn energy.
        // Hubs fire for all inputs → reward ±symmetric → net zero income.
        // Specialists fire for correct class → mostly positive → positive income.
        let reward_energy_coeff = self.config.metabolic.reward_energy_coefficient;
        if reward_energy_coeff > 0.0 {
            let reward = self.modulation.reward_delta();
            if reward > 0.0 {
                for m in self.morphons.values_mut() {
                    if m.fired {
                        m.energy = (m.energy + reward * reward_energy_coeff).min(1.0);
                    }
                }
            }
        }

        // V3: Cluster overhead — fused morphons pay extra maintenance cost.
        // V2: Cluster collective compute — energy pooling + shared homeostasis
        {
            let cluster_overhead = self.config.metabolic.cluster_overhead_per_tick;
            let draw_per_tick = self.config.metabolic.cluster_energy_draw_per_tick;

            // Collect cluster member lists to avoid borrow conflicts
            let cluster_data: Vec<(ClusterId, Vec<MorphonId>)> = self
                .clusters
                .iter()
                .map(|(&cid, c)| (cid, c.members.clone()))
                .collect();

            for (cid, members) in &cluster_data {
                let member_count = members.len();
                if member_count == 0 {
                    continue;
                }

                // 1. Cluster overhead cost (V3 — kept for backward compat)
                if cluster_overhead > 0.0 {
                    for &mid in members {
                        if let Some(m) = self.morphons.get_mut(&mid) {
                            m.energy -= cluster_overhead;
                        }
                    }
                }

                // 2. Each fused morphon draws from the shared pool
                if let Some(cluster) = self.clusters.get_mut(cid) {
                    for &mid in members {
                        let actual_draw =
                            draw_per_tick.min(cluster.shared_energy_pool / member_count as f64);
                        if actual_draw > 0.0 {
                            if let Some(m) = self.morphons.get_mut(&mid) {
                                m.energy = (m.energy + actual_draw).min(1.0);
                            }
                            cluster.shared_energy_pool -= actual_draw;
                        }
                    }
                }

                // 3. Members contribute utility reward back to pool
                for &mid in members {
                    if let Some(m) = self.morphons.get(&mid) {
                        let pe_delta = m.prev_potential.abs() - m.prediction_error;
                        if pe_delta > 0.0 {
                            if let Some(cluster) = self.clusters.get_mut(cid) {
                                cluster.shared_energy_pool += pe_delta * 0.005;
                            }
                        }
                    }
                }

                // 4. Cluster-level homeostasis: shared setpoint → member thresholds
                let cluster_mean_rate: f64 = members
                    .iter()
                    .filter_map(|mid| self.morphons.get(mid))
                    .map(|m| m.activity_history.mean())
                    .sum::<f64>()
                    / member_count as f64;

                if let Some(cluster) = self.clusters.get_mut(cid) {
                    cluster.shared_homeostatic_setpoint +=
                        0.01 * (cluster_mean_rate - cluster.shared_homeostatic_setpoint);
                    cluster.shared_homeostatic_setpoint =
                        cluster.shared_homeostatic_setpoint.clamp(0.05, 0.5);
                    cluster.shared_energy_pool =
                        cluster.shared_energy_pool.clamp(0.0, member_count as f64);

                    let setpoint = cluster.shared_homeostatic_setpoint;
                    for &mid in members {
                        if let Some(m) = self.morphons.get_mut(&mid) {
                            let actual_rate = m.activity_history.mean();
                            m.threshold += 0.01 * (actual_rate - setpoint);
                            let max_threshold = m.activation_fn.max_output();
                            m.threshold = m.threshold.clamp(0.05, max_threshold);
                        }
                    }
                }
            }
        }
        // V3: Enforce constitutional energy floor.
        let energy_floor = self.config.governance.energy_floor;
        if energy_floor > 0.0 {
            for m in self.morphons.values_mut() {
                crate::governance::enforce_energy_floor(m, energy_floor);
            }
        }

        // 5. Adaptive threshold boost (Diehl & Cook).
        //    Boost any fired Associative/Stem morphon (LocalInhibition determines who fired).
        {
            let boost = self.config.homeostasis.winner_boost
                * self.endo.channels.winner_adaptation_mult as f64;
            if boost > 0.0 {
                for m in self.morphons.values_mut() {
                    if m.fired
                        && (m.cell_type == CellType::Associative
                            || m.cell_type == CellType::Stem)
                    {
                        m.threshold += boost;
                    }
                }
            }
        }

        // 5. Spike-timing: boost pre_trace on delivered synapses.
        //    The spike delivery is direct evidence that pre fired recently.
        //    By incrementing pre_trace here, the trace-based STDP in the medium
        //    path will detect the causal relationship even if the pre-synaptic
        //    morphon is now in refractory (fired=false).
        //
        //    Hot path: uses the cached edge_idx captured at propagate-time
        //    instead of `synapse_between`, eliminating ~7K samples/run of
        //    HashMap+SipHash lookups (profiled with samply on standard mnist).
        //
        //    Safety: spike delays (min 0.5 ticks, typical < 10) are far shorter
        //    than the slow-path prune period (100 steps), so edge indices are
        //    stable for the lifetime of any in-flight spike. Debug builds assert
        //    this invariant. `None` on spikes from old snapshots — skip those.
        for spike in &delivered {
            if let Some(raw) = spike.edge_idx {
                let ei = petgraph::graph::EdgeIndex::new(raw as usize);
                #[cfg(debug_assertions)]
                if let Some((src, tgt)) = self.topology.edge_endpoint_ids(ei) {
                    debug_assert_eq!(src, spike.source,
                        "stale edge_idx {raw}: source mismatch (got {src:?}, want {:?})", spike.source);
                    debug_assert_eq!(tgt, spike.target,
                        "stale edge_idx {raw}: target mismatch (got {tgt:?}, want {:?})", spike.target);
                }
                // iSTDP pre-spike: check source cell type before mutating topology.
                let src_is_inh = self
                    .morphons
                    .get(&spike.source)
                    .map(|m| m.cell_type == CellType::InhibitoryInterneuron)
                    .unwrap_or(false);
                if let Some(synapse) = self.topology.synapse_mut(ei) {
                    synapse.pre_trace += 1.0;
                    // iSTDP pre (Vogels 2011): Δw = -η × post_trace.
                    // Strengthens inhibition when inhibitory pre fires and post was recently active.
                    if src_is_inh {
                        let (eta, _) = istdp_params;
                        crate::learning::istdp_pre(synapse, eta);
                    }
                }
            }
        }
        let morphon_ids: Vec<MorphonId> = self.morphons.keys().copied().collect();

        // === DIRECT FEEDBACK ALIGNMENT ===
        // Project output error through fixed random weights to Associative morphons.
        // This (a) drives Associative firing (breaks the 0% deadlock) and
        // (b) provides neuron-specific credit assignment (replaces global reward).
        // (Lillicrap et al. 2016, GLSNN: global feedback + local STDP)
        {
            let td_err = self.last_td_error;
            // Compute error vector: TD error × each motor morphon's centered potential
            let error_vec: Vec<(MorphonId, f64)> = self
                .output_ports
                .iter()
                .filter_map(|&mid| {
                    self.morphons.get(&mid).map(|m| {
                        let sig = 1.0 / (1.0 + (-m.potential).exp());
                        (mid, td_err * (sig - 0.5) * 2.0)
                    })
                })
                .collect();

            // Project error to each Associative morphon through fixed random weights
            let feedback_strength = 1.0;
            for (&assoc_id, weights) in &self.feedback_weights {
                let feedback: f64 = weights
                    .iter()
                    .filter_map(|(mid, bw)| {
                        error_vec
                            .iter()
                            .find(|(id, _)| id == mid)
                            .map(|(_, e)| bw * e)
                    })
                    .sum();

                if let Some(m) = self.morphons.get_mut(&assoc_id) {
                    // Inject as input current — drives firing (breaks deadlock)
                    m.input_accumulator += feedback_strength * feedback;
                    // Store as neuron-specific modulation for weight updates
                    m.feedback_signal = feedback;
                }
            }
        }

        // === MEDIUM SYNC: hot → structs ===
        // Must happen before Endoquilibrium and learning so morphon.potential/fired are current.
        if tick.medium {
            self.sync_hot_to_structs();
        }

        // === ENDOQUILIBRIUM (before learning) ===
        if self.endo.config.enabled && tick.medium {
            let vitals = crate::endoquilibrium::sense_vitals(
                &self.morphons,
                &self.topology,
                &self.diag,
                self.step_count,
                self.recent_performance,
                &self.clusters,
            );
            self.endo.tick(vitals);
            // Clear episode-gated capture count now that endo has consumed it.
            self.diag.episode_captures_pending = 0;
        }

        // === ASTROCYTIC GATE UPDATE (medium path) ===
        // Per-morphon slow state variable (AGMP-inspired) that gates excitatory plasticity.
        // Updated before learning so the gate is fresh when weight updates happen.
        if tick.medium {
            let tau_a = self.config.homeostasis.tau_astrocytic;
            let eta_v = self.config.homeostasis.eta_v;
            let eta_s = self.config.homeostasis.eta_s;
            let eta_f = self.config.homeostasis.eta_f;
            if tau_a > 0.0 {
                let decay = 1.0 - 1.0 / tau_a;
                let inv_tau = 1.0 / tau_a;
                for m in self.morphons.values_mut() {
                    let drive = eta_v * m.potential
                        + eta_s * m.input_accumulator
                        + eta_f * if m.fired { 1.0 } else { 0.0 };
                    m.astrocytic_state = decay * m.astrocytic_state + inv_tau * drive;
                }
            }
        }

        // === iSTDP: per-spike spike-timing updates (fast path above) ===
        // Inhibitory plasticity now runs per-delivered-spike via istdp_pre/istdp_post
        // (Vogels 2011 spike-timing rule) instead of per-medium-tick rate-based updates.
        // This gives faster hub-breaking convergence and true spike-timing causality.

        // === MEDIUM PATH ===
        // Two learning regimes (Frémaux et al. 2013):
        // - Critic morphons: TD-LTP — Δw = lr × δ × pre_trace (direct value learning)
        // - Actor morphons: Three-factor STDP — Δw = eligibility × M(t) × plasticity
        // This breaks the circular dependency where critic and actor share learning rules.
        let captures_this_step = 0_u64;
        if tick.medium {
            // Apply Endo's tau_eligibility_mult (reactive, capture-speed) and
            // slow_trace_leak (stage-dependent baseline) to learning params.
            // slow_trace_leak > 1 = faster decay = divide tau. Both compose multiplicatively.
            let mut learning_params = self.config.learning.clone();
            let trace_leak = self.endo.channels.slow_trace_leak.max(0.1) as f64;
            learning_params.tau_eligibility *=
                self.endo.channels.tau_eligibility_mult as f64 / trace_leak;

            let critic_set: std::collections::HashSet<MorphonId> =
                self.critic_ports.iter().copied().collect();
            let td_err = self.last_td_error;
            let wmax = self.config.learning.weight_max;

            // === Sparse actor visit set ===
            // The actor branch below only does work when at least one of:
            //   (a) post is active (motors are always borderline-active; non-motors must fire)
            //   (b) feedback_sig != 0 (DFA path)
            //   (c) at least one incoming pre fired this tick (LTD via pre-spike)
            // Lazy decay in update_eligibility means skipped synapses catch up correctly
            // when next visited. Only L2 weight decay is approximated (skipped morphons
            // don't get the per-tick 0.05% nudge — bounded effect, weights are clamped).
            let actor_visit_set: std::collections::HashSet<MorphonId> = {
                let mut set = std::collections::HashSet::new();
                for (id, m) in &self.morphons {
                    if critic_set.contains(id) { continue; }
                    if m.cell_type == CellType::Motor
                        || m.fired
                        || m.feedback_signal.abs() > 0.001
                    {
                        set.insert(*id);
                    }
                }
                // Add post-targets of every firing morphon (covers LTD-via-pre-spike).
                let firing_ids: Vec<MorphonId> = self.morphons.iter()
                    .filter(|(_, m)| m.fired)
                    .map(|(id, _)| *id)
                    .collect();
                for pre_id in &firing_ids {
                    // Closure-based iteration: zero allocation per source.
                    self.topology.for_each_outgoing(*pre_id, |target_id, _| {
                        if !critic_set.contains(&target_id) {
                            set.insert(target_id);
                        }
                    });
                }
                set
            };

            // Hoisted incoming-edge buffer reused across every critic + actor
            // morphon. Eliminates the per-call Vec::from_iter allocation that
            // showed up as ~5–10k samples in the post-edge-cache profile.
            let mut incoming_buf: Vec<(MorphonId, petgraph::graph::EdgeIndex)> = Vec::new();

            for &id in &morphon_ids {
                if critic_set.contains(&id) {
                    // === CRITIC: TD-LTP ===
                    // The critic learns to predict V(s) by minimizing TD error.
                    // Weight update depends only on TD error × pre-synaptic trace,
                    // independent of the actor's eligibility computation.
                    self.topology.collect_incoming_edges_into(id, &mut incoming_buf);
                    for &(_, edge_idx) in &incoming_buf {
                        if let Some(synapse) = self.topology.synapse_mut(edge_idx) {
                            // Decay traces (same as actor)
                            let trace_decay = (-dt / self.config.learning.tau_trace).exp();
                            synapse.pre_trace *= trace_decay;
                            synapse.post_trace *= trace_decay;

                            // TD-LTP: Δw = lr × δ × pre_trace
                            // Only LTP direction — critic converges to value estimate
                            let td_lr = 0.03;
                            let delta_w = td_lr * td_err * synapse.pre_trace;
                            synapse.weight = (synapse.weight + delta_w).clamp(-wmax, wmax);
                            synapse.age += 1;
                        }
                    }
                } else if actor_visit_set.contains(&id) {
                    // === ACTOR: Three-factor learning (sparse — visit set only) ===
                    // Motor morphons: standard receptor-gated modulation (TD via Reward channel)
                    // Associative morphons: DFA feedback signal (neuron-specific credit)
                    let (
                        post_activity,
                        post_receptors,
                        post_sensitivity,
                        feedback_sig,
                        is_assoc,
                        plast_rate,
                        astro_state,
                        post_ct,
                        post_energy,
                    ) = self
                        .morphons
                        .get(&id)
                        .map(|m| {
                            let activity = if m.cell_type == CellType::Motor {
                                let sig = 1.0 / (1.0 + (-m.potential).exp());
                                (sig - 0.5) * 2.0
                            } else {
                                if m.fired {
                                    1.0
                                } else {
                                    0.0
                                }
                            };
                            let is_a = m.cell_type == CellType::Associative
                                || m.cell_type == CellType::Stem;
                            (
                                activity,
                                m.receptors.clone(),
                                m.receptor_sensitivity.clone(),
                                m.feedback_signal,
                                is_a,
                                m.plasticity_rate,
                                m.astrocytic_state,
                                m.cell_type,
                                m.energy,
                            )
                        })
                        .unwrap_or((
                            0.0,
                            Default::default(),
                            Default::default(),
                            0.0,
                            false,
                            1.0,
                            0.5,
                            CellType::Stem,
                            1.0,
                        ));
                    // Astrocytic gate: sigmoid(a - threshold). Bypassed for Modulatory/InhibitoryInterneuron.
                    let astro_gate = if post_ct == CellType::Modulatory
                        || post_ct == CellType::InhibitoryInterneuron
                    {
                        1.0
                    } else {
                        let th = self.config.homeostasis.threshold_astrocytic;
                        1.0 / (1.0 + (-(astro_state - th)).exp())
                    };
                    self.topology.collect_incoming_edges_into(id, &mut incoming_buf);

                    for &(pre_id, edge_idx) in &incoming_buf {
                        let pre_fired = self.morphons.get(&pre_id).map_or(false, |m| m.fired);

                        if let Some(synapse) = self.topology.synapse_mut(edge_idx) {
                            learning::update_eligibility(
                                synapse,
                                pre_fired,
                                post_activity,
                                &learning_params,
                                dt,
                                self.step_count,
                            );

                            // V6: Graceful Degradation — survival mode skips all weight updates.
                            // Eligibility traces still decay (update_eligibility ran above).
                            if post_energy > ENERGY_SURVIVAL_THRESHOLD {
                                if is_assoc && feedback_sig.abs() > 0.001 {
                                    // DFA climbing-fiber rule: Δw = pre_trace × feedback_signal × lr
                                    // Uses pre_trace (decaying memory of recent pre-synaptic firing)
                                    // NOT eligibility (which is STDP-gated and attenuates the signal).
                                    // NOT binary pre_fired (too sparse at 5% firing rate).
                                    // pre_trace persists ~10 steps after firing — "is this synapse
                                    // carrying signal?" — the right gate for targeted DFA updates.
                                    // Biologically: climbing fibers override STDP timing requirements.
                                    let dfa_lr = 0.02
                                        * plast_rate
                                        * self.endo.channels.plasticity_mult as f64
                                        * astro_gate; // scaled by Anchor/Sail + Endo + astrocytic gate
                                    let consolidation_scale = 1.0 - synapse.consolidation_level * 0.9;
                                    let weight_decay = 0.001 * synapse.weight;
                                    let delta_w = (synapse.pre_trace * feedback_sig * dfa_lr
                                        - weight_decay)
                                        * consolidation_scale;
                                    synapse.weight = (synapse.weight + delta_w).clamp(-wmax, wmax);

                                    // V3: Record reinforcement event
                                    if delta_w.abs() > 0.001 {
                                        if let Some(ref mut j) = synapse.justification {
                                            j.record_reinforcement(
                                                self.step_count,
                                                delta_w,
                                                self.modulation.reward,
                                            );
                                            // Phase 2d: link synapse to active ANCS memory item
                                            if self.current_ancs_item.is_some() {
                                                j.memory_item_ref = self.current_ancs_item;
                                            }
                                        }
                                    }

                                    // Tag synapses where DFA update was strong.
                                    // Accumulate tag (not set to max) — capture is deferred to
                                    // episode end where performance-relative decisions are made.
                                    let dfa_strength = (synapse.pre_trace * feedback_sig).abs();
                                    if dfa_strength > 0.1 && synapse.consolidation_level < 1.0 {
                                        synapse.tag = (synapse.tag
                                            + dfa_strength
                                                * self.config.learning.tag_accumulation_rate)
                                            .min(1.0);
                                        synapse.tag_strength = synapse.tag_strength.max(dfa_strength);
                                    }
                                    synapse.age += 1;
                                    if synapse.eligibility.abs() > 0.1 {
                                        synapse.usage_count += 1;
                                    }
                                } else {
                                    // Standard three-factor for Motor, Sensory, Modulatory
                                    // Scaled by per-morphon plasticity_rate (Anchor/Sail) × Endo plasticity_mult
                                    let plasticity = self.modulation.plasticity_rate()
                                        * plast_rate
                                        * self.endo.channels.plasticity_mult as f64
                                        * astro_gate;
                                    let endo_gains = [
                                        self.endo.channels.reward_gain as f64,
                                        self.endo.channels.novelty_gain as f64,
                                        self.endo.channels.arousal_gain as f64,
                                        self.endo.channels.homeostasis_gain as f64,
                                    ];
                                    // V6: Degraded mode — zero out reward/novelty/arousal, keep
                                    // only homeostasis channel for structural stability.
                                    let degraded_mod;
                                    let modulation_ref = if post_energy <= ENERGY_DEGRADED_THRESHOLD {
                                        let mut m = self.modulation.clone();
                                        m.reward = 0.0;
                                        m.novelty = 0.0;
                                        m.arousal = 0.0;
                                        degraded_mod = m;
                                        &degraded_mod
                                    } else {
                                        &self.modulation
                                    };
                                    let weight_before = synapse.weight;
                                    let captured = learning::apply_weight_update(
                                        synapse,
                                        modulation_ref,
                                        &self.config.learning,
                                        plasticity,
                                        &post_receptors,
                                        endo_gains,
                                        &post_sensitivity,
                                    );
                                    // Per-tick capture is disabled — capture is now episode-gated
                                    // via report_episode_end(). Tags still accumulate per-tick.
                                    let _ = captured;
                                    // V3: Record reinforcement event
                                    let delta_w_3f = synapse.weight - weight_before;
                                    if delta_w_3f.abs() > 0.001 {
                                        if let Some(ref mut j) = synapse.justification {
                                            j.record_reinforcement(
                                                self.step_count,
                                                delta_w_3f,
                                                self.modulation.reward,
                                            );
                                            // Phase 2d: link synapse to active ANCS memory item
                                            if self.current_ancs_item.is_some() {
                                                j.memory_item_ref = self.current_ancs_item;
                                            }
                                        }
                                    }
                                    // L2 weight decay on all three-factor synapses
                                    synapse.weight -= 0.0005 * synapse.weight;
                                }
                            }
                        }
                    }
                }
            }
        }

        // === COMBINED: Weight normalization + compensatory plasticity + frustration ===
        // Fused into a single loop to walk incoming edges once per morphon instead
        // of 3 separate passes with 3 separate incoming_synapses_mut() allocations.
        if tick.medium {
            let tip_rate = self.config.learning.transmitter_potentiation;
            let hsd_rate = self.config.learning.heterosynaptic_depression;
            let wmax = self.config.learning.weight_max;
            let frustration_enabled = self.config.morphogenesis.frustration.enabled;
            let perturb_scale = self
                .config
                .morphogenesis
                .frustration
                .weight_perturbation_scale;

            // Snapshot morphon states to avoid borrow conflicts with topology
            let morphon_states: Vec<(MorphonId, bool, f64, f64, bool, f64, bool)> = self
                .morphons
                .values()
                .map(|m| {
                    (
                        m.id,
                        m.fired,
                        m.activity_history.mean(),
                        m.plasticity_rate,
                        m.frustration.exploration_mode,
                        m.frustration.frustration_level,
                        m.cell_type == CellType::Associative || m.cell_type == CellType::Stem,
                    )
                })
                .collect();

            // Hoisted incoming-edge buffer for the combined-pass loop.
            let mut edge_buf: Vec<(MorphonId, petgraph::graph::EdgeIndex)> = Vec::new();

            for &(id, fired, post_rate, plasticity_rate, in_exploration, frust_level, is_assoc) in
                &morphon_states
            {
                self.topology.collect_incoming_edges_into(id, &mut edge_buf);
                if edge_buf.is_empty() {
                    continue;
                }

                // --- Pass 1 (assoc only): compute L1 norm for weight normalization ---
                let (norm_scale, do_normalize) = if is_assoc {
                    let mut pos_sum = 0.0_f64;
                    for &(_, ei) in &edge_buf {
                        if let Some(syn) = self.topology.graph.edge_weight(ei) {
                            if syn.weight > 0.0 {
                                pos_sum += syn.weight;
                            }
                        }
                    }
                    let target_norm = edge_buf.len() as f64 * 0.3;
                    if pos_sum > 0.01 {
                        let tol = 0.1 + 0.15 * plasticity_rate.min(2.0);
                        ((target_norm / pos_sum).clamp(1.0 - tol, 1.0 + tol), true)
                    } else {
                        (1.0, false)
                    }
                } else {
                    (1.0, false)
                };

                // --- Pass 2: apply normalization + compensatory plasticity + frustration ---
                for &(pre_id, edge_idx) in &edge_buf {
                    let pre_fired = self.morphons.get(&pre_id).map_or(false, |m| m.fired);

                    if let Some(synapse) = self.topology.synapse_mut(edge_idx) {
                        // (a) Weight normalization (assoc/stem only, positive weights)
                        if do_normalize && synapse.weight > 0.0 {
                            synapse.weight *= norm_scale;
                            synapse.weight = synapse.weight.clamp(0.0, wmax);
                        }

                        // (b) Transmitter-induced potentiation
                        if pre_fired && post_rate < 0.05 {
                            synapse.weight += tip_rate;
                            synapse.weight = synapse.weight.min(wmax);
                        }

                        // (c) Heterosynaptic depression
                        if fired {
                            synapse.weight *= 1.0 - hsd_rate;
                        }

                        // (d) Frustration-driven weight perturbation
                        if frustration_enabled && in_exploration && !synapse.consolidated {
                            let hash = (id.wrapping_mul(synapse.age).wrapping_add(31337) % 10000)
                                as f64
                                / 5000.0
                                - 1.0;
                            let delta = hash * perturb_scale * frust_level * wmax;
                            synapse.weight += delta;
                        }

                        // Blanket weight sanitizer
                        if !synapse.weight.is_finite() {
                            synapse.weight = 0.0;
                        }
                        synapse.weight = synapse.weight.clamp(-wmax, wmax);
                    }
                }
            }
        }

        // === V2: RECORD MODULATION + PE DELTAS FOR RECEPTOR ADAPTATION ===
        if tick.medium {
            let reward_signal = self.modulation.reward_delta().abs();
            let novelty_signal = self.modulation.novelty;
            let arousal_signal = self.modulation.arousal;
            let homeostasis_signal = self.modulation.homeostasis;
            for m in self.morphons.values_mut() {
                // Record PE delta
                let pe_delta = m.prediction_error - m.frustration.prev_pe;
                m.recent_pe_deltas.push(pe_delta);
                // Record per-channel modulation signals
                m.recent_modulation
                    .entry(ModulatorType::Reward)
                    .or_insert_with(|| RingBuffer::new(10))
                    .push(reward_signal);
                m.recent_modulation
                    .entry(ModulatorType::Novelty)
                    .or_insert_with(|| RingBuffer::new(10))
                    .push(novelty_signal);
                m.recent_modulation
                    .entry(ModulatorType::Arousal)
                    .or_insert_with(|| RingBuffer::new(10))
                    .push(arousal_signal);
                m.recent_modulation
                    .entry(ModulatorType::Homeostasis)
                    .or_insert_with(|| RingBuffer::new(10))
                    .push(homeostasis_signal);
            }
        }

        // === MEDIUM PATH: metabolic + PE + frustration + homeostatic regulation ===
        // Called instead of morphon.step() for these slow-path operations.
        if tick.medium {
            let dt_f = dt;
            let metabolic = self.config.metabolic.clone();
            let mut frustration_cfg = self.config.morphogenesis.frustration.clone();
            frustration_cfg.stagnation_threshold *=
                self.endo.channels.frustration_sensitivity_mult as f64;
            let maint: Vec<(MorphonId, f64)> = self
                .cached_maintenance_costs
                .iter()
                .map(|(&id, &cost)| (id, cost))
                .collect();
            let maint_map: std::collections::HashMap<MorphonId, f64> =
                maint.into_iter().collect();
            for m in self.morphons.values_mut() {
                let cost = maint_map.get(&m.id).copied().unwrap_or(0.0);
                m.medium_update(dt_f, cost, &metabolic, &frustration_cfg);
            }
            // Sync updated thresholds back to hot arrays
            self.sync_structs_to_hot();
        }

        // === SLOW PATH ===
        if tick.slow {
            // Apply Endo Phase B levers to slow-path parameters.
            let mut slow_learning = self.config.learning.clone();
            slow_learning.weight_min *= self.endo.channels.pruning_threshold_mult as f64;

            let mut slow_params = self.config.morphogenesis.clone();
            slow_params.migration_rate *= self.endo.channels.migration_rate_mult as f64;
            slow_params.synaptogenesis_threshold *=
                self.endo.channels.synaptogenesis_threshold_mult as f64;

            let slow_report = morphogenesis::step_slow(
                &mut self.morphons,
                &mut self.topology,
                &slow_params,
                &slow_learning,
                self.modulation.homeostasis,
                &self.config.lifecycle,
                &mut self.rng,
                self.field.as_ref(),
                self.config.governance.max_connectivity_per_morphon,
                self.step_count,
                &self.config.governance.mandatory_justification_for,
            );
            report.synapses_created = slow_report.synapses_created;
            report.synapses_pruned = slow_report.synapses_pruned;
            report.migrations = slow_report.migrations;
            self.diag.pruning_events_recent = slow_report.synapses_pruned as u32;
            self.diag.synapses_saved_fwd_recent = slow_report.synapses_saved_fwd as u32;

            // V6: Contradiction-Driven Reconsolidation — un-consolidate synapses
            // whose post-synaptic morphons show chronic high prediction error.
            let _reconsolidated = learning::reconsolidate(
                &mut self.topology,
                &self.morphons,
                &self.config.learning,
            );

            // V6: Auto-Merge — check for co-activating proximate morphon groups.
            // Confirmed groups (second sighting within window) become fusion candidates.
            let fired_ids: Vec<MorphonId> = self
                .morphons
                .values()
                .filter(|m| m.fired)
                .map(|m| m.id)
                .collect();
            let merge_groups = morphogenesis::check_auto_merge(
                &fired_ids,
                &self.morphons,
                &mut self.auto_merge_candidates,
                &self.config.morphogenesis,
                self.step_count,
            );
            // Confirmed groups are queued for the next glacial fusion pass.
            self.pending_merge_groups.extend(merge_groups);

            // V2: Update bioelectric field — write morphon states, diffuse
            if let Some(ref mut field) = self.field {
                field.write_from_morphons(&self.morphons);
                field.diffuse();
                // Update diagnostics with field metrics
                if let Some(pe_layer) = field.layers.get(&crate::field::FieldType::PredictionError)
                {
                    self.diag.field_pe_max = pe_layer.max();
                    self.diag.field_pe_mean = pe_layer.mean();
                }
            }

            // V2: Adapt receptor sensitivities — meta-learning on slow tick.
            // Rate is very conservative (0.001) to avoid destabilizing a tuned pipeline.
            // Adaptation accumulates over hundreds of slow ticks → meaningful over
            // thousands of episodes, not within a single training run.
            for m in self.morphons.values_mut() {
                learning::adapt_receptor_sensitivity(
                    &mut m.receptor_sensitivity,
                    &m.recent_modulation,
                    &m.recent_pe_deltas,
                    0.001,
                );
            }

            // Activity-dependent myelination with neuromodulatory gating.
            // Treadmilling dynamics (Auer 2019): wrapping vs removal equilibrium.
            // Arousal gates OPC sensitivity, reward biases toward tagged pathways,
            // energy pressure suppresses wrapping under metabolic stress.
            // Energy pressure from Associative morphons only — Sensory/Motor energy
            // isn't meaningful for myelination stress (they don't fire competitively).
            let (assoc_energy_sum, assoc_count) = self
                .morphons
                .values()
                .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
                .fold((0.0_f64, 0usize), |(sum, n), m| (sum + m.energy, n + 1));
            let avg_energy = if assoc_count > 0 {
                assoc_energy_sum / assoc_count as f64
            } else {
                1.0
            };
            let myelin_ctx = crate::morphon::MyelinationContext {
                arousal: self.modulation.arousal,
                reward: self.modulation.reward,
                energy_pressure: (1.0 - avg_energy).clamp(0.0, 1.0),
            };
            self.topology.update_all_synapses(|synapse| {
                synapse.update_myelination(1.0, &myelin_ctx);
            });

            // Rebuild hot arrays after structural changes (synaptogenesis, pruning, migration)
            self.rebuild_hot_arrays();
        }

        // === GLACIAL PATH (with checkpoint/rollback protection) ===
        if tick.glacial {
            // Checkpoint before structural changes
            let all_ids: Vec<MorphonId> = self.morphons.keys().copied().collect();
            let count_before = self.morphons.len();
            let checkpoint =
                homeostasis::create_checkpoint(&all_ids, &self.morphons, &self.topology);

            // Apply Endo Phase B lever: division_threshold_mult scales division_threshold.
            let mut glacial_params = self.config.morphogenesis.clone();
            glacial_params.division_threshold *= self.endo.channels.division_threshold_mult as f64;

            let priority_groups = std::mem::take(&mut self.pending_merge_groups);
            let glacial_report = morphogenesis::step_glacial(
                &mut self.morphons,
                &mut self.topology,
                &mut self.clusters,
                &mut self.next_morphon_id,
                &mut self.next_cluster_id,
                &glacial_params,
                self.effective_max_morphons,
                self.config.governance.max_cluster_size_fraction,
                self.config.governance.max_unverified_fraction,
                self.modulation.arousal,
                &self.config.lifecycle,
                &mut self.rng,
                self.config.target_morphology.as_ref(),
                self.step_count,
                &self.config.homeostasis.competition_mode,
                &priority_groups,
            );
            report.morphons_born = glacial_report.morphons_born;
            report.morphons_died = glacial_report.morphons_died;
            self.diag.division_events_recent = glacial_report.morphons_born as u32;
            self.diag.pruning_events_recent = glacial_report.morphons_died as u32;

            // Wire newly born Associative morphons to nearby InhibitoryInterneurons
            if report.morphons_born > 0 {
                let new_ids: Vec<MorphonId> = self
                    .morphons
                    .keys()
                    .filter(|&&id| {
                        id >= self
                            .next_morphon_id
                            .saturating_sub(report.morphons_born as u64 + 10)
                    })
                    .copied()
                    .collect();
                morphogenesis::wire_to_nearby_interneurons(
                    &self.morphons,
                    &mut self.topology,
                    &self.config.homeostasis.competition_mode,
                    &new_ids,
                );
            }

            // Track ALL births/deaths including fusion's inhibitory morphons
            let count_after = self.morphons.len();
            if count_after > count_before {
                self.total_born += count_after - count_before;
            } else if count_before > count_after {
                self.total_died += count_before - count_after;
            }
            report.differentiations = glacial_report.differentiations;
            report.transdifferentiations = glacial_report.transdifferentiations;
            self.total_transdifferentiations += glacial_report.transdifferentiations;
            report.fusions = glacial_report.fusions;
            report.defusions = glacial_report.defusions;

            // V2: Target Morphology — write Identity field + self-healing
            if let Some(ref target) = self.config.target_morphology {
                // Write Identity field layer (if field is enabled and has Identity layer)
                if let Some(ref mut field) = self.field {
                    // Pre-compute projections before borrowing layers mutably
                    let projections: Vec<(usize, usize, i32, f64)> = target
                        .regions
                        .iter()
                        .map(|region| {
                            let (cx, cy) = field.project(&region.center);
                            let grid_radius = (region.radius * field.config.resolution as f64 / 2.0)
                                .ceil() as i32;
                            (cx, cy, grid_radius, region.identity_strength)
                        })
                        .collect();
                    let res = field.config.resolution as i32;

                    if let Some(identity_layer) =
                        field.layers.get_mut(&crate::field::FieldType::Identity)
                    {
                        identity_layer.data.fill(0.0);
                        for (cx, cy, grid_radius, strength) in &projections {
                            for dy in -grid_radius..=*grid_radius {
                                for dx in -grid_radius..=*grid_radius {
                                    let gx = (*cx as i32 + dx).clamp(0, res - 1) as usize;
                                    let gy = (*cy as i32 + dy).clamp(0, res - 1) as usize;
                                    identity_layer.write(gx, gy, *strength);
                                }
                            }
                        }
                    }
                }

                // Self-healing: recruit morphons to underpopulated regions
                if target.self_healing {
                    developmental::target_morphology_heal(
                        target,
                        &mut self.morphons,
                        &mut self.topology,
                        &mut self.next_morphon_id,
                        self.effective_max_morphons,
                    );
                }
            }

            // V3: Update epistemic states for all clusters.
            crate::epistemic::update_all_clusters(
                &mut self.clusters,
                &self.morphons,
                &self.topology,
                self.step_count,
            );
            // V3: Apply epistemic effects (plasticity adjustments, unconsolidation).
            crate::epistemic::apply_epistemic_effects(
                &self.clusters,
                &mut self.morphons,
                &mut self.topology,
                self.step_count,
                self.config.metabolic.reward_for_verification,
            );

            // Rollback synapses if prediction error spiked after structural changes
            let surviving_ids: Vec<MorphonId> = all_ids
                .iter()
                .filter(|id| self.morphons.contains_key(id))
                .copied()
                .collect();
            // Apply Endo's rollback sensitivity multiplier
            let mut effective_homeo = self.config.homeostasis.clone();
            effective_homeo.rollback_pe_threshold *=
                self.endo.channels.rollback_pe_threshold_mult as f64;
            if homeostasis::should_rollback(
                &checkpoint,
                &surviving_ids,
                &self.morphons,
                &effective_homeo,
            ) {
                homeostasis::rollback_synapses(&checkpoint, &mut self.topology);
                self.diag.rollback_triggered = true;
                self.diag.total_rollbacks += 1;
            }

            // Rebuild hot arrays after structural changes (division, death, fusion, apoptosis)
            self.rebuild_hot_arrays();
        }

        // === V2: DREAMING ENGINE ===
        // Runs on glacial tick when system is mature/consolidating or low activity
        if tick.glacial && self.config.dream.enabled {
            let is_mature = matches!(
                self.endo.stage(),
                crate::endoquilibrium::DevelopmentalStage::Mature
                    | crate::endoquilibrium::DevelopmentalStage::Consolidating
            );
            let total_firing: usize = self.diag.firing_by_type.values().map(|(f, _)| *f).sum();
            let total_morphons: usize = self.diag.firing_by_type.values().map(|(_, t)| *t).sum();
            let is_low_activity =
                total_morphons == 0 || (total_firing as f64 / total_morphons as f64) < 0.05;

            if is_mature || is_low_activity {
                self.dream_cycle();
            }
        }

        // === HOMEOSTASIS ===
        if tick.homeostasis {
            homeostasis::synaptic_scaling(
                &self.morphons,
                &mut self.topology,
                self.config.learning.weight_max,
            );
            // Apply anti-hub scaling only when selective features have formed (Mature/Consolidating stages)
            // Prevents depression of useful distributed activity in early learning phases
            if self.endo.should_apply_anti_hub_scaling() {
                homeostasis::anti_hub_scaling(&self.morphons, &mut self.topology);
            }
            homeostasis::inter_cluster_inhibition(
                &mut self.morphons,
                &self.clusters,
                &self.config.homeostasis,
            );
        }

        // === Always: decay neuromodulation ===
        self.modulation.decay();

        // === MEMORY (on schedule) ===
        if tick.memory {
            self.memory.procedural.record(
                self.step_count,
                self.morphons.len(),
                self.topology.synapse_count(),
                self.clusters.len(),
            );

            // Episodic replay: reactivate high-value episodes to strengthen connections
            // This is analogous to hippocampal replay during "rest" periods
            let replayed = self.memory.episodic.replay(3);
            for episode in replayed {
                // Boost novelty slightly during replay to increase plasticity
                self.modulation.inject_novelty(0.1);
                // Re-inject the reward context of the replayed episode
                if episode.reward > 0.1 {
                    self.modulation.inject_reward(episode.reward * 0.3);
                }
            }

            // === ANCS-Core step (every memory tick = every 100 steps) ===
            // Compute energy usage: fraction of morphon energy budget depleted.
            let total_energy: f64 = self.morphons.values().map(|m| m.energy).sum();
            let max_energy = self.morphons.len() as f64;
            let energy_usage = if max_energy > 0.0 {
                1.0 - (total_energy / max_energy).clamp(0.0, 1.0)
            } else {
                0.0
            };

            // Run ANCS step: recalculate importance, epistemic transitions, eviction.
            self.ancs.step(dt, self.step_count, energy_usage);

            // TruthKeeper reconsolidation: drain patterns that transitioned to Contested
            // and re-open their consolidated synapses for re-learning.
            let contested_patterns = self.ancs.take_reconsolidate_patterns();
            for pattern in &contested_patterns {
                learning::reconsolidate_pattern(
                    &mut self.topology,
                    pattern,
                    &self.config.learning,
                );
            }

            // Update SystemHeartbeat with current snapshot.
            let contested_count = self
                .ancs
                .items_with_state(crate::ancs::MemoryEpistemicState::Contested)
                .len();
            self.heartbeat.update(
                self.modulation.arousal,
                self.modulation.novelty,
                self.modulation.reward,
                self.modulation.homeostasis,
                1.0, // plasticity_mult — future Endoquilibrium integration point
                energy_usage,
                self.ancs.len(),
                contested_count,
            );
        }
        self.memory.working.step(dt);

        // Auto-detect novelty: high average prediction error
        let avg_pe = self.avg_prediction_error();
        if avg_pe > 0.5 {
            self.modulation.inject_novelty(avg_pe * 0.3);
        }

        // Encode episodes when novelty is high
        if self.modulation.novelty > 0.3 {
            let pattern: Vec<(MorphonId, f64)> = self
                .morphons
                .values()
                .filter(|m| m.fired)
                .map(|m| (m.id, m.potential))
                .collect();

            if !pattern.is_empty() {
                // Legacy episodic memory (TripleMemory)
                self.memory.episodic.encode(
                    pattern.clone(),
                    self.modulation.reward,
                    self.modulation.novelty,
                    self.step_count,
                );

                // ANCS-Core: VBC-lite tier classification + store
                let ids: Vec<MorphonId> = pattern.iter().map(|(id, _)| *id).collect();
                let working_overlap = self.memory
                    .working
                    .active_patterns()
                    .iter()
                    .map(|wp| {
                        let intersection =
                            ids.iter().filter(|id| wp.pattern.contains(id)).count();
                        let union = ids.len() + wp.pattern.len() - intersection;
                        if union > 0 {
                            intersection as f64 / union as f64
                        } else {
                            0.0
                        }
                    })
                    .fold(0.0f64, f64::max);
                let tier = classify_tier(
                    self.modulation.novelty,
                    self.modulation.reward,
                    working_overlap,
                );
                let ancs_item = MemoryItem::new(
                    tier,
                    pattern,
                    self.modulation.reward,
                    self.modulation.novelty,
                    self.step_count,
                );
                self.ancs.store(ancs_item);
                // Track the ID for justification linking (Phase 2d)
                self.current_ancs_item = self.ancs.last_stored_id();
                // Populate working memory with the current episodic activation pattern.
                // activation = mean absolute membrane potential across pattern morphons,
                // clamped to [0,1] so it serves as a salience weight.
                let wm_activation = {
                    let n = ids.len().max(1);
                    let sum: f64 = ids.iter()
                        .filter_map(|id| self.morphons.get(id))
                        .map(|m| m.potential.abs())
                        .sum();
                    (sum / n as f64).clamp(0.0, 1.0)
                };
                self.memory.working.store(ids, wm_activation);
            }
        }

        // === EPISODE FIRE COUNTS (for winner diversity entropy) ===
        for m in self.morphons.values() {
            if m.fired {
                *self.episode_fire_counts.entry(m.id).or_insert(0) += 1;
            }
        }

        // === DIAGNOSTICS ===
        // Lightweight per-step counters (O(1)):
        self.diag.spikes_delivered_this_step = spikes_delivered;
        self.diag.spikes_pending = self.resonance.pending_count();
        self.diag.captures_this_step = captures_this_step;
        self.diag.total_captures += captures_this_step;

        // Expensive full snapshot only on homeostasis ticks — avoids O(synapses+morphons)
        // traversal every step. Consumers (dream engine, inspect(), external logging)
        // tolerate slightly stale stats.
        if tick.homeostasis || tick.glacial {
            let prev_total_captures = self.diag.total_captures;
            let prev_total_rollbacks = self.diag.total_rollbacks;
            let rollback_triggered = self.diag.rollback_triggered;
            let prev_field_pe_max = self.diag.field_pe_max;
            let prev_field_pe_mean = self.diag.field_pe_mean;
            let prev_episode_captures = self.diag.episode_captures_pending;
            self.diag = Diagnostics::snapshot(&self.morphons, &self.topology);
            self.diag.spikes_delivered_this_step = spikes_delivered;
            self.diag.spikes_pending = self.resonance.pending_count();
            self.diag.captures_this_step = captures_this_step;
            self.diag.total_captures = prev_total_captures;
            self.diag.total_rollbacks = prev_total_rollbacks;
            self.diag.rollback_triggered = rollback_triggered;
            self.diag.field_pe_max = prev_field_pe_max;
            self.diag.field_pe_mean = prev_field_pe_mean;
            self.diag.episode_captures_pending = prev_episode_captures;

            // V3: Populate epistemic diagnostic counts from cluster state
            {
                let (mut supported, mut hypothesis, mut outdated, mut contested) = (0, 0, 0, 0);
                let mut skepticism_sum = 0.0;
                for c in self.clusters.values() {
                    match &c.epistemic_state {
                        crate::epistemic::EpistemicState::Supported { .. } => supported += 1,
                        crate::epistemic::EpistemicState::Hypothesis { .. } => hypothesis += 1,
                        crate::epistemic::EpistemicState::Outdated { .. } => outdated += 1,
                        crate::epistemic::EpistemicState::Contested { .. } => contested += 1,
                    }
                    skepticism_sum += c.epistemic_history.skepticism;
                }
                self.diag.epistemic_supported = supported;
                self.diag.epistemic_hypothesis = hypothesis;
                self.diag.epistemic_outdated = outdated;
                self.diag.epistemic_contested = contested;
                self.diag.avg_skepticism = if self.clusters.is_empty() {
                    0.0
                } else {
                    skepticism_sum / self.clusters.len() as f64
                };
            }
        }

        // Curvature learning: regions with high prediction error get stronger curvature
        // (more space for fine-grained distinctions). Runs on slow schedule.
        if tick.slow {
            let curvature_update = |morphon: &mut Morphon| {
                let target_curvature = 1.0 + morphon.desire * 2.0;
                let rate = 0.01;
                morphon.position.curvature +=
                    rate * (target_curvature - morphon.position.curvature);
                morphon.position.curvature = morphon.position.curvature.clamp(0.1, 5.0);
            };
            #[cfg(feature = "parallel")]
            self.morphons
                .par_iter_mut()
                .for_each(|(_, m)| curvature_update(m));
            #[cfg(not(feature = "parallel"))]
            self.morphons.values_mut().for_each(|m| curvature_update(m));
        }

        report
    }

    /// Feed external input to sensory Morphons via stable port mapping.
    ///
    /// Each input value is broadcast to a group of sensory Morphons (fan-out).
    /// This ensures all inputs reach the network even if there are more sensory
    /// Morphons than input dimensions.
    pub fn feed_input(&mut self, inputs: &[f64]) {
        if self.input_ports.is_empty() || inputs.is_empty() {
            return;
        }
        let ports_per_input = (self.input_ports.len() / inputs.len()).max(1);
        for (i, &value) in inputs.iter().enumerate() {
            let start = i * ports_per_input;
            let end = (start + ports_per_input).min(self.input_ports.len());
            for port_idx in start..end {
                let id = self.input_ports[port_idx];
                if let Some(m) = self.morphons.get_mut(&id) {
                    m.input_accumulator += value;
                }
            }
        }
    }

    /// Read output — uses analog readout if enabled, else motor potential.
    ///
    /// Analog readout: V_j = Σ_i w_ji × sigmoid(assoc_i.potential)
    /// This bypasses the spike propagation pipeline, giving exact gradients.
    /// Spike-based: returns raw motor morphon potential (original behavior).
    pub fn read_output(&self) -> Vec<f64> {
        if self.use_analog_readout && !self.readout_weights.is_empty() {
            self.read_output_analog()
        } else {
            self.output_ports
                .iter()
                .filter_map(|id| self.morphons.get(id))
                .map(|m| m.potential)
                .collect()
        }
    }

    /// Analog readout: weighted sum of associative layer potentials.
    /// V_j = Σ_i readout_weights[j][i] × sigmoid(P_i)
    fn read_output_analog(&self) -> Vec<f64> {
        self.readout_weights
            .iter()
            .enumerate()
            .map(|(j, weights)| {
                let bias = self.readout_bias.get(j).copied().unwrap_or(0.0);
                let v: f64 = weights
                    .iter()
                    .map(|(&id, &w)| {
                        let p = self
                            .morphons
                            .get(&id)
                            .map(|m| {
                                if m.potential.is_finite() {
                                    m.potential.clamp(-10.0, 10.0)
                                } else {
                                    0.0
                                }
                            })
                            .unwrap_or(0.0);
                        // Centered sigmoid: 0 at resting potential, ±0.5 range.
                        // Eliminates constant 0.5 offset that drowns discriminative signal.
                        let act = 1.0 / (1.0 + (-p).exp()) - 0.5;
                        w * act
                    })
                    .sum();
                let out = bias + v;
                if out.is_finite() {
                    out
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Enable analog readout and initialize readout weights.
    ///
    /// Reset all morphon voltages to resting potential between episodes.
    /// Clears refractory timers, input accumulators, and spike queue.
    /// Preserves weights, thresholds, energy, consolidation — only transient state is reset.
    /// Biologically analogous to inter-trial interval baseline recovery.
    pub fn reset_voltages(&mut self) {
        for m in self.morphons.values_mut() {
            m.potential = 0.0;
            m.prev_potential = 0.0;
            m.fired = false;
            m.input_accumulator = 0.0;
            m.refractory_timer = 0.0;
            m.feedback_signal = 0.0;
        }
        self.resonance.clear();
    }

    /// Zero out readout weights for morphons that don't match the filter.
    /// Used to restrict readout to specific cell types (e.g., sensory-only).
    pub fn filter_readout_weights(&mut self, keep: impl Fn(MorphonId) -> bool) {
        for weights in &mut self.readout_weights {
            for (id, w) in weights.iter_mut() {
                if !keep(*id) {
                    *w = 0.0;
                }
            }
        }
    }

    /// Set the performance threshold for enabling consolidation.
    /// Below this level, the system stays fully plastic (no tag captures).
    pub fn set_consolidation_gate(&mut self, gate: f64) {
        self.consolidation_gate = gate;
    }

    /// Creates a weight matrix from all Associative+Stem morphons to each
    /// output port. Weights initialized to small random values (Xavier-scaled).
    /// Call this before training on classification tasks.
    pub fn enable_analog_readout(&mut self) {
        let assoc_ids: Vec<MorphonId> = self
            .morphons
            .values()
            .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
            .map(|m| m.id)
            .collect();

        let n_assoc = assoc_ids.len().max(1) as f64;
        let scale = 1.0 / n_assoc.sqrt();

        let n_outputs = self.output_ports.len();
        self.readout_weights = (0..n_outputs)
            .map(|_| {
                assoc_ids
                    .iter()
                    .map(|&id| (id, self.rng.random_range(-scale..scale)))
                    .collect()
            })
            .collect();

        // Also include sensory morphons as direct input to readout
        let sensory_ids: Vec<MorphonId> = self
            .morphons
            .values()
            .filter(|m| m.cell_type == CellType::Sensory)
            .map(|m| m.id)
            .collect();
        let n_total = (assoc_ids.len() + sensory_ids.len()).max(1) as f64;
        let sens_scale = 1.0 / n_total.sqrt();
        for weights in &mut self.readout_weights {
            for &id in &sensory_ids {
                weights.insert(id, self.rng.random_range(-sens_scale..sens_scale));
            }
        }

        self.readout_bias = vec![0.0; self.output_ports.len()];
        self.use_analog_readout = true;
    }

    /// Train the analog readout weights using softmax cross-entropy.
    ///
    /// error_j = target_j - softmax(outputs)_j
    ///
    /// Softmax normalizes outputs to sum to 1, preventing mode collapse:
    /// when one class dominates (softmax→0.99), its gradient shrinks to 0.01,
    /// breaking the positive feedback loop that per-output sigmoid creates.
    ///
    /// Also backprojects the output error to hidden morphons via feedback_signal.
    pub fn train_readout(&mut self, correct_index: usize, learning_rate: f64) {
        if !self.use_analog_readout || correct_index >= self.output_ports.len() {
            return;
        }

        let n_out = self.readout_weights.len();
        let outputs = self.read_output_analog();
        if outputs.len() != n_out {
            return;
        }

        // Sanitize outputs — NaN from morphon potential overflow kills the chain
        let outputs: Vec<f64> = outputs
            .iter()
            .map(|&x| if x.is_finite() { x } else { 0.0 })
            .collect();

        // Softmax: exp(x_j) / Σ_k exp(x_k), with numerical stability (subtract max)
        let max_out = outputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = outputs
            .iter()
            .map(|&x| (x - max_out).clamp(-50.0, 0.0).exp())
            .collect();
        let sum_exp: f64 = exps.iter().sum();
        if sum_exp < 1e-10 {
            return;
        } // all outputs identical → no gradient
        let softmax: Vec<f64> = exps.iter().map(|e| e / sum_exp).collect();

        // Cross-entropy gradient: error_j = target_j - softmax_j
        let errors: Vec<f64> = (0..n_out)
            .map(|j| {
                let target = if j == correct_index { 1.0 } else { 0.0 };
                target - softmax[j]
            })
            .collect();

        // Collect pre-synaptic activities (centered sigmoid: 0 at rest, ±0.5 range)
        let activities: HashMap<MorphonId, f64> = self.readout_weights[0]
            .keys()
            .filter_map(|&id| {
                self.morphons.get(&id).map(|m| {
                    let p = if m.potential.is_finite() {
                        m.potential
                    } else {
                        0.0
                    };
                    (id, 1.0 / (1.0 + (-p.clamp(-10.0, 10.0)).exp()) - 0.5)
                })
            })
            .collect();

        // Delta rule on readout weights (no L2 decay — it was erasing the
        // discriminative signal that the supervised gradient builds)
        for j in 0..n_out {
            if errors[j].abs() < 0.0001 {
                continue;
            }
            // Update bias
            if let Some(b) = self.readout_bias.get_mut(j) {
                let bias_delta = learning_rate * errors[j];
                if bias_delta.is_finite() {
                    *b = (*b + bias_delta).clamp(-5.0, 5.0);
                }
            }
            // Update weights
            for (&id, w) in self.readout_weights[j].iter_mut() {
                let act = activities.get(&id).copied().unwrap_or(0.0);
                let delta = learning_rate * act * errors[j];
                if delta.is_finite() {
                    *w = (*w + delta).clamp(-5.0, 5.0);
                }
            }
        }

        // Backproject error to hidden layer as feedback_signal (dendritic injection).
        // Each hidden morphon gets a signal proportional to how much it contributed
        // to the output error, weighted by the (fixed) feedback weights.
        // This doesn't use the readout weights (which would be backprop) — it uses
        // the separate fixed random feedback_weights (Direct Feedback Alignment).
        for (&assoc_id, dfa_weights) in &self.feedback_weights {
            let mut fb = 0.0;
            for &(motor_id, fb_w) in dfa_weights {
                if let Some(j) = self.output_ports.iter().position(|&id| id == motor_id) {
                    fb += fb_w * errors[j];
                }
            }
            if let Some(m) = self.morphons.get_mut(&assoc_id) {
                m.feedback_signal = fb;
            }

            // Tag-and-capture on sensory→associative synapses during readout training.
            // This is the RIGHT moment: we have fresh error signals, the feedback_signal
            // is just computed, and the reward reflects the current action quality.
            // Consolidates input→hidden connections that contribute to correct outputs.
            // Accumulate tag from repeated positive feedback (not one-shot).
            // Capture after sustained positive tagging across many presentations.
            // Tag-and-capture: gated on performance.
            // Below consolidation_gate: tags accumulate but never capture.
            // Above gate: captures happen, locking in proven representations.
            // "Sklerotien-Bildung" — only consolidate near a rich nutrient source.
            let cg = self.endo.channels.consolidation_gain as f64;
            if fb > 0.01 && self.recent_performance > self.consolidation_gate {
                let incoming = self.topology.incoming_synapses_mut(assoc_id);
                for (_, edge_idx) in incoming {
                    if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                        if !syn.consolidated {
                            syn.tag += fb.abs() * 0.02;
                            syn.tag = syn.tag.min(1.0);
                            syn.tag_strength = syn.tag;
                        }
                        // Endo consolidation_gain lowers effective capture threshold.
                        // High cg = capture-friendly (stressed/proliferating), low = selective (mature).
                        let effective_threshold = self.config.learning.capture_threshold
                            * self.endo.channels.capture_threshold_mult as f64
                            / cg;
                        if syn.tag > effective_threshold && !syn.consolidated {
                            syn.consolidated = true;
                            syn.tag = 0.0;
                            self.diag.total_captures += 1;
                        }
                    }
                }
            }
        }

        // === H2: READOUT-COUPLED ANCHORING ===
        // "Success breeds stability" — morphons with high readout weight magnitude
        // get their plasticity_rate reduced (become anchors). Morphons the readout
        // doesn't use stay plastic (sails). EMA update at glacial timescale (~0.005).
        // (Pilzak et al. 2026: iTDS — top-down consolidation gating at 100x learning rate)
        let n_out = self.readout_weights.len();
        if n_out > 0 {
            // Compute max readout weight magnitude per hidden morphon
            let mut importance: HashMap<MorphonId, f64> = HashMap::new();
            for j in 0..n_out {
                for (&id, &w) in &self.readout_weights[j] {
                    let entry = importance.entry(id).or_insert(0.0_f64);
                    *entry = entry.max(w.abs());
                }
            }

            // Find the max importance for normalization
            let max_imp = importance.values().cloned().fold(0.01_f64, f64::max);

            // Update plasticity_rate: high importance → lower plasticity (anchor)
            let anchoring_rate = 0.01; // EMA — takes ~100 readout updates to converge
            for (&id, &imp) in &importance {
                if let Some(m) = self.morphons.get_mut(&id) {
                    if m.cell_type == CellType::Associative || m.cell_type == CellType::Stem {
                        let normalized_imp = imp / max_imp; // [0, 1]
                                                            // Target plasticity: 0.2 for max importance, basal for zero importance
                        let target = 0.2 + (1.0 - normalized_imp) * 0.8;
                        m.plasticity_rate += anchoring_rate * (target - m.plasticity_rate);
                        m.plasticity_rate = m.plasticity_rate.clamp(0.1, 2.5);
                        // Dampen noise for important morphons — anchors produce stable output.
                        let noise_target = crate::types::intrinsic_noise_for(m.cell_type) * target;
                        m.intrinsic_noise += anchoring_rate * (noise_target - m.intrinsic_noise);
                        m.intrinsic_noise = m.intrinsic_noise.clamp(0.0, 0.2);
                    }
                }
            }
        }
    }

    /// Train the first readout output as a continuous regression target (MSE / delta rule).
    ///
    /// Unlike `train_readout` (softmax cross-entropy for N-way classification),
    /// this performs direct gradient descent on (output - target)² for scalar outputs.
    /// Use for prediction tasks where the target is a continuous value, not a class index.
    pub fn train_readout_value(&mut self, target: f64, learning_rate: f64) {
        if !self.use_analog_readout || self.readout_weights.is_empty() {
            return;
        }
        let outputs = self.read_output_analog();
        if outputs.is_empty() {
            return;
        }
        let predicted = if outputs[0].is_finite() { outputs[0] } else { 0.0 };
        let error = target - predicted; // gradient of 0.5*(predicted-target)²

        let activities: HashMap<MorphonId, f64> = self.readout_weights[0]
            .keys()
            .filter_map(|&id| {
                self.morphons.get(&id).map(|m| {
                    let p = if m.potential.is_finite() { m.potential.clamp(-10.0, 10.0) } else { 0.0 };
                    (id, 1.0 / (1.0 + (-p).exp()) - 0.5)
                })
            })
            .collect();

        if let Some(b) = self.readout_bias.get_mut(0) {
            let delta = learning_rate * error;
            if delta.is_finite() {
                *b = (*b + delta).clamp(-5.0, 5.0);
            }
        }
        for (&id, w) in self.readout_weights[0].iter_mut() {
            let act = activities.get(&id).copied().unwrap_or(0.0);
            let delta = learning_rate * act * error;
            if delta.is_finite() {
                *w = (*w + delta).clamp(-5.0, 5.0);
            }
        }
    }

    /// Number of input ports (sensory Morphons available for external input).
    pub fn input_size(&self) -> usize {
        self.input_ports.len()
    }

    /// Number of output ports (motor Morphons available for external output).
    pub fn output_size(&self) -> usize {
        self.output_ports.len()
    }

    /// Current readout training mode. In Hybrid mode, this reflects the
    /// active phase (Supervised until consolidation gate, then TDOnly).
    pub fn readout_training_mode(&self) -> ReadoutTrainingMode {
        match self.config.readout_mode {
            ReadoutTrainingMode::Hybrid => {
                if self.recent_performance > self.consolidation_gate {
                    ReadoutTrainingMode::TDOnly
                } else {
                    ReadoutTrainingMode::Supervised
                }
            }
            other => other,
        }
    }

    /// Report recent performance and trigger adaptive consolidation.
    ///
    /// Three regimes:
    /// 1. Below gate (avg < 30): fully plastic, no consolidation.
    /// 2. Above gate + improving: selective consolidation (top synapses only).
    /// 3. Above gate + declining (>20% below peak): deconsolidate weakest 10%
    ///    to restore plasticity. "Metabolic Melting" / "Disturbance-induced Growth".
    pub fn report_performance(&mut self, performance: f64) {
        self.recent_performance += 0.05 * (performance - self.recent_performance);

        if self.recent_performance > self.peak_performance {
            self.peak_performance = self.recent_performance;
        }

        // Deconsolidation: if performance drops >40% from peak, melt weakest synapses.
        // 20% was too sensitive — a few bad CartPole episodes trigger melting that
        // destroys a good policy. 40% requires sustained degradation.
        if self.peak_performance > self.consolidation_gate
            && self.recent_performance < self.peak_performance * 0.6
        {
            self.deconsolidate_weakest(0.05); // melt 5% of consolidated synapses
        }
    }

    /// Report episode end and trigger performance-relative capture.
    ///
    /// If this episode was better than the running average, consolidate tagged synapses
    /// proportional to how much better it was. If worse, decay tags instead.
    /// This replaces per-tick `modulation.reward > threshold` capture, which is broken
    /// for RL because the reward level stays saturated during any alive episode.
    pub fn report_episode_end(&mut self, episode_steps: f64) {
        let delta = episode_steps - self.running_avg_steps;
        self.running_avg_steps = 0.95 * self.running_avg_steps + 0.05 * episode_steps;

        if delta > 0.0 {
            // Above average — increase consolidation_level, scaled by Endo consolidation_gain.
            // Biology: PRP availability gates how much a good episode consolidates.
            let cg = self.endo.channels.consolidation_gain as f64;
            let strength = (delta / self.running_avg_steps.max(1.0)).min(1.0);
            let mut capture_count = 0_u64;
            for ei in self.topology.graph.edge_indices() {
                if let Some(syn) = self.topology.graph.edge_weight_mut(ei) {
                    if syn.tag > 0.1 && syn.consolidation_level < 1.0 {
                        let delta_level = strength * syn.tag_strength.min(1.0) * 0.1 * cg;
                        syn.consolidation_level = (syn.consolidation_level + delta_level).min(1.0);
                        syn.tag *= 0.5;
                        capture_count += 1;
                    }
                }
            }
            // Signal to Endo that captures happened — both the direct counter reset
            // AND the pending count so sense_vitals() sees episode-gated captures
            // in Rule 5 and Rule 11 (not just the ticks_since counter).
            if capture_count > 0 {
                self.endo.ticks_since_last_capture = 0;
                self.diag.episode_captures_pending += capture_count;
            }
        } else if delta < 0.0 {
            // Below average — decay tags, don't consolidate
            self.decay_all_tags(0.5);
        }

        // Compute winner diversity entropy from episode fire counts
        let total_fires: u64 = self.episode_fire_counts.values().sum();
        if total_fires > 0 {
            let n = self.episode_fire_counts.len();
            let h: f64 = self
                .episode_fire_counts
                .values()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / total_fires as f64;
                    -p * p.ln()
                })
                .sum();
            self.diag.winner_diversity_entropy = if n > 1 { h / (n as f64).ln() } else { 0.0 };
        }
        self.episode_fire_counts.clear();
    }

    /// Decay all active tags after a below-average episode.
    fn decay_all_tags(&mut self, factor: f64) {
        for ei in self.topology.graph.edge_indices() {
            if let Some(syn) = self.topology.graph.edge_weight_mut(ei) {
                if syn.tag > 0.01 {
                    syn.tag *= factor;
                    syn.tag_strength *= factor;
                }
            }
        }
    }

    /// Deconsolidate the weakest fraction of consolidated synapses.
    /// "Metabolic Melting" — restores plasticity where the system is underperforming.
    /// Deconsolidated synapses can be re-learned and re-captured.
    fn deconsolidate_weakest(&mut self, fraction: f64) {
        // Collect all consolidated synapses with their weights
        let mut consolidated: Vec<(petgraph::graph::EdgeIndex, f64)> = self
            .topology
            .graph
            .edge_indices()
            .filter_map(|ei| {
                self.topology
                    .graph
                    .edge_weight(ei)
                    .filter(|s| s.consolidated)
                    .map(|s| (ei, s.weight.abs()))
            })
            .collect();

        if consolidated.is_empty() {
            return;
        }

        // Sort by absolute weight (weakest first — these contributed least)
        consolidated.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let n_to_melt = ((consolidated.len() as f64 * fraction).ceil() as usize).max(1);
        for &(ei, _) in &consolidated[..n_to_melt.min(consolidated.len())] {
            if let Some(syn) = self.topology.graph.edge_weight_mut(ei) {
                syn.consolidated = false;
                syn.consolidation_level = 0.0;
                syn.tag = 0.0; // reset tag for fresh accumulation
                syn.activity_trace = 0.0;
                // Halve myelination — treadmill removal will erode the rest
                // now that consolidation_level is zero (wrapping target drops).
                syn.myelination *= 0.5;
            }
        }
    }

    /// Check if consolidation is enabled (performance above gate).
    pub fn consolidation_enabled(&self) -> bool {
        self.recent_performance > self.consolidation_gate
    }

    /// Inject a reward signal.
    pub fn inject_reward(&mut self, strength: f64) {
        self.modulation.inject_reward(strength);
    }

    /// Inject a novelty signal.
    pub fn inject_novelty(&mut self, strength: f64) {
        self.modulation.inject_novelty(strength);
    }

    /// Inject an arousal signal.
    pub fn inject_arousal(&mut self, strength: f64) {
        self.modulation.inject_arousal(strength);
    }

    /// Signal the start of a new sequence (temporal processing tasks).
    ///
    /// Resets temporal context so information from the previous sequence
    /// does not bleed into the next one. Specifically:
    ///
    /// 1. **Working memory cleared** — active patterns from the last sequence
    ///    are irrelevant to the new one.
    /// 2. **Eligibility traces zeroed** — prevents credit from the last
    ///    sequence leaking into this one's weight updates.
    /// 3. **Novelty burst injected** — sequence boundaries are inherently
    ///    novel; high novelty keeps the network maximally attentive at onset.
    ///
    /// Does NOT reset membrane potentials — the network's learned dynamic
    /// state carries over. Only the accumulated temporal context resets.
    ///
    /// For RL tasks, `inject_reward()` + environment reset already serves
    /// this function. This method is for supervised sequence tasks.
    pub fn sequence_reset(&mut self) {
        // 1. Clear working memory
        self.memory.working.clear();

        // 2. Zero eligibility traces on all synapses
        for edge_idx in self.topology.graph.edge_indices() {
            if let Some(syn) = self.topology.graph.edge_weight_mut(edge_idx) {
                syn.eligibility = 0.0;
            }
        }

        // 3. Reset membrane voltages so each trial starts from a clean baseline.
        //    Without this, residual state from trial N bleeds into trial N+1 and
        //    the reservoir can't produce consistent states for the same input pattern.
        for m in self.morphons.values_mut() {
            m.potential = 0.0;
            m.prev_potential = 0.0;
            m.fired = false;
            m.input_accumulator = 0.0;
            m.refractory_timer = 0.0;
        }
        for j in 0..self.hot.active_count {
            self.hot.voltage[j] = 0.0;
            self.hot.input_current[j] = 0.0;
            self.hot.fired[j] = false;
            self.hot.fired_prev[j] = false;
            self.hot.refractory[j] = 0.0;
        }

        // 4. Clear in-flight spikes (stale spikes from previous trial can activate wrong neurons)
        self.resonance.clear();

        // 5. Mild novelty burst — signals context boundary without saturating plasticity
        self.modulation.inject_novelty(0.3);
    }

    /// Reset all per-trial transient state without touching learned weights or
    /// STDP eligibility traces.
    ///
    /// Use this before each evaluation image to prevent temporal bleed between
    /// test samples — the "state-reset evaluator" described in the MNIST roadmap.
    /// Unlike `sequence_reset()`, this does NOT zero eligibility traces and does
    /// NOT inject novelty, so the trained network state is fully preserved.
    pub fn reset_transient_state(&mut self) {
        // Membrane voltages + accumulators (same as present_image, but all morphons)
        for m in self.morphons.values_mut() {
            m.potential = 0.0;
            m.prev_potential = 0.0;
            m.input_accumulator = 0.0;
            m.fired = false;
            m.refractory_timer = 0.0;
        }
        // Hot-array shadow copy
        for j in 0..self.hot.active_count {
            self.hot.voltage[j] = 0.0;
            self.hot.input_current[j] = 0.0;
            self.hot.fired[j] = false;
            self.hot.fired_prev[j] = false;
            self.hot.refractory[j] = 0.0;
        }
        // In-flight spikes
        self.resonance.clear();
        // Working memory context
        self.memory.working.clear();
        // Per-episode fire counts (used by report_episode_end)
        self.episode_fire_counts.clear();
        // ANCS item context
        self.current_ancs_item = None;
    }

    /// Inject a small excitatory boost to morphons that belong to the top working-memory
    /// pattern. Call this at the start of a new timestep (before `step()`) to let the
    /// most-recently-active pattern bias the next processing cycle — a lightweight form of
    /// top-down attentional priming that does not require a separate recurrent pathway.
    ///
    /// `strength` is the potential delta injected per morphon (typical range 0.01–0.1).
    /// Zero or negative `strength` is a no-op.
    pub fn feed_working_memory_feedback(&mut self, strength: f64) {
        if strength <= 0.0 {
            return;
        }
        // Take the pattern with the highest activation weight from working memory.
        let top_pattern: Option<Vec<crate::MorphonId>> = self
            .memory
            .working
            .active_patterns()
            .iter()
            .max_by(|a, b| a.activation.partial_cmp(&b.activation).unwrap_or(std::cmp::Ordering::Equal))
            .map(|wp| wp.pattern.clone());

        if let Some(pattern) = top_pattern {
            let delta = strength as f32;
            for id in &pattern {
                if let Some(m) = self.morphons.get_mut(id) {
                    m.potential += strength;
                    // Mirror into hot-array if this morphon is hot-mapped
                    if let Some(&j) = self.hot.id_to_idx.get(id) {
                        self.hot.voltage[j] = (self.hot.voltage[j] + delta).min(5.0);
                    }
                }
            }
        }
    }

    /// Inject a targeted reward at a specific output port's morphon.
    ///
    /// Two-hop credit assignment (inspired by SADP's population agreement):
    /// 1. Boost eligibility on all incoming synapses to the motor morphon (hop 0)
    /// 2. Propagate backward: for each pre-synaptic morphon that connects to the
    ///    motor, boost eligibility on THEIR incoming synapses proportional to the
    ///    connecting weight (hop 1). This gives the associative layer credit.
    ///
    /// This is local (no chain rule), but provides output-specific credit to
    /// the hidden layer — the critical missing piece for classification tasks.
    pub fn inject_reward_at(&mut self, output_index: usize, strength: f64) {
        let Some(&motor_id) = self.output_ports.get(output_index) else {
            return;
        };

        // Hop 0: boost eligibility on motor's incoming synapses
        let motor_incoming = self.topology.incoming_synapses_mut(motor_id);
        let pre_ids_and_weights: Vec<(MorphonId, f64)> = motor_incoming
            .into_iter()
            .filter_map(|(pre_id, edge_idx)| {
                if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                    syn.eligibility += strength;
                    syn.eligibility = syn.eligibility.clamp(-1.0, 1.0);
                    Some((pre_id, syn.weight))
                } else {
                    None
                }
            })
            .collect();

        // Hop 1: propagate credit backward to the associative layer.
        // Each pre-synaptic morphon's incoming synapses get a boost proportional
        // to the weight of the pre→motor connection (stronger connection = more credit).
        let decay = 0.5; // credit attenuates per hop
        for (pre_id, weight) in pre_ids_and_weights {
            let hop1_strength = strength * decay * weight.abs();
            if hop1_strength < 0.01 {
                continue;
            }
            let pre_incoming = self.topology.incoming_synapses_mut(pre_id);
            for (_, edge_idx) in pre_incoming {
                if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                    syn.eligibility += hop1_strength;
                    syn.eligibility = syn.eligibility.clamp(-1.0, 1.0);
                }
            }
        }

        // V3: Credit motor morphon energy for successful output.
        if strength > 0.0 {
            if let Some(m) = self.morphons.get_mut(&motor_id) {
                m.energy = (m.energy
                    + self.config.metabolic.reward_for_successful_output * strength)
                    .min(1.0);
            }
        }

        // Weak global signal for paths deeper than 2 hops
        self.modulation.inject_reward(strength * 0.1);
    }

    /// Inject a targeted inhibition at a specific output port's morphon.
    /// Inhibit an output port by decaying its eligibility traces toward zero.
    ///
    /// Unlike inject_reward_at which ADDS positive eligibility, inhibition
    /// DECAYS existing eligibility toward zero. This prevents incorrect outputs
    /// from being strengthened by subsequent reward, without driving weights
    /// negative (which would permanently silence the motor morphon).
    pub fn inject_inhibition_at(&mut self, output_index: usize, strength: f64) {
        let Some(&motor_id) = self.output_ports.get(output_index) else {
            return;
        };
        let decay_factor = (1.0 - strength).max(0.0); // strength=0.3 → keep 70% of eligibility

        let motor_incoming = self.topology.incoming_synapses_mut(motor_id);
        let pre_ids: Vec<MorphonId> = motor_incoming
            .into_iter()
            .filter_map(|(pre_id, edge_idx)| {
                if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                    syn.eligibility *= decay_factor; // decay toward zero, don't go negative
                    Some(pre_id)
                } else {
                    None
                }
            })
            .collect();

        // Hop 1: also decay eligibility on paths feeding into the motor's sources
        let hop1_decay = (1.0 - strength * 0.5).max(0.0);
        for pre_id in pre_ids {
            let pre_incoming = self.topology.incoming_synapses_mut(pre_id);
            for (_, edge_idx) in pre_incoming {
                if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                    syn.eligibility *= hop1_decay;
                }
            }
        }
    }

    /// Contrastive reward: reward the correct output, inhibit incorrect outputs.
    ///
    /// This is the key mechanism for breaking mode collapse in classification tasks.
    /// The correct motor morphon gets boosted eligibility (→ strengthened by next reward),
    /// while incorrect motors get reduced eligibility (→ weakened or unaffected).
    ///
    /// `correct_index`: the output port index that should have been selected
    /// `reward_strength`: how much to reward the correct path (0.0-1.0)
    /// `inhibit_strength`: how much to inhibit incorrect paths (0.0-1.0)
    pub fn reward_contrastive(
        &mut self,
        correct_index: usize,
        reward_strength: f64,
        inhibit_strength: f64,
    ) {
        // Broadcast reward into the global modulation channel so that
        // reward-correlated energy (metabolic) and learning (three-factor)
        // both see the signal. Without this, classification tasks that only
        // use reward_contrastive() never set modulation.reward.
        self.modulation.inject_reward(reward_strength);

        // Reward-correlated energy: fired morphons earn energy when reward arrives.
        // Hubs fire for all inputs → reward ±symmetric → net zero income.
        // Specialists fire for correct class → mostly positive → positive income.
        let coeff = self.config.metabolic.reward_energy_coefficient;
        if coeff > 0.0 && reward_strength > 0.0 {
            for m in self.morphons.values_mut() {
                if m.fired {
                    m.energy = (m.energy + reward_strength * coeff).min(1.0);
                }
            }
        }

        let n_outputs = self.output_ports.len();
        for i in 0..n_outputs {
            if i == correct_index {
                self.inject_reward_at(i, reward_strength);
            } else {
                self.inject_inhibition_at(i, inhibit_strength);
            }
        }
    }

    /// SADP-inspired hidden layer teaching signal.
    ///
    /// For classification tasks, this provides output-specific credit to the
    /// associative (hidden) layer without backpropagation. Based on Supervised
    /// SADP (arXiv:2601.08526) which achieves 99.1% MNIST using population-based
    /// agreement between hidden neurons and target outputs.
    ///
    /// For each associative morphon:
    /// 1. Compute agreement with the correct output (did this hidden neuron fire
    ///    when the correct motor morphon was active?)
    /// 2. Boost eligibility on incoming synapses proportional to agreement
    /// 3. Decay eligibility for anti-correlated hidden neurons
    ///
    /// Call this after `process_steps()` and before `reward_contrastive()`.
    /// `correct_index`: which output class is the target.
    /// `strength`: teaching signal magnitude (0.0-1.0).
    /// SADP-inspired hidden layer teaching signal.
    ///
    /// Uses the TARGET output (what the correct class SHOULD fire) instead of
    /// actual motor activity. This is critical: at the start of training all
    /// motors have similar potential, so agreement with ACTUAL output is ~zero.
    /// Using the TARGET provides a non-zero teaching signal from step 1.
    ///
    /// Based on SADP (arXiv:2601.08526): "Hidden layers are trained using spike
    /// agreement-dependent plasticity, which reinforces neurons whose spike trains
    /// agree with the correct-class output neuron beyond chance."
    ///
    /// For each associative morphon, the teaching signal is:
    ///   kappa_j = activity_j × target_correct - activity_j × target_incorrect
    ///           = activity_j × (1.0 - 0.0) = activity_j   (for active hidden neurons)
    ///           = 0.0                                        (for inactive hidden neurons)
    ///
    /// Active hidden neurons get their incoming synapses boosted (they should
    /// help fire the correct output). Inactive ones get no signal (not penalized).
    pub fn teach_hidden(&mut self, correct_index: usize, strength: f64) {
        if correct_index >= self.output_ports.len() {
            return;
        }

        // TARGET signal: 1.0 for correct class, 0.0 for incorrect.
        // This is the supervised signal — what the output SHOULD be.
        let target_correct: f64 = 1.0;
        let target_incorrect: f64 = 0.0;
        let _n_incorrect = (self.output_ports.len() - 1).max(1) as f64;

        let assoc_ids: Vec<MorphonId> = self
            .morphons
            .values()
            .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
            .map(|m| m.id)
            .collect();

        for assoc_id in assoc_ids {
            let assoc_active = self
                .morphons
                .get(&assoc_id)
                .map(|m| if m.fired { 1.0 } else { 0.0 })
                .unwrap_or(0.0);

            if assoc_active < 0.01 {
                continue;
            } // inactive neurons get no signal

            // Kappa-like agreement: how well does this hidden neuron's activity
            // correlate with the TARGET output pattern?
            // agreement = activity × target_correct = activity × 1.0
            // anti_agreement = activity × mean(target_incorrect) = activity × 0.0
            // teaching_signal = agreement - anti_agreement = activity
            let teaching_signal = assoc_active * (target_correct - target_incorrect) * strength;

            // Boost eligibility on incoming synapses to this active hidden neuron.
            // These sensory→hidden connections carried the input that made this
            // hidden neuron fire during the correct class presentation.
            let incoming = self.topology.incoming_synapses_mut(assoc_id);
            for (_, edge_idx) in incoming {
                if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                    syn.eligibility += teaching_signal;
                    syn.eligibility = syn.eligibility.clamp(-1.0, 1.0);
                }
            }

            // Also modulate the outgoing synapses from this hidden neuron
            // to the CORRECT motor morphon (strengthen the correct path).
            let correct_motor_id = self.output_ports[correct_index];
            if let Some((ei, _)) = self.topology.synapse_between(assoc_id, correct_motor_id) {
                if let Some(syn) = self.topology.synapse_mut(ei) {
                    syn.eligibility += teaching_signal;
                    syn.eligibility = syn.eligibility.clamp(-1.0, 1.0);
                }
            }

            // Decay eligibility on outgoing synapses to INCORRECT motor morphons.
            for (i, &motor_id) in self.output_ports.iter().enumerate() {
                if i == correct_index {
                    continue;
                }
                if let Some((ei, _)) = self.topology.synapse_between(assoc_id, motor_id) {
                    if let Some(syn) = self.topology.synapse_mut(ei) {
                        syn.eligibility *= 0.5; // decay, don't go negative
                    }
                }
            }
        }
    }

    /// Direct supervised learning — bypasses three-factor entirely.
    ///
    /// Applies the delta rule directly to weights:
    ///   Δw_ij = learning_rate × pre_activity_i × (target_j - output_j)
    ///
    /// This is what SADP effectively does. No eligibility traces, no modulation.
    /// If this works and three-factor doesn't, the problem is in the modulation
    /// pipeline, not the architecture.
    ///
    /// `correct_index`: which output should be 1.0 (others should be 0.0)
    /// `learning_rate`: step size (0.001-0.1 typical)
    pub fn teach_supervised(&mut self, correct_index: usize, learning_rate: f64) {
        if correct_index >= self.output_ports.len() {
            return;
        }

        // Build target vector: 1.0 for correct, 0.0 for others
        let n_out = self.output_ports.len();
        let targets: Vec<f64> = (0..n_out)
            .map(|i| if i == correct_index { 1.0 } else { 0.0 })
            .collect();

        // Get actual outputs (sigmoid of potential)
        let outputs: Vec<f64> = self
            .output_ports
            .iter()
            .filter_map(|id| self.morphons.get(id))
            .map(|m| 1.0 / (1.0 + (-m.potential).exp()))
            .collect();

        if outputs.len() != n_out {
            return;
        }

        // Delta rule on ALL synapses feeding into each motor morphon
        for j in 0..n_out {
            let error_j = targets[j] - outputs[j];
            if error_j.abs() < 0.001 {
                continue;
            }

            let motor_id = self.output_ports[j];
            let incoming = self.topology.incoming_synapses_mut(motor_id);
            for (pre_id, edge_idx) in incoming {
                // Pre-synaptic activity: continuous sigmoid of potential.
                // Binary gate (fired || activity_history > 0.05) caused a cold-start
                // deadlock: neurons can't learn because they don't fire, and they
                // don't fire because they haven't learned. Sigmoid gives a gradient
                // signal even for cold neurons (sigmoid(0) = 0.5).
                let pre_act = self
                    .morphons
                    .get(&pre_id)
                    .map(|m| 1.0 / (1.0 + (-m.potential).exp()))
                    .unwrap_or(0.0);

                if pre_act < 0.01 {
                    continue;
                } // skip only truly dead neurons

                if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                    let delta_w = learning_rate * pre_act * error_j;
                    syn.weight += delta_w;
                    syn.weight = syn.weight.clamp(
                        -self.config.learning.weight_max,
                        self.config.learning.weight_max,
                    );
                }
            }
        }

        // Hidden layer: delta rule on sensory→hidden synapses using error
        // backpropagated one hop (not backprop — just the output error weighted
        // by the hidden→motor weight, which is a local quantity).
        for j in 0..n_out {
            let error_j = targets[j] - outputs[j];
            if error_j.abs() < 0.001 {
                continue;
            }

            let motor_id = self.output_ports[j];
            let motor_incoming: Vec<(MorphonId, f64)> = self
                .topology
                .incoming(motor_id)
                .into_iter()
                .map(|(pre_id, syn)| (pre_id, syn.weight))
                .collect();

            for (hidden_id, w_hj) in motor_incoming {
                let hidden_act = self
                    .morphons
                    .get(&hidden_id)
                    .map(|m| 1.0 / (1.0 + (-m.potential).exp()))
                    .unwrap_or(0.0);
                // Local error for this hidden neuron from this output
                let hidden_error = error_j * w_hj * hidden_act * (1.0 - hidden_act);

                let hidden_incoming = self.topology.incoming_synapses_mut(hidden_id);
                for (sens_id, edge_idx) in hidden_incoming {
                    let sens_act = self
                        .morphons
                        .get(&sens_id)
                        .map(|m| 1.0 / (1.0 + (-m.potential).exp()))
                        .unwrap_or(0.0);

                    if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                        let delta_w = learning_rate * sens_act * hidden_error;
                        syn.weight += delta_w;
                        syn.weight = syn.weight.clamp(
                            -self.config.learning.weight_max,
                            self.config.learning.weight_max,
                        );
                    }
                }
            }
        }
    }

    /// Direct supervised learning using RAW INPUT values as pre-synaptic activity.
    ///
    /// Unlike teach_supervised() which reads morphon potentials (distorted by
    /// leaky integration, noise, clamping), this uses the actual input values.
    /// Proven to work: external logistic regression with the same data hits 100%.
    ///
    /// Updates sensory→motor direct connections and hidden→motor connections
    /// using the input-port-mapped raw values.
    pub fn teach_supervised_with_input(
        &mut self,
        input: &[f64],
        correct_index: usize,
        learning_rate: f64,
    ) {
        if correct_index >= self.output_ports.len() {
            return;
        }
        let n_out = self.output_ports.len();

        let targets: Vec<f64> = (0..n_out)
            .map(|i| if i == correct_index { 1.0 } else { 0.0 })
            .collect();
        let outputs: Vec<f64> = self
            .output_ports
            .iter()
            .filter_map(|id| self.morphons.get(id))
            .map(|m| 1.0 / (1.0 + (-m.potential).exp()))
            .collect();
        if outputs.len() != n_out {
            return;
        }

        // Build a map from input_port MorphonId → raw input value
        let mut input_map = std::collections::HashMap::new();
        let ports_per_input = (self.input_ports.len() / input.len().max(1)).max(1);
        for (i, &value) in input.iter().enumerate() {
            let start = i * ports_per_input;
            let end = (start + ports_per_input).min(self.input_ports.len());
            for port_idx in start..end {
                input_map.insert(self.input_ports[port_idx], value);
            }
        }

        // Delta rule: update weights on motor incoming synapses
        for j in 0..n_out {
            let error_j = targets[j] - outputs[j];
            if error_j.abs() < 0.001 {
                continue;
            }

            let motor_id = self.output_ports[j];
            let incoming = self.topology.incoming_synapses_mut(motor_id);
            for (pre_id, edge_idx) in incoming {
                // Use raw input value if this is a sensory morphon, else use sigmoid(potential)
                let pre_act = if let Some(&raw) = input_map.get(&pre_id) {
                    raw / 3.0 // normalize to ~[0,1]
                } else {
                    self.morphons
                        .get(&pre_id)
                        .map(|m| (1.0 / (1.0 + (-m.potential).exp())).max(0.01))
                        .unwrap_or(0.0)
                };

                if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                    // Delta rule + mild L2 weight decay
                    let weight_decay = 0.001 * syn.weight;
                    let delta_w = learning_rate * pre_act * error_j - weight_decay;
                    syn.weight += delta_w;
                    syn.weight = syn.weight.clamp(
                        -self.config.learning.weight_max,
                        self.config.learning.weight_max,
                    );
                }
            }
        }
    }

    /// Process input and return output (single inference step with learning).
    pub fn process(&mut self, input: &[f64]) -> Vec<f64> {
        self.feed_input(input);
        self.step();
        self.read_output()
    }

    /// Process input with multiple internal steps to let signals propagate.
    ///
    /// For reactive tasks where the input→output path spans multiple hops,
    /// this runs `n` internal steps per call so motor morphons have time
    /// to receive signals from sensory morphons before the output is read.
    /// Input is fed on every step to maintain sensory drive.
    pub fn process_steps(&mut self, input: &[f64], n: usize) -> Vec<f64> {
        for _ in 0..n {
            self.feed_input(input);
            self.step();
        }
        self.read_output()
    }

    /// Process one step, then apply confidence-based limbic salience modulation.
    ///
    /// If `self.limbic.enabled`, this is the preferred call for classification tasks:
    /// 1. Run the cortical step (`process()`).
    /// 2. Compute salience from readout uncertainty (high uncertainty = hard sample = salient).
    /// 3. Inject a bounded arousal delta for the *next* step — attention boost
    ///    for samples the model found difficult.
    ///
    /// "High road" signal: salience from cortical output, not raw input pixels.
    /// Correctly differentiates hard digits (uncertain readout → high salience)
    /// from easy ones (peaked readout → low salience), without flooding.
    /// Arousal injection capped at `0.3 × salience` — never dominates Endo's baseline.
    ///
    /// If limbic is disabled, equivalent to plain `process()`.
    pub fn process_with_limbic(
        &mut self,
        observation: &[f64],
        _label: Option<usize>,
    ) -> Vec<f64> {
        let output = self.process(observation);
        if !self.limbic.enabled {
            return output;
        }
        // High road: derive salience from readout uncertainty after cortical processing.
        let salience = self.limbic.salience_detector.evaluate_from_output(&output);
        // Gate arousal injection on Endo stage. During Proliferating the model hasn't
        // formed representations yet — confidence is uniformly low for every input, so
        // salience ≈ 0.9 constantly and arousal becomes noise (same failure mode as
        // pixel-EMA flooding). Only inject once we're Consolidating or beyond, when
        // confidence variance is real (easy classes confident, hard classes uncertain).
        use crate::endoquilibrium::DevelopmentalStage;
        let stage = self.endo.stage();
        if stage != DevelopmentalStage::Proliferating {
            self.modulation.inject_arousal(0.3 * salience);
        }
        output
    }

    /// Deliver reward through the Limbic Circuit after processing a sample.
    ///
    /// If `self.limbic.enabled`:
    /// 1. Computes RPE = received_reward − expected_reward_for_class.
    /// 2. Applies RPE-scaled metabolic energy to fired morphons
    ///    (replaces the flat reward-energy in reward_contrastive).
    /// 3. Records outcome in the salience detector's reward-association map.
    /// 4. Stores a high-salience episode in the episodic tagger.
    ///
    /// Call this **after** `reward_contrastive()`. The existing energy boost in
    /// reward_contrastive still runs; this adds the RPE layer on top.
    /// If limbic is disabled this is a no-op.
    pub fn deliver_limbic_reward(
        &mut self,
        observation: &[f64],
        label: usize,
        reward: f64,
    ) {
        if !self.limbic.enabled { return; }

        // Compute RPE and update reward expectation for this class.
        let rpe = self.limbic.motivational_drive.compute_rpe_for_class(label, reward);

        // Record outcome so salience_detector learns which patterns are rewarding.
        self.limbic.salience_detector.record_outcome(observation, reward);

        // RPE-scaled energy: morphons that fire during surprising successes earn more.
        // High RPE (unexpected correct) → large energy boost.
        // Low RPE (expected correct, or incorrect) → small or no boost.
        let coeff = self.config.metabolic.reward_energy_coefficient;
        if coeff > 0.0 && rpe > 0.0 {
            for m in self.morphons.values_mut() {
                if m.fired {
                    m.energy = (m.energy + rpe * coeff).min(1.0);
                }
            }
        }

        // Store episode if salient enough.
        let salience = self.limbic.salience_detector.current_salience;
        let active_morphons: Vec<u64> = self.morphons.values()
            .filter(|m| m.fired)
            .map(|m| m.id)
            .collect();
        self.limbic.episodic_tagger.maybe_record(
            observation,
            &active_morphons,
            reward,
            rpe,
            salience,
            Some(label),
            self.step_count,
        );
    }

    /// V2: Explicitly trigger a dream cycle (e.g., between RL episodes).
    /// Can be called regardless of developmental stage.
    pub fn trigger_dream(&mut self) {
        self.dream_cycle();
    }

    /// V2: Dreaming engine — offline consolidation and self-optimization.
    ///
    /// Three phases:
    /// 1. Consolidation: high-tag synapses get increased consolidation_level
    /// 2. Self-optimization: stale, unused, weak synapses get refreshed
    /// 3. Curiosity signal: clusters with high internal PE variance + low
    ///    external connectivity get mild Novelty injection
    fn dream_cycle(&mut self) {
        if !self.config.dream.enabled {
            return;
        }
        // Don't consolidate during dreaming until performance is above the gate.
        // Early consolidation locks in random weights.
        let can_consolidate = self.recent_performance > self.consolidation_gate;

        // === Phase 1: Dream Consolidation ===
        // Find high-tag-strength synapses and consolidate them
        let dream_lr = self.config.dream.dream_learning_rate;
        let tag_thresh = self.config.dream.dream_tag_threshold;
        let max_synapses = self.config.dream.max_dream_synapses;

        if can_consolidate {
            // Scale dream consolidation by Endo consolidation_gain — same PRP model
            // as waking consolidation. Mature (cg=0.5) → selective dreaming.
            // Differentiating (cg=2.0) → aggressive replay consolidation.
            let cg = self.endo.channels.consolidation_gain as f64;
            let candidates: Vec<petgraph::graph::EdgeIndex> = self
                .topology
                .graph
                .edge_indices()
                .filter(|&ei| {
                    self.topology.graph.edge_weight(ei).is_some_and(|syn| {
                        syn.tag_strength > tag_thresh && syn.consolidation_level < 1.0
                    })
                })
                .take(max_synapses)
                .collect();

            for ei in candidates {
                if let Some(syn) = self.topology.graph.edge_weight_mut(ei) {
                    let delta_level = dream_lr * syn.tag_strength.min(1.0) * 0.1 * cg;
                    syn.consolidation_level = (syn.consolidation_level + delta_level).min(1.0);
                    if syn.consolidation_level > 0.5 {
                        syn.consolidated = true;
                    }
                    syn.tag *= 0.7;
                    syn.tag_strength *= 0.7;
                }
            }
        }

        // Replay episodic memories during dreaming (more than waking)
        let replayed = self.memory.episodic.replay(5);
        for episode in replayed {
            if episode.reward > 0.1 {
                self.modulation.inject_reward(episode.reward * 0.1);
            }
        }

        // ANCS-guided dream replay: prioritize high-importance, low-consolidation items.
        // Biologically: hippocampal replay during slow-wave sleep strengthens
        // salient but not-yet-consolidated memories.
        let ancs_replay: Vec<(u64, f64)> = self
            .ancs
            .top_items(5, None)
            .iter()
            .map(|item| (item.id, item.reward))
            .collect();
        for (id, reward) in ancs_replay {
            if reward > 0.1 {
                self.modulation.inject_reward(reward * 0.1);
            }
            self.modulation.inject_novelty(0.05);
            self.ancs.record_replay(id);
        }

        // === Phase 2: Self-Optimization — refresh stale synapses ===
        let stale_age = self.config.dream.stale_synapse_age;
        let stale_usage = self.config.dream.stale_usage_threshold;
        let reset_scale = self.config.dream.reset_weight_scale;

        for ei in self.topology.graph.edge_indices() {
            if let Some(syn) = self.topology.graph.edge_weight_mut(ei) {
                if syn.age > stale_age
                    && syn.usage_count < stale_usage
                    && !syn.consolidated
                    && syn.weight.abs() < 0.1
                {
                    // Refresh to small value (not prune) — give a second chance
                    let sign = if syn.weight >= 0.0 { 1.0 } else { -1.0 };
                    let pseudo_random =
                        ((syn.age.wrapping_mul(31).wrapping_add(7)) % 100) as f64 / 100.0;
                    syn.weight = sign * reset_scale * (0.5 + 0.5 * pseudo_random);
                    syn.age = 0;
                    syn.usage_count = 0;
                    syn.eligibility = 0.0;
                    syn.pre_trace = 0.0;
                    syn.post_trace = 0.0;
                    syn.activity_trace = 0.0;
                    syn.myelination = 0.0;
                }
            }
        }

        // === Phase 3: Curiosity Signal — topology anomalies ===
        let curiosity_strength = self.config.dream.curiosity_signal_strength;

        for cluster in self.clusters.values() {
            if cluster.members.len() < 2 {
                continue;
            }

            // Internal PE variance
            let pe_vals: Vec<f64> = cluster
                .members
                .iter()
                .filter_map(|mid| self.morphons.get(mid))
                .map(|m| m.prediction_error)
                .collect();
            if pe_vals.is_empty() {
                continue;
            }
            let mean_pe = pe_vals.iter().sum::<f64>() / pe_vals.len() as f64;
            let pe_variance =
                pe_vals.iter().map(|pe| (pe - mean_pe).powi(2)).sum::<f64>() / pe_vals.len() as f64;

            // External connectivity ratio
            let member_set: std::collections::HashSet<MorphonId> =
                cluster.members.iter().copied().collect();
            let external_connections: usize = cluster
                .members
                .iter()
                .flat_map(|&mid| self.topology.outgoing(mid))
                .filter(|(nid, _)| !member_set.contains(nid))
                .count();
            let external_ratio =
                external_connections as f64 / (cluster.members.len() as f64 * 5.0).max(1.0);

            // High internal variance + low external connectivity = anomaly
            if pe_variance > 0.1 && external_ratio < 0.3 {
                for &mid in &cluster.members {
                    if let Some(m) = self.morphons.get_mut(&mid) {
                        m.input_accumulator += curiosity_strength;
                    }
                }
                self.modulation.inject_novelty(curiosity_strength * 0.5);
            }
        }
    }

    /// V6: Multi-index memory retrieval via Reciprocal Rank Fusion.
    ///
    /// Queries both working memory (recent active patterns) and episodic memory
    /// (consolidated high-reward episodes) and fuses the rankings. Patterns
    /// that appear in both stores with good overlap get boosted scores.
    ///
    /// Useful for pattern completion: inject the returned patterns' morphons
    /// to bias the network toward previously learned states.
    /// Multi-index memory retrieval via Reciprocal Rank Fusion.
    ///
    /// Queries both the legacy TripleMemory stores (working + episodic) and the
    /// ANCS-Core backend, then merges the results. ANCS contributes only
    /// Supported and Hypothesis items (epistemic filter, Phase 3c).
    pub fn retrieve_memory(
        &self,
        query: &[MorphonId],
        top_k: usize,
    ) -> Vec<crate::memory::MemoryRetrievalResult> {
        use crate::memory::{MemoryRetrievalResult, MemorySource};

        // Legacy TripleMemory RRF (working + episodic)
        let mut results = self.memory.retrieve_fused(query, top_k);

        // ANCS-Core: epistemic-filtered query
        let ancs_query = crate::ancs::RetrievalQuery { pattern: query.to_vec() };
        for (item_id, score) in self.ancs.retrieve(&ancs_query, top_k) {
            if let Some(pattern) = self.ancs.item_pattern(item_id) {
                results.push(MemoryRetrievalResult {
                    pattern,
                    score,
                    source: MemorySource::Episodic,
                });
            }
        }

        // Re-sort and truncate the merged list
        results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    /// Get system inspection statistics.
    pub fn inspect(&self) -> SystemStats {
        let mut differentiation_map = HashMap::new();
        let mut max_gen: Generation = 0;
        let mut total_energy = 0.0;
        let mut total_pe = 0.0;
        let mut firing_count = 0;

        for m in self.morphons.values() {
            let effective_type = if m.fused_with.is_some() { CellType::Fused } else { m.cell_type };
            *differentiation_map.entry(effective_type).or_insert(0usize) += 1;
            max_gen = max_gen.max(m.generation);
            total_energy += m.energy;
            total_pe += m.prediction_error;
            if m.fired {
                firing_count += 1;
            }
        }

        let n = self.morphons.len().max(1) as f64;

        // Axonal property stats
        let synapse_count = self.topology.synapse_count();
        let (mut total_myelin, mut max_myelin, mut myelinated, mut total_eff_delay) =
            (0.0, 0.0_f64, 0usize, 0.0);
        let (mut min_base_delay, mut max_base_delay) = (f64::MAX, 0.0_f64);
        for (_, _, ei) in self.topology.all_edges() {
            let syn = &self.topology.graph[ei];
            total_myelin += syn.myelination;
            max_myelin = max_myelin.max(syn.myelination);
            if syn.myelination > 0.01 {
                myelinated += 1;
            }
            total_eff_delay += syn.effective_delay();
            min_base_delay = min_base_delay.min(syn.delay);
            max_base_delay = max_base_delay.max(syn.delay);
        }
        if synapse_count == 0 {
            min_base_delay = 0.0;
        }
        let s = synapse_count.max(1) as f64;

        SystemStats {
            max_morphons: self.effective_max_morphons,
            at_morphon_cap: self.morphons.len() >= self.effective_max_morphons,
            total_morphons: self.morphons.len(),
            total_synapses: synapse_count,
            fused_clusters: self.clusters.len(),
            differentiation_map,
            max_generation: max_gen,
            avg_energy: total_energy / n,
            avg_prediction_error: total_pe / n,
            firing_rate: firing_count as f64 / n,
            working_memory_items: self.memory.working.len(),
            episodic_memory_items: self.memory.episodic.len(),
            step_count: self.step_count,
            total_born: self.total_born,
            total_died: self.total_died,
            total_transdifferentiations: self.total_transdifferentiations,
            apoptosis_age_eligible: self.diag.apoptosis_age_eligible,
            apoptosis_silent: self.diag.apoptosis_silent,
            apoptosis_energy_low: self.diag.apoptosis_energy_low,
            assoc_activity_min: self.diag.assoc_activity_min,
            assoc_activity_max: self.diag.assoc_activity_max,
            assoc_activity_mean: self.diag.assoc_activity_mean,
            region_health: self
                .config
                .target_morphology
                .as_ref()
                .map(|tm| tm.region_health(&self.morphons))
                .unwrap_or_default(),
            field_pe_max: self.diag.field_pe_max,
            field_pe_mean: self.diag.field_pe_mean,
            avg_myelination: total_myelin / s,
            max_myelination: max_myelin,
            myelinated_synapses: myelinated,
            avg_effective_delay: total_eff_delay / s,
            min_base_delay,
            max_base_delay,
            ancs_items: self.ancs.len(),
            ancs_contested: self
                .ancs
                .items_with_state(crate::ancs::MemoryEpistemicState::Contested)
                .len(),
            ancs_supported: self
                .ancs
                .items_with_state(crate::ancs::MemoryEpistemicState::Supported)
                .len(),
            ancs_pressure: format!("{:?}", self.ancs.pressure_mode()),
            synapses_saved_fwd_recent: self.diag.synapses_saved_fwd_recent,
        }
    }

    /// Build a lineage tree from the current morphon population.
    ///
    /// Useful for exporting parent-child relationships for visualization
    /// (e.g. arXiv paper figures).
    pub fn lineage_tree(&self) -> LineageTree {
        lineage::build_lineage_tree(&self.morphons)
    }

    /// Get the effective morphon cap (auto-derived or explicitly set).
    pub fn max_morphons(&self) -> usize {
        self.effective_max_morphons
    }

    /// Get the current simulation step.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Get the current learning pipeline diagnostics.
    ///
    /// Updated every step. Use `diagnostics().summary()` for a concise log line,
    /// or `diagnostics().firing_summary()` for per-type firing rates.
    pub fn diagnostics(&self) -> &Diagnostics {
        &self.diag
    }

    /// Read the critic morphons' aggregate potential as V(state).
    /// Returns the mean membrane potential of all critic morphons.
    pub fn critic_value(&self) -> f64 {
        if self.critic_ports.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .critic_ports
            .iter()
            .filter_map(|id| self.morphons.get(id))
            .map(|m| m.potential)
            .sum();
        sum / self.critic_ports.len() as f64
    }

    /// Number of critic morphons.
    pub fn critic_size(&self) -> usize {
        self.critic_ports.len()
    }

    /// Compute TD error and inject it as the reward modulation signal.
    ///
    /// δ = reward + γ·V(s') - V(s)
    ///
    /// This is the core of the actor-critic architecture within the Morphon
    /// framework. The critic morphons learn to predict V(s), and the TD error
    /// drives both the critic's own learning (minimize prediction error) and
    /// the actor's learning (strengthen actions that lead to better-than-expected states).
    ///
    /// Biologically: dopamine neurons compute δ from ventral striatum (critic)
    /// signals and broadcast it to dorsal striatum (actor) and cortex.
    pub fn inject_td_error(&mut self, reward: f64, gamma: f64) -> f64 {
        let v_new = self.critic_value();
        let td_error = reward + gamma * v_new - self.prev_critic_value;
        self.prev_critic_value = v_new;
        self.last_td_error = td_error;

        // Inject positive TD as reward; negative TD → no injection, let decay
        // reduce the channel. This preserves directional signal via reward_delta():
        // positive TD → reward spike → positive delta → strengthen active traces
        // negative TD → reward decays → negative delta → weaken active traces
        if td_error > 0.0 {
            self.modulation.inject_reward(td_error.min(1.0));
        }

        td_error
    }

    /// Average prediction error across all Morphons.
    fn avg_prediction_error(&self) -> f64 {
        if self.morphons.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.morphons.values().map(|m| m.prediction_error).sum();
        sum / self.morphons.len() as f64
    }
}
