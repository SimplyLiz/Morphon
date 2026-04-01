//! The MI System — top-level orchestration of the Morphogenic Intelligence engine.
//!
//! Uses the Dual-Clock Architecture (Section 3.8) to separate fast inference
//! from slow morphogenesis, with homeostatic protection mechanisms throughout.

use serde::{Deserialize, Serialize};
use crate::developmental::{self, DevelopmentalConfig};
use crate::diagnostics::Diagnostics;
use crate::lineage::{self, LineageTree};
use crate::homeostasis::{self, HomeostasisParams};
use crate::learning::{self, LearningParams};
use crate::memory::TripleMemory;
use crate::morphogenesis::{self, Cluster, MorphogenesisParams, MorphogenesisReport};
use crate::morphon::{MetabolicConfig, Morphon};
use crate::neuromodulation::Neuromodulation;
use crate::resonance::ResonanceEngine;
use crate::scheduler::SchedulerConfig;
use crate::topology::Topology;
use crate::types::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
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
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
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
        }
    }
}

/// Inspection results for the system state.
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemStats {
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
    /// Whether to use analog readout (true) or spike-based motor potential (false).
    pub(crate) use_analog_readout: bool,

    /// Next available Morphon ID.
    pub(crate) next_morphon_id: MorphonId,
    /// Next available Cluster ID.
    pub(crate) next_cluster_id: ClusterId,
    /// Current simulation step.
    pub(crate) step_count: u64,

    /// Learning pipeline diagnostics (updated every step).
    pub(crate) diag: Diagnostics,

    /// Cumulative morphon birth count.
    pub(crate) total_born: usize,
    /// Cumulative morphon death count.
    pub(crate) total_died: usize,

    /// Running average of recent episode performance (set by caller via report_performance).
    /// Used to gate consolidation: no captures until performance exceeds a threshold.
    pub(crate) recent_performance: f64,
    /// Performance threshold for enabling consolidation.
    /// Below this, the system stays fully plastic. Above, it starts consolidating.
    pub(crate) consolidation_gate: f64,
    /// Peak performance seen so far — used for deconsolidation trigger.
    pub(crate) peak_performance: f64,
}

impl System {
    /// Create a new MI System by running its developmental program.
    pub fn new(config: SystemConfig) -> Self {
        let mut rng = rand::rng();

        let (mut morphons, topology, next_id) =
            developmental::develop(&config.developmental, &mut rng);

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
        let all_assoc_ids: Vec<MorphonId> = morphons.values()
            .filter(|m| m.cell_type == CellType::Associative)
            .map(|m| m.id)
            .collect();
        let mut feedback_weights: HashMap<MorphonId, Vec<(MorphonId, f64)>> = HashMap::new();
        for (i, &aid) in all_assoc_ids.iter().enumerate() {
            let motor_weights: Vec<(MorphonId, f64)> = motor_ids.iter()
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
            use_analog_readout: false,   // off by default, enabled per-task
            next_morphon_id: next_id,
            next_cluster_id: 0,
            step_count: 0,
            diag: Diagnostics::default(),
            total_born: 0,
            total_died: 0,
            recent_performance: 0.0,
            consolidation_gate: 15.0, // consolidate once above 15 (current avg)
            peak_performance: 0.0,
        };

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
        };

        for step in 0..100_u64 {
            let n_inputs = system.input_ports.len();
            if n_inputs > 0 {
                let noise_input: Vec<f64> = (0..n_inputs)
                    .map(|i| {
                        let v = ((i as u64).wrapping_mul(step.wrapping_add(997)) % 1000) as f64 / 1000.0;
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
                syn.tag = 0.0;
                syn.consolidated = false; // critical: undo warm-up captures
            }
        }
        system.diag.total_captures = 0;

        system
    }

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
        let mut rng = rand::rng();

        let tick = self.config.scheduler.tick(self.step_count);
        let mut report = MorphogenesisReport::default();

        // === FAST PATH (every step) ===
        // 1. Propagate spikes from currently firing Morphons
        self.resonance.propagate(&self.morphons, &self.topology);

        // 2. Deliver spikes that have reached their targets
        let delivered = self.resonance.deliver(&mut self.morphons, dt);

        let spikes_delivered = delivered.len();

        // 3. Update all Morphon states (integrate input, fire/not-fire).
        //    Must run BEFORE spike-timing eligibility so post_fired reflects
        //    whether the delivered spike caused the target to fire this step.
        //    V3 Metabolic Budget: pre-compute synapse counts for energy cost.
        let metabolic = &self.config.metabolic;
        let degree_map: HashMap<MorphonId, usize> = self.morphons.keys()
            .map(|&id| (id, self.topology.degree(id)))
            .collect();
        #[cfg(feature = "parallel")]
        self.morphons.par_iter_mut().for_each(|(id, m)| {
            let sc = degree_map.get(id).copied().unwrap_or(0);
            m.step(dt, sc, metabolic);
        });
        #[cfg(not(feature = "parallel"))]
        self.morphons.values_mut().for_each(|m| {
            let sc = degree_map.get(&m.id).copied().unwrap_or(0);
            m.step(dt, sc, metabolic);
        });

        // 4. k-Winner-Take-All lateral inhibition in the Associative layer.
        //    Only the top-k most active associative morphons survive each step.
        //    All others have their firing suppressed and potential clamped.
        //    This forces different neurons to specialize on different input patterns
        //    (Diehl & Cook 2015). k = ~5% of population allows sparse distributed
        //    representations while maintaining diversity.
        {
            let mut assoc_potentials: Vec<(MorphonId, f64, bool)> = self.morphons.values()
                .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
                .map(|m| (m.id, m.potential, m.fired))
                .collect();

            if !assoc_potentials.is_empty() {
                // k = 15% of population — enough diversity for the readout
                // while maintaining competitive specialization via threshold boost.
                let k = (assoc_potentials.len() as f64 * 0.15).ceil() as usize;
                let k = k.max(3).min(assoc_potentials.len());

                // Sort by potential descending — top-k winners survive
                assoc_potentials.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let winners: std::collections::HashSet<MorphonId> = assoc_potentials[..k]
                    .iter().map(|(id, _, _)| *id).collect();

                // Suppress non-winners: reset fired flag, clamp potential below threshold
                for &(id, _, _) in &assoc_potentials[k..] {
                    if let Some(m) = self.morphons.get_mut(&id) {
                        if m.fired {
                            m.fired = false;
                            m.activity_history.push(0.0); // undo the 1.0 pushed in step()
                        }
                        // Mild suppression — don't destroy potential, just prevent firing
                        m.potential = m.potential.min(m.threshold * 0.5);
                    }
                }

                // Adaptive threshold: winners get a small threshold boost (Diehl & Cook).
                // Prevents any single neuron from dominating all inputs.
                // Slow decay via homeostatic regulation restores threshold over time.
                for &id in &winners {
                    if let Some(m) = self.morphons.get_mut(&id) {
                        m.threshold += 0.02;
                    }
                }
            }
        }

        // 5. Spike-timing: boost pre_trace on delivered synapses.
        //    The spike delivery is direct evidence that pre fired recently.
        //    By incrementing pre_trace here, the trace-based STDP in the medium
        //    path will detect the causal relationship even if the pre-synaptic
        //    morphon is now in refractory (fired=false).
        for spike in &delivered {
            if let Some((ei, _)) = self.topology.synapse_between(spike.source, spike.target) {
                if let Some(synapse) = self.topology.synapse_mut(ei) {
                    synapse.pre_trace += 1.0;
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
            let error_vec: Vec<(MorphonId, f64)> = self.output_ports.iter()
                .filter_map(|&mid| self.morphons.get(&mid).map(|m| {
                    let sig = 1.0 / (1.0 + (-m.potential).exp());
                    (mid, td_err * (sig - 0.5) * 2.0)
                }))
                .collect();

            // Project error to each Associative morphon through fixed random weights
            let feedback_strength = 0.5;
            for (&assoc_id, weights) in &self.feedback_weights {
                let feedback: f64 = weights.iter()
                    .filter_map(|(mid, bw)| {
                        error_vec.iter().find(|(id, _)| id == mid).map(|(_, e)| bw * e)
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

        // === MEDIUM PATH ===
        // Two learning regimes (Frémaux et al. 2013):
        // - Critic morphons: TD-LTP — Δw = lr × δ × pre_trace (direct value learning)
        // - Actor morphons: Three-factor STDP — Δw = eligibility × M(t) × plasticity
        // This breaks the circular dependency where critic and actor share learning rules.
        let mut captures_this_step = 0_u64;
        if tick.medium {
            let critic_set: std::collections::HashSet<MorphonId> =
                self.critic_ports.iter().copied().collect();
            let td_err = self.last_td_error;
            let wmax = self.config.learning.weight_max;

            for &id in &morphon_ids {
                if critic_set.contains(&id) {
                    // === CRITIC: TD-LTP ===
                    // The critic learns to predict V(s) by minimizing TD error.
                    // Weight update depends only on TD error × pre-synaptic trace,
                    // independent of the actor's eligibility computation.
                    let incoming = self.topology.incoming_synapses_mut(id);
                    for (_, edge_idx) in incoming {
                        if let Some(synapse) = self.topology.synapse_mut(edge_idx) {
                            // Decay traces (same as actor)
                            let trace_decay = (-dt / self.config.learning.tau_trace).exp();
                            synapse.pre_trace *= trace_decay;
                            synapse.post_trace *= trace_decay;

                            // TD-LTP: Δw = lr × δ × pre_trace
                            // Only LTP direction — critic converges to value estimate
                            let td_lr = 0.01;
                            let delta_w = td_lr * td_err * synapse.pre_trace;
                            synapse.weight = (synapse.weight + delta_w).clamp(-wmax, wmax);
                            synapse.age += 1;
                        }
                    }
                } else {
                    // === ACTOR: Three-factor learning ===
                    // Motor morphons: standard receptor-gated modulation (TD via Reward channel)
                    // Associative morphons: DFA feedback signal (neuron-specific credit)
                    let (post_activity, post_receptors, feedback_sig, is_assoc) =
                        self.morphons.get(&id)
                            .map(|m| {
                                let activity = if m.cell_type == CellType::Motor {
                                    let sig = 1.0 / (1.0 + (-m.potential).exp());
                                    (sig - 0.5) * 2.0
                                } else {
                                    if m.fired { 1.0 } else { 0.0 }
                                };
                                let is_a = m.cell_type == CellType::Associative
                                    || m.cell_type == CellType::Stem;
                                (activity, m.receptors.clone(), m.feedback_signal, is_a)
                            })
                            .unwrap_or_default();
                    let incoming = self.topology.incoming_synapses_mut(id);

                    for (pre_id, edge_idx) in incoming {
                        let pre_fired = self.morphons.get(&pre_id).map_or(false, |m| m.fired);

                        if let Some(synapse) = self.topology.synapse_mut(edge_idx) {
                            learning::update_eligibility(
                                synapse,
                                pre_fired,
                                post_activity,
                                &self.config.learning,
                                dt,
                            );

                            if is_assoc && feedback_sig.abs() > 0.001 {
                                // DFA climbing-fiber rule: Δw = pre_trace × feedback_signal × lr
                                // Uses pre_trace (decaying memory of recent pre-synaptic firing)
                                // NOT eligibility (which is STDP-gated and attenuates the signal).
                                // NOT binary pre_fired (too sparse at 5% firing rate).
                                // pre_trace persists ~10 steps after firing — "is this synapse
                                // carrying signal?" — the right gate for targeted DFA updates.
                                // Biologically: climbing fibers override STDP timing requirements.
                                let dfa_lr = 0.02;
                                let weight_decay = 0.001 * synapse.weight;
                                let delta_w = synapse.pre_trace * feedback_sig * dfa_lr - weight_decay;
                                synapse.weight = (synapse.weight + delta_w).clamp(-wmax, wmax);

                                // Tag-and-Capture on DFA path: consolidate synapses where
                                // the DFA update was strong AND reward is high.
                                // This gives the hidden layer long-term memory.
                                let dfa_strength = (synapse.pre_trace * feedback_sig).abs();
                                if dfa_strength > 0.1 && !synapse.consolidated {
                                    synapse.tag = 1.0;
                                    synapse.tag_strength = dfa_strength;
                                }
                                // Capture: strong DFA update + high reward → consolidate
                                // Performance-gated capture: no consolidation until proven performance
                                if synapse.tag > 0.1
                                    && self.modulation.reward > 0.3
                                    && !synapse.consolidated
                                    && self.recent_performance > self.consolidation_gate
                                {
                                    synapse.weight += 0.1 * synapse.tag_strength * self.modulation.reward;
                                    synapse.weight = synapse.weight.clamp(-wmax, wmax);
                                    synapse.consolidated = true;
                                    synapse.tag = 0.0;
                                    captures_this_step += 1;
                                }
                                synapse.age += 1;
                                if synapse.eligibility.abs() > 0.1 {
                                    synapse.usage_count += 1;
                                }
                            } else {
                                // Standard three-factor for Motor, Sensory, Modulatory
                                let plasticity = self.modulation.plasticity_rate();
                                let captured = learning::apply_weight_update(
                                    synapse,
                                    &self.modulation,
                                    &self.config.learning,
                                    plasticity,
                                    &post_receptors,
                                );
                                // Performance gate: undo capture if below threshold
                                if captured && self.recent_performance <= self.consolidation_gate {
                                    synapse.consolidated = false; // revert — not ready to consolidate
                                }
                                if captured && self.recent_performance > self.consolidation_gate {
                                    captures_this_step += 1;
                                }
                                // L2 weight decay on all three-factor synapses
                                synapse.weight -= 0.0005 * synapse.weight;
                            }
                        }
                    }
                }
            }
        }

        // === WEIGHT NORMALIZATION (after STDP updates) ===
        // Per-neuron L1 normalization for Associative morphons' incoming weights.
        // Strengthening some inputs forces weakening others — creates synaptic
        // competition that, combined with WTA, produces specialized feature detectors.
        // (Diehl & Cook 2015: "sum of all incoming weights kept constant")
        if tick.medium {
            let assoc_ids: Vec<MorphonId> = self.morphons.values()
                .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
                .map(|m| m.id)
                .collect();

            for &aid in &assoc_ids {
                let incoming = self.topology.incoming_synapses_mut(aid);
                let edge_indices: Vec<_> = incoming.into_iter().map(|(_, ei)| ei).collect();
                if edge_indices.is_empty() { continue; }

                // Compute current L1 norm of positive incoming weights
                let mut pos_sum = 0.0_f64;
                for &ei in &edge_indices {
                    if let Some(syn) = self.topology.graph.edge_weight(ei) {
                        if syn.weight > 0.0 {
                            pos_sum += syn.weight;
                        }
                    }
                }

                // Target norm: proportional to number of connections
                let target_norm = edge_indices.len() as f64 * 0.3; // ~0.3 per synapse avg
                if pos_sum > 0.01 {
                    let scale = target_norm / pos_sum;
                    for &ei in &edge_indices {
                        if let Some(syn) = self.topology.synapse_mut(ei) {
                            if syn.weight > 0.0 {
                                syn.weight *= scale;
                            }
                        }
                    }
                }
            }
        }

        // === SLOW PATH ===
        if tick.slow {
            let slow_report = morphogenesis::step_slow(
                &mut self.morphons,
                &mut self.topology,
                &self.config.morphogenesis,
                &self.config.learning,
                self.modulation.homeostasis,
                &self.config.lifecycle,
                &mut rng,
            );
            report.synapses_created = slow_report.synapses_created;
            report.synapses_pruned = slow_report.synapses_pruned;
            report.migrations = slow_report.migrations;
        }

        // === GLACIAL PATH (with checkpoint/rollback protection) ===
        if tick.glacial {
            // Checkpoint before structural changes
            let all_ids: Vec<MorphonId> = self.morphons.keys().copied().collect();
            let checkpoint = homeostasis::create_checkpoint(
                &all_ids,
                &self.morphons,
                &self.topology,
            );

            let glacial_report = morphogenesis::step_glacial(
                &mut self.morphons,
                &mut self.topology,
                &mut self.clusters,
                &mut self.next_morphon_id,
                &mut self.next_cluster_id,
                &self.config.morphogenesis,
                self.modulation.arousal,
                &self.config.lifecycle,
                &mut rng,
            );
            report.morphons_born = glacial_report.morphons_born;
            report.morphons_died = glacial_report.morphons_died;
            self.total_born += glacial_report.morphons_born;
            self.total_died += glacial_report.morphons_died;
            report.differentiations = glacial_report.differentiations;
            report.fusions = glacial_report.fusions;
            report.defusions = glacial_report.defusions;

            // Rollback synapses if prediction error spiked after structural changes
            let surviving_ids: Vec<MorphonId> = all_ids
                .iter()
                .filter(|id| self.morphons.contains_key(id))
                .copied()
                .collect();
            if homeostasis::should_rollback(
                &checkpoint,
                &surviving_ids,
                &self.morphons,
                &self.config.homeostasis,
            ) {
                homeostasis::rollback_synapses(&checkpoint, &mut self.topology);
                self.diag.rollback_triggered = true;
                self.diag.total_rollbacks += 1;
            }
        }

        // === HOMEOSTASIS ===
        if tick.homeostasis {
            homeostasis::synaptic_scaling(&self.morphons, &mut self.topology);
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
                self.memory.episodic.encode(
                    pattern,
                    self.modulation.reward,
                    self.modulation.novelty,
                    self.step_count,
                );
            }
        }

        // === DIAGNOSTICS ===
        // Compute learning pipeline diagnostics for observability.
        let prev_total_captures = self.diag.total_captures;
        let prev_total_rollbacks = self.diag.total_rollbacks;
        let rollback_triggered = self.diag.rollback_triggered;
        self.diag = Diagnostics::snapshot(&self.morphons, &self.topology);
        self.diag.spikes_delivered_this_step = spikes_delivered;
        self.diag.spikes_pending = self.resonance.pending_count();
        self.diag.captures_this_step = captures_this_step;
        self.diag.total_captures = prev_total_captures + captures_this_step;
        self.diag.total_rollbacks = prev_total_rollbacks;
        self.diag.rollback_triggered = rollback_triggered;

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
            self.morphons.par_iter_mut().for_each(|(_, m)| curvature_update(m));
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
        self.readout_weights.iter().map(|weights| {
            weights.iter().map(|(&assoc_id, &w)| {
                let act = self.morphons.get(&assoc_id)
                    .map(|m| 1.0 / (1.0 + (-m.potential).exp()))
                    .unwrap_or(0.0);
                w * act
            }).sum()
        }).collect()
    }

    /// Enable analog readout and initialize readout weights.
    ///
    /// Creates a weight matrix from all Associative+Stem morphons to each
    /// output port. Weights initialized to small random values (Xavier-scaled).
    /// Call this before training on classification tasks.
    pub fn enable_analog_readout(&mut self) {
        let assoc_ids: Vec<MorphonId> = self.morphons.values()
            .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
            .map(|m| m.id)
            .collect();

        let n_assoc = assoc_ids.len().max(1) as f64;
        let scale = 1.0 / n_assoc.sqrt();
        let mut rng = rand::rng();
        use rand::Rng;

        self.readout_weights = (0..self.output_ports.len()).map(|_| {
            assoc_ids.iter().map(|&id| {
                (id, rng.random_range(-scale..scale))
            }).collect()
        }).collect();

        // Also include sensory morphons as direct input to readout
        let sensory_ids: Vec<MorphonId> = self.morphons.values()
            .filter(|m| m.cell_type == CellType::Sensory)
            .map(|m| m.id)
            .collect();
        let n_total = (assoc_ids.len() + sensory_ids.len()).max(1) as f64;
        let sens_scale = 1.0 / n_total.sqrt();
        for weights in &mut self.readout_weights {
            for &id in &sensory_ids {
                weights.insert(id, rng.random_range(-sens_scale..sens_scale));
            }
        }

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
        if !self.use_analog_readout || correct_index >= self.output_ports.len() { return; }

        let n_out = self.readout_weights.len();
        let outputs = self.read_output_analog();
        if outputs.len() != n_out { return; }

        // Softmax: exp(x_j) / Σ_k exp(x_k), with numerical stability (subtract max)
        let max_out = outputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = outputs.iter().map(|&x| (x - max_out).exp()).collect();
        let sum_exp: f64 = exps.iter().sum();
        let softmax: Vec<f64> = exps.iter().map(|e| e / sum_exp.max(1e-10)).collect();

        // Cross-entropy gradient: error_j = target_j - softmax_j
        let errors: Vec<f64> = (0..n_out).map(|j| {
            let target = if j == correct_index { 1.0 } else { 0.0 };
            target - softmax[j]
        }).collect();

        // Collect pre-synaptic activities (sigmoid of morphon potentials)
        let activities: HashMap<MorphonId, f64> = self.readout_weights[0].keys()
            .filter_map(|&id| {
                self.morphons.get(&id).map(|m| (id, 1.0 / (1.0 + (-m.potential).exp())))
            })
            .collect();

        // Delta rule on readout weights + L2 weight decay
        for j in 0..n_out {
            if errors[j].abs() < 0.0001 { continue; }
            for (&id, w) in self.readout_weights[j].iter_mut() {
                let act = activities.get(&id).copied().unwrap_or(0.0);
                let weight_decay = 0.001 * *w;
                *w += learning_rate * act * errors[j] - weight_decay;
                *w = w.clamp(-5.0, 5.0);
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
            if fb > 0.01 {
                let can_consolidate = self.recent_performance > self.consolidation_gate;
                let incoming = self.topology.incoming_synapses_mut(assoc_id);
                for (_, edge_idx) in incoming {
                    if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                        if !syn.consolidated {
                            syn.tag += fb.abs() * 0.02;
                            syn.tag = syn.tag.min(1.0);
                            syn.tag_strength = syn.tag;
                        }
                        if syn.tag > 0.3 && !syn.consolidated && can_consolidate {
                            syn.consolidated = true;
                            syn.tag = 0.0;
                            self.diag.total_captures += 1;
                        }
                    }
                }
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

        // Deconsolidation: if performance drops >20% from peak, melt weakest synapses
        if self.peak_performance > self.consolidation_gate
            && self.recent_performance < self.peak_performance * 0.8
        {
            self.deconsolidate_weakest(0.1); // melt 10% of consolidated synapses
        }
    }

    /// Deconsolidate the weakest fraction of consolidated synapses.
    /// "Metabolic Melting" — restores plasticity where the system is underperforming.
    /// Deconsolidated synapses can be re-learned and re-captured.
    fn deconsolidate_weakest(&mut self, fraction: f64) {
        // Collect all consolidated synapses with their weights
        let mut consolidated: Vec<(petgraph::graph::EdgeIndex, f64)> = self.topology.graph
            .edge_indices()
            .filter_map(|ei| {
                self.topology.graph.edge_weight(ei)
                    .filter(|s| s.consolidated)
                    .map(|s| (ei, s.weight.abs()))
            })
            .collect();

        if consolidated.is_empty() { return; }

        // Sort by absolute weight (weakest first — these contributed least)
        consolidated.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let n_to_melt = ((consolidated.len() as f64 * fraction).ceil() as usize).max(1);
        for &(ei, _) in &consolidated[..n_to_melt.min(consolidated.len())] {
            if let Some(syn) = self.topology.graph.edge_weight_mut(ei) {
                syn.consolidated = false;
                syn.tag = 0.0; // reset tag for fresh accumulation
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
        let Some(&motor_id) = self.output_ports.get(output_index) else { return };

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
            if hop1_strength < 0.01 { continue; }
            let pre_incoming = self.topology.incoming_synapses_mut(pre_id);
            for (_, edge_idx) in pre_incoming {
                if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                    syn.eligibility += hop1_strength;
                    syn.eligibility = syn.eligibility.clamp(-1.0, 1.0);
                }
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
        let Some(&motor_id) = self.output_ports.get(output_index) else { return };
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
        if correct_index >= self.output_ports.len() { return; }

        // TARGET signal: 1.0 for correct class, 0.0 for incorrect.
        // This is the supervised signal — what the output SHOULD be.
        let target_correct: f64 = 1.0;
        let target_incorrect: f64 = 0.0;
        let n_incorrect = (self.output_ports.len() - 1).max(1) as f64;

        let assoc_ids: Vec<MorphonId> = self.morphons.values()
            .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
            .map(|m| m.id)
            .collect();

        for assoc_id in assoc_ids {
            let assoc_active = self.morphons.get(&assoc_id)
                .map(|m| if m.fired { 1.0 } else { 0.0 })
                .unwrap_or(0.0);

            if assoc_active < 0.01 { continue; } // inactive neurons get no signal

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
                if i == correct_index { continue; }
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
        if correct_index >= self.output_ports.len() { return; }

        // Build target vector: 1.0 for correct, 0.0 for others
        let n_out = self.output_ports.len();
        let targets: Vec<f64> = (0..n_out).map(|i| if i == correct_index { 1.0 } else { 0.0 }).collect();

        // Get actual outputs (sigmoid of potential)
        let outputs: Vec<f64> = self.output_ports.iter()
            .filter_map(|id| self.morphons.get(id))
            .map(|m| 1.0 / (1.0 + (-m.potential).exp()))
            .collect();

        if outputs.len() != n_out { return; }

        // Delta rule on ALL synapses feeding into each motor morphon
        for j in 0..n_out {
            let error_j = targets[j] - outputs[j];
            if error_j.abs() < 0.001 { continue; }

            let motor_id = self.output_ports[j];
            let incoming = self.topology.incoming_synapses_mut(motor_id);
            for (pre_id, edge_idx) in incoming {
                // Pre-synaptic activity: binary (fired or not in recent history).
                // Using sigmoid(potential) fails because cold morphons have sigmoid≈0.
                // Using a floor erases class discrimination.
                // Binary firing is the clearest signal: this sensory neuron was active
                // for this input, so its connection to the correct output should strengthen.
                let pre_act = self.morphons.get(&pre_id)
                    .map(|m| if m.fired || m.activity_history.mean() > 0.05 { 1.0 } else { 0.0 })
                    .unwrap_or(0.0);

                if pre_act < 0.01 { continue; } // skip inactive pre-synaptic neurons

                if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                    let delta_w = learning_rate * error_j; // pre_act is binary gate, not multiplier
                    syn.weight += delta_w;
                    syn.weight = syn.weight.clamp(-self.config.learning.weight_max, self.config.learning.weight_max);
                }
            }
        }

        // Hidden layer: delta rule on sensory→hidden synapses using error
        // backpropagated one hop (not backprop — just the output error weighted
        // by the hidden→motor weight, which is a local quantity).
        for j in 0..n_out {
            let error_j = targets[j] - outputs[j];
            if error_j.abs() < 0.001 { continue; }

            let motor_id = self.output_ports[j];
            let motor_incoming: Vec<(MorphonId, f64)> = self.topology.incoming(motor_id)
                .into_iter()
                .map(|(pre_id, syn)| (pre_id, syn.weight))
                .collect();

            for (hidden_id, w_hj) in motor_incoming {
                let hidden_act = self.morphons.get(&hidden_id)
                    .map(|m| 1.0 / (1.0 + (-m.potential).exp()))
                    .unwrap_or(0.0);
                // Local error for this hidden neuron from this output
                let hidden_error = error_j * w_hj * hidden_act * (1.0 - hidden_act);

                let hidden_incoming = self.topology.incoming_synapses_mut(hidden_id);
                for (sens_id, edge_idx) in hidden_incoming {
                    let sens_act = self.morphons.get(&sens_id)
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
    pub fn teach_supervised_with_input(&mut self, input: &[f64], correct_index: usize, learning_rate: f64) {
        if correct_index >= self.output_ports.len() { return; }
        let n_out = self.output_ports.len();

        let targets: Vec<f64> = (0..n_out).map(|i| if i == correct_index { 1.0 } else { 0.0 }).collect();
        let outputs: Vec<f64> = self.output_ports.iter()
            .filter_map(|id| self.morphons.get(id))
            .map(|m| 1.0 / (1.0 + (-m.potential).exp()))
            .collect();
        if outputs.len() != n_out { return; }

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
            if error_j.abs() < 0.001 { continue; }

            let motor_id = self.output_ports[j];
            let incoming = self.topology.incoming_synapses_mut(motor_id);
            for (pre_id, edge_idx) in incoming {
                // Use raw input value if this is a sensory morphon, else use sigmoid(potential)
                let pre_act = if let Some(&raw) = input_map.get(&pre_id) {
                    raw / 3.0 // normalize to ~[0,1]
                } else {
                    self.morphons.get(&pre_id)
                        .map(|m| (1.0 / (1.0 + (-m.potential).exp())).max(0.01))
                        .unwrap_or(0.0)
                };

                if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                    // Delta rule + L2 weight decay to prevent drift to ±clamp
                    let weight_decay = 0.01 * syn.weight; // pulls toward zero (10x stronger)
                    let delta_w = learning_rate * pre_act * error_j - weight_decay;
                    syn.weight += delta_w;
                    syn.weight = syn.weight.clamp(-self.config.learning.weight_max, self.config.learning.weight_max);
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

    /// Get system inspection statistics.
    pub fn inspect(&self) -> SystemStats {
        let mut differentiation_map = HashMap::new();
        let mut max_gen: Generation = 0;
        let mut total_energy = 0.0;
        let mut total_pe = 0.0;
        let mut firing_count = 0;

        for m in self.morphons.values() {
            *differentiation_map.entry(m.cell_type).or_insert(0usize) += 1;
            max_gen = max_gen.max(m.generation);
            total_energy += m.energy;
            total_pe += m.prediction_error;
            if m.fired {
                firing_count += 1;
            }
        }

        let n = self.morphons.len().max(1) as f64;

        SystemStats {
            total_morphons: self.morphons.len(),
            total_synapses: self.topology.synapse_count(),
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
        }
    }

    /// Build a lineage tree from the current morphon population.
    ///
    /// Useful for exporting parent-child relationships for visualization
    /// (e.g. arXiv paper figures).
    pub fn lineage_tree(&self) -> LineageTree {
        lineage::build_lineage_tree(&self.morphons)
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
        let sum: f64 = self.critic_ports
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

        // Inject TD error as the reward signal — positive = better than expected,
        // negative = worse than expected. Scale to [0, 1] for the modulation channel.
        let scaled = (td_error * 0.3 + 0.5).clamp(0.0, 1.0);
        self.modulation.inject_reward(scaled);

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
