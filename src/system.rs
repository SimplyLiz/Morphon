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
use crate::morphon::Morphon;
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

    /// Next available Morphon ID.
    pub(crate) next_morphon_id: MorphonId,
    /// Next available Cluster ID.
    pub(crate) next_cluster_id: ClusterId,
    /// Current simulation step.
    pub(crate) step_count: u64,

    /// Learning pipeline diagnostics (updated every step).
    pub(crate) diag: Diagnostics,
}

impl System {
    /// Create a new MI System by running its developmental program.
    pub fn new(config: SystemConfig) -> Self {
        let mut rng = rand::rng();

        let (morphons, topology, next_id) =
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

        System {
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
            next_morphon_id: next_id,
            next_cluster_id: 0,
            step_count: 0,
            diag: Diagnostics::default(),
        }
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

        // 3. Spike-timing eligibility: update eligibility traces at the moment
        //    of spike delivery for precise pre->post timing information.
        //    This uses the actual spike arrival (not just the coarse "fired" flag)
        //    to provide STDP-like precision to the three-factor learning rule.
        for spike in &delivered {
            let post_fired = self.morphons.get(&spike.target).map_or(false, |m| m.fired);
            if let Some((ei, _)) = self.topology.synapse_between(spike.source, spike.target) {
                if let Some(synapse) = self.topology.synapse_mut(ei) {
                    // Spike arrival is direct evidence of pre-synaptic firing.
                    // Uses same balanced LTD as hebbian_coincidence (pre=true case).
                    let h = if post_fired { 1.0 } else { -0.06 };
                    synapse.eligibility +=
                        (-synapse.eligibility / self.config.learning.tau_eligibility + h) * dt;
                    synapse.eligibility = synapse.eligibility.clamp(-1.0, 1.0);

                    // Tag on strong coincidence (spike arrived AND post fired)
                    if h > self.config.learning.tag_threshold && !synapse.consolidated {
                        synapse.tag = 1.0;
                        synapse.tag_strength = h;
                    }
                }
            }
        }
        let spikes_delivered = delivered.len();

        // 4. Update all Morphon states (integrate input, fire/not-fire)
        #[cfg(feature = "parallel")]
        self.morphons.par_iter_mut().for_each(|(_, m)| m.step(dt));
        #[cfg(not(feature = "parallel"))]
        self.morphons.values_mut().for_each(|m| m.step(dt));
        let morphon_ids: Vec<MorphonId> = self.morphons.keys().copied().collect();

        // === MEDIUM PATH ===
        let mut captures_this_step = 0_u64;
        if tick.medium {
            // Update eligibility traces and apply receptor-gated weight changes
            for &id in &morphon_ids {
                let (post_fired, post_receptors) = self.morphons.get(&id)
                    .map(|m| (m.fired, m.receptors.clone()))
                    .unwrap_or_default();
                let incoming = self.topology.incoming_synapses_mut(id);

                for (pre_id, edge_idx) in incoming {
                    let pre_fired = self.morphons.get(&pre_id).map_or(false, |m| m.fired);

                    if let Some(synapse) = self.topology.synapse_mut(edge_idx) {
                        learning::update_eligibility(
                            synapse,
                            pre_fired,
                            post_fired,
                            &self.config.learning,
                            dt,
                        );

                        let plasticity = self.modulation.plasticity_rate();
                        let captured = learning::apply_weight_update(
                            synapse,
                            &self.modulation,
                            &self.config.learning,
                            plasticity,
                            &post_receptors,
                        );
                        if captured {
                            captures_this_step += 1;
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

    /// Read output from motor Morphons via stable port mapping.
    ///
    /// Returns one value per output port. If there are more motor Morphons
    /// than needed, groups are averaged.
    pub fn read_output(&self) -> Vec<f64> {
        self.output_ports
            .iter()
            .filter_map(|id| self.morphons.get(id))
            .map(|m| m.potential)
            .collect()
    }

    /// Number of input ports (sensory Morphons available for external input).
    pub fn input_size(&self) -> usize {
        self.input_ports.len()
    }

    /// Number of output ports (motor Morphons available for external output).
    pub fn output_size(&self) -> usize {
        self.output_ports.len()
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
    /// Instead of global broadcast, this directly boosts the eligibility traces
    /// of all incoming synapses to the specified motor morphon. This provides
    /// output-specific credit assignment within the three-factor framework:
    /// the modulation is still local (eligibility × reward), but spatially targeted.
    ///
    /// Analogous to how dopaminergic projections target specific brain regions,
    /// not the whole cortex uniformly.
    pub fn inject_reward_at(&mut self, output_index: usize, strength: f64) {
        if let Some(&id) = self.output_ports.get(output_index) {
            // Boost eligibility traces on all incoming synapses to this motor morphon
            let incoming = self.topology.incoming_synapses_mut(id);
            for (_, edge_idx) in incoming {
                if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                    syn.eligibility += strength;
                    syn.eligibility = syn.eligibility.clamp(-1.0, 1.0);
                }
            }
        }
        // Also inject globally (but weaker) so interior paths benefit too
        self.modulation.inject_reward(strength * 0.3);
    }

    /// Inject a targeted inhibition at a specific output port's morphon.
    ///
    /// Reduces eligibility traces on incoming synapses to the specified motor morphon,
    /// making those paths less likely to be strengthened by future reward.
    pub fn inject_inhibition_at(&mut self, output_index: usize, strength: f64) {
        if let Some(&id) = self.output_ports.get(output_index) {
            let incoming = self.topology.incoming_synapses_mut(id);
            for (_, edge_idx) in incoming {
                if let Some(syn) = self.topology.synapse_mut(edge_idx) {
                    syn.eligibility -= strength;
                    syn.eligibility = syn.eligibility.clamp(-1.0, 1.0);
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

    /// Average prediction error across all Morphons.
    fn avg_prediction_error(&self) -> f64 {
        if self.morphons.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.morphons.values().map(|m| m.prediction_error).sum();
        sum / self.morphons.len() as f64
    }
}
