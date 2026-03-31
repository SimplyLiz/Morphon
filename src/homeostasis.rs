//! Homeostatic Protection Mechanisms — solving the "Stable-Dynamic" problem.
//!
//! A system that constantly changes its structure risks losing what it has learned.
//! These mechanisms, inspired by biological neuroprotection, prevent that:
//!
//! A) Synaptic Scaling — proportional weight normalization preserving relative ratios
//! B) Inhibitory Inter-Cluster Morphons — prevent over-synchronization ("epilepsy")
//! C) Migration Damping — cooldown prevents topological instability
//! D) Checkpoint/Rollback — revert structural changes that increase prediction error

use crate::morphogenesis::Cluster;
use crate::morphon::Morphon;
use crate::topology::Topology;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Parameters for homeostatic mechanisms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeostasisParams {
    /// How often (in steps) to run synaptic scaling.
    pub scaling_interval: u64,
    /// Inter-cluster inhibition strength.
    pub inhibition_strength: f64,
    /// Correlation threshold above which inter-cluster inhibition activates.
    pub inhibition_correlation_threshold: f64,
    /// Migration cooldown duration after a migration event.
    pub migration_cooldown_duration: f64,
    /// Prediction error increase threshold that triggers checkpoint rollback.
    /// Lowered from 0.5 to 0.2 so rollback actually protects against
    /// destabilizing structural changes.
    pub rollback_pe_threshold: f64,
}

impl Default for HomeostasisParams {
    fn default() -> Self {
        Self {
            scaling_interval: 50,
            inhibition_strength: 0.3,
            inhibition_correlation_threshold: 0.9,
            migration_cooldown_duration: 20.0,
            rollback_pe_threshold: 0.2,
        }
    }
}

// === A) Synaptic Scaling ===

/// Apply synaptic scaling to maintain target firing rates.
///
/// For each Morphon, if actual firing rate deviates from the homeostatic setpoint,
/// all incoming synaptic weights are proportionally scaled. This preserves the
/// relative weight ratios (i.e., what was learned) while stabilizing overall activity.
pub fn synaptic_scaling(
    morphons: &HashMap<MorphonId, Morphon>,
    topology: &mut Topology,
) {
    for morphon in morphons.values() {
        let actual_rate = morphon.activity_history.mean();
        if actual_rate < 1e-6 {
            continue; // avoid division by zero
        }

        let scaling_factor = morphon.homeostatic_setpoint / actual_rate;
        // Clamp to prevent extreme scaling
        let scaling_factor = scaling_factor.clamp(0.5, 2.0);

        if (scaling_factor - 1.0).abs() < 0.01 {
            continue; // already close enough
        }

        let incoming = topology.incoming_synapses_mut(morphon.id);
        for (_, edge_idx) in incoming {
            if let Some(syn) = topology.synapse_mut(edge_idx) {
                syn.weight *= scaling_factor;
            }
        }
    }
}

// === B) Inhibitory Inter-Cluster Morphons ===

/// Activate inhibitory inter-cluster Morphons proportional to cluster synchrony.
///
/// Per concept doc section 3.7B: between each cluster pair, inhibitory Morphons
/// (type: Modulatory, with negative weights) exist in the topology. These fire
/// proportionally to the activity correlation of the connected clusters. When two
/// clusters become too synchronous, inhibition rises, preventing over-synchronization
/// ("system epilepsy").
///
/// This function drives the inhibitory morphons' firing by injecting input
/// proportional to the synchrony between the cluster pairs they bridge.
/// The actual inhibition is delivered through the negative-weight synapses
/// in the topology during normal spike propagation.
///
/// Returns the number of inhibition events (inhibitory morphons activated).
pub fn inter_cluster_inhibition(
    morphons: &mut HashMap<MorphonId, Morphon>,
    clusters: &HashMap<ClusterId, Cluster>,
    params: &HomeostasisParams,
) -> usize {
    let cluster_list: Vec<&Cluster> = clusters.values().collect();
    let mut inhibitions = 0;

    // Build a set of all inhibitory morphon IDs across all clusters
    let mut inhibitory_ids: std::collections::HashSet<MorphonId> = std::collections::HashSet::new();
    for cluster in clusters.values() {
        for &inh_id in &cluster.inhibitory_morphons {
            inhibitory_ids.insert(inh_id);
        }
    }

    for i in 0..cluster_list.len() {
        for j in (i + 1)..cluster_list.len() {
            let a = cluster_list[i];
            let b = cluster_list[j];

            // Compute activity correlation between clusters
            let a_rate = cluster_mean_activity(a, morphons);
            let b_rate = cluster_mean_activity(b, morphons);

            // Simple synchrony proxy: both clusters have similar high activity
            if a_rate > 0.3 && b_rate > 0.3 {
                let sync = 1.0 - (a_rate - b_rate).abs();
                if sync > params.inhibition_correlation_threshold {
                    // Find shared inhibitory morphons between these two clusters
                    let a_inh: std::collections::HashSet<MorphonId> =
                        a.inhibitory_morphons.iter().copied().collect();
                    let shared_inh: Vec<MorphonId> = b
                        .inhibitory_morphons
                        .iter()
                        .filter(|id| a_inh.contains(id))
                        .copied()
                        .collect();

                    if !shared_inh.is_empty() {
                        // Drive inhibitory morphons proportionally to synchrony
                        let drive = sync * params.inhibition_strength;
                        for &inh_id in &shared_inh {
                            if let Some(inh_m) = morphons.get_mut(&inh_id) {
                                inh_m.input_accumulator += drive;
                            }
                        }
                        inhibitions += 1;
                    } else {
                        // Fallback: if no inhibitory morphons exist yet (e.g. legacy
                        // clusters created before this mechanism), apply direct
                        // potential reduction as before.
                        for &mid in &a.members {
                            if let Some(m) = morphons.get_mut(&mid) {
                                m.potential -= params.inhibition_strength;
                            }
                        }
                        for &mid in &b.members {
                            if let Some(m) = morphons.get_mut(&mid) {
                                m.potential -= params.inhibition_strength;
                            }
                        }
                        inhibitions += 1;
                    }
                }
            }
        }
    }

    inhibitions
}

/// Compute mean firing rate of a cluster's members.
fn cluster_mean_activity(cluster: &Cluster, morphons: &HashMap<MorphonId, Morphon>) -> f64 {
    let rates: Vec<f64> = cluster
        .members
        .iter()
        .filter_map(|id| morphons.get(id))
        .map(|m| m.activity_history.mean())
        .collect();

    if rates.is_empty() {
        return 0.0;
    }
    rates.iter().sum::<f64>() / rates.len() as f64
}

// === C) Migration Damping ===

/// Check if a Morphon is allowed to migrate (cooldown expired).
pub fn can_migrate(morphon: &Morphon) -> bool {
    morphon.migration_cooldown <= 0.0
}

/// Set migration cooldown after a successful migration.
pub fn apply_migration_cooldown(morphon: &mut Morphon, params: &HomeostasisParams) {
    morphon.migration_cooldown = params.migration_cooldown_duration;
}

/// Compute the system-wide migration rate based on homeostasis level.
/// High homeostasis (stability) → low migration rate.
/// High prediction error → more migration allowed.
pub fn migration_rate_modifier(homeostasis_level: f64, avg_prediction_error: f64) -> f64 {
    let stability_brake = 1.0 - homeostasis_level.clamp(0.0, 1.0) * 0.8;
    let error_boost = avg_prediction_error.clamp(0.0, 1.0);
    (stability_brake * (0.2 + 0.8 * error_boost)).clamp(0.0, 1.0)
}

// === D) Checkpoint/Rollback ===

/// A local checkpoint of a region's state before structural changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalCheckpoint {
    /// Morphon states at checkpoint time.
    pub morphon_states: Vec<(MorphonId, f64, f64)>, // (id, prediction_error, potential)
    /// Average prediction error at checkpoint time.
    pub avg_prediction_error: f64,
    /// Synapse states: (from, to, weight).
    pub synapse_states: Vec<(MorphonId, MorphonId, f64)>,
}

/// Create a checkpoint of the local area around a set of Morphon IDs.
pub fn create_checkpoint(
    morphon_ids: &[MorphonId],
    morphons: &HashMap<MorphonId, Morphon>,
    topology: &Topology,
) -> LocalCheckpoint {
    let morphon_states: Vec<_> = morphon_ids
        .iter()
        .filter_map(|id| morphons.get(id))
        .map(|m| (m.id, m.prediction_error, m.potential))
        .collect();

    let avg_pe = if morphon_states.is_empty() {
        0.0
    } else {
        morphon_states.iter().map(|(_, pe, _)| pe).sum::<f64>() / morphon_states.len() as f64
    };

    let mut synapse_states = Vec::new();
    for &id in morphon_ids {
        for (src, syn) in topology.incoming(id) {
            synapse_states.push((src, id, syn.weight));
        }
    }

    LocalCheckpoint {
        morphon_states,
        avg_prediction_error: avg_pe,
        synapse_states,
    }
}

/// Check if prediction error increased significantly after a structural change.
/// Returns true if rollback is recommended.
pub fn should_rollback(
    checkpoint: &LocalCheckpoint,
    morphon_ids: &[MorphonId],
    morphons: &HashMap<MorphonId, Morphon>,
    params: &HomeostasisParams,
) -> bool {
    let current_states: Vec<_> = morphon_ids
        .iter()
        .filter_map(|id| morphons.get(id))
        .map(|m| m.prediction_error)
        .collect();

    if current_states.is_empty() {
        return false;
    }

    let current_avg = current_states.iter().sum::<f64>() / current_states.len() as f64;
    let increase = current_avg - checkpoint.avg_prediction_error;

    increase > params.rollback_pe_threshold
}

/// Rollback synapse weights to checkpoint state.
pub fn rollback_synapses(
    checkpoint: &LocalCheckpoint,
    topology: &mut Topology,
) {
    for &(from, to, weight) in &checkpoint.synapse_states {
        if let Some((ei, _)) = topology.synapse_between(from, to) {
            if let Some(syn) = topology.synapse_mut(ei) {
                syn.weight = weight;
            }
        }
    }
}
