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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphon::{Morphon, Synapse};
    use crate::topology::Topology;
    use crate::types::HyperbolicPoint;

    fn make_morphon(id: MorphonId, activity_mean: f64) -> Morphon {
        let mut m = Morphon::new(id, HyperbolicPoint::origin(3));
        // Fill activity history to produce desired mean
        for _ in 0..100 {
            m.activity_history.push(activity_mean);
        }
        m
    }

    // === Synaptic Scaling ===

    #[test]
    fn synaptic_scaling_scales_weights_toward_setpoint() {
        let mut morphons = HashMap::new();
        // Morphon with activity well above setpoint (0.1)
        let mut m = make_morphon(1, 0.5);
        m.homeostatic_setpoint = 0.1;
        morphons.insert(1, m);

        let mut topo = Topology::new();
        topo.add_morphon(0);
        topo.add_morphon(1);
        topo.add_synapse(0, 1, Synapse::new(1.0));

        synaptic_scaling(&morphons, &mut topo);

        let (_, syn) = topo.synapse_between(0, 1).unwrap();
        // Activity (0.5) > setpoint (0.1), so scaling_factor = 0.1/0.5 = 0.2
        // Clamped to [0.5, 2.0] → 0.5
        assert!(
            syn.weight < 1.0,
            "weight should decrease when activity exceeds setpoint: {}",
            syn.weight
        );
    }

    #[test]
    fn synaptic_scaling_increases_weights_when_underactive() {
        let mut morphons = HashMap::new();
        let mut m = make_morphon(1, 0.05);
        m.homeostatic_setpoint = 0.1;
        morphons.insert(1, m);

        let mut topo = Topology::new();
        topo.add_morphon(0);
        topo.add_morphon(1);
        topo.add_synapse(0, 1, Synapse::new(0.5));

        synaptic_scaling(&morphons, &mut topo);

        let (_, syn) = topo.synapse_between(0, 1).unwrap();
        assert!(
            syn.weight > 0.5,
            "weight should increase when activity below setpoint: {}",
            syn.weight
        );
    }

    #[test]
    fn synaptic_scaling_no_change_at_setpoint() {
        let mut morphons = HashMap::new();
        let mut m = make_morphon(1, 0.1);
        m.homeostatic_setpoint = 0.1;
        morphons.insert(1, m);

        let mut topo = Topology::new();
        topo.add_morphon(0);
        topo.add_morphon(1);
        topo.add_synapse(0, 1, Synapse::new(0.5));

        synaptic_scaling(&morphons, &mut topo);

        let (_, syn) = topo.synapse_between(0, 1).unwrap();
        assert!(
            (syn.weight - 0.5).abs() < 0.01,
            "weight should not change when at setpoint: {}",
            syn.weight
        );
    }

    #[test]
    fn synaptic_scaling_preserves_relative_ratios() {
        let mut morphons = HashMap::new();
        let mut m = make_morphon(1, 0.5);
        m.homeostatic_setpoint = 0.1;
        morphons.insert(1, m);

        let mut topo = Topology::new();
        topo.add_morphon(0);
        topo.add_morphon(1);
        topo.add_morphon(2);
        topo.add_synapse(0, 1, Synapse::new(1.0));
        topo.add_synapse(2, 1, Synapse::new(0.5));

        synaptic_scaling(&morphons, &mut topo);

        let (_, s1) = topo.synapse_between(0, 1).unwrap();
        let (_, s2) = topo.synapse_between(2, 1).unwrap();
        let ratio = s1.weight / s2.weight;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "relative weight ratio should be preserved: got {ratio}"
        );
    }

    // === Migration Damping ===

    #[test]
    fn can_migrate_when_cooldown_zero() {
        let m = Morphon::new(1, HyperbolicPoint::origin(3));
        assert!(can_migrate(&m));
    }

    #[test]
    fn cannot_migrate_during_cooldown() {
        let mut m = Morphon::new(1, HyperbolicPoint::origin(3));
        m.migration_cooldown = 10.0;
        assert!(!can_migrate(&m));
    }

    #[test]
    fn apply_migration_cooldown_sets_timer() {
        let params = HomeostasisParams::default();
        let mut m = Morphon::new(1, HyperbolicPoint::origin(3));
        apply_migration_cooldown(&mut m, &params);
        assert_eq!(m.migration_cooldown, params.migration_cooldown_duration);
    }

    // === Migration Rate Modifier ===

    #[test]
    fn high_homeostasis_reduces_migration() {
        let low_h = migration_rate_modifier(0.1, 0.5);
        let high_h = migration_rate_modifier(0.9, 0.5);
        assert!(
            high_h < low_h,
            "high homeostasis should reduce migration rate: {high_h} vs {low_h}"
        );
    }

    #[test]
    fn high_error_increases_migration() {
        let low_e = migration_rate_modifier(0.5, 0.1);
        let high_e = migration_rate_modifier(0.5, 0.9);
        assert!(
            high_e > low_e,
            "high prediction error should increase migration rate: {high_e} vs {low_e}"
        );
    }

    #[test]
    fn migration_rate_modifier_clamped() {
        let rate = migration_rate_modifier(100.0, 100.0);
        assert!(rate >= 0.0 && rate <= 1.0);
        let rate = migration_rate_modifier(-1.0, -1.0);
        assert!(rate >= 0.0 && rate <= 1.0);
    }

    // === Checkpoint/Rollback ===

    #[test]
    fn create_checkpoint_captures_state() {
        let mut morphons = HashMap::new();
        let mut m1 = Morphon::new(1, HyperbolicPoint::origin(3));
        m1.prediction_error = 0.3;
        m1.potential = 0.5;
        let mut m2 = Morphon::new(2, HyperbolicPoint::origin(3));
        m2.prediction_error = 0.1;
        m2.potential = 0.2;
        morphons.insert(1, m1);
        morphons.insert(2, m2);

        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);
        topo.add_synapse(1, 2, Synapse::new(0.7));

        let cp = create_checkpoint(&[1, 2], &morphons, &topo);
        assert_eq!(cp.morphon_states.len(), 2);
        assert!((cp.avg_prediction_error - 0.2).abs() < 1e-10);
        assert_eq!(cp.synapse_states.len(), 1);
        assert!((cp.synapse_states[0].2 - 0.7).abs() < 1e-10);
    }

    #[test]
    fn should_rollback_detects_pe_increase() {
        let params = HomeostasisParams::default();
        let mut morphons = HashMap::new();
        let mut m = Morphon::new(1, HyperbolicPoint::origin(3));
        m.prediction_error = 0.1;
        morphons.insert(1, m);

        let cp = LocalCheckpoint {
            morphon_states: vec![(1, 0.1, 0.0)],
            avg_prediction_error: 0.1,
            synapse_states: vec![],
        };

        // No increase → no rollback
        assert!(!should_rollback(&cp, &[1], &morphons, &params));

        // Increase PE beyond threshold
        morphons.get_mut(&1).unwrap().prediction_error = 0.5;
        assert!(
            should_rollback(&cp, &[1], &morphons, &params),
            "should rollback when PE increases by > threshold ({})",
            params.rollback_pe_threshold
        );
    }

    #[test]
    fn rollback_restores_synapse_weights() {
        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);
        topo.add_synapse(1, 2, Synapse::new(0.7));

        let cp = LocalCheckpoint {
            morphon_states: vec![],
            avg_prediction_error: 0.0,
            synapse_states: vec![(1, 2, 0.7)],
        };

        // Modify weight
        let (ei, _) = topo.synapse_between(1, 2).unwrap();
        topo.synapse_mut(ei).unwrap().weight = 0.2;

        // Rollback
        rollback_synapses(&cp, &mut topo);

        let (_, syn) = topo.synapse_between(1, 2).unwrap();
        assert!(
            (syn.weight - 0.7).abs() < 1e-10,
            "weight should be restored to checkpoint value"
        );
    }

    #[test]
    fn should_rollback_false_for_empty_morphons() {
        let params = HomeostasisParams::default();
        let morphons: HashMap<MorphonId, Morphon> = HashMap::new();
        let cp = LocalCheckpoint {
            morphon_states: vec![],
            avg_prediction_error: 0.5,
            synapse_states: vec![],
        };
        assert!(!should_rollback(&cp, &[99], &morphons, &params));
    }
}
