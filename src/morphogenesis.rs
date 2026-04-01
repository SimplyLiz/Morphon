//! Morphogenesis — topology changes at runtime.
//!
//! Seven mechanisms operating on different timescales:
//! A) Synaptic plasticity (fast, ~ms) — handled in learning.rs
//! B) Synaptogenesis/Pruning (medium, ~s to min)
//! C) Cell division / Mitosis (slow, ~min to h)
//! D) Differentiation (slow, ~min to h)
//! E) Fusion / Autonomy loss (slow, ~h to days)
//! F) Migration (slow, ~min to h)
//! G) Apoptosis / Programmed cell death (slow, ~h to days)

use crate::learning::{self, LearningParams};
use crate::morphon::{Morphon, Synapse};
use crate::topology::Topology;
use crate::types::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Parameters controlling morphogenesis behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphogenesisParams {
    /// Correlation threshold for synaptogenesis (new connection growth).
    pub synaptogenesis_threshold: f64,
    /// Minimum age for a synapse to be pruning-eligible.
    pub pruning_min_age: u64,
    /// Division pressure threshold to trigger mitosis.
    pub division_threshold: f64,
    /// Minimum energy to allow division.
    pub division_min_energy: f64,
    /// Correlation threshold for cluster fusion.
    /// Lowered from 0.95 to 0.75 to allow functional cluster formation earlier.
    pub fusion_correlation_threshold: f64,
    /// Minimum cluster size for fusion.
    pub fusion_min_size: usize,
    /// Migration step size.
    pub migration_rate: f64,
    /// Apoptosis — minimum age before eligible.
    pub apoptosis_min_age: u64,
    /// Apoptosis — energy threshold below which death is possible.
    pub apoptosis_energy_threshold: f64,
    /// Maximum number of morphons (prevent unbounded growth).
    pub max_morphons: usize,
}

impl Default for MorphogenesisParams {
    fn default() -> Self {
        Self {
            synaptogenesis_threshold: 0.6,
            pruning_min_age: 100,
            division_threshold: 1.0,
            division_min_energy: 0.3,
            fusion_correlation_threshold: 0.75,
            fusion_min_size: 3,
            migration_rate: 0.05,
            apoptosis_min_age: 1000,
            apoptosis_energy_threshold: 0.1,
            max_morphons: 10_000,
        }
    }
}

/// Result of a morphogenesis step — describes what changed.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MorphogenesisReport {
    pub synapses_created: usize,
    pub synapses_pruned: usize,
    pub morphons_born: usize,
    pub morphons_died: usize,
    pub differentiations: usize,
    pub fusions: usize,
    pub defusions: usize,
    pub migrations: usize,
}

/// Run synaptogenesis — create new connections between correlated Morphons
/// that don't yet have a direct connection.
pub fn synaptogenesis(
    morphons: &HashMap<MorphonId, Morphon>,
    topology: &mut Topology,
    _params: &MorphogenesisParams,
    rng: &mut impl Rng,
) -> usize {
    let mut created = 0;
    let ids: Vec<MorphonId> = morphons.keys().copied().collect();

    // Check pairs of morphons that both fired recently
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            let a = &morphons[&ids[i]];
            let b = &morphons[&ids[j]];

            // Both must have fired recently (high mean activity)
            if a.activity_history.mean() < 0.3 || b.activity_history.mean() < 0.3 {
                continue;
            }

            // Must be in spatial proximity
            let distance = a.position.distance(&b.position);
            if distance > 2.0 {
                continue;
            }

            // Don't create if connection already exists
            if topology.has_connection(a.id, b.id) || topology.has_connection(b.id, a.id) {
                continue;
            }

            // Respect cell type hierarchy:
            // - Don't create connections INTO Sensory (they're input-only)
            // - Don't create connections OUT OF Motor (they're output-only)
            let (from, to) = (a, b);
            let valid_direction = to.cell_type != CellType::Sensory
                && from.cell_type != CellType::Motor;
            if !valid_direction { continue; }

            let prob = (1.0 - distance / 2.0) * 0.1;
            if rng.random_range(0.0..1.0) < prob {
                let weight = rng.random_range(-0.5..0.5);
                topology.add_synapse(from.id, to.id, Synapse::new(weight));
                created += 1;
            }
        }
    }

    created
}

/// Run pruning — remove weak, unused synapses.
pub fn pruning(
    topology: &mut Topology,
    learning_params: &LearningParams,
) -> usize {
    let edges_to_remove: Vec<_> = topology
        .all_edges()
        .into_iter()
        .filter(|(_, _, ei)| {
            if let Some((_, syn)) = topology.graph.edge_endpoints(*ei).map(|(s, _t)| {
                (s, &topology.graph[*ei])
            }) {
                learning::should_prune(syn, learning_params)
            } else {
                false
            }
        })
        .map(|(_, _, ei)| ei)
        .collect();

    let count = edges_to_remove.len();
    for ei in edges_to_remove {
        topology.remove_synapse(ei);
    }
    count
}

/// Run cell division (mitosis) — overloaded Morphons split.
pub fn division(
    morphons: &mut HashMap<MorphonId, Morphon>,
    topology: &mut Topology,
    next_id: &mut MorphonId,
    params: &MorphogenesisParams,
    rng: &mut impl Rng,
) -> usize {
    if morphons.len() >= params.max_morphons {
        return 0;
    }

    let candidates: Vec<MorphonId> = morphons
        .values()
        .filter(|m| m.should_divide(params.division_threshold))
        .map(|m| m.id)
        .collect();

    let mut born = 0;
    for parent_id in candidates {
        if morphons.len() >= params.max_morphons {
            break;
        }

        let child_id = *next_id;
        *next_id += 1;

        // Create child via mitosis
        let child = morphons[&parent_id].divide(child_id, rng);

        // Parent loses half its energy and resets division pressure
        if let Some(parent) = morphons.get_mut(&parent_id) {
            parent.energy *= 0.5;
            parent.division_pressure = 0.0;
        }

        // Add child to topology and duplicate ~50% of parent's connections
        topology.add_morphon(child_id);
        topology.duplicate_connections(parent_id, child_id, rng);

        morphons.insert(child_id, child);
        born += 1;
    }

    born
}

/// Run differentiation — Morphons specialize based on their activity patterns.
pub fn differentiation(
    morphons: &mut HashMap<MorphonId, Morphon>,
    _topology: &Topology,
) -> usize {
    let mut count = 0;

    for morphon in morphons.values_mut() {
        if morphon.cell_type != CellType::Stem {
            continue;
        }
        if morphon.age < 200 {
            continue; // need substantial activity history before differentiating
        }
        if morphon.activity_history.len() < 50 {
            continue; // not enough data
        }

        let mean_activity = morphon.activity_history.mean();
        let variance = morphon.activity_history.variance();

        // Most stem cells should stay stem (become Associative by default).
        // Only differentiate under clear activity signatures.
        let target = if mean_activity > 0.4 && variance < 0.1 {
            CellType::Associative // high consistent → Associative (the workhorse type)
        } else if mean_activity > 0.3 && variance > 0.2 {
            CellType::Associative // high variable → also Associative
        } else {
            continue; // stay stem — don't eagerly classify as Modulatory
        };

        if morphon.differentiate(target) {
            count += 1;
        }
    }

    count
}

/// Run dedifferentiation — stressed Morphons return to flexible state.
pub fn dedifferentiation(
    morphons: &mut HashMap<MorphonId, Morphon>,
    arousal_level: f64,
) -> usize {
    let mut count = 0;

    for morphon in morphons.values_mut() {
        if morphon.cell_type == CellType::Stem {
            continue;
        }
        // Very high prediction error + very high arousal → dedifferentiate
        // Requires extreme stress to override developmental differentiation
        if morphon.desire > 0.9 && arousal_level > 0.8 && morphon.differentiation_level < 0.5 {
            morphon.dedifferentiate();
            count += 1;
        }
    }

    count
}

/// Cluster state for fusion tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    pub id: ClusterId,
    pub members: Vec<MorphonId>,
    pub shared_threshold: f64,
    /// Inhibitory Morphons (type: Modulatory, negative-weight connections) created
    /// between this cluster and its neighbors to prevent over-synchronization.
    /// See concept doc section 3.7B.
    pub inhibitory_morphons: Vec<MorphonId>,
}

/// Run fusion — highly correlated Morphon groups merge into clusters.
///
/// Per concept doc section 3.7B, fusion is only allowed when prediction error
/// drops (not just when firing is correlated). This prevents meaningless
/// synchronization from triggering structural changes.
pub fn fusion(
    morphons: &mut HashMap<MorphonId, Morphon>,
    clusters: &mut HashMap<ClusterId, Cluster>,
    next_cluster_id: &mut ClusterId,
    next_morphon_id: &mut MorphonId,
    topology: &mut Topology,
    params: &MorphogenesisParams,
) -> usize {
    // Find groups of tightly connected, correlated morphons
    // Simple heuristic: look at groups of neighbors that all fire together
    let mut fused = 0;

    let ids: Vec<MorphonId> = morphons
        .values()
        .filter(|m| m.fused_with.is_none() && m.activity_history.mean() > 0.3)
        .map(|m| m.id)
        .collect();

    for &id in &ids {
        // Skip if this morphon was already fused by an earlier iteration
        if morphons.get(&id).map_or(true, |m| m.fused_with.is_some()) {
            continue;
        }

        let neighbors = topology.outgoing(id);
        let correlated: Vec<MorphonId> = neighbors
            .iter()
            .filter(|(nid, _)| {
                if let Some(n) = morphons.get(nid) {
                    n.fused_with.is_none()
                        && n.activity_history.mean() > 0.3
                        && n.fired == morphons[&id].fired // simple correlation proxy
                } else {
                    false
                }
            })
            .map(|(nid, _)| *nid)
            .collect();

        if correlated.len() + 1 >= params.fusion_min_size {
            let mut members = vec![id];
            members.extend(correlated.iter());

            // === Prediction error gate (section 3.7B) ===
            // Fusion is only allowed if the candidate group shows prediction error
            // reduction — i.e. their mean prediction error is below their mean desire
            // (long-term PE average). Correlated firing alone is not sufficient.
            let (pe_sum, desire_sum, count) = members
                .iter()
                .filter_map(|mid| morphons.get(mid))
                .fold((0.0, 0.0, 0usize), |(pe, d, c), m| {
                    (pe + m.prediction_error, d + m.desire, c + 1)
                });

            if count == 0 {
                continue;
            }
            let mean_pe = pe_sum / count as f64;
            let mean_desire = desire_sum / count as f64;

            // Prediction error must be trending down (current PE < long-term average)
            if mean_pe >= mean_desire && mean_desire > 0.01 {
                continue; // no evidence of prediction error reduction — deny fusion
            }

            let cluster_id = *next_cluster_id;
            *next_cluster_id += 1;

            // Calculate shared threshold
            let avg_threshold: f64 = members
                .iter()
                .filter_map(|mid| morphons.get(mid))
                .map(|m| m.threshold)
                .sum::<f64>()
                / members.len() as f64;

            // Update morphon states
            for &mid in &members {
                if let Some(m) = morphons.get_mut(&mid) {
                    m.fused_with = Some(cluster_id);
                    m.autonomy = 0.5; // partial fusion
                    m.threshold = avg_threshold;
                }
            }

            // === Create inhibitory morphons to neighboring clusters (section 3.7B) ===
            let inhibitory_morphons = create_inhibitory_morphons_for_cluster(
                cluster_id,
                &members,
                morphons,
                clusters,
                topology,
                next_morphon_id,
                params.max_morphons,
            );

            clusters.insert(
                cluster_id,
                Cluster {
                    id: cluster_id,
                    members,
                    shared_threshold: avg_threshold,
                    inhibitory_morphons,
                },
            );
            fused += 1;
        }
    }

    fused
}

/// Create inhibitory Morphon nodes between a newly-formed cluster and each
/// existing neighboring cluster. Each inhibitory morphon is CellType::Modulatory
/// with negative-weight connections to members of both clusters.
fn create_inhibitory_morphons_for_cluster(
    new_cluster_id: ClusterId,
    new_members: &[MorphonId],
    morphons: &mut HashMap<MorphonId, Morphon>,
    clusters: &mut HashMap<ClusterId, Cluster>,
    topology: &mut Topology,
    next_morphon_id: &mut MorphonId,
    max_morphons: usize,
) -> Vec<MorphonId> {
    let mut created = Vec::new();
    let existing_cluster_ids: Vec<ClusterId> = clusters.keys().copied().collect();

    for &other_cid in &existing_cluster_ids {
        if other_cid == new_cluster_id {
            continue;
        }

        let other_members: Vec<MorphonId> = match clusters.get(&other_cid) {
            Some(c) => c.members.clone(),
            None => continue,
        };

        // Respect morphon cap — don't leak morphons past max_morphons
        if morphons.len() >= max_morphons {
            break;
        }

        // Create one inhibitory morphon for this cluster pair
        let inh_id = *next_morphon_id;
        *next_morphon_id += 1;

        // Compute a midpoint position between the two clusters (average of all members)
        let all_member_ids: Vec<MorphonId> = new_members
            .iter()
            .chain(other_members.iter())
            .copied()
            .collect();
        let dim = morphons
            .values()
            .next()
            .map(|m| m.position.coords.len())
            .unwrap_or(3);
        let mut avg_coords = vec![0.0; dim];
        let mut pos_count = 0;
        for &mid in &all_member_ids {
            if let Some(m) = morphons.get(&mid) {
                for (i, c) in m.position.coords.iter().enumerate() {
                    if i < avg_coords.len() {
                        avg_coords[i] += c;
                    }
                }
                pos_count += 1;
            }
        }
        if pos_count > 0 {
            for c in &mut avg_coords {
                *c /= pos_count as f64;
            }
        }
        // Clamp inside Poincare ball
        let norm: f64 = avg_coords.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.9 {
            let scale = 0.9 / norm;
            for c in &mut avg_coords {
                *c *= scale;
            }
        }

        let position = crate::types::HyperbolicPoint {
            coords: avg_coords,
            curvature: 1.0,
        };

        let mut inh_morphon = crate::morphon::Morphon::new(inh_id, position);
        inh_morphon.cell_type = CellType::Modulatory;
        inh_morphon.activation_fn = crate::types::ActivationFn::Oscillatory;
        inh_morphon.receptors = crate::types::default_receptors(CellType::Modulatory);
        inh_morphon.differentiation_level = 1.0; // terminally differentiated

        // Add to topology
        topology.add_morphon(inh_id);

        // Create negative-weight connections from the inhibitory morphon to all members
        // of both clusters
        let inhibition_weight = -0.3;
        for &mid in new_members.iter().chain(other_members.iter()) {
            topology.add_synapse(
                inh_id,
                mid,
                crate::morphon::Synapse::new(inhibition_weight),
            );
        }

        morphons.insert(inh_id, inh_morphon);
        created.push(inh_id);

        // Also register this inhibitory morphon in the other cluster
        if let Some(other_cluster) = clusters.get_mut(&other_cid) {
            other_cluster.inhibitory_morphons.push(inh_id);
        }
    }

    created
}

/// Run defusion — clusters under stress break apart.
///
/// When a cluster defuses, its inhibitory morphons are removed from the
/// topology and morphon map, and references are cleaned from other clusters.
pub fn defusion(
    morphons: &mut HashMap<MorphonId, Morphon>,
    clusters: &mut HashMap<ClusterId, Cluster>,
    topology: &mut Topology,
) -> usize {
    let mut defused = 0;
    let mut clusters_to_remove = Vec::new();

    for (cluster_id, cluster) in clusters.iter() {
        // Check if members have diverging prediction errors
        let errors: Vec<f64> = cluster
            .members
            .iter()
            .filter_map(|mid| morphons.get(mid))
            .map(|m| m.prediction_error)
            .collect();

        if errors.is_empty() {
            clusters_to_remove.push(*cluster_id);
            continue;
        }

        let mean_error: f64 = errors.iter().sum::<f64>() / errors.len() as f64;
        let error_variance: f64 = errors
            .iter()
            .map(|e| (e - mean_error).powi(2))
            .sum::<f64>()
            / errors.len() as f64;

        // High variance in prediction errors → defuse
        if error_variance > 0.5 {
            for &mid in &cluster.members {
                if let Some(m) = morphons.get_mut(&mid) {
                    m.fused_with = None;
                    m.autonomy = 1.0;
                }
            }
            clusters_to_remove.push(*cluster_id);
            defused += 1;
        }
    }

    // Collect all inhibitory morphon IDs to remove from defusing clusters
    let mut inhibitory_to_remove: Vec<MorphonId> = Vec::new();
    for &cid in &clusters_to_remove {
        if let Some(cluster) = clusters.get(&cid) {
            inhibitory_to_remove.extend(cluster.inhibitory_morphons.iter().copied());
        }
    }

    // Remove inhibitory morphons from topology and morphon map
    for &inh_id in &inhibitory_to_remove {
        morphons.remove(&inh_id);
        topology.remove_morphon(inh_id);
    }

    // Clean references to removed inhibitory morphons from surviving clusters
    for cluster in clusters.values_mut() {
        cluster
            .inhibitory_morphons
            .retain(|id| !inhibitory_to_remove.contains(id));
    }

    for id in clusters_to_remove {
        clusters.remove(&id);
    }

    defused
}

/// Run migration — Morphons with high desire move in hyperbolic information space.
///
/// Uses the logarithmic map to compute tangent vectors towards neighbors with
/// lower prediction error, then the exponential map to project onto the manifold.
/// Migration is damped by cooldowns and the homeostasis channel.
pub fn migration(
    morphons: &mut HashMap<MorphonId, Morphon>,
    topology: &Topology,
    params: &MorphogenesisParams,
    homeostasis_level: f64,
) -> usize {
    let mut migrated = 0;

    // System-wide migration rate modifier (high homeostasis → less migration)
    let system_migration_mod = crate::homeostasis::migration_rate_modifier(
        homeostasis_level,
        morphons.values().map(|m| m.prediction_error).sum::<f64>()
            / morphons.len().max(1) as f64,
    );

    // Collect positions and prediction errors of all morphons
    let positions: HashMap<MorphonId, (Position, f64)> = morphons
        .values()
        .map(|m| (m.id, (m.position.clone(), m.prediction_error)))
        .collect();

    for morphon in morphons.values_mut() {
        let dim = morphon.position.coords.len();

        // --- Radial anchoring (always applied, no cooldown) ---
        // Each cell type has a target radius; push morphons back when they drift.
        let target_radius = match morphon.cell_type {
            CellType::Sensory => 0.6,
            CellType::Motor => 0.6,
            CellType::Associative => 0.35,
            CellType::Modulatory => 0.5,
            _ => 0.3, // Stem, Fused
        };
        let current_radius = morphon.position.specificity();
        let radial_error = target_radius - current_radius;
        // Only correct if meaningfully off-target (> 10% of target)
        if radial_error.abs() > target_radius * 0.1 {
            let radial_strength = 0.15 * radial_error;
            let mut radial_tangent = vec![0.0; dim];
            if current_radius > 1e-6 {
                for (i, t) in radial_tangent.iter_mut().enumerate() {
                    *t = morphon.position.coords[i] / current_radius * radial_strength;
                }
            } else {
                // At exact origin — use morphon id to pick a unique outward direction
                let phi = (morphon.id as f64) * 2.399963;
                let cos_theta = 1.0 - 2.0 * (((morphon.id as f64) * 0.618034) % 1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                if dim >= 1 { radial_tangent[0] = phi.cos() * sin_theta * radial_strength; }
                if dim >= 2 { radial_tangent[1] = phi.sin() * sin_theta * radial_strength; }
                if dim >= 3 { radial_tangent[2] = cos_theta * radial_strength; }
            }
            morphon.position = morphon.position.exp_map(&radial_tangent);
            migrated += 1;
        }

        // --- Neighbor-chasing migration (gated by desire, cooldown, etc.) ---
        if morphon.desire < 0.3 {
            continue;
        }
        if morphon.fused_with.is_some() && morphon.autonomy < 0.5 {
            continue;
        }
        if !crate::homeostasis::can_migrate(morphon) {
            continue;
        }

        let neighbors = topology.outgoing(morphon.id);
        if neighbors.is_empty() {
            continue;
        }

        let mut tangent = vec![0.0; dim];
        let mut count = 0;

        for (nid, _) in &neighbors {
            if let Some((pos, pe)) = positions.get(nid) {
                if *pe < morphon.prediction_error {
                    let log_v = morphon.position.log_map(pos);
                    for (i, t) in tangent.iter_mut().enumerate() {
                        if i < log_v.len() {
                            *t += log_v[i];
                        }
                    }
                    count += 1;
                }
            }
        }

        if count > 0 {
            let scale = params.migration_rate * morphon.desire * system_migration_mod;
            // Project tangent onto tangential component only (remove radial part)
            // so neighbor-chasing moves laterally, not inward
            let current_radius = morphon.position.specificity();
            if current_radius > 1e-6 {
                let radial_dot: f64 = tangent.iter().enumerate()
                    .map(|(i, t)| t * morphon.position.coords[i] / current_radius)
                    .sum();
                for (i, t) in tangent.iter_mut().enumerate() {
                    *t -= radial_dot * morphon.position.coords[i] / current_radius;
                }
            }
            for t in &mut tangent {
                *t = *t / count as f64 * scale;
            }
            morphon.position = morphon.position.exp_map(&tangent);
            morphon.migration_cooldown = 20.0;
            migrated += 1;
        }
    }

    migrated
}

/// Run apoptosis — remove useless Morphons.
pub fn apoptosis(
    morphons: &mut HashMap<MorphonId, Morphon>,
    topology: &mut Topology,
    params: &MorphogenesisParams,
) -> usize {
    let to_remove: Vec<MorphonId> = morphons
        .values()
        .filter(|m| {
            m.age > params.apoptosis_min_age
                && m.energy < params.apoptosis_energy_threshold
                && m.activity_history.mean() < 0.01
                && m.fused_with.is_none()
                && topology.degree(m.id) < 3 // poorly connected
        })
        .map(|m| m.id)
        .collect();

    let count = to_remove.len();
    for id in to_remove {
        morphons.remove(&id);
        topology.remove_morphon(id);
    }
    count
}

/// Run morphogenesis mechanisms based on scheduler tick.
///
/// Uses the dual-clock architecture: slow-path processes (synaptogenesis, pruning,
/// migration) run on `slow` ticks; glacial-path processes (division, differentiation,
/// fusion, apoptosis) run on `glacial` ticks.
pub fn step_slow(
    morphons: &mut HashMap<MorphonId, Morphon>,
    topology: &mut Topology,
    params: &MorphogenesisParams,
    learning_params: &LearningParams,
    homeostasis_level: f64,
    lifecycle: &LifecycleConfig,
    rng: &mut impl Rng,
) -> MorphogenesisReport {
    let mut report = MorphogenesisReport::default();

    report.synapses_created = synaptogenesis(morphons, topology, params, rng);
    report.synapses_pruned = pruning(topology, learning_params);

    if lifecycle.migration {
        report.migrations = migration(morphons, topology, params, homeostasis_level);
    }

    report
}

/// Run glacial-path morphogenesis (division, differentiation, fusion, apoptosis).
pub fn step_glacial(
    morphons: &mut HashMap<MorphonId, Morphon>,
    topology: &mut Topology,
    clusters: &mut HashMap<ClusterId, Cluster>,
    next_morphon_id: &mut MorphonId,
    next_cluster_id: &mut ClusterId,
    params: &MorphogenesisParams,
    arousal_level: f64,
    lifecycle: &LifecycleConfig,
    rng: &mut impl Rng,
) -> MorphogenesisReport {
    let mut report = MorphogenesisReport::default();

    if lifecycle.division {
        report.morphons_born = division(morphons, topology, next_morphon_id, params, rng);
    }

    if lifecycle.differentiation {
        report.differentiations = differentiation(morphons, topology);
        dedifferentiation(morphons, arousal_level);
    }

    if lifecycle.fusion {
        report.fusions = fusion(morphons, clusters, next_cluster_id, next_morphon_id, topology, params);
        report.defusions = defusion(morphons, clusters, topology);
    }

    if lifecycle.apoptosis {
        report.morphons_died = apoptosis(morphons, topology, params);
    }

    report
}
