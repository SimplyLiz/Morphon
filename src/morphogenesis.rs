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
    /// Transdifferentiation — minimum desire (chronic prediction error) to trigger
    /// direct A→B cell type conversion without passing through Stem.
    pub transdifferentiation_desire_threshold: f64,
    /// Transdifferentiation — minimum age before a morphon is eligible.
    pub transdifferentiation_min_age: u64,
    /// Maximum number of morphons (prevent unbounded growth).
    /// `None` = auto-derive from I/O dimensions: `max(500, (input + output) * 3)`.
    /// `Some(n)` = explicit override.
    pub max_morphons: Option<usize>,
    /// V3 Governor: minimum morphon count — apoptosis stops below this.
    pub min_morphons: usize,
    /// V3 Governor: minimum fraction of Sensory morphons (prevent I/O starvation).
    pub min_sensory_fraction: f64,
    /// V3 Governor: minimum fraction of Motor morphons (prevent output death).
    pub min_motor_fraction: f64,
    /// V3 Governor: maximum fraction of any single cell type (prevent Modulatory explosion).
    pub max_single_type_fraction: f64,

    /// V2: Frustration-driven stochastic exploration configuration.
    #[serde(default)]
    pub frustration: FrustrationConfig,
}

impl Default for MorphogenesisParams {
    fn default() -> Self {
        Self {
            synaptogenesis_threshold: 0.6,
            pruning_min_age: 100,
            division_threshold: 0.5,  // lowered from 1.0 — active k-WTA winners reproduce faster
            division_min_energy: 0.3,
            fusion_correlation_threshold: 0.75,
            fusion_min_size: 3,
            migration_rate: 0.05,
            apoptosis_min_age: 1000,
            apoptosis_energy_threshold: 0.05,
            transdifferentiation_desire_threshold: 0.5,
            transdifferentiation_min_age: 500,
            max_morphons: None,  // auto-derive from I/O dimensions
            min_morphons: 10,
            min_sensory_fraction: 0.05,
            min_motor_fraction: 0.02,
            max_single_type_fraction: 0.80,
            frustration: FrustrationConfig::default(),
        }
    }
}

/// Default morphon cap when no I/O dimensions are available.
pub const DEFAULT_MAX_MORPHONS: usize = 500;

impl MorphogenesisParams {
    /// Resolve the effective morphon cap from I/O dimensions.
    /// Called once during `System::new()`.
    pub fn resolve_max_morphons(&self, target_input: Option<usize>, target_output: Option<usize>) -> usize {
        if let Some(explicit) = self.max_morphons {
            return explicit;
        }
        let io_total = target_input.unwrap_or(0) + target_output.unwrap_or(0);
        if io_total > 0 {
            DEFAULT_MAX_MORPHONS.max(io_total * 3)
        } else {
            DEFAULT_MAX_MORPHONS
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
    pub transdifferentiations: usize,
    pub migrations: usize,
}

/// Run synaptogenesis — create new connections between correlated Morphons
/// that don't yet have a direct connection.
///
/// Uses random pair sampling (O(N)) instead of exhaustive pair scan (O(N²)).
/// For small active populations the sample count covers all pairs; for large
/// populations, statistical coverage accumulates across slow ticks.
pub fn synaptogenesis(
    morphons: &HashMap<MorphonId, Morphon>,
    topology: &mut Topology,
    params: &MorphogenesisParams,
    rng: &mut impl Rng,
    max_connectivity: usize,
    step_count: u64,
) -> usize {
    let mut created = 0;

    // Pre-filter: only morphons with recent activity are candidates.
    // InhibitoryInterneurons are excluded — their connectivity is managed by iSTDP.
    let activity_threshold = params.synaptogenesis_threshold * 0.5; // scale: 0.6 * 0.5 = 0.3 at default
    let active: Vec<MorphonId> = morphons
        .values()
        .filter(|m| m.cell_type != CellType::InhibitoryInterneuron
            && m.activity_history.mean() >= activity_threshold)
        .map(|m| m.id)
        .collect();

    if active.len() < 2 {
        return 0;
    }

    // Median degree for anti-hub gating (computed once, not per sample).
    let median_degree = {
        let mut degrees: Vec<usize> = active.iter()
            .map(|&id| topology.degree(id))
            .collect();
        degrees.sort();
        degrees[degrees.len() / 2].max(1)
    };

    // Sample O(N) random pairs. For small N this exceeds total pairs,
    // giving equivalent coverage to the exhaustive scan.
    let total_pairs = active.len() * (active.len() - 1) / 2;
    let num_samples = (active.len() * 5).min(total_pairs);

    for _ in 0..num_samples {
        let i = rng.random_range(0..active.len());
        let mut j = rng.random_range(0..active.len() - 1);
        if j >= i { j += 1; } // avoid i == j without rejection sampling

        let a = &morphons[&active[i]];
        let b = &morphons[&active[j]];

        // Must be in spatial proximity
        let distance = a.position.distance(&b.position);
        if distance > 2.0 {
            continue;
        }

        // Don't create if connection already exists (either direction)
        if topology.has_connection(a.id, b.id) || topology.has_connection(b.id, a.id) {
            continue;
        }

        // Respect cell type hierarchy:
        // - Don't create connections INTO Sensory (they're input-only)
        // - Don't create connections OUT OF Motor (they're output-only)
        // Try a→b first, fall back to b→a
        let (from, to) = if b.cell_type != CellType::Sensory && a.cell_type != CellType::Motor {
            (a, b)
        } else if a.cell_type != CellType::Sensory && b.cell_type != CellType::Motor {
            (b, a)
        } else {
            continue;
        };

        let prob = (1.0 - distance / 2.0) * 0.1;
        if rng.random_range(0.0..1.0) < prob {
            // V3 Governor: check connectivity cap
            if !crate::governance::check_connectivity(topology, from.id, max_connectivity)
                || !crate::governance::check_connectivity(topology, to.id, max_connectivity)
            {
                continue;
            }
            // Anti-hub: reduce synaptogenesis probability for high-degree nodes.
            // Morphons with >2x median degree get exponentially less likely to gain
            // new connections, preventing rich-get-richer feedback.
            let from_ratio = topology.degree(from.id) as f64 / median_degree as f64;
            let to_ratio = topology.degree(to.id) as f64 / median_degree as f64;
            if from_ratio > 2.0 && rng.random_range(0.0..1.0) < (from_ratio - 2.0) * 0.5 {
                continue;
            }
            if to_ratio > 2.0 && rng.random_range(0.0..1.0) < (to_ratio - 2.0) * 0.5 {
                continue;
            }
            let weight = rng.random_range(-0.5..0.5);
            // Distance-dependent delay: nearby morphons get fast connections (~0.5),
            // distant ones get slower propagation (up to ~2.0). Myelination can
            // reduce effective delay later for consolidated pathways.
            let delay = 0.5 + distance * 0.75;
            let justification = crate::justification::SynapticJustification::new(
                crate::justification::FormationCause::ProximityFormation { distance },
                step_count,
            );
            topology.add_synapse(from.id, to.id, Synapse::new_justified(weight, justification).with_delay(delay));
            created += 1;
        }
    }

    created
}

/// Run pruning — remove weak, unused synapses.
/// Uses distance-dependent cost: expensive long-distance synapses are pruned more
/// aggressively when they carry little weight.
pub fn pruning(
    topology: &mut Topology,
    learning_params: &LearningParams,
    morphons: &HashMap<MorphonId, Morphon>,
) -> usize {
    let edges_to_remove: Vec<_> = topology
        .all_edges()
        .into_iter()
        .filter(|(src_id, tgt_id, ei)| {
            let syn = &topology.graph[*ei];
            // Dimensionless cost factor: how much more expensive is this synapse
            // relative to a local, unmyelinated one?
            // dist=0 → 1.0 (baseline), dist=2.0 → 2.0, + myelin maintenance
            let cost_factor = match (morphons.get(src_id), morphons.get(tgt_id)) {
                (Some(src), Some(tgt)) => {
                    let dist = src.position.distance(&tgt.position);
                    (1.0 + dist * 0.5) * (1.0 + syn.myelination * 2.0)
                }
                _ => 1.0,
            };
            learning::should_prune_with_cost(syn, learning_params, cost_factor)
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
    effective_max_morphons: usize,
    rng: &mut impl Rng,
    step_count: u64,
) -> usize {
    if morphons.len() >= effective_max_morphons {
        return 0;
    }

    // V3 Governor: count cell types for diversity guard during division.
    // Children start as Stem, but if Stem already exceeds max_single_type_fraction,
    // block further division to prevent Stem/Modulatory explosion.
    let total = morphons.len() as f64;
    let stem_count = morphons.values().filter(|m| m.cell_type == CellType::Stem).count();
    let stem_fraction = stem_count as f64 / total.max(1.0);
    if stem_fraction > params.max_single_type_fraction {
        return 0; // too many Stem cells — block ALL division until they differentiate
    }

    let candidates: Vec<MorphonId> = morphons
        .values()
        .filter(|m| {
            // Only Associative and Stem can divide — Sensory/Motor are fixed I/O ports,
            // Modulatory is controlled by fusion. Without this, Sensory morphons
            // (100% firing from constant input) reproduce endlessly.
            (m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
            && m.should_divide(params.division_threshold)
        })
        .map(|m| m.id)
        .collect();

    let mut born = 0;
    // Cap births per glacial tick to prevent synapse explosion from connection duplication.
    // 654 births at once → 19K new synapses. Cap at 10% of current population.
    let max_births_per_tick = (morphons.len() / 10).max(5);
    for parent_id in candidates {
        if morphons.len() >= effective_max_morphons || born >= max_births_per_tick {
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
        topology.duplicate_connections(parent_id, child_id, rng, step_count);

        morphons.insert(child_id, child);
        born += 1;
    }

    born
}

/// Wire newly born Associative morphons to nearby InhibitoryInterneurons.
/// Called after division to ensure new morphons participate in local competition.
pub fn wire_to_nearby_interneurons(
    morphons: &HashMap<MorphonId, Morphon>,
    topology: &mut Topology,
    competition_mode: &crate::homeostasis::CompetitionMode,
    new_morphon_ids: &[MorphonId],
) {
    let initial_inh_weight = match competition_mode {
        crate::homeostasis::CompetitionMode::LocalInhibition { initial_inh_weight, .. } => {
            *initial_inh_weight
        }
        _ => return,
    };

    let interneurons: Vec<(MorphonId, crate::types::HyperbolicPoint)> = morphons.values()
        .filter(|m| m.cell_type == CellType::InhibitoryInterneuron)
        .map(|m| (m.id, m.position.clone()))
        .collect();

    for &new_id in new_morphon_ids {
        let new_m = match morphons.get(&new_id) {
            Some(m) if m.cell_type == CellType::Associative || m.cell_type == CellType::Stem => m,
            _ => continue,
        };

        // Connect to the closest interneuron
        let mut closest: Option<(MorphonId, f64)> = None;
        for (inh_id, inh_pos) in &interneurons {
            let dist = new_m.position.distance(inh_pos);
            if closest.is_none() || dist < closest.unwrap().1 {
                closest = Some((*inh_id, dist));
            }
        }

        if let Some((inh_id, _)) = closest {
            // Bidirectional wiring
            if topology.synapse_between(new_id, inh_id).is_none() {
                topology.add_synapse(new_id, inh_id,
                    crate::morphon::Synapse::new(0.3).with_delay(0.5));
            }
            if topology.synapse_between(inh_id, new_id).is_none() {
                topology.add_synapse(inh_id, new_id,
                    crate::morphon::Synapse::new(initial_inh_weight).with_delay(0.5));
            }
        }
    }
}

/// Run differentiation — Morphons specialize based on their activity patterns.
/// If a target morphology is provided, morphons inside a target region are
/// biased toward that region's cell type.
pub fn differentiation(
    morphons: &mut HashMap<MorphonId, Morphon>,
    _topology: &Topology,
    target: Option<&crate::developmental::TargetMorphology>,
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
        let mut diff_target = if mean_activity > 0.4 && variance < 0.1 {
            CellType::Associative // high consistent → Associative (the workhorse type)
        } else if mean_activity > 0.3 && variance > 0.2 {
            CellType::Associative // high variable → also Associative
        } else {
            // V2: Even if activity signature is weak, target morphology can
            // override differentiation toward the region's desired cell type.
            if let Some(tm) = target {
                let nearest = tm.regions.iter()
                    .filter(|r| morphon.position.distance(&r.center) <= r.radius)
                    .min_by(|a, b| {
                        morphon.position.distance(&a.center)
                            .partial_cmp(&morphon.position.distance(&b.center))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                if let Some(region) = nearest {
                    region.target_cell_type
                } else {
                    continue; // stay stem
                }
            } else {
                continue; // stay stem — don't eagerly classify as Modulatory
            }
        };

        // V2: If inside a target region, override differentiation target
        if let Some(tm) = target {
            let nearest = tm.regions.iter()
                .filter(|r| morphon.position.distance(&r.center) <= r.radius)
                .min_by(|a, b| {
                    morphon.position.distance(&a.center)
                        .partial_cmp(&morphon.position.distance(&b.center))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(region) = nearest {
                diff_target = region.target_cell_type;
            }
        }

        if morphon.differentiate(diff_target) {
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

/// Run transdifferentiation — direct A→B cell type conversion without Stem detour.
///
/// Biological pendant: pancreas alpha cells converting directly to insulin-producing
/// beta cells. Triggered by chronic "mismatch" — the morphon's current function
/// doesn't match the inputs it receives (concept doc section 3.4D).
///
/// Mismatch detection: if the majority of a morphon's incoming connections come from
/// a cell type that implies a different role, the morphon converts. For example, a
/// Modulatory morphon receiving mostly Sensory input should become Associative.
pub fn transdifferentiation(
    morphons: &mut HashMap<MorphonId, Morphon>,
    topology: &Topology,
    params: &MorphogenesisParams,
) -> usize {
    // First pass: collect (morphon_id, suggested_type) decisions.
    // We read topology + other morphons here, so we collect first, mutate second.
    let mut conversions: Vec<(MorphonId, CellType)> = Vec::new();

    for morphon in morphons.values() {
        // Only non-Stem, non-Fused, sufficiently old, under chronic mismatch
        if morphon.cell_type == CellType::Stem || morphon.cell_type == CellType::Fused {
            continue;
        }
        if morphon.age < params.transdifferentiation_min_age {
            continue;
        }
        if morphon.desire < params.transdifferentiation_desire_threshold {
            continue;
        }
        // Don't touch Sensory or Motor morphons — they're I/O boundary, not convertible
        if morphon.cell_type == CellType::Sensory || morphon.cell_type == CellType::Motor {
            continue;
        }

        // Analyze incoming connectivity: what types are feeding this morphon?
        let incoming = topology.incoming(morphon.id);
        if incoming.is_empty() {
            continue;
        }

        let mut type_counts: HashMap<CellType, usize> = HashMap::new();
        for (src_id, _synapse) in &incoming {
            if let Some(src) = morphons.get(src_id) {
                *type_counts.entry(src.cell_type).or_insert(0) += 1;
            }
        }

        let total = incoming.len() as f64;
        let suggested = infer_role_from_inputs(&type_counts, total);

        if let Some(target) = suggested {
            if target != morphon.cell_type {
                conversions.push((morphon.id, target));
            }
        }
    }

    // Second pass: apply conversions
    let mut count = 0;
    for (id, target) in conversions {
        if let Some(morphon) = morphons.get_mut(&id) {
            // Direct conversion: differentiate() handles the type switch + activation fn +
            // receptor update. Rate is 0.01 for non-Stem (slower than normal differentiation).
            if morphon.differentiate(target) {
                count += 1;
            }
        }
    }

    count
}

/// Infer what cell type a morphon *should* be based on its input sources.
/// Returns Some(target) if there's a clear signal, None if ambiguous.
fn infer_role_from_inputs(
    type_counts: &HashMap<CellType, usize>,
    total: f64,
) -> Option<CellType> {
    // Dominant input type must be >60% of connections for a clear signal
    let threshold = 0.6;

    let sensory_frac = *type_counts.get(&CellType::Sensory).unwrap_or(&0) as f64 / total;
    let motor_frac = *type_counts.get(&CellType::Motor).unwrap_or(&0) as f64 / total;
    let assoc_frac = *type_counts.get(&CellType::Associative).unwrap_or(&0) as f64 / total;
    let modul_frac = *type_counts.get(&CellType::Modulatory).unwrap_or(&0) as f64 / total;

    if sensory_frac > threshold {
        // Mostly sensory input → should be Associative (process patterns)
        Some(CellType::Associative)
    } else if motor_frac > threshold {
        // Mostly motor feedback → should be Modulatory (regulate output)
        Some(CellType::Modulatory)
    } else if modul_frac > threshold {
        // Mostly modulatory input → should be Associative (integrate signals)
        Some(CellType::Associative)
    } else if assoc_frac > threshold {
        // Mostly associative → could go either way, but if mismatched,
        // become Associative (join the processing layer)
        Some(CellType::Associative)
    } else {
        None // mixed inputs, no clear signal
    }
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

    /// V2: Shared energy pool — members contribute on fusion, draw during operation.
    #[serde(default)]
    pub shared_energy_pool: f64,

    /// V2: Shared homeostatic setpoint — single computation for the group.
    #[serde(default)]
    pub shared_homeostatic_setpoint: f64,

    /// V3: Epistemic state — how confident is the system in this cluster's knowledge?
    #[serde(default)]
    pub epistemic_state: crate::epistemic::EpistemicState,

    /// V3: Epistemic scarring — history of epistemic failures.
    #[serde(default)]
    pub epistemic_history: crate::epistemic::EpistemicHistory,
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
    effective_max_morphons: usize,
    max_cluster_size_fraction: f64,
    max_unverified_fraction: f64,
    competition_mode: &crate::homeostasis::CompetitionMode,
) -> usize {
    // V3 Governor: block new fusion when too many clusters are unverified.
    // Prevents unbounded cluster creation before existing ones are validated.
    if !clusters.is_empty() {
        let unverified = clusters.values()
            .filter(|c| matches!(c.epistemic_state, crate::epistemic::EpistemicState::Hypothesis { .. }))
            .count();
        if unverified as f64 / clusters.len() as f64 > max_unverified_fraction {
            return 0;
        }
    }

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

            // V3 Governor: cluster size check — no cluster may exceed max fraction
            let candidate_size = members.len();
            let total = morphons.len();
            if total > 0 && (candidate_size as f64 / total as f64) > max_cluster_size_fraction {
                continue;
            }

            let cluster_id = *next_cluster_id;
            *next_cluster_id += 1;

            // Calculate shared threshold and homeostatic setpoint
            let member_count = members.len() as f64;
            let avg_threshold: f64 = members
                .iter()
                .filter_map(|mid| morphons.get(mid))
                .map(|m| m.threshold)
                .sum::<f64>()
                / member_count;

            let avg_setpoint: f64 = members
                .iter()
                .filter_map(|mid| morphons.get(mid))
                .map(|m| m.homeostatic_setpoint)
                .sum::<f64>()
                / member_count;

            // V2: Pool 30% of each member's energy into shared pool
            let pooled_energy: f64 = members
                .iter()
                .filter_map(|mid| morphons.get(mid))
                .map(|m| m.energy * 0.3)
                .sum();

            // Update morphon states
            for &mid in &members {
                if let Some(m) = morphons.get_mut(&mid) {
                    m.fused_with = Some(cluster_id);
                    m.autonomy = 0.5; // partial fusion
                    m.threshold = avg_threshold;
                    m.energy *= 0.7; // contributed 30% to pool
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
                effective_max_morphons,
            );

            // === Create intra-cluster InhibitoryInterneurons (Section 3.4) ===
            // Only in LocalInhibition mode. These interneurons receive excitation from
            // all cluster members and send inhibition back, implementing local WTA
            // competition that is self-tuned by iSTDP rather than a global sort.
            if let crate::homeostasis::CompetitionMode::LocalInhibition {
                interneuron_ratio, initial_inh_weight, ..
            } = competition_mode {
                create_local_inhibitory_interneurons(
                    &members,
                    morphons,
                    topology,
                    next_morphon_id,
                    effective_max_morphons,
                    *initial_inh_weight,
                    *interneuron_ratio,
                );
            }

            clusters.insert(
                cluster_id,
                Cluster {
                    id: cluster_id,
                    members,
                    shared_threshold: avg_threshold,
                    inhibitory_morphons,
                    shared_energy_pool: pooled_energy,
                    shared_homeostatic_setpoint: avg_setpoint,
                    epistemic_state: crate::epistemic::EpistemicState::Hypothesis {
                        formation_step: 0, // caller doesn't have step_count; updated on next glacial eval
                    },
                    epistemic_history: Default::default(),
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
            let justification = crate::justification::SynapticJustification::new(
                crate::justification::FormationCause::FusionBridge { cluster: new_cluster_id },
                0, // step_count not available here; updated on next glacial eval
            );
            topology.add_synapse(
                inh_id,
                mid,
                crate::morphon::Synapse::new_justified(inhibition_weight, justification).with_delay(0.5),
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

/// Create intra-cluster InhibitoryInterneuron morphons for a newly-formed cluster.
///
/// Each interneuron receives excitatory input from all cluster members and sends
/// inhibitory (negative-weight) synapses back. iSTDP (Vogels et al. 2011) tunes
/// the inhibitory weights on the medium path to maintain target firing rates.
///
/// Number of interneurons: max(1, round(members.len() * interneuron_ratio)).
/// Biological baseline: ~10-20% inhibitory (Brunel 2000); start at 10% and let
/// iSTDP compensate if the ratio is imperfect.
fn create_local_inhibitory_interneurons(
    members: &[MorphonId],
    morphons: &mut HashMap<MorphonId, Morphon>,
    topology: &mut Topology,
    next_morphon_id: &mut MorphonId,
    max_morphons: usize,
    initial_inh_weight: f64,
    interneuron_ratio: f64,
) -> Vec<MorphonId> {
    let excitatory: Vec<MorphonId> = members.iter()
        .filter(|&&mid| morphons.get(&mid).map_or(false, |m|
            m.cell_type == CellType::Associative || m.cell_type == CellType::Stem))
        .copied()
        .collect();

    if excitatory.is_empty() {
        return Vec::new();
    }

    let n_interneurons = ((excitatory.len() as f64 * interneuron_ratio).round() as usize).max(1);
    let mut created = Vec::new();

    // Compute cluster centroid in Poincare ball
    let dim = morphons.values().next().map(|m| m.position.coords.len()).unwrap_or(3);
    let mut centroid = vec![0.0_f64; dim];
    let mut count = 0usize;
    for &mid in &excitatory {
        if let Some(m) = morphons.get(&mid) {
            for (i, c) in m.position.coords.iter().enumerate() {
                if i < centroid.len() { centroid[i] += c; }
            }
            count += 1;
        }
    }
    if count > 0 {
        for c in &mut centroid { *c /= count as f64; }
    }
    // Clamp inside Poincare ball
    let norm: f64 = centroid.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.85 {
        let scale = 0.85 / norm;
        for c in &mut centroid { *c *= scale; }
    }

    for _ in 0..n_interneurons {
        if morphons.len() >= max_morphons {
            break;
        }

        let inh_id = *next_morphon_id;
        *next_morphon_id += 1;

        let position = crate::types::HyperbolicPoint {
            coords: centroid.clone(),
            curvature: 1.0,
        };

        let mut inh = crate::morphon::Morphon::new(inh_id, position);
        inh.cell_type = CellType::InhibitoryInterneuron;
        inh.activation_fn = crate::types::ActivationFn::Sigmoid;

        topology.add_morphon(inh_id);

        // Excitatory member → interneuron (positive drive)
        for &mid in &excitatory {
            if topology.synapse_between(mid, inh_id).is_none() {
                topology.add_synapse(
                    mid, inh_id,
                    crate::morphon::Synapse::new(0.3).with_delay(0.5),
                );
            }
        }

        // Interneuron → all excitatory members (inhibitory feedback)
        let inh_weight = initial_inh_weight.min(-0.001);
        for &mid in &excitatory {
            if topology.synapse_between(inh_id, mid).is_none() {
                topology.add_synapse(
                    inh_id, mid,
                    crate::morphon::Synapse::new(inh_weight).with_delay(0.5),
                );
            }
        }

        morphons.insert(inh_id, inh);
        created.push(inh_id);
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
            // V2: Return pooled energy equally to ex-members
            let per_member = cluster.shared_energy_pool / cluster.members.len().max(1) as f64;
            for &mid in &cluster.members {
                if let Some(m) = morphons.get_mut(&mid) {
                    m.fused_with = None;
                    m.autonomy = 1.0;
                    m.energy = (m.energy + per_member).min(1.0);
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
    field: Option<&crate::field::MorphonField>,
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
        // V2: Frustration-gated migration — frustrated morphons can migrate
        // even without the desire>=0.3 gate, and use random direction when stuck.
        let frustrated = params.frustration.enabled
            && params.frustration.frustration_migration
            && morphon.frustration.exploration_mode
            && morphon.frustration.frustration_level > params.frustration.random_migration_threshold;

        // V2: Field-motivated migration — when a bioelectric field is present,
        // morphons with even mild PE (>0.05) can migrate along field gradients.
        let field_motivated = field.is_some() && morphon.desire > 0.05;

        if morphon.desire < 0.3 && !frustrated && !field_motivated {
            continue;
        }
        if morphon.fused_with.is_some() && morphon.autonomy < 0.5 {
            continue;
        }
        // Migration damping: respect cooldown
        if !crate::homeostasis::can_migrate(morphon) {
            continue;
        }

        let neighbors = topology.outgoing(morphon.id);
        if neighbors.is_empty() && !frustrated {
            continue;
        }

        // Compute tangent vector in hyperbolic space via log_map
        let mut tangent = vec![0.0; morphon.position.coords.len()];
        let mut count = 0;

        for (nid, _) in &neighbors {
            if let Some((pos, pe)) = positions.get(nid) {
                if *pe < morphon.prediction_error {
                    // Log map: tangent vector from morphon's position to neighbor
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

        // V2: Frustrated morphon with no gradient — use random exploration direction
        if count == 0 && frustrated {
            let dims = morphon.position.coords.len();
            for (i, t) in tangent.iter_mut().enumerate().take(dims) {
                let hash = (morphon.id.wrapping_mul(morphon.age).wrapping_add(9973 + i as u64) % 10000) as f64;
                *t = (hash / 5000.0 - 1.0) * params.migration_rate * morphon.frustration.frustration_level;
            }
            count = 1; // proceed to exp_map
        }

        if count > 0 {
            let scale = params.migration_rate * morphon.desire.max(0.1) * system_migration_mod;
            for t in &mut tangent {
                *t = *t / count as f64 * scale;
            }

            // V2: Blend field gradient into migration direction.
            // PE field gradient points toward HIGH PE — negate to move away.
            if let Some(field) = field {
                let field_weight = field.config.migration_field_weight;
                if field_weight > 0.0 {
                    if let Some((gx, gy)) = field.gradient_at(&morphon.position, crate::field::FieldType::PredictionError) {
                        let neighbor_weight = 1.0 - field_weight;
                        if tangent.len() >= 2 {
                            tangent[0] = tangent[0] * neighbor_weight + (-gx) * field_weight * params.migration_rate;
                            tangent[1] = tangent[1] * neighbor_weight + (-gy) * field_weight * params.migration_rate;
                        }
                    }
                    // V2: Identity field gradient — attract toward regions that need morphons.
                    // Positive direction (move TOWARD high identity strength).
                    if let Some((ix, iy)) = field.gradient_at(&morphon.position, crate::field::FieldType::Identity) {
                        let identity_weight = 0.1;
                        if tangent.len() >= 2 {
                            tangent[0] += ix * identity_weight * params.migration_rate;
                            tangent[1] += iy * identity_weight * params.migration_rate;
                        }
                    }
                }
            }

            // Strip inward radial component — only allow lateral or outward migration.
            // This prevents the global collapse toward origin that happens when
            // interior morphons have lower prediction error.
            let current_radius = morphon.position.specificity();
            if current_radius > 1e-6 {
                let radial_dot: f64 = tangent.iter().enumerate()
                    .map(|(i, t)| t * morphon.position.coords[i] / current_radius)
                    .sum();
                if radial_dot < 0.0 {
                    // Negative = pointing inward — remove only the inward part
                    for (i, t) in tangent.iter_mut().enumerate() {
                        *t -= radial_dot * morphon.position.coords[i] / current_radius;
                    }
                }
            }

            // Exponential map: project tangent onto hyperbolic manifold
            morphon.position = morphon.position.exp_map(&tangent);
            morphon.migration_cooldown = 20.0; // set cooldown
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
    recent_births: usize,
) -> usize {
    // V3 Governor: energy floor — don't apoptose below minimum morphon count
    if morphons.len() <= params.min_morphons {
        return 0;
    }

    // Count cell types for diversity guard
    let mut type_counts: HashMap<CellType, usize> = HashMap::new();
    for m in morphons.values() {
        *type_counts.entry(m.cell_type).or_insert(0) += 1;
    }
    let total = morphons.len() as f64;

    let to_remove: Vec<MorphonId> = morphons
        .values()
        .filter(|m| {
            // InhibitoryInterneurons are structural — never apoptose.
            m.cell_type != CellType::InhibitoryInterneuron
                && m.age > params.apoptosis_min_age
                && m.fused_with.is_none()
                // Two paths to apoptosis:
                // (a) Sustained energy deficit: below threshold for 500+ ticks.
                //     With reward-correlated utility + superlinear firing cost,
                //     hubs drain energy (high cost, uncorrelated reward). Duration
                //     check prevents killing morphons with transient dips.
                // (b) Silent: activity < 0.1% — useless regardless of energy.
                && (m.ticks_below_energy_threshold > 500
                    || m.activity_history.mean() < 0.001)

                // V3 Governor: protect minimum cell type fractions
                && match m.cell_type {
                    CellType::Sensory => {
                        let count = *type_counts.get(&CellType::Sensory).unwrap_or(&0);
                        count as f64 / total > params.min_sensory_fraction
                    }
                    CellType::Motor => {
                        let count = *type_counts.get(&CellType::Motor).unwrap_or(&0);
                        count as f64 / total > params.min_motor_fraction
                    }
                    _ => true,
                }
        })
        .map(|m| m.id)
        .collect();

    // Adaptive rate limit: deaths track births, not a fixed percentage.
    // Kill at most (recent_births + 2) per glacial step. This ensures:
    // - If division is active (births > 0): apoptosis can match + slight surplus
    // - If division is stalled (births = 0): at most 2 die per step (gentle bleed)
    // - The population naturally converges to a size where births ≈ deaths
    let max_deaths = recent_births + 2;
    let to_remove = if to_remove.len() > max_deaths {
        to_remove[..max_deaths].to_vec()
    } else {
        to_remove
    };

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
    field: Option<&crate::field::MorphonField>,
    max_connectivity: usize,
    step_count: u64,
) -> MorphogenesisReport {
    let mut report = MorphogenesisReport::default();

    report.synapses_created = synaptogenesis(morphons, topology, params, rng, max_connectivity, step_count);
    report.synapses_pruned = pruning(topology, learning_params, morphons);

    if lifecycle.migration {
        report.migrations = migration(morphons, topology, params, homeostasis_level, field);
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
    effective_max_morphons: usize,
    max_cluster_size_fraction: f64,
    max_unverified_fraction: f64,
    arousal_level: f64,
    lifecycle: &LifecycleConfig,
    rng: &mut impl Rng,
    target: Option<&crate::developmental::TargetMorphology>,
    step_count: u64,
    competition_mode: &crate::homeostasis::CompetitionMode,
) -> MorphogenesisReport {
    let mut report = MorphogenesisReport::default();

    if lifecycle.division {
        report.morphons_born = division(morphons, topology, next_morphon_id, params, effective_max_morphons, rng, step_count);
    }

    if lifecycle.differentiation {
        report.differentiations = differentiation(morphons, topology, target);
        report.transdifferentiations = transdifferentiation(morphons, topology, params);
        dedifferentiation(morphons, arousal_level);
    }

    if lifecycle.fusion {
        report.fusions = fusion(morphons, clusters, next_cluster_id, next_morphon_id, topology, params, effective_max_morphons, max_cluster_size_fraction, max_unverified_fraction, competition_mode);
        report.defusions = defusion(morphons, clusters, topology);
    }

    if lifecycle.apoptosis {
        report.morphons_died = apoptosis(morphons, topology, params, report.morphons_born);
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphon::Morphon;
    use crate::types::HyperbolicPoint;

    fn make_morphon(id: MorphonId, cell_type: CellType) -> Morphon {
        let mut m = Morphon::new(id, HyperbolicPoint::random(3, &mut rand::rng()));
        m.cell_type = cell_type;
        m.receptors = crate::types::default_receptors(cell_type);
        m
    }


    // === Auto-derivation of max_morphons ===

    #[test]
    fn resolve_max_morphons_auto_derives_from_io() {
        let params = MorphogenesisParams::default();
        // No I/O → fallback to DEFAULT_MAX_MORPHONS
        assert_eq!(params.resolve_max_morphons(None, None), 500);
        // Small I/O → still 500 (floor)
        assert_eq!(params.resolve_max_morphons(Some(4), Some(2)), 500);
        // Large I/O → 3× total
        assert_eq!(params.resolve_max_morphons(Some(784), Some(10)), 2382);
    }

    #[test]
    fn resolve_max_morphons_explicit_overrides() {
        let params = MorphogenesisParams { max_morphons: Some(42), ..Default::default() };
        // Explicit always wins, even if smaller than I/O
        assert_eq!(params.resolve_max_morphons(Some(784), Some(10)), 42);
    }

    // === Synaptogenesis ===

    #[test]
    fn synaptogenesis_creates_connections_between_active_nearby_morphons() {
        let mut rng = rand::rng();
        let pos = HyperbolicPoint { coords: vec![0.1, 0.0, 0.0], curvature: 1.0 };
        let pos2 = HyperbolicPoint { coords: vec![0.15, 0.0, 0.0], curvature: 1.0 };

        let mut m1 = Morphon::new(1, pos);
        m1.cell_type = CellType::Associative;
        for _ in 0..100 { m1.activity_history.push(0.5); }

        let mut m2 = Morphon::new(2, pos2);
        m2.cell_type = CellType::Associative;
        for _ in 0..100 { m2.activity_history.push(0.5); }

        let mut morphons = HashMap::new();
        morphons.insert(1, m1);
        morphons.insert(2, m2);

        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);

        let params = MorphogenesisParams::default();

        // Run many times since it's probabilistic
        let mut total_created = 0;
        for _ in 0..100 {
            total_created += synaptogenesis(&morphons, &mut topo, &params, &mut rng, 50, 0);
        }
        // With very close morphons and high activity, should eventually create connections
        assert!(total_created > 0, "synaptogenesis should create some connections over 100 attempts");
    }

    #[test]
    fn synaptogenesis_respects_cell_type_hierarchy() {
        let mut rng = rand::rng();
        let pos = HyperbolicPoint { coords: vec![0.1, 0.0, 0.0], curvature: 1.0 };

        // Motor → anything should not create (Motor is output-only)
        let mut motor = Morphon::new(1, pos.clone());
        motor.cell_type = CellType::Motor;
        for _ in 0..100 { motor.activity_history.push(0.5); }

        let mut assoc = Morphon::new(2, pos.clone());
        assoc.cell_type = CellType::Associative;
        for _ in 0..100 { assoc.activity_history.push(0.5); }

        let mut morphons = HashMap::new();
        morphons.insert(1, motor);
        morphons.insert(2, assoc);

        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);

        let params = MorphogenesisParams::default();
        for _ in 0..100 {
            synaptogenesis(&morphons, &mut topo, &params, &mut rng, 50, 0);
        }
        // Motor as source should be blocked
        assert!(!topo.has_connection(1, 2), "Motor should not have outgoing connections to non-Motor");
    }

    #[test]
    fn synaptogenesis_skips_inactive_morphons() {
        let mut rng = rand::rng();
        let pos = HyperbolicPoint { coords: vec![0.1, 0.0, 0.0], curvature: 1.0 };

        let m1 = Morphon::new(1, pos.clone()); // default activity = 0
        let m2 = Morphon::new(2, pos.clone());

        let mut morphons = HashMap::new();
        morphons.insert(1, m1);
        morphons.insert(2, m2);

        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);

        let params = MorphogenesisParams::default();
        let created = synaptogenesis(&morphons, &mut topo, &params, &mut rng, 50, 0);
        assert_eq!(created, 0, "inactive morphons should not form new connections");
    }

    // === Pruning ===

    #[test]
    fn pruning_removes_weak_old_unused_synapses() {
        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);

        let mut weak_syn = Synapse::new(0.0001); // below weight_min
        weak_syn.age = 200;
        weak_syn.usage_count = 0;
        topo.add_synapse(1, 2, weak_syn);

        let params = LearningParams::default();
        let morphons = HashMap::new();
        let pruned = pruning(&mut topo, &params, &morphons);
        assert_eq!(pruned, 1);
        assert_eq!(topo.synapse_count(), 0);
    }

    #[test]
    fn pruning_keeps_strong_synapses() {
        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);

        let mut strong_syn = Synapse::new(1.0);
        strong_syn.age = 200;
        strong_syn.usage_count = 0;
        topo.add_synapse(1, 2, strong_syn);

        let params = LearningParams::default();
        let morphons = HashMap::new();
        let pruned = pruning(&mut topo, &params, &morphons);
        assert_eq!(pruned, 0);
        assert_eq!(topo.synapse_count(), 1);
    }

    // === Division ===

    #[test]
    fn division_creates_child_morphon() {
        let mut rng = rand::rng();
        let mut morphons = HashMap::new();
        let mut parent = make_morphon(1, CellType::Associative);
        parent.division_pressure = 2.0; // above threshold (1.0)
        parent.energy = 0.8;
        morphons.insert(1, parent);

        let mut topo = Topology::new();
        topo.add_morphon(1);

        let params = MorphogenesisParams::default();
        let mut next_id = 100;
        let born = division(&mut morphons, &mut topo, &mut next_id, &params, DEFAULT_MAX_MORPHONS, &mut rng, 0);

        assert_eq!(born, 1);
        assert_eq!(morphons.len(), 2);
        assert!(morphons.contains_key(&100));
        assert_eq!(next_id, 101);

        // Child should be Stem
        assert_eq!(morphons[&100].cell_type, CellType::Stem);
        // Parent energy should be halved
        assert!(morphons[&1].energy < 0.8);
        // Parent division pressure should be reset
        assert_eq!(morphons[&1].division_pressure, 0.0);
    }

    #[test]
    fn division_respects_max_morphons() {
        let mut rng = rand::rng();
        let mut morphons = HashMap::new();
        for i in 0..10 {
            let mut m = make_morphon(i, CellType::Associative);
            m.division_pressure = 2.0;
            m.energy = 0.8;
            morphons.insert(i, m);
        }

        let mut topo = Topology::new();
        for i in 0..10 { topo.add_morphon(i); }

        let params = MorphogenesisParams::default();
        let effective_max = 12;
        let mut next_id = 100;
        let born = division(&mut morphons, &mut topo, &mut next_id, &params, effective_max, &mut rng, 0);

        assert!(born <= 2, "should not exceed max_morphons");
        assert!(morphons.len() <= 12);
    }

    #[test]
    fn division_skips_low_energy_morphons() {
        let mut rng = rand::rng();
        let mut morphons = HashMap::new();
        let mut m = make_morphon(1, CellType::Associative);
        m.division_pressure = 2.0;
        m.energy = 0.1; // below division_min_energy (0.3)
        morphons.insert(1, m);

        let mut topo = Topology::new();
        topo.add_morphon(1);

        let params = MorphogenesisParams::default();
        let mut next_id = 100;
        let born = division(&mut morphons, &mut topo, &mut next_id, &params, DEFAULT_MAX_MORPHONS, &mut rng, 0);

        assert_eq!(born, 0);
    }

    // === Differentiation ===

    #[test]
    fn differentiation_converts_mature_active_stem_cells() {
        let mut morphons = HashMap::new();
        let mut m = make_morphon(1, CellType::Stem);
        m.age = 300; // above 200
        // High consistent activity → Associative
        for _ in 0..100 { m.activity_history.push(0.5); }
        morphons.insert(1, m);

        let topo = Topology::new();
        let count = differentiation(&mut morphons, &topo, None);

        assert_eq!(count, 1);
        assert_eq!(morphons[&1].cell_type, CellType::Associative);
    }

    #[test]
    fn differentiation_skips_young_morphons() {
        let mut morphons = HashMap::new();
        let mut m = make_morphon(1, CellType::Stem);
        m.age = 50; // below 200
        for _ in 0..100 { m.activity_history.push(0.5); }
        morphons.insert(1, m);

        let topo = Topology::new();
        let count = differentiation(&mut morphons, &topo, None);
        assert_eq!(count, 0);
    }

    #[test]
    fn differentiation_skips_already_differentiated() {
        let mut morphons = HashMap::new();
        let mut m = make_morphon(1, CellType::Associative); // not Stem
        m.age = 300;
        for _ in 0..100 { m.activity_history.push(0.5); }
        morphons.insert(1, m);

        let topo = Topology::new();
        let count = differentiation(&mut morphons, &topo, None);
        assert_eq!(count, 0);
    }

    // === Dedifferentiation ===

    #[test]
    fn dedifferentiation_under_extreme_stress() {
        let mut morphons = HashMap::new();
        let mut m = make_morphon(1, CellType::Associative);
        m.desire = 0.95; // very high PE
        m.differentiation_level = 0.3; // below 0.5
        morphons.insert(1, m);

        let count = dedifferentiation(&mut morphons, 0.9); // high arousal
        assert_eq!(count, 1);
    }

    #[test]
    fn dedifferentiation_does_not_affect_stem() {
        let mut morphons = HashMap::new();
        let m = make_morphon(1, CellType::Stem);
        morphons.insert(1, m);

        let count = dedifferentiation(&mut morphons, 0.9);
        assert_eq!(count, 0);
    }

    // === Apoptosis ===

    #[test]
    fn apoptosis_removes_old_inactive_low_energy_morphons() {
        let mut morphons = HashMap::new();
        let mut m = Morphon::new(1, HyperbolicPoint::origin(3));
        m.age = 2000; // above min_age
        m.energy = 0.01; // below threshold
        // Activity near zero (default)
        morphons.insert(1, m);

        // Add enough healthy morphons to exceed min_morphons governor
        let mut topo = Topology::new();
        topo.add_morphon(1);
        for id in 100..112 {
            let healthy = Morphon::new(id, HyperbolicPoint::origin(3));
            morphons.insert(id, healthy);
            topo.add_morphon(id);
        }

        let params = MorphogenesisParams::default();
        let died = apoptosis(&mut morphons, &mut topo, &params, 100);

        assert_eq!(died, 1);
        assert!(!morphons.contains_key(&1), "apoptosed morphon should be removed");
        assert_eq!(morphons.len(), 12, "healthy morphons should remain");
    }

    #[test]
    fn apoptosis_keeps_young_morphons() {
        let mut morphons = HashMap::new();
        let mut m = Morphon::new(1, HyperbolicPoint::origin(3));
        m.age = 100; // below min_age
        m.energy = 0.01;
        morphons.insert(1, m);

        let mut topo = Topology::new();
        topo.add_morphon(1);

        let params = MorphogenesisParams::default();
        let died = apoptosis(&mut morphons, &mut topo, &params, 100);
        assert_eq!(died, 0);
    }

    #[test]
    fn apoptosis_keeps_fused_morphons() {
        let mut morphons = HashMap::new();
        let mut m = Morphon::new(1, HyperbolicPoint::origin(3));
        m.age = 2000;
        m.energy = 0.01;
        m.fused_with = Some(42);
        morphons.insert(1, m);

        let mut topo = Topology::new();
        topo.add_morphon(1);

        let params = MorphogenesisParams::default();
        let died = apoptosis(&mut morphons, &mut topo, &params, 100);
        assert_eq!(died, 0, "fused morphons should be protected from apoptosis");
    }

    #[test]
    fn apoptosis_keeps_energy_rich_morphons() {
        let mut morphons = HashMap::new();
        let mut m = Morphon::new(1, HyperbolicPoint::origin(3));
        m.age = 2000;
        m.energy = 0.5; // above threshold — survives regardless of connectivity
        // Give it some activity so it doesn't die via silent path (b)
        for _ in 0..10 {
            m.activity_history.push(1.0);
        }
        morphons.insert(1, m);

        // Add enough healthy morphons to exceed min_morphons governor
        let mut topo = Topology::new();
        topo.add_morphon(1);
        for id in 100..112 {
            let healthy = Morphon::new(id, HyperbolicPoint::origin(3));
            morphons.insert(id, healthy);
            topo.add_morphon(id);
        }

        let params = MorphogenesisParams::default();
        let died = apoptosis(&mut morphons, &mut topo, &params, 100);
        assert_eq!(died, 0, "energy-rich morphons should survive regardless of connectivity");
    }

    #[test]
    fn apoptosis_kills_energy_starved_well_connected_morphons() {
        let mut morphons = HashMap::new();
        let mut m = Morphon::new(1, HyperbolicPoint::origin(3));
        m.age = 2000;
        m.energy = 0.03;
        m.ticks_below_energy_threshold = 600; // sustained deficit > 500 ticks
        morphons.insert(1, m);

        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);
        topo.add_morphon(3);
        topo.add_morphon(4);
        topo.add_synapse(2, 1, Synapse::new(0.5));
        topo.add_synapse(3, 1, Synapse::new(0.3));
        topo.add_synapse(1, 4, Synapse::new(0.2));
        // degree = 3, but energy-starved — should die

        // Add enough healthy morphons to exceed min_morphons governor
        for id in 100..112 {
            let healthy = Morphon::new(id, HyperbolicPoint::origin(3));
            morphons.insert(id, healthy);
            topo.add_morphon(id);
        }

        let params = MorphogenesisParams::default();
        let died = apoptosis(&mut morphons, &mut topo, &params, 100);
        assert_eq!(died, 1, "energy-starved morphons should die regardless of connectivity");
        assert!(!morphons.contains_key(&1));
    }

    #[test]
    fn apoptosis_silent_path_kills_active_energy_rich_but_silent_morphons() {
        let mut morphons = HashMap::new();
        let mut m = Morphon::new(1, HyperbolicPoint::origin(3));
        m.age = 2000;
        m.energy = 0.8; // healthy energy
        // Activity near zero (default RingBuffer) — silent for 100+ steps
        morphons.insert(1, m);

        let mut topo = Topology::new();
        topo.add_morphon(1);
        // Give it high degree — should still die via silent path
        for id in 2..6 {
            topo.add_morphon(id);
            topo.add_synapse(id, 1, Synapse::new(0.5));
        }

        // Add enough healthy morphons to exceed min_morphons governor
        for id in 100..112 {
            let healthy = Morphon::new(id, HyperbolicPoint::origin(3));
            morphons.insert(id, healthy);
            topo.add_morphon(id);
        }

        let params = MorphogenesisParams::default();
        let died = apoptosis(&mut morphons, &mut topo, &params, 100);
        assert_eq!(died, 1, "silent morphon should die via path (b) regardless of energy or connectivity");
        assert!(!morphons.contains_key(&1));
    }

    #[test]
    fn apoptosis_rate_limiting() {
        let mut morphons = HashMap::new();
        let mut topo = Topology::new();

        // Create 20 apoptosis-eligible morphons (old, low energy, default silent activity)
        for id in 1..=20 {
            let mut m = Morphon::new(id, HyperbolicPoint::origin(3));
            m.age = 2000;
            m.energy = 0.01;
            morphons.insert(id, m);
            topo.add_morphon(id);
        }
        // Add healthy morphons to exceed min_morphons
        for id in 100..130 {
            let healthy = Morphon::new(id, HyperbolicPoint::origin(3));
            morphons.insert(id, healthy);
            topo.add_morphon(id);
        }

        let params = MorphogenesisParams::default();
        // recent_births=3 → max_deaths = 3 + 2 = 5
        let died = apoptosis(&mut morphons, &mut topo, &params, 3);
        assert_eq!(died, 5, "apoptosis should be rate-limited to recent_births + 2");
    }

    #[test]
    fn apoptosis_min_morphons_governor() {
        let mut morphons = HashMap::new();
        let mut topo = Topology::new();

        // Create exactly min_morphons eligible morphons
        let params = MorphogenesisParams::default();
        for id in 1..=(params.min_morphons as u64) {
            let mut m = Morphon::new(id, HyperbolicPoint::origin(3));
            m.age = 2000;
            m.energy = 0.01;
            morphons.insert(id, m);
            topo.add_morphon(id);
        }

        let died = apoptosis(&mut morphons, &mut topo, &params, 100);
        assert_eq!(died, 0, "apoptosis should not fire when at min_morphons floor");
    }

    #[test]
    fn apoptosis_protects_last_sensory_morphon() {
        let mut morphons = HashMap::new();
        let mut topo = Topology::new();

        // One sensory morphon — eligible for death but protected by diversity guard
        let mut sensory = Morphon::new(1, HyperbolicPoint::origin(3));
        sensory.age = 2000;
        sensory.energy = 0.01;
        sensory.cell_type = CellType::Sensory;
        morphons.insert(1, sensory);
        topo.add_morphon(1);

        // Add enough non-sensory morphons so removing the sensory one
        // would violate min_sensory_fraction (0.05).
        // With 1 sensory out of 12 total = 8.3% > 5%, so it can die.
        // With 1 sensory out of 25 total = 4% < 5%, so it's protected.
        for id in 100..124 {
            let mut m = Morphon::new(id, HyperbolicPoint::origin(3));
            m.cell_type = CellType::Associative;
            morphons.insert(id, m);
            topo.add_morphon(id);
        }

        let params = MorphogenesisParams::default();
        let died = apoptosis(&mut morphons, &mut topo, &params, 100);
        assert_eq!(died, 0, "last sensory morphon near min_sensory_fraction should be protected");
        assert!(morphons.contains_key(&1));
    }

    // === Fusion ===

    #[test]
    fn fusion_groups_correlated_active_morphons() {
        let mut morphons = HashMap::new();
        let pos = HyperbolicPoint { coords: vec![0.1, 0.0, 0.0], curvature: 1.0 };

        // Create 3 morphons that all fire together (fired = true, same state)
        for i in 0..3 {
            let mut m = Morphon::new(i, pos.clone());
            m.cell_type = CellType::Associative;
            m.fired = true;
            m.prediction_error = 0.01; // low PE → passes PE gate
            m.desire = 0.05;           // mean_pe < mean_desire won't block
            for _ in 0..100 { m.activity_history.push(0.5); }
            morphons.insert(i, m);
        }
        // Add bystander morphons so the fusion candidate (3/13 ≈ 23%) passes
        // the max_cluster_size_fraction check (30% cap).
        for i in 10..20 {
            let far_pos = HyperbolicPoint { coords: vec![0.5, 0.5, 0.0], curvature: 1.0 };
            let mut m = Morphon::new(i, far_pos);
            m.cell_type = CellType::Associative;
            morphons.insert(i, m);
        }

        let mut topo = Topology::new();
        for &id in morphons.keys() { topo.add_morphon(id); }
        topo.add_synapse(0, 1, Synapse::new(0.5));
        topo.add_synapse(0, 2, Synapse::new(0.3));

        let mut clusters = HashMap::new();
        let mut next_cluster_id = 0;
        let mut next_morphon_id = 100;
        let params = MorphogenesisParams { fusion_min_size: 3, ..Default::default() };

        let fused = fusion(
            &mut morphons, &mut clusters, &mut next_cluster_id,
            &mut next_morphon_id, &mut topo, &params, DEFAULT_MAX_MORPHONS, 0.3, 0.5,
            &crate::homeostasis::CompetitionMode::default(),
        );

        assert_eq!(fused, 1, "should form one cluster");
        assert_eq!(clusters.len(), 1);
        let cluster = clusters.values().next().unwrap();
        assert!(cluster.members.len() >= 3);
    }

    // === Defusion ===

    #[test]
    fn defusion_breaks_clusters_with_diverging_errors() {
        let mut morphons = HashMap::new();
        let pos = HyperbolicPoint::origin(3);

        let mut m1 = Morphon::new(1, pos.clone());
        m1.fused_with = Some(0);
        m1.autonomy = 0.5;
        m1.prediction_error = 0.1;
        morphons.insert(1, m1);

        let mut m2 = Morphon::new(2, pos.clone());
        m2.fused_with = Some(0);
        m2.autonomy = 0.5;
        m2.prediction_error = 1.5; // variance = ((0.1-0.8)^2 + (1.5-0.8)^2)/2 = 0.49... need more
        morphons.insert(2, m2);

        // Add a third member with extreme divergence to push variance > 0.5
        let mut m3 = Morphon::new(3, pos.clone());
        m3.fused_with = Some(0);
        m3.autonomy = 0.5;
        m3.prediction_error = 2.0;
        morphons.insert(3, m3);

        let mut clusters = HashMap::new();
        clusters.insert(0, Cluster {
            id: 0,
            members: vec![1, 2, 3],
            shared_threshold: 0.3,
            inhibitory_morphons: vec![],
            shared_energy_pool: 0.0,
            shared_homeostatic_setpoint: 0.15,
            epistemic_state: Default::default(),
            epistemic_history: Default::default(),
        });

        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);
        topo.add_morphon(3);

        let defused = defusion(&mut morphons, &mut clusters, &mut topo);
        assert_eq!(defused, 1);
        assert!(clusters.is_empty());
        assert!(morphons[&1].fused_with.is_none());
        assert_eq!(morphons[&1].autonomy, 1.0);
    }

    #[test]
    fn defusion_cleans_up_inhibitory_morphons() {
        let mut morphons = HashMap::new();
        let pos = HyperbolicPoint::origin(3);

        let mut m1 = Morphon::new(1, pos.clone());
        m1.fused_with = Some(0);
        m1.prediction_error = 0.0;
        morphons.insert(1, m1);

        let mut m2 = Morphon::new(2, pos.clone());
        m2.fused_with = Some(0);
        m2.prediction_error = 2.0; // extreme divergence
        morphons.insert(2, m2);

        let mut m3 = Morphon::new(3, pos.clone());
        m3.fused_with = Some(0);
        m3.prediction_error = 3.0;
        morphons.insert(3, m3);

        // Inhibitory morphon
        let inh = Morphon::new(99, pos.clone());
        morphons.insert(99, inh);

        let mut clusters = HashMap::new();
        clusters.insert(0, Cluster {
            id: 0,
            members: vec![1, 2, 3],
            shared_threshold: 0.3,
            inhibitory_morphons: vec![99],
            shared_energy_pool: 0.0,
            shared_homeostatic_setpoint: 0.15,
            epistemic_state: Default::default(),
            epistemic_history: Default::default(),
        });

        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);
        topo.add_morphon(3);
        topo.add_morphon(99);

        let defused = defusion(&mut morphons, &mut clusters, &mut topo);
        assert_eq!(defused, 1);
        assert!(!morphons.contains_key(&99), "inhibitory morphon should be removed");
    }

    // === Migration ===

    #[test]
    fn migration_moves_high_desire_morphons() {
        let mut morphons = HashMap::new();
        let pos1 = HyperbolicPoint { coords: vec![0.3, 0.0, 0.0], curvature: 1.0 };
        let pos2 = HyperbolicPoint { coords: vec![0.5, 0.0, 0.0], curvature: 1.0 };

        let mut m1 = Morphon::new(1, pos1.clone());
        m1.desire = 0.8;
        m1.prediction_error = 0.5;
        morphons.insert(1, m1);

        let mut m2 = Morphon::new(2, pos2);
        m2.prediction_error = 0.1; // lower PE → attractive target
        morphons.insert(2, m2);

        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);
        topo.add_synapse(1, 2, Synapse::new(0.5));

        let params = MorphogenesisParams::default();
        let migrated = migration(&mut morphons, &topo, &params, 0.0, None);

        assert_eq!(migrated, 1);
        let new_pos = &morphons[&1].position;
        assert_ne!(
            new_pos.coords, pos1.coords,
            "morphon should have moved"
        );
        assert!(morphons[&1].migration_cooldown > 0.0, "cooldown should be set");
    }

    #[test]
    fn migration_skips_low_desire_morphons() {
        let mut morphons = HashMap::new();
        let mut m = Morphon::new(1, HyperbolicPoint::origin(3));
        m.desire = 0.1; // below 0.3 threshold
        morphons.insert(1, m);

        let topo = Topology::new();
        let params = MorphogenesisParams::default();
        let migrated = migration(&mut morphons, &topo, &params, 0.0, None);
        assert_eq!(migrated, 0);
    }

    #[test]
    fn migration_skips_fused_low_autonomy() {
        let mut morphons = HashMap::new();
        let mut m = Morphon::new(1, HyperbolicPoint::origin(3));
        m.desire = 0.8;
        m.fused_with = Some(0);
        m.autonomy = 0.3; // below 0.5
        morphons.insert(1, m);

        let topo = Topology::new();
        let params = MorphogenesisParams::default();
        let migrated = migration(&mut morphons, &topo, &params, 0.0, None);
        assert_eq!(migrated, 0);
    }

    // === step_slow / step_glacial ===

    #[test]
    fn step_slow_returns_report() {
        let mut rng = rand::rng();
        let mut morphons = HashMap::new();
        morphons.insert(1, make_morphon(1, CellType::Associative));

        let mut topo = Topology::new();
        topo.add_morphon(1);

        let params = MorphogenesisParams::default();
        let lp = LearningParams::default();
        let lifecycle = LifecycleConfig::default();

        let report = step_slow(&mut morphons, &mut topo, &params, &lp, 0.5, &lifecycle, &mut rng, None, 50, 0);
        // Just verify it runs and returns a valid report
        // Report is valid (fields are populated)
        let _ = report.synapses_created;
        let _ = report.synapses_pruned;
    }

    #[test]
    fn step_glacial_respects_lifecycle_config() {
        let mut rng = rand::rng();
        let mut morphons = HashMap::new();
        let mut m = make_morphon(1, CellType::Associative);
        m.division_pressure = 2.0;
        m.energy = 0.8;
        morphons.insert(1, m);

        let mut topo = Topology::new();
        topo.add_morphon(1);
        let mut clusters = HashMap::new();
        let mut next_mid = 100;
        let mut next_cid = 0;
        let params = MorphogenesisParams::default();

        // Division disabled
        let lifecycle = LifecycleConfig { division: false, ..Default::default() };
        let report = step_glacial(
            &mut morphons, &mut topo, &mut clusters,
            &mut next_mid, &mut next_cid, &params, DEFAULT_MAX_MORPHONS, 0.3, 0.5, 0.0, &lifecycle, &mut rng, None, 0,
            &crate::homeostasis::CompetitionMode::default(),
        );
        assert_eq!(report.morphons_born, 0);
    }

    // === Transdifferentiation ===

    #[test]
    fn transdifferentiation_converts_mismatched_morphon() {
        let pos = HyperbolicPoint { coords: vec![0.1, 0.0, 0.0], curvature: 1.0 };

        let mut target = Morphon::new(10, pos.clone());
        target.cell_type = CellType::Modulatory;
        target.differentiation_level = 0.5;
        target.activation_fn = ActivationFn::Oscillatory;
        target.receptors = default_receptors(CellType::Modulatory);
        target.age = 600;
        target.desire = 0.8;

        let s1 = make_morphon(1, CellType::Sensory);
        let s2 = make_morphon(2, CellType::Sensory);
        let s3 = make_morphon(3, CellType::Sensory);

        let mut morphons = HashMap::new();
        morphons.insert(10, target);
        morphons.insert(1, s1);
        morphons.insert(2, s2);
        morphons.insert(3, s3);

        let mut topo = Topology::new();
        for &id in &[1, 2, 3, 10] {
            topo.add_morphon(id);
        }
        topo.add_synapse(1, 10, Synapse::new(0.5));
        topo.add_synapse(2, 10, Synapse::new(0.5));
        topo.add_synapse(3, 10, Synapse::new(0.5));

        let params = MorphogenesisParams::default();
        let count = transdifferentiation(&mut morphons, &topo, &params);

        assert_eq!(count, 1);
        let m = morphons.get(&10).unwrap();
        assert_eq!(m.cell_type, CellType::Associative);
        assert_eq!(m.activation_fn, ActivationFn::LeakyIntegrator);
    }

    #[test]
    fn transdifferentiation_skips_low_desire() {
        let pos = HyperbolicPoint { coords: vec![0.1, 0.0, 0.0], curvature: 1.0 };

        let mut target = Morphon::new(10, pos.clone());
        target.cell_type = CellType::Modulatory;
        target.differentiation_level = 0.5;
        target.age = 600;
        target.desire = 0.1;

        let s1 = make_morphon(1, CellType::Sensory);

        let mut morphons = HashMap::new();
        morphons.insert(10, target);
        morphons.insert(1, s1);

        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(10);
        topo.add_synapse(1, 10, Synapse::new(0.5));

        let params = MorphogenesisParams::default();
        let count = transdifferentiation(&mut morphons, &topo, &params);
        assert_eq!(count, 0);
    }

    #[test]
    fn transdifferentiation_skips_io_boundary() {
        let pos = HyperbolicPoint { coords: vec![0.1, 0.0, 0.0], curvature: 1.0 };

        let mut motor = Morphon::new(10, pos.clone());
        motor.cell_type = CellType::Motor;
        motor.differentiation_level = 0.5;
        motor.age = 600;
        motor.desire = 0.8;

        let s1 = make_morphon(1, CellType::Sensory);

        let mut morphons = HashMap::new();
        morphons.insert(10, motor);
        morphons.insert(1, s1);

        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(10);
        topo.add_synapse(1, 10, Synapse::new(0.5));

        let params = MorphogenesisParams::default();
        let count = transdifferentiation(&mut morphons, &topo, &params);
        assert_eq!(count, 0);
    }

    #[test]
    fn fusion_creates_energy_pool() {
        let (mut morphons, mut topology, params, mut cluster_id, mut morphon_id) = setup_fusion_test();

        // Set up conditions for fusion: high activity, correlated firing, low PE
        for m in morphons.values_mut() {
            for _ in 0..50 { m.activity_history.push(1.0); }
            m.fired = true;
            m.prediction_error = 0.01;
            m.desire = 0.1;
            m.energy = 0.9;
        }

        let mut clusters = HashMap::new();
        let fused = fusion(
            &mut morphons, &mut clusters, &mut cluster_id, &mut morphon_id,
            &mut topology, &params, DEFAULT_MAX_MORPHONS, 0.3, 0.5,
            &crate::homeostasis::CompetitionMode::default(),
        );

        if fused > 0 {
            let cluster = clusters.values().next().unwrap();
            assert!(cluster.shared_energy_pool > 0.0,
                "cluster should have pooled energy: {}", cluster.shared_energy_pool);
            // Members should have reduced energy (contributed 30%)
            for &mid in &cluster.members {
                if let Some(m) = morphons.get(&mid) {
                    assert!(m.energy < 0.9, "member energy should be reduced after pooling");
                }
            }
        }
    }

    #[test]
    fn defusion_returns_energy() {
        let (mut morphons, mut topology, params, mut cluster_id, mut morphon_id) = setup_fusion_test();

        // Force fusion
        for m in morphons.values_mut() {
            for _ in 0..50 { m.activity_history.push(1.0); }
            m.fired = true;
            m.prediction_error = 0.01;
            m.desire = 0.1;
            m.energy = 0.9;
        }

        let mut clusters = HashMap::new();
        fusion(
            &mut morphons, &mut clusters, &mut cluster_id, &mut morphon_id,
            &mut topology, &params, DEFAULT_MAX_MORPHONS, 0.3, 0.5,
            &crate::homeostasis::CompetitionMode::default(),
        );

        if !clusters.is_empty() {
            // Record energy before defusion
            let member_ids: Vec<MorphonId> = clusters.values()
                .next().unwrap().members.clone();
            let energy_before: Vec<f64> = member_ids.iter()
                .filter_map(|mid| morphons.get(mid))
                .map(|m| m.energy)
                .collect();

            // Force defusion by giving high PE variance
            for (i, &mid) in member_ids.iter().enumerate() {
                if let Some(m) = morphons.get_mut(&mid) {
                    m.prediction_error = if i % 2 == 0 { 0.9 } else { 0.0 };
                }
            }

            let defused = defusion(&mut morphons, &mut clusters, &mut topology);
            if defused > 0 {
                // Energy should have been returned
                for (i, &mid) in member_ids.iter().enumerate() {
                    if let Some(m) = morphons.get(&mid) {
                        assert!(m.fused_with.is_none(), "member should be defused");
                        assert!(m.energy >= energy_before[i],
                            "member energy should not decrease after defusion");
                    }
                }
            }
        }
    }

    /// Helper: set up a minimal scenario for fusion testing
    fn setup_fusion_test() -> (HashMap<MorphonId, Morphon>, Topology, MorphogenesisParams, ClusterId, MorphonId) {
        let mut morphons = HashMap::new();
        let mut topology = Topology::new();

        // Create 4 connected morphons
        for i in 0..4u64 {
            let pos = HyperbolicPoint::origin(4);
            let mut m = Morphon::new(i, pos);
            m.cell_type = CellType::Associative;
            m.differentiation_level = 0.5;
            topology.add_morphon(i);
            morphons.insert(i, m);
        }

        // Connect them densely
        for i in 0..4u64 {
            for j in 0..4u64 {
                if i != j {
                    topology.add_synapse(i, j, Synapse::new(0.3));
                }
            }
        }

        let params = MorphogenesisParams {
            fusion_min_size: 3,
            fusion_correlation_threshold: 0.75,
            ..Default::default()
        };

        (morphons, topology, params, 0, 100)
    }

    // === Distance-dependent delay in synaptogenesis ===

    #[test]
    fn synaptogenesis_sets_delay_from_distance() {
        use crate::topology::Topology;

        let mut morphons = HashMap::new();
        let mut topology = Topology::new();
        let mut rng = rand::rng();

        // Place two morphons close together (distance ≈ 0)
        let origin = HyperbolicPoint { coords: vec![0.0, 0.0, 0.0], curvature: 1.0 };
        let near = HyperbolicPoint { coords: vec![0.01, 0.0, 0.0], curvature: 1.0 };
        // And one far away (distance > 1.0 in hyperbolic space)
        let far = HyperbolicPoint { coords: vec![0.7, 0.0, 0.0], curvature: 1.0 };

        let mut m0 = Morphon::new(0, origin.clone());
        m0.cell_type = CellType::Associative;
        let mut m1 = Morphon::new(1, near);
        m1.cell_type = CellType::Associative;
        let mut m2 = Morphon::new(2, far);
        m2.cell_type = CellType::Associative;

        for m in [&m0, &m1, &m2] {
            topology.add_morphon(m.id);
        }
        morphons.insert(0, m0);
        morphons.insert(1, m1);
        morphons.insert(2, m2);

        let params = MorphogenesisParams::default();

        // Run synaptogenesis many times to ensure connections form
        for _ in 0..500 {
            synaptogenesis(&morphons, &mut topology, &params, &mut rng, 50, 0);
        }

        // Check that formed synapses have distance-dependent delays
        if let Some((_, syn_near)) = topology.synapse_between(0, 1)
            .or(topology.synapse_between(1, 0))
        {
            // Close morphons: distance ≈ 0.02, delay ≈ 0.5 + 0.02*0.75 ≈ 0.515
            assert!(syn_near.delay < 1.0,
                "near synapse should have short delay, got {}", syn_near.delay);
            assert!(syn_near.delay >= 0.5,
                "delay floor is 0.5, got {}", syn_near.delay);
        }

        if let Some((_, syn_far)) = topology.synapse_between(0, 2)
            .or(topology.synapse_between(2, 0))
        {
            // Far morphons: distance > 1.0, delay > 0.5 + 1.0*0.75 = 1.25
            assert!(syn_far.delay > 1.0,
                "far synapse should have longer delay, got {}", syn_far.delay);
        }
    }

    #[test]
    fn synaptogenesis_delay_myelination_chain() {
        // Verify the full chain: distance → delay → myelination → reduced effective_delay
        let mut syn = Synapse::new(0.5).with_delay(2.0); // distant synapse
        let base_effective = syn.effective_delay();
        assert!((base_effective - 2.0).abs() < 1e-10);

        // Simulate consolidation + activity → myelination (neutral context)
        let ctx = crate::morphon::MyelinationContext {
            arousal: 0.0, reward: 0.0, energy_pressure: 0.0,
        };
        syn.consolidation_level = 1.0;
        syn.activity_trace = 1.0;
        for _ in 0..5000 {
            syn.update_myelination(1.0, &ctx);
        }

        assert!(syn.myelination > 0.3,
            "should have meaningful myelination after 5000 steps: {}", syn.myelination);
        assert!(syn.effective_delay() < base_effective,
            "myelination should reduce effective delay: {} vs {}", syn.effective_delay(), base_effective);
        assert!(syn.effective_delay() >= 0.5,
            "effective delay floor respected: {}", syn.effective_delay());
    }
}
