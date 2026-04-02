//! Developmental Programs — bootstrapping MI systems from a seed.
//!
//! Instead of random initialization, MI systems grow through an
//! embryogenesis-inspired developmental program:
//! 1. Seed phase: small number of Morphons with minimal connectivity
//! 2. Proliferation: cell division creates more Morphons
//! 3. Differentiation: regions develop different characteristics
//! 4. Pruning: excess connections are refined

use crate::morphon::{Morphon, Synapse};
use crate::topology::Topology;
use crate::types::*;
use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for a developmental program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevelopmentalConfig {
    /// Which program to use.
    pub program: DevelopmentalProgram,
    /// Number of initial seed Morphons.
    pub seed_size: usize,
    /// Dimensionality of the information space.
    pub dimensions: usize,
    /// Initial connectivity probability between seed Morphons.
    pub initial_connectivity: f64,
    /// Number of proliferation rounds.
    pub proliferation_rounds: usize,
    /// Target ratio of each cell type after differentiation.
    pub type_ratios: CellTypeRatios,
    /// If set, overrides the number of Sensory morphons to match the task input size.
    /// The developmental program will create exactly this many Sensory morphons,
    /// adjusting seed_size upward if necessary.
    pub target_input_size: Option<usize>,
    /// If set, overrides the number of Motor morphons to match the task output size.
    pub target_output_size: Option<usize>,
}

/// Target ratios of cell types (must sum to 1.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellTypeRatios {
    pub sensory: f64,
    pub associative: f64,
    pub motor: f64,
    pub modulatory: f64,
}

impl Default for DevelopmentalConfig {
    fn default() -> Self {
        Self {
            program: DevelopmentalProgram::Cortical,
            seed_size: 100,
            dimensions: 8,
            initial_connectivity: 0.1,
            proliferation_rounds: 3,
            type_ratios: CellTypeRatios {
                sensory: 0.2,
                associative: 0.5,
                motor: 0.2,
                modulatory: 0.1,
            },
            target_input_size: None,
            target_output_size: None,
        }
    }
}

impl DevelopmentalConfig {
    /// Preset: Cortical — for classification and pattern recognition.
    pub fn cortical() -> Self {
        Self {
            program: DevelopmentalProgram::Cortical,
            type_ratios: CellTypeRatios {
                sensory: 0.2,
                associative: 0.5,
                motor: 0.2,
                modulatory: 0.1,
            },
            ..Default::default()
        }
    }

    /// Preset: Hippocampal — for sequential learning and time series.
    pub fn hippocampal() -> Self {
        Self {
            program: DevelopmentalProgram::Hippocampal,
            type_ratios: CellTypeRatios {
                sensory: 0.3,
                associative: 0.4,
                motor: 0.1,
                modulatory: 0.2,
            },
            ..Default::default()
        }
    }

    /// Preset: Cerebellar — for motor control and robotics.
    pub fn cerebellar() -> Self {
        Self {
            program: DevelopmentalProgram::Cerebellar,
            type_ratios: CellTypeRatios {
                sensory: 0.15,
                associative: 0.3,
                motor: 0.45,
                modulatory: 0.1,
            },
            ..Default::default()
        }
    }
}

/// Execute the developmental program to bootstrap an MI system.
///
/// Returns the initial set of Morphons, Topology, and the next available ID.
pub fn develop(
    config: &DevelopmentalConfig,
    homeostasis: &crate::homeostasis::HomeostasisParams,
    rng: &mut impl Rng,
) -> (HashMap<MorphonId, Morphon>, Topology, MorphonId) {
    let mut morphons = HashMap::new();
    let mut topology = Topology::new();
    let mut next_id: MorphonId = 0;

    // Auto-scale seed size if target I/O sizes require more morphons.
    // Need at least input + output + 20% interior morphons.
    let min_io = config.target_input_size.unwrap_or(0) + config.target_output_size.unwrap_or(0);
    let effective_seed = config.seed_size.max((min_io as f64 * 1.2) as usize);

    // === Phase 1: Seed ===
    for _ in 0..effective_seed {
        let id = next_id;
        next_id += 1;

        let position = Position::random(config.dimensions, rng);
        let morphon = Morphon::new(id, position);

        topology.add_morphon(id);
        morphons.insert(id, morphon);
    }

    // Skip random connectivity before differentiation — it creates connections
    // that violate the cell type hierarchy (e.g. into Sensory, out of Motor).
    // The I/O pathway phase (Phase 4) creates the feedforward structure,
    // and synaptogenesis grows additional connections during runtime.

    // === Phase 2: Proliferation ===
    // Controlled cell division to reach target size
    for _round in 0..config.proliferation_rounds {
        let current_ids: Vec<MorphonId> = morphons.keys().copied().collect();
        for parent_id in current_ids {
            // Each morphon has a chance to divide
            if rng.random_range(0.0..1.0) < 0.3 {
                let child_id = next_id;
                next_id += 1;

                let child = morphons[&parent_id].divide(child_id, rng);
                topology.add_morphon(child_id);
                topology.duplicate_connections(parent_id, child_id, rng, 0);
                morphons.insert(child_id, child);
            }
        }
    }

    // === Phase 3: Positional Differentiation ===
    // Assign cell types based on position in information space
    // (simulating morphogen gradients)
    differentiate_by_position(&mut morphons, config);

    // === Phase 4: Dense I/O Pathways ===
    // LNDP research shows 70-80% connectivity density is needed for RL tasks.
    // We create dense feedforward connections: Sensory → Associative → Motor.
    // Each motor morphon must "see" a large fraction of the input through its
    // associative fan-in. Each associative must receive from many sensory.
    {
        let sensory: Vec<MorphonId> = morphons.values()
            .filter(|m| m.cell_type == CellType::Sensory).map(|m| m.id).collect();
        let associative: Vec<MorphonId> = morphons.values()
            .filter(|m| m.cell_type == CellType::Associative).map(|m| m.id).collect();
        let motor: Vec<MorphonId> = morphons.values()
            .filter(|m| m.cell_type == CellType::Motor).map(|m| m.id).collect();

        // Fan-in from sensory → associative: each associative receives from ~30% of sensory.
        // Use fan-IN perspective (iterate over associative, sample sensory) to ensure
        // uniform receptive field sizes. The old fan-OUT approach created uneven
        // distributions where some associative morphons received 2-3x more input,
        // seeding hub dominance from initialization.
        let sens_per_assoc = (sensory.len() as f64 * 0.3).ceil() as usize;

        for &a in &associative {
            // Shuffle sensory and take first sens_per_assoc
            let mut sens_shuffled = sensory.clone();
            sens_shuffled.shuffle(rng);
            for &s in sens_shuffled.iter().take(sens_per_assoc) {
                if !topology.has_connection(s, a) {
                    let w = rng.random_range(0.3..0.8); // positive, above threshold
                    let j = crate::justification::SynapticJustification::new(
                        crate::justification::FormationCause::External { source: "developmental".into() },
                        0,
                    );
                    topology.add_synapse(s, a, Synapse::new_justified(w, j).with_delay(0.1));
                }
            }
        }

        // Fan-in from associative → motor: EVERY motor connects to ALL associative.
        // Small random Xavier-scaled weights — breaks symmetry so each motor class
        // starts with a unique receptive field. Zero init causes mode collapse
        // (all motors respond identically → first class to win locks in).
        let assoc_to_motor_scale = 1.0 / (associative.len() as f64).max(1.0).sqrt();
        for &a in &associative {
            for &m in &motor {
                if !topology.has_connection(a, m) {
                    let w = rng.random_range(-assoc_to_motor_scale..assoc_to_motor_scale);
                    let j = crate::justification::SynapticJustification::new(
                        crate::justification::FormationCause::External { source: "developmental".into() },
                        0,
                    );
                    topology.add_synapse(a, m, Synapse::new_justified(w, j).with_delay(0.1));
                }
            }
        }

        // Direct sensory → motor shortcuts (sparse, small random).
        let direct_scale = 0.1 / (sensory.len() as f64).max(1.0).sqrt();
        for (i, &s) in sensory.iter().enumerate() {
            let t = motor[i % motor.len().max(1)];
            if !topology.has_connection(s, t) {
                let w = rng.random_range(-direct_scale..direct_scale);
                let j = crate::justification::SynapticJustification::new(
                    crate::justification::FormationCause::External { source: "developmental".into() },
                    0,
                );
                topology.add_synapse(s, t, Synapse::new_justified(w, j).with_delay(0.1));
            }
        }
    }

    // === Phase 4.5: Local Inhibitory Interneurons (iSTDP competition) ===
    // When CompetitionMode::LocalInhibition, create inhibitory interneurons
    // wired to Associative morphons for emergent lateral inhibition.
    create_local_inhibitory_interneurons(
        &mut morphons,
        &mut topology,
        &mut next_id,
        &homeostasis.competition_mode,
        rng,
    );

    // === Phase 5: Refinement Pruning ===
    // Remove weakest random connections to sharpen the network
    let edges = topology.all_edges();
    for (_, _, ei) in edges {
        if let Some(syn) = topology.graph.edge_weight(ei) {
            if syn.weight.abs() < 0.05 {
                topology.remove_synapse(ei);
            }
        }
    }

    (morphons, topology, next_id)
}

/// Create intra-group inhibitory interneurons for local competition (iSTDP).
///
/// Groups Associative morphons by spatial proximity in the Poincare ball,
/// creates one InhibitoryInterneuron per group, and wires bidirectionally:
/// - Excitatory: Assoc → Interneuron (drives the interneuron when the group is active)
/// - Inhibitory: Interneuron → Assoc (suppresses the group, creating competition)
///
/// iSTDP (Vogels et al. 2011) will tune the inhibitory weights during training
/// to maintain a target firing rate — no explicit k-WTA parameter needed.
fn create_local_inhibitory_interneurons(
    morphons: &mut HashMap<MorphonId, Morphon>,
    topology: &mut Topology,
    next_id: &mut MorphonId,
    competition_mode: &crate::homeostasis::CompetitionMode,
    _rng: &mut impl Rng,
) {
    let (interneuron_ratio, initial_inh_weight, inhibition_radius) = match competition_mode {
        crate::homeostasis::CompetitionMode::LocalInhibition {
            interneuron_ratio, initial_inh_weight, inhibition_radius, ..
        } => (*interneuron_ratio, *initial_inh_weight, *inhibition_radius),
        _ => return, // Only activate for LocalInhibition mode
    };

    // Collect Associative morphon IDs and positions
    let assoc_data: Vec<(MorphonId, HyperbolicPoint)> = morphons.values()
        .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
        .map(|m| (m.id, m.position.clone()))
        .collect();

    if assoc_data.is_empty() {
        return;
    }

    let num_interneurons = (assoc_data.len() as f64 * interneuron_ratio).ceil() as usize;
    if num_interneurons == 0 {
        return;
    }

    // Simple grouping: evenly partition Assoc morphons, one interneuron per group.
    // Each interneuron is positioned at the centroid of its group.
    let group_size = (assoc_data.len() + num_interneurons - 1) / num_interneurons;

    for chunk in assoc_data.chunks(group_size) {
        let dim = chunk[0].1.coords.len();

        // Compute centroid of this group
        let mut centroid = vec![0.0; dim];
        for (_, pos) in chunk {
            for (i, c) in pos.coords.iter().enumerate() {
                centroid[i] += c;
            }
        }
        for c in &mut centroid {
            *c /= chunk.len() as f64;
        }
        // Clamp inside Poincare ball
        let norm: f64 = centroid.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.85 {
            let scale = 0.85 / norm;
            for c in &mut centroid {
                *c *= scale;
            }
        }

        let position = HyperbolicPoint { coords: centroid, curvature: 1.0 };
        let inh_id = *next_id;
        *next_id += 1;

        let mut inh_morphon = Morphon::new(inh_id, position.clone());
        inh_morphon.cell_type = CellType::InhibitoryInterneuron;
        inh_morphon.activation_fn = ActivationFn::Sigmoid;
        inh_morphon.receptors = default_receptors(CellType::InhibitoryInterneuron);
        inh_morphon.differentiation_level = 1.0; // terminally differentiated
        inh_morphon.homeostatic_setpoint = 0.3; // interneurons should be active

        topology.add_morphon(inh_id);

        // Wire bidirectionally to group members (or all Assoc if radius = 0)
        for (assoc_id, assoc_pos) in chunk {
            let in_range = inhibition_radius <= 0.0
                || position.distance(assoc_pos) < inhibition_radius;
            if !in_range {
                continue;
            }

            // Excitatory: Assoc → Interneuron (drives the interneuron)
            let exc_syn = Synapse::new(0.3).with_delay(0.5);
            topology.add_synapse(*assoc_id, inh_id, exc_syn);

            // Inhibitory: Interneuron → Assoc (competition)
            let inh_syn = Synapse::new(initial_inh_weight).with_delay(0.5);
            topology.add_synapse(inh_id, *assoc_id, inh_syn);
        }

        morphons.insert(inh_id, inh_morphon);
    }
}

// === V2: Target Morphology ===

/// A target region in the information space.
/// Defines what the system should look like in a spatial area.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetRegion {
    /// Human-readable name for this region.
    pub name: String,
    /// Center of the region in hyperbolic space.
    pub center: HyperbolicPoint,
    /// Radius in hyperbolic distance units.
    pub radius: f64,
    /// The cell type that morphons in this region should differentiate toward.
    pub target_cell_type: CellType,
    /// Target number of morphons in the region.
    pub target_density: usize,
    /// Target average degree (in+out connections) per morphon in the region.
    pub target_connectivity: f64,
    /// How strongly the Identity field broadcasts this region's presence.
    pub identity_strength: f64,
}

/// The complete target morphology: describes the desired spatial organization.
/// When active, the system actively maintains these targets — if morphons
/// die or migrate away, new ones are recruited via division pressure or seeding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetMorphology {
    /// All target regions.
    pub regions: Vec<TargetRegion>,
    /// Whether self-healing is enabled (recruit morphons to underpopulated regions).
    pub self_healing: bool,
    /// Minimum population deficit ratio to trigger recruitment.
    /// 0.5 = region is at half its target density or less.
    pub healing_threshold: f64,
}

impl Default for TargetMorphology {
    fn default() -> Self {
        Self {
            regions: Vec::new(),
            self_healing: true,
            healing_threshold: 0.5,
        }
    }
}

impl TargetMorphology {
    /// Cortical template: sensory periphery, associative core, motor output region.
    pub fn cortical(dimensions: usize) -> Self {
        // Build direction vectors for region centers
        let mut sensory_dir = vec![0.0; dimensions];
        sensory_dir[0] = 0.6; // positive x direction
        let mut motor_dir = vec![0.0; dimensions];
        motor_dir[0] = -0.6; // negative x direction

        let origin = HyperbolicPoint::origin(dimensions);

        Self {
            regions: vec![
                TargetRegion {
                    name: "sensory_cortex".into(),
                    center: origin.exp_map(&sensory_dir),
                    radius: 0.5,
                    target_cell_type: CellType::Sensory,
                    target_density: 20,
                    target_connectivity: 5.0,
                    identity_strength: 1.0,
                },
                TargetRegion {
                    name: "associative_core".into(),
                    center: origin.clone(),
                    radius: 0.6,
                    target_cell_type: CellType::Associative,
                    target_density: 50,
                    target_connectivity: 8.0,
                    identity_strength: 0.5,
                },
                TargetRegion {
                    name: "motor_output".into(),
                    center: origin.exp_map(&motor_dir),
                    radius: 0.5,
                    target_cell_type: CellType::Motor,
                    target_density: 20,
                    target_connectivity: 5.0,
                    identity_strength: 1.0,
                },
            ],
            self_healing: true,
            healing_threshold: 0.5,
        }
    }

    /// Cerebellar template: large motor region, compact sensory input.
    pub fn cerebellar(dimensions: usize) -> Self {
        let mut sensory_dir = vec![0.0; dimensions];
        sensory_dir[0] = 0.6;
        let mut motor_dir = vec![0.0; dimensions];
        motor_dir[0] = -0.4;
        let origin = HyperbolicPoint::origin(dimensions);

        Self {
            regions: vec![
                TargetRegion {
                    name: "sensory_input".into(),
                    center: origin.exp_map(&sensory_dir),
                    radius: 0.3,
                    target_cell_type: CellType::Sensory,
                    target_density: 15,
                    target_connectivity: 4.0,
                    identity_strength: 0.8,
                },
                TargetRegion {
                    name: "associative_bridge".into(),
                    center: origin.clone(),
                    radius: 0.4,
                    target_cell_type: CellType::Associative,
                    target_density: 30,
                    target_connectivity: 6.0,
                    identity_strength: 0.3,
                },
                TargetRegion {
                    name: "motor_cortex".into(),
                    center: origin.exp_map(&motor_dir),
                    radius: 0.6,
                    target_cell_type: CellType::Motor,
                    target_density: 45,
                    target_connectivity: 8.0,
                    identity_strength: 1.0,
                },
            ],
            self_healing: true,
            healing_threshold: 0.5,
        }
    }

    /// Hippocampal template: sequential processing chain.
    pub fn hippocampal(dimensions: usize) -> Self {
        let mut sensory_dir = vec![0.0; dimensions];
        sensory_dir[0] = 0.6;
        let mut assoc_dir = vec![0.0; dimensions];
        if dimensions > 1 { assoc_dir[1] = 0.3; }
        let mut motor_dir = vec![0.0; dimensions];
        motor_dir[0] = -0.6;
        let origin = HyperbolicPoint::origin(dimensions);

        Self {
            regions: vec![
                TargetRegion {
                    name: "input_gate".into(),
                    center: origin.exp_map(&sensory_dir),
                    radius: 0.4,
                    target_cell_type: CellType::Sensory,
                    target_density: 30,
                    target_connectivity: 4.0,
                    identity_strength: 0.8,
                },
                TargetRegion {
                    name: "processing_chain".into(),
                    center: origin.exp_map(&assoc_dir),
                    radius: 0.5,
                    target_cell_type: CellType::Associative,
                    target_density: 40,
                    target_connectivity: 6.0,
                    identity_strength: 0.5,
                },
                TargetRegion {
                    name: "output_gate".into(),
                    center: origin.exp_map(&motor_dir),
                    radius: 0.3,
                    target_cell_type: CellType::Motor,
                    target_density: 10,
                    target_connectivity: 5.0,
                    identity_strength: 0.8,
                },
            ],
            self_healing: true,
            healing_threshold: 0.5,
        }
    }

    /// Count morphons in each region and return health status.
    /// Returns Vec of (region_index, current_count, target_density).
    pub fn region_health(&self, morphons: &HashMap<MorphonId, Morphon>) -> Vec<(usize, usize, usize)> {
        self.regions.iter().enumerate().map(|(i, region)| {
            let count = morphons.values()
                .filter(|m| m.position.distance(&region.center) <= region.radius)
                .count();
            (i, count, region.target_density)
        }).collect()
    }
}

/// Self-healing: compare actual vs target state per region.
/// Boost division pressure for underpopulated regions, seed new morphons
/// if a region is completely empty.
/// Returns the number of healing actions taken.
pub fn target_morphology_heal(
    target: &TargetMorphology,
    morphons: &mut HashMap<MorphonId, Morphon>,
    topology: &mut crate::topology::Topology,
    next_id: &mut MorphonId,
    max_morphons: usize,
) -> usize {
    let mut actions = 0;

    for region in &target.regions {
        let in_region: Vec<MorphonId> = morphons.values()
            .filter(|m| m.position.distance(&region.center) <= region.radius)
            .map(|m| m.id)
            .collect();

        let current = in_region.len();
        let deficit_ratio = current as f64 / region.target_density.max(1) as f64;

        if deficit_ratio < target.healing_threshold {
            // Boost division pressure scaled by deficit severity.
            // Mild deficit (+0.1), severe deficit (+0.5) — enables faster recovery.
            let deficit = 1.0 - deficit_ratio;
            let boost = 0.1 + deficit * 0.4;
            for &id in &in_region {
                if let Some(m) = morphons.get_mut(&id) {
                    m.division_pressure += boost;
                    actions += 1;
                }
            }

            // If region is empty, seed a new morphon near the center
            if current == 0 && morphons.len() < max_morphons {
                let new_id = *next_id;
                *next_id += 1;
                // Small random offset from center
                let tangent: Vec<f64> = (0..region.center.coords.len())
                    .map(|i| {
                        let hash = (new_id.wrapping_mul(7919).wrapping_add(i as u64) % 1000) as f64;
                        (hash / 500.0 - 1.0) * 0.05
                    })
                    .collect();
                let pos = region.center.exp_map(&tangent);
                let mut m = Morphon::new(new_id, pos);
                m.differentiate(region.target_cell_type);
                m.differentiation_level = 0.8; // pre-committed, resists dedifferentiation
                m.age = 200; // bypass the age >= 200 differentiation gate
                topology.add_morphon(new_id);
                morphons.insert(new_id, m);
                actions += 1;
            }
        }
    }

    actions
}

/// Assign cell types based on position (morphogen gradient simulation).
/// If target_input_size / target_output_size are set, those exact counts are used.
fn differentiate_by_position(
    morphons: &mut HashMap<MorphonId, Morphon>,
    config: &DevelopmentalConfig,
) {
    let mut sorted: Vec<MorphonId> = morphons.keys().copied().collect();
    sorted.sort_by(|a, b| {
        let pa = morphons[a].position.coords.first().copied().unwrap_or(0.0);
        let pb = morphons[b].position.coords.first().copied().unwrap_or(0.0);
        pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let total = sorted.len();

    // Compute actual counts — override with target sizes if specified
    let n_sensory = config.target_input_size
        .unwrap_or((total as f64 * config.type_ratios.sensory) as usize)
        .min(total);
    let n_motor = config.target_output_size
        .unwrap_or((total as f64 * config.type_ratios.motor) as usize)
        .min(total.saturating_sub(n_sensory));
    let n_modulatory = {
        let raw = ((total.saturating_sub(n_sensory).saturating_sub(n_motor)) as f64
            * config.type_ratios.modulatory
            / (config.type_ratios.associative + config.type_ratios.modulatory).max(0.01))
            as usize;
        // V3 guard: cap Modulatory at 15% of total to prevent over-representation
        raw.min((total as f64 * 0.15) as usize)
    };
    let n_associative = total.saturating_sub(n_sensory).saturating_sub(n_motor).saturating_sub(n_modulatory);

    let sensory_end = n_sensory;
    let assoc_end = sensory_end + n_associative;
    let motor_end = assoc_end + n_motor;

    for (i, id) in sorted.iter().enumerate() {
        let target = if i < sensory_end {
            CellType::Sensory
        } else if i < assoc_end {
            CellType::Associative
        } else if i < motor_end {
            CellType::Motor
        } else {
            CellType::Modulatory
        };

        if let Some(m) = morphons.get_mut(id) {
            m.differentiate(target);
            m.differentiation_level = 0.6;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::Topology;

    #[test]
    fn cortical_template_has_three_regions() {
        let tm = TargetMorphology::cortical(4);
        assert_eq!(tm.regions.len(), 3);
        assert_eq!(tm.regions[0].target_cell_type, CellType::Sensory);
        assert_eq!(tm.regions[1].target_cell_type, CellType::Associative);
        assert_eq!(tm.regions[2].target_cell_type, CellType::Motor);
    }

    #[test]
    fn cerebellar_template_motor_dominant() {
        let tm = TargetMorphology::cerebellar(4);
        assert_eq!(tm.regions.len(), 3);
        let motor_region = tm.regions.iter().find(|r| r.target_cell_type == CellType::Motor).unwrap();
        let sensory_region = tm.regions.iter().find(|r| r.target_cell_type == CellType::Sensory).unwrap();
        assert!(motor_region.target_density > sensory_region.target_density);
    }

    #[test]
    fn self_healing_seeds_empty_region() {
        let tm = TargetMorphology::cortical(4);
        let mut morphons = HashMap::new();
        let mut topo = Topology::new();
        let mut next_id = 1;

        // No morphons exist — all regions are empty
        let actions = target_morphology_heal(&tm, &mut morphons, &mut topo, &mut next_id, 100);

        // Should have seeded one morphon per empty region
        assert!(actions > 0, "should take healing actions");
        assert!(!morphons.is_empty(), "should have seeded morphons");
        // Verify the seeded morphons have correct cell types
        for m in morphons.values() {
            let in_any_region = tm.regions.iter().any(|r| r.target_cell_type == m.cell_type);
            assert!(in_any_region, "seeded morphon should match a region's target type");
        }
    }

    #[test]
    fn self_healing_boosts_division_in_underpopulated_region() {
        let tm = TargetMorphology::cortical(4);
        let sensory_center = tm.regions[0].center.clone();

        let mut morphons = HashMap::new();
        let mut topo = Topology::new();
        let mut next_id = 100;

        // Place one morphon inside the sensory region (target density = 20)
        let mut m = Morphon::new(1, sensory_center.clone());
        m.differentiate(CellType::Sensory);
        let dp_before = m.division_pressure;
        morphons.insert(1, m);
        topo.add_morphon(1);

        target_morphology_heal(&tm, &mut morphons, &mut topo, &mut next_id, 100);

        // The existing morphon should have boosted division pressure
        assert!(morphons[&1].division_pressure > dp_before,
            "division pressure should be boosted in underpopulated region");
    }

    #[test]
    fn region_health_reports_correctly() {
        let tm = TargetMorphology::cortical(4);
        let sensory_center = tm.regions[0].center.clone();

        let mut morphons = HashMap::new();
        // Place 3 morphons near the sensory region center
        for i in 0..3 {
            let tangent: Vec<f64> = (0..4).map(|d| if d == 0 { 0.01 * i as f64 } else { 0.0 }).collect();
            let pos = sensory_center.exp_map(&tangent);
            morphons.insert(i as u64, Morphon::new(i as u64, pos));
        }

        let health = tm.region_health(&morphons);
        assert_eq!(health.len(), 3);
        // First region (sensory) should have 3 morphons, target 20
        assert_eq!(health[0].0, 0);
        assert_eq!(health[0].1, 3);
        assert_eq!(health[0].2, 20);
    }
}
