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

    // Create initial random connectivity
    let ids: Vec<MorphonId> = morphons.keys().copied().collect();
    for &from in &ids {
        for &to in &ids {
            if from == to {
                continue;
            }
            if rng.random_range(0.0..1.0) < config.initial_connectivity {
                let weight = rng.random_range(-0.5..0.5);
                let delay = rng.random_range(0.1..1.0);
                topology.add_synapse(from, to, Synapse::new(weight).with_delay(delay));
            }
        }
    }

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
                topology.duplicate_connections(parent_id, child_id, rng);
                morphons.insert(child_id, child);
            }
        }
    }

    // === Phase 3: Positional Differentiation ===
    // Assign cell types based on position in information space
    // (simulating morphogen gradients)
    differentiate_by_position(&mut morphons, config);

    // === Phase 4: Ensure I/O Pathways ===
    // Create strong feedforward connections: Sensory → Associative → Motor
    // Without this, random connectivity may not create viable signal paths.
    {
        let sensory: Vec<MorphonId> = morphons.values()
            .filter(|m| m.cell_type == CellType::Sensory).map(|m| m.id).collect();
        let associative: Vec<MorphonId> = morphons.values()
            .filter(|m| m.cell_type == CellType::Associative).map(|m| m.id).collect();
        let motor: Vec<MorphonId> = morphons.values()
            .filter(|m| m.cell_type == CellType::Motor).map(|m| m.id).collect();

        // Each sensory connects to a few associative morphons
        for (i, &s) in sensory.iter().enumerate() {
            let targets = &associative;
            for j in 0..3.min(targets.len()) {
                let t = targets[(i * 3 + j) % targets.len()];
                if !topology.has_connection(s, t) {
                    let w = rng.random_range(0.3..0.8);
                    topology.add_synapse(s, t, Synapse::new(w).with_delay(0.1));
                }
            }
        }
        // Each associative connects to a few motor morphons
        for (i, &a) in associative.iter().enumerate() {
            let targets = &motor;
            for j in 0..3.min(targets.len()) {
                let t = targets[(i * 3 + j) % targets.len()];
                if !topology.has_connection(a, t) {
                    let w = rng.random_range(0.3..0.8);
                    topology.add_synapse(a, t, Synapse::new(w).with_delay(0.1));
                }
            }
        }
        // A few direct sensory → motor connections (shortcut)
        for (i, &s) in sensory.iter().enumerate() {
            if i < motor.len() {
                let t = motor[i % motor.len()];
                if !topology.has_connection(s, t) {
                    let w = rng.random_range(0.2..0.5);
                    topology.add_synapse(s, t, Synapse::new(w).with_delay(0.1));
                }
            }
        }
    }

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
    let n_modulatory = ((total.saturating_sub(n_sensory).saturating_sub(n_motor)) as f64
        * config.type_ratios.modulatory
        / (config.type_ratios.associative + config.type_ratios.modulatory).max(0.01))
        as usize;
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
