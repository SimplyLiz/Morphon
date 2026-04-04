//! V3 Governance Layer — Constitutional Constraints.
//!
//! Hard invariants that lie **outside the learning loop** and cannot be modified
//! by the system itself.  Only a human oracle (or explicit API call) can amend
//! them.  Biological analogy: DNA-coded checkpoint programs that epigenetic
//! modification cannot alter.

use crate::morphogenesis::Cluster;
use crate::morphon::Morphon;
use crate::topology::Topology;
use crate::types::{CellType, MorphonId};
use serde::{Deserialize, Serialize};

/// Constitutional constraints — the "fundamental laws" of the system.
///
/// These are enforced at every structural decision point (synaptogenesis,
/// division, fusion, apoptosis) and override any learned behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionalConstraints {
    /// Maximum number of connections (in + out) per morphon.
    /// Prevents "superhub" pathology where one morphon dominates.
    pub max_connectivity_per_morphon: usize,

    /// No single cluster may exceed this fraction of total morphons.
    pub max_cluster_size_fraction: f64,

    /// Maximum fraction of the system allowed in Hypothesis (unverified) state.
    /// Phase 1 soft limit — logged but not enforced until Phase 2.
    pub max_unverified_fraction: f64,

    /// Cell types that MUST have justification records on their synapses.
    pub mandatory_justification_for: Vec<CellType>,

    /// Maximum depth for cascade invalidation traversal.
    pub cascade_depth_limit: usize,

    /// Maximum fraction of clusters allowed to fuse per glacial epoch.
    pub max_fusion_rate_per_epoch: f64,

    /// Total structural change budget per glacial epoch (division + fusion + apoptosis).
    pub max_structural_changes_per_epoch: usize,

    /// Minimum energy floor — morphon energy never drops below this.
    /// Prevents total starvation while still allowing energy-based selection.
    pub energy_floor: f64,

    /// Maximum morphon population cap.
    /// `None` = auto-derive from I/O dimensions: `max(500, (input + output) * 3)`.
    /// `Some(n)` = explicit override. The system cannot raise its own cap.
    pub max_morphons: Option<usize>,
}

impl Default for ConstitutionalConstraints {
    fn default() -> Self {
        Self {
            max_connectivity_per_morphon: 50,
            max_cluster_size_fraction: 0.3,
            max_unverified_fraction: 0.5,
            mandatory_justification_for: vec![CellType::Motor],
            cascade_depth_limit: 10,
            max_fusion_rate_per_epoch: 0.1,
            max_structural_changes_per_epoch: 50,
            energy_floor: 0.0, // permissive default — existing behavior
            max_morphons: None, // auto-derive from I/O dimensions
        }
    }
}

/// Returns `true` if the morphon can accept another connection without
/// violating the connectivity cap.
pub fn check_connectivity(
    topology: &Topology,
    morphon_id: MorphonId,
    max: usize,
) -> bool {
    let degree = topology.degree(morphon_id);
    degree < max
}

/// Returns `true` if the cluster size is within the allowed fraction.
pub fn check_cluster_size(
    cluster: &Cluster,
    total_morphons: usize,
    max_fraction: f64,
) -> bool {
    if total_morphons == 0 {
        return true;
    }
    (cluster.members.len() as f64 / total_morphons as f64) <= max_fraction
}

/// Clamp a morphon's energy to at least `floor`.
pub fn enforce_energy_floor(morphon: &mut Morphon, floor: f64) {
    if morphon.energy < floor {
        morphon.energy = floor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphon::Morphon;
    use crate::types::HyperbolicPoint;

    #[test]
    fn default_constraints_are_permissive() {
        let c = ConstitutionalConstraints::default();
        assert_eq!(c.max_connectivity_per_morphon, 50);
        assert_eq!(c.energy_floor, 0.0);
        assert_eq!(c.max_structural_changes_per_epoch, 50);
    }

    #[test]
    fn check_cluster_size_enforces_fraction() {
        use crate::morphogenesis::Cluster;
        let cluster = Cluster {
            id: 1,
            members: vec![1, 2, 3, 4],
            shared_threshold: 0.5,
            inhibitory_morphons: vec![],
            shared_energy_pool: 0.0,
            shared_homeostatic_setpoint: 0.15,
            epistemic_state: Default::default(),
            epistemic_history: Default::default(),
        };
        // 4 out of 10 = 0.4 > 0.3 → should fail
        assert!(!check_cluster_size(&cluster, 10, 0.3));
        // 4 out of 20 = 0.2 ≤ 0.3 → should pass
        assert!(check_cluster_size(&cluster, 20, 0.3));
    }

    #[test]
    fn energy_floor_enforced() {
        let pos = HyperbolicPoint { coords: vec![0.0; 3], curvature: 1.0 };
        let mut m = Morphon::new(1, pos);
        m.energy = 0.01;
        enforce_energy_floor(&mut m, 0.05);
        assert!((m.energy - 0.05).abs() < 1e-10);
    }

    #[test]
    fn energy_floor_no_op_when_above() {
        let pos = HyperbolicPoint { coords: vec![0.0; 3], curvature: 1.0 };
        let mut m = Morphon::new(1, pos);
        m.energy = 0.5;
        enforce_energy_floor(&mut m, 0.05);
        assert!((m.energy - 0.5).abs() < 1e-10);
    }
}
