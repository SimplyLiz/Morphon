//! V3 Epistemic Model — four-state knowledge tracking with scarring.
//!
//! Every cluster has an epistemic state that reflects the system's confidence
//! in the knowledge encoded by that cluster's synaptic topology.  State
//! transitions are driven by justification records on member synapses.
//!
//! **Epistemic Scarring**: clusters that have been burned (repeatedly Outdated
//! or Contested) develop higher skepticism thresholds, requiring stronger
//! evidence before accepting new beliefs.

use crate::morphogenesis::Cluster;
use crate::morphon::Morphon;
use crate::topology::Topology;
use crate::types::{ClusterId, MorphonId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The four epistemic states a cluster can be in.
///
/// Modeled after TruthKeeper's knowledge lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EpistemicState {
    /// Verified against current evidence — cluster is protected and stable.
    Supported {
        /// Confidence score [0, 1] — fraction of justified, consolidated synapses.
        confidence: f64,
        /// Step when last verified.
        last_verified: u64,
    },

    /// Evidence has gone stale — needs re-verification.
    Outdated {
        /// Step when staleness was detected.
        since: u64,
    },

    /// Conflicting evidence from multiple pathways.
    Contested {
        /// Number of member synapses with positive recent reinforcement.
        evidence_for: usize,
        /// Number of member synapses with negative recent reinforcement.
        evidence_against: usize,
    },

    /// Newly formed, not yet verified.
    Hypothesis {
        /// Step when the cluster was formed.
        formation_step: u64,
    },
}

impl Default for EpistemicState {
    fn default() -> Self {
        EpistemicState::Hypothesis { formation_step: 0 }
    }
}

/// Epistemic scarring — tracks a cluster's history of epistemic failures.
///
/// Clusters that have been repeatedly Outdated or Contested develop higher
/// skepticism, requiring stronger evidence for Hypothesis → Supported.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpistemicHistory {
    /// How many times this cluster entered Outdated.
    pub stale_count: u32,
    /// How many times this cluster entered Contested.
    pub contested_count: u32,
    /// How many times a Supported cluster was later found to be wrong.
    pub false_positive_count: u32,
    /// Derived skepticism score [0, 1].
    pub skepticism: f64,
}

impl EpistemicHistory {
    /// Recompute skepticism from the accumulated failure counts.
    pub fn update_skepticism(&mut self) {
        let raw = (self.stale_count as f64 * 0.1
            + self.contested_count as f64 * 0.3
            + self.false_positive_count as f64 * 0.5)
            .min(5.0);
        self.skepticism = raw / 5.0; // normalize to [0, 1]
    }

    /// The confidence threshold required for this cluster to transition
    /// from Hypothesis to Supported.
    pub fn required_confidence(&self) -> f64 {
        let base = 0.8;
        let scar_bonus = self.skepticism * 0.15;
        (base + scar_bonus).min(0.98)
    }

    /// Slowly decay scar counts — prevents permanent scarring from early noise.
    /// Call once per glacial epoch.
    pub fn decay(&mut self) {
        // Decay 1 count every 10 glacial epochs (~10K steps)
        // We track this by decrementing once per call but only every N calls.
        // Simplified: just decrement by 1 if nonzero (caller controls frequency).
        if self.stale_count > 0 {
            self.stale_count -= 1;
        }
        if self.contested_count > 0 {
            self.contested_count -= 1;
        }
        self.update_skepticism();
    }
}

/// Staleness threshold: if no reinforcement event within this many steps,
/// a Supported cluster transitions to Outdated.
const STALENESS_THRESHOLD: u64 = 5000;

/// Evaluate the epistemic state for a single cluster.
pub fn evaluate_cluster_state(
    cluster: &Cluster,
    _morphons: &HashMap<MorphonId, Morphon>,
    topology: &Topology,
    step_count: u64,
) -> EpistemicState {
    if cluster.members.is_empty() {
        return EpistemicState::Hypothesis { formation_step: step_count };
    }

    // Gather stats from member synapses
    let mut total_synapses = 0usize;
    let mut justified_count = 0usize;
    let mut consolidated_count = 0usize;
    let mut latest_reinforcement: u64 = 0;
    let mut positive_evidence = 0usize;
    let mut negative_evidence = 0usize;

    for &member_id in &cluster.members {
        // Examine incoming synapses of each member
        let incoming = topology.incoming_synapses(member_id);
        for (_, synapse) in &incoming {
            total_synapses += 1;
            if synapse.justification.is_some() {
                justified_count += 1;
                if let Some(ref j) = synapse.justification {
                    let last = j.last_reinforcement_step();
                    if last > latest_reinforcement {
                        latest_reinforcement = last;
                    }
                    // Check recent reinforcement direction
                    if let Some(recent) = j.reinforcement_history.back() {
                        if recent.delta_weight > 0.0 {
                            positive_evidence += 1;
                        } else if recent.delta_weight < 0.0 {
                            negative_evidence += 1;
                        }
                    }
                }
            }
            if synapse.consolidated {
                consolidated_count += 1;
            }
        }
    }

    if total_synapses == 0 {
        return EpistemicState::Hypothesis { formation_step: step_count };
    }

    // Check for contestation: significant evidence on both sides
    let total_evidence = positive_evidence + negative_evidence;
    if total_evidence > 2 {
        let minority_ratio = positive_evidence.min(negative_evidence) as f64 / total_evidence as f64;
        if minority_ratio > 0.25 {
            // At least 25% of evidence contradicts the majority → contested
            return EpistemicState::Contested {
                evidence_for: positive_evidence,
                evidence_against: negative_evidence,
            };
        }
    }

    // Compute confidence as fraction of justified + consolidated synapses
    let justified_frac = justified_count as f64 / total_synapses as f64;
    let consolidated_frac = consolidated_count as f64 / total_synapses as f64;
    let confidence = (justified_frac * 0.4 + consolidated_frac * 0.6).min(1.0);

    // Check required confidence (influenced by scarring)
    let required = cluster.epistemic_history.required_confidence();

    if confidence >= required {
        // Check for staleness
        if step_count > latest_reinforcement + STALENESS_THRESHOLD && latest_reinforcement > 0 {
            return EpistemicState::Outdated { since: step_count };
        }
        return EpistemicState::Supported {
            confidence,
            last_verified: step_count,
        };
    }

    EpistemicState::Hypothesis { formation_step: step_count }
}

/// Update epistemic states and scarring for all clusters.
/// Called on the glacial tick.
pub fn update_all_clusters(
    clusters: &mut HashMap<ClusterId, Cluster>,
    morphons: &HashMap<MorphonId, Morphon>,
    topology: &Topology,
    step_count: u64,
) {
    for cluster in clusters.values_mut() {
        let old_state = cluster.epistemic_state.clone();
        let new_state = evaluate_cluster_state(cluster, morphons, topology, step_count);

        // Track state transitions for scarring
        match (&old_state, &new_state) {
            // Entering Outdated
            (_, EpistemicState::Outdated { .. }) if !matches!(old_state, EpistemicState::Outdated { .. }) => {
                cluster.epistemic_history.stale_count += 1;
            }
            // Entering Contested
            (_, EpistemicState::Contested { .. }) if !matches!(old_state, EpistemicState::Contested { .. }) => {
                cluster.epistemic_history.contested_count += 1;
            }
            // Supported → Contested or Outdated = false positive
            (EpistemicState::Supported { .. }, EpistemicState::Contested { .. })
            | (EpistemicState::Supported { .. }, EpistemicState::Outdated { .. }) => {
                cluster.epistemic_history.false_positive_count += 1;
            }
            _ => {}
        }

        cluster.epistemic_history.update_skepticism();
        cluster.epistemic_state = new_state;
    }
}

/// Apply epistemic effects to morphon plasticity.
/// Called after state evaluation on the glacial tick.
pub fn apply_epistemic_effects(
    clusters: &HashMap<ClusterId, Cluster>,
    morphons: &mut HashMap<MorphonId, Morphon>,
    topology: &mut Topology,
    step_count: u64,
    verification_reward: f64,
) {
    for cluster in clusters.values() {
        match &cluster.epistemic_state {
            EpistemicState::Hypothesis { .. } => {
                // Fully plastic — boost plasticity for member morphons
                for &id in &cluster.members {
                    if let Some(m) = morphons.get_mut(&id) {
                        m.plasticity_rate = (m.plasticity_rate * 1.5).min(3.0);
                    }
                }
            }
            EpistemicState::Outdated { .. } => {
                // Open up — unconsolidate stale synapses to allow relearning.
                // Two-pass: collect candidates (immutable), then mutate.
                let mut to_unconsolidate = Vec::new();
                for &id in &cluster.members {
                    let incoming = topology.incoming_synapses(id);
                    for (from, synapse) in &incoming {
                        if synapse.consolidated {
                            let stale = synapse.justification.as_ref()
                                .map(|j| step_count > j.last_reinforcement_step() + STALENESS_THRESHOLD)
                                .unwrap_or(true);
                            if stale {
                                to_unconsolidate.push((*from, id));
                            }
                        }
                    }
                }
                for (from, to) in to_unconsolidate {
                    if let Some((ei, _)) = topology.synapse_between(from, to) {
                        if let Some(s) = topology.synapse_mut(ei) {
                            s.consolidated = false;
                        }
                    }
                }
            }
            EpistemicState::Contested { .. } => {
                // Increase arousal for members — drives re-evaluation
                for &id in &cluster.members {
                    if let Some(m) = morphons.get_mut(&id) {
                        m.plasticity_rate = (m.plasticity_rate * 1.2).min(3.0);
                    }
                }
            }
            EpistemicState::Supported { .. } => {
                // Reward members for verified knowledge — energy incentive for
                // forming clusters that pass epistemic scrutiny.
                if verification_reward > 0.0 {
                    for &id in &cluster.members {
                        if let Some(m) = morphons.get_mut(&id) {
                            m.energy = (m.energy + verification_reward).min(1.0);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_epistemic_state_is_hypothesis() {
        let state = EpistemicState::default();
        assert!(matches!(state, EpistemicState::Hypothesis { .. }));
    }

    #[test]
    fn skepticism_increases_with_failures() {
        let mut h = EpistemicHistory::default();
        assert_eq!(h.skepticism, 0.0);

        h.stale_count = 5;
        h.contested_count = 3;
        h.update_skepticism();
        // (5*0.1 + 3*0.3 + 0*0.5) = 1.4 → 1.4/5 = 0.28
        assert!((h.skepticism - 0.28).abs() < 0.01);
    }

    #[test]
    fn required_confidence_increases_with_skepticism() {
        let mut h = EpistemicHistory::default();
        let base = h.required_confidence();
        assert!((base - 0.8).abs() < 0.01);

        h.skepticism = 1.0;
        let scarred = h.required_confidence();
        assert!((scarred - 0.95).abs() < 0.01);
    }

    #[test]
    fn required_confidence_capped_at_098() {
        let mut h = EpistemicHistory::default();
        h.skepticism = 2.0; // extreme — beyond normal range
        assert!(h.required_confidence() <= 0.98);
    }

    #[test]
    fn decay_reduces_scar_counts() {
        let mut h = EpistemicHistory {
            stale_count: 3,
            contested_count: 2,
            false_positive_count: 0,
            skepticism: 0.0,
        };
        h.decay();
        assert_eq!(h.stale_count, 2);
        assert_eq!(h.contested_count, 1);
    }
}
