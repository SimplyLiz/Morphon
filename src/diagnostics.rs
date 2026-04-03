//! Learning diagnostics — observability into the learning pipeline.
//!
//! Tracks weight distributions, eligibility traces, tag-and-capture events,
//! firing rates by cell type, and spike delivery metrics. Essential for
//! tuning three-factor learning and debugging convergence issues.

use crate::morphon::Morphon;
use crate::topology::Topology;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-step diagnostics snapshot of the learning pipeline.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Diagnostics {
    // === Weight statistics ===
    /// Mean synapse weight across all connections.
    pub weight_mean: f64,
    /// Standard deviation of synapse weights.
    pub weight_std: f64,
    /// Maximum absolute synapse weight.
    pub weight_abs_max: f64,
    /// Total number of synapses.
    pub total_synapses: usize,

    // === Eligibility trace statistics ===
    /// Mean absolute eligibility trace value.
    pub eligibility_mean_abs: f64,
    /// Maximum absolute eligibility trace value.
    pub eligibility_max_abs: f64,
    /// Number of synapses with |eligibility| > 0.01.
    pub eligibility_nonzero_count: usize,

    // === Tag-and-capture ===
    /// Number of synapses with active tags (tag > 0.1).
    pub active_tags: usize,
    /// Capture events this step.
    pub captures_this_step: u64,
    /// Episode-gated captures pending endo consumption.
    /// Written by `report_episode_end()`, read+cleared by `sense_vitals()`.
    pub episode_captures_pending: u64,
    /// Cumulative capture events since system creation.
    pub total_captures: u64,

    // === Firing rates by cell type ===
    /// (firing_count, total_count) per CellType.
    pub firing_by_type: HashMap<CellType, (usize, usize)>,

    // === Spike delivery ===
    /// Spikes delivered to targets this step.
    pub spikes_delivered_this_step: usize,
    /// Spikes still in transit.
    pub spikes_pending: usize,

    // === Consolidation ===
    /// Number of synapses that have been consolidated (captured).
    pub consolidated_count: usize,
    /// Fraction of all synapses that are consolidated.
    pub consolidated_fraction: f64,
    /// Average energy across all morphons.
    pub avg_energy: f64,

    // === Apoptosis eligibility breakdown ===
    /// Morphons old enough for apoptosis (age > min_age).
    pub apoptosis_age_eligible: usize,
    /// Of those, how many are silent (activity < 0.005).
    pub apoptosis_silent: usize,
    /// Of those, how many have low energy (< threshold).
    pub apoptosis_energy_low: usize,
    /// Min/max/mean activity of Associative morphons (for debugging k-WTA).
    pub assoc_activity_min: f64,
    pub assoc_activity_max: f64,
    pub assoc_activity_mean: f64,

    // === Structural events ===
    /// Whether checkpoint rollback was triggered this step.
    pub rollback_triggered: bool,
    /// Cumulative rollback events since system creation.
    pub total_rollbacks: u64,

    // === V2: Frustration metrics ===
    /// Average frustration level across all morphons.
    pub avg_frustration: f64,
    /// Number of morphons currently in exploration_mode.
    pub exploration_mode_count: usize,
    /// Maximum frustration level among all morphons.
    pub max_frustration: f64,

    // === V2: Field metrics ===
    /// Peak value in the PredictionError field layer (0.0 if field disabled).
    pub field_pe_max: f64,
    /// Mean value in the PredictionError field layer (0.0 if field disabled).
    pub field_pe_mean: f64,

    // === V2: Target Morphology metrics ===
    /// Per-region population vs target: (region_index, current_count, target_density).
    pub region_health: Vec<(usize, usize, usize)>,

    // === V3: Epistemic metrics ===
    /// Number of clusters in each epistemic state.
    pub epistemic_supported: usize,
    pub epistemic_hypothesis: usize,
    pub epistemic_outdated: usize,
    pub epistemic_contested: usize,
    /// Fraction of all synapses with justification records.
    pub justified_fraction: f64,
    /// Average skepticism across all clusters.
    pub avg_skepticism: f64,

    // === V2: Competition metrics (Endo V2) ===
    /// Treves-Rolls population sparsity of associative activations this step.
    pub population_sparsity: f64,
    /// Mean lifetime sparsity (Treves-Rolls) across associative morphons.
    pub lifetime_sparsity: f64,
    /// Shannon entropy of per-morphon firing counts (normalized [0,1]).
    /// Computed at episode end; NaN between episodes.
    pub winner_diversity_entropy: f64,
}

impl Diagnostics {
    /// Compute a snapshot of weight, eligibility, and firing statistics
    /// from the current system state.
    pub fn snapshot(
        morphons: &HashMap<MorphonId, Morphon>,
        topology: &Topology,
    ) -> Self {
        let total_synapses = topology.synapse_count();
        let mut weight_sum = 0.0;
        let mut weight_sq_sum = 0.0;
        let mut weight_abs_max = 0.0_f64;
        let mut eligibility_abs_sum = 0.0;
        let mut eligibility_max_abs = 0.0_f64;
        let mut eligibility_nonzero = 0_usize;
        let mut active_tags = 0_usize;
        let mut consolidated_count = 0_usize;
        let mut justified_count = 0_usize;

        for ei in topology.graph.edge_indices() {
            if let Some(syn) = topology.graph.edge_weight(ei) {
                let w = if syn.weight.is_finite() { syn.weight.clamp(-100.0, 100.0) } else { 0.0 };
                weight_sum += w;
                weight_sq_sum += w * w;
                weight_abs_max = weight_abs_max.max(w.abs());

                let e_abs = if syn.eligibility.is_finite() { syn.eligibility.abs() } else { 0.0 };
                eligibility_abs_sum += e_abs;
                eligibility_max_abs = eligibility_max_abs.max(e_abs);
                if e_abs > 0.01 {
                    eligibility_nonzero += 1;
                }

                if syn.tag > 0.1 {
                    active_tags += 1;
                }
                if syn.consolidated {
                    consolidated_count += 1;
                }
                if syn.justification.is_some() {
                    justified_count += 1;
                }
            }
        }

        let n = total_synapses.max(1) as f64;
        let weight_mean = weight_sum / n;
        let weight_variance = (weight_sq_sum / n) - (weight_mean * weight_mean);
        let weight_std = weight_variance.max(0.0).sqrt();

        // Firing rates by cell type + average energy + apoptosis eligibility
        let mut firing_by_type: HashMap<CellType, (usize, usize)> = HashMap::new();
        let mut energy_sum = 0.0;
        let mut apoptosis_age_eligible = 0_usize;
        let mut apoptosis_silent = 0_usize;
        let mut apoptosis_energy_low = 0_usize;
        let mut assoc_activities: Vec<f64> = Vec::new();
        let mut frustration_sum = 0.0_f64;
        let mut frustration_max = 0.0_f64;
        let mut exploration_mode_count = 0_usize;

        for m in morphons.values() {
            let entry = firing_by_type.entry(m.cell_type).or_insert((0, 0));
            entry.1 += 1;
            if m.fired {
                entry.0 += 1;
            }
            energy_sum += m.energy;

            // Apoptosis eligibility tracking
            if m.age > 1000 {
                apoptosis_age_eligible += 1;
                if m.activity_history.mean() < 0.005 {
                    apoptosis_silent += 1;
                }
                if m.energy < 0.1 {
                    apoptosis_energy_low += 1;
                }
            }

            // Associative activity stats
            if m.cell_type == CellType::Associative || m.cell_type == CellType::Stem {
                assoc_activities.push(m.activity_history.mean());
            }

            // V2: Frustration stats
            frustration_sum += m.frustration.frustration_level;
            frustration_max = frustration_max.max(m.frustration.frustration_level);
            if m.frustration.exploration_mode {
                exploration_mode_count += 1;
            }
        }

        let assoc_activity_min = assoc_activities.iter().cloned().fold(f64::INFINITY, f64::min);
        let assoc_activity_max = assoc_activities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let assoc_activity_mean = if assoc_activities.is_empty() { 0.0 }
            else { assoc_activities.iter().sum::<f64>() / assoc_activities.len() as f64 };
        let avg_energy = energy_sum / morphons.len().max(1) as f64;

        // Competition metrics: population sparsity (from current step firing)
        let assoc_fired: Vec<f64> = morphons.values()
            .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
            .map(|m| if m.fired { 1.0 } else { 0.0 })
            .collect();
        let population_sparsity = treves_rolls_sparsity(&assoc_fired);

        // Competition metrics: lifetime sparsity (from activity_history per morphon)
        let lifetime_sparsities: Vec<f64> = morphons.values()
            .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
            .filter_map(|m| {
                let history: Vec<f64> = m.activity_history.iter().copied().collect();
                let ls = treves_rolls_sparsity(&history);
                if ls.is_finite() { Some(ls) } else { None }
            })
            .collect();
        let lifetime_sparsity = if lifetime_sparsities.is_empty() { 0.0 }
            else { lifetime_sparsities.iter().sum::<f64>() / lifetime_sparsities.len() as f64 };

        Self {
            weight_mean,
            weight_std,
            weight_abs_max,
            total_synapses,
            eligibility_mean_abs: eligibility_abs_sum / n,
            eligibility_max_abs,
            eligibility_nonzero_count: eligibility_nonzero,
            active_tags,
            consolidated_count,
            consolidated_fraction: consolidated_count as f64 / total_synapses.max(1) as f64,
            justified_fraction: justified_count as f64 / total_synapses.max(1) as f64,
            avg_energy,
            firing_by_type,
            apoptosis_age_eligible,
            apoptosis_silent,
            apoptosis_energy_low,
            assoc_activity_min: if assoc_activity_min.is_finite() { assoc_activity_min } else { 0.0 },
            assoc_activity_max: if assoc_activity_max.is_finite() { assoc_activity_max } else { 0.0 },
            assoc_activity_mean,
            avg_frustration: frustration_sum / morphons.len().max(1) as f64,
            exploration_mode_count,
            max_frustration: frustration_max,
            population_sparsity,
            lifetime_sparsity,
            winner_diversity_entropy: 0.0, // computed at episode end, not per-step
            ..Default::default()
        }
    }

    /// Format a concise one-line summary for logging.
    pub fn summary(&self) -> String {
        format!(
            "w={:.4}\u{00b1}{:.4} e={:.4}({}) tags={} con={}/{} E={:.2} spk={} frust={:.2}({})",
            self.weight_mean,
            self.weight_std,
            self.eligibility_mean_abs,
            self.eligibility_nonzero_count,
            self.active_tags,
            self.consolidated_count,
            self.total_synapses,
            self.avg_energy,
            self.spikes_delivered_this_step,
            self.avg_frustration,
            self.exploration_mode_count,
        )
    }

    /// Format per-type firing rates for logging.
    pub fn firing_summary(&self) -> String {
        let mut parts = Vec::new();
        // Sort by cell type for consistent output
        let mut entries: Vec<_> = self.firing_by_type.iter().collect();
        entries.sort_by_key(|(ct, _)| format!("{:?}", ct));
        for (ct, (firing, total)) in entries {
            if *total > 0 {
                parts.push(format!(
                    "{:?}={:.1}%",
                    ct,
                    *firing as f64 / *total as f64 * 100.0
                ));
            }
        }
        parts.join(" ")
    }
}

/// Treves-Rolls sparsity measure.
/// Returns 0.0 (dense / all equal) to 1.0 (maximally sparse / one active).
/// S = (1 - (mean(r))^2 / mean(r^2)) / (1 - 1/N)
pub fn treves_rolls_sparsity(values: &[f64]) -> f64 {
    let n = values.len();
    if n <= 1 { return 0.0; }
    let mean = values.iter().sum::<f64>() / n as f64;
    let mean_sq = values.iter().map(|x| x * x).sum::<f64>() / n as f64;
    if mean_sq < 1e-12 { return 0.0; }
    (1.0 - (mean * mean) / mean_sq) / (1.0 - 1.0 / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_treves_rolls_maximally_sparse() {
        // One active, rest silent → maximally sparse
        let values = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let s = treves_rolls_sparsity(&values);
        assert!(s > 0.9, "one active should be near 1.0: {}", s);
    }

    #[test]
    fn test_treves_rolls_uniform() {
        // All equal → minimum sparsity (0.0)
        let values = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let s = treves_rolls_sparsity(&values);
        assert!(s < 0.01, "uniform should be near 0.0: {}", s);
    }

    #[test]
    fn test_treves_rolls_empty() {
        assert_eq!(treves_rolls_sparsity(&[]), 0.0);
        assert_eq!(treves_rolls_sparsity(&[1.0]), 0.0);
    }

    #[test]
    fn test_treves_rolls_all_zero() {
        let values = vec![0.0, 0.0, 0.0];
        let s = treves_rolls_sparsity(&values);
        assert_eq!(s, 0.0, "all zeros should be 0.0");
    }
}
