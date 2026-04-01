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

    // === Structural events ===
    /// Whether checkpoint rollback was triggered this step.
    pub rollback_triggered: bool,
    /// Cumulative rollback events since system creation.
    pub total_rollbacks: u64,
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

        for ei in topology.graph.edge_indices() {
            if let Some(syn) = topology.graph.edge_weight(ei) {
                let w = if syn.weight.is_finite() { syn.weight } else { 0.0 };
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
            }
        }

        let n = total_synapses.max(1) as f64;
        let weight_mean = weight_sum / n;
        let weight_variance = (weight_sq_sum / n) - (weight_mean * weight_mean);
        let weight_std = weight_variance.max(0.0).sqrt();

        // Firing rates by cell type + average energy
        let mut firing_by_type: HashMap<CellType, (usize, usize)> = HashMap::new();
        let mut energy_sum = 0.0;
        for m in morphons.values() {
            let entry = firing_by_type.entry(m.cell_type).or_insert((0, 0));
            entry.1 += 1;
            if m.fired {
                entry.0 += 1;
            }
            energy_sum += m.energy;
        }
        let avg_energy = energy_sum / morphons.len().max(1) as f64;

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
            avg_energy,
            firing_by_type,
            ..Default::default()
        }
    }

    /// Format a concise one-line summary for logging.
    pub fn summary(&self) -> String {
        format!(
            "w={:.4}\u{00b1}{:.4} e={:.4}({}) tags={} con={}/{} E={:.2} spk={}",
            self.weight_mean,
            self.weight_std,
            self.eligibility_mean_abs,
            self.eligibility_nonzero_count,
            self.active_tags,
            self.consolidated_count,
            self.total_synapses,
            self.avg_energy,
            self.spikes_delivered_this_step,
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
