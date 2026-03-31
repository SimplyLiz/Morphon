//! Resonance — local, topology-based signal propagation.
//!
//! Instead of global quadratic attention, MI uses local signal propagation:
//! 1. A firing Morphon sends signals to direct neighbors (weighted + delayed)
//! 2. Correlated firing creates synchronization waves (resonance cascades)
//! 3. Rare long-range connections couple distant regions
//!
//! Complexity: O(k·N) where k = average connectivity, N = morphon count.

use crate::morphon::Morphon;
use crate::topology::Topology;
use crate::types::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A spike event traveling through the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeEvent {
    /// Source Morphon that fired.
    pub source: MorphonId,
    /// Target Morphon to receive the signal.
    pub target: MorphonId,
    /// Signal strength (weight * source activation).
    pub strength: f64,
    /// Remaining delay before delivery (in timesteps).
    pub delay: f64,
}

/// The resonance engine manages signal propagation through the network.
pub struct ResonanceEngine {
    /// Queue of pending spike events (delayed delivery).
    pending_spikes: VecDeque<SpikeEvent>,
}

impl ResonanceEngine {
    pub fn new() -> Self {
        Self {
            pending_spikes: VecDeque::new(),
        }
    }

    /// Generate spike events from all currently firing Morphons.
    pub fn propagate(
        &mut self,
        morphons: &std::collections::HashMap<MorphonId, Morphon>,
        topology: &Topology,
    ) {
        // Collect firing morphons
        let firing: Vec<MorphonId> = morphons
            .values()
            .filter(|m| m.fired)
            .map(|m| m.id)
            .collect();

        // Generate spike events for each firing morphon's outgoing connections
        let spike_gen = |&source_id: &MorphonId| -> Vec<SpikeEvent> {
            topology.outgoing(source_id)
                .into_iter()
                .map(|(target_id, synapse)| SpikeEvent {
                    source: source_id,
                    target: target_id,
                    strength: synapse.weight,
                    delay: synapse.delay,
                })
                .collect()
        };

        #[cfg(feature = "parallel")]
        let new_spikes: Vec<SpikeEvent> = firing.par_iter().flat_map(spike_gen).collect();
        #[cfg(not(feature = "parallel"))]
        let new_spikes: Vec<SpikeEvent> = firing.iter().flat_map(spike_gen).collect();

        self.pending_spikes.extend(new_spikes);
    }

    /// Deliver spikes that have reached their target (delay expired).
    /// Returns the list of delivered events for learning updates.
    pub fn deliver(
        &mut self,
        morphons: &mut std::collections::HashMap<MorphonId, Morphon>,
        dt: f64,
    ) -> Vec<SpikeEvent> {
        let mut delivered = Vec::new();
        let mut still_pending = VecDeque::new();

        while let Some(mut spike) = self.pending_spikes.pop_front() {
            spike.delay -= dt;
            if spike.delay <= 0.0 {
                // Deliver: add to target's input accumulator
                if let Some(target) = morphons.get_mut(&spike.target) {
                    target.input_accumulator += spike.strength;
                }
                delivered.push(spike);
            } else {
                still_pending.push_back(spike);
            }
        }

        self.pending_spikes = still_pending;
        delivered
    }

    /// Number of spikes currently in transit.
    pub fn pending_count(&self) -> usize {
        self.pending_spikes.len()
    }

    /// Clear all pending spikes.
    pub fn clear(&mut self) {
        self.pending_spikes.clear();
    }
}
