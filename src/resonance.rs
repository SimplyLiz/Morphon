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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphon::{Morphon, Synapse};
    use crate::topology::Topology;
    use crate::types::HyperbolicPoint;
    use std::collections::HashMap;

    fn make_morphon(id: MorphonId, fired: bool) -> Morphon {
        let mut m = Morphon::new(id, HyperbolicPoint::origin(3));
        m.fired = fired;
        m
    }

    #[test]
    fn propagate_generates_spikes_from_firing_morphons() {
        let mut morphons = HashMap::new();
        morphons.insert(1, make_morphon(1, true));
        morphons.insert(2, make_morphon(2, false));
        morphons.insert(3, make_morphon(3, false));

        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);
        topo.add_morphon(3);
        topo.add_synapse(1, 2, Synapse::new(0.5).with_delay(2.0));
        topo.add_synapse(1, 3, Synapse::new(0.3).with_delay(1.0));

        let mut engine = ResonanceEngine::new();
        engine.propagate(&morphons, &topo);

        assert_eq!(engine.pending_count(), 2);
    }

    #[test]
    fn non_firing_morphons_generate_no_spikes() {
        let mut morphons = HashMap::new();
        morphons.insert(1, make_morphon(1, false));
        morphons.insert(2, make_morphon(2, false));

        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);
        topo.add_synapse(1, 2, Synapse::new(0.5));

        let mut engine = ResonanceEngine::new();
        engine.propagate(&morphons, &topo);

        assert_eq!(engine.pending_count(), 0);
    }

    #[test]
    fn deliver_respects_delay() {
        let mut morphons = HashMap::new();
        morphons.insert(2, make_morphon(2, false));

        let mut engine = ResonanceEngine::new();
        engine.pending_spikes.push_back(SpikeEvent {
            source: 1,
            target: 2,
            strength: 0.5,
            delay: 3.0,
        });

        // Step 1: delay 3 -> 2, not delivered
        let delivered = engine.deliver(&mut morphons, 1.0);
        assert_eq!(delivered.len(), 0);
        assert_eq!(engine.pending_count(), 1);

        // Step 2: delay 2 -> 1
        let delivered = engine.deliver(&mut morphons, 1.0);
        assert_eq!(delivered.len(), 0);

        // Step 3: delay 1 -> 0, delivered
        let delivered = engine.deliver(&mut morphons, 1.0);
        assert_eq!(delivered.len(), 1);
        assert_eq!(engine.pending_count(), 0);
    }

    #[test]
    fn delivered_spike_adds_to_input_accumulator() {
        let mut morphons = HashMap::new();
        morphons.insert(2, make_morphon(2, false));

        let mut engine = ResonanceEngine::new();
        engine.pending_spikes.push_back(SpikeEvent {
            source: 1,
            target: 2,
            strength: 0.7,
            delay: 0.5, // will be delivered in one step with dt=1.0
        });

        let _delivered = engine.deliver(&mut morphons, 1.0);
        assert!(
            (morphons[&2].input_accumulator - 0.7).abs() < 1e-10,
            "spike strength should be added to target input_accumulator"
        );
    }

    #[test]
    fn multiple_spikes_accumulate() {
        let mut morphons = HashMap::new();
        morphons.insert(2, make_morphon(2, false));

        let mut engine = ResonanceEngine::new();
        engine.pending_spikes.push_back(SpikeEvent {
            source: 1,
            target: 2,
            strength: 0.3,
            delay: 0.0,
        });
        engine.pending_spikes.push_back(SpikeEvent {
            source: 3,
            target: 2,
            strength: 0.4,
            delay: 0.0,
        });

        let delivered = engine.deliver(&mut morphons, 1.0);
        assert_eq!(delivered.len(), 2);
        assert!(
            (morphons[&2].input_accumulator - 0.7).abs() < 1e-10,
            "multiple spikes should accumulate"
        );
    }

    #[test]
    fn clear_removes_all_pending() {
        let mut engine = ResonanceEngine::new();
        engine.pending_spikes.push_back(SpikeEvent {
            source: 1,
            target: 2,
            strength: 0.5,
            delay: 5.0,
        });
        assert_eq!(engine.pending_count(), 1);
        engine.clear();
        assert_eq!(engine.pending_count(), 0);
    }

    #[test]
    fn spike_to_nonexistent_target_does_not_panic() {
        let mut morphons: HashMap<MorphonId, Morphon> = HashMap::new();
        // target 99 doesn't exist

        let mut engine = ResonanceEngine::new();
        engine.pending_spikes.push_back(SpikeEvent {
            source: 1,
            target: 99,
            strength: 0.5,
            delay: 0.0,
        });

        let delivered = engine.deliver(&mut morphons, 1.0);
        assert_eq!(delivered.len(), 1); // spike is consumed even if target missing
    }
}
