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
use std::collections::HashMap;
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
    /// Original delay at creation — used by the visualizer to compute progress.
    pub initial_delay: f64,
    /// Cached petgraph edge index of the synapse that produced this spike.
    /// Captured at propagate-time so the delivery loop can bump `pre_trace`
    /// via direct edge lookup instead of paying a `synapse_between` HashMap
    /// walk per spike. Stored as raw u32 to keep SpikeEvent free of petgraph
    /// types in its serde representation; reconstruct with
    /// `petgraph::graph::EdgeIndex::new(raw as usize)`.
    ///
    /// `None` when deserialized from old snapshots — the delivery loop skips
    /// the pre_trace bump in that case rather than silently updating edge 0.
    #[serde(default)]
    pub edge_idx: Option<u32>,
}

/// The resonance engine manages signal propagation through the network.
pub struct ResonanceEngine {
    /// Queue of pending spike events (delayed delivery).
    pending_spikes: VecDeque<SpikeEvent>,
    /// Spikes delivered on the most recent deliver() call (for visualization).
    last_delivered: Vec<SpikeEvent>,
}

impl ResonanceEngine {
    pub fn new() -> Self {
        Self {
            pending_spikes: VecDeque::new(),
            last_delivered: Vec::new(),
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

        // Generate spike events for each firing morphon's outgoing connections.
        // Spike strength is scaled by source morphon energy — metabolically
        // stressed morphons produce weaker signals (graceful degradation).
        //
        // Per-source allocation pattern: we still allocate one Vec per firing
        // morphon (rayon's flat_map needs an IntoIterator), but we use the
        // closure-based `for_each_outgoing_with_edge` so the topology helper
        // itself does not allocate an intermediate Vec. The per-source Vec is
        // pre-sized to a typical fan-out (32) to avoid 0→4→8→16→32 reallocs.
        let spike_gen = |&source_id: &MorphonId| -> Vec<SpikeEvent> {
            let energy_factor = morphons.get(&source_id)
                .map(|m| (m.energy / 0.6).min(1.0))
                .unwrap_or(1.0);
            let mut spikes: Vec<SpikeEvent> = Vec::with_capacity(32);
            topology.for_each_outgoing_with_edge(source_id, |target_id, edge_idx, synapse| {
                let eff_delay = synapse.effective_delay();
                spikes.push(SpikeEvent {
                    source: source_id,
                    target: target_id,
                    strength: synapse.weight * energy_factor,
                    delay: eff_delay,
                    initial_delay: eff_delay,
                    edge_idx: Some(edge_idx.index() as u32),
                });
            });
            spikes
        };

        #[cfg(feature = "parallel")]
        let new_spikes: Vec<SpikeEvent> = firing.par_iter().flat_map(spike_gen).collect();
        #[cfg(not(feature = "parallel"))]
        let new_spikes: Vec<SpikeEvent> = firing.iter().flat_map(spike_gen).collect();

        self.pending_spikes.extend(new_spikes);
    }

    /// Deliver spikes that have reached their target (delay expired).
    /// Writes spike current directly into the hot-path `input_current` array,
    /// bypassing the per-morphon HashMap lookup that `input_accumulator` required.
    /// Returns the list of delivered events for learning updates (pre_trace, STDP).
    pub fn deliver(
        &mut self,
        input_current: &mut Vec<f32>,
        id_to_idx: &HashMap<MorphonId, usize>,
        dt: f64,
    ) -> Vec<SpikeEvent> {
        let mut delivered = Vec::new();
        let mut still_pending = VecDeque::new();

        while let Some(mut spike) = self.pending_spikes.pop_front() {
            spike.delay -= dt;
            if spike.delay <= 0.0 {
                // Deliver: write directly to hot input_current (Vec index, no HashMap walk).
                // Spikes to targets that have since died (not in id_to_idx) are silently dropped.
                if let Some(&j) = id_to_idx.get(&spike.target) {
                    input_current[j] += spike.strength as f32;
                }
                delivered.push(spike);
            } else {
                still_pending.push_back(spike);
            }
        }

        self.pending_spikes = still_pending;
        self.last_delivered = delivered.clone();
        delivered
    }

    /// Spikes delivered on the most recent deliver() call.
    pub fn last_delivered(&self) -> &[SpikeEvent] {
        &self.last_delivered
    }

    /// Number of spikes currently in transit.
    pub fn pending_count(&self) -> usize {
        self.pending_spikes.len()
    }

    /// Clear all pending spikes.
    pub fn clear(&mut self) {
        self.pending_spikes.clear();
    }

    /// Read-only access to pending spikes (for visualization).
    pub fn pending_spikes(&self) -> &VecDeque<SpikeEvent> {
        &self.pending_spikes
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

    fn make_hot(targets: &[(MorphonId, usize)], size: usize) -> (Vec<f32>, HashMap<MorphonId, usize>) {
        let input_current = vec![0.0f32; size];
        let id_to_idx: HashMap<MorphonId, usize> = targets.iter().copied().collect();
        (input_current, id_to_idx)
    }

    #[test]
    fn deliver_respects_delay() {
        let (mut input_current, id_to_idx) = make_hot(&[(2, 0)], 1);

        let mut engine = ResonanceEngine::new();
        engine.pending_spikes.push_back(SpikeEvent {
            source: 1,
            target: 2,
            strength: 0.5,
            delay: 3.0,
            initial_delay: 3.0,
            edge_idx: None,
        });

        // Step 1: delay 3 -> 2, not delivered
        let delivered = engine.deliver(&mut input_current, &id_to_idx, 1.0);
        assert_eq!(delivered.len(), 0);
        assert_eq!(engine.pending_count(), 1);

        // Step 2: delay 2 -> 1
        let delivered = engine.deliver(&mut input_current, &id_to_idx, 1.0);
        assert_eq!(delivered.len(), 0);

        // Step 3: delay 1 -> 0, delivered
        let delivered = engine.deliver(&mut input_current, &id_to_idx, 1.0);
        assert_eq!(delivered.len(), 1);
        assert_eq!(engine.pending_count(), 0);
    }

    #[test]
    fn delivered_spike_adds_to_input_current() {
        let (mut input_current, id_to_idx) = make_hot(&[(2, 0)], 1);

        let mut engine = ResonanceEngine::new();
        engine.pending_spikes.push_back(SpikeEvent {
            source: 1,
            target: 2,
            strength: 0.7,
            delay: 0.5, // will be delivered in one step with dt=1.0
            initial_delay: 0.5,
            edge_idx: None,
        });

        let _delivered = engine.deliver(&mut input_current, &id_to_idx, 1.0);
        assert!(
            (input_current[0] - 0.7f32).abs() < 1e-6,
            "spike strength should be added to hot input_current"
        );
    }

    #[test]
    fn multiple_spikes_accumulate() {
        let (mut input_current, id_to_idx) = make_hot(&[(2, 0)], 1);

        let mut engine = ResonanceEngine::new();
        engine.pending_spikes.push_back(SpikeEvent {
            source: 1,
            target: 2,
            strength: 0.3,
            delay: 0.0,
            initial_delay: 0.0,
            edge_idx: None,
        });
        engine.pending_spikes.push_back(SpikeEvent {
            source: 3,
            target: 2,
            strength: 0.4,
            delay: 0.0,
            initial_delay: 0.0,
            edge_idx: None,
        });

        let delivered = engine.deliver(&mut input_current, &id_to_idx, 1.0);
        assert_eq!(delivered.len(), 2);
        assert!(
            (input_current[0] - 0.7f32).abs() < 1e-6,
            "multiple spikes should accumulate in hot input_current"
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
            initial_delay: 5.0,
            edge_idx: None,
        });
        assert_eq!(engine.pending_count(), 1);
        engine.clear();
        assert_eq!(engine.pending_count(), 0);
    }

    #[test]
    fn spike_to_nonexistent_target_does_not_panic() {
        // target 99 not in id_to_idx — spike should be consumed without panic
        let (mut input_current, id_to_idx) = make_hot(&[], 0);

        let mut engine = ResonanceEngine::new();
        engine.pending_spikes.push_back(SpikeEvent {
            source: 1,
            target: 99,
            strength: 0.5,
            delay: 0.0,
            initial_delay: 0.0,
            edge_idx: None,
        });

        let delivered = engine.deliver(&mut input_current, &id_to_idx, 1.0);
        assert_eq!(delivered.len(), 1); // spike is consumed even if target missing
    }

    #[test]
    fn propagate_caches_correct_edge_idx() {
        // Verifies that edge_idx in the generated SpikeEvent matches the actual
        // EdgeIndex returned by the topology, so the delivery loop's pre_trace
        // bump targets the right synapse.
        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);
        let ei = topo.add_synapse(1, 2, Synapse::new(0.5)).unwrap();

        let mut morphons = HashMap::new();
        morphons.insert(1, make_morphon(1, true));
        morphons.insert(2, make_morphon(2, false));

        let mut engine = ResonanceEngine::new();
        engine.propagate(&morphons, &topo);

        assert_eq!(engine.pending_count(), 1);
        let spike = engine.pending_spikes.front().unwrap();
        assert_eq!(spike.source, 1);
        assert_eq!(spike.target, 2);
        assert_eq!(spike.edge_idx, Some(ei.index() as u32),
            "cached edge_idx must match the actual EdgeIndex");
    }

    #[test]
    fn energy_dependent_spike_strength() {
        let synapse_weight = 0.8;

        // Low-energy morphon: energy=0.3, factor = 0.3/0.6 = 0.5
        let mut low_energy = make_morphon(1, true);
        low_energy.energy = 0.3;

        // Healthy morphon: energy=0.9, factor = min(0.9/0.6, 1.0) = 1.0
        let mut healthy = make_morphon(2, true);
        healthy.energy = 0.9;

        let mut morphons = HashMap::new();
        morphons.insert(1, low_energy);
        morphons.insert(2, healthy);
        morphons.insert(3, make_morphon(3, false)); // target

        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);
        topo.add_morphon(3);
        topo.add_synapse(1, 3, Synapse::new(synapse_weight));
        topo.add_synapse(2, 3, Synapse::new(synapse_weight));

        let mut engine = ResonanceEngine::new();
        engine.propagate(&morphons, &topo);

        // Find spikes from each source
        let spikes: Vec<&SpikeEvent> = engine.pending_spikes.iter().collect();
        let low_spike = spikes.iter().find(|s| s.source == 1).unwrap();
        let healthy_spike = spikes.iter().find(|s| s.source == 2).unwrap();

        let expected_low = synapse_weight * (0.3 / 0.6);
        assert!((low_spike.strength - expected_low).abs() < 0.001,
            "low-energy spike should be ~{expected_low:.3}, got {:.3}", low_spike.strength);
        assert!((healthy_spike.strength - synapse_weight).abs() < 0.001,
            "healthy spike should be full weight {synapse_weight}, got {:.3}", healthy_spike.strength);
    }
}
