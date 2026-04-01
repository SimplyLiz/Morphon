//! System snapshot serialization — save and load MI system state.
//!
//! Captures the full system state (Morphons, connections, clusters, modulation,
//! memory) into a serializable format for checkpoint export/import.

use crate::morphogenesis::Cluster;
use crate::morphon::{Morphon, Synapse};
use crate::neuromodulation::Neuromodulation;
use crate::system::{System, SystemConfig};
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A serializable snapshot of the full system state.
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemSnapshot {
    /// Configuration.
    pub config: SystemConfig,
    /// All Morphons.
    pub morphons: HashMap<MorphonId, Morphon>,
    /// All connections as (from, to, synapse) triples.
    pub connections: Vec<(MorphonId, MorphonId, Synapse)>,
    /// All clusters.
    pub clusters: HashMap<ClusterId, Cluster>,
    /// Neuromodulation state.
    pub modulation: Neuromodulation,
    /// Next Morphon ID.
    pub next_morphon_id: MorphonId,
    /// Next Cluster ID.
    pub next_cluster_id: ClusterId,
    /// Current step count.
    pub step_count: u64,
    /// Stable input port mapping.
    pub input_ports: Vec<MorphonId>,
    /// Stable output port mapping.
    pub output_ports: Vec<MorphonId>,
}

impl System {
    /// Create a serializable snapshot of the current system state.
    pub fn snapshot(&self) -> SystemSnapshot {
        let connections: Vec<(MorphonId, MorphonId, Synapse)> = self
            .topology
            .all_edges()
            .into_iter()
            .filter_map(|(from, to, ei)| {
                self.topology
                    .graph
                    .edge_weight(ei)
                    .map(|syn| (from, to, syn.clone()))
            })
            .collect();

        SystemSnapshot {
            config: self.config.clone(),
            morphons: self.morphons.clone(),
            connections,
            clusters: self.clusters.clone(),
            modulation: self.modulation.clone(),
            next_morphon_id: self.next_morphon_id,
            next_cluster_id: self.next_cluster_id,
            step_count: self.step_count,
            input_ports: self.input_ports.clone(),
            output_ports: self.output_ports.clone(),
        }
    }

    /// Restore a system from a snapshot.
    pub fn from_snapshot(snapshot: SystemSnapshot) -> Self {
        use crate::memory::TripleMemory;
        use crate::resonance::ResonanceEngine;
        use crate::topology::Topology;

        let mut topology = Topology::new();

        // Add all morphons to topology
        for &id in snapshot.morphons.keys() {
            topology.add_morphon(id);
        }

        // Restore all connections
        for (from, to, synapse) in snapshot.connections {
            topology.add_synapse(from, to, synapse);
        }

        System {
            morphons: snapshot.morphons,
            topology,
            resonance: ResonanceEngine::new(),
            modulation: snapshot.modulation,
            clusters: snapshot.clusters,
            memory: TripleMemory::new(
                snapshot.config.working_memory_capacity,
                snapshot.config.episodic_memory_capacity,
            ),
            config: snapshot.config,
            input_ports: snapshot.input_ports,
            output_ports: snapshot.output_ports,
            next_morphon_id: snapshot.next_morphon_id,
            next_cluster_id: snapshot.next_cluster_id,
            step_count: snapshot.step_count,
            critic_ports: Vec::new(),
            prev_critic_value: 0.0,
            last_td_error: 0.0,
            diag: crate::diagnostics::Diagnostics::default(),
            total_born: 0,
            total_died: 0,
            feedback_weights: std::collections::HashMap::new(),
            readout_weights: Vec::new(),
            use_analog_readout: false,
            consolidation_gate: 30.0,
            recent_performance: 0.0,
            peak_performance: 0.0,
        }
    }

    /// Save the system state to a JSON string.
    pub fn save_json(&self) -> Result<String, serde_json::Error> {
        let snapshot = self.snapshot();
        serde_json::to_string(&snapshot)
    }

    /// Save the system state to a pretty-printed JSON string.
    pub fn save_json_pretty(&self) -> Result<String, serde_json::Error> {
        let snapshot = self.snapshot();
        serde_json::to_string_pretty(&snapshot)
    }

    /// Load a system from a JSON string.
    pub fn load_json(json: &str) -> Result<Self, serde_json::Error> {
        let snapshot: SystemSnapshot = serde_json::from_str(json)?;
        Ok(Self::from_snapshot(snapshot))
    }
}
