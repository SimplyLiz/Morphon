//! Topology management — the dynamic graph of Morphon connections.
//!
//! Uses petgraph with dynamic node/edge management to support
//! synaptogenesis, pruning, division, fusion, and apoptosis.

use crate::morphon::Synapse;
use crate::types::*;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// The network topology graph.
pub struct Topology {
    /// The directed graph: nodes are MorphonIds, edges are Synapses.
    pub graph: DiGraph<MorphonId, Synapse>,
    /// Map from MorphonId to NodeIndex for fast lookup.
    id_to_node: HashMap<MorphonId, NodeIndex>,
}

impl Topology {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            id_to_node: HashMap::new(),
        }
    }

    /// Add a Morphon to the topology.
    pub fn add_morphon(&mut self, id: MorphonId) -> NodeIndex {
        let idx = self.graph.add_node(id);
        self.id_to_node.insert(id, idx);
        idx
    }

    /// Remove a Morphon and all its connections.
    pub fn remove_morphon(&mut self, id: MorphonId) -> bool {
        if let Some(&idx) = self.id_to_node.get(&id) {
            self.graph.remove_node(idx);
            self.id_to_node.remove(&id);
            // petgraph may invalidate indices after removal, rebuild map
            self.rebuild_index_map();
            true
        } else {
            false
        }
    }

    /// Add a synapse between two Morphons.
    pub fn add_synapse(&mut self, from: MorphonId, to: MorphonId, synapse: Synapse) -> Option<EdgeIndex> {
        let from_idx = self.id_to_node.get(&from)?;
        let to_idx = self.id_to_node.get(&to)?;
        Some(self.graph.add_edge(*from_idx, *to_idx, synapse))
    }

    /// Remove a specific synapse.
    pub fn remove_synapse(&mut self, edge: EdgeIndex) {
        self.graph.remove_edge(edge);
    }

    /// Get all incoming connections for a Morphon.
    pub fn incoming(&self, id: MorphonId) -> Vec<(MorphonId, &Synapse)> {
        let Some(&idx) = self.id_to_node.get(&id) else {
            return Vec::new();
        };
        self.graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .map(|edge| {
                let source_idx = edge.source();
                let source_id = self.graph[source_idx];
                (source_id, edge.weight())
            })
            .collect()
    }

    /// Get all outgoing connections for a Morphon.
    pub fn outgoing(&self, id: MorphonId) -> Vec<(MorphonId, &Synapse)> {
        let Some(&idx) = self.id_to_node.get(&id) else {
            return Vec::new();
        };
        self.graph
            .edges_directed(idx, petgraph::Direction::Outgoing)
            .map(|edge| {
                let target_idx = edge.target();
                let target_id = self.graph[target_idx];
                (target_id, edge.weight())
            })
            .collect()
    }

    /// Get mutable access to all incoming synapses for a Morphon.
    pub fn incoming_synapses_mut(&mut self, id: MorphonId) -> Vec<(MorphonId, EdgeIndex)> {
        let Some(&idx) = self.id_to_node.get(&id) else {
            return Vec::new();
        };
        self.graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .map(|edge| {
                let source_id = self.graph[edge.source()];
                (source_id, edge.id())
            })
            .collect()
    }

    /// Get a mutable reference to a synapse by edge index.
    pub fn synapse_mut(&mut self, edge: EdgeIndex) -> Option<&mut Synapse> {
        self.graph.edge_weight_mut(edge)
    }

    /// Get the synapse between two specific Morphons.
    pub fn synapse_between(&self, from: MorphonId, to: MorphonId) -> Option<(EdgeIndex, &Synapse)> {
        let from_idx = self.id_to_node.get(&from)?;
        let to_idx = self.id_to_node.get(&to)?;
        self.graph.find_edge(*from_idx, *to_idx)
            .map(|ei| (ei, &self.graph[ei]))
    }

    /// Check if a connection exists between two Morphons.
    pub fn has_connection(&self, from: MorphonId, to: MorphonId) -> bool {
        self.synapse_between(from, to).is_some()
    }

    /// Get the NodeIndex for a MorphonId.
    pub fn node_index(&self, id: MorphonId) -> Option<NodeIndex> {
        self.id_to_node.get(&id).copied()
    }

    /// Number of Morphons in the topology.
    pub fn morphon_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of synapses in the topology.
    pub fn synapse_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get all MorphonIds in the topology.
    pub fn all_morphon_ids(&self) -> Vec<MorphonId> {
        self.graph.node_weights().copied().collect()
    }

    /// Get all edges with their source and target IDs.
    pub fn all_edges(&self) -> Vec<(MorphonId, MorphonId, EdgeIndex)> {
        self.graph
            .edge_indices()
            .filter_map(|ei| {
                let (src, tgt) = self.graph.edge_endpoints(ei)?;
                Some((self.graph[src], self.graph[tgt], ei))
            })
            .collect()
    }

    /// Get the number of connections (in + out) for a Morphon.
    pub fn degree(&self, id: MorphonId) -> usize {
        let Some(&idx) = self.id_to_node.get(&id) else {
            return 0;
        };
        self.graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .count()
            + self
                .graph
                .edges_directed(idx, petgraph::Direction::Outgoing)
                .count()
    }

    /// Duplicate a subset of connections from parent to child (for mitosis).
    /// Copies ~50% of parent's connections to the child with small weight mutations.
    pub fn duplicate_connections(
        &mut self,
        parent_id: MorphonId,
        child_id: MorphonId,
        rng: &mut impl rand::Rng,
    ) {
        // Copy ~50% of incoming connections
        let incoming: Vec<_> = self.incoming(parent_id)
            .into_iter()
            .map(|(src, syn)| (src, syn.clone()))
            .collect();

        for (src, syn) in incoming {
            if rng.random_bool(0.5) {
                let mut child_syn = syn.clone();
                child_syn.weight += rng.random_range(-0.05..0.05);
                child_syn.age = 0;
                child_syn.usage_count = 0;
                child_syn.eligibility = 0.0;
                child_syn.tag = 0.0;
                child_syn.tag_strength = 0.0;
                child_syn.consolidated = false;
                self.add_synapse(src, child_id, child_syn);
            }
        }

        // Copy ~50% of outgoing connections
        let outgoing: Vec<_> = self.outgoing(parent_id)
            .into_iter()
            .map(|(tgt, syn)| (tgt, syn.clone()))
            .collect();

        for (tgt, syn) in outgoing {
            if rng.random_bool(0.5) {
                let mut child_syn = syn.clone();
                child_syn.weight += rng.random_range(-0.05..0.05);
                child_syn.age = 0;
                child_syn.usage_count = 0;
                child_syn.eligibility = 0.0;
                self.add_synapse(child_id, tgt, child_syn);
            }
        }
    }

    /// Rebuild the id-to-node index map after removals.
    fn rebuild_index_map(&mut self) {
        self.id_to_node.clear();
        for idx in self.graph.node_indices() {
            let id = self.graph[idx];
            self.id_to_node.insert(id, idx);
        }
    }
}
