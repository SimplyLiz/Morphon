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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphon::Synapse;

    fn setup_topology() -> Topology {
        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);
        topo.add_morphon(3);
        topo
    }

    #[test]
    fn add_and_count_morphons() {
        let topo = setup_topology();
        assert_eq!(topo.morphon_count(), 3);
        assert_eq!(topo.all_morphon_ids().len(), 3);
    }

    #[test]
    fn add_synapse_creates_connection() {
        let mut topo = setup_topology();
        let ei = topo.add_synapse(1, 2, Synapse::new(0.5));
        assert!(ei.is_some());
        assert!(topo.has_connection(1, 2));
        assert!(!topo.has_connection(2, 1), "directed graph — reverse should not exist");
        assert_eq!(topo.synapse_count(), 1);
    }

    #[test]
    fn add_synapse_returns_none_for_missing_node() {
        let mut topo = setup_topology();
        let ei = topo.add_synapse(1, 99, Synapse::new(0.5));
        assert!(ei.is_none());
    }

    #[test]
    fn remove_morphon_removes_connections() {
        let mut topo = setup_topology();
        topo.add_synapse(1, 2, Synapse::new(0.5));
        topo.add_synapse(2, 3, Synapse::new(0.3));
        topo.add_synapse(3, 1, Synapse::new(0.1));

        assert!(topo.remove_morphon(2));
        assert_eq!(topo.morphon_count(), 2);
        assert_eq!(topo.synapse_count(), 1); // only 3->1 remains
        assert!(!topo.has_connection(1, 2));
        assert!(!topo.has_connection(2, 3));
        assert!(topo.has_connection(3, 1));
    }

    #[test]
    fn remove_nonexistent_morphon_returns_false() {
        let mut topo = setup_topology();
        assert!(!topo.remove_morphon(99));
    }

    #[test]
    fn incoming_outgoing_queries() {
        let mut topo = setup_topology();
        topo.add_synapse(1, 2, Synapse::new(0.5));
        topo.add_synapse(3, 2, Synapse::new(0.3));

        let incoming = topo.incoming(2);
        assert_eq!(incoming.len(), 2);

        let outgoing = topo.outgoing(1);
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].0, 2);
    }

    #[test]
    fn synapse_between_returns_correct_edge() {
        let mut topo = setup_topology();
        topo.add_synapse(1, 2, Synapse::new(0.7));

        let result = topo.synapse_between(1, 2);
        assert!(result.is_some());
        let (_, syn) = result.unwrap();
        assert!((syn.weight - 0.7).abs() < 1e-10);

        assert!(topo.synapse_between(2, 1).is_none());
    }

    #[test]
    fn degree_counts_both_directions() {
        let mut topo = setup_topology();
        topo.add_synapse(1, 2, Synapse::new(0.5));
        topo.add_synapse(3, 2, Synapse::new(0.3));
        topo.add_synapse(2, 3, Synapse::new(0.1));

        assert_eq!(topo.degree(2), 3); // 2 incoming + 1 outgoing
        assert_eq!(topo.degree(99), 0); // nonexistent
    }

    #[test]
    fn synapse_mut_modifies_weight() {
        let mut topo = setup_topology();
        let ei = topo.add_synapse(1, 2, Synapse::new(0.5)).unwrap();
        if let Some(syn) = topo.synapse_mut(ei) {
            syn.weight = 0.9;
        }
        let (_, syn) = topo.synapse_between(1, 2).unwrap();
        assert!((syn.weight - 0.9).abs() < 1e-10);
    }

    #[test]
    fn remove_synapse_works() {
        let mut topo = setup_topology();
        let ei = topo.add_synapse(1, 2, Synapse::new(0.5)).unwrap();
        assert_eq!(topo.synapse_count(), 1);
        topo.remove_synapse(ei);
        assert_eq!(topo.synapse_count(), 0);
        assert!(!topo.has_connection(1, 2));
    }

    #[test]
    fn duplicate_connections_copies_subset() {
        let mut topo = Topology::new();
        topo.add_morphon(1);
        topo.add_morphon(2);
        topo.add_morphon(3);
        topo.add_morphon(4); // child

        // Parent (2) has connections
        topo.add_synapse(1, 2, Synapse::new(0.5));
        topo.add_synapse(3, 2, Synapse::new(0.3));
        topo.add_synapse(2, 3, Synapse::new(0.7));

        let mut rng = rand::rng();
        // Run many times to check it doesn't panic and produces reasonable results
        let mut total_child_degree = 0;
        for _ in 0..20 {
            let mut topo_copy = Topology::new();
            topo_copy.add_morphon(1);
            topo_copy.add_morphon(2);
            topo_copy.add_morphon(3);
            topo_copy.add_morphon(4);
            topo_copy.add_synapse(1, 2, Synapse::new(0.5));
            topo_copy.add_synapse(3, 2, Synapse::new(0.3));
            topo_copy.add_synapse(2, 3, Synapse::new(0.7));

            topo_copy.duplicate_connections(2, 4, &mut rng);
            total_child_degree += topo_copy.degree(4);
        }
        // With 3 parent connections and 50% copy probability, average child degree ~1.5 per run
        // Over 20 runs, should be > 0
        assert!(total_child_degree > 0, "duplicate_connections should create some connections");
    }

    #[test]
    fn all_edges_returns_all() {
        let mut topo = setup_topology();
        topo.add_synapse(1, 2, Synapse::new(0.5));
        topo.add_synapse(2, 3, Synapse::new(0.3));
        topo.add_synapse(3, 1, Synapse::new(0.1));

        let edges = topo.all_edges();
        assert_eq!(edges.len(), 3);
    }
}
