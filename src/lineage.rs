//! Lineage tracking and export for visualization (e.g. arXiv paper figures).
//!
//! Builds a tree structure from the parent-child relationships encoded in
//! `Morphon::lineage` / `Morphon::generation`, and serializes it to JSON
//! for consumption by external plotting tools.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::morphon::Morphon;
use crate::types::{CellType, Generation, MorphonId};

/// A single node in the lineage tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageNode {
    pub morphon_id: MorphonId,
    pub parent_id: Option<MorphonId>,
    pub generation: Generation,
    pub cell_type: CellType,
    pub age: u64,
    pub energy: f64,
    /// Distance from the Poincaré-ball origin — a proxy for specialization depth.
    pub position_specificity: f64,
}

/// A complete lineage tree built from the current Morphon population.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageTree {
    /// Every node, keyed by morphon id.
    pub nodes: HashMap<MorphonId, LineageNode>,
    /// Pre-computed parent → children index for fast lookup.
    #[serde(skip)]
    children_index: HashMap<MorphonId, Vec<MorphonId>>,
    /// Morphons with no parent (roots / seed cells).
    #[serde(skip)]
    roots: Vec<MorphonId>,
}

impl LineageTree {
    /// Serialize the full tree to a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("LineageTree serialization should not fail")
    }

    /// Return the IDs of all root morphons (those with no parent).
    pub fn root_ids(&self) -> Vec<MorphonId> {
        self.roots.clone()
    }

    /// Return the IDs of the direct children of `id`.
    pub fn children_of(&self, id: MorphonId) -> Vec<MorphonId> {
        self.children_index.get(&id).cloned().unwrap_or_default()
    }

    /// Maximum depth of the tree (the highest generation number present).
    pub fn max_depth(&self) -> u32 {
        self.nodes.values().map(|n| n.generation).max().unwrap_or(0)
    }
}

/// Build a [`LineageTree`] from the current set of morphons.
pub fn build_lineage_tree(morphons: &HashMap<MorphonId, Morphon>) -> LineageTree {
    let mut nodes = HashMap::with_capacity(morphons.len());
    let mut children_index: HashMap<MorphonId, Vec<MorphonId>> = HashMap::new();
    let mut roots = Vec::new();

    for morphon in morphons.values() {
        let node = LineageNode {
            morphon_id: morphon.id,
            parent_id: morphon.lineage,
            generation: morphon.generation,
            cell_type: morphon.cell_type,
            age: morphon.age,
            energy: morphon.energy,
            position_specificity: morphon.position.specificity(),
        };

        match morphon.lineage {
            Some(parent_id) => {
                children_index
                    .entry(parent_id)
                    .or_default()
                    .push(morphon.id);
            }
            None => {
                roots.push(morphon.id);
            }
        }

        nodes.insert(morphon.id, node);
    }

    // Sort for deterministic output.
    roots.sort();
    for children in children_index.values_mut() {
        children.sort();
    }

    LineageTree {
        nodes,
        children_index,
        roots,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphon::Morphon;
    use crate::types::Position;

    fn make_morphon(id: MorphonId, parent: Option<MorphonId>, generation: Generation) -> Morphon {
        let mut m = Morphon::new(id, Position::origin(4));
        m.lineage = parent;
        m.generation = generation;
        m
    }

    #[test]
    fn empty_set_produces_empty_tree() {
        let morphons = HashMap::new();
        let tree = build_lineage_tree(&morphons);
        assert!(tree.root_ids().is_empty());
        assert_eq!(tree.max_depth(), 0);
    }

    #[test]
    fn single_root() {
        let mut morphons = HashMap::new();
        morphons.insert(0, make_morphon(0, None, 0));
        let tree = build_lineage_tree(&morphons);
        assert_eq!(tree.root_ids(), vec![0]);
        assert!(tree.children_of(0).is_empty());
        assert_eq!(tree.max_depth(), 0);
    }

    #[test]
    fn parent_child_relationship() {
        let mut morphons = HashMap::new();
        morphons.insert(0, make_morphon(0, None, 0));
        morphons.insert(1, make_morphon(1, Some(0), 1));
        morphons.insert(2, make_morphon(2, Some(0), 1));
        morphons.insert(3, make_morphon(3, Some(1), 2));

        let tree = build_lineage_tree(&morphons);
        assert_eq!(tree.root_ids(), vec![0]);
        assert_eq!(tree.children_of(0), vec![1, 2]);
        assert_eq!(tree.children_of(1), vec![3]);
        assert!(tree.children_of(2).is_empty());
        assert_eq!(tree.max_depth(), 2);
    }

    #[test]
    fn json_roundtrip() {
        let mut morphons = HashMap::new();
        morphons.insert(0, make_morphon(0, None, 0));
        morphons.insert(1, make_morphon(1, Some(0), 1));

        let tree = build_lineage_tree(&morphons);
        let json = tree.to_json();
        // Verify it's valid JSON by parsing it back.
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("nodes").is_some());
    }
}
