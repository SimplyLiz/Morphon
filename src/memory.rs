//! Triple Memory Architecture — Working, Episodic, and Procedural memory.
//!
//! - Working Memory: persistent activity patterns (attractors) in Morphon clusters
//! - Episodic Memory: fast synaptic changes (one-shot learning via high Novelty)
//! - Procedural Memory: the topology itself — structure IS knowledge

use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A working memory item — a pattern of active Morphon IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemoryItem {
    /// The set of Morphon IDs that form this attractor pattern.
    pub pattern: Vec<MorphonId>,
    /// How strongly this pattern is currently activated.
    pub activation: f64,
    /// Time since last reactivation (decays without refresh).
    pub decay_timer: f64,
}

/// Working Memory — implemented through persistent activity patterns.
/// Capacity-limited (like Miller's 7±2).
pub struct WorkingMemory {
    /// Active patterns.
    items: VecDeque<WorkingMemoryItem>,
    /// Maximum number of simultaneous patterns.
    capacity: usize,
    /// Decay rate per timestep.
    decay_rate: f64,
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            items: VecDeque::new(),
            capacity,
            decay_rate: 0.05,
        }
    }

    /// Store a new pattern (or refresh if similar pattern exists).
    pub fn store(&mut self, pattern: Vec<MorphonId>, activation: f64) {
        // Check for similar existing pattern (>50% overlap)
        for item in &mut self.items {
            let overlap = pattern
                .iter()
                .filter(|id| item.pattern.contains(id))
                .count();
            let similarity = overlap as f64 / pattern.len().max(item.pattern.len()) as f64;
            if similarity > 0.5 {
                // Refresh existing pattern
                item.activation = (item.activation + activation).min(1.0);
                item.decay_timer = 0.0;
                return;
            }
        }

        // Add new pattern
        if self.items.len() >= self.capacity {
            // Remove least activated
            if let Some(min_idx) = self
                .items
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.activation
                        .partial_cmp(&b.activation)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.items.remove(min_idx);
            }
        }

        self.items.push_back(WorkingMemoryItem {
            pattern,
            activation,
            decay_timer: 0.0,
        });
    }

    /// Decay all items. Remove those that have decayed below threshold.
    pub fn step(&mut self, dt: f64) {
        for item in &mut self.items {
            item.decay_timer += dt;
            item.activation *= 1.0 - self.decay_rate * dt;
        }
        self.items.retain(|item| item.activation > 0.01);
    }

    /// Get currently active patterns.
    pub fn active_patterns(&self) -> &VecDeque<WorkingMemoryItem> {
        &self.items
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// An episodic memory trace — a snapshot of a significant event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// The pattern of Morphon activations at the time of encoding.
    pub pattern: Vec<(MorphonId, f64)>,
    /// Reward signal at the time of encoding.
    pub reward: f64,
    /// Novelty at encoding time.
    pub novelty: f64,
    /// Simulation step when encoded.
    pub timestamp: u64,
    /// Consolidation level: 0.0 = fresh, 1.0 = fully consolidated.
    pub consolidation: f64,
    /// Number of times this episode has been replayed.
    pub replay_count: u32,
}

/// Episodic Memory — fast one-shot storage with consolidation.
pub struct EpisodicMemory {
    /// Stored episodes.
    episodes: Vec<Episode>,
    /// Maximum number of episodes before forgetting.
    capacity: usize,
    /// Consolidation rate per replay.
    consolidation_rate: f64,
}

impl EpisodicMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            episodes: Vec::new(),
            capacity,
            consolidation_rate: 0.1,
        }
    }

    /// Encode a new episode (triggered by high novelty).
    pub fn encode(
        &mut self,
        pattern: Vec<(MorphonId, f64)>,
        reward: f64,
        novelty: f64,
        timestamp: u64,
    ) {
        if self.episodes.len() >= self.capacity {
            // Remove least consolidated, oldest episode
            if let Some(idx) = self
                .episodes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.consolidation
                        .partial_cmp(&b.consolidation)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.episodes.remove(idx);
            }
        }

        self.episodes.push(Episode {
            pattern,
            reward,
            novelty,
            timestamp,
            consolidation: 0.0,
            replay_count: 0,
        });
    }

    /// Replay episodes for consolidation (like hippocampal replay during sleep).
    /// Returns episodes that should be replayed to strengthen connections.
    pub fn replay(&mut self, count: usize) -> Vec<&Episode> {
        // Prioritize high-reward, high-novelty, under-consolidated episodes
        self.episodes.sort_by(|a, b| {
            let score_a = a.reward + a.novelty - a.consolidation;
            let score_b = b.reward + b.novelty - b.consolidation;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let to_replay: Vec<usize> = (0..count.min(self.episodes.len())).collect();

        for &idx in &to_replay {
            self.episodes[idx].replay_count += 1;
            self.episodes[idx].consolidation =
                (self.episodes[idx].consolidation + self.consolidation_rate).min(1.0);
        }

        to_replay
            .iter()
            .map(|&idx| &self.episodes[idx])
            .collect()
    }

    pub fn episodes(&self) -> &[Episode] {
        &self.episodes
    }

    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }
}

/// Procedural Memory — the topology IS the memory.
///
/// This is not a separate data structure — it's the insight that
/// the network's structure encodes procedural knowledge. This struct
/// provides analysis and query capabilities over that topology.
pub struct ProceduralMemory {
    /// Snapshots of topology statistics over time.
    pub topology_history: Vec<TopologySnapshot>,
    /// Maximum history length.
    capacity: usize,
}

/// A snapshot of the topology at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologySnapshot {
    pub timestamp: u64,
    pub morphon_count: usize,
    pub synapse_count: usize,
    pub avg_connectivity: f64,
    pub cluster_count: usize,
}

impl ProceduralMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            topology_history: Vec::new(),
            capacity,
        }
    }

    /// Record a snapshot of the current topology state.
    pub fn record(
        &mut self,
        timestamp: u64,
        morphon_count: usize,
        synapse_count: usize,
        cluster_count: usize,
    ) {
        let avg_connectivity = if morphon_count > 0 {
            synapse_count as f64 / morphon_count as f64
        } else {
            0.0
        };

        if self.topology_history.len() >= self.capacity {
            self.topology_history.remove(0);
        }

        self.topology_history.push(TopologySnapshot {
            timestamp,
            morphon_count,
            synapse_count,
            avg_connectivity,
            cluster_count,
        });
    }
}

/// The combined triple-memory system.
pub struct TripleMemory {
    pub working: WorkingMemory,
    pub episodic: EpisodicMemory,
    pub procedural: ProceduralMemory,
}

impl TripleMemory {
    pub fn new(working_capacity: usize, episodic_capacity: usize) -> Self {
        Self {
            working: WorkingMemory::new(working_capacity),
            episodic: EpisodicMemory::new(episodic_capacity),
            procedural: ProceduralMemory::new(1000),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === WorkingMemory tests ===

    #[test]
    fn working_memory_store_and_retrieve() {
        let mut wm = WorkingMemory::new(5);
        assert!(wm.is_empty());

        wm.store(vec![1, 2, 3], 0.8);
        assert_eq!(wm.len(), 1);
        assert_eq!(wm.active_patterns()[0].pattern, vec![1, 2, 3]);
    }

    #[test]
    fn working_memory_refreshes_similar_pattern() {
        let mut wm = WorkingMemory::new(5);
        wm.store(vec![1, 2, 3], 0.5);
        wm.store(vec![1, 2, 4], 0.3); // >50% overlap with [1,2,3]

        assert_eq!(wm.len(), 1, "similar pattern should refresh, not create new entry");
        assert!(
            wm.active_patterns()[0].activation > 0.5,
            "activation should increase on refresh"
        );
    }

    #[test]
    fn working_memory_stores_dissimilar_patterns_separately() {
        let mut wm = WorkingMemory::new(5);
        wm.store(vec![1, 2, 3], 0.5);
        wm.store(vec![10, 20, 30], 0.5); // no overlap

        assert_eq!(wm.len(), 2);
    }

    #[test]
    fn working_memory_evicts_weakest_at_capacity() {
        let mut wm = WorkingMemory::new(3);
        wm.store(vec![1], 0.3);
        wm.store(vec![2], 0.9);
        wm.store(vec![3], 0.7);
        assert_eq!(wm.len(), 3);

        // This should evict the weakest (pattern [1] with activation 0.3)
        wm.store(vec![4], 0.5);
        assert_eq!(wm.len(), 3);

        let patterns: Vec<Vec<u64>> = wm
            .active_patterns()
            .iter()
            .map(|p| p.pattern.clone())
            .collect();
        assert!(!patterns.contains(&vec![1]), "weakest pattern should be evicted");
    }

    #[test]
    fn working_memory_decay_removes_weak_items() {
        let mut wm = WorkingMemory::new(5);
        wm.store(vec![1], 0.02); // very low activation

        // Decay until removed
        for _ in 0..100 {
            wm.step(1.0);
        }
        assert!(wm.is_empty(), "very weak items should decay away");
    }

    #[test]
    fn working_memory_activation_capped_at_one() {
        let mut wm = WorkingMemory::new(5);
        wm.store(vec![1, 2], 0.8);
        wm.store(vec![1, 2], 0.8); // refresh same pattern

        assert!(
            wm.active_patterns()[0].activation <= 1.0,
            "activation should be capped at 1.0"
        );
    }

    // === EpisodicMemory tests ===

    #[test]
    fn episodic_encode_and_len() {
        let mut em = EpisodicMemory::new(10);
        assert!(em.is_empty());

        em.encode(vec![(1, 0.5), (2, 0.8)], 0.9, 0.7, 100);
        assert_eq!(em.len(), 1);
    }

    #[test]
    fn episodic_evicts_least_consolidated_at_capacity() {
        let mut em = EpisodicMemory::new(3);
        em.encode(vec![(1, 0.5)], 0.5, 0.5, 100);
        em.encode(vec![(2, 0.5)], 0.5, 0.5, 200);
        em.encode(vec![(3, 0.5)], 0.5, 0.5, 300);
        assert_eq!(em.len(), 3);

        em.encode(vec![(4, 0.5)], 0.9, 0.9, 400);
        assert_eq!(em.len(), 3, "should stay at capacity");
    }

    #[test]
    fn episodic_replay_prioritizes_high_reward_high_novelty() {
        let mut em = EpisodicMemory::new(10);
        em.encode(vec![(1, 0.5)], 0.1, 0.1, 100); // low priority
        em.encode(vec![(2, 0.5)], 0.9, 0.9, 200); // high priority
        em.encode(vec![(3, 0.5)], 0.5, 0.5, 300); // medium priority

        let replayed = em.replay(2);
        assert_eq!(replayed.len(), 2);
        // Highest priority episode should be first
        assert!(
            replayed[0].reward + replayed[0].novelty >= replayed[1].reward + replayed[1].novelty,
            "replay should prioritize high reward+novelty"
        );
    }

    #[test]
    fn episodic_replay_consolidates() {
        let mut em = EpisodicMemory::new(10);
        em.encode(vec![(1, 0.5)], 0.8, 0.8, 100);

        let replayed = em.replay(1);
        assert_eq!(replayed[0].replay_count, 1);
        assert!(replayed[0].consolidation > 0.0);

        // Replay again
        let replayed = em.replay(1);
        assert_eq!(replayed[0].replay_count, 2);
    }

    // === ProceduralMemory tests ===

    #[test]
    fn procedural_record_and_history() {
        let mut pm = ProceduralMemory::new(5);
        pm.record(100, 50, 120, 3);
        pm.record(200, 55, 130, 4);

        assert_eq!(pm.topology_history.len(), 2);
        assert_eq!(pm.topology_history[0].morphon_count, 50);
        assert_eq!(pm.topology_history[1].cluster_count, 4);
    }

    #[test]
    fn procedural_evicts_oldest_at_capacity() {
        let mut pm = ProceduralMemory::new(3);
        pm.record(100, 10, 20, 1);
        pm.record(200, 20, 40, 2);
        pm.record(300, 30, 60, 3);
        pm.record(400, 40, 80, 4);

        assert_eq!(pm.topology_history.len(), 3);
        assert_eq!(
            pm.topology_history[0].timestamp, 200,
            "oldest entry should be evicted"
        );
    }

    #[test]
    fn procedural_avg_connectivity_computed() {
        let mut pm = ProceduralMemory::new(10);
        pm.record(100, 10, 30, 1);

        assert!(
            (pm.topology_history[0].avg_connectivity - 3.0).abs() < 1e-10,
            "avg_connectivity should be synapses/morphons"
        );
    }

    #[test]
    fn procedural_zero_morphons_no_panic() {
        let mut pm = ProceduralMemory::new(10);
        pm.record(100, 0, 0, 0);
        assert!((pm.topology_history[0].avg_connectivity - 0.0).abs() < 1e-10);
    }

    // === TripleMemory tests ===

    #[test]
    fn triple_memory_creates_all_subsystems() {
        let tm = TripleMemory::new(7, 100);
        assert!(tm.working.is_empty());
        assert!(tm.episodic.is_empty());
        assert!(tm.procedural.topology_history.is_empty());
    }
}
