//! V3 Synaptic Justification Records — provenance tracking for connections.
//!
//! Every synapse tracks *why* it was formed, *what* reinforced it, and *what*
//! depends on it.  This is the foundation for the epistemic model: you cannot
//! assess the validity of a belief without knowing its justification chain.
//!
//! Memory is bounded: reinforcement history is a ring buffer (capacity 16).

use crate::types::{ClusterId, MorphonId};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Maximum number of reinforcement events stored per synapse.
const MAX_REINFORCEMENT_HISTORY: usize = 16;

/// Why a synapse was formed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormationCause {
    /// Hebb'ian coincidence: pre and post fired in correlation.
    HebbianCoincidence { step: u64 },

    /// Inherited from parent morphon during mitosis.
    InheritedFromDivision { parent: MorphonId },

    /// Formed by spatial proximity during synaptogenesis.
    ProximityFormation { distance: f64 },

    /// Bridge created during cluster fusion.
    FusionBridge { cluster: ClusterId },

    /// Externally specified (developmental program, user injection).
    External { source: String },
}

/// A single reinforcement event — records when and how a synapse was modified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReinforcementEvent {
    /// Simulation step when reinforcement occurred.
    pub step: u64,
    /// Weight change applied.
    pub delta_weight: f64,
    /// Dominant modulation level at time of update.
    pub modulation_level: f64,
}

/// Provenance record for a synapse.
///
/// Tracks formation cause and bounded reinforcement history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticJustification {
    /// Why this synapse was created.
    pub formation_cause: FormationCause,
    /// Simulation step at formation.
    pub formation_step: u64,
    /// Bounded ring of recent reinforcement events.
    pub reinforcement_history: VecDeque<ReinforcementEvent>,
    /// ANCS-Core: the memory item ID that was active when this synapse was
    /// last significantly reinforced. Used by TruthKeeper to locate synapses
    /// that encoded a Contested memory and route them for reconsolidation.
    #[serde(default)]
    pub memory_item_ref: Option<u64>,  // MemoryItemId (u64) — no circular dep
}

impl SynapticJustification {
    /// Create a new justification record.
    pub fn new(cause: FormationCause, step: u64) -> Self {
        Self {
            formation_cause: cause,
            formation_step: step,
            reinforcement_history: VecDeque::with_capacity(MAX_REINFORCEMENT_HISTORY),
            memory_item_ref: None,
        }
    }

    /// Record a reinforcement event, evicting the oldest if at capacity.
    pub fn record_reinforcement(&mut self, step: u64, delta_weight: f64, modulation_level: f64) {
        if self.reinforcement_history.len() >= MAX_REINFORCEMENT_HISTORY {
            self.reinforcement_history.pop_front();
        }
        self.reinforcement_history.push_back(ReinforcementEvent {
            step,
            delta_weight,
            modulation_level,
        });
    }

    /// Step of the most recent reinforcement, or formation step if never reinforced.
    pub fn last_reinforcement_step(&self) -> u64 {
        self.reinforcement_history
            .back()
            .map(|e| e.step)
            .unwrap_or(self.formation_step)
    }

    /// True if the synapse has been reinforced at least once.
    pub fn has_reinforcement(&self) -> bool {
        !self.reinforcement_history.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reinforcement_history_bounded() {
        let mut j = SynapticJustification::new(
            FormationCause::HebbianCoincidence { step: 0 },
            0,
        );
        for i in 0..20 {
            j.record_reinforcement(i, 0.01, 0.5);
        }
        assert_eq!(j.reinforcement_history.len(), MAX_REINFORCEMENT_HISTORY);
        // oldest should be step 4 (0-3 evicted)
        assert_eq!(j.reinforcement_history.front().unwrap().step, 4);
    }

    #[test]
    fn last_reinforcement_step_returns_formation_if_empty() {
        let j = SynapticJustification::new(
            FormationCause::External { source: "test".into() },
            42,
        );
        assert_eq!(j.last_reinforcement_step(), 42);
    }

    #[test]
    fn last_reinforcement_step_returns_latest() {
        let mut j = SynapticJustification::new(
            FormationCause::HebbianCoincidence { step: 0 },
            0,
        );
        j.record_reinforcement(10, 0.1, 0.5);
        j.record_reinforcement(20, 0.2, 0.8);
        assert_eq!(j.last_reinforcement_step(), 20);
    }
}
