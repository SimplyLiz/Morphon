//! Dual-Clock Architecture — separating fast inference from slow morphogenesis.
//!
//! Not all processes run on the same timescale. MI uses structural quantization:
//!
//! | Path     | Processes                                           | Period  |
//! |----------|-----------------------------------------------------|---------|
//! | Fast     | Spike propagation, resonance, threshold comparisons | Every 1 |
//! | Medium   | Synaptic plasticity, eligibility traces, scaling    | Every 10|
//! | Slow     | Synaptogenesis/pruning, migration, tag-and-capture  | Every 100|
//! | Glacial  | Division, differentiation, fusion/defusion, apoptosis| Every 1000|
//!
//! The fast path must run at real-time even with 1M Morphons.
//! The slow path is deliberately rate-limited to maintain stability.

use serde::{Deserialize, Serialize};

/// Scheduler configuration — controls the timing of each process path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Period for medium-path processes (synaptic plasticity, scaling).
    pub medium_period: u64,
    /// Period for slow-path processes (synaptogenesis, pruning, migration).
    pub slow_period: u64,
    /// Period for glacial-path processes (division, differentiation, fusion, apoptosis).
    pub glacial_period: u64,
    /// Period for homeostatic checks (synaptic scaling, inter-cluster inhibition).
    pub homeostasis_period: u64,
    /// Period for memory system updates (procedural topology snapshots).
    pub memory_period: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            medium_period: 10,
            slow_period: 100,
            glacial_period: 1000,
            homeostasis_period: 50,
            memory_period: 100,
        }
    }
}

/// Determines which processes should run at a given step.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SchedulerTick {
    /// Always true — fast path runs every step.
    pub fast: bool,
    /// Synaptic plasticity, eligibility traces.
    pub medium: bool,
    /// Synaptogenesis, pruning, migration.
    pub slow: bool,
    /// Division, differentiation, fusion, apoptosis.
    pub glacial: bool,
    /// Synaptic scaling, inter-cluster inhibition.
    pub homeostasis: bool,
    /// Memory system updates.
    pub memory: bool,
}

impl SchedulerConfig {
    /// Determine which paths should execute at the given step number.
    pub fn tick(&self, step: u64) -> SchedulerTick {
        SchedulerTick {
            fast: true,
            medium: step % self.medium_period == 0,
            slow: step % self.slow_period == 0,
            glacial: step % self.glacial_period == 0,
            homeostasis: step % self.homeostasis_period == 0,
            memory: step % self.memory_period == 0,
        }
    }
}
