//! Cache-friendly dense arrays for the fast-path inner loop.
//!
//! These MIRROR fields from each Morphon struct.
//! Source of truth remains the Morphon structs.
//! Sync happens once per medium-path tick (every 10 steps).

use crate::types::MorphonId;
use std::collections::HashMap;

/// Hot-path arrays for cache-friendly spike processing.
pub struct HotArrays {
    /// Membrane potential. Written every tick by fast_integrate().
    pub voltage: Vec<f32>,
    /// Adaptive firing threshold. Written by sync_structs_to_hot() on medium tick.
    pub threshold: Vec<f32>,
    /// Synapse input current for this tick. Written by extract loop, read by fast_integrate().
    pub input_current: Vec<f32>,
    /// Did this morphon fire this tick?
    pub fired: Vec<bool>,
    /// Previous tick's fired state. Swapped with fired at start of each tick.
    pub fired_prev: Vec<bool>,
    /// Refractory countdown (in ticks). Decremented on each fast tick.
    pub refractory: Vec<f32>,
    /// Per-morphon voltage decay (1 - leak_rate). Motor=0.0, others=0.9.
    pub leak_decay: Vec<f32>,
    /// Metabolic energy [0.0, 1.0]. Deducted on fire; clamped to 0.0 min.
    /// Checked in threshold loop — zero-energy morphons cannot fire.
    /// Replenished on medium tick via medium_update().
    pub energy: Vec<f32>,
    /// Maps hot index → MorphonId. Stable within a slow-path epoch.
    pub idx_to_id: Vec<MorphonId>,
    /// Maps MorphonId → hot index. Sparse.
    pub id_to_idx: HashMap<MorphonId, usize>,
    /// How many active morphons are in the arrays.
    pub active_count: usize,
}

impl HotArrays {
    pub fn new() -> Self {
        Self {
            voltage: Vec::new(),
            threshold: Vec::new(),
            input_current: Vec::new(),
            fired: Vec::new(),
            fired_prev: Vec::new(),
            refractory: Vec::new(),
            leak_decay: Vec::new(),
            energy: Vec::new(),
            idx_to_id: Vec::new(),
            id_to_idx: HashMap::new(),
            active_count: 0,
        }
    }

    /// Allocate storage for n morphons (does not set active_count).
    pub fn allocate(&mut self, n: usize) {
        self.voltage = vec![0.0f32; n];
        self.threshold = vec![0.3f32; n];
        self.input_current = vec![0.0f32; n];
        self.fired = vec![false; n];
        self.fired_prev = vec![false; n];
        self.refractory = vec![0.0f32; n];
        self.leak_decay = vec![0.9f32; n];
        self.energy = vec![1.0f32; n];
        self.idx_to_id = vec![0; n];
        // id_to_idx cleared on rebuild
    }

    pub fn get_idx(&self, id: MorphonId) -> Option<usize> {
        self.id_to_idx.get(&id).copied()
    }
}
