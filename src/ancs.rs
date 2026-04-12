//! ANCS-Core — Native Rust Memory Substrate for Morphon.
//!
//! Implements the Adaptive Node-Cluster System memory architecture directly
//! in-process: no HTTP, no PostgreSQL, no separate runtime. Memory access is
//! a pointer dereference, not a network round-trip.
//!
//! ## Biological mapping
//! - `InMemoryBackend`  ↔ Hippocampus (episodic binding, consolidation, replay)
//! - AXION importance   ↔ SM-2 forgetting curves (6-factor composite score)
//! - TruthKeeper        ↔ Epistemic immune system (cascading invalidation)
//! - `MemoryTier`       ↔ Neocortical consolidation gradient (vivid → schematic)
//!
//! ## Architecture
//! ```text
//! Old:  MORPHON (Rust) ──HTTP──→ ANCS (Node.js) ──SQL──→ PostgreSQL
//! New:  MORPHON + ANCS-Core: one binary, shared memory, zero serialization
//! ```
//!
//! ## Phases implemented here
//! - Phase 0: `MemoryBackend` trait + `MemoryItem` + `InMemoryBackend`
//! - Phase 1: VBC-lite tier classification + AXION 6-factor importance + F7 pressure modes
//! - Phase 2: TruthKeeper epistemic transitions + blast-radius cascade + reconsolidation
//! - Phase 3: Epistemic-filtered RRF retrieval (Phase 3 RRF across stores lives in memory.rs)
//!
//! Phase 4 (forward-importance pruning) is already implemented via `Synapse::reward_correlation`
//! in `learning.rs`. Phase 5 (SOMNUS sleep/wake) is deferred.

use crate::types::MorphonId;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Unique identifier for a memory item. Counter-based, monotonically increasing.
pub type MemoryItemId = u64;

// ─── Tier ────────────────────────────────────────────────────────────────────

/// Memory tier — determines encoding fidelity and eviction priority.
///
/// VBC-lite: the system's own neuromodulatory signals determine tier.
/// No external classifier; the same signals that drive learning classify storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryTier {
    /// T0: Verbatim — high-significance moment, immutable, hash-verified.
    /// Protected from eviction; only demoted when importance drops and consolidation is low.
    Verbatim,
    /// T1: Structural — lossless copy of an active working-memory pattern.
    Structural,
    /// T2: Semantic — relational, lossy (standard episodic capture).
    Semantic,
    /// T3: Procedural — background topology snapshot, lowest priority.
    Procedural,
}

// ─── Epistemic state ─────────────────────────────────────────────────────────

/// 6-state epistemic lifecycle for individual memory items.
///
/// Separate from `EpistemicState` in `epistemic.rs` (which applies to *clusters*).
/// Items transition through states based on consolidation, importance, and
/// conflict detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryEpistemicState {
    /// Just encoded, not yet consolidated.
    Hypothesis,
    /// Confirmed: consolidation ≥ 0.5 or positive replay correlation.
    Supported,
    /// Source pattern drifted: importance fell below `stale_threshold`.
    Stale,
    /// Transitive dependent of a Stale item (within `blast_window` steps).
    Suspect,
    /// Contradicted by a newer item with opposing reward sign.
    /// Synapses encoded during this item's window are queued for reconsolidation.
    Contested,
    /// Superseded or timed-out — candidate for eviction.
    Outdated,
}

// ─── Pressure ────────────────────────────────────────────────────────────────

/// Energy pressure mode — gates eviction aggressiveness (F7 system).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PressureMode {
    /// energy_usage < 0.70 — full 6-factor AXION scoring, normal eviction.
    Normal,
    /// 0.70–0.85 — fast-path eviction: importance + pinned only.
    Pressure,
    /// 0.85–0.95 — only pinned (consolidated) items survive.
    Emergency,
    /// ≥ 0.95 — safe mode, governor takes over.
    Critical,
}

impl PressureMode {
    /// Derive from `energy_usage` ∈ [0, 1].
    /// `energy_usage = 1 - (total_energy / max_energy)`.
    pub fn from_usage(usage: f64) -> Self {
        if usage < 0.70 {
            PressureMode::Normal
        } else if usage < 0.85 {
            PressureMode::Pressure
        } else if usage < 0.95 {
            PressureMode::Emergency
        } else {
            PressureMode::Critical
        }
    }
}

// ─── MemoryItem ───────────────────────────────────────────────────────────────

/// A unified memory item spanning all tiers.
///
/// Created from episodic events, classified into tiers via VBC-lite, and
/// scored continuously by AXION importance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: MemoryItemId,
    pub tier: MemoryTier,
    /// Morphon activations at encoding time: (MorphonId, activation).
    pub pattern: Vec<(MorphonId, f64)>,
    pub reward: f64,
    pub novelty: f64,
    /// System step at encoding.
    pub timestamp: u64,
    pub epistemic: MemoryEpistemicState,
    /// Consolidation level [0.0, 1.0]. 0 = fresh, 1 = fully consolidated.
    pub consolidation: f64,
    /// AXION composite importance score [0.0, 1.0], updated on each `step()`.
    pub importance: f64,
    /// SM-2 stability — governs forgetting rate. Grows with replays.
    pub stability: f64,
    pub replay_count: u32,
    pub access_count: u32,
    /// T0 only: std `DefaultHasher` fingerprint of the pattern for integrity checks.
    pub content_hash: Option<u64>,
    /// Step when epistemic state last changed — used for timeout transitions.
    pub state_changed_at: u64,
}

impl MemoryItem {
    /// Construct a new item in `Hypothesis` state.
    pub fn new(
        tier: MemoryTier,
        pattern: Vec<(MorphonId, f64)>,
        reward: f64,
        novelty: f64,
        timestamp: u64,
    ) -> Self {
        let content_hash = if tier == MemoryTier::Verbatim {
            Some(hash_pattern(&pattern))
        } else {
            None
        };
        let init_importance = novelty * 0.5 + reward.abs() * 0.5;
        Self {
            id: 0, // assigned by InMemoryBackend::store()
            tier,
            pattern,
            reward,
            novelty,
            timestamp,
            epistemic: MemoryEpistemicState::Hypothesis,
            consolidation: 0.0,
            importance: init_importance,
            stability: 1.0,
            replay_count: 0,
            access_count: 0,
            content_hash,
            state_changed_at: timestamp,
        }
    }

    /// MorphonIds participating in this memory's pattern.
    pub fn morphon_ids(&self) -> Vec<MorphonId> {
        self.pattern.iter().map(|(id, _)| *id).collect()
    }

    /// Verify T0 verbatim integrity. Returns `true` for non-Verbatim items.
    pub fn verify_integrity(&self) -> bool {
        match self.content_hash {
            None => true,
            Some(stored) => hash_pattern(&self.pattern) == stored,
        }
    }
}

fn hash_pattern(pattern: &[(MorphonId, f64)]) -> u64 {
    let mut h = DefaultHasher::new();
    for (id, v) in pattern {
        id.hash(&mut h);
        v.to_bits().hash(&mut h);
    }
    h.finish()
}

// ─── RetrievalQuery ──────────────────────────────────────────────────────────

/// Query for memory retrieval against the ANCS-Core store.
pub struct RetrievalQuery {
    /// MorphonIds to match against stored patterns.
    pub pattern: Vec<MorphonId>,
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the ANCS-Core memory backend.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AncsConfig {
    /// Hard capacity limit. Items evicted by importance when exceeded.
    pub capacity: usize,
    /// Steps before an unconsolidated Hypothesis item is marked Outdated.
    pub hypothesis_timeout: u64,
    /// Steps before a Contested or Stale item is marked Outdated.
    pub contested_timeout: u64,
    /// Half-window (in steps) for blast-radius cascade from a Stale item.
    pub blast_window: u64,
    /// Importance threshold below which a Supported item transitions to Stale.
    pub stale_threshold: f64,
    /// Jaccard overlap required to flag a new item as conflicting an existing one.
    pub conflict_overlap_threshold: f64,
}

impl Default for AncsConfig {
    fn default() -> Self {
        Self {
            capacity: 2000,
            hypothesis_timeout: 500,
            contested_timeout: 1000,
            blast_window: 50,
            stale_threshold: 0.15,
            conflict_overlap_threshold: 0.30,
        }
    }
}

// ─── MemoryBackend trait ─────────────────────────────────────────────────────

/// Uniform interface over the ANCS-Core memory store.
///
/// `System::step()` uses this trait to drive ANCS without coupling to the
/// concrete backend type. A future persistent backend (redb-backed) can
/// implement the same trait.
pub trait MemoryBackend: Send + Sync {
    /// Store a new item. The backend assigns a fresh `MemoryItemId`.
    fn store(&mut self, item: MemoryItem);
    /// Retrieve top-k items by pattern overlap. Epistemic filter: Supported + Hypothesis only.
    fn retrieve(&self, query: &RetrievalQuery, top_k: usize) -> Vec<(MemoryItemId, f64)>;
    /// Advance time: recalculate importance, run epistemic transitions, evict under pressure.
    fn step(&mut self, dt: f64, current_step: u64, energy_usage: f64);
    /// Record a retrieval access (bumps access_count, affects f2 importance factor).
    fn record_access(&mut self, id: MemoryItemId);
    /// Current AXION importance score for a given item.
    fn importance(&self, id: MemoryItemId) -> f64;
    /// Current epistemic state for a given item.
    fn epistemic_state(&self, id: MemoryItemId) -> Option<MemoryEpistemicState>;
    /// All item IDs currently in the given epistemic state.
    fn items_with_state(&self, state: MemoryEpistemicState) -> Vec<MemoryItemId>;
    /// Force-transition an item to Stale and trigger blast-radius cascade.
    fn mark_stale(&mut self, id: MemoryItemId, current_step: u64);
    /// Current F7 pressure mode.
    fn pressure_mode(&self) -> PressureMode;
    /// Drain patterns pending synapse reconsolidation.
    ///
    /// Each `Vec<(MorphonId, f64)>` is the pattern of a memory item that
    /// transitioned to Contested. System should call `learning::reconsolidate_pattern`
    /// for each one to re-open the corresponding synapses.
    fn take_reconsolidate_patterns(&mut self) -> Vec<Vec<(MorphonId, f64)>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// ID of the most recently stored item, if any.
    fn last_stored_id(&self) -> Option<MemoryItemId>;
    /// Pattern (MorphonIds) of the item with the given id, if it exists.
    fn item_pattern(&self, id: MemoryItemId) -> Option<Vec<MorphonId>>;
}

// ─── VBC-lite tier classification ────────────────────────────────────────────

/// VBC-lite: classify an encoding event into a `MemoryTier` using neuromodulatory signals.
///
/// Replaces an external ML classifier with the system's own internal signals.
/// Biologically: salience circuits (LC-NE, VTA-DA) gate hippocampal encoding.
///
/// - `novelty`: current global novelty (ACh analog), 0.0–1.0
/// - `reward`: current reward signal (dopamine analog), −1.0–1.0
/// - `working_overlap`: Jaccard overlap with any active working-memory pattern, 0.0–1.0
pub fn classify_tier(novelty: f64, reward: f64, working_overlap: f64) -> MemoryTier {
    if novelty > 0.7 && reward.abs() > 0.5 {
        // High-salience moment: protect verbatim, hash-verify later
        MemoryTier::Verbatim
    } else if working_overlap > 0.5 {
        // Reinforcing an already-active working-memory pattern
        MemoryTier::Structural
    } else if novelty > 0.3 {
        // Standard episodic capture (matches existing novelty > 0.3 gate)
        MemoryTier::Semantic
    } else {
        // Background topology trace
        MemoryTier::Procedural
    }
}

// ─── AXION 6-factor importance ────────────────────────────────────────────────

/// Compute AXION 6-factor composite importance score ∈ [0, 1].
///
/// Factors:
/// - **f1** retrievability: SM-2 forgetting curve — probability of recall right now
/// - **f2** access frequency: normalized 1 − e^(−0.3·count)
/// - **f3** structural centrality: caller-provided hint (0.5 = unknown/average)
/// - **f4** encoding surprise: novelty at time of encoding
/// - **f5** pinned: whether consolidation > 0.8 (protected)
/// - **f6** task relevance: reward magnitude at encoding
///
/// Weights: f1 = 0.25, f2–f6 = 0.15 each.
///
/// The `centrality_hint` parameter lets `System` pass degree/max_degree for the
/// pattern's primary morphon without requiring a full topology traversal here.
/// Pass 0.5 when unknown.
pub fn compute_importance(item: &MemoryItem, current_step: u64, centrality_hint: f64) -> f64 {
    let age = current_step.saturating_sub(item.timestamp) as f64;
    let stability = item.stability.max(0.1);

    // f1: SM-2 retrievability — S-shaped forgetting (Leitner/Wozniak SM-2)
    let f1 = (1.0 + (19.0 / 81.0) * age / stability).powf(-0.5);

    // f2: Access frequency (saturates around count ≈ 10)
    let f2 = 1.0 - (-0.3 * (item.access_count as f64 + 1.0)).exp();

    // f3: Structural centrality (topology-dependent, approximated by caller)
    let f3 = centrality_hint.clamp(0.0, 1.0);

    // f4: Encoding surprise
    let f4 = item.novelty;

    // f5: Pinned (consolidated memories are constitutionally protected)
    let f5 = if item.consolidation > 0.8 {
        1.0
    } else {
        item.consolidation
    };

    // f6: Task relevance — reward magnitude at encoding
    let f6 = item.reward.abs().min(1.0);

    0.25 * f1 + 0.15 * f2 + 0.15 * f3 + 0.15 * f4 + 0.15 * f5 + 0.15 * f6
}

// ─── InMemoryBackend ─────────────────────────────────────────────────────────

/// In-process ANCS-Core backend — zero external dependencies.
///
/// Items are stored in a flat `Vec<MemoryItem>`. The write path (`store`/`step`)
/// has exclusive `&mut self`; no locks needed since `System::step()` is the
/// sole writer and runs single-threaded on the hot path.
pub struct InMemoryBackend {
    items: Vec<MemoryItem>,
    next_id: MemoryItemId,
    pressure: PressureMode,
    config: AncsConfig,
    /// Patterns of items that just transitioned to Contested.
    /// Drained by `take_reconsolidate_patterns()` in `System::step()`.
    pending_reconsolidate: Vec<Vec<(MorphonId, f64)>>,
}

impl InMemoryBackend {
    pub fn new(config: AncsConfig) -> Self {
        Self {
            items: Vec::new(),
            next_id: 1,
            pressure: PressureMode::Normal,
            config,
            pending_reconsolidate: Vec::new(),
        }
    }

    fn next_id(&mut self) -> MemoryItemId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Record a replay event: bump replay_count and update SM-2 stability.
    pub fn record_replay(&mut self, id: MemoryItemId) {
        if let Some(item) = self.items.iter_mut().find(|i| i.id == id) {
            item.replay_count += 1;
            let ln_replay = (item.replay_count as f64).ln().max(1.0);
            item.stability *= 1.0 + 0.1 * ln_replay;
            item.consolidation = (item.consolidation + 0.05).min(1.0);
        }
    }

    /// Items in epistemic-valid states (Supported + Hypothesis), sorted by importance desc.
    pub fn top_items(&self, n: usize, tier: Option<MemoryTier>) -> Vec<&MemoryItem> {
        let mut candidates: Vec<&MemoryItem> = self
            .items
            .iter()
            .filter(|i| tier.map(|t| i.tier == t).unwrap_or(true))
            .filter(|i| {
                matches!(
                    i.epistemic,
                    MemoryEpistemicState::Supported | MemoryEpistemicState::Hypothesis
                )
            })
            .collect();
        candidates.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(n);
        candidates
    }

    /// Detect conflict between `new_item` and existing Supported items.
    ///
    /// A conflict is: ≥ `conflict_overlap_threshold` Jaccard overlap AND opposing reward sign.
    /// Conflicting existing items → Contested, pattern queued for synapse reconsolidation.
    fn detect_conflict(&mut self, new_item: &MemoryItem) {
        let new_ids = new_item.morphon_ids();
        let reward_sign = new_item.reward.signum();
        if reward_sign == 0.0 {
            return;
        }
        let threshold = self.config.conflict_overlap_threshold;
        let new_ts = new_item.timestamp;

        // Two-pass: collect contested patterns first, then push to pending.
        let mut newly_contested: Vec<Vec<(MorphonId, f64)>> = Vec::new();

        for existing in &mut self.items {
            if existing.epistemic != MemoryEpistemicState::Supported {
                continue;
            }
            if existing.reward.signum() == reward_sign {
                continue; // same direction → no conflict
            }
            let ex_ids = existing.morphon_ids();
            let intersection = new_ids.iter().filter(|id| ex_ids.contains(id)).count();
            let union = new_ids.len() + ex_ids.len() - intersection;
            if union > 0 && (intersection as f64 / union as f64) >= threshold {
                existing.epistemic = MemoryEpistemicState::Contested;
                existing.state_changed_at = new_ts;
                newly_contested.push(existing.pattern.clone());
            }
        }

        self.pending_reconsolidate.extend(newly_contested);
    }

    /// Blast-radius cascade: when an item goes Stale, mark temporal neighbors Suspect.
    ///
    /// Walks items encoded within `blast_window` steps of the stale item's timestamp.
    /// Only Supported and Hypothesis items are affected (not already-degraded states).
    fn blast_radius_cascade(&mut self, stale_timestamp: u64, current_step: u64) {
        let window = self.config.blast_window;
        let low = stale_timestamp.saturating_sub(window);
        let high = stale_timestamp + window;

        for item in &mut self.items {
            if item.timestamp < low || item.timestamp > high {
                continue;
            }
            match item.epistemic {
                MemoryEpistemicState::Supported | MemoryEpistemicState::Hypothesis => {
                    item.epistemic = MemoryEpistemicState::Suspect;
                    item.state_changed_at = current_step;
                }
                _ => {}
            }
        }
    }

    /// Tier demotion for items whose importance has fallen below tier thresholds.
    fn demote_tiers(&mut self) {
        for item in &mut self.items {
            match item.tier {
                MemoryTier::Verbatim if item.importance < 0.5 && item.consolidation <= 0.8 => {
                    // Demote to Semantic: allow lossy representation
                    item.tier = MemoryTier::Semantic;
                    item.content_hash = None;
                }
                MemoryTier::Semantic if item.importance < 0.3 => {
                    item.tier = MemoryTier::Procedural;
                }
                MemoryTier::Procedural if item.importance < 0.1 => {
                    item.epistemic = MemoryEpistemicState::Outdated;
                }
                _ => {}
            }
        }
    }

    /// Evict items according to the current pressure mode.
    fn evict_under_pressure(&mut self) {
        match self.pressure {
            PressureMode::Normal => {
                // Only evict Outdated items with very low importance
                self.items.retain(|i| {
                    !(i.epistemic == MemoryEpistemicState::Outdated && i.importance < 0.05)
                });
            }
            PressureMode::Pressure => {
                // Keep Verbatim, pinned, or important items
                self.items.retain(|i| {
                    i.tier == MemoryTier::Verbatim || i.importance >= 0.25 || i.consolidation > 0.8
                });
            }
            PressureMode::Emergency => {
                // Only Verbatim and pinned survive
                self.items.retain(|i| {
                    i.tier == MemoryTier::Verbatim || i.consolidation > 0.8
                });
            }
            PressureMode::Critical => {
                // Safe mode: Verbatim + Supported with high importance only
                self.items.retain(|i| {
                    i.tier == MemoryTier::Verbatim
                        || (i.epistemic == MemoryEpistemicState::Supported && i.importance > 0.6)
                });
            }
        }

        // Hard capacity limit: evict lowest-importance items (protecting Verbatim)
        if self.items.len() > self.config.capacity {
            // Sort ascending by importance
            self.items.sort_by(|a, b| {
                a.importance
                    .partial_cmp(&b.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let overshoot = self.items.len() - self.config.capacity;
            // Skip over Verbatim items at the start (they sort near the middle/top)
            let mut evicted = 0;
            self.items.retain(|i| {
                if evicted < overshoot && i.tier != MemoryTier::Verbatim {
                    evicted += 1;
                    false
                } else {
                    true
                }
            });
        }
    }
}

impl MemoryBackend for InMemoryBackend {
    fn store(&mut self, mut item: MemoryItem) {
        item.id = self.next_id();
        self.detect_conflict(&item);
        self.items.push(item);
    }

    fn retrieve(&self, query: &RetrievalQuery, top_k: usize) -> Vec<(MemoryItemId, f64)> {
        const RRF_K: f64 = 60.0;

        // Filter: only epistemic ally valid items participate in standard retrieval
        let mut ranked: Vec<(MemoryItemId, f64)> = self
            .items
            .iter()
            .filter(|i| {
                matches!(
                    i.epistemic,
                    MemoryEpistemicState::Supported | MemoryEpistemicState::Hypothesis
                )
            })
            .map(|item| {
                let item_ids = item.morphon_ids();
                let denom = query.pattern.len().max(item_ids.len()).max(1);
                let intersection = query
                    .pattern
                    .iter()
                    .filter(|id| item_ids.contains(id))
                    .count();
                let score = (intersection as f64 / denom as f64) * item.importance;
                (item.id, score)
            })
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply RRF weighting over the importance-ranked list
        ranked
            .iter()
            .enumerate()
            .map(|(rank, (id, _))| (*id, 1.0 / (RRF_K + rank as f64 + 1.0)))
            .take(top_k)
            .collect()
    }

    fn step(&mut self, _dt: f64, current_step: u64, energy_usage: f64) {
        // Update pressure mode from system energy state
        self.pressure = PressureMode::from_usage(energy_usage);

        // Collect timestamps of items that transition to Stale this tick
        // (for blast-radius cascade after the main loop closes the borrow)
        let mut newly_stale: Vec<u64> = Vec::new();

        for item in &mut self.items {
            // Recalculate AXION importance (centrality approximated at 0.5)
            item.importance = compute_importance(item, current_step, 0.5);

            let age_in_state = current_step.saturating_sub(item.state_changed_at);

            match item.epistemic {
                MemoryEpistemicState::Hypothesis => {
                    if item.consolidation > 0.5 {
                        // Consolidated → Supported
                        item.epistemic = MemoryEpistemicState::Supported;
                        item.state_changed_at = current_step;
                    } else if age_in_state > self.config.hypothesis_timeout
                        && item.importance < self.config.stale_threshold
                    {
                        // Old low-importance hypothesis → Outdated
                        item.epistemic = MemoryEpistemicState::Outdated;
                        item.state_changed_at = current_step;
                    }
                }
                MemoryEpistemicState::Supported => {
                    if item.importance < self.config.stale_threshold {
                        // Importance decayed → Stale (proxy for source pattern drift)
                        item.epistemic = MemoryEpistemicState::Stale;
                        item.state_changed_at = current_step;
                        newly_stale.push(item.timestamp);
                    }
                }
                MemoryEpistemicState::Stale => {
                    if age_in_state > self.config.contested_timeout {
                        item.epistemic = MemoryEpistemicState::Outdated;
                        item.state_changed_at = current_step;
                    }
                }
                MemoryEpistemicState::Suspect => {
                    if item.importance > 0.5 {
                        // Importance recovered → back to Supported
                        item.epistemic = MemoryEpistemicState::Supported;
                        item.state_changed_at = current_step;
                    } else if age_in_state > self.config.contested_timeout
                        && item.importance < self.config.stale_threshold
                    {
                        item.epistemic = MemoryEpistemicState::Outdated;
                        item.state_changed_at = current_step;
                    }
                }
                MemoryEpistemicState::Contested => {
                    if age_in_state > self.config.contested_timeout {
                        item.epistemic = MemoryEpistemicState::Outdated;
                        item.state_changed_at = current_step;
                    }
                }
                MemoryEpistemicState::Outdated => {}
            }
        }

        // Blast-radius cascade for newly-stale items
        for ts in newly_stale {
            self.blast_radius_cascade(ts, current_step);
        }

        // Tier demotion
        self.demote_tiers();

        // Pressure-based eviction
        self.evict_under_pressure();
    }

    fn record_access(&mut self, id: MemoryItemId) {
        if let Some(item) = self.items.iter_mut().find(|i| i.id == id) {
            item.access_count += 1;
        }
    }

    fn importance(&self, id: MemoryItemId) -> f64 {
        self.items
            .iter()
            .find(|i| i.id == id)
            .map(|i| i.importance)
            .unwrap_or(0.0)
    }

    fn epistemic_state(&self, id: MemoryItemId) -> Option<MemoryEpistemicState> {
        self.items.iter().find(|i| i.id == id).map(|i| i.epistemic)
    }

    fn items_with_state(&self, state: MemoryEpistemicState) -> Vec<MemoryItemId> {
        self.items
            .iter()
            .filter(|i| i.epistemic == state)
            .map(|i| i.id)
            .collect()
    }

    fn mark_stale(&mut self, id: MemoryItemId, current_step: u64) {
        let ts = self
            .items
            .iter()
            .find(|i| i.id == id)
            .map(|i| i.timestamp);
        if let Some(ts) = ts {
            if let Some(item) = self.items.iter_mut().find(|i| i.id == id) {
                item.epistemic = MemoryEpistemicState::Stale;
                item.state_changed_at = current_step;
            }
            self.blast_radius_cascade(ts, current_step);
        }
    }

    fn pressure_mode(&self) -> PressureMode {
        self.pressure
    }

    fn take_reconsolidate_patterns(&mut self) -> Vec<Vec<(MorphonId, f64)>> {
        std::mem::take(&mut self.pending_reconsolidate)
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn last_stored_id(&self) -> Option<MemoryItemId> {
        self.items.last().map(|i| i.id)
    }

    fn item_pattern(&self, id: MemoryItemId) -> Option<Vec<MorphonId>> {
        self.items.iter().find(|i| i.id == id).map(|i| i.morphon_ids())
    }
}

// ─── SystemHeartbeat ─────────────────────────────────────────────────────────

/// Cache-line-sized atomic snapshot of global system state.
///
/// Updated at the start of each step. Any module can read it with a single
/// load (Relaxed ordering — stale by at most one tick, acceptable for
/// neuromodulatory signals).
///
/// Uses `AtomicU32` with `f32::to_bits`/`from_bits` because `AtomicF32`
/// does not exist on stable Rust.
#[repr(C, align(64))]
pub struct SystemHeartbeat {
    // Neuromodulation snapshot (already computed in System each step)
    pub global_arousal: AtomicU32,
    pub global_novelty: AtomicU32,
    pub global_reward: AtomicU32,
    pub global_homeostasis: AtomicU32,
    pub plasticity_mult: AtomicU32,

    // ANCS-Core state
    pub energy_pressure: AtomicU32,
    pub ancs_item_count: AtomicU32,
    pub contested_count: AtomicU32,

    // SOMNUS (deferred — Phase 5)
    pub sleep_phase: AtomicBool,
}

impl Default for SystemHeartbeat {
    fn default() -> Self {
        Self {
            global_arousal: AtomicU32::new(0),
            global_novelty: AtomicU32::new(0),
            global_reward: AtomicU32::new(0),
            global_homeostasis: AtomicU32::new(0),
            plasticity_mult: AtomicU32::new(f32::to_bits(1.0)),
            energy_pressure: AtomicU32::new(0),
            ancs_item_count: AtomicU32::new(0),
            contested_count: AtomicU32::new(0),
            sleep_phase: AtomicBool::new(false),
        }
    }
}

impl SystemHeartbeat {
    /// Write an f64 value into an AtomicU32 via f32 bits (precision loss is fine for signals).
    #[inline]
    pub fn write_f32(slot: &AtomicU32, val: f64) {
        slot.store(f32::to_bits(val as f32), Ordering::Relaxed);
    }

    /// Read an f64 value from an AtomicU32 stored as f32 bits.
    #[inline]
    pub fn read_f32(slot: &AtomicU32) -> f64 {
        f32::from_bits(slot.load(Ordering::Relaxed)) as f64
    }

    /// Write a complete system snapshot in one call.
    pub fn update(
        &self,
        arousal: f64,
        novelty: f64,
        reward: f64,
        homeostasis: f64,
        plasticity: f64,
        energy_pressure: f64,
        ancs_count: usize,
        contested: usize,
    ) {
        Self::write_f32(&self.global_arousal, arousal);
        Self::write_f32(&self.global_novelty, novelty);
        Self::write_f32(&self.global_reward, reward);
        Self::write_f32(&self.global_homeostasis, homeostasis);
        Self::write_f32(&self.plasticity_mult, plasticity);
        Self::write_f32(&self.energy_pressure, energy_pressure);
        self.ancs_item_count
            .store(ancs_count as u32, Ordering::Relaxed);
        self.contested_count
            .store(contested as u32, Ordering::Relaxed);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(novelty: f64, reward: f64, timestamp: u64, pattern: Vec<(MorphonId, f64)>) -> MemoryItem {
        MemoryItem::new(MemoryTier::Semantic, pattern, reward, novelty, timestamp)
    }

    // === VBC-lite ===

    #[test]
    fn classify_verbatim_on_high_salience() {
        assert_eq!(classify_tier(0.8, 0.6, 0.0), MemoryTier::Verbatim);
    }

    #[test]
    fn classify_structural_on_high_overlap() {
        // working_overlap dominates when novelty/reward don't hit Verbatim
        assert_eq!(classify_tier(0.2, 0.2, 0.7), MemoryTier::Structural);
    }

    #[test]
    fn classify_semantic_default() {
        assert_eq!(classify_tier(0.5, 0.2, 0.1), MemoryTier::Semantic);
    }

    #[test]
    fn classify_procedural_low_everything() {
        assert_eq!(classify_tier(0.1, 0.1, 0.1), MemoryTier::Procedural);
    }

    // === Pressure modes ===

    #[test]
    fn pressure_mode_thresholds() {
        assert_eq!(PressureMode::from_usage(0.5), PressureMode::Normal);
        assert_eq!(PressureMode::from_usage(0.75), PressureMode::Pressure);
        assert_eq!(PressureMode::from_usage(0.88), PressureMode::Emergency);
        assert_eq!(PressureMode::from_usage(0.97), PressureMode::Critical);
    }

    // === InMemoryBackend ===

    #[test]
    fn store_assigns_monotone_ids() {
        let mut backend = InMemoryBackend::new(AncsConfig::default());
        backend.store(make_item(0.5, 0.3, 0, vec![(1, 0.5)]));
        backend.store(make_item(0.5, 0.3, 10, vec![(2, 0.5)]));
        // IDs should be 1, 2
        assert_eq!(backend.items[0].id, 1);
        assert_eq!(backend.items[1].id, 2);
    }

    #[test]
    fn retrieve_filters_contested() {
        let mut backend = InMemoryBackend::new(AncsConfig::default());
        let mut item = make_item(0.5, 0.3, 0, vec![(1, 0.5), (2, 0.8)]);
        item.epistemic = MemoryEpistemicState::Contested;
        backend.store(item);

        let query = RetrievalQuery { pattern: vec![1, 2] };
        let results = backend.retrieve(&query, 5);
        assert!(results.is_empty(), "Contested items must not appear in retrieval");
    }

    #[test]
    fn retrieve_returns_supported_items() {
        let mut backend = InMemoryBackend::new(AncsConfig::default());
        let mut item = make_item(0.5, 0.3, 0, vec![(1, 0.5), (2, 0.8)]);
        item.epistemic = MemoryEpistemicState::Supported;
        backend.store(item);

        let query = RetrievalQuery { pattern: vec![1, 2] };
        let results = backend.retrieve(&query, 5);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn hypothesis_transitions_to_supported_on_consolidation() {
        let mut backend = InMemoryBackend::new(AncsConfig::default());
        let mut item = make_item(0.5, 0.3, 0, vec![(1, 0.5)]);
        item.consolidation = 0.6; // above threshold
        backend.store(item);

        backend.step(1.0, 10, 0.3);

        assert_eq!(
            backend.epistemic_state(1).unwrap(),
            MemoryEpistemicState::Supported
        );
    }

    #[test]
    fn conflict_detection_marks_existing_contested() {
        let mut backend = InMemoryBackend::new(AncsConfig::default());

        // Supported item with positive reward and overlapping pattern
        let mut existing = make_item(0.5, 0.8, 0, vec![(1, 0.8), (2, 0.6), (3, 0.4)]);
        existing.epistemic = MemoryEpistemicState::Supported;
        backend.store(existing);

        // New item with opposing reward, same pattern
        let conflict = make_item(0.5, -0.8, 100, vec![(1, 0.7), (2, 0.5), (3, 0.3)]);
        backend.store(conflict);

        assert_eq!(
            backend.epistemic_state(1).unwrap(),
            MemoryEpistemicState::Contested,
            "existing item should become Contested"
        );
        let pending = backend.take_reconsolidate_patterns();
        assert_eq!(pending.len(), 1, "one pattern queued for reconsolidation");
    }

    #[test]
    fn blast_radius_cascade_marks_nearby_suspect() {
        let mut backend = InMemoryBackend::new(AncsConfig::default());

        // Two items close in time
        let mut a = make_item(0.5, 0.3, 100, vec![(1, 0.5)]);
        a.epistemic = MemoryEpistemicState::Supported;
        backend.store(a);

        let mut b = make_item(0.5, 0.3, 130, vec![(2, 0.5)]);
        b.epistemic = MemoryEpistemicState::Supported;
        backend.store(b);

        // Trigger cascade from timestamp 100 (window=50: covers 50–150)
        backend.blast_radius_cascade(100, 200);

        // Both should be Suspect
        assert_eq!(backend.epistemic_state(1).unwrap(), MemoryEpistemicState::Suspect);
        assert_eq!(backend.epistemic_state(2).unwrap(), MemoryEpistemicState::Suspect);
    }

    #[test]
    fn importance_decreases_with_age() {
        let old = make_item(0.5, 0.3, 100, vec![(1, 0.5)]);
        let young = make_item(0.5, 0.3, 900, vec![(1, 0.5)]);

        let imp_old = compute_importance(&old, 1000, 0.5);
        let imp_young = compute_importance(&young, 1000, 0.5);

        assert!(imp_young > imp_old, "younger items have higher retrievability");
    }

    #[test]
    fn verbatim_item_has_content_hash() {
        let item = MemoryItem::new(
            MemoryTier::Verbatim,
            vec![(10, 0.9)],
            0.9,
            0.8,
            0,
        );
        assert!(item.content_hash.is_some());
        assert!(item.verify_integrity());
    }

    #[test]
    fn heartbeat_write_read_roundtrip() {
        let hb = SystemHeartbeat::default();
        SystemHeartbeat::write_f32(&hb.global_arousal, 0.75);
        let val = SystemHeartbeat::read_f32(&hb.global_arousal);
        assert!((val - 0.75).abs() < 0.001, "f32 precision should be sufficient for signals");
    }
}
