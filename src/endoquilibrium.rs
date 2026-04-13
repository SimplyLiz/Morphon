//! Endoquilibrium — Predictive Neuroendocrine Regulation Engine.
//!
//! Maintains network health by sensing vital signs, predicting healthy state
//! via dual-timescale EMAs, and adjusting neuromodulatory channels through
//! proportional control. Biological analogy: the endocrine system (allostasis),
//! not the nervous system (homeostasis).
//!
//! Runs on the medium path tick. Never modifies weights or topology directly —
//! it modulates the environment in which the Builder operates.

use crate::diagnostics::Diagnostics;
use crate::morphon::Morphon;
use crate::topology::Topology;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ─── Configuration ───────────────────────────────────────────────────

/// Endoquilibrium configuration. All fields have `#[serde(default)]` via
/// the struct-level Default impl, so existing configs deserialize cleanly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndoConfig {
    /// Master switch. When false, all channels stay at neutral defaults.
    pub enabled: bool,
    /// Fast EMA time constant (ticks). Tracks acute changes.
    pub fast_tau: f32,
    /// Slow EMA time constant (ticks). Tracks developmental trajectory.
    pub slow_tau: f32,
    /// Smoothing factor for channel adjustments (prevents oscillation).
    pub smoothing_alpha: f32,

    // Rule 1: Firing rate regulation coefficients
    pub fr_deficit_threshold_k: f32,
    pub fr_deficit_arousal_k: f32,
    pub fr_deficit_novelty_k: f32,
    pub fr_excess_threshold_k: f32,
    pub fr_excess_homeo_k: f32,

    // Rule 2: Eligibility density coefficients
    pub elig_low_novelty_k: f32,
    pub elig_low_plast_k: f32,
    pub elig_high_homeo_k: f32,
    pub elig_high_plast_k: f32,

    // Rule 3: Weight entropy coefficients
    pub entropy_low_novelty_k: f32,
    pub entropy_low_plast_k: f32,
    pub entropy_high_plast_k: f32,
    pub entropy_high_homeo_k: f32,

    // Rule 5: Tag-capture coefficients
    pub tag_capture_reward_boost: f32,
    pub tag_capture_stale_ticks: u64,

    /// Minimum plasticity_mult Endo can output. Prevents Mature stage from throttling
    /// below this floor. Default 0.0 (no floor). Set to ~1.5 for supervised learning
    /// tasks where premature Mature detection must not suppress plasticity.
    pub plasticity_floor: f32,
    /// Ablation flag: restore pre-v4.6.0 behaviour where Rule 7 energy_emergency/critical
    /// suppresses novelty_gain (ng *= 0.2 / ng = 0.0). Default false (fixed behaviour).
    /// Set true only for ablation studies isolating the ng-collapse fix contribution.
    #[serde(default)]
    pub suppress_novelty_on_energy: bool,
    /// Minimum total_updates before the Mature stage can be entered via the time-gate
    /// fallback (`total_updates >= mature_min_updates`). Default 2000 (backward compat).
    /// Raise for supervised tasks (e.g. MNIST) where early reward stability is misleading:
    /// the training-window accuracy saturates before test accuracy converges, causing Endo
    /// to declare Mature prematurely and lock in conservative parameters (cg=0.5, pm=0.5).
    /// Set to e.g. 8000 (≈ first 2700 supervised images at 3 steps/image) to defer Mature
    /// until the system has seen enough supervised signal to genuinely converge.
    #[serde(default = "default_mature_min_updates")]
    pub mature_min_updates: usize,
}

fn default_mature_min_updates() -> usize { 2000 }

impl Default for EndoConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fast_tau: 50.0,
            slow_tau: 500.0,
            smoothing_alpha: 0.1,
            // Rule 1 (from spec §5.2)
            fr_deficit_threshold_k: 0.5,
            fr_deficit_arousal_k: 0.3,
            fr_deficit_novelty_k: 0.2,
            fr_excess_threshold_k: 0.8,
            fr_excess_homeo_k: 0.5,
            // Rule 2
            elig_low_novelty_k: 0.2,
            elig_low_plast_k: 1.2,
            elig_high_homeo_k: 0.3,
            elig_high_plast_k: 0.8,
            // Rule 3
            entropy_low_novelty_k: 0.4,
            entropy_low_plast_k: 1.5,
            entropy_high_plast_k: 0.5,
            entropy_high_homeo_k: 0.4,
            // Rule 5
            tag_capture_reward_boost: 1.5,
            tag_capture_stale_ticks: 500,
            plasticity_floor: 0.0,
            suppress_novelty_on_energy: false,
            mature_min_updates: default_mature_min_updates(), // 2000
        }
    }
}

// ─── Vital Signs ─────────────────────────────────────────────────────

/// Network health metrics sensed every medium-path tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitalSigns {
    pub timestamp: u64,
    // Per-type firing rates (fraction, 0.0–1.0)
    pub fr_sensory: f32,
    pub fr_associative: f32,
    pub fr_motor: f32,
    pub fr_modulatory: f32,
    // Learning pipeline
    pub eligibility_density: f32,
    pub weight_entropy: f32,
    pub tag_count: u32,
    pub capture_count: u32,
    // Structural
    pub cell_type_fractions: [f32; 7],
    pub total_morphons: u32,
    pub total_synapses: u32,
    // Metabolic
    pub energy_utilization: f32,
    // Task performance
    pub prediction_error_mean: f32,
    /// Rolling reward/performance metric (task-agnostic, set by caller).
    /// For CartPole: episode steps. For MNIST: classification accuracy.
    pub reward_avg: f32,
    // Competition monitoring (Section 8, Phase A)
    /// Mean number of cluster members that fired after competition resolved.
    /// Only meaningful in LocalInhibition mode; 0 when no clusters exist.
    pub winners_per_cluster: f32,
    // Morphogenesis rates (Section 8, Phase B)
    /// Division events in the last glacial tick.
    pub division_rate: f32,
    /// Pruning (synapse) events in the last slow tick.
    pub pruning_rate: f32,
    /// Mean frustration level across all morphons.
    pub frustration_mean: f32,
}

impl Default for VitalSigns {
    fn default() -> Self {
        Self {
            timestamp: 0,
            fr_sensory: 0.1,
            fr_associative: 0.1,
            fr_motor: 0.1,
            fr_modulatory: 0.1,
            eligibility_density: 0.3,
            weight_entropy: 3.0,
            tag_count: 0,
            capture_count: 0,
            cell_type_fractions: [0.0; 7],
            total_morphons: 0,
            total_synapses: 0,
            energy_utilization: 0.5,
            prediction_error_mean: 0.1,
            reward_avg: 0.0,
            winners_per_cluster: 0.0,
            division_rate: 0.0,
            pruning_rate: 0.0,
            frustration_mean: 0.0,
        }
    }
}

/// Sense current vital signs from the network. Free function to avoid
/// borrow conflicts with `&mut self.endo` in `System::step()`.
pub fn sense_vitals(
    morphons: &HashMap<MorphonId, Morphon>,
    topology: &Topology,
    diag: &Diagnostics,
    step: u64,
    reward_avg: f64,
    clusters: &HashMap<crate::types::ClusterId, crate::morphogenesis::Cluster>,
) -> VitalSigns {
    let n = morphons.len().max(1) as f32;

    // Firing rates by type (reuse Diagnostics data)
    let fr = |ct: CellType| -> f32 {
        diag.firing_by_type
            .get(&ct)
            .map(|&(fired, total)| {
                if total > 0 {
                    fired as f32 / total as f32
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0)
    };

    // Cell type fractions
    let mut type_counts = [0u32; 7];
    let mut pe_sum = 0.0_f32;
    let mut energy_sum = 0.0_f32;
    for m in morphons.values() {
        let idx = cell_type_index(m.cell_type);
        type_counts[idx] += 1;
        pe_sum += m.prediction_error as f32;
        energy_sum += m.energy as f32;
    }
    let mut fractions = [0.0_f32; 7];
    for i in 0..7 {
        fractions[i] = type_counts[i] as f32 / n;
    }

    // Weight entropy: bin weights into 20 buckets, compute Shannon entropy
    let weight_entropy = compute_weight_entropy(topology);

    // Eligibility density
    let total_syn = diag.total_synapses.max(1) as f32;
    let eligibility_density = diag.eligibility_nonzero_count as f32 / total_syn;

    // Energy utilization: 1.0 = fully depleted, 0.0 = fully charged
    let avg_energy = energy_sum / n;
    let energy_utilization = 1.0 - avg_energy.clamp(0.0, 1.0);

    // winners_per_cluster: mean fired count per cluster (LocalInhibition observability)
    let winners_per_cluster = if clusters.is_empty() {
        0.0_f32
    } else {
        let total_winners: usize = clusters
            .values()
            .map(|c| {
                c.members
                    .iter()
                    .filter(|&&mid| morphons.get(&mid).map_or(false, |m| m.fired))
                    .count()
            })
            .sum();
        total_winners as f32 / clusters.len() as f32
    };

    VitalSigns {
        timestamp: step,
        fr_sensory: fr(CellType::Sensory),
        fr_associative: fr(CellType::Associative),
        fr_motor: fr(CellType::Motor),
        fr_modulatory: fr(CellType::Modulatory),
        eligibility_density,
        weight_entropy,
        tag_count: diag.active_tags as u32,
        capture_count: (diag.captures_this_step + diag.episode_captures_pending) as u32,
        cell_type_fractions: fractions,
        total_morphons: morphons.len() as u32,
        total_synapses: diag.total_synapses as u32,
        energy_utilization,
        prediction_error_mean: pe_sum / n,
        reward_avg: reward_avg as f32,
        winners_per_cluster,
        division_rate: diag.division_events_recent as f32,
        pruning_rate: diag.pruning_events_recent as f32,
        frustration_mean: diag.avg_frustration as f32,
    }
}

/// Map CellType to array index for `cell_type_fractions`.
fn cell_type_index(ct: CellType) -> usize {
    match ct {
        CellType::Stem => 0,
        CellType::Sensory => 1,
        CellType::Associative => 2,
        CellType::Motor => 3,
        CellType::Modulatory => 4,
        CellType::Fused => 5,
        CellType::InhibitoryInterneuron => 6,
    }
}

/// Compute Shannon entropy of the weight distribution.
/// Bins weights into NUM_BINS buckets across [-max, max].
fn compute_weight_entropy(topology: &Topology) -> f32 {
    const NUM_BINS: usize = 20;
    let mut bins = [0u32; NUM_BINS];
    let mut count = 0u32;
    let mut abs_max = 0.0_f64;

    // First pass: find range
    for ei in topology.graph.edge_indices() {
        if let Some(syn) = topology.graph.edge_weight(ei) {
            let w = if syn.weight.is_finite() {
                syn.weight
            } else {
                0.0
            };
            abs_max = abs_max.max(w.abs());
            count += 1;
        }
    }

    if count == 0 || abs_max < 1e-10 {
        return 0.0;
    }

    let range = abs_max * 2.0;

    // Second pass: bin
    for ei in topology.graph.edge_indices() {
        if let Some(syn) = topology.graph.edge_weight(ei) {
            let w = if syn.weight.is_finite() {
                syn.weight
            } else {
                0.0
            };
            let normalized = (w + abs_max) / range; // [0, 1]
            let bin = ((normalized * NUM_BINS as f64) as usize).min(NUM_BINS - 1);
            bins[bin] += 1;
        }
    }

    // Shannon entropy
    let n = count as f32;
    let mut entropy = 0.0_f32;
    for &b in &bins {
        if b > 0 {
            let p = b as f32 / n;
            entropy -= p * p.ln();
        }
    }
    entropy
}

// ─── Channel State (the 10 levers) ──────────────────────────────────

/// The actuators Endoquilibrium controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelState {
    pub reward_gain: f32,
    pub novelty_gain: f32,
    pub arousal_gain: f32,
    pub homeostasis_gain: f32,
    pub threshold_bias: f32,
    pub plasticity_mult: f32,
    /// Consolidation gain: how aggressively tagged synapses get captured.
    pub consolidation_gain: f32,
    // === Endo V2 levers ===
    /// Scales activity-dependent winner threshold boost (Diehl & Cook).
    #[serde(default = "default_channel_one")]
    pub winner_adaptation_mult: f32,
    /// Scales capture_threshold for tag-and-capture consolidation.
    #[serde(default = "default_channel_one")]
    pub capture_threshold_mult: f32,
    /// Scales rollback_pe_threshold.
    #[serde(default = "default_channel_one")]
    pub rollback_pe_threshold_mult: f32,
    /// Scales tau_eligibility (eligibility trace time constant).
    /// >1 = wider credit assignment window (slow captures), <1 = tighter (fast captures).
    /// Driven by ticks_since_last_capture — the single lever that most directly
    /// enables temporal processing.
    #[serde(default = "default_channel_one")]
    pub tau_eligibility_mult: f32,
    /// Scales slow trace leak rate. Plumbing for temporal processing —
    /// no regulation rule yet, just the lever slot.
    #[serde(default = "default_channel_one")]
    pub slow_trace_leak: f32,
    // === Endo V2 Phase B levers (structural plasticity) ===
    /// Scales morphogenesis division threshold. <1 = easier division (Proliferating),
    /// >1 = harder division (Mature). Multiplied into params.division_threshold.
    #[serde(default = "default_channel_one")]
    pub division_threshold_mult: f32,
    /// Scales pruning weight_min threshold. <1 = harder to prune (Proliferating),
    /// >1 = easier to prune (Consolidating). Multiplied into params.weight_min.
    #[serde(default = "default_channel_one")]
    pub pruning_threshold_mult: f32,
    /// Scales frustration sensitivity thresholds. <1 = more sensitive (Mature),
    /// >1 = more tolerant (Proliferating). Multiplied into stagnation_threshold.
    #[serde(default = "default_channel_one")]
    pub frustration_sensitivity_mult: f32,
    /// Scales migration rate in hyperbolic space. >1 = more migration (Proliferating),
    /// <1 = less migration (Mature/Consolidating). Multiplied into params.migration_rate.
    #[serde(default = "default_channel_one")]
    pub migration_rate_mult: f32,
    /// Scales synaptogenesis activity threshold. <1 = easier new connections (Proliferating),
    /// >1 = harder (Mature). Multiplied into the activity threshold for synaptogenesis.
    #[serde(default = "default_channel_one")]
    pub synaptogenesis_threshold_mult: f32,
}

fn default_channel_one() -> f32 {
    1.0
}

impl Default for ChannelState {
    fn default() -> Self {
        Self {
            reward_gain: 1.0,
            novelty_gain: 1.0,
            arousal_gain: 1.0,
            homeostasis_gain: 1.0,
            threshold_bias: 0.0,
            plasticity_mult: 1.0,
            consolidation_gain: 1.0,
            winner_adaptation_mult: 1.0,
            capture_threshold_mult: 1.0,
            rollback_pe_threshold_mult: 1.0,
            tau_eligibility_mult: 1.0,
            slow_trace_leak: 1.0,
            division_threshold_mult: 1.0,
            pruning_threshold_mult: 1.0,
            frustration_sensitivity_mult: 1.0,
            migration_rate_mult: 1.0,
            synaptogenesis_threshold_mult: 1.0,
        }
    }
}

impl ChannelState {
    /// Clamp all channels to their valid ranges.
    fn clamp(&mut self) {
        self.reward_gain = self.reward_gain.clamp(0.1, 3.0);
        self.novelty_gain = self.novelty_gain.clamp(0.0, 2.0);
        self.arousal_gain = self.arousal_gain.clamp(0.1, 2.0);
        self.homeostasis_gain = self.homeostasis_gain.clamp(0.3, 2.0);
        self.threshold_bias = self.threshold_bias.clamp(-0.3, 0.3);
        self.plasticity_mult = self.plasticity_mult.clamp(0.1, 5.0);
        self.consolidation_gain = self.consolidation_gain.clamp(0.2, 3.0);
        self.winner_adaptation_mult = self.winner_adaptation_mult.clamp(0.3, 2.5);
        self.capture_threshold_mult = self.capture_threshold_mult.clamp(0.5, 1.5);
        self.rollback_pe_threshold_mult = self.rollback_pe_threshold_mult.clamp(0.25, 2.0);
        self.tau_eligibility_mult = self.tau_eligibility_mult.clamp(0.5, 2.0);
        self.slow_trace_leak = self.slow_trace_leak.clamp(0.1, 3.0);
        self.division_threshold_mult = self.division_threshold_mult.clamp(0.5, 2.0);
        self.pruning_threshold_mult = self.pruning_threshold_mult.clamp(0.5, 2.0);
        self.frustration_sensitivity_mult = self.frustration_sensitivity_mult.clamp(0.5, 2.0);
        self.migration_rate_mult = self.migration_rate_mult.clamp(0.3, 2.0);
        self.synaptogenesis_threshold_mult = self.synaptogenesis_threshold_mult.clamp(0.5, 2.0);
    }
}

// ─── Developmental Stage ─────────────────────────────────────────────

/// Detected developmental stage — drives setpoint adaptation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevelopmentalStage {
    Proliferating,
    Differentiating,
    Consolidating,
    Mature,
    Stressed,
}

impl Default for DevelopmentalStage {
    fn default() -> Self {
        Self::Proliferating
    }
}

/// Stage-dependent setpoints for regulation rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DevelopmentalSetpoints {
    fr_assoc_min: f32,
    fr_assoc_max: f32,
    elig_min: f32,
    elig_max: f32,
    entropy_min: f32,
    entropy_max: f32,
    type_targets: [f32; 7], // S, A, M, Mod, Stem, Fused, InhibitoryInterneuron
}

impl Default for DevelopmentalSetpoints {
    fn default() -> Self {
        Self::for_stage(DevelopmentalStage::Proliferating)
    }
}

impl DevelopmentalSetpoints {
    fn for_stage(stage: DevelopmentalStage) -> Self {
        match stage {
            DevelopmentalStage::Proliferating => Self {
                fr_assoc_min: 0.12,
                fr_assoc_max: 0.18,
                elig_min: 0.40,
                elig_max: 0.70,
                entropy_min: 3.0,
                entropy_max: 4.5,
                type_targets: [0.15, 0.20, 0.35, 0.10, 0.15, 0.05, 0.0],
            },
            DevelopmentalStage::Differentiating => Self {
                fr_assoc_min: 0.10,
                fr_assoc_max: 0.15,
                elig_min: 0.30,
                elig_max: 0.60,
                entropy_min: 2.5,
                entropy_max: 4.0,
                type_targets: [0.10, 0.20, 0.40, 0.10, 0.15, 0.05, 0.0],
            },
            DevelopmentalStage::Consolidating | DevelopmentalStage::Mature => Self {
                fr_assoc_min: 0.08,
                fr_assoc_max: 0.12,
                elig_min: 0.20,
                elig_max: 0.40,
                entropy_min: 2.0,
                entropy_max: 3.5,
                type_targets: [0.05, 0.20, 0.45, 0.10, 0.15, 0.05, 0.0],
            },
            DevelopmentalStage::Stressed => Self {
                fr_assoc_min: 0.08,
                fr_assoc_max: 0.15,
                elig_min: 0.15,
                elig_max: 0.50,
                entropy_min: 2.0,
                entropy_max: 4.0,
                type_targets: [0.10, 0.20, 0.40, 0.10, 0.15, 0.05, 0.0],
            },
        }
    }
}

// ─── Allostasis Predictor ────────────────────────────────────────────

/// Dual-timescale EMA predictor + developmental stage detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AllostasisPredictor {
    fast_emas: VitalSigns,
    slow_emas: VitalSigns,
    pe_history: VecDeque<f32>,
    morphon_count_history: VecDeque<u32>,
    reward_history: VecDeque<f32>,
    stage: DevelopmentalStage,
    /// Cumulative tick count — used for Mature gate instead of reward_history.len()
    /// because the ring buffer is capped at 500 but Mature requires 2000 ticks.
    #[serde(default)]
    total_updates: usize,
    /// Ticks spent in the current stage. Guards against rapid oscillation (thrashing).
    #[serde(default)]
    stage_age_ticks: usize,
}

impl Default for AllostasisPredictor {
    fn default() -> Self {
        Self {
            fast_emas: VitalSigns::default(),
            slow_emas: VitalSigns::default(),
            pe_history: VecDeque::with_capacity(200),
            morphon_count_history: VecDeque::with_capacity(200),
            reward_history: VecDeque::with_capacity(500),
            stage: DevelopmentalStage::Proliferating,
            total_updates: 0,
            stage_age_ticks: 0,
        }
    }
}

impl AllostasisPredictor {
    fn update(&mut self, vitals: &VitalSigns, fast_alpha: f32, slow_alpha: f32, mature_min_updates: usize) {
        // Update fast EMAs
        ema_update_f32(
            &mut self.fast_emas.fr_sensory,
            vitals.fr_sensory,
            fast_alpha,
        );
        ema_update_f32(
            &mut self.fast_emas.fr_associative,
            vitals.fr_associative,
            fast_alpha,
        );
        ema_update_f32(&mut self.fast_emas.fr_motor, vitals.fr_motor, fast_alpha);
        ema_update_f32(
            &mut self.fast_emas.fr_modulatory,
            vitals.fr_modulatory,
            fast_alpha,
        );
        ema_update_f32(
            &mut self.fast_emas.eligibility_density,
            vitals.eligibility_density,
            fast_alpha,
        );
        ema_update_f32(
            &mut self.fast_emas.weight_entropy,
            vitals.weight_entropy,
            fast_alpha,
        );
        ema_update_f32(
            &mut self.fast_emas.energy_utilization,
            vitals.energy_utilization,
            fast_alpha,
        );
        ema_update_f32(
            &mut self.fast_emas.prediction_error_mean,
            vitals.prediction_error_mean,
            fast_alpha,
        );
        ema_update_f32(
            &mut self.fast_emas.reward_avg,
            vitals.reward_avg,
            fast_alpha,
        );

        // Update slow EMAs
        ema_update_f32(
            &mut self.slow_emas.fr_sensory,
            vitals.fr_sensory,
            slow_alpha,
        );
        ema_update_f32(
            &mut self.slow_emas.fr_associative,
            vitals.fr_associative,
            slow_alpha,
        );
        ema_update_f32(&mut self.slow_emas.fr_motor, vitals.fr_motor, slow_alpha);
        ema_update_f32(
            &mut self.slow_emas.fr_modulatory,
            vitals.fr_modulatory,
            slow_alpha,
        );
        ema_update_f32(
            &mut self.slow_emas.eligibility_density,
            vitals.eligibility_density,
            slow_alpha,
        );
        ema_update_f32(
            &mut self.slow_emas.weight_entropy,
            vitals.weight_entropy,
            slow_alpha,
        );
        ema_update_f32(
            &mut self.slow_emas.energy_utilization,
            vitals.energy_utilization,
            slow_alpha,
        );
        ema_update_f32(
            &mut self.slow_emas.prediction_error_mean,
            vitals.prediction_error_mean,
            slow_alpha,
        );
        ema_update_f32(
            &mut self.slow_emas.reward_avg,
            vitals.reward_avg,
            slow_alpha,
        );

        self.total_updates += 1;

        // History for trend detection
        self.pe_history.push_back(vitals.prediction_error_mean);
        if self.pe_history.len() > 200 {
            self.pe_history.pop_front();
        }
        self.morphon_count_history.push_back(vitals.total_morphons);
        if self.morphon_count_history.len() > 200 {
            self.morphon_count_history.pop_front();
        }
        // Only push reward when it meaningfully changed — between report_performance()
        // calls, reward_avg is stale and repeating the same value inflates the history
        // with artificial flatness, triggering premature Consolidating.
        let last_reward = self.reward_history.back().copied().unwrap_or(-1.0);
        if (vitals.reward_avg - last_reward).abs() > 1e-6 {
            self.reward_history.push_back(vitals.reward_avg);
            if self.reward_history.len() > 500 {
                self.reward_history.pop_front();
            }
        }

        // Detect developmental stage — log on transition with triggering vitals.
        // MIN_CONSOL_DWELL guards only the Consolidating→Stressed direction to prevent
        // rapid thrashing at performance plateaus where reward_trend bounces near
        // the Stressed threshold. Progressive transitions (Prolif→Diff→Consol→Mature)
        // are not guarded — they should respond immediately to genuine improvement.
        const MIN_CONSOL_DWELL: usize = 150;
        let new_stage = self.detect_stage(mature_min_updates);
        self.stage_age_ticks += 1;
        let thrash_guard = self.stage == DevelopmentalStage::Consolidating
            && new_stage == DevelopmentalStage::Stressed
            && self.stage_age_ticks < MIN_CONSOL_DWELL;
        if new_stage != self.stage && !thrash_guard {
            let reward_slow = self.slow_emas.reward_avg;
            let reward_fast = self.fast_emas.reward_avg;
            let reward_std = self.std_f32(&self.reward_history);
            let reward_cv = if reward_fast.abs() > 0.01 {
                reward_std / reward_fast.abs()
            } else {
                99.0
            };
            let reward_trend = self.trend_f32(&self.reward_history);
            let slow_abs = reward_slow.abs().max(0.01_f32);
            eprintln!("[ENDO] {:?} → {:?} | reward_slow={:.4} reward_cv={:.4} trend={:.6} (slow_abs={:.4}) hist={} age={}",
                self.stage, new_stage,
                reward_slow, reward_cv, reward_trend, slow_abs,
                self.reward_history.len(), self.stage_age_ticks);
            self.stage = new_stage;
            self.stage_age_ticks = 0;
        }
    }

    fn detect_stage(&self, mature_min_updates: usize) -> DevelopmentalStage {
        // Reward-based stage detection: relative to the system's own history.
        // Uses slow EMA as "what this system normally achieves" baseline.
        // No hardcoded absolute thresholds — works for CartPole, MNIST, any task.
        let reward_trend = self.trend_f32(&self.reward_history);
        let reward_fast = self.fast_emas.reward_avg;
        let reward_slow = self.slow_emas.reward_avg;
        let mc_trend = self.trend(&self.morphon_count_history);

        // Need enough history for meaningful trends
        if self.reward_history.len() < 20 {
            return DevelopmentalStage::Proliferating;
        }

        // Stressed: reward DROPPING significantly relative to own baseline.
        // Not "seeing new things" (PE rising) but "getting worse at the task."
        // Use max(0.01) not max(1.0) — for MNIST rewards in [0,1], max(1.0)
        // makes the threshold absolute and triggers Stressed on tiny dips.
        //
        // Hysteresis: enter Stressed only on a meaningful dip (-7% × slow_abs),
        // but once in Stressed stay there until genuine recovery (-2% × slow_abs).
        // Prevents the boundary oscillation where reward_trend sits exactly at
        // the threshold and small noise causes rapid Consolidating↔Stressed cycling.
        let slow_abs = reward_slow.abs().max(0.01);
        let stress_threshold = if self.stage == DevelopmentalStage::Stressed {
            -0.02 * slow_abs // exit only on clear recovery
        } else {
            -0.07 * slow_abs // enter only on meaningful dip
        };
        if reward_trend < stress_threshold && self.reward_history.len() >= 50 {
            return DevelopmentalStage::Stressed;
        }

        // Mature: reward stable (low variance) and near slow EMA — consistent performance.
        // Threshold is 0.3 (not 0.05): prevents premature Mature on classification tasks
        // where contrastive reward is dense+stable even at 20% accuracy. Endo must see
        // genuinely high reward before declaring the system mature.
        let reward_std = self.std_f32(&self.reward_history);
        let reward_cv = if reward_fast.abs() > 0.01 {
            reward_std / reward_fast.abs()
        } else {
            1.0
        };
        // RPE gate: use prediction error convergence as the principled maturity signal
        // instead of a fixed 2000-tick time gate (which was a calibrated patch).
        // pe_converged = PE slow EMA low + stable (low CV) + enough PE history accumulated.
        // PE slow is task-agnostic: when the network stops being surprised by inputs,
        // it has genuinely converged — independent of the reward schedule or task structure.
        // Minimum 500 ticks prevents noise-based early triggering during Proliferating;
        // the 2000-tick time gate is kept as fallback for reward-only tasks where PE
        // is not a reliable signal (e.g., CartPole where PE correlates with pole angle noise).
        // Recovery experiment proved: pm=2.16 (Differentiating) → 49%, pm=0.60 (Mature) → 27%.
        // So we only gate into Mature when we're sure learning has genuinely plateaued.
        let pe_slow = self.slow_emas.prediction_error_mean;
        let pe_std = self.std_f32(&self.pe_history);
        let pe_cv = if pe_slow.abs() > 0.01 { pe_std / pe_slow.abs() } else { 99.0 };
        let rpe_converged = pe_slow < 0.15
            && pe_cv < 0.3
            && self.pe_history.len() >= 100
            && self.total_updates >= 500;
        if reward_slow > 0.3
            && reward_cv < 0.15
            && reward_trend.abs() < 0.005 * slow_abs
            && (rpe_converged || self.total_updates >= mature_min_updates)
        {
            return DevelopmentalStage::Mature;
        }

        // Consolidating: reward near own ceiling (fast ≥ 95% of slow), stable.
        // 500 ticks — allows natural Differentiating↔Consolidating oscillation
        // which provides explore/exploit rhythm (pm=0.96↔1.77). The oscillation
        // IS the feature: cortical learning alternates theta bursts (exploration)
        // with sharp-wave ripples (consolidation). Mature at 2000 prevents
        // permanent throttling; Consolidating at 500 provides periodic stabilization.
        if reward_slow > 0.05
            && reward_fast > reward_slow * 0.95
            && reward_trend.abs() < 0.05 * slow_abs
            && self.reward_history.len() >= 500
        {
            return DevelopmentalStage::Consolidating;
        }

        // Differentiating: reward actively climbing with a meaningful gradient.
        // Threshold mirrors the Stressed condition (0.05 × slow_abs) but positive.
        // Previously used `> 0.0` which fired on any noise — in late training when
        // reward has plateaued, tiny fluctuations caused constant Consolidating↔
        // Differentiating thrashing (cg oscillates 1.0↔2.0, pm 0.8↔1.2), destabilising
        // the readout at exactly the measurement point. Requiring a proportional trend
        // preserves the intentional theta/SWR explore-exploit rhythm during genuine
        // improvement phases while suppressing noise-driven late-training oscillation.
        if reward_trend > 0.05 * slow_abs {
            return DevelopmentalStage::Differentiating;
        }

        // Proliferating: morphon count rising, reward not yet meaningful.
        if mc_trend > 0.01 {
            return DevelopmentalStage::Proliferating;
        }

        DevelopmentalStage::Differentiating
    }

    /// Returns true if anti-hub scaling should be applied.
    /// This is biologically appropriate mainly in Mature and Consolidating stages
    /// when selective feature detectors have formed and need protection from
    /// indiscriminate hubs. In early stages (Proliferating, Differentiating),
    /// uniform activity patterns are normal and should not be suppressed.
    pub fn should_apply_anti_hub_scaling(&self) -> bool {
        matches!(
            self.stage,
            DevelopmentalStage::Mature | DevelopmentalStage::Consolidating
        )
    }

    /// Compute standard deviation of f32 history.
    fn std_f32(&self, history: &VecDeque<f32>) -> f32 {
        if history.len() < 2 {
            return 0.0;
        }
        let n = history.len() as f32;
        let mean: f32 = history.iter().sum::<f32>() / n;
        let variance: f32 = history.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        variance.sqrt()
    }

    /// Compute normalized trend (slope / mean) of u32 history.
    fn trend(&self, history: &VecDeque<u32>) -> f32 {
        if history.len() < 10 {
            return 0.0;
        }
        let n = history.len();
        let recent = history.iter().rev().take(n / 2).sum::<u32>() as f32 / (n / 2) as f32;
        let older = history.iter().take(n / 2).sum::<u32>() as f32 / (n / 2) as f32;
        let mean = (recent + older) / 2.0;
        if mean < 1.0 {
            return 0.0;
        }
        (recent - older) / mean
    }

    /// Compute normalized trend of f32 history.
    fn trend_f32(&self, history: &VecDeque<f32>) -> f32 {
        if history.len() < 10 {
            return 0.0;
        }
        let n = history.len();
        let half = n / 2;
        let recent: f32 = history.iter().rev().take(half).sum::<f32>() / half as f32;
        let older: f32 = history.iter().take(half).sum::<f32>() / half as f32;
        recent - older
    }
}

fn ema_update_f32(current: &mut f32, value: f32, alpha: f32) {
    *current = *current * (1.0 - alpha) + value * alpha;
}

// ─── Intervention logging ────────────────────────────────────────────

/// Record of a single regulation action for diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intervention {
    pub rule: String,
    pub vital: String,
    pub actual: f32,
    pub setpoint: f32,
    pub lever: String,
    pub adjustment: f32,
}

/// Diagnostics snapshot from the last Endoquilibrium tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndoDiagnostics {
    pub stage: DevelopmentalStage,
    pub channels: ChannelState,
    pub interventions: Vec<Intervention>,
    pub health_score: f32,
}

impl Default for EndoDiagnostics {
    fn default() -> Self {
        Self {
            stage: DevelopmentalStage::Proliferating,
            channels: ChannelState::default(),
            interventions: Vec::new(),
            health_score: 1.0,
        }
    }
}

// ─── Endoquilibrium (main struct) ────────────────────────────────────

/// Predictive neuroendocrine regulation engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Endoquilibrium {
    /// Current channel gains/biases applied to the system.
    pub channels: ChannelState,
    /// Configuration.
    pub config: EndoConfig,
    /// Dual-timescale predictor.
    predictor: AllostasisPredictor,
    /// Stage-dependent setpoints.
    setpoints: DevelopmentalSetpoints,
    /// Ticks since the last tag-capture event.
    pub(crate) ticks_since_last_capture: u64,
    /// Last diagnostics snapshot.
    pub last_diag: EndoDiagnostics,
}

impl Default for Endoquilibrium {
    fn default() -> Self {
        Self::new(EndoConfig::default())
    }
}

impl Endoquilibrium {
    pub fn new(config: EndoConfig) -> Self {
        Self {
            channels: ChannelState::default(),
            predictor: AllostasisPredictor::default(),
            setpoints: DevelopmentalSetpoints::default(),
            ticks_since_last_capture: 0,
            last_diag: EndoDiagnostics::default(),
            config,
        }
    }

    /// Run one regulation tick: update predictor, apply rules, smooth channels.
    pub fn tick(&mut self, vitals: VitalSigns) {
        if !self.config.enabled {
            return;
        }

        let fast_alpha = 1.0 / self.config.fast_tau;
        let slow_alpha = 1.0 / self.config.slow_tau;
        self.predictor.update(&vitals, fast_alpha, slow_alpha, self.config.mature_min_updates);

        // Update setpoints for current developmental stage
        self.setpoints = DevelopmentalSetpoints::for_stage(self.predictor.stage);

        // Track tag-capture health
        if vitals.capture_count > 0 {
            self.ticks_since_last_capture = 0;
        } else {
            self.ticks_since_last_capture += 1;
        }

        // Compute regulation adjustments
        let (raw, interventions) = self.regulate(&vitals);

        // Smooth and clamp
        self.apply_smoothing(&raw);

        // Compute health score (0–1, composite)
        let health = self.compute_health(&vitals);

        self.last_diag = EndoDiagnostics {
            stage: self.predictor.stage,
            channels: self.channels.clone(),
            interventions,
            health_score: health,
        };
    }

    /// Apply the 6 regulation rules. Returns raw (unsmoothed) channel targets
    /// and a list of interventions for logging.
    fn regulate(&self, vitals: &VitalSigns) -> (ChannelState, Vec<Intervention>) {
        let mut ch = ChannelState::default();
        let mut interventions = Vec::new();
        let sp = &self.setpoints;
        let cfg = &self.config;

        // Use fast EMA for reactive regulation (responds to acute changes)
        let fr_a = self.predictor.fast_emas.fr_associative;
        let elig = self.predictor.fast_emas.eligibility_density;
        let entropy = self.predictor.fast_emas.weight_entropy;
        let energy = self.predictor.fast_emas.energy_utilization;

        // ── Rule 1: Firing Rate Regulation ──
        if fr_a < sp.fr_assoc_min {
            let deficit = sp.fr_assoc_min - fr_a;
            ch.threshold_bias -= deficit * cfg.fr_deficit_threshold_k;
            ch.arousal_gain += deficit * cfg.fr_deficit_arousal_k;
            ch.novelty_gain += deficit * cfg.fr_deficit_novelty_k;
            interventions.push(Intervention {
                rule: "firing_rate_low".into(),
                vital: "fr_associative".into(),
                actual: fr_a,
                setpoint: sp.fr_assoc_min,
                lever: "threshold_bias/arousal/novelty".into(),
                adjustment: -deficit,
            });
        }
        if fr_a > sp.fr_assoc_max {
            let excess = fr_a - sp.fr_assoc_max;
            ch.threshold_bias += excess * cfg.fr_excess_threshold_k;
            ch.homeostasis_gain += excess * cfg.fr_excess_homeo_k;
            interventions.push(Intervention {
                rule: "firing_rate_high".into(),
                vital: "fr_associative".into(),
                actual: fr_a,
                setpoint: sp.fr_assoc_max,
                lever: "threshold_bias/homeostasis".into(),
                adjustment: excess,
            });
        }

        // ── Rule 2: Eligibility Density Regulation ──
        if elig < sp.elig_min {
            ch.novelty_gain += cfg.elig_low_novelty_k;
            ch.plasticity_mult *= cfg.elig_low_plast_k;
            interventions.push(Intervention {
                rule: "eligibility_low".into(),
                vital: "eligibility_density".into(),
                actual: elig,
                setpoint: sp.elig_min,
                lever: "novelty/plasticity_mult".into(),
                adjustment: sp.elig_min - elig,
            });
        }
        if elig > sp.elig_max {
            ch.homeostasis_gain += cfg.elig_high_homeo_k;
            ch.plasticity_mult *= cfg.elig_high_plast_k;
            interventions.push(Intervention {
                rule: "eligibility_high".into(),
                vital: "eligibility_density".into(),
                actual: elig,
                setpoint: sp.elig_max,
                lever: "homeostasis/plasticity_mult".into(),
                adjustment: elig - sp.elig_max,
            });
        }

        // ── Rule 3: Weight Distribution Health ──
        if entropy < sp.entropy_min {
            ch.novelty_gain += cfg.entropy_low_novelty_k;
            ch.plasticity_mult *= cfg.entropy_low_plast_k;
            interventions.push(Intervention {
                rule: "entropy_collapse".into(),
                vital: "weight_entropy".into(),
                actual: entropy,
                setpoint: sp.entropy_min,
                lever: "novelty/plasticity_mult".into(),
                adjustment: sp.entropy_min - entropy,
            });
        }
        if entropy > sp.entropy_max {
            ch.plasticity_mult *= cfg.entropy_high_plast_k;
            ch.homeostasis_gain += cfg.entropy_high_homeo_k;
            interventions.push(Intervention {
                rule: "entropy_explosion".into(),
                vital: "weight_entropy".into(),
                actual: entropy,
                setpoint: sp.entropy_max,
                lever: "plasticity_mult/homeostasis".into(),
                adjustment: entropy - sp.entropy_max,
            });
        }

        // ── Rule 4: Cell Type Balance ──
        // Detect overrepresented types and apply structural pressure:
        // raise division threshold (slow growth) and pruning threshold (trim excess).
        let mut max_excess: f32 = 0.0;
        for (i, &fraction) in vitals.cell_type_fractions.iter().enumerate() {
            let target = sp.type_targets[i];
            let excess = fraction - target;
            if excess > 0.15 {
                interventions.push(Intervention {
                    rule: "type_imbalance".into(),
                    vital: "cell_type_fraction".into(),
                    actual: fraction,
                    setpoint: target,
                    lever: "division_threshold_mult/pruning_threshold_mult".into(),
                    adjustment: excess,
                });
                if excess > max_excess {
                    max_excess = excess;
                }
            }
        }
        if max_excess > 0.0 {
            // Scale proportionally to imbalance severity (max +0.4 each).
            // This stacks with the stage-dependent Rules 12-13, applied later.
            let pressure = (max_excess / 0.30).min(1.0); // normalize: 0.30 excess = full pressure
            ch.division_threshold_mult *= 1.0 + pressure * 0.4;
            ch.pruning_threshold_mult *= 1.0 + pressure * 0.4;
        }

        // ── Rule 5: Tag-and-Capture Health ──
        if vitals.tag_count > 100 && vitals.capture_count == 0 {
            ch.reward_gain *= cfg.tag_capture_reward_boost;
            interventions.push(Intervention {
                rule: "tag_capture_stalled".into(),
                vital: "tag_capture_rate".into(),
                actual: 0.0,
                setpoint: 0.02,
                lever: "reward_gain".into(),
                adjustment: cfg.tag_capture_reward_boost - 1.0,
            });
        }

        // ── Rule 6: Consolidation Gain (PRP availability) ──
        // Biology: dopamine/norepinephrine gate PRP synthesis → capture.
        // Proliferating: learn everything fast (nothing to protect).
        // Differentiating: refining, still capturing aggressively.
        // Consolidating: normal capture, slowing down.
        // Mature: protect what works — very selective consolidation.
        // Stressed: HIGH plasticity (explore) but LOW consolidation
        //   (don't lock in bad patterns during a crisis).
        let (cg, pm_stage) = match self.predictor.stage {
            DevelopmentalStage::Proliferating => (2.5, 1.5),
            DevelopmentalStage::Differentiating => (2.0, 1.2),
            DevelopmentalStage::Consolidating => (1.0, 0.8),
            DevelopmentalStage::Mature => (0.5, 0.5),
            DevelopmentalStage::Stressed => (0.70, 1.5), // raised from 0.50 — protects learned representations during dips
        };
        ch.consolidation_gain = cg;
        ch.plasticity_mult *= pm_stage;
        if cg != 1.0 {
            interventions.push(Intervention {
                rule: "stage_consolidation".into(),
                vital: "developmental_stage".into(),
                actual: 0.0,
                setpoint: 1.0,
                lever: "consolidation_gain/plasticity_mult".into(),
                adjustment: cg - 1.0,
            });
        }

        // ── Rule 7: Energy Pressure ──
        if energy > 0.95 {
            // Critical: safe mode — suppress plasticity and consolidation but NOT
            // novelty_gain. Novelty (ACh) is a salience/detection signal; suppressing
            // it cuts the very signal that drives STDP and three-factor learning.
            // Under metabolic stress, heightened novelty is biologically appropriate
            // (increased alertness). Energy pressure is already expressed via plasticity.
            // suppress_novelty_on_energy=true restores pre-v4.6.0 behaviour for ablations.
            ch.plasticity_mult = 0.0;
            ch.homeostasis_gain = 2.0;
            if self.config.suppress_novelty_on_energy {
                ch.novelty_gain = 0.0;
            }
            interventions.push(Intervention {
                rule: "energy_critical".into(),
                vital: "energy_utilization".into(),
                actual: energy,
                setpoint: 0.70,
                lever: if self.config.suppress_novelty_on_energy { "plasticity/homeostasis/novelty" } else { "plasticity/homeostasis" }.into(),
                adjustment: energy - 0.95,
            });
        } else if energy > 0.85 {
            // Emergency — reduce plasticity but preserve novelty signal.
            // Prior behaviour (novelty_gain *= 0.2) was the root cause of ng-collapse
            // during MNIST training: depleted morphon energy repeatedly drove the EMA
            // toward 0.2, throttling STDP and suppressing learning mid-run.
            // suppress_novelty_on_energy=true restores old behaviour for ablations.
            ch.plasticity_mult *= 0.3;
            if self.config.suppress_novelty_on_energy {
                ch.novelty_gain *= 0.2;
            }
            interventions.push(Intervention {
                rule: "energy_emergency".into(),
                vital: "energy_utilization".into(),
                actual: energy,
                setpoint: 0.70,
                lever: if self.config.suppress_novelty_on_energy { "plasticity/novelty" } else { "plasticity_mult" }.into(),
                adjustment: energy - 0.85,
            });
        } else if energy > 0.70 {
            // Pressure
            ch.plasticity_mult *= 0.7;
            interventions.push(Intervention {
                rule: "energy_pressure".into(),
                vital: "energy_utilization".into(),
                actual: energy,
                setpoint: 0.70,
                lever: "plasticity_mult".into(),
                adjustment: energy - 0.70,
            });
        }

        // ── Rule 8: Winner Adaptation Multiplier (stage-dependent) ──
        let wam = match self.predictor.stage {
            DevelopmentalStage::Proliferating => 1.5,
            DevelopmentalStage::Differentiating => 1.2,
            DevelopmentalStage::Consolidating => 1.0,
            DevelopmentalStage::Mature => 0.7,
            DevelopmentalStage::Stressed => 1.3,
        };
        ch.winner_adaptation_mult = wam;
        if (wam - 1.0).abs() > 0.01 {
            interventions.push(Intervention {
                rule: "stage_winner_adaptation".into(),
                vital: "developmental_stage".into(),
                actual: 0.0,
                setpoint: 1.0,
                lever: "winner_adaptation_mult".into(),
                adjustment: wam - 1.0,
            });
        }

        // ── Rule 9: Capture Threshold Multiplier (tag-capture health) ──
        let ctm = if self.ticks_since_last_capture > cfg.tag_capture_stale_ticks {
            0.7 // long stall → lower threshold (easier capture)
        } else if self.ticks_since_last_capture > cfg.tag_capture_stale_ticks / 2 {
            0.85
        } else {
            1.0
        };
        ch.capture_threshold_mult = ctm;
        if ctm != 1.0 {
            interventions.push(Intervention {
                rule: "capture_threshold_health".into(),
                vital: "ticks_since_capture".into(),
                actual: self.ticks_since_last_capture as f32,
                setpoint: cfg.tag_capture_stale_ticks as f32,
                lever: "capture_threshold_mult".into(),
                adjustment: ctm - 1.0,
            });
        }

        // ── Rule 10: Rollback PE Threshold Multiplier (stage-dependent) ──
        let rpm = match self.predictor.stage {
            DevelopmentalStage::Proliferating => 1.5, // tolerant during growth
            DevelopmentalStage::Differentiating => 1.0,
            DevelopmentalStage::Consolidating => 0.7,
            DevelopmentalStage::Mature => 0.5, // strict when mature
            DevelopmentalStage::Stressed => 0.25, // very strict under stress
        };
        ch.rollback_pe_threshold_mult = rpm;
        if (rpm - 1.0).abs() > 0.01 {
            interventions.push(Intervention {
                rule: "stage_rollback_sensitivity".into(),
                vital: "developmental_stage".into(),
                actual: 0.0,
                setpoint: 1.0,
                lever: "rollback_pe_threshold_mult".into(),
                adjustment: rpm - 1.0,
            });
        }

        // ── Rule 11: Tau Eligibility Multiplier (capture-speed driven) ──
        // Slow captures → widen the credit assignment window (tau_e up).
        // Fast captures → tighten it for precision (tau_e down).
        // This is the single lever that most directly enables temporal processing.
        let tem = if self.ticks_since_last_capture > cfg.tag_capture_stale_ticks * 2 {
            1.8 // very slow captures → wide window
        } else if self.ticks_since_last_capture > cfg.tag_capture_stale_ticks {
            1.4 // slow captures → wider
        } else if self.ticks_since_last_capture < cfg.tag_capture_stale_ticks / 4
            && vitals.capture_count > 0
        {
            0.7 // fast captures → tighter for precision
        } else {
            1.0
        };
        ch.tau_eligibility_mult = tem;
        if (tem - 1.0).abs() > 0.01 {
            interventions.push(Intervention {
                rule: "tau_eligibility_capture_speed".into(),
                vital: "ticks_since_capture".into(),
                actual: self.ticks_since_last_capture as f32,
                setpoint: cfg.tag_capture_stale_ticks as f32,
                lever: "tau_eligibility_mult".into(),
                adjustment: tem - 1.0,
            });
        }

        // ── Rule 15: Slow Trace Leak (stage-dependent) ──
        // Controls eligibility trace decay rate: >1 = faster decay (tighter credit window),
        // <1 = slower decay (broader credit window).
        // Proliferating: broad credit assignment (explore), Mature: precise (exploit).
        let stl = match self.predictor.stage {
            DevelopmentalStage::Proliferating => 0.6,
            DevelopmentalStage::Differentiating => 0.8,
            DevelopmentalStage::Consolidating => 1.0,
            DevelopmentalStage::Mature => 1.4,
            DevelopmentalStage::Stressed => 0.7,
        };
        ch.slow_trace_leak = stl;
        if (stl - 1.0).abs() > 0.01 {
            interventions.push(Intervention {
                rule: "stage_slow_trace_leak".into(),
                vital: "developmental_stage".into(),
                actual: 0.0,
                setpoint: 1.0,
                lever: "slow_trace_leak".into(),
                adjustment: stl - 1.0,
            });
        }

        // ── Rule 12: Division Threshold Multiplier (stage-dependent) ──
        // Proliferating: low threshold → easier division (explore topology).
        // Consolidating/Mature: high threshold → protect structure.
        // Stressed: moderate → allow some structural escape.
        let dtm = match self.predictor.stage {
            DevelopmentalStage::Proliferating => 0.6,
            DevelopmentalStage::Differentiating => 0.8,
            DevelopmentalStage::Consolidating => 1.3,
            DevelopmentalStage::Mature => 1.8,
            DevelopmentalStage::Stressed => 0.9,
        };
        ch.division_threshold_mult = dtm;
        if (dtm - 1.0).abs() > 0.01 {
            interventions.push(Intervention {
                rule: "stage_division_threshold".into(),
                vital: "developmental_stage".into(),
                actual: 0.0,
                setpoint: 1.0,
                lever: "division_threshold_mult".into(),
                adjustment: dtm - 1.0,
            });
        }

        // ── Rule 13: Pruning Threshold Multiplier (stage-dependent) ──
        // Proliferating: low → harder to prune (keep new connections).
        // Consolidating/Mature: high → aggressively prune weak connections.
        let ptm = match self.predictor.stage {
            DevelopmentalStage::Proliferating => 0.6,
            DevelopmentalStage::Differentiating => 0.8,
            DevelopmentalStage::Consolidating => 1.3,
            DevelopmentalStage::Mature => 1.5,
            DevelopmentalStage::Stressed => 1.0,
        };
        ch.pruning_threshold_mult = ptm;
        if (ptm - 1.0).abs() > 0.01 {
            interventions.push(Intervention {
                rule: "stage_pruning_threshold".into(),
                vital: "developmental_stage".into(),
                actual: 0.0,
                setpoint: 1.0,
                lever: "pruning_threshold_mult".into(),
                adjustment: ptm - 1.0,
            });
        }

        // ── Rule 14: Frustration Sensitivity Multiplier (stage-dependent) ──
        // Proliferating: high → more tolerant (failures expected during exploration).
        // Mature: low → more sensitive (something broke).
        let fsm = match self.predictor.stage {
            DevelopmentalStage::Proliferating => 1.5,
            DevelopmentalStage::Differentiating => 1.2,
            DevelopmentalStage::Consolidating => 1.0,
            DevelopmentalStage::Mature => 0.7,
            DevelopmentalStage::Stressed => 0.5,
        };
        ch.frustration_sensitivity_mult = fsm;
        if (fsm - 1.0).abs() > 0.01 {
            interventions.push(Intervention {
                rule: "stage_frustration_sensitivity".into(),
                vital: "developmental_stage".into(),
                actual: 0.0,
                setpoint: 1.0,
                lever: "frustration_sensitivity_mult".into(),
                adjustment: fsm - 1.0,
            });
        }

        // ── Rule 16: Migration Rate Multiplier (stage-dependent) ──
        // Proliferating: high migration (explore topology space).
        // Mature: low migration (protect spatial organization).
        let mrm = match self.predictor.stage {
            DevelopmentalStage::Proliferating => 1.5,
            DevelopmentalStage::Differentiating => 1.2,
            DevelopmentalStage::Consolidating => 0.7,
            DevelopmentalStage::Mature => 0.4,
            DevelopmentalStage::Stressed => 1.0,
        };
        ch.migration_rate_mult = mrm;
        if (mrm - 1.0).abs() > 0.01 {
            interventions.push(Intervention {
                rule: "stage_migration_rate".into(),
                vital: "developmental_stage".into(),
                actual: 0.0,
                setpoint: 1.0,
                lever: "migration_rate_mult".into(),
                adjustment: mrm - 1.0,
            });
        }

        // ── Rule 17: Synaptogenesis Threshold Multiplier (stage-dependent) ──
        // Proliferating: low threshold (form connections easily, explore).
        // Mature: high threshold (only strong correlations warrant new connections).
        let stm = match self.predictor.stage {
            DevelopmentalStage::Proliferating => 0.7,
            DevelopmentalStage::Differentiating => 0.85,
            DevelopmentalStage::Consolidating => 1.0,
            DevelopmentalStage::Mature => 1.3,
            DevelopmentalStage::Stressed => 0.8,
        };
        ch.synaptogenesis_threshold_mult = stm;
        if (stm - 1.0).abs() > 0.01 {
            interventions.push(Intervention {
                rule: "stage_synaptogenesis_threshold".into(),
                vital: "developmental_stage".into(),
                actual: 0.0,
                setpoint: 1.0,
                lever: "synaptogenesis_threshold_mult".into(),
                adjustment: stm - 1.0,
            });
        }

        (ch, interventions)
    }

    /// Smooth raw channel adjustments toward current state (EMA) and clamp.
    fn apply_smoothing(&mut self, raw: &ChannelState) {
        let a = self.config.smoothing_alpha;
        self.channels.reward_gain = lerp(self.channels.reward_gain, raw.reward_gain, a);
        self.channels.novelty_gain = lerp(self.channels.novelty_gain, raw.novelty_gain, a);
        self.channels.arousal_gain = lerp(self.channels.arousal_gain, raw.arousal_gain, a);
        self.channels.homeostasis_gain =
            lerp(self.channels.homeostasis_gain, raw.homeostasis_gain, a);
        self.channels.threshold_bias = lerp(self.channels.threshold_bias, raw.threshold_bias, a);
        self.channels.plasticity_mult = lerp(self.channels.plasticity_mult, raw.plasticity_mult, a);
        self.channels.consolidation_gain =
            lerp(self.channels.consolidation_gain, raw.consolidation_gain, a);
        self.channels.winner_adaptation_mult = lerp(
            self.channels.winner_adaptation_mult,
            raw.winner_adaptation_mult,
            a,
        );
        self.channels.capture_threshold_mult = lerp(
            self.channels.capture_threshold_mult,
            raw.capture_threshold_mult,
            a,
        );
        self.channels.rollback_pe_threshold_mult = lerp(
            self.channels.rollback_pe_threshold_mult,
            raw.rollback_pe_threshold_mult,
            a,
        );
        self.channels.tau_eligibility_mult = lerp(
            self.channels.tau_eligibility_mult,
            raw.tau_eligibility_mult,
            a,
        );
        self.channels.slow_trace_leak = lerp(self.channels.slow_trace_leak, raw.slow_trace_leak, a);
        self.channels.division_threshold_mult = lerp(
            self.channels.division_threshold_mult,
            raw.division_threshold_mult,
            a,
        );
        self.channels.pruning_threshold_mult = lerp(
            self.channels.pruning_threshold_mult,
            raw.pruning_threshold_mult,
            a,
        );
        self.channels.frustration_sensitivity_mult = lerp(
            self.channels.frustration_sensitivity_mult,
            raw.frustration_sensitivity_mult,
            a,
        );
        self.channels.migration_rate_mult = lerp(
            self.channels.migration_rate_mult,
            raw.migration_rate_mult,
            a,
        );
        self.channels.synaptogenesis_threshold_mult = lerp(
            self.channels.synaptogenesis_threshold_mult,
            raw.synaptogenesis_threshold_mult,
            a,
        );
        self.channels.clamp();
        // Apply plasticity floor after clamping — prevents Mature from suppressing learning
        // below the configured minimum (useful for supervised tasks).
        if self.config.plasticity_floor > 0.0 {
            self.channels.plasticity_mult = self
                .channels
                .plasticity_mult
                .max(self.config.plasticity_floor);
        }
    }

    /// Composite health score (0–1). 1.0 = all vitals within setpoints.
    fn compute_health(&self, vitals: &VitalSigns) -> f32 {
        let sp = &self.setpoints;
        let mut score = 1.0_f32;

        // FR health: penalty for being outside [min, max]
        let fr_a = self.predictor.fast_emas.fr_associative;
        if fr_a < sp.fr_assoc_min {
            score -= ((sp.fr_assoc_min - fr_a) / sp.fr_assoc_min.max(0.01)).min(1.0) * 0.3;
        } else if fr_a > sp.fr_assoc_max {
            score -= ((fr_a - sp.fr_assoc_max) / sp.fr_assoc_max.max(0.01)).min(1.0) * 0.2;
        }

        // Eligibility health
        let elig = self.predictor.fast_emas.eligibility_density;
        if elig < sp.elig_min {
            score -= 0.2;
        }

        // Entropy health
        let ent = self.predictor.fast_emas.weight_entropy;
        if ent < sp.entropy_min || ent > sp.entropy_max {
            score -= 0.2;
        }

        // Energy health
        if vitals.energy_utilization > 0.85 {
            score -= 0.3;
        }

        score.max(0.0)
    }

    /// Format a concise summary for logging.
    pub fn summary(&self) -> String {
        let s = &self.predictor.stage;
        let c = &self.channels;
        format!(
            "endo: stage={:?} rg={:.2} ng={:.2} ag={:.2} hg={:.2} tb={:.3} pm={:.2} cg={:.2} wam={:.2} ctm={:.2} tem={:.2} hp={:.2}",
            s, c.reward_gain, c.novelty_gain, c.arousal_gain,
            c.homeostasis_gain, c.threshold_bias, c.plasticity_mult,
            c.consolidation_gain, c.winner_adaptation_mult,
            c.capture_threshold_mult, c.tau_eligibility_mult,
            self.last_diag.health_score,
        )
    }

    /// Current developmental stage.
    pub fn stage(&self) -> DevelopmentalStage {
        self.predictor.stage
    }

    /// Slow EMA of reward signal (what Mature detection uses).
    pub fn reward_slow(&self) -> f32 {
        self.predictor.slow_emas.reward_avg
    }

    /// Coefficient of variation of reward history (Mature trigger: cv < 0.15).
    pub fn reward_cv(&self) -> f32 {
        let fast = self.predictor.fast_emas.reward_avg;
        if fast.abs() > 0.01 {
            self.predictor.std_f32(&self.predictor.reward_history) / fast.abs()
        } else {
            99.0
        }
    }

    /// Target associative firing rate (midpoint of setpoint range).
    /// Used by iSTDP to derive its homeostatic target when not explicitly configured.
    pub fn target_assoc_firing_rate(&self) -> f64 {
        ((self.setpoints.fr_assoc_min + self.setpoints.fr_assoc_max) / 2.0) as f64
    }

    /// Delegates to AllostasisPredictor — anti-hub scaling is stage-gated.
    pub fn should_apply_anti_hub_scaling(&self) -> bool {
        self.predictor.should_apply_anti_hub_scaling()
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vitals(fr_assoc: f32, elig: f32, entropy: f32, energy: f32) -> VitalSigns {
        VitalSigns {
            fr_associative: fr_assoc,
            eligibility_density: elig,
            weight_entropy: entropy,
            energy_utilization: energy,
            ..Default::default()
        }
    }

    #[test]
    fn test_disabled_is_neutral() {
        let mut endo = Endoquilibrium::new(EndoConfig::default());
        assert!(!endo.config.enabled);
        let vitals = make_vitals(0.0, 0.0, 0.0, 0.99);
        endo.tick(vitals);
        // When disabled, channels stay at defaults
        assert_eq!(endo.channels.threshold_bias, 0.0);
        assert_eq!(endo.channels.plasticity_mult, 1.0);
        assert_eq!(endo.channels.reward_gain, 1.0);
    }

    #[test]
    fn test_rule1_fr_low() {
        let mut endo = Endoquilibrium::new(EndoConfig {
            enabled: true,
            ..Default::default()
        });
        // Feed 0% FR for enough ticks that fast EMA drops
        for _ in 0..100 {
            endo.tick(make_vitals(0.0, 0.3, 3.0, 0.5));
        }
        // threshold_bias should be negative (lowering thresholds)
        assert!(
            endo.channels.threshold_bias < 0.0,
            "threshold_bias should be negative when FR=0%, got {}",
            endo.channels.threshold_bias
        );
        // arousal and novelty should be elevated
        assert!(
            endo.channels.arousal_gain > 1.0,
            "arousal_gain should be >1.0, got {}",
            endo.channels.arousal_gain
        );
    }

    #[test]
    fn test_rule1_fr_high() {
        let mut endo = Endoquilibrium::new(EndoConfig {
            enabled: true,
            ..Default::default()
        });
        for _ in 0..100 {
            endo.tick(make_vitals(0.40, 0.3, 3.0, 0.5));
        }
        // threshold_bias should be positive (raising thresholds)
        assert!(
            endo.channels.threshold_bias > 0.0,
            "threshold_bias should be positive when FR=40%, got {}",
            endo.channels.threshold_bias
        );
        assert!(endo.channels.homeostasis_gain > 1.0);
    }

    #[test]
    fn test_rule3_entropy_collapse() {
        let mut endo = Endoquilibrium::new(EndoConfig {
            enabled: true,
            ..Default::default()
        });
        for _ in 0..100 {
            endo.tick(make_vitals(0.10, 0.3, 0.5, 0.5)); // entropy=0.5, well below min
        }
        // Novelty should be boosted, plasticity elevated
        assert!(
            endo.channels.novelty_gain > 1.0,
            "novelty should boost on entropy collapse, got {}",
            endo.channels.novelty_gain
        );
        assert!(
            endo.channels.plasticity_mult > 1.0,
            "plasticity should boost on entropy collapse, got {}",
            endo.channels.plasticity_mult
        );
    }

    #[test]
    fn test_rule6_energy_critical() {
        let mut endo = Endoquilibrium::new(EndoConfig {
            enabled: true,
            ..Default::default()
        });
        for _ in 0..200 {
            endo.tick(make_vitals(0.10, 0.3, 3.0, 0.96));
        }
        // plasticity should be near zero, homeostasis maxed
        assert!(
            endo.channels.plasticity_mult < 0.2,
            "plasticity should be near 0 at critical energy, got {}",
            endo.channels.plasticity_mult
        );
    }

    #[test]
    fn test_smoothing_clamps() {
        let mut endo = Endoquilibrium::new(EndoConfig {
            enabled: true,
            ..Default::default()
        });
        // Even with extreme vitals, channels stay within bounds
        for _ in 0..500 {
            endo.tick(make_vitals(0.0, 0.0, 0.0, 0.99));
        }
        assert!(endo.channels.threshold_bias >= -0.3);
        assert!(endo.channels.threshold_bias <= 0.3);
        assert!(endo.channels.plasticity_mult >= 0.1);
        assert!(endo.channels.plasticity_mult <= 5.0);
        assert!(endo.channels.reward_gain >= 0.1);
        assert!(endo.channels.reward_gain <= 3.0);
        assert!(endo.channels.novelty_gain >= 0.0);
        assert!(endo.channels.novelty_gain <= 2.0);
    }

    #[test]
    fn test_ema_convergence() {
        let mut endo = Endoquilibrium::new(EndoConfig {
            enabled: true,
            ..Default::default()
        });
        let target_fr = 0.15;
        // Feed constant vitals
        for _ in 0..600 {
            endo.tick(make_vitals(target_fr, 0.3, 3.0, 0.5));
        }
        // Fast EMA should be very close (within 1%)
        let fast_fr = endo.predictor.fast_emas.fr_associative;
        assert!(
            (fast_fr - target_fr).abs() < 0.01,
            "fast EMA should converge to {}, got {}",
            target_fr,
            fast_fr
        );
        // Slow EMA should be close (within 5% after 600 ticks with tau=500)
        let slow_fr = endo.predictor.slow_emas.fr_associative;
        assert!(
            (slow_fr - target_fr).abs() < target_fr * 0.15,
            "slow EMA should approach {}, got {}",
            target_fr,
            slow_fr
        );
    }

    #[test]
    fn test_stage_detection_proliferating() {
        let mut endo = Endoquilibrium::new(EndoConfig {
            enabled: true,
            ..Default::default()
        });
        // Too little reward history → Proliferating
        for _ in 0..15 {
            let mut v = make_vitals(0.10, 0.3, 3.0, 0.5);
            v.reward_avg = 10.0;
            endo.tick(v);
        }
        assert_eq!(endo.stage(), DevelopmentalStage::Proliferating);
    }

    #[test]
    fn test_stage_detection_differentiating() {
        let mut endo = Endoquilibrium::new(EndoConfig {
            enabled: true,
            ..Default::default()
        });
        // Feed rising reward → Differentiating
        for i in 0..100 {
            let mut v = make_vitals(0.10, 0.3, 3.0, 0.5);
            v.reward_avg = 10.0 + i as f32 * 0.5;
            endo.tick(v);
        }
        assert_eq!(endo.stage(), DevelopmentalStage::Differentiating);
    }

    #[test]
    fn test_stage_detection_mature() {
        let mut endo = Endoquilibrium::new(EndoConfig {
            enabled: true,
            ..Default::default()
        });
        // Feed stable reward with low variance → Mature
        // Needs 2000+ ticks (Mature history gate raised to prevent premature triggering)
        // Alternate between two close values to pass dedup while keeping cv low and trend ~0
        for i in 0..2500 {
            let mut v = make_vitals(0.10, 0.3, 3.0, 0.5);
            v.reward_avg = 195.0 + if i % 2 == 0 { 0.01 } else { -0.01 };
            endo.tick(v);
        }
        assert_eq!(endo.stage(), DevelopmentalStage::Mature);
    }

    #[test]
    fn test_stage_detection_stressed() {
        let mut endo = Endoquilibrium::new(EndoConfig {
            enabled: true,
            ..Default::default()
        });
        // Build up a baseline, then drop reward → Stressed
        for i in 0..100 {
            let mut v = make_vitals(0.10, 0.3, 3.0, 0.5);
            v.reward_avg = 150.0 + (i as f32) * 0.001;
            endo.tick(v);
        }
        for i in 0..100 {
            let mut v = make_vitals(0.10, 0.3, 3.0, 0.5);
            v.reward_avg = 80.0 - (i as f32) * 0.001; // significant drop
            endo.tick(v);
        }
        assert_eq!(endo.stage(), DevelopmentalStage::Stressed);
    }

    #[test]
    fn test_weight_entropy_empty() {
        let topology = Topology::new();
        assert_eq!(compute_weight_entropy(&topology), 0.0);
    }

    #[test]
    fn test_healthy_vitals_stable() {
        let mut endo = Endoquilibrium::new(EndoConfig {
            enabled: true,
            ..Default::default()
        });
        // Feed perfectly healthy vitals with stable reward
        // Small jitter to bypass reward_history deduplication
        for i in 0..200 {
            let mut v = make_vitals(0.10, 0.30, 2.5, 0.5);
            v.reward_avg = 100.0 + (i as f32) * 0.001;
            endo.tick(v);
        }
        // Channels should be reasonable (stage may be Mature with pm_stage=0.5)
        assert!(
            (endo.channels.threshold_bias).abs() < 0.1,
            "healthy vitals should produce near-zero bias, got {}",
            endo.channels.threshold_bias
        );
        assert!(
            endo.channels.plasticity_mult >= 0.1,
            "plasticity should stay within bounds, got {}",
            endo.channels.plasticity_mult
        );
        assert!(endo.last_diag.health_score > 0.7);
    }

    // === Phase B lever tests ===

    /// Helper: drive endo into a specific stage and return the channel state.
    /// Patterns match the passing stage detection tests exactly (with jitter).
    fn drive_to_stage(stage: DevelopmentalStage) -> ChannelState {
        let mut endo = Endoquilibrium::new(EndoConfig {
            enabled: true,
            ..Default::default()
        });
        match stage {
            DevelopmentalStage::Proliferating => {
                for _ in 0..15 {
                    let mut v = make_vitals(0.10, 0.3, 3.0, 0.5);
                    v.reward_avg = 10.0;
                    endo.tick(v);
                }
            }
            DevelopmentalStage::Differentiating => {
                for i in 0..100 {
                    let mut v = make_vitals(0.10, 0.3, 3.0, 0.5);
                    v.reward_avg = 10.0 + i as f32 * 0.5;
                    endo.tick(v);
                }
            }
            DevelopmentalStage::Mature => {
                // Needs 2000+ ticks to pass Mature history gate
                for i in 0..2500 {
                    let mut v = make_vitals(0.10, 0.3, 3.0, 0.5);
                    v.reward_avg = 195.0 + if i % 2 == 0 { 0.01 } else { -0.01 };
                    endo.tick(v);
                }
            }
            DevelopmentalStage::Stressed => {
                for i in 0..100 {
                    let mut v = make_vitals(0.10, 0.3, 3.0, 0.5);
                    v.reward_avg = 150.0 + (i as f32) * 0.001;
                    endo.tick(v);
                }
                for i in 0..100 {
                    let mut v = make_vitals(0.10, 0.3, 3.0, 0.5);
                    v.reward_avg = 80.0 - (i as f32) * 0.001;
                    endo.tick(v);
                }
            }
            DevelopmentalStage::Consolidating => {
                // Rising reward then plateau near ceiling
                for i in 0..100 {
                    let mut v = make_vitals(0.10, 0.3, 3.0, 0.5);
                    v.reward_avg = 50.0 + i as f32 * 1.0;
                    endo.tick(v);
                }
                for i in 0..50 {
                    let mut v = make_vitals(0.10, 0.3, 3.0, 0.5);
                    v.reward_avg = 148.0 + (i as f32) * 0.001;
                    endo.tick(v);
                }
            }
        }
        assert_eq!(endo.stage(), stage, "failed to drive to {:?}", stage);
        endo.channels.clone()
    }

    #[test]
    fn phase_b_division_threshold_mult_by_stage() {
        let prolif = drive_to_stage(DevelopmentalStage::Proliferating);
        let mature = drive_to_stage(DevelopmentalStage::Mature);

        assert!(
            prolif.division_threshold_mult < 1.0,
            "Proliferating should lower division threshold, got {}",
            prolif.division_threshold_mult
        );
        assert!(
            mature.division_threshold_mult > 1.0,
            "Mature should raise division threshold, got {}",
            mature.division_threshold_mult
        );
        assert!(
            mature.division_threshold_mult > prolif.division_threshold_mult,
            "Mature dtm ({}) should be higher than Proliferating ({})",
            mature.division_threshold_mult,
            prolif.division_threshold_mult
        );
    }

    #[test]
    fn phase_b_pruning_threshold_mult_by_stage() {
        let prolif = drive_to_stage(DevelopmentalStage::Proliferating);
        let mature = drive_to_stage(DevelopmentalStage::Mature);

        assert!(
            prolif.pruning_threshold_mult < 1.0,
            "Proliferating should make pruning harder, got {}",
            prolif.pruning_threshold_mult
        );
        assert!(
            mature.pruning_threshold_mult > 1.0,
            "Mature should make pruning easier, got {}",
            mature.pruning_threshold_mult
        );
    }

    #[test]
    fn phase_b_frustration_sensitivity_mult_by_stage() {
        let prolif = drive_to_stage(DevelopmentalStage::Proliferating);
        let mature = drive_to_stage(DevelopmentalStage::Mature);

        assert!(
            prolif.frustration_sensitivity_mult > 1.0,
            "Proliferating should be more tolerant, got {}",
            prolif.frustration_sensitivity_mult
        );
        assert!(
            mature.frustration_sensitivity_mult < 1.0,
            "Mature should be more sensitive, got {}",
            mature.frustration_sensitivity_mult
        );
    }

    #[test]
    fn phase_b_stressed_allows_structural_escape() {
        let stressed = drive_to_stage(DevelopmentalStage::Stressed);

        // Stressed: division slightly easier than default (0.9 < 1.0),
        // frustration very sensitive (0.5), pruning neutral (1.0).
        assert!(
            stressed.division_threshold_mult < 1.0,
            "Stressed should allow some division for escape, got {}",
            stressed.division_threshold_mult
        );
        assert!(
            stressed.frustration_sensitivity_mult < 1.0,
            "Stressed should be frustration-sensitive, got {}",
            stressed.frustration_sensitivity_mult
        );
    }

    #[test]
    fn phase_b_levers_stay_in_clamp_range() {
        for stage in [
            DevelopmentalStage::Proliferating,
            DevelopmentalStage::Differentiating,
            DevelopmentalStage::Mature,
            DevelopmentalStage::Stressed,
        ] {
            let ch = drive_to_stage(stage);
            assert!(
                ch.division_threshold_mult >= 0.5 && ch.division_threshold_mult <= 2.0,
                "{:?}: dtm {} out of range",
                stage,
                ch.division_threshold_mult
            );
            assert!(
                ch.pruning_threshold_mult >= 0.5 && ch.pruning_threshold_mult <= 2.0,
                "{:?}: ptm {} out of range",
                stage,
                ch.pruning_threshold_mult
            );
            assert!(
                ch.frustration_sensitivity_mult >= 0.5 && ch.frustration_sensitivity_mult <= 2.0,
                "{:?}: fsm {} out of range",
                stage,
                ch.frustration_sensitivity_mult
            );
        }
    }

    // === sense_vitals new fields ===

    fn make_sense_vitals_inputs() -> (
        HashMap<MorphonId, Morphon>,
        Topology,
        Diagnostics,
        HashMap<crate::types::ClusterId, crate::morphogenesis::Cluster>,
    ) {
        (HashMap::new(), Topology::new(), Diagnostics::default(), HashMap::new())
    }

    #[test]
    fn sense_vitals_winners_per_cluster_zero_when_no_clusters() {
        let (morphons, topo, diag, clusters) = make_sense_vitals_inputs();
        let v = sense_vitals(&morphons, &topo, &diag, 0, 0.0, &clusters);
        assert_eq!(v.winners_per_cluster, 0.0);
    }

    #[test]
    fn sense_vitals_winners_per_cluster_counts_fired_members() {
        use crate::morphogenesis::Cluster;
        use crate::types::HyperbolicPoint;
        use crate::epistemic::EpistemicState;

        let pos = HyperbolicPoint::origin(3);
        let mut morphons: HashMap<MorphonId, Morphon> = HashMap::new();

        for id in 1..=4u64 {
            let mut m = Morphon::new(id, pos.clone());
            m.cell_type = CellType::Associative;
            m.fired = id <= 2; // members 1 and 2 fired, 3 and 4 didn't
            morphons.insert(id, m);
        }

        let mut clusters = HashMap::new();
        clusters.insert(0u64, Cluster {
            id: 0,
            members: vec![1, 2, 3, 4],
            shared_threshold: 0.5,
            inhibitory_morphons: vec![],
            shared_energy_pool: 0.0,
            shared_homeostatic_setpoint: 0.5,
            epistemic_state: EpistemicState::Hypothesis { formation_step: 0 },
            epistemic_history: Default::default(),
        });

        let v = sense_vitals(&morphons, &Topology::new(), &Diagnostics::default(), 0, 0.0, &clusters);
        assert_eq!(v.winners_per_cluster, 2.0, "2 of 4 members fired");
    }

    #[test]
    fn sense_vitals_winners_per_cluster_averages_across_clusters() {
        use crate::morphogenesis::Cluster;
        use crate::types::HyperbolicPoint;
        use crate::epistemic::EpistemicState;

        let pos = HyperbolicPoint::origin(3);
        let mut morphons: HashMap<MorphonId, Morphon> = HashMap::new();

        // Cluster A: members 1-4, all fired → 4 winners
        for id in 1..=4u64 {
            let mut m = Morphon::new(id, pos.clone());
            m.cell_type = CellType::Associative;
            m.fired = true;
            morphons.insert(id, m);
        }
        // Cluster B: members 5-8, none fired → 0 winners
        for id in 5..=8u64 {
            let mut m = Morphon::new(id, pos.clone());
            m.cell_type = CellType::Associative;
            m.fired = false;
            morphons.insert(id, m);
        }

        let mut clusters = HashMap::new();
        clusters.insert(0u64, Cluster {
            id: 0,
            members: vec![1, 2, 3, 4],
            shared_threshold: 0.5,
            inhibitory_morphons: vec![],
            shared_energy_pool: 0.0,
            shared_homeostatic_setpoint: 0.5,
            epistemic_state: EpistemicState::Hypothesis { formation_step: 0 },
            epistemic_history: Default::default(),
        });
        clusters.insert(1u64, Cluster {
            id: 1,
            members: vec![5, 6, 7, 8],
            shared_threshold: 0.5,
            inhibitory_morphons: vec![],
            shared_energy_pool: 0.0,
            shared_homeostatic_setpoint: 0.5,
            epistemic_state: EpistemicState::Hypothesis { formation_step: 0 },
            epistemic_history: Default::default(),
        });

        let v = sense_vitals(&morphons, &Topology::new(), &Diagnostics::default(), 0, 0.0, &clusters);
        // (4 + 0) / 2 clusters = 2.0 mean winners
        assert_eq!(v.winners_per_cluster, 2.0, "mean winners across 2 clusters");
    }

    #[test]
    fn sense_vitals_division_and_pruning_rate_from_diagnostics() {
        let (morphons, topo, mut diag, clusters) = make_sense_vitals_inputs();
        diag.division_events_recent = 5;
        diag.pruning_events_recent = 12;
        let v = sense_vitals(&morphons, &topo, &diag, 0, 0.0, &clusters);
        assert_eq!(v.division_rate, 5.0);
        assert_eq!(v.pruning_rate, 12.0);
    }

    #[test]
    fn sense_vitals_frustration_mean_from_diagnostics() {
        let (morphons, topo, mut diag, clusters) = make_sense_vitals_inputs();
        diag.avg_frustration = 0.42;
        let v = sense_vitals(&morphons, &topo, &diag, 0, 0.0, &clusters);
        assert!((v.frustration_mean - 0.42).abs() < 1e-4);
    }
}
