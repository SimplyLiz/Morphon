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
}

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
    pub cell_type_fractions: [f32; 6],
    pub total_morphons: u32,
    pub total_synapses: u32,
    // Metabolic
    pub energy_utilization: f32,
    // Task performance
    pub prediction_error_mean: f32,
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
            cell_type_fractions: [0.0; 6],
            total_morphons: 0,
            total_synapses: 0,
            energy_utilization: 0.5,
            prediction_error_mean: 0.1,
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
) -> VitalSigns {
    let n = morphons.len().max(1) as f32;

    // Firing rates by type (reuse Diagnostics data)
    let fr = |ct: CellType| -> f32 {
        diag.firing_by_type
            .get(&ct)
            .map(|&(fired, total)| if total > 0 { fired as f32 / total as f32 } else { 0.0 })
            .unwrap_or(0.0)
    };

    // Cell type fractions
    let mut type_counts = [0u32; 6];
    let mut pe_sum = 0.0_f32;
    let mut energy_sum = 0.0_f32;
    for m in morphons.values() {
        let idx = cell_type_index(m.cell_type);
        type_counts[idx] += 1;
        pe_sum += m.prediction_error as f32;
        energy_sum += m.energy as f32;
    }
    let mut fractions = [0.0_f32; 6];
    for i in 0..6 {
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

    VitalSigns {
        timestamp: step,
        fr_sensory: fr(CellType::Sensory),
        fr_associative: fr(CellType::Associative),
        fr_motor: fr(CellType::Motor),
        fr_modulatory: fr(CellType::Modulatory),
        eligibility_density,
        weight_entropy,
        tag_count: diag.active_tags as u32,
        capture_count: diag.captures_this_step as u32,
        cell_type_fractions: fractions,
        total_morphons: morphons.len() as u32,
        total_synapses: diag.total_synapses as u32,
        energy_utilization,
        prediction_error_mean: pe_sum / n,
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
            let w = if syn.weight.is_finite() { syn.weight } else { 0.0 };
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
            let w = if syn.weight.is_finite() { syn.weight } else { 0.0 };
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

// ─── Channel State (the 6 levers) ───────────────────────────────────

/// The 6 actuators Endoquilibrium controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelState {
    pub reward_gain: f32,
    pub novelty_gain: f32,
    pub arousal_gain: f32,
    pub homeostasis_gain: f32,
    pub threshold_bias: f32,
    pub plasticity_mult: f32,
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
    type_targets: [f32; 6], // S, A, M, Mod, Stem, Fused
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
                fr_assoc_min: 0.12, fr_assoc_max: 0.18,
                elig_min: 0.40, elig_max: 0.70,
                entropy_min: 3.0, entropy_max: 4.5,
                type_targets: [0.15, 0.20, 0.35, 0.10, 0.15, 0.05],
            },
            DevelopmentalStage::Differentiating => Self {
                fr_assoc_min: 0.10, fr_assoc_max: 0.15,
                elig_min: 0.30, elig_max: 0.60,
                entropy_min: 2.5, entropy_max: 4.0,
                type_targets: [0.10, 0.20, 0.40, 0.10, 0.15, 0.05],
            },
            DevelopmentalStage::Consolidating | DevelopmentalStage::Mature => Self {
                fr_assoc_min: 0.08, fr_assoc_max: 0.12,
                elig_min: 0.20, elig_max: 0.40,
                entropy_min: 2.0, entropy_max: 3.5,
                type_targets: [0.05, 0.20, 0.45, 0.10, 0.15, 0.05],
            },
            DevelopmentalStage::Stressed => Self {
                fr_assoc_min: 0.08, fr_assoc_max: 0.15,
                elig_min: 0.15, elig_max: 0.50,
                entropy_min: 2.0, entropy_max: 4.0,
                type_targets: [0.10, 0.20, 0.40, 0.10, 0.15, 0.05],
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
    stage: DevelopmentalStage,
}

impl Default for AllostasisPredictor {
    fn default() -> Self {
        Self {
            fast_emas: VitalSigns::default(),
            slow_emas: VitalSigns::default(),
            pe_history: VecDeque::with_capacity(200),
            morphon_count_history: VecDeque::with_capacity(200),
            stage: DevelopmentalStage::Proliferating,
        }
    }
}

impl AllostasisPredictor {
    fn update(&mut self, vitals: &VitalSigns, fast_alpha: f32, slow_alpha: f32) {
        // Update fast EMAs
        ema_update_f32(&mut self.fast_emas.fr_sensory, vitals.fr_sensory, fast_alpha);
        ema_update_f32(&mut self.fast_emas.fr_associative, vitals.fr_associative, fast_alpha);
        ema_update_f32(&mut self.fast_emas.fr_motor, vitals.fr_motor, fast_alpha);
        ema_update_f32(&mut self.fast_emas.fr_modulatory, vitals.fr_modulatory, fast_alpha);
        ema_update_f32(&mut self.fast_emas.eligibility_density, vitals.eligibility_density, fast_alpha);
        ema_update_f32(&mut self.fast_emas.weight_entropy, vitals.weight_entropy, fast_alpha);
        ema_update_f32(&mut self.fast_emas.energy_utilization, vitals.energy_utilization, fast_alpha);
        ema_update_f32(&mut self.fast_emas.prediction_error_mean, vitals.prediction_error_mean, fast_alpha);

        // Update slow EMAs
        ema_update_f32(&mut self.slow_emas.fr_sensory, vitals.fr_sensory, slow_alpha);
        ema_update_f32(&mut self.slow_emas.fr_associative, vitals.fr_associative, slow_alpha);
        ema_update_f32(&mut self.slow_emas.fr_motor, vitals.fr_motor, slow_alpha);
        ema_update_f32(&mut self.slow_emas.fr_modulatory, vitals.fr_modulatory, slow_alpha);
        ema_update_f32(&mut self.slow_emas.eligibility_density, vitals.eligibility_density, slow_alpha);
        ema_update_f32(&mut self.slow_emas.weight_entropy, vitals.weight_entropy, slow_alpha);
        ema_update_f32(&mut self.slow_emas.energy_utilization, vitals.energy_utilization, slow_alpha);
        ema_update_f32(&mut self.slow_emas.prediction_error_mean, vitals.prediction_error_mean, slow_alpha);

        // History for trend detection
        self.pe_history.push_back(vitals.prediction_error_mean);
        if self.pe_history.len() > 200 {
            self.pe_history.pop_front();
        }
        self.morphon_count_history.push_back(vitals.total_morphons);
        if self.morphon_count_history.len() > 200 {
            self.morphon_count_history.pop_front();
        }

        // Detect developmental stage
        self.stage = self.detect_stage();
    }

    fn detect_stage(&self) -> DevelopmentalStage {
        let mc_trend = self.trend(&self.morphon_count_history);
        let pe_trend = self.trend_f32(&self.pe_history);

        // Stressed: PE rising for extended period
        if pe_trend > 0.001 && self.pe_history.len() >= 100 {
            return DevelopmentalStage::Stressed;
        }
        // Proliferating: morphon count rising
        if mc_trend > 0.01 {
            return DevelopmentalStage::Proliferating;
        }
        // Consolidating: morphon count falling
        if mc_trend < -0.005 {
            return DevelopmentalStage::Consolidating;
        }
        // Differentiating: some structural change but count stable
        if mc_trend.abs() > 0.002 {
            return DevelopmentalStage::Differentiating;
        }
        DevelopmentalStage::Mature
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
        if mean < 1.0 { return 0.0; }
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
    ticks_since_last_capture: u64,
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
        self.predictor.update(&vitals, fast_alpha, slow_alpha);

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
                rule: "firing_rate_low".into(), vital: "fr_associative".into(),
                actual: fr_a, setpoint: sp.fr_assoc_min,
                lever: "threshold_bias/arousal/novelty".into(),
                adjustment: -deficit,
            });
        }
        if fr_a > sp.fr_assoc_max {
            let excess = fr_a - sp.fr_assoc_max;
            ch.threshold_bias += excess * cfg.fr_excess_threshold_k;
            ch.homeostasis_gain += excess * cfg.fr_excess_homeo_k;
            interventions.push(Intervention {
                rule: "firing_rate_high".into(), vital: "fr_associative".into(),
                actual: fr_a, setpoint: sp.fr_assoc_max,
                lever: "threshold_bias/homeostasis".into(),
                adjustment: excess,
            });
        }

        // ── Rule 2: Eligibility Density Regulation ──
        if elig < sp.elig_min {
            ch.novelty_gain += cfg.elig_low_novelty_k;
            ch.plasticity_mult *= cfg.elig_low_plast_k;
            interventions.push(Intervention {
                rule: "eligibility_low".into(), vital: "eligibility_density".into(),
                actual: elig, setpoint: sp.elig_min,
                lever: "novelty/plasticity_mult".into(),
                adjustment: sp.elig_min - elig,
            });
        }
        if elig > sp.elig_max {
            ch.homeostasis_gain += cfg.elig_high_homeo_k;
            ch.plasticity_mult *= cfg.elig_high_plast_k;
            interventions.push(Intervention {
                rule: "eligibility_high".into(), vital: "eligibility_density".into(),
                actual: elig, setpoint: sp.elig_max,
                lever: "homeostasis/plasticity_mult".into(),
                adjustment: elig - sp.elig_max,
            });
        }

        // ── Rule 3: Weight Distribution Health ──
        if entropy < sp.entropy_min {
            ch.novelty_gain += cfg.entropy_low_novelty_k;
            ch.plasticity_mult *= cfg.entropy_low_plast_k;
            interventions.push(Intervention {
                rule: "entropy_collapse".into(), vital: "weight_entropy".into(),
                actual: entropy, setpoint: sp.entropy_min,
                lever: "novelty/plasticity_mult".into(),
                adjustment: sp.entropy_min - entropy,
            });
        }
        if entropy > sp.entropy_max {
            ch.plasticity_mult *= cfg.entropy_high_plast_k;
            ch.homeostasis_gain += cfg.entropy_high_homeo_k;
            interventions.push(Intervention {
                rule: "entropy_explosion".into(), vital: "weight_entropy".into(),
                actual: entropy, setpoint: sp.entropy_max,
                lever: "plasticity_mult/homeostasis".into(),
                adjustment: entropy - sp.entropy_max,
            });
        }

        // ── Rule 4: Cell Type Balance ──
        for (i, &fraction) in vitals.cell_type_fractions.iter().enumerate() {
            let target = sp.type_targets[i];
            if fraction > target + 0.15 {
                interventions.push(Intervention {
                    rule: "type_imbalance".into(), vital: "cell_type_fraction".into(),
                    actual: fraction, setpoint: target,
                    lever: "logged_only".into(),
                    adjustment: fraction - target,
                });
            }
        }

        // ── Rule 5: Tag-and-Capture Health ──
        if vitals.tag_count > 100 && vitals.capture_count == 0 {
            ch.reward_gain *= cfg.tag_capture_reward_boost;
            interventions.push(Intervention {
                rule: "tag_capture_stalled".into(), vital: "tag_capture_rate".into(),
                actual: 0.0, setpoint: 0.02,
                lever: "reward_gain".into(),
                adjustment: cfg.tag_capture_reward_boost - 1.0,
            });
        }

        // ── Rule 6: Energy Pressure ──
        if energy > 0.95 {
            // Critical: safe mode
            ch.plasticity_mult = 0.0;
            ch.novelty_gain = 0.0;
            ch.homeostasis_gain = 2.0;
            interventions.push(Intervention {
                rule: "energy_critical".into(), vital: "energy_utilization".into(),
                actual: energy, setpoint: 0.70,
                lever: "plasticity/novelty/homeostasis".into(),
                adjustment: energy - 0.95,
            });
        } else if energy > 0.85 {
            // Emergency
            ch.plasticity_mult *= 0.3;
            ch.novelty_gain *= 0.2;
            interventions.push(Intervention {
                rule: "energy_emergency".into(), vital: "energy_utilization".into(),
                actual: energy, setpoint: 0.70,
                lever: "plasticity/novelty".into(),
                adjustment: energy - 0.85,
            });
        } else if energy > 0.70 {
            // Pressure
            ch.plasticity_mult *= 0.7;
            interventions.push(Intervention {
                rule: "energy_pressure".into(), vital: "energy_utilization".into(),
                actual: energy, setpoint: 0.70,
                lever: "plasticity_mult".into(),
                adjustment: energy - 0.70,
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
        self.channels.homeostasis_gain = lerp(self.channels.homeostasis_gain, raw.homeostasis_gain, a);
        self.channels.threshold_bias = lerp(self.channels.threshold_bias, raw.threshold_bias, a);
        self.channels.plasticity_mult = lerp(self.channels.plasticity_mult, raw.plasticity_mult, a);
        self.channels.clamp();
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
            "endo: stage={:?} rg={:.2} ng={:.2} ag={:.2} hg={:.2} tb={:.3} pm={:.2} hp={:.2}",
            s, c.reward_gain, c.novelty_gain, c.arousal_gain,
            c.homeostasis_gain, c.threshold_bias, c.plasticity_mult,
            self.last_diag.health_score,
        )
    }

    /// Current developmental stage.
    pub fn stage(&self) -> DevelopmentalStage {
        self.predictor.stage
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
        let mut endo = Endoquilibrium::new(EndoConfig { enabled: true, ..Default::default() });
        // Feed 0% FR for enough ticks that fast EMA drops
        for _ in 0..100 {
            endo.tick(make_vitals(0.0, 0.3, 3.0, 0.5));
        }
        // threshold_bias should be negative (lowering thresholds)
        assert!(endo.channels.threshold_bias < 0.0,
            "threshold_bias should be negative when FR=0%, got {}", endo.channels.threshold_bias);
        // arousal and novelty should be elevated
        assert!(endo.channels.arousal_gain > 1.0,
            "arousal_gain should be >1.0, got {}", endo.channels.arousal_gain);
    }

    #[test]
    fn test_rule1_fr_high() {
        let mut endo = Endoquilibrium::new(EndoConfig { enabled: true, ..Default::default() });
        for _ in 0..100 {
            endo.tick(make_vitals(0.40, 0.3, 3.0, 0.5));
        }
        // threshold_bias should be positive (raising thresholds)
        assert!(endo.channels.threshold_bias > 0.0,
            "threshold_bias should be positive when FR=40%, got {}", endo.channels.threshold_bias);
        assert!(endo.channels.homeostasis_gain > 1.0);
    }

    #[test]
    fn test_rule3_entropy_collapse() {
        let mut endo = Endoquilibrium::new(EndoConfig { enabled: true, ..Default::default() });
        for _ in 0..100 {
            endo.tick(make_vitals(0.10, 0.3, 0.5, 0.5)); // entropy=0.5, well below min
        }
        // Novelty should be boosted, plasticity elevated
        assert!(endo.channels.novelty_gain > 1.0,
            "novelty should boost on entropy collapse, got {}", endo.channels.novelty_gain);
        assert!(endo.channels.plasticity_mult > 1.0,
            "plasticity should boost on entropy collapse, got {}", endo.channels.plasticity_mult);
    }

    #[test]
    fn test_rule6_energy_critical() {
        let mut endo = Endoquilibrium::new(EndoConfig { enabled: true, ..Default::default() });
        for _ in 0..200 {
            endo.tick(make_vitals(0.10, 0.3, 3.0, 0.96));
        }
        // plasticity should be near zero, homeostasis maxed
        assert!(endo.channels.plasticity_mult < 0.2,
            "plasticity should be near 0 at critical energy, got {}", endo.channels.plasticity_mult);
    }

    #[test]
    fn test_smoothing_clamps() {
        let mut endo = Endoquilibrium::new(EndoConfig { enabled: true, ..Default::default() });
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
        let mut endo = Endoquilibrium::new(EndoConfig { enabled: true, ..Default::default() });
        let target_fr = 0.15;
        // Feed constant vitals
        for _ in 0..600 {
            endo.tick(make_vitals(target_fr, 0.3, 3.0, 0.5));
        }
        // Fast EMA should be very close (within 1%)
        let fast_fr = endo.predictor.fast_emas.fr_associative;
        assert!((fast_fr - target_fr).abs() < 0.01,
            "fast EMA should converge to {}, got {}", target_fr, fast_fr);
        // Slow EMA should be close (within 5% after 600 ticks with tau=500)
        let slow_fr = endo.predictor.slow_emas.fr_associative;
        assert!((slow_fr - target_fr).abs() < target_fr * 0.15,
            "slow EMA should approach {}, got {}", target_fr, slow_fr);
    }

    #[test]
    fn test_stage_detection_proliferating() {
        let mut endo = Endoquilibrium::new(EndoConfig { enabled: true, ..Default::default() });
        // Feed rising morphon count
        for i in 0..50 {
            let mut v = make_vitals(0.10, 0.3, 3.0, 0.5);
            v.total_morphons = 100 + i * 5;
            endo.tick(v);
        }
        assert_eq!(endo.stage(), DevelopmentalStage::Proliferating);
    }

    #[test]
    fn test_stage_detection_mature() {
        let mut endo = Endoquilibrium::new(EndoConfig { enabled: true, ..Default::default() });
        // Feed stable morphon count
        for _ in 0..50 {
            let mut v = make_vitals(0.10, 0.3, 3.0, 0.5);
            v.total_morphons = 300;
            endo.tick(v);
        }
        assert_eq!(endo.stage(), DevelopmentalStage::Mature);
    }

    #[test]
    fn test_weight_entropy_empty() {
        let topology = Topology::new();
        assert_eq!(compute_weight_entropy(&topology), 0.0);
    }

    #[test]
    fn test_healthy_vitals_no_intervention() {
        let mut endo = Endoquilibrium::new(EndoConfig { enabled: true, ..Default::default() });
        // Feed perfectly healthy vitals (within Mature setpoints)
        for _ in 0..200 {
            endo.tick(make_vitals(0.10, 0.30, 2.5, 0.5));
        }
        // Channels should stay near defaults
        assert!((endo.channels.threshold_bias).abs() < 0.05,
            "healthy vitals should produce near-zero bias, got {}", endo.channels.threshold_bias);
        assert!((endo.channels.plasticity_mult - 1.0).abs() < 0.3,
            "healthy vitals should keep plasticity near 1.0, got {}", endo.channels.plasticity_mult);
        assert!(endo.last_diag.health_score > 0.8);
    }
}
