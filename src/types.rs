//! Core types and enums for the Morphogenic Intelligence engine.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Unique identifier for a Morphon.
pub type MorphonId = u64;

/// Unique identifier for a Cluster (fused group of Morphons).
pub type ClusterId = u64;

/// Unique identifier for a Synapse.
pub type SynapseId = u64;

/// Unique identifier for a Lineage (tracks parent-child relationships).
pub type LineageId = u64;

/// Generation counter (how many divisions since the seed).
pub type Generation = u32;

/// Cell types that a Morphon can differentiate into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CellType {
    /// Pluripotent — can become anything. Initial state.
    Stem,
    /// Input processing — sensitive to external stimuli.
    Sensory,
    /// Pattern recognition and association.
    Associative,
    /// Output generation — drives actions.
    Motor,
    /// Internal regulation — produces modulatory signals.
    Modulatory,
    /// Part of a fused cluster — no longer fully autonomous.
    Fused,
}

impl Default for CellType {
    fn default() -> Self {
        CellType::Stem
    }
}

/// Neuromodulator types — the four broadcast channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModulatorType {
    /// Dopamine analog — reinforces recently active traces.
    Reward,
    /// Acetylcholine analog — increases plasticity systemwide.
    Novelty,
    /// Noradrenaline analog — increases sensitivity/alertness.
    Arousal,
    /// Serotonin analog — regulates baseline activity.
    Homeostasis,
}

/// The set of modulators a Morphon is sensitive to.
pub type ReceptorSet = HashSet<ModulatorType>;

/// Predefined developmental programs for bootstrapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevelopmentalProgram {
    /// For classification, pattern recognition.
    Cortical,
    /// For sequential learning, time series.
    Hippocampal,
    /// For motor control, robotics.
    Cerebellar,
    /// User-defined growth rules.
    Custom,
}

/// Lifecycle events that can be enabled/disabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleConfig {
    /// Allow cell division (mitosis).
    pub division: bool,
    /// Allow functional specialization changes.
    pub differentiation: bool,
    /// Allow cluster formation.
    pub fusion: bool,
    /// Allow programmed cell death.
    pub apoptosis: bool,
    /// Allow migration in information space.
    pub migration: bool,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            division: true,
            differentiation: true,
            fusion: true,
            apoptosis: true,
            migration: true,
        }
    }
}

/// Activation function type — changes with differentiation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActivationFn {
    /// General-purpose, used by Stem cells.
    Sigmoid,
    /// Sharp threshold, used by Sensory cells.
    HardThreshold,
    /// Integrating/accumulating, used by Associative cells.
    LeakyIntegrator,
    /// Burst-capable, used by Motor cells.
    Burst,
    /// Slow oscillatory, used by Modulatory cells.
    Oscillatory,
}

impl Default for ActivationFn {
    fn default() -> Self {
        ActivationFn::Sigmoid
    }
}

impl ActivationFn {
    /// Apply the activation function to an input value.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFn::HardThreshold => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFn::LeakyIntegrator => x.max(0.01 * x), // leaky ReLU variant
            ActivationFn::Burst => {
                // Burst: sigmoid-like baseline with amplified response above 0.5.
                // Below 0.5: gentle sigmoid (still responsive, unlike the old 0.1x dead zone).
                // Above 0.5: rapid ramp to saturation (burst behavior).
                let base = 1.0 / (1.0 + (-x).exp()); // sigmoid baseline
                if x > 0.5 { base + (x - 0.5).tanh() * 0.5 } else { base }
            }
            ActivationFn::Oscillatory => (x * std::f64::consts::PI).sin(),
        }
    }

    /// Maximum output value this activation function can produce.
    /// Used to clamp the homeostatic threshold so it never exceeds
    /// the activation's output range (which would permanently silence the morphon).
    pub fn max_output(&self) -> f64 {
        match self {
            ActivationFn::Sigmoid => 0.95,          // approaches 1.0 asymptotically
            ActivationFn::HardThreshold => 0.95,     // outputs exactly 1.0
            ActivationFn::LeakyIntegrator => 5.0,    // unbounded, use weight_max as proxy
            ActivationFn::Burst => 1.4,              // sigmoid() + tanh()*0.5, max ~1.5
            ActivationFn::Oscillatory => 0.95,       // sin(), max 1.0
        }
    }

    /// Returns the default activation function for a given cell type.
    pub fn for_cell_type(cell_type: CellType) -> Self {
        match cell_type {
            CellType::Stem => ActivationFn::Sigmoid,
            CellType::Sensory => ActivationFn::HardThreshold,
            CellType::Associative => ActivationFn::LeakyIntegrator,
            CellType::Motor => ActivationFn::Burst,
            CellType::Modulatory => ActivationFn::Oscillatory,
            CellType::Fused => ActivationFn::Sigmoid,
        }
    }
}

/// Default receptor sets for each cell type.
pub fn default_receptors(cell_type: CellType) -> ReceptorSet {
    let mut set = HashSet::new();
    match cell_type {
        CellType::Stem => {
            // Stem cells respond to all modulators
            set.insert(ModulatorType::Reward);
            set.insert(ModulatorType::Novelty);
            set.insert(ModulatorType::Arousal);
            set.insert(ModulatorType::Homeostasis);
        }
        CellType::Sensory => {
            set.insert(ModulatorType::Novelty);
            set.insert(ModulatorType::Arousal);
        }
        CellType::Associative => {
            set.insert(ModulatorType::Reward);
            set.insert(ModulatorType::Novelty);
        }
        CellType::Motor => {
            set.insert(ModulatorType::Reward);
            set.insert(ModulatorType::Arousal);
        }
        CellType::Modulatory => {
            set.insert(ModulatorType::Homeostasis);
        }
        CellType::Fused => {
            set.insert(ModulatorType::Reward);
            set.insert(ModulatorType::Homeostasis);
        }
    }
    set
}

/// Default receptor sensitivities for a cell type.
/// Present receptors get 1.0, absent get 0.01 (never fully deaf).
/// V2: Sub-morphon plasticity — these values adapt during learning.
pub fn default_receptor_sensitivity(cell_type: CellType) -> HashMap<ModulatorType, f64> {
    let receptors = default_receptors(cell_type);
    [
        ModulatorType::Reward,
        ModulatorType::Novelty,
        ModulatorType::Arousal,
        ModulatorType::Homeostasis,
    ]
    .iter()
    .map(|&ch| {
        let val = if receptors.contains(&ch) { 1.0 } else { 0.01 };
        (ch, val)
    })
    .collect()
}

/// Position in N-dimensional hyperbolic information space (Poincaré ball model).
///
/// The Poincaré ball maps the entire hyperbolic space into the unit ball.
/// Points near the origin are "general", points near the boundary are "specific".
/// Hierarchies are encoded logarithmically (not exponentially as in Euclidean space).
///
/// Based on: Nickel & Kiela (2017) "Poincaré Embeddings for Learning Hierarchical
/// Representations", Ganea et al. (2018) "Hyperbolic Neural Networks".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicPoint {
    /// Coordinates in the Poincaré ball (must satisfy ||coords|| < 1).
    pub coords: Vec<f64>,
    /// Learnable curvature — regions with high complexity get stronger curvature.
    pub curvature: f64,
}

impl HyperbolicPoint {
    /// Create the origin point (most general position).
    pub fn origin(dimensions: usize) -> Self {
        Self {
            coords: vec![0.0; dimensions],
            curvature: 1.0,
        }
    }

    /// Create a random point inside the Poincaré ball.
    /// Points are generated with moderate norm to avoid boundary instability.
    pub fn random(dimensions: usize, rng: &mut impl rand::Rng) -> Self {
        let mut coords: Vec<f64> = (0..dimensions)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        // Project into ball with max radius 0.9 to avoid boundary singularity
        let norm = euclidean_norm(&coords);
        if norm > 0.9 {
            let scale = 0.9 / norm;
            for c in &mut coords {
                *c *= scale;
            }
        }
        Self {
            coords,
            curvature: 1.0,
        }
    }

    /// Squared Euclidean norm of the coordinates.
    fn norm_sq(&self) -> f64 {
        self.coords.iter().map(|x| x * x).sum()
    }

    /// Euclidean norm of the coordinates.
    fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Hyperbolic distance in the Poincaré ball model.
    ///
    /// d(x, y) = (1/√c) · arcosh(1 + 2c · ||x-y||² / ((1 - c·||x||²)(1 - c·||y||²)))
    pub fn distance(&self, other: &HyperbolicPoint) -> f64 {
        let c = self.curvature;
        let diff_sq: f64 = self
            .coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let norm_x_sq = self.norm_sq();
        let norm_y_sq = other.norm_sq();

        let denom = (1.0 - c * norm_x_sq).max(1e-10) * (1.0 - c * norm_y_sq).max(1e-10);
        let arg = 1.0 + 2.0 * c * diff_sq / denom;

        // arcosh(x) = ln(x + sqrt(x²-1)), for x >= 1
        let arg = arg.max(1.0 + 1e-10); // numerical stability
        (1.0 / c.sqrt()) * (arg + (arg * arg - 1.0).sqrt()).ln()
    }

    /// Möbius addition in the Poincaré ball: x ⊕ y
    /// Used for translation and the exponential map.
    fn mobius_add(&self, y: &[f64]) -> Vec<f64> {
        let c = self.curvature;
        let x = &self.coords;
        let x_sq = self.norm_sq();
        let y_sq: f64 = y.iter().map(|v| v * v).sum();
        let xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        let num_coeff_x = 1.0 + 2.0 * c * xy + c * y_sq;
        let num_coeff_y = 1.0 - c * x_sq;
        let denom = (1.0 + 2.0 * c * xy + c * c * x_sq * y_sq).max(1e-10);

        x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (num_coeff_x * xi + num_coeff_y * yi) / denom)
            .collect()
    }

    /// Exponential map: project a tangent vector at this point onto the manifold.
    /// Used for gradient-based migration in hyperbolic space.
    pub fn exp_map(&self, tangent: &[f64]) -> HyperbolicPoint {
        let c = self.curvature;
        let sqrt_c = c.sqrt();
        let conformal = 1.0 - c * self.norm_sq();
        // If too close to the boundary, just return self (can't move further out)
        if conformal < 1e-6 {
            return self.clone();
        }
        let lambda = 2.0 / conformal;
        let t_norm = euclidean_norm(tangent).max(1e-10);

        let arg = (sqrt_c * lambda * t_norm / 2.0).min(18.0); // clamp to prevent tanh saturation
        let coeff = arg.tanh() / (sqrt_c * t_norm);
        let scaled: Vec<f64> = tangent.iter().map(|t| t * coeff).collect();

        let mut new_coords = self.mobius_add(&scaled);

        // Sanitize: replace NaN/Inf with zero, then clamp to ball
        for c in &mut new_coords {
            if !c.is_finite() { *c = 0.0; }
        }
        let norm = euclidean_norm(&new_coords);
        if norm >= 1.0 / sqrt_c {
            let max_norm = (1.0 / sqrt_c) - 1e-5;
            let scale = max_norm / norm;
            for c in &mut new_coords {
                *c *= scale;
            }
        }

        HyperbolicPoint {
            coords: new_coords,
            curvature: self.curvature,
        }
    }

    /// Logarithmic map: compute the tangent vector from this point to another.
    /// Inverse of exp_map.
    pub fn log_map(&self, target: &HyperbolicPoint) -> Vec<f64> {
        let c = self.curvature;
        let sqrt_c = c.sqrt();
        let lambda = 2.0 / (1.0 - c * self.norm_sq()).max(1e-10);

        // -x ⊕ y (Möbius subtraction)
        let neg_x: Vec<f64> = self.coords.iter().map(|v| -v).collect();
        let neg_self = HyperbolicPoint {
            coords: neg_x,
            curvature: c,
        };
        let diff = neg_self.mobius_add(&target.coords);
        let diff_norm = euclidean_norm(&diff).max(1e-10);

        let atanh_arg = (sqrt_c * diff_norm).min(0.999); // clamp to avoid atanh(1) = infinity
        let coeff = (2.0 / (lambda * sqrt_c)) * atanh_arg.atanh() / diff_norm;
        diff.iter().map(|d| d * coeff).collect()
    }

    /// How "specific" this point is (distance from origin).
    /// Points near the boundary are more specialized.
    pub fn specificity(&self) -> f64 {
        self.norm()
    }
}

/// Helper: Euclidean norm of a vector.
fn euclidean_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

impl fmt::Display for HyperbolicPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "H(")?;
        for (i, c) in self.coords.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.3}", c)?;
        }
        write!(f, "; c={:.2})", self.curvature)
    }
}

/// Legacy alias — all positions are now hyperbolic.
pub type Position = HyperbolicPoint;

/// Per-morphon frustration state for stochastic exploration.
///
/// When prediction error stagnates (desire is high but not improving),
/// frustration grows, increasing noise to escape local minima.
/// Biological analog: locus coeruleus noradrenaline burst under
/// sustained prediction failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrustrationState {
    /// Consecutive fast ticks where |PE_delta| < stagnation_threshold.
    pub stagnation_counter: u32,
    /// Normalized frustration level in [0.0, 1.0].
    pub frustration_level: f64,
    /// Current noise amplitude multiplier applied to potential (1.0 = baseline).
    pub noise_amplitude: f64,
    /// Whether this morphon is in active exploration mode.
    pub exploration_mode: bool,
    /// Previous prediction_error value (for computing delta).
    pub prev_pe: f64,
}

impl Default for FrustrationState {
    fn default() -> Self {
        Self {
            stagnation_counter: 0,
            frustration_level: 0.0,
            noise_amplitude: 1.0,
            exploration_mode: false,
            prev_pe: 0.0,
        }
    }
}

/// Configuration for frustration-driven stochastic exploration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrustrationConfig {
    /// Whether frustration-driven exploration is enabled.
    pub enabled: bool,
    /// Minimum PE change to consider non-stagnant.
    pub stagnation_threshold: f64,
    /// Stagnation counter value at which frustration saturates.
    pub saturation_steps: u32,
    /// Frustration level above which exploration_mode activates.
    pub exploration_threshold: f64,
    /// Maximum noise multiplier when fully frustrated.
    pub max_noise_multiplier: f64,
    /// Weight perturbation amplitude on medium tick (fraction of weight_max).
    pub weight_perturbation_scale: f64,
    /// Whether frustrated morphons can migrate without the desire>=0.3 gate.
    pub frustration_migration: bool,
    /// Frustration level above which migration direction is randomized.
    pub random_migration_threshold: f64,
}

impl Default for FrustrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            stagnation_threshold: 0.005,
            saturation_steps: 200,
            exploration_threshold: 0.3,
            max_noise_multiplier: 5.0,
            weight_perturbation_scale: 0.01,
            frustration_migration: true,
            random_migration_threshold: 0.6,
        }
    }
}

/// Configuration for the V2 dreaming engine — offline consolidation and self-optimization.
///
/// The dreaming engine runs during glacial ticks when the system is mature or idle.
/// It consolidates tagged synapses, refreshes stale connections, and broadcasts
/// curiosity signals to topology anomalies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamConfig {
    /// Whether dreaming is enabled.
    pub enabled: bool,
    /// Learning rate multiplier during dream mode (reduced vs waking).
    pub dream_learning_rate: f64,
    /// Minimum tag strength to select a synapse for dream consolidation.
    pub dream_tag_threshold: f64,
    /// Maximum number of synapses to process per dream cycle.
    pub max_dream_synapses: usize,
    /// Synapse age threshold for self-optimization reset candidates.
    pub stale_synapse_age: u64,
    /// Maximum usage count below which a stale synapse is a reset candidate.
    pub stale_usage_threshold: u64,
    /// Weight reset magnitude for stale synapses (fraction of weight_max).
    pub reset_weight_scale: f64,
    /// Curiosity signal strength for topology anomalies.
    pub curiosity_signal_strength: f64,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dream_learning_rate: 0.3,
            dream_tag_threshold: 0.2,
            max_dream_synapses: 50,
            stale_synapse_age: 5000,
            stale_usage_threshold: 3,
            reset_weight_scale: 0.1,
            curiosity_signal_strength: 0.15,
        }
    }
}

/// Ring buffer for activity history tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingBuffer {
    data: Vec<f64>,
    head: usize,
    len: usize,
}

impl Default for RingBuffer {
    fn default() -> Self {
        Self::new(10)
    }
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            head: 0,
            len: 0,
        }
    }

    /// Overwrite the most recently pushed value (rewind one position).
    /// Used by k-WTA to correct activity_history when a neuron's firing
    /// is suppressed after morphon.step() already pushed 1.0.
    pub fn overwrite_last(&mut self, value: f64) {
        if self.len > 0 {
            let last = if self.head == 0 { self.data.len() - 1 } else { self.head - 1 };
            self.data[last] = value;
        }
    }

    pub fn push(&mut self, value: f64) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.data.len();
        if self.len < self.data.len() {
            self.len += 1;
        }
    }

    pub fn mean(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        self.data[..self.len].iter().sum::<f64>() / self.len as f64
    }

    pub fn variance(&self) -> f64 {
        if self.len < 2 {
            return 0.0;
        }
        let mean = self.mean();
        self.data[..self.len]
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (self.len - 1) as f64
    }

    pub fn max(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        self.data[..self.len]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        // Return items in chronological order
        let start = if self.len < self.data.len() {
            0
        } else {
            self.head
        };
        let cap = self.data.len();
        (0..self.len).map(move |i| &self.data[(start + i) % cap])
    }
}
