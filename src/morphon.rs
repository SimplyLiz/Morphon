//! The Morphon — fundamental autonomous compute unit of MI.

use crate::types::*;
use serde::{Deserialize, Serialize};

/// A Synapse connecting two Morphons.
///
/// Includes both fast eligibility traces (τ ~ 100ms) for standard three-factor
/// learning, and slow synaptic tags (τ ~ minutes) for delayed reward via the
/// Tag-and-Capture mechanism (Frey & Morris 1997).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    /// Connection weight.
    pub weight: f64,
    /// Signal delay (learnable).
    pub delay: f64,
    /// Fast eligibility trace — transient marking for three-factor learning (τ ~ 100ms).
    pub eligibility: f64,
    /// Slow synaptic tag — long-lived marking for delayed reward (τ ~ minutes).
    pub tag: f64,
    /// Strength of the tag (proportional to Hebbian coincidence at tagging time).
    pub tag_strength: f64,
    /// Whether this synapse has been captured (tag converted to permanent weight change).
    pub consolidated: bool,
    /// Age in simulation steps.
    pub age: u64,
    /// How often this synapse has been activated.
    pub usage_count: u64,
    /// Pre-synaptic trace — decaying memory of recent pre-synaptic spikes.
    /// Incremented on pre-spike, decays exponentially with tau_trace.
    pub pre_trace: f64,
    /// Post-synaptic trace — decaying memory of recent post-synaptic spikes.
    /// Incremented on post-spike, decays exponentially with tau_trace.
    pub post_trace: f64,
}

impl Synapse {
    pub fn new(weight: f64) -> Self {
        Self {
            weight,
            delay: 1.0,
            eligibility: 0.0,
            tag: 0.0,
            tag_strength: 0.0,
            consolidated: false,
            age: 0,
            usage_count: 0,
            pre_trace: 0.0,
            post_trace: 0.0,
        }
    }

    pub fn with_delay(mut self, delay: f64) -> Self {
        self.delay = delay;
        self
    }
}

/// The Morphon — an autonomous agent in the MI network.
///
/// Not a classical neuron, but a self-governing compute unit with identity,
/// internal state, connectivity, learning state, and lifecycle management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Morphon {
    // === Identity ===
    /// Unique identifier.
    pub id: MorphonId,
    /// Position in the N-dimensional information space.
    pub position: Position,
    /// Lineage — which parent Morphon this was born from.
    pub lineage: Option<MorphonId>,
    /// How many divisions since the seed.
    pub generation: Generation,

    // === Cell Type & Differentiation ===
    /// Current functional type.
    pub cell_type: CellType,
    /// 0.0 = pluripotent (stem), 1.0 = terminally differentiated.
    pub differentiation_level: f64,
    /// Current activation function (changes with differentiation).
    pub activation_fn: ActivationFn,
    /// Which neuromodulators this Morphon responds to.
    pub receptors: ReceptorSet,

    // === Internal State ===
    /// Membrane potential analog.
    pub potential: f64,
    /// Adaptive firing threshold (homeostatically regulated).
    pub threshold: f64,
    /// Refractory timer — cooldown after firing.
    pub refractory_timer: f64,
    /// Running difference between expectation and input.
    pub prediction_error: f64,
    /// Long-term mean of prediction error (drives migration).
    pub desire: f64,

    // === Learning State ===
    /// Activity history for structural plasticity decisions.
    pub activity_history: RingBuffer,
    /// Whether this Morphon fired in the current step.
    pub fired: bool,
    /// Accumulated input for the current step.
    pub input_accumulator: f64,
    /// Previous potential (for prediction error computation).
    pub prev_potential: f64,

    // === Lifecycle ===
    /// Age in simulation steps.
    pub age: u64,
    /// Metabolic energy budget.
    pub energy: f64,
    /// Pressure to divide (accumulates with overload).
    pub division_pressure: f64,

    // === Fusion ===
    /// If part of a fused cluster, its ID.
    pub fused_with: Option<ClusterId>,
    /// 1.0 = fully autonomous, 0.0 = fully integrated in cluster.
    pub autonomy: f64,

    // === Homeostatic Protection ===
    /// Cooldown timer after migration — prevents topological instability.
    pub migration_cooldown: f64,
    /// Direct Feedback Alignment signal — neuron-specific modulation from
    /// output error projected through fixed random weights. Replaces global
    /// reward broadcast for hidden layer credit assignment.
    pub feedback_signal: f64,
    /// Target firing rate for synaptic scaling (homeostatic anchor).
    pub homeostatic_setpoint: f64,
}

impl Morphon {
    /// Create a new stem-cell Morphon at the given position.
    pub fn new(id: MorphonId, position: Position) -> Self {
        Self {
            id,
            position,
            lineage: None,
            generation: 0,
            cell_type: CellType::Stem,
            differentiation_level: 0.0,
            activation_fn: ActivationFn::Sigmoid,
            receptors: default_receptors(CellType::Stem),
            potential: 0.0,
            threshold: 0.3,
            refractory_timer: 0.0,
            prediction_error: 0.0,
            desire: 0.0,
            activity_history: RingBuffer::new(100),
            fired: false,
            input_accumulator: 0.0,
            prev_potential: 0.0,
            age: 0,
            energy: 1.0,
            division_pressure: 0.0,
            fused_with: None,
            autonomy: 1.0,
            migration_cooldown: 0.0,
            homeostatic_setpoint: 0.1,
            feedback_signal: 0.0,
        }
    }

    /// Create a child Morphon from this one (mitosis).
    /// The child inherits state with stochastic mutations.
    pub fn divide(&self, child_id: MorphonId, rng: &mut impl rand::Rng) -> Self {
        // Use exponential map for offset in hyperbolic space
        let tangent: Vec<f64> = (0..self.position.coords.len())
            .map(|_| rng.random_range(-0.1..0.1))
            .collect();
        let child_position = self.position.exp_map(&tangent);

        let mutation_scale = 0.05;

        Self {
            id: child_id,
            position: child_position,
            lineage: Some(self.id),
            generation: self.generation + 1,
            // Child starts as Stem (asymmetric division: parent keeps identity, child is plastic)
            cell_type: CellType::Stem,
            differentiation_level: 0.0,
            activation_fn: ActivationFn::Sigmoid,
            receptors: default_receptors(CellType::Stem),
            potential: 0.0,
            threshold: self.threshold + rng.random_range(-mutation_scale..mutation_scale),
            refractory_timer: 0.0,
            prediction_error: 0.0,
            desire: 0.0,
            activity_history: RingBuffer::new(100),
            fired: false,
            input_accumulator: 0.0,
            prev_potential: 0.0,
            age: 0,
            energy: self.energy * 0.5, // energy split between parent and child
            division_pressure: 0.0,
            fused_with: None,
            autonomy: 1.0,
            migration_cooldown: 0.0,
            homeostatic_setpoint: 0.1,
            feedback_signal: 0.0,
        }
    }

    /// Process accumulated input and determine if the Morphon fires.
    pub fn step(&mut self, dt: f64) {
        self.age += 1;

        // Refractory period
        if self.refractory_timer > 0.0 {
            self.refractory_timer -= dt;
            self.fired = false;
            self.input_accumulator = 0.0;
            return;
        }

        // Update potential with leaky integration + spontaneous noise
        // Noise ensures baseline Hebbian coincidences even without external input
        let leak_rate = 0.1;
        // Pseudo-random noise centered at zero (range [-0.05, +0.05])
        let noise_raw = (self.id.wrapping_mul(self.age).wrapping_add(7919) % 1000) as f64 / 1000.0;
        let noise = (noise_raw - 0.5) * 0.1;
        self.prev_potential = self.potential;
        self.potential = self.potential * (1.0 - leak_rate * dt) + self.input_accumulator + noise;
        // Clamp potential to prevent saturation from dense connectivity.
        // Without this, 300+ inputs each adding 0.07 = 21.0, sigmoid(21) ≈ 1.0 always.
        self.potential = self.potential.clamp(-5.0, 5.0);
        if !self.potential.is_finite() {
            self.potential = 0.0;
        }

        // Apply activation function to determine output
        let activation = self.activation_fn.apply(self.potential);

        // Fire if above threshold and has energy to spend
        self.fired = activation > self.threshold && self.energy > 0.0;
        if self.fired {
            self.refractory_timer = 1.0; // refractory period (1 step)
            self.energy = (self.energy - 0.005).max(0.0); // firing costs energy, floor at 0
            self.activity_history.push(1.0);
        } else {
            self.activity_history.push(0.0);
        }

        // Update prediction error
        let expected = self.prev_potential; // simple prediction: previous state
        self.prediction_error = (self.potential - expected).abs();

        // Update desire (exponential moving average of prediction error)
        let desire_alpha = 0.01;
        self.desire = self.desire * (1.0 - desire_alpha) + self.prediction_error * desire_alpha;

        // Homeostatic threshold regulation
        // Upper clamp respects the activation function's output range — without this,
        // threshold can exceed max activation output, permanently silencing the morphon.
        let actual_rate = self.activity_history.mean();
        self.threshold += 0.01 * (actual_rate - self.homeostatic_setpoint);
        let max_threshold = self.activation_fn.max_output();
        self.threshold = self.threshold.clamp(0.05, max_threshold);

        // Division pressure accumulates when chronically overloaded
        if actual_rate > 0.5 {
            self.division_pressure += 0.01;
        } else {
            self.division_pressure = (self.division_pressure - 0.005).max(0.0);
        }

        // Energy regeneration (slow), clamped to [0, 1]
        self.energy = (self.energy + 0.001).clamp(0.0, 1.0);

        // Tick down migration cooldown
        if self.migration_cooldown > 0.0 {
            self.migration_cooldown -= dt;
        }

        // Reset accumulator for next step
        self.input_accumulator = 0.0;
    }

    /// Check if this Morphon should undergo apoptosis (programmed death).
    pub fn should_apoptose(&self) -> bool {
        self.age > 1000
            && self.energy < 0.1
            && self.activity_history.mean() < 0.01
            && self.fused_with.is_none()
    }

    /// Check if this Morphon is ready to divide.
    pub fn should_divide(&self, division_threshold: f64) -> bool {
        self.division_pressure > division_threshold && self.energy > 0.3
    }

    /// Differentiate towards a target cell type.
    /// Returns true if differentiation occurred.
    pub fn differentiate(&mut self, target: CellType) -> bool {
        if self.differentiation_level >= 1.0 {
            return false; // terminally differentiated
        }

        let rate = match self.cell_type {
            CellType::Stem => 0.05,        // stems differentiate easily
            _ => 0.01,                      // transdifferentiation is slower
        };

        self.cell_type = target;
        self.differentiation_level = (self.differentiation_level + rate).min(1.0);
        self.activation_fn = ActivationFn::for_cell_type(target);
        self.receptors = default_receptors(target);
        true
    }

    /// Dedifferentiate — return towards stem-like flexibility.
    /// Triggered by sustained high prediction error + arousal.
    pub fn dedifferentiate(&mut self) {
        if self.differentiation_level <= 0.0 {
            return;
        }
        self.differentiation_level = (self.differentiation_level - 0.02).max(0.0);
        if self.differentiation_level < 0.2 {
            self.cell_type = CellType::Stem;
            self.activation_fn = ActivationFn::Sigmoid;
            self.receptors = default_receptors(CellType::Stem);
        }
    }
}
