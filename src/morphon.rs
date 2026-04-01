//! The Morphon — fundamental autonomous compute unit of MI.

use crate::justification::SynapticJustification;
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
    /// Continuous consolidation level (0.0 = fully plastic, 1.0 = fully consolidated).
    /// Weight updates are scaled by `(1.0 - consolidation_level * 0.9)`.
    #[serde(default)]
    pub consolidation_level: f64,
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

    /// V3: Provenance record — why this synapse was formed and what reinforced it.
    #[serde(default)]
    pub justification: Option<SynapticJustification>,
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
            consolidation_level: 0.0,
            age: 0,
            usage_count: 0,
            pre_trace: 0.0,
            post_trace: 0.0,
            justification: None,
        }
    }

    /// Create a synapse with a justification record (V3).
    pub fn new_justified(weight: f64, justification: SynapticJustification) -> Self {
        Self {
            justification: Some(justification),
            ..Self::new(weight)
        }
    }

    pub fn with_delay(mut self, delay: f64) -> Self {
        self.delay = delay;
        self
    }
}

fn default_plasticity_rate() -> f64 { 1.0 }
fn default_cluster_overhead() -> f64 { 0.0005 }
fn default_reward_for_output() -> f64 { 0.05 }

/// V3 Metabolic Budget — energy earned through utility, not flat regeneration.
///
/// Morphons pay a base cost plus per-synapse maintenance each step.
/// Energy is earned by reducing prediction error (utility) instead of
/// unconditional regeneration. This forces the system toward minimal
/// topology at maximal performance — ideal for edge deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetabolicConfig {
    /// Base energy cost per step for being alive.
    pub base_cost: f64,
    /// Additional cost per outgoing synapse per step.
    pub synapse_cost: f64,
    /// Energy earned per unit of prediction error reduction.
    pub utility_reward: f64,
    /// Small unconditional trickle to prevent total starvation of quiet morphons.
    pub basal_regen: f64,
    /// Extra firing cost on top of the base cost when the morphon spikes.
    pub firing_cost: f64,

    /// V3: Additional cost per step for morphons in a fused cluster.
    #[serde(default = "default_cluster_overhead")]
    pub cluster_overhead_per_tick: f64,

    /// V3: Energy bonus for motor morphons when output is rewarded.
    #[serde(default = "default_reward_for_output")]
    pub reward_for_successful_output: f64,

    /// V3: Energy bonus for successful epistemic verification (Phase 2).
    #[serde(default)]
    pub reward_for_verification: f64,
}

impl Default for MetabolicConfig {
    fn default() -> Self {
        Self {
            base_cost: 0.001,
            synapse_cost: 0.0001,  // reduced 5× — was starving hidden layer
            utility_reward: 0.02,
            basal_regen: 0.005,    // generous trickle — silent morphons survive for anchor/sail
            firing_cost: 0.002,    // reduced 2× — firing should be cheap
            cluster_overhead_per_tick: 0.0005,
            reward_for_successful_output: 0.05,
            reward_for_verification: 0.0, // Phase 2 placeholder
        }
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
    /// Plasticity rate — scales all weight updates for this morphon's synapses.
    /// "Anchor" morphons (low plasticity) provide stable features for the readout.
    /// "Sail" morphons (high plasticity) explore the state space.
    /// Initialized from log-normal distribution; modulated by readout importance (H2).
    #[serde(default = "default_plasticity_rate")]
    pub plasticity_rate: f64,

    // === V2: Frustration-Driven Exploration ===
    /// Per-morphon frustration state for escaping local minima via adaptive noise.
    #[serde(default)]
    pub frustration: FrustrationState,
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
            homeostatic_setpoint: 0.15,
            feedback_signal: 0.0,
            plasticity_rate: 1.0, // default; overwritten by developmental program
            frustration: FrustrationState::default(),
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
            homeostatic_setpoint: 0.15,
            feedback_signal: 0.0,
            // Child inherits parent's plasticity with mutation — anchors beget anchors
            plasticity_rate: (self.plasticity_rate + rng.random_range(-0.1..0.1)).clamp(0.1, 2.0),
            frustration: FrustrationState::default(),
        }
    }

    /// Process accumulated input and determine if the Morphon fires.
    ///
    /// `synapse_count`: number of outgoing synapses (for metabolic maintenance cost).
    /// `metabolic`: V3 metabolic budget configuration.
    pub fn step(&mut self, dt: f64, synapse_count: usize, metabolic: &MetabolicConfig, frustration_config: &FrustrationConfig, threshold_bias: f64) {
        self.age += 1;

        // Refractory period
        if self.refractory_timer > 0.0 {
            self.refractory_timer -= dt;
            self.fired = false;
            self.input_accumulator = 0.0;
            return;
        }

        // Update potential with leaky integration + spontaneous noise
        // Motor morphons have FULL leak (reset to 0 each step) so they reflect
        // only the current input, not accumulated history. Without this, noise
        // accumulates over hundreds of steps and drifts motors to ±clamp.
        let leak_rate = if self.cell_type == CellType::Motor { 1.0 } else { 0.1 };
        // Pseudo-random noise centered at zero.
        // Motor morphons get reduced noise (0.02) to prevent accumulation drift.
        // Other types get standard noise (0.1) for baseline activity.
        let noise_raw = (self.id.wrapping_mul(self.age).wrapping_add(7919) % 1000) as f64 / 1000.0;
        let noise_scale = match self.cell_type {
            CellType::Motor => 0.0,
            // Reduced noise for Associative morphons — preserves pattern stability
            // for readout learning while frustration scaling still drives exploration.
            CellType::Associative => 0.02 * self.frustration.noise_amplitude,
            _ => 0.1 * self.frustration.noise_amplitude,
        };
        let noise = (noise_raw - 0.5) * noise_scale;
        self.prev_potential = self.potential;
        self.potential = self.potential * (1.0 - leak_rate * dt) + self.input_accumulator + noise;
        // Clamp potential to prevent saturation from dense connectivity.
        // [-10, 10] is wide enough that motor morphons don't hit the wall easily
        // but still prevents overflow from 300+ simultaneous inputs.
        self.potential = self.potential.clamp(-10.0, 10.0);
        if !self.potential.is_finite() {
            self.potential = 0.0;
        }

        // Apply activation function to determine output
        let activation = self.activation_fn.apply(self.potential);

        // Fire if above threshold (+ Endoquilibrium bias) and has energy to spend
        self.fired = activation > (self.threshold + threshold_bias) && self.energy > 0.0;
        if self.fired {
            self.refractory_timer = 1.0; // refractory period (1 step)
            self.energy -= metabolic.firing_cost;
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

        // V2: Update frustration state — detect PE stagnation and scale noise.
        if frustration_config.enabled {
            let pe_delta = (self.prediction_error - self.frustration.prev_pe).abs();
            self.frustration.prev_pe = self.prediction_error;

            let prev_counter = self.frustration.stagnation_counter;
            if pe_delta < frustration_config.stagnation_threshold && self.desire > 0.1 {
                self.frustration.stagnation_counter = self.frustration.stagnation_counter.saturating_add(1);
            } else {
                // Fast decay on improvement — 5:1 ratio means recovery is quick
                self.frustration.stagnation_counter = self.frustration.stagnation_counter.saturating_sub(5);
            }

            // Only recompute derived values when counter changed
            if self.frustration.stagnation_counter != prev_counter {
                let ratio = self.frustration.stagnation_counter as f64 / frustration_config.saturation_steps as f64;
                self.frustration.frustration_level = (ratio * 3.0).tanh();
                self.frustration.noise_amplitude = 1.0 + self.frustration.frustration_level
                    * (frustration_config.max_noise_multiplier - 1.0);
                self.frustration.exploration_mode =
                    self.frustration.frustration_level > frustration_config.exploration_threshold;
            }
        }

        // Homeostatic threshold regulation
        // Upper clamp respects the activation function's output range — without this,
        // threshold can exceed max activation output, permanently silencing the morphon.
        let actual_rate = self.activity_history.mean();
        self.threshold += 0.01 * (actual_rate - self.homeostatic_setpoint);
        let max_threshold = self.activation_fn.max_output();
        self.threshold = self.threshold.clamp(0.05, max_threshold);

        // Division pressure accumulates from two sources:
        // 1. Active morphons accumulate division pressure.
        //    With k-WTA capped at 20 winners out of ~300, the max firing rate
        //    for any morphon is ~7%. Gate at 3% so consistent winners can divide.
        // 2. High DFA feedback error — "error-driven proliferation"
        if actual_rate > 0.03 {
            self.division_pressure += 0.005;
        } else {
            self.division_pressure = (self.division_pressure - 0.003).max(0.0);
        }
        // DFA-driven pressure: strong feedback signal means this morphon is in
        // a region where the output is wrong. More capacity needed here.
        if self.feedback_signal.abs() > 0.1 {
            self.division_pressure += self.feedback_signal.abs() * 0.005;
        }

        // V3 Metabolic Budget: energy earned through utility, not flat regen.
        // 1. Base maintenance cost (just being alive)
        self.energy -= metabolic.base_cost;
        // 2. Per-synapse maintenance cost (connections are expensive)
        self.energy -= synapse_count as f64 * metabolic.synapse_cost;
        // 3. Utility reward: earn energy by reducing prediction error
        //    PE decrease means the morphon is doing useful work.
        let pe_delta = self.prev_potential.abs() - self.prediction_error;
        if pe_delta > 0.0 {
            self.energy += pe_delta * metabolic.utility_reward;
        }
        // 4. Basal trickle: small unconditional regen prevents total starvation
        //    of quiet morphons that may become useful later.
        self.energy += metabolic.basal_regen;
        // Clamp to [0, 1]
        self.energy = self.energy.clamp(0.0, 1.0);

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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_morphon() -> Morphon {
        Morphon::new(1, HyperbolicPoint::origin(4))
    }

    fn default_frustration_config() -> FrustrationConfig {
        FrustrationConfig::default()
    }

    #[test]
    fn frustration_accumulates_on_stagnant_pe() {
        let mut m = make_morphon();
        m.desire = 0.5; // pre-set above the 0.1 gate
        let metabolic = MetabolicConfig::default();
        let fc = default_frustration_config();

        // Feed constant input — PE delta will be small, frustration should build.
        for _ in 0..300 {
            m.input_accumulator = 0.5;
            m.step(1.0, 0, &metabolic, &fc, 0.0);
        }

        assert!(m.frustration.stagnation_counter > 0, "stagnation counter should grow");
        assert!(m.frustration.frustration_level > 0.0, "frustration should be non-zero");
        assert!(m.frustration.noise_amplitude > 1.0, "noise amplitude should exceed baseline");
    }

    #[test]
    fn frustration_resets_on_pe_improvement() {
        let mut m = make_morphon();
        m.desire = 0.2;
        let metabolic = MetabolicConfig::default();
        let fc = default_frustration_config();

        // Build up frustration
        for _ in 0..200 {
            m.input_accumulator = 0.5;
            m.step(1.0, 0, &metabolic, &fc, 0.0);
        }
        let frustrated_level = m.frustration.frustration_level;
        assert!(frustrated_level > 0.0);

        // Now inject varying input — PE changes, stagnation counter should decay
        for i in 0..100 {
            m.input_accumulator = (i as f64) * 0.1;
            m.step(1.0, 0, &metabolic, &fc, 0.0);
        }

        assert!(
            m.frustration.frustration_level < frustrated_level,
            "frustration should decrease after PE changes"
        );
    }

    #[test]
    fn frustration_noise_amplitude_scaling() {
        let state = FrustrationState::default();
        assert!((state.noise_amplitude - 1.0).abs() < f64::EPSILON,
            "baseline noise amplitude should be 1.0");

        let mut m = make_morphon();
        // Manually set high frustration
        m.frustration.frustration_level = 1.0;
        m.frustration.noise_amplitude = 1.0 + 1.0 * (FrustrationConfig::default().max_noise_multiplier - 1.0);
        assert!((m.frustration.noise_amplitude - 5.0).abs() < f64::EPSILON,
            "max frustration should give max_noise_multiplier");
    }

    #[test]
    fn motor_morphons_ignore_frustration_noise() {
        let mut m = make_morphon();
        m.differentiate(CellType::Motor);
        m.desire = 0.2;
        m.frustration.frustration_level = 1.0;
        m.frustration.noise_amplitude = 5.0;
        let metabolic = MetabolicConfig::default();
        let fc = default_frustration_config();

        // Motor morphons have noise_scale = 0.0 regardless of frustration
        let potential_before = m.potential;
        m.input_accumulator = 0.0;
        m.step(1.0, 0, &metabolic, &fc, 0.0);
        // Motor has full leak (leak_rate=1.0), so potential = 0*(1-1) + 0 + 0 = 0
        // No noise should be added
        assert!((m.potential - 0.0).abs() < f64::EPSILON,
            "motor morphon potential should not have noise");
    }
}
