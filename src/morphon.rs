//! The Morphon — fundamental autonomous compute unit of MI.

use crate::justification::SynapticJustification;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A Synapse connecting two Morphons.
///
/// Includes both fast eligibility traces (τ ~ 100ms) for standard three-factor
/// learning, and slow synaptic tags (τ ~ minutes) for delayed reward via the
/// Tag-and-Capture mechanism (Frey & Morris 1997).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    /// Connection weight.
    pub weight: f64,
    /// Signal delay — initialized from hyperbolic distance, then modulated by myelination.
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

    /// Slow-decaying activity trace for myelination gating (τ=200 medium ticks ≈ 2000 sim steps).
    /// Bumped in apply_weight_update (medium path), read in update_myelination (slow path).
    /// Solves the timing mismatch where fast traces (eligibility τ=20, pre/post τ=10)
    /// decay to zero between slow ticks.
    #[serde(default)]
    pub activity_trace: f64,

    /// Activity-dependent myelination level (0.0 = unmyelinated, 1.0 = fully myelinated).
    /// Increases with consolidation × usage on slow timescale. Reduces effective signal
    /// delay, giving established pathways a temporal advantage in local competition.
    #[serde(default)]
    pub myelination: f64,

    /// V3: Provenance record — why this synapse was formed and what reinforced it.
    #[serde(default)]
    pub justification: Option<SynapticJustification>,

    /// Step number at which traces/eligibility/tag were last updated. Used by
    /// sparse eligibility updates to fast-forward exponential decay over skipped
    /// medium ticks. Mathematically equivalent to per-step decay (exponentials
    /// compose) — the synapse only needs visiting when pre or post is active.
    #[serde(default)]
    pub last_update_step: u64,

    /// V6: Rolling EMA of |eligibility| × |reward_channel| — forward-reference density.
    /// Measures whether this synapse's activity correlates with downstream reward.
    /// Synapses with high reward correlation are protected from pruning even if their
    /// weight is currently low (they carry credit-assignment signal).
    /// EMA α=0.01 (slow decay, window ~100 medium ticks).
    #[serde(default)]
    pub reward_correlation: f64,

    /// Phase 4 (ANCS-Core): Forward-importance EMA — tracks how much reward-correlated
    /// credit flows *through* this synapse to downstream morphons.
    /// Complements `reward_correlation` (backward: was this synapse rewarded?) with a
    /// forward signal: does reward flow through this synapse to its targets?
    /// EMA α=0.05 (faster decay than reward_correlation — responds to recent reward flow).
    /// Synapses with high forward_importance are protected from pruning even when
    /// their weight is weak or usage_count is low.
    #[serde(default)]
    pub forward_importance: f64,
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
            activity_trace: 0.0,
            myelination: 0.0,
            justification: None,
            last_update_step: 0,
            reward_correlation: 0.0,
            forward_importance: 0.0,
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

    /// Effective signal delay accounting for myelination.
    /// Fully myelinated synapses deliver up to 5× faster than unmyelinated ones.
    pub fn effective_delay(&self) -> f64 {
        let speed_factor = 1.0 + self.myelination * 4.0; // 1× to 5× faster
        (self.delay / speed_factor).max(0.5) // minimum 0.5 tick delay
    }

    /// Update myelination via treadmilling dynamics with neuromodulatory gating.
    ///
    /// Biology (Auer et al. 2019): myelin thickness is a dynamic equilibrium —
    /// new layers added at inner tongue (activity-driven), old layers removed
    /// at outer tongue (constitutive). Three interacting processes:
    ///
    /// 1. **Wrapping** (additive): local activity × consolidation, gated by
    ///    arousal sensitivity. Reward biases toward tagged synapses.
    /// 2. **Treadmill removal** (subtractive): constitutive slow removal,
    ///    representing outer-tongue turnover (thrombin/neurofascin 155).
    /// 3. **Stress inhibition**: energy pressure suppresses wrapping;
    ///    removal continues → net demyelination under sustained stress.
    pub fn update_myelination(&mut self, dt: f64, ctx: &MyelinationContext) {
        let tau_wrap = 5000.0;      // wrapping timescale (slow)
        let tau_removal = 50000.0;  // treadmill removal — slower than wrapping so
                                    // equilibrium settles near target, not far below it

        // --- Wrapping (inner tongue addition) ---
        // Local signal: consolidated + active pathway → OPC process wraps.
        let activity_signal = if self.activity_trace > 0.1 { 1.0 } else { 0.0 };
        let local_drive = self.consolidation_level * activity_signal;

        // Neuromodulatory gating of OPC sensitivity (threshold shift, not rate).
        // High arousal: OPCs respond to weaker activity → faster myelination.
        // Maps to: norepinephrine driving OPC differentiation (Barron & Kim 2023).
        let sensitivity = 1.0 + ctx.arousal * 0.5;

        // Reward bias: recently-tagged synapses get a myelination boost.
        // Maps to: dopamine D1/D2 on OL lineage cells (Monje 2024).
        let reward_bias = if self.tag > 0.1 { ctx.reward * 0.3 } else { 0.0 };

        // Stress gate: energy pressure slows wrapping (doesn't block it).
        // Maps to: cortisol down-regulating IGF-1 → OPC differentiation slowed.
        // 0.5× multiplier: at full energy pressure (1.0), wrapping is halved, not zeroed.
        let stress_gate = (1.0 - ctx.energy_pressure * 0.5).max(0.2);

        let wrap_target = (local_drive * sensitivity + reward_bias) * stress_gate;

        // --- Treadmill removal (outer tongue turnover) ---
        // Constitutive, proportional to current level. Always running.
        let removal = self.myelination * (dt / tau_removal);

        // --- Net change ---
        let wrapping = (wrap_target - self.myelination) * (dt / tau_wrap);
        self.myelination = (self.myelination + wrapping - removal).clamp(0.0, 1.0);
    }
}

/// Neuromodulatory context for myelination decisions.
/// Passed from System on the slow path — the synapse doesn't know about
/// global state, so the caller provides what the "oligodendrocyte" senses.
#[derive(Debug, Clone, Copy)]
pub struct MyelinationContext {
    /// Arousal (norepinephrine) — promotes OPC differentiation.
    /// High arousal → OPCs more sensitive to activity → faster myelination.
    pub arousal: f64,
    /// Reward (dopamine) — biases myelination toward recently-rewarded pathways.
    pub reward: f64,
    /// Energy pressure (0.0 = abundant, 1.0 = depleted).
    /// High pressure → inhibit myelination (cortisol/stress analog).
    pub energy_pressure: f64,
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

    /// V2: Base cost reduction factor for fused morphons [0.0, 1.0].
    /// Fused morphons pay `base_cost * (1.0 - cluster_base_cost_reduction)`.
    #[serde(default = "default_cluster_reduction")]
    pub cluster_base_cost_reduction: f64,

    /// V2: Energy drawn from cluster pool per tick by each fused morphon.
    #[serde(default = "default_cluster_draw")]
    pub cluster_energy_draw_per_tick: f64,

    /// Energy earned by morphons that fire when reward signal is positive.
    /// Hubs fire for all inputs → reward is ±symmetric → net zero income.
    /// Specialists fire for correct class → mostly positive → positive income.
    #[serde(default)]
    pub reward_energy_coefficient: f64,

    /// Superlinear firing cost factor. Firing cost scales as:
    /// `firing_cost * (1.0 + activity_rate * superlinear_firing_factor)`
    /// At 50% rate with factor=2.0, cost is 3× base. At 5%, cost is 1.1× base.
    #[serde(default)]
    pub superlinear_firing_factor: f64,
}

fn default_cluster_reduction() -> f64 { 0.4 }
fn default_cluster_draw() -> f64 { 0.0003 }

impl Default for MetabolicConfig {
    fn default() -> Self {
        Self {
            base_cost: 0.001,
            synapse_cost: 0.0001,  // reduced 5× — was starving hidden layer
            utility_reward: 0.02,
            basal_regen: 0.003,    // tight trickle — hubs with superlinear cost go net-negative
            firing_cost: 0.002,    // reduced 2× — firing should be cheap
            cluster_overhead_per_tick: 0.0005,
            reward_for_successful_output: 0.05,
            reward_for_verification: 0.0, // Phase 2 feature — disabled in Phase 1
            cluster_base_cost_reduction: 0.4,
            cluster_energy_draw_per_tick: 0.0003,
            reward_energy_coefficient: 0.0,
            superlinear_firing_factor: 0.0,
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

    // === Metabolic duration tracking ===
    /// Consecutive ticks below apoptosis energy threshold.
    /// Apoptosis requires sustained energy deficit, not a transient dip.
    #[serde(default)]
    pub ticks_below_energy_threshold: u32,

    // === V2: Frustration-Driven Exploration ===
    /// Per-morphon frustration state for escaping local minima via adaptive noise.
    #[serde(default)]
    pub frustration: FrustrationState,

    // === V2: Sub-Morphon Plasticity (Adaptive Receptors) ===
    /// Continuous receptor sensitivity per modulator channel [0.01, 2.0].
    /// Modulates amplitude of neuromodulatory signals during weight updates.
    /// Adapts based on correlation between modulation and PE reduction.
    #[serde(default)]
    pub receptor_sensitivity: HashMap<ModulatorType, f64>,
    /// Recent modulation signals per channel for adaptation tracking.
    #[serde(default)]
    pub recent_modulation: HashMap<ModulatorType, RingBuffer>,
    /// Recent prediction error deltas for adaptation tracking.
    #[serde(default)]
    pub recent_pe_deltas: RingBuffer,

    // === V2: Astrocytic Gate (AGMP-inspired) ===
    /// Slow state variable gating excitatory plasticity. Updated every medium tick.
    /// gate = sigmoid(astrocytic_state - threshold_a), multiplied into weight updates.
    #[serde(default = "default_astrocytic_state")]
    pub astrocytic_state: f64,

    // === V2: Per-Morphon Noise ===
    /// Base noise amplitude for membrane potential perturbation.
    /// Initialized from cell type (Motor=0.0, Associative=0.08, others=0.1),
    /// inherited with mutation in divide(), updated on differentiation.
    /// Final noise = intrinsic_noise × frustration.noise_amplitude.
    #[serde(default = "default_intrinsic_noise")]
    pub intrinsic_noise: f64,
}

fn default_astrocytic_state() -> f64 { 1.0 }
fn default_intrinsic_noise() -> f64 { 0.1 }

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
            ticks_below_energy_threshold: 0,
            frustration: FrustrationState::default(),
            receptor_sensitivity: default_receptor_sensitivity(CellType::Stem),
            recent_modulation: HashMap::new(),
            recent_pe_deltas: RingBuffer::new(10),
            astrocytic_state: 1.0,
            intrinsic_noise: crate::types::intrinsic_noise_for(CellType::Stem),
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
            ticks_below_energy_threshold: 0,
            frustration: FrustrationState::default(),
            // V2: Child inherits parent's receptor sensitivities with small mutation
            receptor_sensitivity: self.receptor_sensitivity.iter().map(|(&ch, &val)| {
                let mutated = (val + rng.random_range(-0.05..0.05)).clamp(0.01, 2.0);
                (ch, mutated)
            }).collect(),
            recent_modulation: HashMap::new(),
            recent_pe_deltas: RingBuffer::new(10),
            astrocytic_state: self.astrocytic_state,
            // Child inherits parent's noise with mutation — noisy parents beget noisy children
            intrinsic_noise: (self.intrinsic_noise + rng.random_range(-0.01..0.01)).clamp(0.0, 0.2),
        }
    }

    /// Process accumulated input and determine if the Morphon fires.
    ///
    /// `synapse_maintenance_cost`: total maintenance cost for this morphon's outgoing
    /// synapses, precomputed by the caller. Includes distance-dependent and myelination
    /// maintenance factors.
    /// `metabolic`: V3 metabolic budget configuration.
    pub fn step(&mut self, dt: f64, synapse_maintenance_cost: f64, metabolic: &MetabolicConfig, frustration_config: &FrustrationConfig, threshold_bias: f64) {
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
        // Pseudo-random noise centered at zero, scaled by per-morphon intrinsic_noise.
        // intrinsic_noise is set per cell type (Motor=0, Associative=0.08, others=0.1),
        // inherited in divide(), updated on differentiation. Frustration amplifies on top.
        let noise_raw = (self.id.wrapping_mul(self.age).wrapping_add(7919) % 1000) as f64 / 1000.0;
        let noise_scale = self.intrinsic_noise * self.frustration.noise_amplitude;
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
            let rate = self.activity_history.mean();
            self.energy -= metabolic.firing_cost * (1.0 + rate * metabolic.superlinear_firing_factor);
            self.activity_history.push(1.0);
        } else {
            self.activity_history.push(0.0);
        }

        // Medium-path operations: PE, frustration, homeostatic regulation, metabolic.
        self.medium_update(dt, synapse_maintenance_cost, metabolic, frustration_config);
    }

    /// Medium-path update: prediction error, desire, frustration, homeostatic threshold
    /// regulation, division pressure, metabolic energy, and cooldown bookkeeping.
    ///
    /// Extracted from `step()` for the hot-array world where the fast path runs on dense
    /// arrays and medium-path ops run once every 10 ticks on Morphon structs directly.
    /// Called by `step()` for backward compat, and by `System::step()` on the medium tick.
    pub fn medium_update(
        &mut self,
        dt: f64,
        synapse_maintenance_cost: f64,
        metabolic: &MetabolicConfig,
        frustration_config: &FrustrationConfig,
    ) {
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
        // V2: Fused morphons defer to cluster-level homeostasis in system.rs
        // Upper clamp respects the activation function's output range — without this,
        // threshold can exceed max activation output, permanently silencing the morphon.
        let actual_rate = self.activity_history.mean();
        if self.fused_with.is_none() {
            self.threshold += 0.01 * (actual_rate - self.homeostatic_setpoint);
            let max_threshold = self.activation_fn.max_output();
            self.threshold = self.threshold.clamp(0.05, max_threshold);
        }

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
        // 1. Base maintenance cost — V2: fused morphons get reduced cost (offload overhead)
        let effective_base_cost = if self.fused_with.is_some() {
            metabolic.base_cost * (1.0 - metabolic.cluster_base_cost_reduction)
        } else {
            metabolic.base_cost
        };
        self.energy -= effective_base_cost;
        // 2. Per-synapse maintenance cost (connections are expensive).
        // Cost includes distance-dependent and myelination maintenance factors
        // precomputed by the caller.
        self.energy -= synapse_maintenance_cost;
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

        // Track sustained energy deficit for duration-based apoptosis.
        // Hubs drain energy through superlinear firing cost; this counter
        // ensures apoptosis only fires after sustained deficit, not transient dips.
        if self.energy < 0.1 {
            self.ticks_below_energy_threshold = self.ticks_below_energy_threshold.saturating_add(1);
        } else {
            self.ticks_below_energy_threshold = 0;
        }

        // Tick down migration cooldown
        if self.migration_cooldown > 0.0 {
            self.migration_cooldown -= dt;
        }

        // Reset accumulator for next step
        self.input_accumulator = 0.0;
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
        self.receptor_sensitivity = default_receptor_sensitivity(target);
        self.intrinsic_noise = crate::types::intrinsic_noise_for(target);
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
            self.receptor_sensitivity = default_receptor_sensitivity(CellType::Stem);
            self.intrinsic_noise = crate::types::intrinsic_noise_for(CellType::Stem);
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
            m.step(1.0, 0.0, &metabolic, &fc, 0.0);
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
            m.step(1.0, 0.0, &metabolic, &fc, 0.0);
        }
        let frustrated_level = m.frustration.frustration_level;
        assert!(frustrated_level > 0.0);

        // Now inject varying input — PE changes, stagnation counter should decay
        for i in 0..100 {
            m.input_accumulator = (i as f64) * 0.1;
            m.step(1.0, 0.0, &metabolic, &fc, 0.0);
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
        let _potential_before = m.potential;
        m.input_accumulator = 0.0;
        m.step(1.0, 0.0, &metabolic, &fc, 0.0);
        // Motor has full leak (leak_rate=1.0), so potential = 0*(1-1) + 0 + 0 = 0
        // No noise should be added
        assert!((m.potential - 0.0).abs() < f64::EPSILON,
            "motor morphon potential should not have noise");
    }

    // === Myelination ===

    /// Neutral context: no neuromodulatory influence.
    fn neutral_ctx() -> MyelinationContext {
        MyelinationContext { arousal: 0.0, reward: 0.0, energy_pressure: 0.0 }
    }

    #[test]
    fn myelination_increases_with_consolidation_and_activity() {
        let mut syn = Synapse::new(0.5);
        syn.consolidation_level = 0.8;
        syn.activity_trace = 1.0; // above 0.1 gate
        let ctx = neutral_ctx();

        for _ in 0..1000 {
            syn.update_myelination(1.0, &ctx);
        }

        assert!(syn.myelination > 0.0, "myelination should increase: {}", syn.myelination);
        assert!(syn.myelination < 0.3, "myelination should be slow: {}", syn.myelination);
    }

    #[test]
    fn myelination_stays_zero_without_consolidation() {
        let mut syn = Synapse::new(0.5);
        syn.consolidation_level = 0.0;
        syn.activity_trace = 1.0;
        let ctx = neutral_ctx();

        for _ in 0..1000 {
            syn.update_myelination(1.0, &ctx);
        }

        assert!(syn.myelination.abs() < 1e-6,
            "unmyelinated without consolidation: {}", syn.myelination);
    }

    #[test]
    fn effective_delay_decreases_with_myelination() {
        let mut syn = Synapse::new(0.5).with_delay(3.0);
        let base_delay = syn.effective_delay();
        assert!((base_delay - 3.0).abs() < 1e-10, "no myelination = base delay");

        syn.myelination = 0.5;
        let mid_delay = syn.effective_delay();
        assert!(mid_delay < base_delay, "partial myelination reduces delay");

        syn.myelination = 1.0;
        let full_delay = syn.effective_delay();
        assert!((full_delay - 0.6).abs() < 1e-10,
            "full myelination: 3.0 / 5.0 = 0.6, got {}", full_delay);
    }

    #[test]
    fn effective_delay_has_minimum() {
        let mut syn = Synapse::new(0.5).with_delay(1.0);
        syn.myelination = 1.0;
        assert!((syn.effective_delay() - 0.5).abs() < 1e-10,
            "effective delay floor is 0.5: {}", syn.effective_delay());
    }

    #[test]
    fn myelination_decays_without_activity() {
        let mut syn = Synapse::new(0.5);
        syn.myelination = 0.5;
        syn.consolidation_level = 0.8;
        syn.activity_trace = 0.0; // inactive — no recent signal
        let ctx = neutral_ctx();

        for _ in 0..2000 {
            syn.update_myelination(1.0, &ctx);
        }

        assert!(syn.myelination < 0.5,
            "myelination should decay without activity: {}", syn.myelination);
    }

    #[test]
    fn arousal_accelerates_myelination() {
        let mut syn_calm = Synapse::new(0.5);
        syn_calm.consolidation_level = 0.8;
        syn_calm.activity_trace = 1.0;
        let mut syn_aroused = syn_calm.clone();

        let calm = neutral_ctx();
        let aroused = MyelinationContext { arousal: 1.0, reward: 0.0, energy_pressure: 0.0 };

        for _ in 0..1000 {
            syn_calm.update_myelination(1.0, &calm);
            syn_aroused.update_myelination(1.0, &aroused);
        }

        assert!(syn_aroused.myelination > syn_calm.myelination,
            "arousal should accelerate myelination: aroused={} vs calm={}",
            syn_aroused.myelination, syn_calm.myelination);
    }

    #[test]
    fn reward_biases_tagged_synapses() {
        let mut syn_tagged = Synapse::new(0.5);
        syn_tagged.consolidation_level = 0.3; // low consolidation
        syn_tagged.activity_trace = 0.5;
        syn_tagged.tag = 0.5; // has active tag
        let mut syn_untagged = syn_tagged.clone();
        syn_untagged.tag = 0.0; // no tag

        let ctx = MyelinationContext { arousal: 0.0, reward: 1.0, energy_pressure: 0.0 };

        for _ in 0..1000 {
            syn_tagged.update_myelination(1.0, &ctx);
            syn_untagged.update_myelination(1.0, &ctx);
        }

        assert!(syn_tagged.myelination > syn_untagged.myelination,
            "reward should bias myelination toward tagged synapses: tagged={} vs untagged={}",
            syn_tagged.myelination, syn_untagged.myelination);
    }

    #[test]
    fn energy_pressure_inhibits_myelination() {
        let mut syn_healthy = Synapse::new(0.5);
        syn_healthy.consolidation_level = 0.8;
        syn_healthy.activity_trace = 1.0;
        let mut syn_stressed = syn_healthy.clone();

        let healthy = neutral_ctx();
        let stressed = MyelinationContext { arousal: 0.0, reward: 0.0, energy_pressure: 0.9 };

        for _ in 0..1000 {
            syn_healthy.update_myelination(1.0, &healthy);
            syn_stressed.update_myelination(1.0, &stressed);
        }

        assert!(syn_healthy.myelination > syn_stressed.myelination,
            "energy pressure should inhibit myelination: healthy={} vs stressed={}",
            syn_healthy.myelination, syn_stressed.myelination);
    }

    #[test]
    fn treadmill_erodes_myelin_without_activity() {
        // Start with full myelination but no activity or consolidation.
        // Treadmill removal should erode it over time.
        let mut syn = Synapse::new(0.5);
        syn.myelination = 1.0;
        syn.consolidation_level = 0.0;
        syn.activity_trace = 0.0;
        let ctx = neutral_ctx();

        for _ in 0..10000 {
            syn.update_myelination(1.0, &ctx);
        }

        assert!(syn.myelination < 0.3,
            "treadmill should erode unmaintained myelin: {}", syn.myelination);
    }

    #[test]
    fn superlinear_firing_cost_penalizes_high_rate_morphons() {
        let mut metabolic = MetabolicConfig::default();
        metabolic.superlinear_firing_factor = 2.0;
        let fc = default_frustration_config();

        // High-rate morphon: fill activity history with 50% firing
        let mut hub = make_morphon();
        hub.energy = 1.0;
        hub.threshold = -100.0; // force firing
        for _ in 0..50 {
            hub.activity_history.push(1.0);
        }
        for _ in 0..50 {
            hub.activity_history.push(0.0);
        }
        let hub_energy_before = hub.energy;
        hub.input_accumulator = 1.0;
        hub.step(1.0, 0.0, &metabolic, &fc, 0.0);
        let hub_cost = hub_energy_before - hub.energy;

        // Low-rate morphon: fill activity history with 5% firing
        let mut specialist = make_morphon();
        specialist.energy = 1.0;
        specialist.threshold = -100.0;
        for _ in 0..5 {
            specialist.activity_history.push(1.0);
        }
        for _ in 0..95 {
            specialist.activity_history.push(0.0);
        }
        let spec_energy_before = specialist.energy;
        specialist.input_accumulator = 1.0;
        specialist.step(1.0, 0.0, &metabolic, &fc, 0.0);
        let spec_cost = spec_energy_before - specialist.energy;

        assert!(hub.fired && specialist.fired, "both should fire");
        assert!(hub_cost > spec_cost * 1.5,
            "hub should pay significantly more: hub={hub_cost:.4}, spec={spec_cost:.4}");
    }
}
