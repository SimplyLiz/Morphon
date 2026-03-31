//! Three-Factor Learning — eligibility traces modulated by global signals.
//!
//! The core learning rule:
//!   ẇᵢⱼ = eᵢⱼ(t) · M(t)
//!   ėᵢⱼ = -eᵢⱼ/τₑ + H(preᵢ, postⱼ)
//!
//! Extended with Tag-and-Capture (Frey & Morris, 1997) for delayed reward:
//!   - Synaptic tags mark synapses at strong Hebbian coincidence (slow decay, τ ~ minutes)
//!   - When a strong reward signal arrives, tagged synapses are "captured" (permanent weight change)
//!   - This solves credit assignment for delayed reward without global gradients

use crate::morphon::Synapse;
use crate::neuromodulation::Neuromodulation;
use crate::types::{ModulatorType, ReceptorSet};
use serde::{Deserialize, Serialize};

/// Parameters for the three-factor learning rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningParams {
    /// Time constant for fast eligibility trace decay.
    pub tau_eligibility: f64,
    /// Time constant for slow synaptic tag decay (much longer).
    pub tau_tag: f64,
    /// Hebbian coincidence threshold for setting a synaptic tag.
    pub tag_threshold: f64,
    /// Reward threshold for capturing a tagged synapse.
    pub capture_threshold: f64,
    /// Learning rate for tag capture.
    pub capture_rate: f64,
    /// Maximum absolute weight value.
    pub weight_max: f64,
    /// Minimum absolute weight value (below this, prune candidate).
    pub weight_min: f64,
    /// Modulation channel weights for the combined signal.
    pub alpha_reward: f64,
    pub alpha_novelty: f64,
    pub alpha_arousal: f64,
    pub alpha_homeostasis: f64,
}

impl Default for LearningParams {
    fn default() -> Self {
        Self {
            tau_eligibility: 20.0,      // ~20 timesteps (fast)
            tau_tag: 6000.0,            // ~6000 timesteps ≈ minutes (slow)
            tag_threshold: 0.7,         // strong Hebbian coincidence needed
            capture_threshold: 0.5,     // moderate reward needed for capture
            capture_rate: 0.1,
            weight_max: 5.0,
            weight_min: 0.001,
            alpha_reward: 1.0,
            alpha_novelty: 0.5,
            alpha_arousal: 0.3,
            alpha_homeostasis: 0.1,
        }
    }
}

/// Compute the Hebbian coincidence function H(pre, post).
///
/// LTD magnitudes are calibrated so that at the homeostatic setpoint (10% firing),
/// the expected H is approximately zero. This prevents systematic negative eligibility
/// drift that would make all weight updates depressive regardless of reward.
///
/// At p = q = 0.1:
///   E[H] = 0.01*(+1.0) + 0.09*(-0.06) + 0.09*(-0.05) + 0.81*(0.0) ≈ 0.0
///
/// Correlated pairs (both fire together more than chance) get net positive eligibility.
/// Anti-correlated pairs get net negative. Uncorrelated pairs stay near zero.
fn hebbian_coincidence(pre_fired: bool, post_fired: bool) -> f64 {
    match (pre_fired, post_fired) {
        (true, true) => 1.0,     // LTP: both active — strong potentiation
        (true, false) => -0.06,  // mild LTD: pre fired alone
        (false, true) => -0.05,  // mild LTD: post fired alone
        (false, false) => 0.0,   // no change
    }
}

/// Update the eligibility trace for a synapse.
///
///   ėᵢⱼ = -eᵢⱼ/τₑ + H(preᵢ, postⱼ)
pub fn update_eligibility(
    synapse: &mut Synapse,
    pre_fired: bool,
    post_fired: bool,
    params: &LearningParams,
    dt: f64,
) {
    let h = hebbian_coincidence(pre_fired, post_fired);

    // Fast eligibility trace: exponential decay + Hebbian input
    synapse.eligibility += (-synapse.eligibility / params.tau_eligibility + h) * dt;
    synapse.eligibility = synapse.eligibility.clamp(-1.0, 1.0);

    // Slow synaptic tag: set on strong Hebbian coincidence, decay slowly
    if h > params.tag_threshold && !synapse.consolidated {
        synapse.tag = 1.0;
        synapse.tag_strength = h;
    }
    // Tag decays exponentially (much slower than eligibility)
    synapse.tag *= (-dt / params.tau_tag).exp();
}

/// Apply the receptor-gated three-factor learning rule.
///
/// Only modulation channels that the post-synaptic morphon has receptors for
/// are included in M(t). This creates differential learning:
///   - Motor morphons (Reward + Arousal receptors) learn from reward signals
///   - Sensory morphons (Novelty + Arousal receptors) learn from novelty
///   - Associative morphons (Reward + Novelty receptors) learn from both
///   - Modulatory morphons (Homeostasis receptors) learn from stability signals
///
/// Also handles Tag-and-Capture for delayed reward.
/// Returns `true` if a tag-and-capture consolidation event occurred.
pub fn apply_weight_update(
    synapse: &mut Synapse,
    modulation: &Neuromodulation,
    params: &LearningParams,
    plasticity_rate: f64,
    post_receptors: &ReceptorSet,
) -> bool {
    // Receptor-gated modulation: only include channels the post-synaptic morphon responds to
    let r = if post_receptors.contains(&ModulatorType::Reward) {
        params.alpha_reward * modulation.reward
    } else {
        0.0
    };
    let n = if post_receptors.contains(&ModulatorType::Novelty) {
        params.alpha_novelty * modulation.novelty
    } else {
        0.0
    };
    let a = if post_receptors.contains(&ModulatorType::Arousal) {
        params.alpha_arousal * modulation.arousal
    } else {
        0.0
    };
    let h = if post_receptors.contains(&ModulatorType::Homeostasis) {
        params.alpha_homeostasis * modulation.homeostasis
    } else {
        0.0
    };
    let m = r + n + a + h;

    // Standard three-factor: fast eligibility × receptor-gated modulation
    let delta_w = synapse.eligibility * m * plasticity_rate;
    synapse.weight += delta_w;

    // Tag-and-Capture: only if morphon has Reward receptor
    let captured = post_receptors.contains(&ModulatorType::Reward)
        && synapse.tag > 0.1
        && modulation.reward > params.capture_threshold
        && !synapse.consolidated;

    if captured {
        synapse.weight += params.capture_rate * synapse.tag_strength * modulation.reward;
        synapse.consolidated = true;
        synapse.tag = 0.0;
    }

    synapse.weight = synapse.weight.clamp(-params.weight_max, params.weight_max);

    if synapse.eligibility.abs() > 0.1 {
        synapse.usage_count += 1;
    }
    synapse.age += 1;

    captured
}

/// Determine if a synapse should be pruned based on activity.
pub fn should_prune(synapse: &Synapse, params: &LearningParams) -> bool {
    // Old synapses with very low weight and low usage
    // Consolidated synapses are protected from pruning
    !synapse.consolidated
        && synapse.age > 100
        && synapse.weight.abs() < params.weight_min
        && synapse.usage_count < 5
}
