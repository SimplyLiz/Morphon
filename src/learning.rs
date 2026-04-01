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
    /// Time constant for pre/post-synaptic spike traces (Frémaux & Gerstner 2016).
    /// Controls the STDP window width — how long after a spike the trace persists.
    pub tau_trace: f64,
    /// LTP magnitude (applied when post fires and pre_trace > 0).
    pub a_plus: f64,
    /// LTD magnitude (applied when pre fires and post_trace > 0). Negative.
    pub a_minus: f64,
    /// Time constant for slow synaptic tag decay (much longer).
    pub tau_tag: f64,
    /// Eligibility threshold for setting a synaptic tag.
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
            tau_trace: 10.0,            // STDP trace window (Lava/Loihi: 10)
            a_plus: 1.0,               // LTP magnitude
            a_minus: -1.0,             // LTD magnitude (symmetric)
            tau_tag: 6000.0,            // ~6000 timesteps ≈ minutes (slow)
            tag_threshold: 0.3,         // eligibility threshold for tagging
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

/// Update eligibility trace using trace-based STDP (Frémaux & Gerstner 2016).
///
/// Instead of binary coincidence detection per timestep, pre- and post-synaptic
/// traces maintain a decaying memory of recent spikes. When pre fires, the
/// post_trace determines LTD. When post fires, the pre_trace determines LTP.
/// This widens the effective STDP window from 1 timestep to ~tau_trace steps,
/// solving the co-firing problem caused by refractory periods and spike delays.
///
/// The eligibility trace accumulates these STDP contributions and decays slowly,
/// serving as the "tag" that the three-factor modulation signal acts on.
pub fn update_eligibility(
    synapse: &mut Synapse,
    pre_fired: bool,
    post_fired: bool,
    params: &LearningParams,
    dt: f64,
) {
    // Decay traces
    let trace_decay = (-dt / params.tau_trace).exp();
    synapse.pre_trace *= trace_decay;
    synapse.post_trace *= trace_decay;

    // STDP: event-driven eligibility updates
    let mut stdp = 0.0;
    if pre_fired {
        // Pre-before-post: LTD proportional to how recently post fired
        stdp += params.a_minus * synapse.post_trace;
        synapse.pre_trace += 1.0;
    }
    if post_fired {
        // Post-after-pre: LTP proportional to how recently pre fired
        stdp += params.a_plus * synapse.pre_trace;
        synapse.post_trace += 1.0;
    }

    // Eligibility trace: exponential decay + STDP contributions
    synapse.eligibility += (-synapse.eligibility / params.tau_eligibility + stdp) * dt;
    synapse.eligibility = synapse.eligibility.clamp(-1.0, 1.0);

    // Slow synaptic tag: set when eligibility is strongly positive
    if synapse.eligibility > params.tag_threshold && !synapse.consolidated {
        synapse.tag = 1.0;
        synapse.tag_strength = synapse.eligibility;
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
    // Receptor-gated modulation: only include channels the post-synaptic morphon responds to.
    // Reward channel uses ADVANTAGE (reward - baseline) instead of raw reward.
    // This eliminates the unsupervised bias that causes systematic weight drift
    // (Frémaux et al. 2010: mean(reward) × mean(eligibility) term dominates otherwise).
    let r = if post_receptors.contains(&ModulatorType::Reward) {
        params.alpha_reward * modulation.reward_advantage()
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
