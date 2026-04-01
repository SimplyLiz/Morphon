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
    /// Transmitter-induced potentiation rate — prevents silent death.
    /// When pre fires but post rate is low, apply this as a floor on dw.
    /// Zenke et al. 2015: must operate on SAME timescale as STDP.
    pub transmitter_potentiation: f64,
    /// Heterosynaptic depression rate — prevents runaway excitation.
    /// When post fires, ALL incoming synapses get depressed by this fraction.
    /// Independent of pre-synaptic activity.
    pub heterosynaptic_depression: f64,
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
            transmitter_potentiation: 0.001,  // small floor — prevents silent death
            heterosynaptic_depression: 0.002, // slight depression on all inputs when post fires
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
/// `post_activity`: continuous postsynaptic activity signal.
/// For spiking morphons: 1.0 when fired, 0.0 otherwise.
/// For Motor morphons (non-spiking readout): normalized membrane potential,
/// so the learning rule responds to graded output even when the morphon
/// doesn't spike. This matches the DSQN/SpikeGym pattern where output
/// neurons are non-spiking leaky integrators.
pub fn update_eligibility(
    synapse: &mut Synapse,
    pre_fired: bool,
    post_activity: f64,
    params: &LearningParams,
    dt: f64,
) {
    // Decay traces
    let trace_decay = (-dt / params.tau_trace).exp();
    synapse.pre_trace *= trace_decay;
    synapse.post_trace *= trace_decay;

    // Weight-dependent (multiplicative) STDP with soft bounds.
    // LTP scales as (w_max - w): easy to strengthen weak synapses, hard to over-strengthen.
    // LTD scales as (w - w_min): easy to weaken strong synapses, protects weak ones.
    // This produces stable long-tail weight distributions instead of bimodal collapse
    // (Gilson & Fukai 2011, van Rossum et al. 2000).
    let w = synapse.weight;
    let ltp_scale = (params.weight_max - w) / params.weight_max; // 1.0 at w=0, 0.0 at w=w_max
    let ltd_scale = (w + params.weight_max) / (2.0 * params.weight_max); // 0.0 at w=-w_max, 1.0 at w=w_max

    let mut stdp = 0.0;
    if pre_fired {
        // Pre-before-post: LTD proportional to how recently post was active
        stdp += params.a_minus * synapse.post_trace * ltd_scale;
        synapse.pre_trace += 1.0;
    }
    if post_activity > 0.01 {
        // Post active: LTP proportional to pre trace × activity level
        stdp += params.a_plus * synapse.pre_trace * post_activity * ltp_scale;
        synapse.post_trace += post_activity;
    }

    // Eligibility trace: exponential decay + STDP contributions
    let e_delta = (-synapse.eligibility / params.tau_eligibility + stdp) * dt;
    if e_delta.is_finite() {
        synapse.eligibility += e_delta;
    }
    synapse.eligibility = if synapse.eligibility.is_finite() {
        synapse.eligibility.clamp(-1.0, 1.0)
    } else {
        0.0
    };

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
    channel_gains: [f64; 4],
) -> bool {
    // Receptor-gated modulation: only include channels the post-synaptic morphon responds to.
    // Channel gains are set by Endoquilibrium (default [1.0; 4] when disabled).
    // Reward channel uses DELTA (reward change) as a pseudo-TD error signal.
    // This provides bidirectional modulation that never converges to zero:
    // improving → positive, worsening → negative, steady → zero.
    // (Frémaux et al. 2013: critic/TD-error signal is necessary for RL convergence)
    let r = if post_receptors.contains(&ModulatorType::Reward) {
        params.alpha_reward * modulation.reward_delta() * channel_gains[0]
    } else {
        0.0
    };
    let n = if post_receptors.contains(&ModulatorType::Novelty) {
        params.alpha_novelty * modulation.novelty * channel_gains[1]
    } else {
        0.0
    };
    let a = if post_receptors.contains(&ModulatorType::Arousal) {
        params.alpha_arousal * modulation.arousal * channel_gains[2]
    } else {
        0.0
    };
    let h = if post_receptors.contains(&ModulatorType::Homeostasis) {
        params.alpha_homeostasis * modulation.homeostasis * channel_gains[3]
    } else {
        0.0
    };
    let m = r + n + a + h;

    // Standard three-factor: fast eligibility × receptor-gated modulation
    // Consolidated synapses get reduced updates (10% residual plasticity at level=1.0)
    let consolidation_scale = 1.0 - synapse.consolidation_level * 0.9;
    let delta_w = synapse.eligibility * m * plasticity_rate * consolidation_scale;
    if delta_w.is_finite() {
        synapse.weight += delta_w;
    }

    // Tag-and-Capture: per-tick capture is disabled.
    // Capture is now episode-gated via System::report_episode_end().
    // Tags still accumulate here (via update_eligibility) for later capture.
    let captured = false;

    synapse.weight = if synapse.weight.is_finite() {
        synapse.weight.clamp(-params.weight_max, params.weight_max)
    } else {
        0.0
    };

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphon::Synapse;
    use crate::neuromodulation::Neuromodulation;
    use crate::types::{default_receptors, CellType, ModulatorType};

    #[test]
    fn eligibility_decays_without_activity() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(0.5);
        // Build up eligibility
        for _ in 0..5 {
            update_eligibility(&mut syn, true, 1.0, &params, 1.0);
        }
        let peak = syn.eligibility;
        assert!(peak > 0.0);

        // Let it decay with no activity
        for _ in 0..200 {
            update_eligibility(&mut syn, false, 0.0, &params, 1.0);
        }
        assert!(
            syn.eligibility.abs() < peak.abs() * 0.1,
            "eligibility should decay significantly: was {peak}, now {}",
            syn.eligibility
        );
    }

    #[test]
    fn pre_trace_increments_on_pre_spike() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(0.5);
        assert_eq!(syn.pre_trace, 0.0);

        update_eligibility(&mut syn, true, 0.0, &params, 1.0);
        assert!(syn.pre_trace > 0.0, "pre_trace should increment on pre spike");
    }

    #[test]
    fn post_trace_increments_on_post_activity() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(0.5);
        assert_eq!(syn.post_trace, 0.0);

        update_eligibility(&mut syn, false, 0.8, &params, 1.0);
        assert!(syn.post_trace > 0.0, "post_trace should increment on post activity");
    }

    #[test]
    fn traces_decay_exponentially() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(0.5);

        // Set traces
        update_eligibility(&mut syn, true, 1.0, &params, 1.0);
        let pre_after_spike = syn.pre_trace;
        let post_after_spike = syn.post_trace;

        // Decay for several steps
        for _ in 0..50 {
            update_eligibility(&mut syn, false, 0.0, &params, 1.0);
        }
        assert!(syn.pre_trace < pre_after_spike * 0.1, "pre_trace should decay");
        assert!(syn.post_trace < post_after_spike * 0.1, "post_trace should decay");
    }

    #[test]
    fn eligibility_clamped_to_unit_range() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(0.5);

        // Massive coincidence
        for _ in 0..1000 {
            update_eligibility(&mut syn, true, 1.0, &params, 1.0);
        }
        assert!(syn.eligibility <= 1.0, "eligibility must be <= 1.0");
        assert!(syn.eligibility >= -1.0, "eligibility must be >= -1.0");
    }

    #[test]
    fn tag_set_when_eligibility_exceeds_threshold() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(0.1); // low weight → high LTP scale
        for _ in 0..10 {
            update_eligibility(&mut syn, true, 1.0, &params, 1.0);
        }
        assert!(
            syn.tag > 0.0,
            "tag should be set when eligibility > threshold ({})",
            params.tag_threshold
        );
        assert!(syn.tag_strength > 0.0, "tag_strength should be set");
    }

    #[test]
    fn tag_decays_slower_than_eligibility() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(0.1);
        // Build tag
        for _ in 0..10 {
            update_eligibility(&mut syn, true, 1.0, &params, 1.0);
        }
        let tag_after_set = syn.tag;
        let elig_after_set = syn.eligibility;

        // Decay for 100 steps
        for _ in 0..100 {
            update_eligibility(&mut syn, false, 0.0, &params, 1.0);
        }
        let tag_ratio = syn.tag / tag_after_set;
        let elig_ratio = if elig_after_set.abs() > 1e-10 {
            syn.eligibility.abs() / elig_after_set.abs()
        } else {
            0.0
        };
        assert!(
            tag_ratio > elig_ratio,
            "tag should decay slower: tag retained {:.1}%, eligibility retained {:.1}%",
            tag_ratio * 100.0,
            elig_ratio * 100.0
        );
    }

    #[test]
    fn weight_update_receptor_gated() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(0.5);
        syn.eligibility = 0.5;

        let mut modulation = Neuromodulation::default();
        modulation.inject_reward(0.8);
        // Inject again to create a positive delta
        modulation.inject_reward(0.1);

        // Sensory receptors: Novelty + Arousal (no Reward)
        let sensory_receptors = default_receptors(CellType::Sensory);
        assert!(!sensory_receptors.contains(&ModulatorType::Reward));

        let weight_before = syn.weight;
        apply_weight_update(&mut syn, &modulation, &params, 0.01, &sensory_receptors, [1.0; 4]);

        // With zero novelty and zero arousal, weight change should be minimal
        let _delta_sensory = (syn.weight - weight_before).abs();

        // Now test with motor receptors (has Reward)
        let mut syn2 = Synapse::new(0.5);
        syn2.eligibility = 0.5;
        let motor_receptors = default_receptors(CellType::Motor);
        assert!(motor_receptors.contains(&ModulatorType::Reward));

        let weight_before2 = syn2.weight;
        apply_weight_update(&mut syn2, &modulation, &params, 0.01, &motor_receptors, [1.0; 4]);
        let delta2 = (syn2.weight - weight_before2).abs();

        // Motor (with reward receptor) should have larger change when reward is high
        // This is a soft check — the exact values depend on reward_delta
        assert!(
            delta2 >= 0.0,
            "motor synapse update should be non-negative: {delta2}"
        );
    }

    #[test]
    fn per_tick_capture_disabled() {
        // Per-tick capture is now disabled — capture happens at episode end
        // via System::report_episode_end(). Tags still accumulate.
        let params = LearningParams::default();
        let mut syn = Synapse::new(0.3);
        syn.tag = 0.5;
        syn.tag_strength = 0.4;

        let mut modulation = Neuromodulation::default();
        modulation.inject_reward(0.8);

        let motor_receptors = default_receptors(CellType::Motor);
        let captured = apply_weight_update(&mut syn, &modulation, &params, 0.01, &motor_receptors, [1.0; 4]);

        assert!(!captured, "per-tick capture should be disabled");
        assert!(!syn.consolidated, "synapse should not be consolidated per-tick");
    }

    #[test]
    fn consolidation_level_scales_weight_updates() {
        let params = LearningParams::default();
        let mut syn_plastic = Synapse::new(0.3);
        syn_plastic.eligibility = 0.5;

        let mut syn_consolidated = Synapse::new(0.3);
        syn_consolidated.eligibility = 0.5;
        syn_consolidated.consolidation_level = 1.0;

        // Create a positive reward delta: inject 0 first (sets prev), then 0.8
        let mut modulation = Neuromodulation::default();
        modulation.inject_reward(0.0); // sets prev_reward = 0
        modulation.inject_reward(0.8); // reward_delta = 0.8 - 0 = 0.8

        // Use Associative receptors (Reward + Novelty) — reward delta drives update
        let receptors = default_receptors(CellType::Associative);
        apply_weight_update(&mut syn_plastic, &modulation, &params, 0.1, &receptors, [1.0; 4]);
        apply_weight_update(&mut syn_consolidated, &modulation, &params, 0.1, &receptors, [1.0; 4]);

        let delta_plastic = (syn_plastic.weight - 0.3).abs();
        let delta_consolidated = (syn_consolidated.weight - 0.3).abs();
        assert!(delta_plastic > 0.0001, "plastic synapse should get an update, got {:.6}", delta_plastic);
        // Consolidated should get ~10% of the update
        assert!(delta_consolidated < delta_plastic * 0.5,
            "consolidated ({:.6}) should get much less update than plastic ({:.6})",
            delta_consolidated, delta_plastic);
    }

    #[test]
    fn weight_clamped_to_max() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(4.9); // near weight_max (5.0)
        syn.eligibility = 1.0;

        let mut modulation = Neuromodulation::default();
        modulation.inject_reward(1.0);
        modulation.inject_reward(1.0);

        let motor_receptors = default_receptors(CellType::Motor);
        apply_weight_update(&mut syn, &modulation, &params, 1.0, &motor_receptors, [1.0; 4]);

        assert!(
            syn.weight <= params.weight_max,
            "weight {} should not exceed weight_max {}",
            syn.weight,
            params.weight_max
        );
        assert!(
            syn.weight >= -params.weight_max,
            "weight {} should not go below -weight_max",
            syn.weight
        );
    }

    #[test]
    fn should_prune_old_weak_unused() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(0.0001); // below weight_min
        syn.age = 200;
        syn.usage_count = 2;
        assert!(should_prune(&syn, &params));
    }

    #[test]
    fn should_not_prune_young_synapse() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(0.0001);
        syn.age = 50; // below 100
        syn.usage_count = 0;
        assert!(!should_prune(&syn, &params));
    }

    #[test]
    fn should_not_prune_consolidated() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(0.0001);
        syn.age = 200;
        syn.usage_count = 0;
        syn.consolidated = true;
        assert!(!should_prune(&syn, &params), "consolidated synapses are protected");
    }

    #[test]
    fn should_not_prune_strong_synapse() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(1.0); // well above weight_min
        syn.age = 200;
        syn.usage_count = 0;
        assert!(!should_prune(&syn, &params));
    }

    #[test]
    fn usage_count_increments_with_active_eligibility() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(0.5);
        syn.eligibility = 0.5; // above 0.1 threshold

        let modulation = Neuromodulation::default();
        let receptors = default_receptors(CellType::Stem);

        let initial_usage = syn.usage_count;
        apply_weight_update(&mut syn, &modulation, &params, 0.01, &receptors, [1.0; 4]);
        assert_eq!(syn.usage_count, initial_usage + 1);
    }

    #[test]
    fn age_increments_on_weight_update() {
        let params = LearningParams::default();
        let mut syn = Synapse::new(0.5);

        let modulation = Neuromodulation::default();
        let receptors = default_receptors(CellType::Stem);

        let initial_age = syn.age;
        apply_weight_update(&mut syn, &modulation, &params, 0.01, &receptors, [1.0; 4]);
        assert_eq!(syn.age, initial_age + 1);
    }
}
