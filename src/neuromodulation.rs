//! Neuromodulation — four broadcast channels replacing backpropagation.
//!
//! Each channel is a global signal that modulates local eligibility traces.
//! The specificity comes from the interaction: only synapses with active
//! eligibility traces are affected by the broadcast.

use crate::types::ModulatorType;
use serde::{Deserialize, Serialize};

/// The four neuromodulatory channels as a single broadcast state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuromodulation {
    /// Dopamine analog — reinforces recently active eligibility traces.
    pub reward: f64,
    /// Acetylcholine analog — increases plasticity rate systemwide.
    pub novelty: f64,
    /// Noradrenaline analog — increases threshold sensitivity.
    pub arousal: f64,
    /// Serotonin analog — regulates baseline activity level.
    pub homeostasis: f64,

    /// Running average of reward signal — used for advantage computation.
    pub reward_baseline: f64,
    /// Previous step's reward — used for reward delta (pseudo-TD error).
    /// Frémaux et al. 2013 showed a critic (TD error) is necessary for RL tasks.
    /// The reward delta R(t) - R(t-1) is the simplest approximation: it tracks
    /// instantaneous improvement/deterioration without needing a value function.
    pub prev_reward: f64,

    /// Decay rates for each channel.
    reward_decay: f64,
    novelty_decay: f64,
    arousal_decay: f64,
    homeostasis_decay: f64,
}

impl Default for Neuromodulation {
    fn default() -> Self {
        Self {
            reward: 0.0,
            novelty: 0.0,
            arousal: 0.0,
            homeostasis: 0.5, // baseline stability
            reward_baseline: 0.0,
            prev_reward: 0.0,
            reward_decay: 0.95,
            novelty_decay: 0.90,
            arousal_decay: 0.85,
            homeostasis_decay: 0.99, // very slow decay
        }
    }
}

impl Neuromodulation {
    /// Inject a reward signal (0.0 to 1.0).
    /// Tracks previous reward for delta computation and updates EMA baseline.
    pub fn inject_reward(&mut self, strength: f64) {
        self.prev_reward = self.reward;
        self.reward = (self.reward + strength).clamp(0.0, 1.0);
        self.reward_baseline += 0.01 * (strength - self.reward_baseline);
    }

    /// Reward delta: pseudo-TD error approximating the critic signal.
    ///
    /// R(t) - R(t-1) tracks instantaneous improvement/deterioration:
    /// - Positive: situation improving → strengthen active pathways
    /// - Negative: situation worsening → weaken active pathways
    /// - Zero: steady state → no change (nothing to learn)
    ///
    /// Unlike advantage (reward - baseline), this NEVER converges to zero
    /// because it tracks change, not level. This is the simplest approximation
    /// to a TD error signal without requiring a separate value function.
    /// (Frémaux et al. 2013 showed a critic signal is necessary for RL tasks)
    pub fn reward_delta(&self) -> f64 {
        self.reward - self.prev_reward
    }

    /// The advantage signal (reward - baseline), clamped non-negative.
    /// Used for classification tasks where reward is sparse and binary.
    /// For RL tasks, prefer reward_delta() which provides bidirectional signal.
    pub fn reward_advantage(&self) -> f64 {
        (self.reward - self.reward_baseline).max(0.0)
    }

    /// Inject a novelty signal (0.0 to 1.0).
    pub fn inject_novelty(&mut self, strength: f64) {
        self.novelty = (self.novelty + strength).clamp(0.0, 1.0);
    }

    /// Inject an arousal signal (0.0 to 1.0).
    pub fn inject_arousal(&mut self, strength: f64) {
        self.arousal = (self.arousal + strength).clamp(0.0, 1.0);
    }

    /// Inject a homeostasis signal (0.0 to 1.0).
    pub fn inject_homeostasis(&mut self, strength: f64) {
        self.homeostasis = (self.homeostasis + strength).clamp(0.0, 1.0);
    }

    /// Inject a signal into the specified channel.
    pub fn inject(&mut self, channel: ModulatorType, strength: f64) {
        match channel {
            ModulatorType::Reward => self.inject_reward(strength),
            ModulatorType::Novelty => self.inject_novelty(strength),
            ModulatorType::Arousal => self.inject_arousal(strength),
            ModulatorType::Homeostasis => self.inject_homeostasis(strength),
        }
    }

    /// Get the current level of a specific channel.
    pub fn level(&self, channel: ModulatorType) -> f64 {
        match channel {
            ModulatorType::Reward => self.reward,
            ModulatorType::Novelty => self.novelty,
            ModulatorType::Arousal => self.arousal,
            ModulatorType::Homeostasis => self.homeostasis,
        }
    }

    /// Compute the combined modulation signal M(t) for weight updates.
    ///
    /// ẇᵢⱼ = eᵢⱼ · (αᵣ·R(t) + αₙ·N(t) + αₐ·A(t) + αₕ·H(t))
    pub fn combined_signal(&self, alpha_r: f64, alpha_n: f64, alpha_a: f64, alpha_h: f64) -> f64 {
        alpha_r * self.reward
            + alpha_n * self.novelty
            + alpha_a * self.arousal
            + alpha_h * self.homeostasis
    }

    /// Compute the default combined modulation signal.
    pub fn default_signal(&self) -> f64 {
        self.combined_signal(1.0, 0.5, 0.3, 0.1)
    }

    /// Decay all channels towards their resting state.
    pub fn decay(&mut self) {
        self.reward *= self.reward_decay;
        self.novelty *= self.novelty_decay;
        self.arousal *= self.arousal_decay;
        // Homeostasis decays towards 0.5 (baseline)
        self.homeostasis = 0.5 + (self.homeostasis - 0.5) * self.homeostasis_decay;
    }

    /// The current plasticity rate — influenced primarily by novelty.
    /// Higher novelty = faster learning.
    pub fn plasticity_rate(&self) -> f64 {
        0.01 + 0.09 * self.novelty // base rate 0.01, up to 0.10 with full novelty
    }
}
