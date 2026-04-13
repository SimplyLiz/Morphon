//! Limbic Circuit — salience detection, motivational drive, and episodic tagging.
//!
//! The evaluative layer for morphogenic intelligence. While Endoquilibrium handles
//! global homeostatic regulation, the Limbic Circuit provides per-stimulus evaluation:
//! "Is this input novel? How much reward do I expect? Is this worth remembering?"
//!
//! Three components, each biologically grounded:
//! - [`SalienceDetector`] — amygdala analog. Detects novelty + reward expectation.
//! - [`MotivationalDrive`] — nucleus accumbens analog. Tracks RPE per stimulus class.
//! - [`EpisodicTagger`] — hippocampal analog. Records high-salience episodes for replay.
//!
//! All components are gated by `LimbicCircuit::enabled` (default: false). When disabled
//! the circuit is a zero-cost no-op — existing behavior is unchanged.
//!
//! ## Integration in examples
//!
//! ```rust,ignore
//! // Before processing the sample:
//! let output = system.process_with_limbic(&input, Some(label));
//!
//! // After computing correctness and calling reward_contrastive():
//! if correct {
//!     system.deliver_limbic_reward(&input, label, 0.5);
//! }
//! ```

use crate::types::RingBuffer;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// ─── Salience Detector ───────────────────────────────────────────────────────

/// Evaluates each input for novelty and reward association before cortical processing.
/// Analog to the amygdala's rapid ("low road") salience assessment.
///
/// Output: scalar salience ∈ [0, 1].
/// High salience → more arousal → more morphons fire → broader feature exploration.
/// Low salience → baseline arousal → only tuned specialists fire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalienceDetector {
    /// Exponential moving average of input components — the "familiar" baseline.
    input_ema: Vec<f64>,
    /// Running variance of input components — used to normalise deviation.
    input_var: Vec<f64>,
    /// EMA decay rate. Slow (0.99) = long memory; fast (0.9) = adapts quickly.
    #[serde(default = "default_ema_decay")]
    pub ema_decay: f64,
    /// Deviation from EMA (in std-devs) required to trigger high salience.
    #[serde(default = "default_salience_threshold")]
    pub salience_threshold: f64,
    /// Per-pattern reward association: input_hash → EMA of reward received.
    reward_association: HashMap<u64, f64>,
    /// Cap on reward_association size (prevents unbounded growth).
    #[serde(default = "default_max_associations")]
    pub max_associations: usize,
    /// Salience score for the most recently evaluated input.
    pub current_salience: f64,
    /// Novelty component of the last salience score (for diagnostics).
    pub novelty_component: f64,
    /// Reward-expectation component of the last salience score (for diagnostics).
    pub reward_expectation_component: f64,
}

fn default_ema_decay() -> f64 { 0.99 }
fn default_salience_threshold() -> f64 { 1.5 }
fn default_max_associations() -> usize { 1000 }

impl SalienceDetector {
    pub fn new(input_size: usize) -> Self {
        Self {
            input_ema: vec![0.0; input_size],
            input_var: vec![1.0; input_size],
            ema_decay: default_ema_decay(),
            salience_threshold: default_salience_threshold(),
            reward_association: HashMap::new(),
            max_associations: default_max_associations(),
            current_salience: 0.5,
            novelty_component: 0.0,
            reward_expectation_component: 0.0,
        }
    }

    /// Evaluate the salience of `input` before it enters the cortical path.
    /// Returns salience ∈ [0, 1] and caches the result in `current_salience`.
    pub fn evaluate(&mut self, input: &[f64]) -> f64 {
        // Novelty: mean z-score deviation from running EMA.
        let mut deviation_sum = 0.0;
        let mut count = 0.0;
        for (i, &val) in input.iter().enumerate() {
            if i >= self.input_ema.len() { break; }
            let diff = (val - self.input_ema[i]).abs();
            let std = self.input_var[i].sqrt().max(0.01);
            deviation_sum += diff / std;
            count += 1.0;
        }
        let mean_deviation = if count > 0.0 { deviation_sum / count } else { 0.0 };
        // Sigmoid centred on salience_threshold.
        self.novelty_component =
            1.0 / (1.0 + (-(mean_deviation - self.salience_threshold) * 2.0).exp());

        // Reward expectation: has this pattern been rewarding in the past?
        let input_hash = Self::hash_input(input);
        let expected_reward = self.reward_association.get(&input_hash).copied().unwrap_or(0.0);
        // High |expected_reward| → high salience (rewarding OR aversive patterns matter).
        self.reward_expectation_component = (expected_reward.abs() * 2.0).min(1.0);

        self.current_salience = self.novelty_component
            .max(self.reward_expectation_component)
            .clamp(0.0, 1.0);

        // Update running statistics.
        let alpha = 1.0 - self.ema_decay;
        for (i, &val) in input.iter().enumerate() {
            if i >= self.input_ema.len() { break; }
            let old_ema = self.input_ema[i];
            self.input_ema[i] = self.ema_decay * old_ema + alpha * val;
            let diff = val - self.input_ema[i];
            self.input_var[i] = self.ema_decay * self.input_var[i] + alpha * diff * diff;
        }

        self.current_salience
    }

    /// Evaluate salience from the model's readout output (post-cortical path).
    ///
    /// This is the "high road" salience signal — computed *after* the cortical path
    /// has processed the input, using the model's own uncertainty as the salience proxy.
    ///
    /// **Why this works better than pixel-EMA for classification:**
    /// Pixel-EMA novelty stays ~1.0 for MNIST (every sparse digit deviates from the
    /// grey mean). Readout uncertainty correctly tracks per-sample difficulty: if the
    /// model is confident (peaked output), the input is familiar → low salience.
    /// If the model is uncertain (flat output), the input is hard/novel → high salience.
    ///
    /// salience = 1 − (max_output / softmax_sum) = 1 − confidence
    ///
    /// Updates `current_salience` and `novelty_component`. Returns salience ∈ [0, 1].
    /// Call after `present_image()` / `read_output()` and before `inject_arousal()`.
    pub fn evaluate_from_output(&mut self, readout: &[f64]) -> f64 {
        if readout.is_empty() {
            self.current_salience = 0.5;
            return 0.5;
        }
        // Softmax-based confidence: numerically stable via max-subtraction.
        let max_val = readout.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = readout.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f64 = exps.iter().sum();
        let confidence = if sum_exp > 0.0 { exps.iter().cloned().fold(f64::NEG_INFINITY, f64::max) / sum_exp } else { 1.0 / readout.len() as f64 };
        // salience = uncertainty: confident model → low salience, uncertain → high
        let salience = (1.0 - confidence).clamp(0.0, 1.0);
        self.novelty_component = salience; // reuse field; reward_expectation_component unchanged
        self.current_salience = salience
            .max(self.reward_expectation_component)
            .clamp(0.0, 1.0);
        self.current_salience
    }

    /// Record the reward outcome for an input after processing.
    /// Updates the reward association map so future similar inputs are recognised.
    pub fn record_outcome(&mut self, input: &[f64], reward: f64) {
        let hash = Self::hash_input(input);
        let entry = self.reward_association.entry(hash).or_insert(0.0);
        *entry = 0.9 * *entry + 0.1 * reward;

        // Evict least-informative entry when at capacity.
        if self.reward_association.len() > self.max_associations {
            if let Some((&key, _)) = self.reward_association
                .iter()
                .min_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap_or(std::cmp::Ordering::Equal))
            {
                self.reward_association.remove(&key);
            }
        }
    }

    /// Coarse hash of input for pattern matching.
    /// Quantises to 4 levels and hashes first 50 dimensions — groups similar inputs.
    pub fn hash_input(input: &[f64]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &val in input.iter().take(50) {
            let quantized = (val * 4.0) as i32;
            quantized.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Number of distinct input patterns with stored reward associations.
    pub fn associations_stored(&self) -> usize {
        self.reward_association.len()
    }
}

// ─── Motivational Drive ───────────────────────────────────────────────────────

/// Tracks reward expectation per stimulus class and computes Reward Prediction Error (RPE).
/// Analog to the nucleus accumbens / VTA dopaminergic system.
///
/// RPE = received_reward − expected_reward.
/// Positive RPE → surprising success → boost learning and energy.
/// Negative RPE → unexpected failure → trigger exploration.
/// Zero RPE → as predicted → maintenance only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotivationalDrive {
    /// Expected reward per stimulus key (class index or state hash).
    reward_expectation: HashMap<u64, f64>,
    /// Fallback expected reward when a stimulus key is unseen.
    pub global_expectation: f64,
    /// EMA decay for reward expectation updates.
    #[serde(default = "default_expectation_decay")]
    pub expectation_decay: f64,
    /// RPE for the most recently processed stimulus.
    pub current_rpe: f64,
    /// Motivational state derived from recent RPE history.
    pub drive_state: DriveState,
    /// Rolling RPE history for drive-state detection.
    rpe_history: RingBuffer,
}

/// Motivational state derived from the recent RPE trend.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DriveState {
    /// Consistently positive RPE — learning is productive, reward exceeds expectation.
    Seeking,
    /// Consistently negative RPE — strategy is failing, reward below expectation.
    Avoiding,
    /// Mixed RPE — uncertain, maintain current approach.
    Neutral,
}

fn default_expectation_decay() -> f64 { 0.95 }

impl MotivationalDrive {
    pub fn new() -> Self {
        Self {
            reward_expectation: HashMap::new(),
            global_expectation: 0.0,
            expectation_decay: default_expectation_decay(),
            current_rpe: 0.0,
            drive_state: DriveState::Neutral,
            rpe_history: RingBuffer::new(50),
        }
    }

    /// Compute RPE for a stimulus identified by `stimulus_key`, given `received_reward`.
    /// Updates reward expectation and drive state.
    pub fn compute_rpe(&mut self, stimulus_key: u64, received_reward: f64) -> f64 {
        let expected = self.reward_expectation
            .get(&stimulus_key)
            .copied()
            .unwrap_or(self.global_expectation);

        self.current_rpe = received_reward - expected;

        let d = self.expectation_decay;
        let entry = self.reward_expectation.entry(stimulus_key).or_insert(0.0);
        *entry = d * *entry + (1.0 - d) * received_reward;
        self.global_expectation = d * self.global_expectation + (1.0 - d) * received_reward;

        self.rpe_history.push(self.current_rpe);
        self.update_drive_state();

        self.current_rpe
    }

    /// Compute RPE using a class label (convenience wrapper for classification tasks).
    pub fn compute_rpe_for_class(&mut self, class_label: usize, received_reward: f64) -> f64 {
        self.compute_rpe(class_label as u64, received_reward)
    }

    /// Get the expected reward for a stimulus (useful before processing for pre-decisions).
    pub fn expected_reward(&self, stimulus_key: u64) -> f64 {
        self.reward_expectation
            .get(&stimulus_key)
            .copied()
            .unwrap_or(self.global_expectation)
    }

    /// Expected reward by class label.
    pub fn expected_reward_for_class(&self, class_label: usize) -> f64 {
        self.expected_reward(class_label as u64)
    }

    /// Per-class expected reward array for diagnostics. Returns up to `n_classes` values.
    pub fn class_expectations(&self, n_classes: usize) -> Vec<f64> {
        (0..n_classes)
            .map(|c| self.expected_reward(c as u64))
            .collect()
    }

    fn update_drive_state(&mut self) {
        let mean_rpe = self.rpe_history.mean();
        self.drive_state = if mean_rpe > 0.1 {
            DriveState::Seeking
        } else if mean_rpe < -0.1 {
            DriveState::Avoiding
        } else {
            DriveState::Neutral
        };
    }
}

impl Default for MotivationalDrive {
    fn default() -> Self { Self::new() }
}

// ─── Episodic Tagger ─────────────────────────────────────────────────────────

/// Records high-salience experiences for later replay and prioritised consolidation.
/// Analog to hippocampal episodic memory formation.
///
/// Episodes are gated by `recording_threshold` — only salient experiences are stored.
/// Replay sampling is weighted by `replay_priority = salience × |RPE|`, so the system
/// revisits surprising, important events more than routine ones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicTagger {
    /// Fixed-size ring buffer of episodes.
    episodes: Vec<EpisodeRecord>,
    /// Maximum number of stored episodes.
    #[serde(default = "default_max_episodes")]
    pub max_episodes: usize,
    /// Minimum salience for recording.
    #[serde(default = "default_recording_threshold")]
    pub recording_threshold: f64,
    /// Write pointer (ring-buffer position).
    write_idx: usize,
}

/// A single recorded episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeRecord {
    /// Coarse hash of the input pattern (from SalienceDetector::hash_input).
    pub input_hash: u64,
    /// Raw input snapshot for replay. Enables re-presenting the exact stimulus.
    pub input: Vec<f64>,
    /// Morphon IDs that fired during this episode.
    pub active_morphons: Vec<u64>,
    /// Reward received.
    pub reward: f64,
    /// RPE at time of recording.
    pub rpe: f64,
    /// Salience score at time of recording.
    pub salience: f64,
    /// Class label (classification tasks) or None (RL/unsupervised).
    pub label: Option<usize>,
    /// System step_count when recorded.
    pub tick: u64,
    /// Replay priority = salience × |RPE|. Higher = sampled more often.
    pub replay_priority: f64,
}

fn default_max_episodes() -> usize { 500 }
fn default_recording_threshold() -> f64 { 0.3 }

impl EpisodicTagger {
    pub fn new(max_episodes: usize) -> Self {
        Self {
            episodes: Vec::with_capacity(max_episodes),
            max_episodes,
            recording_threshold: default_recording_threshold(),
            write_idx: 0,
        }
    }

    /// Record an episode if `salience` exceeds the recording threshold.
    pub fn maybe_record(
        &mut self,
        input: &[f64],
        active_morphons: &[u64],
        reward: f64,
        rpe: f64,
        salience: f64,
        label: Option<usize>,
        tick: u64,
    ) {
        if salience < self.recording_threshold { return; }

        let record = EpisodeRecord {
            input_hash: SalienceDetector::hash_input(input),
            input: input.to_vec(),
            active_morphons: active_morphons.to_vec(),
            reward,
            rpe,
            salience,
            label,
            tick,
            replay_priority: salience * rpe.abs(),
        };

        if self.episodes.len() < self.max_episodes {
            self.episodes.push(record);
        } else {
            self.episodes[self.write_idx] = record;
        }
        self.write_idx = (self.write_idx + 1) % self.max_episodes.max(1);
    }

    /// Sample up to `n` episodes, weighted by replay_priority.
    /// High-priority episodes (high salience × high |RPE|) are sampled more often.
    pub fn sample_for_replay<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<&EpisodeRecord> {
        if self.episodes.is_empty() { return Vec::new(); }

        let total_priority: f64 = self.episodes.iter()
            .map(|e| e.replay_priority.max(0.01))
            .sum();

        let mut sampled = Vec::with_capacity(n);
        for _ in 0..n {
            let mut target = rng.random::<f64>() * total_priority;
            for episode in &self.episodes {
                target -= episode.replay_priority.max(0.01);
                if target <= 0.0 {
                    sampled.push(episode);
                    break;
                }
            }
        }
        sampled
    }

    /// All episodes for a specific class label.
    pub fn episodes_for_class(&self, class: usize) -> Vec<&EpisodeRecord> {
        self.episodes.iter().filter(|e| e.label == Some(class)).collect()
    }

    /// Aggregate statistics for diagnostics output.
    pub fn stats(&self) -> EpisodicStats {
        let count = self.episodes.len();
        let mean_salience = if count > 0 {
            self.episodes.iter().map(|e| e.salience).sum::<f64>() / count as f64
        } else { 0.0 };
        let mean_rpe = if count > 0 {
            self.episodes.iter().map(|e| e.rpe.abs()).sum::<f64>() / count as f64
        } else { 0.0 };
        EpisodicStats { count, mean_salience, mean_rpe }
    }
}

/// Aggregate statistics from the EpisodicTagger for logging / JSON output.
#[derive(Debug, Clone)]
pub struct EpisodicStats {
    pub count: usize,
    pub mean_salience: f64,
    pub mean_rpe: f64,
}

// ─── LimbicCircuit ────────────────────────────────────────────────────────────

/// The Limbic Circuit — evaluative layer for morphogenic intelligence.
///
/// Provides per-stimulus salience detection (amygdala), reward prediction with RPE
/// (nucleus accumbens), and episodic memory formation (hippocampus).
///
/// All behaviour is gated by `enabled`. When `false` every call is a no-op and
/// existing System behaviour is fully preserved — safe to add to System without
/// touching any example that doesn't opt in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimbicCircuit {
    pub salience_detector: SalienceDetector,
    pub motivational_drive: MotivationalDrive,
    pub episodic_tagger: EpisodicTagger,
    /// Master enable switch. Default false — zero-cost no-op when disabled.
    #[serde(default)]
    pub enabled: bool,
}

impl LimbicCircuit {
    /// Create a new Limbic Circuit sized for `input_size`-dimensional inputs.
    /// Disabled by default — call `circuit.enabled = true` to activate.
    pub fn new(input_size: usize) -> Self {
        Self {
            salience_detector: SalienceDetector::new(input_size),
            motivational_drive: MotivationalDrive::new(),
            episodic_tagger: EpisodicTagger::new(500),
            enabled: false,
        }
    }

    /// Salience score for the last evaluated input (0.0 if never called).
    pub fn current_salience(&self) -> f64 {
        self.salience_detector.current_salience
    }

    /// Current RPE from the most recent reward delivery.
    pub fn current_rpe(&self) -> f64 {
        self.motivational_drive.current_rpe
    }

    /// Current motivational drive state.
    pub fn drive_state(&self) -> DriveState {
        self.motivational_drive.drive_state
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn salience_high_for_novel_input() {
        let mut sd = SalienceDetector::new(10);
        // Warm up with zeros long enough for variance EMA to converge.
        // ema_decay=0.99 means var decays as 0.99^N from 1.0; need ~500 steps
        // to get std low enough that a |1.0 - 0.0| deviation clears the threshold.
        let zeros = vec![0.0_f64; 10];
        for _ in 0..500 {
            sd.evaluate(&zeros);
        }
        // Novel input far from established baseline should score high.
        let novel = vec![1.0_f64; 10];
        let salience = sd.evaluate(&novel);
        assert!(salience > 0.5, "novel input salience={salience:.3} expected >0.5");
    }

    #[test]
    fn salience_low_for_repeated_input() {
        let mut sd = SalienceDetector::new(10);
        let repeated = vec![0.5_f64; 10];
        // Repeat enough times for variance EMA to adapt to this pattern.
        for _ in 0..500 {
            sd.evaluate(&repeated);
        }
        let salience = sd.evaluate(&repeated);
        assert!(salience < 0.5, "repeated input salience={salience:.3} expected <0.5");
    }

    #[test]
    fn rpe_positive_for_surprise() {
        let mut md = MotivationalDrive::new();
        // No prior experience → expected_reward = 0.
        let rpe = md.compute_rpe_for_class(3, 0.8);
        // received(0.8) - expected(0.0) = 0.8
        assert!(rpe > 0.5, "surprising correct rpe={rpe:.3} expected >0.5");
    }

    #[test]
    fn rpe_near_zero_for_expected_reward() {
        let mut md = MotivationalDrive::new();
        // Train the expectation toward 0.8.
        for _ in 0..100 {
            md.compute_rpe_for_class(3, 0.8);
        }
        // After convergence, RPE should be near zero.
        let rpe = md.compute_rpe_for_class(3, 0.8);
        assert!(rpe.abs() < 0.1, "converged rpe={rpe:.3} expected ≈0");
    }

    #[test]
    fn episodic_tagger_records_above_threshold() {
        let mut tagger = EpisodicTagger::new(100);
        let input = vec![0.5_f64; 10];
        let morphons = vec![1u64, 2, 3];
        // Below threshold — should not record.
        tagger.maybe_record(&input, &morphons, 0.5, 0.4, 0.1, Some(0), 1);
        assert_eq!(tagger.stats().count, 0);
        // Above threshold — should record.
        tagger.maybe_record(&input, &morphons, 0.5, 0.4, 0.8, Some(0), 2);
        assert_eq!(tagger.stats().count, 1);
    }

    #[test]
    fn episodic_tagger_ring_buffer_wraps() {
        let mut tagger = EpisodicTagger::new(3);
        let input = vec![0.5_f64; 5];
        let morphons = vec![1u64];
        for tick in 0..6 {
            tagger.maybe_record(&input, &morphons, 0.5, 0.5, 0.9, Some(0), tick);
        }
        // Ring buffer caps at max_episodes.
        assert_eq!(tagger.stats().count, 3);
    }

    #[test]
    fn limbic_circuit_disabled_by_default() {
        let circuit = LimbicCircuit::new(10);
        assert!(!circuit.enabled);
    }
}
