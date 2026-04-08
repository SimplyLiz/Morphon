# Limbic Circuit — Salience Detection, Episodic Tagging & Motivational Drive
## The Evaluative Layer for Morphogenic Intelligence
### Technical Specification v1.0 — TasteHub GmbH, April 2026

---

| | |
|---|---|
| **System name** | Limbic Circuit |
| **Components** | Salience Detector (amygdala), Episodic Tagger (hippocampus), Motivational Drive (nucleus accumbens) |
| **Not part of** | Endoquilibrium (Endo is regulation; Limbic is evaluation) |
| **Lives in** | New module: `src/limbic.rs` |
| **Depends on** | Endoquilibrium V1 (implemented), reward delivery pipeline |
| **Recommended after** | Connectivity fix (full S→A), metabolic pressure validation |
| **Effort** | Salience Detector: 4–6 hours. Episodic Tagger: 3–4 hours. Motivational Drive: 2–3 hours. |
| **Priority** | Salience Detector = Phase 2D. Episodic Tagger = Phase 8 (ANCS-Core). Motivational Drive = Phase 2D alongside Salience. |

---

## 1. What's Missing: The Evaluative Gap

### 1.1 The Biological Limbic System

The limbic system is not a single structure — it's a circuit of interacting brain regions that evaluate incoming stimuli for biological significance and coordinate the appropriate behavioral and physiological response. It sits between perception (cortex) and regulation (hypothalamus/endocrine), providing the critical "is this important?" assessment that neither perception nor regulation can provide alone.

| Structure | Function | Timescale | Pathway |
|---|---|---|---|
| **Amygdala** | Salience detection — is this novel, threatening, rewarding? | Fast (50–200ms) | Direct from thalamus (low road) + from cortex (high road) |
| **Hippocampus** | Episodic memory — what happened last time in this context? | Medium (200ms–seconds) | From cortex + amygdala, outputs to cortex + hypothalamus |
| **Nucleus accumbens** | Reward prediction — how much reward do I expect? RPE = received - expected | Fast (100–300ms) | Dopaminergic input from VTA, outputs to motor and prefrontal |
| **Cingulate cortex** | Conflict monitoring — are my predictions conflicting? | Medium (200–500ms) | From multiple cortical areas, outputs to prefrontal + motor |
| **Hypothalamus** | Homeostatic drives — am I hungry, tired, threatened? | Slow (seconds–minutes) | From limbic structures, outputs to endocrine + autonomic |

### 1.2 What MORPHON Currently Has

| Limbic function | Current implementation | Quality |
|---|---|---|
| Homeostatic regulation | Endoquilibrium (7 vitals, 6 rules, 5 stages) | Good |
| Reward delivery | inject_reward(), reward_contrastive() | Partial — global broadcast, no per-stimulus valuation |
| Prediction error | PE sensing in Endo, checkpoint/rollback | Partial — aggregate PE, not per-stimulus |
| Memory consolidation | Tag-and-capture, episode-gated capture | Partial — synaptic consolidation, no episodic indexing |
| Salience detection | **Missing** | Not covered |
| Reward prediction / RPE | **Missing** | Not covered |
| Conflict monitoring | **Missing** | Not covered |
| Per-stimulus attention gating | **Missing** | Not covered |

### 1.3 Why This Matters for Current Bottlenecks

**The hub dominance problem** is partly a salience problem. Every input receives equal processing resources — digit "1" (easy, common) and digit "8" (hard, complex) get the same arousal, the same attention, the same learning investment. In biology, the amygdala detects that digit "8" is more surprising/difficult and triggers an arousal burst that allocates more processing resources to it.

**The metabolic selection problem** is partly a reward prediction problem. Currently, reward-correlated energy rewards all morphons that fire during a correct classification equally. But a correct classification of digit "1" (which the system already gets right 95% of the time) shouldn't generate the same metabolic reward as a correct classification of digit "8" (which the system almost never gets right). The reward prediction error (RPE = actual - expected) would make energy allocation proportional to surprise, not just correctness.

**The learning efficiency problem** is partly an episodic tagging problem. The system trains on 3000 images but treats each one independently. There's no mechanism to say "I've seen this type of pattern before and it was rewarding — allocate more processing." Episodic tagging enables experience replay and prioritized learning.

---

## 2. Architecture: Where the Limbic Circuit Lives

```
Input arrives
    │
    ├──────────────────────→ [Sensory → Associative → Motor]
    │                         (cortical path: slow, detailed, full features)
    │                                    ↑
    │                                    │ arousal_modulation (per-stimulus)
    │                                    │ rpe_energy_scaling
    │                                    │
    └──→ [Salience Detector] ──→ [Motivational Drive] ──→ Endo arousal channel
          (fast, coarse)          (RPE computation)         metabolic energy scaling
               │                        │
               │                        │
               └──→ [Episodic Tagger] ──┘
                    (stores episode records for replay)
```

The limbic circuit runs **in parallel** with the cortical path but resolves **faster** (direct sensory connections, simpler processing). By the time the associative layer has finished integrating features, the limbic system has already decided:

1. How novel/surprising is this input? (Salience Detector)
2. How much reward do I expect from it? (Motivational Drive)
3. Should I store this experience for later? (Episodic Tagger)

The limbic outputs modulate the cortical path's processing — they don't replace it.

### 2.1 Relationship to Existing Systems

| System | Relationship to Limbic |
|---|---|
| **Endoquilibrium** | Endo sets global baselines. Limbic provides per-stimulus modulation on top of those baselines. Endo says "the system is in Mature stage, base arousal = 0.3." Limbic says "this particular input is novel, boost arousal to 0.8 for this stimulus only." |
| **Metabolic system** | Limbic's RPE scales the reward-correlated energy signal. Instead of flat energy for correct classification, morphons earn energy proportional to RPE × fired. High RPE (surprising correct) = more energy. Low RPE (expected correct) = less energy. |
| **Reward delivery** | Limbic wraps the existing reward pipeline. inject_reward() still works as before. Limbic adds reward expectation tracking and RPE computation on top. |
| **Tag-and-capture** | Episodic Tagger can provide prioritized replay targets for consolidation during Endo's Mature stage. |
| **Competition** | Salience-modulated arousal affects firing thresholds — high-salience inputs lower thresholds (more morphons fire, more features explored), low-salience inputs raise thresholds (fewer morphons fire, only tuned specialists respond). |

---

## 3. Component 1: Salience Detector (Amygdala Analog)

### 3.1 What It Does

The Salience Detector evaluates each input for novelty and importance before the cortical path has finished processing. It produces a scalar salience score ∈ [0, 1] that modulates arousal and attention for this specific stimulus.

**High salience** (novel input, unexpected pattern, previously-rewarding stimulus):
- Arousal boosted → more morphons fire → broader feature exploration
- Learning rate increased → STDP updates are larger for this stimulus
- Episodic Tagger activated → this experience is worth remembering

**Low salience** (familiar input, expected pattern, previously-neutral stimulus):
- Arousal unchanged or reduced → only tuned specialists fire
- Learning rate at baseline → maintenance learning only
- Episodic Tagger not activated → routine, don't waste memory

### 3.2 Biological Basis

The amygdala receives input through two pathways: the "low road" (direct from thalamus — fast, coarse, 12ms) and the "high road" (from cortex — slow, detailed, 30ms). The low road provides a rapid initial evaluation: "Is this input similar to something I've experienced before? Was that experience rewarding or threatening?" This evaluation happens before the cortex has finished feature extraction.

In MORPHON, the "low road" is a direct connection from sensory morphons to the salience detector, bypassing the associative layer. The "high road" evaluation comes from the associative layer's output, but by then the salience detector has already set the attentional tone.

### 3.3 Implementation

```rust
/// The Salience Detector — per-stimulus evaluation of novelty and importance.
/// Analog to the amygdala's rapid salience assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalienceDetector {
    /// Running statistics of input patterns for novelty detection.
    /// Exponential moving average of input vector components.
    input_ema: Vec<f64>,
    
    /// Running variance of input patterns.
    input_var: Vec<f64>,
    
    /// EMA decay rate — how fast the "familiar" model adapts.
    /// Slow decay = long memory = only truly novel inputs trigger salience.
    /// Fast decay = short memory = most inputs trigger salience.
    #[serde(default = "default_ema_decay")]
    pub ema_decay: f64,
    
    /// Salience threshold — deviation from EMA needed to trigger high salience.
    /// Measured in standard deviations.
    #[serde(default = "default_salience_threshold")]
    pub salience_threshold: f64,
    
    /// Association memory: input_hash → average reward received.
    /// Tracks whether similar inputs were previously rewarding.
    reward_association: HashMap<u64, f64>,
    
    /// Maximum entries in reward association (prevents unbounded growth).
    #[serde(default = "default_max_associations")]
    pub max_associations: usize,
    
    /// Current output — the salience score for the most recent input.
    pub current_salience: f64,
    
    /// Breakdown of salience components for diagnostics.
    pub novelty_component: f64,
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
    
    /// Evaluate the salience of an input vector.
    /// Called BEFORE the input is fed to the cortical path.
    /// Returns salience ∈ [0, 1].
    pub fn evaluate(&mut self, input: &[f64]) -> f64 {
        // === Novelty detection: how far is this input from the running average? ===
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
        
        // Novelty score: sigmoid of deviation relative to threshold
        self.novelty_component = 1.0 / (1.0 + (-(mean_deviation - self.salience_threshold) * 2.0).exp());
        
        // === Reward expectation: has this type of input been rewarding before? ===
        let input_hash = Self::hash_input(input);
        let expected_reward = self.reward_association.get(&input_hash).copied().unwrap_or(0.0);
        // High expected reward → high salience (pay attention to rewarding stimuli)
        // But also: very LOW expected reward → moderate salience (pay attention to threats)
        // Neutral → low salience (boring)
        self.reward_expectation_component = (expected_reward.abs() * 2.0).min(1.0);
        
        // === Combined salience: max of novelty and reward expectation ===
        self.current_salience = self.novelty_component
            .max(self.reward_expectation_component)
            .clamp(0.0, 1.0);
        
        // === Update running statistics ===
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
    
    /// Record the reward outcome for an input pattern.
    /// Called after the system processes the input and receives reward.
    pub fn record_outcome(&mut self, input: &[f64], reward: f64) {
        let hash = Self::hash_input(input);
        let entry = self.reward_association.entry(hash).or_insert(0.0);
        *entry = 0.9 * *entry + 0.1 * reward;  // EMA of reward for this pattern
        
        // Evict old entries if at capacity (LRU would be better, but HashMap is simpler)
        if self.reward_association.len() > self.max_associations {
            // Remove entry with smallest absolute reward (least informative)
            if let Some((&key, _)) = self.reward_association.iter()
                .min_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            {
                self.reward_association.remove(&key);
            }
        }
    }
    
    /// Fast hash of input vector for pattern matching.
    /// Coarse — intentionally groups similar inputs together.
    fn hash_input(input: &[f64]) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        // Quantize to 4 levels and hash — groups similar inputs
        for &val in input.iter().take(50) {  // only hash first 50 dimensions for speed
            let quantized = (val * 4.0) as i32;
            quantized.hash(&mut hasher);
        }
        hasher.finish()
    }
}
```

### 3.4 Integration Point: Arousal Modulation

The salience score modulates the Endoquilibrium arousal channel on a per-stimulus basis:

```rust
// In system.rs, BEFORE feed_input():
let salience = self.limbic.salience_detector.evaluate(&observation);

// Modulate arousal for this stimulus
let base_arousal = self.endo.channels.arousal_gain;
let stimulus_arousal = base_arousal * (0.5 + salience);  // range: [0.5×, 1.5×] of base
self.modulation.inject_arousal_override(stimulus_arousal);

// After processing + reward:
self.limbic.salience_detector.record_outcome(&observation, reward);
```

This means:
- Novel/important stimuli get 1.5× the base arousal → more morphons fire, more features explored
- Familiar/neutral stimuli get 0.5× the base arousal → only tuned specialists fire, maintenance mode
- The base arousal is still set by Endoquilibrium's developmental stage — the limbic system modulates on top

---

## 4. Component 2: Motivational Drive (Nucleus Accumbens Analog)

### 4.1 What It Does

The Motivational Drive tracks **reward expectation** per stimulus class and computes the **Reward Prediction Error (RPE)**. RPE is the difference between received reward and expected reward — the dopaminergic learning signal that drives both the VTA→cortex teaching signal and the metabolic energy allocation.

### 4.2 Biological Basis

The nucleus accumbens is the primary target of dopaminergic neurons in the VTA (ventral tegmental area). When reward exceeds expectation, dopamine bursts — driving learning, motivation, and approach behavior. When reward falls below expectation, dopamine dips — signaling that something went wrong and triggering exploration or avoidance. When reward matches expectation, dopamine is unchanged — nothing to learn.

This is the Temporal Difference (TD) error that RL uses. MORPHON already has TD error injection in CartPole (`inject_td_error()`). The Motivational Drive extends this to classification tasks where TD error isn't naturally defined.

### 4.3 Implementation

```rust
/// Motivational Drive — reward prediction and RPE computation.
/// Analog to the nucleus accumbens / VTA dopaminergic system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotivationalDrive {
    /// Expected reward per stimulus class.
    /// For classification: class_index → EMA of reward.
    /// For RL: state_hash → EMA of reward.
    reward_expectation: HashMap<u64, f64>,
    
    /// Global expected reward (when class/state is unknown).
    global_expectation: f64,
    
    /// EMA decay for reward expectations.
    #[serde(default = "default_expectation_decay")]
    pub expectation_decay: f64,
    
    /// Current RPE — the output signal.
    /// Positive: better than expected → boost learning.
    /// Negative: worse than expected → trigger exploration.
    /// Zero: as expected → maintenance.
    pub current_rpe: f64,
    
    /// Motivational state derived from recent RPE history.
    pub drive_state: DriveState,
    
    /// RPE history for drive state detection.
    rpe_history: RingBuffer,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DriveState {
    /// Consistently positive RPE — things are going well, learning is productive.
    Seeking,
    /// Consistently negative RPE — things are going badly, need to change strategy.
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
    
    /// Compute RPE for a given stimulus and received reward.
    /// Call this when reward is delivered after processing a stimulus.
    pub fn compute_rpe(&mut self, stimulus_key: u64, received_reward: f64) -> f64 {
        // Get expected reward for this stimulus type
        let expected = self.reward_expectation
            .get(&stimulus_key)
            .copied()
            .unwrap_or(self.global_expectation);
        
        // RPE = received - expected
        self.current_rpe = received_reward - expected;
        
        // Update expectation for this stimulus
        let entry = self.reward_expectation.entry(stimulus_key).or_insert(0.0);
        *entry = self.expectation_decay * *entry + (1.0 - self.expectation_decay) * received_reward;
        
        // Update global expectation
        self.global_expectation = self.expectation_decay * self.global_expectation
            + (1.0 - self.expectation_decay) * received_reward;
        
        // Update RPE history and drive state
        self.rpe_history.push(self.current_rpe);
        self.update_drive_state();
        
        self.current_rpe
    }
    
    /// Compute RPE using a class label (for classification tasks).
    pub fn compute_rpe_for_class(&mut self, class_label: usize, received_reward: f64) -> f64 {
        self.compute_rpe(class_label as u64, received_reward)
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
    
    /// Get the expected reward for a stimulus (for pre-processing decisions).
    pub fn expected_reward(&self, stimulus_key: u64) -> f64 {
        self.reward_expectation
            .get(&stimulus_key)
            .copied()
            .unwrap_or(self.global_expectation)
    }
}
```

### 4.4 Integration Point: RPE-Scaled Energy

The RPE replaces raw reward in the metabolic energy calculation:

```rust
// In the reward-correlated energy loop (system.rs):
let rpe = self.limbic.motivational_drive.compute_rpe_for_class(label, reward);

// RPE-scaled energy instead of raw reward-scaled energy
let reward_energy_coeff = self.config.metabolic.reward_energy_coefficient;
if reward_energy_coeff > 0.0 && rpe > 0.0 {
    for m in self.morphons.values_mut() {
        if m.fired {
            m.energy = (m.energy + rpe * reward_energy_coeff).min(1.0);
        }
    }
}
```

**Why RPE instead of raw reward:**

- Correctly classifying digit "1" (95% accuracy) → expected reward ~0.95 → RPE ≈ 0.05 → tiny energy boost
- Correctly classifying digit "8" (5% accuracy) → expected reward ~0.05 → RPE ≈ 0.95 → large energy boost
- Morphons that fire for easy digits earn almost nothing. Morphons that fire for hard digits earn a lot.
- Hubs that fire for everything get the average RPE across all classes — which approaches zero as the system learns (because expectation catches up to reality).

This is the principled solution to the metabolic selection problem: energy allocation proportional to **surprise**, not just **correctness**. Specialists that crack difficult patterns are metabolically rewarded. Hubs that ride easy patterns are not.

### 4.5 Integration Point: Drive State → Endoquilibrium

The motivational drive state feeds into Endoquilibrium's stage detection:

```rust
// In Endo's regulate() function:
match self.limbic.motivational_drive.drive_state {
    DriveState::Seeking => {
        // Positive RPE trend — boost consolidation, reduce exploration
        // This accelerates convergence when learning is productive
        self.channels.consolidation_gain *= 1.1;
    }
    DriveState::Avoiding => {
        // Negative RPE trend — boost exploration, increase plasticity
        // This triggers the "stressed" response to find new strategies
        self.channels.novelty_gain *= 1.2;
        self.channels.plasticity_mult *= 1.1;
    }
    DriveState::Neutral => {
        // RPE flat — maintenance mode
    }
}
```

---

## 5. Component 3: Episodic Tagger (Hippocampal Analog)

### 5.1 What It Does

The Episodic Tagger creates compact records of important experiences — linking the stimulus, the morphon activation pattern, the reward outcome, and the salience level into an indexed episode. These episodes enable experience replay and prioritized consolidation.

### 5.2 Biological Basis

The hippocampus creates episodic memories by binding cortical representations (what was seen) with amygdala evaluations (how important it was) and reward outcomes (what happened). During sleep, the hippocampus replays these episodes to the cortex, driving synaptic consolidation. High-salience episodes are replayed more frequently — prioritized experience replay is a biological mechanism, not just an RL algorithm.

### 5.3 Implementation

```rust
/// Episodic Tagger — records important experiences for replay.
/// Analog to hippocampal episodic memory formation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicTagger {
    /// Ring buffer of recent episodes.
    episodes: Vec<EpisodeRecord>,
    
    /// Maximum number of stored episodes.
    #[serde(default = "default_max_episodes")]
    pub max_episodes: usize,
    
    /// Salience threshold for recording — only store important episodes.
    #[serde(default = "default_recording_threshold")]
    pub recording_threshold: f64,
    
    /// Write pointer for ring buffer.
    write_idx: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeRecord {
    /// Fast hash of the input pattern.
    pub input_hash: u64,
    
    /// Which morphons fired during this episode.
    /// Stored as a sorted Vec of MorphonIds for compact representation.
    pub active_morphons: Vec<u64>,
    
    /// Reward outcome.
    pub reward: f64,
    
    /// RPE at time of recording.
    pub rpe: f64,
    
    /// Salience at time of recording.
    pub salience: f64,
    
    /// Class label (for classification tasks).
    pub label: Option<usize>,
    
    /// Tick when recorded.
    pub tick: u64,
    
    /// Priority for replay — high-salience, high-RPE episodes are replayed more.
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
    
    /// Record an episode if salience exceeds threshold.
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
        self.write_idx = (self.write_idx + 1) % self.max_episodes;
    }
    
    /// Sample episodes for replay, prioritized by salience × |RPE|.
    /// High-priority episodes are sampled more frequently.
    pub fn sample_for_replay(
        &self,
        n: usize,
        rng: &mut impl Rng,
    ) -> Vec<&EpisodeRecord> {
        if self.episodes.is_empty() { return Vec::new(); }
        
        // Weighted sampling by replay_priority
        let total_priority: f64 = self.episodes.iter()
            .map(|e| e.replay_priority.max(0.01))
            .sum();
        
        let mut sampled = Vec::with_capacity(n);
        for _ in 0..n {
            let mut target = rng.gen::<f64>() * total_priority;
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
    
    /// Get episodes for a specific class (for class-specific replay).
    pub fn episodes_for_class(&self, class: usize) -> Vec<&EpisodeRecord> {
        self.episodes.iter()
            .filter(|e| e.label == Some(class))
            .collect()
    }
    
    /// Statistics for diagnostics.
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

#[derive(Debug, Clone)]
pub struct EpisodicStats {
    pub count: usize,
    pub mean_salience: f64,
    pub mean_rpe: f64,
}
```

### 5.4 Integration Point: Experience Replay

During Endoquilibrium's Mature stage (when the system should consolidate rather than explore), the Episodic Tagger provides replay targets:

```rust
// In the glacial path, during Mature stage:
if self.endo.developmental_stage == DevelopmentalStage::Mature {
    let replay_episodes = self.limbic.episodic_tagger.sample_for_replay(5, &mut self.rng);
    for episode in replay_episodes {
        // Re-activate the stored morphon pattern
        for &morphon_id in &episode.active_morphons {
            if let Some(m) = self.morphons.get_mut(&morphon_id) {
                m.input_accumulator += 0.3;  // gentle reactivation
            }
        }
        // Run a few ticks with the reactivated pattern
        self.step_internal(3);
        // Deliver the stored reward to reinforce the pattern
        self.modulation.inject_reward(episode.reward * 0.5);  // half-strength replay
    }
}
```

This is biological sleep replay — the hippocampus reactivates stored patterns during slow-wave sleep, and the cortex consolidates the associated synaptic changes. In MORPHON, replay happens during Mature stage's glacial ticks, not during active processing.

---

## 6. The Limbic Circuit as a Whole

### 6.1 The Complete Struct

```rust
/// The Limbic Circuit — evaluative layer for morphogenic intelligence.
/// Provides per-stimulus salience detection, reward prediction, and episodic memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimbicCircuit {
    pub salience_detector: SalienceDetector,
    pub motivational_drive: MotivationalDrive,
    pub episodic_tagger: EpisodicTagger,
    
    /// Master switch — if false, all limbic functions are skipped.
    #[serde(default)]
    pub enabled: bool,
}

impl LimbicCircuit {
    pub fn new(input_size: usize) -> Self {
        Self {
            salience_detector: SalienceDetector::new(input_size),
            motivational_drive: MotivationalDrive::new(),
            episodic_tagger: EpisodicTagger::new(500),
            enabled: false,
        }
    }
}
```

### 6.2 The Complete Processing Loop

```rust
// In system.rs, during process_steps() or the MNIST training loop:

fn process_with_limbic(&mut self, observation: &[f64], label: Option<usize>) -> Vec<f64> {
    if !self.limbic.enabled {
        return self.process_steps(observation, self.config.internal_steps);
    }
    
    // === LIMBIC FAST PATH (before cortical processing) ===
    
    // 1. Evaluate salience
    let salience = self.limbic.salience_detector.evaluate(observation);
    
    // 2. Get reward expectation
    let stimulus_key = label.map(|l| l as u64).unwrap_or(0);
    let expected_reward = self.limbic.motivational_drive.expected_reward(stimulus_key);
    
    // 3. Modulate arousal based on salience
    let base_arousal = self.endo.channels.arousal_gain;
    let modulated_arousal = base_arousal * (0.5 + salience);
    
    // Apply per-stimulus arousal (temporarily override Endo's global arousal)
    let original_arousal = self.endo.channels.arousal_gain;
    self.endo.channels.arousal_gain = modulated_arousal;
    
    // === CORTICAL PATH (standard processing) ===
    let output = self.process_steps(observation, self.config.internal_steps);
    
    // Restore original arousal
    self.endo.channels.arousal_gain = original_arousal;
    
    output
}

fn deliver_limbic_reward(&mut self, observation: &[f64], label: usize, reward: f64) {
    if !self.limbic.enabled { return; }
    
    // 1. Compute RPE
    let rpe = self.limbic.motivational_drive.compute_rpe_for_class(label, reward);
    
    // 2. Record outcome in salience detector
    self.limbic.salience_detector.record_outcome(observation, reward);
    
    // 3. RPE-scaled energy for active morphons
    let reward_energy_coeff = self.config.metabolic.reward_energy_coefficient;
    if reward_energy_coeff > 0.0 && rpe > 0.0 {
        for m in self.morphons.values_mut() {
            if m.fired {
                m.energy = (m.energy + rpe * reward_energy_coeff).min(1.0);
            }
        }
    }
    
    // 4. Maybe record episode
    let salience = self.limbic.salience_detector.current_salience;
    let active_morphons: Vec<u64> = self.morphons.values()
        .filter(|m| m.fired)
        .map(|m| m.id)
        .collect();
    self.limbic.episodic_tagger.maybe_record(
        observation, &active_morphons, reward, rpe, salience, Some(label), self.tick,
    );
}
```

---

## 7. Diagnostics

```json
{
  "limbic": {
    "enabled": true,
    "salience": {
      "current": 0.73,
      "novelty_component": 0.68,
      "reward_expectation_component": 0.45,
      "ema_adaptation_rate": 0.99,
      "associations_stored": 342
    },
    "motivation": {
      "current_rpe": 0.31,
      "drive_state": "Seeking",
      "global_expectation": 0.24,
      "per_class_expectation": [0.95, 0.82, 0.15, 0.08, 0.03, 0.12, 0.45, 0.71, 0.02, 0.33]
    },
    "episodes": {
      "stored": 287,
      "mean_salience": 0.61,
      "mean_rpe": 0.42,
      "replays_this_epoch": 15
    }
  }
}
```

**Key diagnostic:** The `per_class_expectation` array shows the system's learned reward expectation per digit. If class 1 has expectation 0.95 and class 8 has expectation 0.02, the system expects to get class 1 right and class 8 wrong. When class 8 is correctly classified, RPE = 0.98 — a massive surprise that drives strong learning. When class 1 is correctly classified, RPE = 0.05 — expected, minimal learning.

---

## 8. Implementation Plan

### Phase 1: Salience Detector + Motivational Drive (4–5 hours)

| Step | What | Lines |
|---|---|---|
| 1 | Create `src/limbic.rs` with LimbicCircuit, SalienceDetector, MotivationalDrive structs | 150 |
| 2 | Add `limbic: LimbicCircuit` to System struct, initialize in System::new() | 10 |
| 3 | Add `LimbicConfig` to SystemConfig with `enabled: false` default | 15 |
| 4 | Integrate salience evaluation before feed_input() in process_steps() | 10 |
| 5 | Integrate RPE computation in reward delivery path | 10 |
| 6 | Replace raw reward with RPE in reward-correlated energy loop | 5 |
| 7 | Tests: salience scores high for novel input, low for repeated; RPE positive for surprise, zero for expected | 30 |
| 8 | Validate: CartPole unchanged (limbic disabled by default) | Run |

### Phase 2: Episodic Tagger (3–4 hours)

| Step | What | Lines |
|---|---|---|
| 9 | Add EpisodicTagger to LimbicCircuit | 80 |
| 10 | Integrate recording in reward delivery path | 10 |
| 11 | Integrate replay in glacial path (Mature stage) | 20 |
| 12 | Tests: episodes recorded above threshold, replay samples by priority | 20 |

### Phase 3: MNIST Validation

| Step | What |
|---|---|
| 13 | Run MNIST with limbic enabled, compare accuracy with and without |
| 14 | Key metric: RPE per class over time — does class 8's RPE drop as the system learns it? |
| 15 | Key metric: salience distribution — novel digits get higher salience than familiar ones? |

**Total: ~360 lines of Rust. 8–12 hours.**

---

## 9. Interaction with Planned Features

| Feature | Interaction |
|---|---|
| **Metabolic pressure** | RPE replaces raw reward in energy calculation. Directly addresses the "hubs earn energy from easy classifications" problem. |
| **Local inhibition** | Salience-modulated arousal affects the iSTDP target firing rate — high salience → lower target → more winners → broader exploration. |
| **Astrocytic gate** | Salience could modulate the astrocytic gate threshold — high salience → lower threshold → gates open → more plasticity for important stimuli. |
| **Coupled Oscillatory Dynamics** | Salience could modulate coupling strength — high salience → stronger coupling → tighter synchronization → more precise temporal binding for important stimuli. |
| **DeMorphon** | DeMorphon's collective utility is RPE-weighted, not raw-reward-weighted. DeMorphons that crack difficult patterns are metabolically favored. |
| **MorphonGenome** | Salience sensitivity could be a heritable genome field — some lineages evolve high sensitivity to novelty (explorative), others low (stable). |
| **ANCS-Core** | Episodic Tagger is the minimal version of what ANCS-Core's memory backend provides. Migration path: replace EpisodicTagger's HashMap with ANCS TripleMemory. |
| **Multi-instance** | Teacher's RPE stream is the "motivational" social signal — students can learn what the teacher finds surprising, bootstrapping their own salience model. |

---

## 10. For the Paper

Current paper: future work. "MORPHON's current global neuromodulation broadcasts reward equally to all active morphons. A limbic evaluation layer — providing per-stimulus salience detection (amygdala analog), reward prediction error computation (nucleus accumbens analog), and episodic memory formation (hippocampal analog) — would enable differential resource allocation to novel or difficult stimuli, addressing the observed limitation of uniform metabolic rewards across varying input difficulty."

Future paper: if RPE-scaled energy measurably improves MNIST accuracy (especially on difficult digit classes 3, 4, 8), the Limbic Circuit becomes a section in the Endoquilibrium paper with comparison tables (raw reward vs. RPE-scaled energy, per-class accuracy analysis, 10 seeds, Welch's t-test).

---

## 11. References

Duggins, P. et al. (2024). A scalable spiking amygdala model that explains fear conditioning, extinction, renewal and generalization. European Journal of Neuroscience, 59(8), 1911–1934.

LeDoux, J. (1996). The Emotional Brain. Simon & Schuster.

Schultz, W. (1998). Predictive Reward Signal of Dopamine Neurons. Journal of Neurophysiology, 80(1), 1–27.

Schultz, W., Dayan, P., & Montague, P. R. (1997). A Neural Substrate of Prediction and Reward. Science, 275(5306), 1593–1599.

Buzsáki, G. (2006). Rhythms of the Brain. Oxford University Press.

Menon, V. (2015). Salience Network. In Brain Mapping: An Encyclopedic Reference, pp. 597–611. Academic Press.

Fries, P. (2005). A mechanism for cognitive dynamics: neuronal communication through neuronal coherence. Trends in Cognitive Sciences, 9(10), 474–480.

Seeley, W. W. (2019). The Salience Network: A Neural System for Perceiving and Responding to Homeostatic Demands. Journal of Neuroscience, 39(50), 9878–9882.

---

*The Limbic Circuit — because intelligence isn't just knowing what you see. It's knowing what matters.*

*TasteHub GmbH, Wien, April 2026*
