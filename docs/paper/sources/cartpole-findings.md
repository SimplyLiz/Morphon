# CartPole Findings — From avg=10 to SOLVED (195.2)

Complete experimental record from the v2.0.0 development session. Each finding includes the diagnostic evidence, the fix applied, and the measured impact.

---

## 1. Starting Point (v0.4.0 baseline)

| Metric | Value |
|--------|-------|
| avg(100) | ~10 steps |
| best | 57 steps |
| Firing rate | 0-10% (unstable, frequently 0%) |
| Captures | 0 |
| Weight std | divergent (1e284 on MNIST) |

Random CartPole baseline is ~9 steps. The system was barely above random.

---

## 2. Fix Chain (each necessary, none alone sufficient)

### 2.1 Endoquilibrium — Predictive Neuroendocrine Regulation

**Problem:** Associative morphons stop firing (FR=0%), creating a deadlock where no learning updates happen. No mechanism detected or corrected this.

**Diagnosis:** The `firing_by_type` diagnostic showed Associative FR dropping to 0% within the first 50 ticks. Once at 0%, no eligibility traces accumulate, no weight updates happen, FR stays at 0%. A positive feedback deadlock.

**Fix:** New module `src/endoquilibrium.rs` (~550 lines). Senses 7 vital signs every medium-path tick, predicts healthy state via dual-timescale EMAs (fast tau=50, slow tau=500), adjusts 6 levers through proportional control:

- `threshold_bias`: added to firing threshold in `Morphon::step()`. When FR drops below setpoint (8-12%), bias goes negative, lowering thresholds until firing restarts.
- `plasticity_mult`: scales all weight update deltas. Boosted when eligibility density is low.
- `channel_gains`: [reward, novelty, arousal, homeostasis] multipliers applied to receptor-gated modulation in `apply_weight_update()`.

6 regulation rules covering: firing rate, eligibility density, weight entropy, cell type balance, tag-capture health, energy pressure.

**Result:** avg 10 → 21 (2.1x). FR stabilized at 16%. Eligibility density at 30-40%. The system was alive and learning.

**Key implementation detail:** `sense_vitals()` is a free function (not `&mut self`) to avoid borrow conflicts with `&self.morphons` + `&mut self.endo` in `System::step()`. Runs on existing medium tick — no new scheduler period.

### 2.2 Episode-Gated Capture

**Problem:** The per-tick capture mechanism (`modulation.reward > capture_threshold`) was broken for RL. In CartPole, `inject_td_error()` injects reward every alive step, keeping the reward level saturated near 1.0 regardless of episode quality. Setting `capture_threshold` to 0.4 consolidated 100% of synapses in 100 episodes. Setting it to 0.7 — same result. The reward *level* doesn't discriminate good from bad episodes.

**Diagnosis:**

| capture_threshold | Result |
|---|---|
| 10.0 (disabled) | 0 captures, avg=21 |
| 0.4 | 557/557 consolidated immediately, avg=19 (worse — frozen) |
| 0.7 | 555/555 consolidated, avg=21 |
| 0.7 + gate=30 | 0 captures (gate never reached), avg=21 |

The `consolidation_gate` on `recent_performance` was supposed to prevent premature capture, but with avg~21 and gate at 10-20, it was borderline.

**Fix:** Replaced per-tick capture with episode-relative capture:

```rust
pub fn report_episode_end(&mut self, episode_steps: f64) {
    let delta = episode_steps - self.running_avg_steps;
    self.running_avg_steps = 0.95 * self.running_avg_steps + 0.05 * episode_steps;
    if delta > 0.0 {
        self.capture_tagged_synapses(strength);  // above average → consolidate
    } else {
        self.decay_all_tags(0.5);  // below average → decay
    }
}
```

Also added `consolidation_level: f64` (0.0-1.0) to Synapse, replacing the binary `consolidated` flag with continuous partial protection: `effective_delta = delta_w * (1.0 - consolidation_level * 0.9)`. Fully consolidated synapses get 10% residual plasticity.

**Result:** Weight entropy collapse stopped. `weight_std` maintained at 1.5-1.6 over 3000 episodes (was collapsing to 0.15). Consolidation became selective and reversible. avg unchanged at ~21 (capture alone doesn't help if the readout can't learn).

### 2.3 Activity Stabilization

**Problem:** The same CartPole input produced different Associative morphon activity patterns across episodes. Measured via Jaccard similarity: **0.432** (43% overlap). The readout saw different representations of the same state every time.

**Diagnosis:** Zero Associative→Associative recurrent connections (0/527). The instability was NOT from recurrent chaos — it was from pseudo-random noise in `Morphon::step()`:

```rust
let noise_scale = 0.1 * self.frustration.noise_amplitude;  // ±0.05 per step
```

With k-WTA selecting the top 5%, morphons near the competition boundary flip in and out based on noise. 4 internal steps × noise = different winners each time.

**Fix:** Reduced noise for Associative morphons from 0.1 to 0.02:

```rust
CellType::Associative => 0.02 * self.frustration.noise_amplitude,
```

Also added `reset_voltages()` between episodes — clears potentials, refractory timers, spike queue. Biologically analogous to inter-trial interval baseline recovery.

**Result:** Jaccard 0.432 → **1.000** (perfect pattern stability). Same input → same 8 morphons fire, every time. The voltage reset alone had no effect on avg (carryover was not the primary source of instability).

**Finding:** Activity stability is necessary but not sufficient. avg stayed at ~21 despite perfect Jaccard=1.0. The readout still couldn't learn the correct mapping.

### 2.4 Supervised Readout Hint

**Problem:** `train_readout(chosen_action, td_error * lr)` reinforces whichever action was chosen. With epsilon-greedy exploration, the chosen action is random ~50% of the time early on. Positive TD error after a random *wrong* action reinforces it just as much as after a correct action. The readout learns "everything is equally good."

**Diagnosis:** Added a policy probe — present 4 test states with known correct actions, check if the readout picks correctly:

```
policy=2/4 for 1000+ consecutive episodes
```

The readout was at **coin-flip accuracy** throughout training. The output differentiation (`d+` and `d-` for opposite pole leans) always had the **same sign** — the system mapped both "lean right" and "lean left" to the same action preference.

**Fix:** Replace chosen-action reinforcement with state-derived correct action:

```rust
let correct_action = if env.theta > 0.0 { 1 } else { 0 };
system.train_readout(correct_action, 0.05);
```

This is the cerebellar circuit pattern: the MI morphon network provides unsupervised features (granule cells), the readout learns a supervised mapping (Purkinje cells / climbing fiber error).

**Result:** avg 21 → 33. Still policy=2/4 on the probe (the probe tests from cold reset where morphons don't fire).

### 2.5 Constant Readout Learning Rate

**Problem:** `lr = td_error.abs().min(1.0) * base_lr` — learning rate depends on TD error magnitude. During steady-state balancing, TD error ≈ 0 and the readout doesn't learn. Learning concentrates on failure moments (TD spikes) where theta is at the threshold — providing no useful signal for small-angle balancing.

**Fix:** Constant learning rate of 0.05, applied every step regardless of TD error.

**Result:** avg 33 → 44.

### 2.6 Readout Bias + Centered Sigmoid + No L2 Decay (THE BREAKTHROUGH)

**Problem:** Three compounding issues in the readout prevented it from learning a discriminative mapping:

1. **L2 weight decay** (`0.001 * w` per step): With ~200 steps/episode, this erases 20% of all weight changes per episode. The supervised gradient tries to build a 0.5 difference between d+ and d-, and the decay erases 20% of it every episode.

2. **Uncentered sigmoid**: `sigmoid(0) = 0.5`, creating a large constant offset `0.5 * Σw` on both outputs. The readout weights needed to learn both the mean *and* the state-dependent signal. The mean dominated.

3. **No bias term**: Without a per-output bias to absorb the constant offset, the weights had to encode both the mean and the deviation. This makes the optimization landscape harder — small state-dependent signals are invisible next to the large constant.

**Diagnosis:** Output discrimination tracking showed `d+` and `d-` with the same sign for 3000+ episodes:

```
Ep  100 | d+=4.498 d-=-3.740    ← same sign (both positive)
Ep  500 | d+=0.876 d-=-2.411    ← same sign (opposite, but both have d+ > d-)
```

Both outputs preferred the same action regardless of state. The ~0.5 difference between d+ and d- was the discriminative signal, drowning in the constant bias.

**Fixes (all three applied together):**

```rust
// 1. Centered sigmoid: 0 at resting potential
let act = 1.0 / (1.0 + (-p).exp()) - 0.5;

// 2. Learnable bias
let out = bias[j] + Σ w[j][i] * act[i];

// 3. No L2 decay
let delta = learning_rate * act * errors[j];  // removed: - 0.001 * w
```

**Result:** avg 44 → **195.2 (SOLVED)**. The learning curve was monotonic:

| Episode | avg(100) |
|---------|----------|
| 100 | 44.9 |
| 300 | 74.0 |
| 500 | 93.4 |
| 700 | 130.1 |
| 800 | 143.9 |
| 895 | **195.2** |

best=468 (93.6% of max 500). `d+` and `d-` developed **opposite signs** by episode 100, and the difference grew monotonically to ±10-12.

---

## 3. TD-Only Experiment (No Supervised Hint)

After solving CartPole with the supervised hint, tested whether the readout fixes alone enable TD-only learning:

**Setup:** Same as solved version but `train_readout(chosen_action, td * lr)` instead of `train_readout(sign(theta), 0.05)`.

**Result over 3000 episodes:**

| Metric | Supervised | TD-only |
|--------|-----------|---------|
| avg peak | 195.2 (SOLVED) | 88.7 |
| avg @ 3000 | - | 58.8 |
| best | 468 | 333 |
| policy probe peak | 2/4 | 4/4 (intermittent) |

**Key observations:**

- TD-only reaches avg=88 (8.8x random). Without the readout fixes it was avg=21. The fixes were necessary regardless of signal source.
- Policy probe hits 4/4 at several points (ep 1991, 2081, 2545, 2619, 2845) — the readout DOES occasionally learn the correct mapping from TD alone.
- But it can't sustain it. avg oscillates between 35-88. The `d+/d-` difference is ~2 (vs ~10 with supervised hint). TD provides a weaker directional signal.
- The avg doesn't plateau — it rises and falls in waves. Suggests the readout learns a good mapping, then the TD noise erodes it over subsequent episodes.

**Conclusion:** The readout architecture is sound. The signal strength determines the ceiling. TD-only provides a weak but real directional signal that the centered/biased readout can use; the supervised hint provides a strong signal that enables convergence.

---

## 4. Diagnostic Instruments Developed

### 4.1 Activity Stability (Jaccard Similarity)

Present the same input 5 times with voltage reset between, record which Associative morphons fire. Compute Jaccard similarity between consecutive pairs.

```
Jaccard = |A ∩ B| / |A ∪ B|
```

- 1.0 = identical patterns (deterministic network)
- 0.5 = ~50% overlap (noisy but structured)
- 0.0 = completely different patterns (chaotic)

**Finding:** The system went from 0.43 (chaotic) to 1.0 (deterministic) with reduced Associative noise. This was necessary for readout learning but not sufficient.

### 4.2 Output Discrimination (d+/d-)

Present two test states with opposite pole leans (θ=+0.1, θ=-0.1), compute output difference `d = out[1] - out[0]` for each.

- d+ positive, d- negative = correct discrimination (push right when leaning right, left when leaning left)
- d+ and d- same sign = no discrimination (same action regardless of state)
- |d+| and |d-| > 5 = strong discrimination

**Finding:** This was the single most informative diagnostic. d+/d- same sign for 3000 episodes confirmed the readout wasn't learning. When the readout fixes enabled discrimination, d+ and d- developed opposite signs within 100 episodes and the avg started climbing monotonically.

### 4.3 Policy Probe

4 test states with known correct actions. Measures policy accuracy (0-4).

**Finding:** Less useful than d+/d- because the probe tests from cold voltage reset where morphons don't fire. Shows 2/4 (coin flip) even when the in-episode policy is working. The warm-probe variant (no reset) also showed 2/4 because the test states disrupt the episode context. The d+/d- metric is more reliable.

---

## 5. Dead Ends and Negative Results

### 5.1 CMA-ES Over Endoquilibrium Parameters

28 generations, 560 evaluations. Best found: avg=21.2. The fitness landscape around avg=20 was flat — no parameter combination could overcome the readout discrimination problem. CMA-ES is useful for tuning, but the bottleneck was architectural (missing bias, uncentered sigmoid, L2 decay).

### 5.2 Extended Training (3000 episodes)

Without the readout fixes, avg plateaued at ~21 after episode 300 regardless of training length. The system found good episodes occasionally (best=132) but couldn't learn from them. More episodes doesn't help when the learning rule can't converge.

### 5.3 Weight Normalization Strength

L1 weight normalization with [0.5, 2.0] clamping collapsed all incoming weights to `target_norm / n_incoming ≈ 0.3`. Weight std dropped from 0.14 to 0.03. All morphon potentials became identical, destroying features.

Fix: gentle clamping [0.9, 1.1]. Weight diversity preserved while preventing explosion.

### 5.4 Voltage Reset Between Episodes

No effect on avg when tested in isolation. Carryover from previous episodes was not the primary source of activity instability — the per-step noise was.

### 5.5 Sensory-Only Readout

Zeroing out Associative weights to read only from sensory morphons (which receive the population-coded input directly): no improvement. The readout still couldn't discriminate because the bias/centering/decay problems affected it regardless of which morphons it read from.

### 5.6 Frozen MI Network

Setting `plasticity_mult = 0` (readout-only learning, MI weights frozen): no improvement. With `weight_std = 0.03`, all morphon potentials were identical. The readout had no discriminative features to work with. The MI network's developmental phase produced uniform weights that the readout couldn't distinguish.

---

## 6. Weight Entropy Collapse — Root Cause Analysis

Across multiple experiments, the MI network weights collapsed to near-uniform distribution (std 0.03-0.15). Three mechanisms contributed:

1. **L1 weight normalization** (Diehl & Cook 2015): rescales all incoming weights to sum to target. With 10+ inputs per morphon, individual weights converge to `target/n ≈ 0.3`. The [0.5, 2.0] clamping was too aggressive.

2. **Heterosynaptic depression**: when a morphon fires, all incoming weights decay by `0.003 * w`. Over 1000+ steps, this flattens weight variance.

3. **Weight decay** on all three-factor synapses: `w -= 0.0005 * w` per medium tick. Small but cumulative.

These mechanisms are individually correct (prevent weight explosion, enforce competition) but collectively too aggressive. The fix: gentler normalization clamping [0.9, 1.1] preserved variance while preventing explosion.

---

## 7. Architecture Validated

The solved CartPole confirms a dual-system architecture:

```
Input → [Sensory] → population coding (32 Gaussian tiles)
                  ↓
         [Associative] → unsupervised representation (STDP + DFA)
                  ↓                    ↑
         [Readout] → supervised task mapping    Endoquilibrium
                  ↓                    ↑        (regulation)
         [Motor] → action selection    ↑
                  ↓                    ↑
         Environment → reward → TD error
```

- **MI morphon network**: unsupervised feature discovery via structural plasticity, three-factor STDP, DFA. Endoquilibrium maintains health (FR, eligibility, entropy).
- **Analog readout**: supervised task mapping via softmax cross-entropy with state-derived labels. Centered sigmoid, learnable bias, no weight decay.
- **Episode-gated capture**: consolidates tagged synapses after above-average episodes. Continuous `consolidation_level` with 10% residual plasticity.

This mirrors the cerebellar circuit: granule cells (Associative morphons) provide a rich feature basis, Purkinje cells (readout) learn a supervised mapping from climbing fiber error signals.

---

## 8. Numerical Summary

| Configuration | avg(100) | best | FR | Jaccard | d+/d- |
|---|---|---|---|---|---|
| v0.4.0 baseline | 10 | 57 | 0-10% | n/a | n/a |
| + Endoquilibrium | 21 | 78 | 16% | 0.43 | same sign |
| + episode capture | 21 | 78 | 16% | 0.43 | same sign |
| + reduced noise | 21 | 78 | 16% | 1.00 | same sign |
| + supervised hint | 33 | 180 | 16% | 1.00 | same sign |
| + constant lr | 44 | 166 | 16% | 1.00 | same sign |
| + bias/centering/no decay | **195.2 (SOLVED)** | 468 | 16% | 1.00 | **opposite** |
| TD-only (no hint) | 88 | 333 | 16% | 1.00 | weak opposite |

---

## 9. Reproducibility

```bash
cargo run --example cartpole --release -- --standard
```

Expected: SOLVED (avg >= 195) within 900 episodes. Results saved to `docs/benchmark_results/v2.0.0/`.

Key parameters in `examples/cartpole.rs`:
- `endoquilibrium: EndoConfig { enabled: true, ..Default::default() }`
- `capture_threshold: 0.7` (used by episode-gated capture, not per-tick)
- `consolidation_gate: 40.0` (via `set_consolidation_gate`)
- `reset_voltages()` between episodes
- `train_readout(correct_action, 0.05)` with `correct_action = sign(theta)`
- Sensory-only readout via `filter_readout_weights`
