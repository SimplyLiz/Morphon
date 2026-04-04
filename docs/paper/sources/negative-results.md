# Negative Results and Dead Ends

Experiments that didn't work, why they failed, and what they taught us.

---

## 1. CMA-ES Over Endoquilibrium + Learning Parameters

**Setup:** 10-dimensional search: 5 Endo regulation gains (fr_deficit_threshold_k, fr_deficit_arousal_k, fr_deficit_novelty_k, smoothing_alpha, fast_tau) + 5 learning params (readout_lr, alpha_reward, tau_eligibility, a_minus, capture_threshold). Population 20, 200 max generations. Fitness = avg steps over 200 episodes.

**Result:** 28 generations (560 evaluations), best avg=21.2. Sigma expanding (2.57) — CMA-ES never found a gradient. The fitness landscape was flat.

**Why it failed:** The bottleneck was architectural (readout bias/centering/decay), not parametric. No combination of regulation gains or learning rates can fix a linear readout that's fighting a constant offset. CMA-ES is a powerful optimizer for smooth landscapes, but the avg=20 plateau was a hard ceiling set by the readout architecture.

**Lesson:** When an optimization algorithm fails to improve beyond a baseline, the search space may not contain the solution. Check architectural assumptions before tuning parameters.

---

## 2. Extended Training (3000 episodes, no readout fixes)

**Setup:** Standard CartPole with Endoquilibrium enabled, 3000 episodes.

**Result:** avg plateaued at ~21 after episode 300. best=132. Weight std collapsed from 2.4 to 0.15 by episode 400.

**Key observation:** best=132 vs avg=21 means the system occasionally finds good weight configurations (6.5x its average) but can't sustain them. The learning rule works momentarily but drifts. More training time doesn't help because the readout can't learn — it's not a convergence speed issue.

**Lesson:** If avg doesn't improve after 300 episodes, running 3000 won't help. The best-to-avg ratio reveals whether the system CAN solve the task (high best) vs whether it CAN LEARN to solve it (high avg). A large gap means the representation is there but the learning rule isn't converging.

---

## 3. Capture Threshold Tuning

**Setup:** Tested capture_threshold values of 0.4, 0.7, and 10.0 (disabled).

**Result:** Both 0.4 and 0.7 caused 100% consolidation within 100 episodes. The system froze solid. avg=19 (worse than no consolidation).

**Why it failed:** The `modulation.reward` level stays saturated at ~1.0 during any alive episode because `inject_td_error` injects reward every step. The capture check compares against a signal that doesn't discriminate episode quality. Any reachable threshold triggers mass consolidation.

**Lesson:** Per-tick capture is fundamentally incompatible with continuous reward injection. The capture mechanism needs an episodic signal (episode-relative performance), not an instantaneous one (current reward level).

---

## 4. Consolidation Gate Tuning

**Setup:** Set `consolidation_gate` to 20 (expected avg) and 30 (3x random).

**Result:**
- Gate=20: avg ~21, so gate is borderline. Sometimes captures, sometimes doesn't. Unstable.
- Gate=30: avg never reaches 30, so 0 captures in 3000 episodes. The gate prevents all consolidation.

**Why it failed:** The gate is a binary threshold on `recent_performance`. If avg oscillates around the gate value, consolidation flickers on and off. Too low → premature consolidation. Too high → no consolidation ever. There's no Goldilocks value when avg is stuck.

**Lesson:** Episode-gated capture is better than a fixed gate because it's relative (above/below current average) rather than absolute (above/below a fixed number).

---

## 5. Voltage Reset in Isolation

**Setup:** Added `reset_voltages()` between episodes, no other changes.

**Result:** No effect on avg. Same ~21.

**Why it failed:** The primary source of activity instability was per-step noise (noise_scale=0.1), not episode-to-episode carryover. After 4 internal steps of noise accumulation, the initial conditions are forgotten. The voltage reset helps with the first step but noise dominates by step 2-4.

**Lesson:** Always identify the dominant noise source before fixing a specific one. The voltage reset was a reasonable hypothesis, but the diagnostic (Jaccard=0.43 WITH reset) immediately showed that carryover wasn't the main issue.

---

## 6. Frozen MI Network (Readout-Only Learning)

**Setup:** Set `endo.channels.plasticity_mult = 0.0` and `endo.config.enabled = false`. Only the readout learns; MI network weights are frozen at developmental values.

**Result:** avg ~31 (similar to unfrozen). policy=2/4. Weight std=0.03 (near-uniform).

**Why it failed:** The MI network's developmental phase produced nearly uniform weights (mean=0.27, std=0.03). All morphon potentials are essentially identical regardless of input. The readout sees ~80 copies of the same value — there's no feature to discriminate on.

**Lesson:** The MI network's self-organization needs to produce diverse weights for the readout to have useful features. The L1 weight normalization (Diehl & Cook 2015) with aggressive clamping [0.5, 2.0] was actively destroying weight diversity. After gentling it to [0.9, 1.1], weight std improved to 0.14.

---

## 7. Sensory-Only Readout

**Setup:** Zeroed out Associative morphon weights in the readout via `filter_readout_weights()`. Readout reads only from the 32 sensory morphons that receive the population-coded input.

**Result:** avg ~31. policy=2/4. Same as full readout.

**Why it failed:** The readout bias/centering/decay problems affected it regardless of which morphons it read from. The sensory morphons have discriminative potentials (different Gaussian tiles activate for different theta), but the uncentered sigmoid + L2 decay + no bias prevented the readout from learning the mapping.

**Lesson:** The readout architecture problems were upstream of the feature source. Fixing the features (sensory-only) without fixing the readout (centering/bias/decay) doesn't help. After fixing the readout, both sensory-only and full readouts work.

---

## 8. TD-Scaled Readout Learning Rate

**Setup:** `lr = td_error.abs().min(1.0) * base_lr` — learning rate proportional to TD error magnitude.

**Result:** avg=33 (improved from 21 with supervised hint, but well below the 195 achieved with constant lr).

**Why it failed:** TD error is near zero during steady-state balancing (pole upright, small angle). The readout doesn't learn during the most informative period (small-angle control). Learning concentrates on failure moments (TD spikes) where theta is at the threshold — a regime where the correct action is always "push toward center" regardless of earlier dynamics.

**Lesson:** For supervised readout training, use a constant learning rate. The supervised signal (`correct_action = sign(theta)`) is informative at every step, not just during TD spikes. Decoupling the readout lr from TD error lets the readout learn from every observation.

---

## 9. Weight Normalization [0.5, 2.0] Clamping

**Setup:** L1 weight normalization with scale clamped to [0.5, 2.0] per medium tick.

**Result:** All incoming weights converge to `target_norm / n_incoming ≈ 0.3`. Weight std collapses to 0.03. No feature diversity.

**Why it failed:** The clamping allows up to 2x scaling per tick. With 10+ incoming connections, a morphon whose total weight is 1.5 (target 3.3) gets scaled by 2.0 every tick until convergence. Within ~10 ticks, all weights are at the target ratio. The normalization intended to prevent weight explosion instead enforced weight uniformity.

**Fix:** Gentle clamping [0.9, 1.1] — max ±10% adjustment per tick. Weights converge slowly enough that learning-driven differentiation outpaces normalization-driven uniformity. Weight std maintained at 0.14 instead of collapsing to 0.03.

**Lesson:** Normalization strength must be calibrated against learning rate. If normalization corrects faster than learning differentiates, all weights converge to the same value.
