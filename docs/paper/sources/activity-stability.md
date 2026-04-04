# Activity Stability — From Chaos to Determinism

How the MI network's activity patterns were stabilized, what caused the instability, and why stability is necessary but not sufficient.

---

## 1. The Measurement: Jaccard Similarity

**Protocol:** Present the same CartPole input 5 times with `reset_voltages()` between each. Record which Associative morphons fire. Compute pairwise Jaccard similarity:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

| Jaccard | Interpretation |
|---------|---------------|
| 1.0 | Identical pattern every time (deterministic) |
| 0.7-0.9 | Mostly stable, some noise |
| 0.4-0.6 | Semi-random (k-WTA boundary effects) |
| 0.0-0.3 | Chaotic (network state dominates input) |

---

## 2. Source of Instability

### What we expected: Recurrent chaos
Associative→Associative connections amplifying small perturbations into divergent firing patterns (butterfly effect).

### What we found: Zero recurrence
```
Recurrent A→A: 0/527 (0%)
```

The developmental program (cerebellar template) creates feedforward connections: Sensory → Associative → Motor. No Associative→Associative recurrence exists. The instability has a different source.

### Actual cause: Per-step noise + sharp k-WTA competition

In `Morphon::step()`:
```rust
let noise_raw = (self.id.wrapping_mul(self.age) ...) // pseudo-random
let noise_scale = 0.1 * self.frustration.noise_amplitude;  // ±0.05
self.potential = self.potential * (1.0 - leak_rate) + input + noise;
```

With k-WTA selecting the top 5% of Associative morphons by `input_accumulator`:
- Morphons near the k-WTA boundary have potentials within ±0.05 of each other
- Noise of ±0.05 pushes them above or below the boundary randomly
- Different winners → different activity pattern → different readout input

With `INTERNAL_STEPS=4`, there are 4 rounds of integration per observation. Noise accumulates across steps: effective noise ≈ `0.05 × √4 = 0.1`, comparable to the input-driven potential differences.

---

## 3. The Fix: Reduced Associative Noise

```rust
// Before:
let noise_scale = if self.cell_type == CellType::Motor { 0.0 }
    else { 0.1 * self.frustration.noise_amplitude };

// After:
let noise_scale = match self.cell_type {
    CellType::Motor => 0.0,
    CellType::Associative => 0.02 * self.frustration.noise_amplitude,
    _ => 0.1 * self.frustration.noise_amplitude,
};
```

0.1 → 0.02 (5x reduction) for Associative morphons only. Sensory, Stem, and Modulatory morphons keep full noise for exploration.

**Result:** Jaccard 0.43 → **1.000**. Same input → same morphons fire, every time. The competition outcome becomes deterministic because the potential differences (~0.3-0.5 from input-driven activation) dominate the noise (~0.01).

**Note:** Frustration-driven noise scaling (`noise_amplitude` rises with stagnation) still works. A frustrated Associative morphon gets `0.02 × 5.0 = 0.1` noise — back to original levels. The reduction only affects the baseline; exploration still activates when needed.

---

## 4. Voltage Reset Between Episodes

Added `reset_voltages()` — clears potentials, refractory timers, input accumulators, spike queue. Called at the start of each episode.

**Hypothesis:** Leftover state from the previous episode (refractory morphons, accumulated potential) causes different transient responses to the same first observation.

**Result:** No effect on avg when tested in isolation. Carryover was not the primary instability source. The noise was.

**Kept anyway:** The voltage reset is cheap and makes the network's episode-initial response cleaner. Biologically analogous to inter-trial interval baseline recovery.

---

## 5. Stability Is Necessary But Not Sufficient

| Configuration | Jaccard | avg |
|---|---|---|
| Full noise (0.1) | 0.43 | 21 |
| Reduced noise (0.02) | 1.00 | 21 |
| + supervised readout fixes | 1.00 | 195 (SOLVED) |

Perfect pattern stability (Jaccard=1.0) did NOT improve avg by itself. The readout still couldn't learn because of the bias/centering/decay problems in `train_readout`.

**Why stability is necessary:** Without it, the readout sees different representations of the same state across episodes. The delta rule gradient averages to zero because the "correct" weight direction changes every time the representation changes. The readout can't converge on a fixed mapping because the mapping isn't fixed.

**Why stability isn't sufficient:** A deterministic representation that doesn't discriminate between states is useless. With uniform MI network weights (std=0.03), all morphons have nearly identical potentials regardless of input. The representation is stable but uninformative. The readout needs both stability AND discriminative features to learn.

In practice, the population-coded sensory morphons provide the discriminative features (different Gaussian tiles activate for different pole angles), and the stability fix ensures these features produce consistent readout inputs. The Associative morphons contribute stability (same ones fire each time) but not discrimination (their potentials are too similar).

---

## 6. Implications for Scale

The noise-driven instability scales with network size. With more Associative morphons, more are near the k-WTA boundary, and the sensitivity to noise increases. The fix (reduced noise) should scale — the 0.02 base is independent of network size.

However, for larger networks or tasks requiring more diverse representations, the noise reduction may be too aggressive. A better approach: adaptive noise scaling based on the k-WTA margin (difference between k-th and (k+1)-th morphon potentials). When the margin is large, noise doesn't matter. When it's small, reduce noise. This would maintain stability where it's needed and exploration where it's safe.
