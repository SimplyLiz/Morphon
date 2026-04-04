# Capture Mechanism — From Per-Tick to Episode-Gated

How synaptic consolidation works, why per-tick capture failed for RL, and the episode-relative replacement.

---

## 1. Tag-and-Capture: The Concept

Tag-and-capture implements biological synaptic tagging and capture (Frey & Morris 1997):

1. **Tagging**: When a synapse has high Hebbian coincidence (eligibility > threshold), it's "tagged" — marked as a candidate for consolidation. Tags decay slowly (tau_tag = 500 steps).

2. **Capture**: When a strong reward signal arrives while the tag is active, the synapse is "captured" — its weight change becomes permanent. Captured synapses are protected from pruning and get reduced plasticity.

This solves delayed credit assignment: the tag marks *which* synapses were active during a decision, and the later reward determines *whether* that decision was good. No backpropagation needed.

---

## 2. Why Per-Tick Capture Failed for RL

### The original mechanism:

```rust
if synapse.tag > 0.1
    && modulation.reward > capture_threshold  // reward LEVEL, not delta
    && !synapse.consolidated
{
    synapse.weight += capture_rate * tag_strength * reward;
    synapse.consolidated = true;
}
```

### The problem:

In CartPole, `inject_td_error()` injects reward every alive step:

```rust
if td_error > 0.0 {
    self.modulation.inject_reward(td_error.min(1.0));
}
```

The `Neuromodulation.reward` field accumulates (additive injection) and decays slowly (decay=0.95). During a typical 20-step episode with positive TD:

```
Step 1: reward = 0.0 + 0.8 = 0.8
Step 2: reward = 0.8 × 0.95 + 0.7 = 1.46 → clamped to 1.0
Step 3: reward = 1.0 × 0.95 + 0.6 = 1.55 → clamped to 1.0
...
```

The reward level stays at 1.0 for the entire episode. Any `capture_threshold` below 1.0 triggers consolidation on every step. The mechanism can't distinguish a great episode from a terrible one — the reward level is the same.

### Experimental confirmation:

| capture_threshold | Synapses consolidated | avg |
|---|---|---|
| 10.0 (disabled) | 0/557 | 21 |
| 0.7 | 555/555 (100%) | 21 |
| 0.4 | 557/557 (100%) | 19 (worse — frozen) |

The `consolidation_gate` (performance threshold) was supposed to prevent premature capture, but with avg~21 and gate at 10-20, the gate was satisfied and everything consolidated.

---

## 3. Episode-Gated Capture

### Design:

Capture decisions are deferred to episode end. The question changes from "Is reward high right now?" to "Was this episode better than average?"

```rust
pub fn report_episode_end(&mut self, episode_steps: f64) {
    let delta = episode_steps - self.running_avg_steps;
    self.running_avg_steps = 0.95 * self.running_avg_steps + 0.05 * episode_steps;

    if delta > 0.0 {
        let strength = (delta / self.running_avg_steps.max(1.0)).min(1.0);
        self.capture_tagged_synapses(strength);
    } else {
        self.decay_all_tags(0.5);
    }
}
```

### Properties:

- **Performance-relative**: A 50-step episode in a system averaging 20 triggers strong capture. A 50-step episode in a system averaging 100 triggers weak capture.
- **Selective**: Only tagged synapses are captured. Tags accumulate per-tick via STDP/DFA as before. Capture only happens at episode boundaries.
- **Anti-consolidation on failure**: Below-average episodes decay tags by 50%, preventing bad policies from accumulating enough tag strength to capture.
- **Proportional**: `strength` scales with how much better the episode was. A 132-step episode with avg=21 gives `strength = min(111/21, 1) = 1.0` (max capture). A 25-step episode with avg=21 gives `strength = 4/21 = 0.19` (weak capture).

### Continuous Consolidation Level:

Replaced boolean `consolidated` with continuous `consolidation_level: f64` (0.0 = fully plastic, 1.0 = fully consolidated):

```rust
// In capture_tagged_synapses:
let delta_level = strength * tag_strength.min(1.0) * 0.3;
syn.consolidation_level = (syn.consolidation_level + delta_level).min(1.0);

// In apply_weight_update:
let consolidation_scale = 1.0 - synapse.consolidation_level * 0.9;
let delta_w = synapse.eligibility * m * plasticity_rate * consolidation_scale;
```

At level=1.0, the synapse gets 10% of normal weight updates. Not fully frozen — can still slowly adapt if the environment changes. The binary `consolidated` flag is preserved for backward compatibility (set when level > 0.5, used for pruning protection).

---

## 4. Observed Behavior

With episode-gated capture:

| Episode range | Consolidation count | What happened |
|---|---|---|
| 0-100 | Rapid rise to ~480/500 | Early above-average episodes triggered mass capture |
| 100-700 | Fluctuating 90-490 | Deconsolidation melting weak synapses, recapture on good episodes |
| 700+ | Stable ~495/510 | Most synapses consolidated with high level |

The deconsolidation mechanism (`deconsolidate_weakest`) fires when `recent_performance < peak_performance * 0.8`, melting 10% of weakest consolidated synapses. This creates a dynamic equilibrium: good episodes consolidate, bad streaks deconsolidate, and the system oscillates between plasticity and stability.

### Weight entropy preservation:

| Mechanism | weight_std over 3000 ep |
|---|---|
| Per-tick capture (disabled) | Collapsed from 2.4 → 0.15 |
| Per-tick capture (enabled) | Frozen at initial distribution |
| Episode-gated capture | Maintained at 1.5-1.6 |

Episode-gated capture preserves weight diversity because consolidation is selective and gradual. Consolidated synapses maintain their diverse weights (protected from normalization and decay). Unconsolidated synapses remain plastic and are reshaped by learning rules.

---

## 5. Limitations

Episode-gated capture requires an explicit episode boundary (`report_episode_end`). This works for episodic RL (CartPole, Atari) but not for continuous tasks (real-time control, online learning) where there's no natural episode boundary.

For continuous tasks, a time-windowed variant could work: evaluate performance over rolling windows and trigger capture when the current window exceeds the running average. The `running_avg_steps` EMA already operates this way — it just needs a periodic trigger instead of an episode-end trigger.
