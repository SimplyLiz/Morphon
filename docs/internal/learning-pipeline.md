# Learning Pipeline

How the MI engine learns — three parallel credit assignment systems, two benchmark strategies, and current status.

---

## The Three Learning Paths

All three paths run during the MEDIUM scheduler tick. They operate on different synapse populations with different credit assignment strategies.

```
Input → [Sensory] → [Associative] → [Motor] → Output
              ↑            ↑             ↑
         (path 1)     (path 2)      (path 3)
        three-factor   DFA climb    TD-LTP critic
           STDP        fiber rule
```

Plus the **analog readout** (path 4), which bypasses spike propagation entirely and is trained by external calls to `train_readout()`.

### Path 1: Three-Factor STDP (Actor)

The biological learning rule. Applies to all synapses on Motor and Associative morphons.

**Eligibility trace** (per-synapse):
- Trace-based STDP (Fremaux & Gerstner 2016) with weight-dependent bounds (Gilson & Fukai 2011)
- Pre/post spike traces decay with `tau_trace` (~10 steps), widening the STDP window
- LTP scales as `(w_max - w) / w_max` — easy to strengthen weak, hard to over-strengthen strong
- LTD scales as `(w + w_max) / (2 * w_max)` — protects weak synapses
- Eligibility integrates STDP events: `e += (-e/tau_e + stdp) * dt`, clamped to [-1, 1]

**Weight update** (receptor-gated):
- Motor morphons respond to Reward + Arousal channels
- Associative morphons respond to Reward + Novelty channels
- `dw = eligibility * M(t) * plasticity_rate`
- Plasticity rate is novelty-modulated: `0.01 + 0.09 * novelty`
- Reward uses **advantage** (reward - baseline EMA, clamped >= 0)
- L2 decay: `w -= 0.0005 * w`

**Tag-and-capture**: When eligibility exceeds threshold, a slow synaptic tag is set (tau ~200-6000 steps). If strong reward arrives while the tag is active, the synapse is permanently captured (consolidated). Consolidated synapses are protected from pruning.

### Path 2: DFA Climbing-Fiber Rule (Hidden Layer)

Direct Feedback Alignment (Lillicrap et al. 2016) for credit assignment to the associative layer without backprop.

**Fixed random weights** (never updated): Each associative morphon gets random projections from all motor morphons, initialized once during `System::new()`. These project output error backward.

**Error signal**: When TD error is available, it's scaled by each motor's sigmoid activation and projected to associative morphons via the fixed weights: `feedback_j = sum(dfa_weight[j,i] * error_i)`.

**Weight update**: Uses `pre_trace` (not eligibility, not binary fired):
```
dw = pre_trace * feedback_signal * 0.02 - 0.001 * w
```
`pre_trace` is the right gate because at ~5% firing rate, binary `pre_fired` is too sparse, while `eligibility` is STDP-gated and attenuates signal. `pre_trace` persists ~10 steps — enough to carry the credit signal.

**Tagging**: When `|pre_trace * feedback_signal| > 0.1`, the synapse is tagged. Captures require `reward > 0.3` and `recent_performance > 30.0` (performance gate).

### Path 3: TD-LTP (Critic Morphons)

A subset of Associative morphons (~15%) are designated as critic morphons during system initialization. They predict state value V(s).

**TD error computation**:
```
V(s) = mean(potential[critic_id] for all critic_ports)
delta = reward + gamma * V(s') - V(s)
```

**Weight update**: Direct TD-LTP on incoming synapses:
```
dw = 0.01 * td_error * pre_trace
```

Critic morphons have restricted receptors: {Reward, Homeostasis} only — they don't respond to Novelty or Arousal, keeping the value function stable.

### Path 4: Analog Readout (Purkinje-style Bypass)

A separate output pathway that reads associative potentials directly (no spike propagation). Enabled by `enable_analog_readout()`, trained by `train_readout(correct_index, lr)`.

**Forward pass**:
```
output_j = sum(readout_weights[j][i] * sigmoid(assoc_i.potential))
```

**Delta rule training**:
```
target_j = 1.0 if j == correct_index, else 0.0
error_j = target_j - sigmoid(output_j)
dw_ji = lr * sigmoid(assoc_i.potential) * error_j - 0.001 * w
```

Readout weights initialized with Xavier scaling: `1/sqrt(n_assoc)`.

After readout training, output errors are backprojected through the same fixed DFA weights to inject `feedback_signal` into the associative layer — closing the loop between the readout and the hidden layer's STDP learning.

**Tag-and-capture on input synapses**: During readout training, sensory->associative synapses are tagged based on the backprojected feedback signal. This consolidates input pathways that contribute to correct classification.

---

## k-WTA + Weight Normalization

These mechanisms create hidden layer specialization (Diehl & Cook 2015), running every fast step:

**k-Winner-Take-All**: Only the top ~5% of associative morphons (by potential) survive firing each step. Non-winners get their firing suppressed and potential clamped to `threshold * 0.5`. Winners get a small threshold boost (+0.02) to prevent any single neuron from dominating.

**L1 Weight Normalization**: On every medium tick, incoming positive weights for each associative morphon are normalized to a target of `n_incoming * 0.3`. Strengthening one input forces weakening of others — synaptic competition that, combined with k-WTA, produces specialized feature detectors.

---

## Performance Gating

`system.report_performance(score)` maintains an EMA (alpha=0.05) of recent performance. This gates consolidation:

- **Below 30.0**: Tags accumulate but captures are blocked — the system hasn't proven competence yet.
- **Above 30.0**: Captures proceed, permanently locking in useful representations.

This prevents catastrophic early consolidation where random early patterns get permanently frozen.

---

## CartPole Strategy

The CartPole example uses both internal and external TD critics with population-coded observations.

### Architecture
- 32 sensory inputs (4 state dimensions x 8 Gaussian tiles, population coding)
- 2 motor outputs (left/right action)
- 4 internal steps per action decision
- Internal MI critic (~8-10 Associative morphons, td_lr=0.03)
- External linear critic with 8 features (raw + squared state, lr=0.02)

### Training Loop
1. Encode state as 32D population-coded vector (Gaussian tuning curves, width=0.3, amp=4.0)
2. `process_steps(obs, 4)` — 4 internal MI steps to settle
3. Epsilon-greedy action: epsilon decays from 0.5 to 0.05 over training
4. Reward shaping: `R = 1.0 + 0.5 * (1 - |theta|/threshold)` if alive, else `-1.0`
5. Internal TD critic: `system.inject_td_error(reward, 0.99)` — sets `last_td_error` (enables DFA) + injects reward
6. External TD critic drives readout training:
   - **Readout training**: `train_readout(action, min(|delta|, 1.0) * 0.15)` (positive delta trains correct action; negative delta trains the other action at 0.5x rate)
   - **Arousal on failure**: `inject_arousal(0.8)` on episode end
   - **Novelty on danger**: `inject_novelty()` when pole angle > 50% of threshold

### Key Fixes (2026-04-01)
- **DFA activated**: `inject_td_error()` sets `last_td_error` — was dead before (zero for entire run)
- **Reward signal fixed**: Only inject on positive TD, let decay create directional signal. Old scaling `(td * 0.3 + 0.5)` saturated channel at 1.0
- **Population coding**: 32 Gaussian channels (was 8 binary pos/neg). Each state region produces distinct activation pattern
- **Consolidation disabled**: `capture_threshold=10.0` — prevents premature locking of random early weights

### Current Results (standard profile, 1000 episodes)
- Best: 106 steps (up from 59)
- Average (last 100): ~15-18 at peak, declining to ~12 by ep 1000
- Representational drift causes degradation — see `docs/research/cartpole-representational-drift.md`
- Not solved (threshold: avg >= 195)

### Open Research: Representational Drift
The hidden layer's representations shift under STDP + DFA, degrading the readout's learned mapping. Even with frozen three-factor learning (alpha=0), drift occurs through DFA weight updates and homeostatic threshold regulation. Primary hypothesis: **Anchor & Sail heterogeneous plasticity** (20% stable anchors + 80% plastic sails). See `docs/research/anchor-sail-design.md`.

---

## MNIST Strategy (Two-Phase, Diehl & Cook 2015)

### Phase 1: Unsupervised Feature Learning

No labels used. The hidden layer self-organizes through STDP + k-WTA:

1. Present each image: `process_steps(image, 5)`
2. Inject novelty: `inject_novelty(0.3)` — primary driver of plasticity (CMA-ES finding)
3. One extra step: `feed_input(image); step()` — lets STDP propagate
4. k-WTA + weight normalization force different neurons to specialize on different digit features

Hidden representations must stabilize before Phase 2.

### Phase 2: Supervised Readout

STDP is frozen (medium_period set to 999999). Only the analog readout learns:

1. `enable_analog_readout()` — initialize readout weights with Xavier scaling
2. For each image: classify via `process_steps(image, 5)`, then `train_readout(label, 0.1)`
3. Hidden weights frozen; only readout weights learn via delta rule

### Input Encoding
Zero-bias: `(pixel / 255) * 3.0` maps to [0, 3]. No sigmoid bias — full dynamic range.

### Current Results (standard profile)
- Test accuracy: ~9-11%
- Mode collapse still present — readout tends to predict one class
- Per-class: some digits reach 40-100% individually, but at the expense of others
- Active tags: high (~79k), but 0 captures (performance gate not reached)

---

## What's Working, What's Not

### Working
- Three-factor STDP converges on 2-class tasks (62% accuracy)
- Tag-and-capture consolidates hundreds of synapses in CartPole
- DFA climbing-fiber rule provides hidden layer credit assignment
- k-WTA creates sparse distributed representations
- Analog readout trains correctly in isolation (100% with external logistic regression)
- Motor drift prevention (full leak + zero noise) keeps motors responsive

### Current Gaps
- **CartPole**: best=106, avg peaks ~18 then declines. **Representational drift** — the hidden layer drifts while the readout tries to learn. DFA and reward signal are now working (fixed 2026-04-01). Next step: heterogeneous plasticity (Anchor & Sail).
- **MNIST**: mode collapse persists in readout. Phase 1 self-organization produces some feature detectors, but Phase 2 readout training collapses to one class.
- **Consolidation catch-22**: Consolidate early → locks in random weights. Never consolidate → nothing stable. Need readout-coupled consolidation (stabilize what the readout finds useful).

### Key Lessons Learned
1. **Global modulation can't do classification**: Three-factor STDP alone applies the same reward scalar to all synapses — it can't express "this output right, that one wrong". DFA + analog readout were needed.
2. **Novelty drives plasticity more than reward**: CMA-ES found alpha_novelty=3.0, alpha_reward=0.5 — novelty is 6x more important for hidden layer learning.
3. **Motor morphons need special treatment**: Full leak (memoryless), zero noise, and potential clamping are essential to prevent saturation drift.
4. **Spike propagation adds noise**: The MI propagation pipeline (multi-step, delays, noise) distorts the forward pass. The analog readout bypass was necessary to get clean classification gradients.
5. **Two-phase learning is biologically valid**: Cortex self-organizes first (sleep/development), then supervised pathways refine (cerebellum). Trying to do both simultaneously destabilizes both.
6. **Dead signal pathways are silent killers**: The DFA and reward channel bugs (2026-04-01) went undetected because the system still "ran" — it just couldn't learn. Always verify that learning signals are actually nonzero.
7. **Population coding is critical**: Binary pos/neg encoding (8 channels) produced indistinguishable hidden layer activations for similar states. Gaussian population coding (32 channels) gave the readout actual features to discriminate on. Best episode jumped 59→106.
8. **Representational drift is the real challenge**: Once signal pathways work, the bottleneck shifts from "can the system learn?" to "can it retain what it learned?" Heterogeneous plasticity (stable anchors + plastic explorers) is the biological solution.
9. **Consolidation must be earned, not given**: Early consolidation locks in random associations. Readout-coupled consolidation (stabilize features the readout uses) is more principled than performance-gated consolidation.

---

## Parameter Reference

### Learning Rates

| Path | Parameter | Value | Notes |
|------|-----------|-------|-------|
| Three-factor STDP | plasticity_rate | 0.01-0.10 | Novelty-modulated |
| DFA climbing-fiber | lr | 0.02 | Fixed |
| TD-LTP (critic) | lr | 0.03 | Fixed (was 0.01, bumped 2026-04-01) |
| Analog readout | lr | 0.1-0.2 | Passed by caller |
| L2 decay (three-factor) | lambda | 0.0005 | Per step |
| L2 decay (DFA) | lambda | 0.001 | 2x stronger |
| L1 norm target | target | 0.3/synapse | Weight normalization |

### Time Constants

| Trace | tau | Purpose |
|-------|-----|---------|
| pre_trace / post_trace | 10 | STDP window |
| eligibility | 15-20 | Fast credit assignment |
| tag | 200-6000 | Delayed reward consolidation |

### Gating Thresholds

| Gate | Threshold | Effect |
|------|-----------|--------|
| Tag formation | eligibility > 0.3 | Set slow synaptic tag |
| DFA tag | \|pre_trace * feedback\| > 0.1 | Tag on DFA path |
| Capture (three-factor) | reward > 0.5 | Consolidate tagged synapse |
| Capture (DFA) | reward > capture_threshold | Consolidate tagged synapse (was hardcoded 0.3, now configurable) |
| Performance gate | recent_perf > 10.0 | Allow captures (was 15.0, lowered 2026-04-01) |
| k-WTA | top 15% | Sparse activation |

### Metabolic Budget (V3)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| base_cost | 0.001 | Being alive |
| synapse_cost | 0.0001 | Per connection per step |
| utility_reward | 0.02 | Per unit PE reduction |
| basal_regen | 0.003 | Unconditional trickle |
| firing_cost | 0.002 | Per spike |
