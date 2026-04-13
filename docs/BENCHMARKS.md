# Morphon Benchmarks

This document describes every benchmark in the `examples/` directory: what it tests, how to run it, what to expect, and how to interpret the results.

All benchmarks save JSON results to `docs/benchmark_results/v{version}/` for tracking progress across versions.

---

## Quick Reference

| Benchmark | Tests | Best Result (v4.9.0) | Runtime (standard) |
|-----------|-------|---------------------|--------------------|
| `cartpole`   | RL control                        | **avg=195.2 SOLVED** | ~10 min |
| `mnist_v2`   | Supervised classification         | **87.0% stateless** (5-seed mean) | ~50 min |
| `milestones` | Developmental (Piagetian M0–M2)   | M1 PASS (96%↓, 12.5×); M2 None pre-training | ~10s |
| `drone`      | 3D quadrotor hover                | avg=91.5 steps, pos_err=1.17m | ~30 min |
| `anomaly`    | Streaming anomaly detection       | (qualitative) | ~1 min |

CMA-ES optimizers (separate, very long):
| `cma_cartpole`, `cma_endo`, `cma_optimize` | Hyperparameter search | (varies) | hours |

---

## Profile Convention

Most benchmarks support three profiles via CLI flags:

```bash
cargo run --example <name> --release              # quick (default, ~minutes)
cargo run --example <name> --release -- --standard # standard (~10x longer)
cargo run --example <name> --release -- --extended # extended (full sweep, ~hours)
```

Quick is for iteration. Standard is for reportable numbers. Extended is for paper-grade verification.

---

## CartPole

**File:** `examples/cartpole.rs`

**Task:** Balance a pole on a cart by applying left/right forces. Classic RL benchmark (CartPole-v1).

**Why this benchmark:** Minimal test of the full learning pipeline. If three-factor learning, eligibility traces, tag-and-capture, contrastive reward, and Endoquilibrium regulation all work end-to-end, CartPole solves. If any link is broken, the system stays at random baseline (~10 steps).

### Setup

- 4 observations (cart x, x_dot, pole θ, θ_dot) → population coding into **32 sensory morphons** (8 Gaussian receptive-field tiles per observation)
- **2 motor morphons** (left/right force)
- Cerebellar developmental preset, seed_size=60
- Linear TD-error critic external to MI provides the dopamine-analogue modulation signal (Frémaux et al. 2013)
- Episodes capped at 200 steps (gym standard) or 500 (extended)
- LocalInhibition (iSTDP) competition mode

### How to run

```bash
cargo run --example cartpole --release              # quick: 200 episodes
cargo run --example cartpole --release -- --standard # standard: 1000 episodes
```

### Expected results

**SOLVED criterion:** average of last 100 episodes ≥ 195 steps.

| Profile | Episodes | Expected |
|---------|----------|---------|
| quick   | 200      | avg ~150–180 (intermittent) |
| standard | 1000    | **SOLVED, avg=195.2** |
| extended | 3000    | SOLVED + extended stability |

### What to look for

- `avg(100)`: rolling average — the SOLVED metric
- `best`: longest single episode
- `Endo` line: stage transitions (Differentiating → Consolidating → Mature)

---

## MNIST V2

**File:** `examples/mnist_v2.rs`

**Task:** Classify 28×28 grayscale digit images into 10 classes.

**Why this benchmark:** Tests whether MI can grow a discriminative topology from scratch on a non-trivial classification task. Also includes the **damage-and-recovery** protocol.

### Setup

- 784 inputs (28×28 pixels), rate-coded
- 10 output classes via analog readout (Purkinje-style bypass)
- Cortical developmental preset, seed_size=200
- **Stateless training (V3-SL):** state reset before each training image — decouples representation learning from sequential context
- **Limbic Circuit:** stage-gated confidence salience + RPE-amplified contrastive reward
- Endoquilibrium with mature_min_updates=8000 (prevents premature Mature)

### How to run

**Prerequisites:** MNIST files in `./data/` (unzipped).

```bash
# Standard: 3 epochs × 5000 samples (~50 min)
cargo run --example mnist_v2 --release -- --standard

# With limbic circuit
cargo run --example mnist_v2 --release -- --standard --limbic

# Specific seed (default 42)
cargo run --example mnist_v2 --release -- --standard --seed=43

# Quick smoke test
cargo run --example mnist_v2 --release -- --fast
```

### Expected results (v4.9.0, standard profile)

| Variant | Mean accuracy (5 seeds) | Notes |
|---------|------------------------|-------|
| V3-SL baseline | **86.87%** stateless | Seeds 42–46 |
| V3-SL + limbic | **87.0%** stateless | 4/5 seeds ≥ baseline |
| Damage+recovery | ~39–44% online (post-recovery) | 30% assoc. morphons killed, 1-epoch regrowth |

**Stateless vs online:** Stateless resets system state before each test image (fair comparison to MLPs). Online is sequential — full temporal context, significantly harder.

**Damage recovery:** After 30% associative morphon ablation, Endo re-enters high-plasticity Differentiating stage (plasticity_mult≈2.16). One additional epoch recovers to within ~5pp of intact online accuracy. Recovery consistently exceeds damaged baseline across seeds (mean +14.8pp), confirming the forced developmental restart is doing real work.

### What to look for

Progress lines:
```
[V3SL] ep1 1000/5000 acc=72.3% lr=0.0200 m=1234 s=78000 endo: stage=Differentiating pm=1.80
```
- `acc=`: stateless test accuracy on 100 held-out samples
- `pm=`: plasticity multiplier (1.80 = learning active, 0.60 = throttled)
- `stage=`: Endoquilibrium developmental stage
- `m=`/`s=`: morphon/synapse counts (slow decrease from pruning expected)

### Output JSON

`docs/benchmark_results/v{version}/mnist_v2_{timestamp}.json`

---

## Piagetian Milestones

**File:** `examples/milestones.rs`

**Task:** Evaluate Morphon against the BSB (Behavioral Stages of Belief) developmental milestone ladder — a cognitive-science-grounded benchmark ladder that measures *what kind of understanding* the system has acquired, not just what task it can solve.

**Why this benchmark:** A trained MLP can reach 99% MNIST without any of the properties that distinguish genuine intelligence from curve fitting. Developmental milestones test for representation quality that task benchmarks miss: habituation, working memory, temporal prediction.

### Milestone ladder

| ID | Name | What it tests |
|----|------|--------------|
| M0 | Sensorimotor Response | Discriminably different outputs for different inputs |
| M1 | Habituation / Dishabituation | PE decreases with repetition, spikes on novelty |
| M2 | Object Permanence | Representation survives input removal (working memory) |
| M3 | Temporal Prediction | Anticipates sequence completions (eligibility traces) |
| M4 | Causal Attribution | Self-caused vs. external state transitions (lower PE on own actions) |
| M5–M7 | Self/Other, Imitation, Theory of Mind | Multi-agent (future work) |

M0–M2 are runnable today. M3 requires `examples/temporal_sequences.rs` (not yet built). M4 requires CartPole + perturbation extension.

### How to run

```bash
cargo run --example milestones --release              # standard: M0+M1+M2
cargo run --example milestones --release -- --quick   # quick: M0+M1 only
cargo run --example milestones --release -- --seed=43 # specific seed
```

### M0 — Sensorimotor Response

**Protocol:** 8 orthogonal input patterns, 10 steps each. Pairwise cosine distance on output vectors.

**Success criterion:** mean pairwise distance > 0.20 AND no pair < 0.05.

**Current status:** FAIL pre-training (expected — untrained readout collapses outputs). Meaningful only after supervised training epoch. Mean cosine dist is healthy (~0.5), but one pair collapses to ~0.001. M0 should auto-pass after MNIST training.

### M1 — Habituation / Dishabituation

**Protocol:** 200-step habituation on pattern X. Record PE at start, after habituation, and on novel pattern Y.

**Success criterion:** PE drop ≥ 30% AND dishabituation ratio ≥ 1.5×.

**Current result (v4.9.0, seed=42):** PASS — 96.6% PE drop, 12.49× dishabituation ratio. Both margins are large (far from threshold), confirming the prediction-error mechanism works as intended. The PE signal genuinely habituates to familiar inputs and spikes on novelty.

### M2 — Object Permanence

**Protocol:** Encode pattern for 30 steps, then feed zero input for up to 100 steps. Measure consecutive correct classification during occlusion. Probe recovery: half-strength pattern at end.

**Medal tiers (majority vote: ≥50% of classes must achieve):**
- Bronze: ≥10 consecutive occlusion steps
- Silver: ≥30 consecutive steps
- Gold: ≥100 consecutive steps + probe recovery

**Current result (v4.9.0, seed=42, pre-training):** None. Best single class: 2 consecutive steps.

**Interpretation:** Pre-training, the readout hasn't learned class-specific representations, so there's nothing to maintain. The metric also reveals that previous runs showing high `last_correct_step` values (up to 100) were misleading — `last_correct_step` is not the same as consecutive persistence. Class 7 at `last=100` but `consec=2` means it briefly guessed correctly at step 100, not that it held the representation throughout.

**Next measurement:** Run after MNIST epoch 1 (trained readout). The spec predicts Bronze is likely; Silver would be a novel result; Gold would be paper-worthy.

### Output JSON

`docs/benchmark_results/v{version}/milestones_{timestamp}.json`:
```json
{
  "m0": { "passed": false, "mean_cosine_distance": 0.51, "min_pairwise_distance": 0.001 },
  "m1": { "passed": true, "habituation_drop_pct": 96.6, "dishabituation_ratio": 12.49 },
  "m2": { "medal": "None", "best_single_class_persist_steps": 2, "per_class": [...] }
}
```

---

## Drone3D

**File:** `examples/drone.rs`

**Task:** Hover a 3D quadrotor at altitude waypoints, then navigate to spatial targets.

**Why this benchmark:** Tests MI on continuous 6-DOF control with a physics-based reward landscape. More difficult than CartPole: 12-dimensional state space, combined hover + navigation objectives.

### How to run

```bash
cargo run --example drone --release -- --standard
```

### Expected results (v4.9.0, standard profile, 1000 episodes)

- Hover: avg=91.5 steps
- Position error: 1.17m (3D)

---

## Anomaly Detection

**File:** `examples/anomaly.rs`

Streaming anomaly detection on a synthetic non-stationary time series. Qualitative — no single accuracy number. Topology change rate should correlate with regime shifts in the input distribution.

```bash
cargo run --example anomaly --release
```

---

## CMA-ES Optimizers

**Files:** `examples/cma_cartpole.rs`, `examples/cma_endo.rs`, `examples/cma_optimize.rs`

These are NOT benchmarks — they are hyperparameter optimizers. They run CMA-ES over the configuration space to find good parameters for a given task. **Multi-hour runtime.**

```bash
cargo run --example cma_cartpole --release   # search CartPole hyperparams
cargo run --example cma_endo --release       # search Endoquilibrium hyperparams
```

Use only after major architectural changes. Don't run routinely.

---

## How to add a new benchmark

1. Create `examples/your_benchmark.rs`
2. Use the profile pattern (`--quick` / `--standard` / `--extended`)
3. Save results to `docs/benchmark_results/v{env!("CARGO_PKG_VERSION")}/your_benchmark_{timestamp}.json`
4. Include seed, profile, and all key hyperparameters in the JSON for reproducibility
5. Add a row to the Quick Reference table at the top of this document
6. Add a section explaining what it tests and how to interpret the results

---

## Reproducing the paper results (v2, April 2026)

All result JSONs are in `docs/benchmark_results/`. The paper numbers come from:

```bash
# CartPole SOLVED (avg=195.2): v0.5.0, standard profile
# Reproduced at v4.1.0 (avg=195.5)
cargo run --example cartpole --release -- --standard

# MNIST 87.0% stateless (V3-SL + limbic, 5-seed mean): v4.9.0, standard
for seed in 42 43 44 45 46; do
  cargo run --example mnist_v2 --release -- --standard --limbic --seed=$seed
done

# Damage recovery (+14.8pp mean): same runs, damage phase results

# Drone3D hover avg=91.5: v4.9.0, standard profile
cargo run --example drone --release -- --standard
```

Paper baseline (pre-limbic, V3-SL only): 86.67% mean, 5 seeds. Current (with limbic): 87.0% mean.
