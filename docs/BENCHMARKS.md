# Morphon Benchmarks

This document describes every benchmark in the `examples/` directory: what it tests, how to run it, what to expect, and how to interpret the results.

All benchmarks save JSON results to `docs/benchmark_results/v{version}/` for tracking progress across versions.

---

## Quick Reference

| Benchmark | Tests | Best Result (v3.0.0) | Runtime (quick) |
|-----------|-------|---------------------|-----------------|
| `cartpole`        | RL control                   | **avg=195.2 SOLVED** (v0.5.0) | ~10 min |
| `mnist_v2`        | Static classification        | 31% intact / **52.5% post-damage** | ~25 min |
| `nlp_readiness`   | Text-like pattern processing | Level 3/3 (analog readout) | ~3 min |
| `mnist`           | Two-phase STDP + post-hoc    | ~20% (legacy)            | ~25 min |
| `classify_tiny`   | 3-class smoke test           | (regression detector)    | ~15 sec |
| `classify_3class` | 3-class smoke test           | (regression detector)    | ~15 sec |
| `learn_compare`   | 2-class A/B learn rule       | 62% (v0.5.0)             | ~30 sec |
| `anomaly`         | Streaming anomaly detection  | (qualitative)            | ~1 min |
| `v2_validation`   | V2 primitives smoke          | (regression detector)    | ~30 sec |
| `v3_governance`   | V3 governance smoke          | (regression detector)    | ~30 sec |

CMA-ES optimizers (run separately, very long):
| `cma_cartpole`, `cma_endo`, `cma_optimize` | Hyperparameter search | (varies) | hours |

---

## Profile Convention

Most benchmarks support three profiles via CLI flags:

```bash
cargo run --example <name> --release              # quick (default, ~minutes)
cargo run --example <name> --release -- --standard # standard (~10x)
cargo run --example <name> --release -- --extended # extended (full sweep, ~hours)
```

Quick is for iteration. Standard is for reportable numbers. Extended is for paper-grade verification.

---

## CartPole

**File:** `examples/cartpole.rs`

**Task:** Balance a pole on a cart by applying left/right forces. Classic RL benchmark from OpenAI Gym (CartPole-v1).

**Why this benchmark:** It is the minimal test of the full learning pipeline. If three-factor learning, eligibility traces, tag-and-capture, contrastive reward, and Endoquilibrium regulation all work end-to-end, CartPole solves. If any link is broken, it stays at random baseline (~10 steps).

### Setup

- 4 observations (cart x, x_dot, pole θ, θ_dot) → population coding into **32 sensory morphons** (8 Gaussian receptive-field tiles per observation)
- **2 motor morphons** (left/right force)
- Cerebellar developmental preset, seed_size=60
- Linear TD-error critic external to MI provides the dopamine-analogue modulation signal (Frémaux et al. 2013)
- Episodes capped at 200 steps (gym standard) or 500 (extended profile)

### How to run

```bash
# Default (quick): 200 episodes
cargo run --example cartpole --release

# Standard: 1000 episodes
cargo run --example cartpole --release -- --standard

# A/B test: GlobalKWTA (default) vs LocalInhibition (iSTDP interneurons)
cargo run --example cartpole --release -- --local-inhibition
```

### Expected results

**SOLVED criterion:** average of last 100 episodes ≥ 195 steps.

| Profile | Episodes | Best documented result |
|---------|----------|----------------------|
| quick   | 200      | avg ~150–180 (intermittent) |
| standard | 1000    | **SOLVED at episode 895, avg=195.2** |
| extended | 3000    | SOLVED + extended stability |

### What to look for in the output

- `avg(100)`: rolling average of last 100 episodes — the SOLVED metric
- `best`: longest single episode
- `Endo` line: developmental stage transitions (Differentiating → Consolidating → Mature)
- Morphon counts: should stay near seed size (no division during RL)
- `tags=` and `con=`: tag-and-capture activity (consolidation events)

### Output JSON

`docs/benchmark_results/v{version}/cartpole_{timestamp}.json`:
```json
{
  "benchmark": "cartpole",
  "results": { "best_steps": 468, "avg_last_100": 195.2, "solved": true },
  "system": { "morphons": 95, "synapses": 425, ... },
  "diagnostics": { "weight_mean": ..., "active_tags": ..., ... }
}
```

---

## MNIST V2 (Recommended)

**File:** `examples/mnist_v2.rs`

**Task:** Classify 28×28 grayscale digit images into 10 classes. The standard benchmark for spiking neural networks.

**Why this benchmark:** Tests whether MI can grow a discriminative topology from scratch on a non-trivial classification task. Also includes the **damage-and-recovery** protocol that reveals self-healing.

### Setup

- 784 input pixels (28×28), rate-coded
- 10 output classes via analog readout (Purkinje-style bypass)
- Cortical developmental preset, seed_size=200
- 30% sensory→associative connectivity (hardcoded in `developmental.rs` Phase 4)
- Endoquilibrium with corrected stage detection (Mature gate raised to 2000 ticks)

### How to run

**Prerequisites:** Download MNIST files to `./data/` (unzipped).

```bash
# Quick: 3 epochs × 3000 samples (~25 min)
cargo run --example mnist_v2 --release

# Fast (debugging): 2 epochs × 500 samples (~5 min)
cargo run --example mnist_v2 --release -- --fast

# Standard: 3 epochs × 5000 samples (~50 min)
cargo run --example mnist_v2 --release -- --standard

# A/B: GlobalKWTA vs LocalInhibition (iSTDP)
cargo run --example mnist_v2 --release -- --local-inhibition

# With seed override (default 42)
cargo run --example mnist_v2 --release -- --seed=123
```

### Expected results

| Phase | Accuracy | Notes |
|-------|----------|-------|
| Baseline (intact) | **30%** | Endo with Mature gate=2000 |
| Post-damage (30% killed) | ~28% | Random ablation of associative morphons |
| **Post-recovery** | **48%** | After 500 recovery samples + lifecycle re-enabled |

**The post-recovery number is the headline result.** A trained MI network that loses 30% of its hidden layer and is allowed to regrow ends up **better** than the original (+18pp). This is the self-healing finding documented in the paper. Earlier runs on v2.4.0 reached 52.5%; the current v3.0.0 result is 48% — both well above the intact baseline.

### What to look for

- Per-checkpoint progress lines:
  ```
  [V2  ] ep1 1000/3000 acc=18.0% lr=0.0200 m=1234 s=78000 endo: stage=Differentiating pm=1.80
  ```
  - `acc=`: test accuracy on 100 held-out samples
  - `m=`/`s=`: morphon and synapse counts (should slowly decrease — pruning)
  - `pm=`: plasticity multiplier (1.80 healthy, 0.60 = throttled)
  - `endo: stage=`: Endoquilibrium developmental stage

- Stage transition logs (`[ENDO] ... → ...`): shows Endo's reaction to reward dynamics
- Final summary box with Baseline / V2 / Post-damage / Post-recovery

### Output JSON

`docs/benchmark_results/v{version}/mnist_v2_{timestamp}.json`:
```json
{
  "baseline_acc": 31.0,
  "v2_acc": 29.5,
  "damaged_acc": 30.0,
  "recovery_acc": 52.5,
  "v2_morphons": 1253,
  "v2_synapses": 78314
}
```

### Known issues

- The 31% intact ceiling is **regulatory, not representational**. The substrate can reach 52% (proven by recovery). The intact training trajectory is held back by Endoquilibrium's plasticity throttling. Future fix: Limbic Circuit's RPE-based regulation.
- Performance is the main bottleneck for iteration. ~25 min per quick run on a 2024 MacBook Pro. Sparse eligibility updates would reduce this 5×.

---

## NLP Readiness

**File:** `examples/nlp_readiness.rs`

**Task:** A 4-tier synthetic benchmark measuring how close MI is to handling language. **Not** an NLP task — tests *prerequisites* for language.

**Why this benchmark:** Language requires (1) handling text-like input encoding, (2) positional sensitivity, (3) temporal memory, (4) compositional semantics. This benchmark tests each capability separately so you can see which is missing.

### Tiers

| Tier | What it tests | Input | Task | Pass threshold |
|------|--------------|-------|------|----------------|
| 0: **Bag-of-Chars**  | Text-like encoding handling | 27-dim freq | Vowel-heavy vs consonant-heavy | 65% |
| 1: **One-Hot Scale** | Dimensional scaling          | 135-dim flat | Same task at scale | 60% |
| 2: **Memory**        | Temporal context             | 27-dim × 3 sequential | Classify by FIRST char | 55% |
| 3: **Composition**   | XOR-like compositional       | 54-dim       | Token-pair XOR | 60% |

### How to run

```bash
cargo run --example nlp_readiness --release              # quick (~3 min)
cargo run --example nlp_readiness --release -- --standard
cargo run --example nlp_readiness --release -- --extended
```

All data is **synthetic** — no external downloads needed. All tiers use the analog readout pathway.

### Expected results (v3.0.0)

| Tier | Accuracy | Status |
|------|---------|--------|
| 0: Bag-of-Chars  | **99%** | PASS |
| 1: One-Hot Scale | **62%** | PASS |
| 2: Memory        | **85%** | PASS |
| 3: Composition   | 42%     | FAIL (XOR needs nonlinear hidden features) |

**NLP Readiness Level: 3/3** with analog readout.

### Interpretation

The pattern across tiers is the central finding:
- **The MI substrate creates discriminative representations.** Tiers 0-2 prove this — analog readout extracts 69-99% accuracy from morphon potentials.
- **The spike pipeline destroys the information.** When the same network is read via spikes (`teach_supervised`), all tiers drop to chance (~50%). The information IS in the potentials — spike conversion + propagation + integration loses it.
- **Compositional reasoning is missing.** Tier 3 (XOR) fails because the analog readout is linear and XOR requires nonlinear hidden features. This needs a working spike-based credit-assignment path through the hidden layer.
- **Temporal memory exists at the potential level.** Tier 2 at 88% means residual potentials retain information across sequential `process()` calls. The substrate for the temporal sequence processing spec already works.

### Output JSON

`docs/benchmark_results/v{version}/nlp_{timestamp}.json`:
```json
{
  "benchmark": "nlp_readiness",
  "readiness_level": 3,
  "composition_capable": false,
  "tiers": {
    "tier0_bag_of_chars": { "accuracy": 99.0, "passed": true, ... },
    "tier1_onehot_scale": { "accuracy": 69.0, "passed": true, ... },
    "tier2_memory":       { "accuracy": 88.0, "passed": true, ... },
    "tier3_composition":  { "accuracy": 40.0, "passed": false, ... }
  }
}
```

---

## MNIST (Legacy)

**File:** `examples/mnist.rs`

The original MNIST benchmark using two-phase learning: unsupervised STDP feature formation followed by post-hoc Diehl & Cook neuron labeling. Superseded by `mnist_v2.rs` which uses the analog readout and produces better results.

```bash
cargo run --example mnist --release
```

Kept for historical comparison and as a regression detector for the unsupervised STDP path.

---

## Smoke Tests (Regression Detectors)

These exist to catch regressions in the basic learning pipeline. They run in seconds and should always work; if they break, something fundamental is broken.

### `classify_tiny`
3-class classification, 8 inputs, 30 seed morphons, no morphogenesis. Uses pure delta rule. **If this is at chance level, the supervised pipeline is broken.**

```bash
cargo run --example classify_tiny --release
```

### `classify_3class`
Same as above but with a different topology setup. Catches regressions in different code paths.

### `learn_compare`
2-class A/B comparison testing different learning rule combinations. Best result: 62% (v0.5.0).

### `v2_validation`, `v3_governance`
Smoke tests for V2 primitives (field, frustration, target morphology) and V3 governance constraints. Should produce stable output across versions.

### `anomaly`
Streaming anomaly detection on a synthetic non-stationary time series. Qualitative — no single accuracy number, but topology change rate should correlate with regime shifts.

---

## CMA-ES Optimizers

**Files:** `examples/cma_cartpole.rs`, `examples/cma_endo.rs`, `examples/cma_optimize.rs`

These are NOT benchmarks — they are hyperparameter optimizers. They run CMA-ES over the configuration space of MI to find good parameters for a given task. Each generation runs the underlying benchmark (e.g. CartPole) many times. **Multi-hour runtime.**

```bash
cargo run --example cma_cartpole --release        # search CartPole hyperparams
cargo run --example cma_endo --release            # search Endoquilibrium hyperparams
cargo run --example cma_optimize --release        # generic search
```

Use only when you want to retune after a major architectural change.

---

## How to add a new benchmark

1. Create `examples/your_benchmark.rs`
2. Use the profile pattern (`--quick` / `--standard` / `--extended`)
3. Save results to `docs/benchmark_results/v{env!("CARGO_PKG_VERSION")}/your_benchmark_{timestamp}.json`
4. Include enough config in the JSON to be reproducible: seed, profile, all key hyperparameters
5. Add a row to the Quick Reference table at the top of this document
6. Add a section explaining what it tests and how to interpret the results

---

## Reproducing the paper results

The numbers in `docs/paper/paper/main.tex` come from:

```bash
# CartPole SOLVED (avg=195.2): historical, v0.5.0 standard profile
# This may not reproduce on v3.0.0 due to subsequent regressions and fixes

# MNIST 31% intact / 52.5% post-recovery: v3.0.0 quick profile
cargo run --example mnist_v2 --release

# NLP readiness Level 3/3: v3.0.0 quick profile
cargo run --example nlp_readiness --release
```

All result JSONs are saved to `docs/benchmark_results/v{version}/` so you can verify the exact numbers used in the paper.
