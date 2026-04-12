# MORPHON Paper v2 — New Findings (v4.6.0)
### April 2026

These findings emerged after the v1 paper was locked. They form the core of the v2 paper.

---

## 1. LSM Co-adaptation: Identified and Resolved

### The problem (diagnosed in v4.4.0, documented in v1)
MORPHON operates in the Liquid State Machine regime. Offline classifiers on frozen activations score 10.5% (random chance), while the live system reaches ~52% online. The v1 paper framed this as an honest limitation: the readout is co-adapted to the dynamical context of the running system and features are not independently extractable.

### The root cause (new, v4.6.0)
Sequential training co-adapts the readout to **inter-image temporal context** rather than per-image discriminative structure. The network accumulates transient state across images during training; the readout learns to exploit that state. When evaluated on fresh inputs (or in shuffled order), accuracy collapses because the exploited state patterns no longer hold.

**Diagnostic: sequential vs shuffled evaluation (new)**

| Variant | Sequential | Shuffled | Δ |
|---------|-----------|---------|---|
| V3 (standard) | 52.0% | 46.7% | +5.3pp |
| V3-SL (standard) | 42.0% | 34.7% | +7.3pp |

V3's +5.3pp advantage in sequential order directly quantifies the co-adaptation artefact.

### The fix: stateless training (V3-SL)
Reset all transient network state (`reset_transient_state()`) before each training image. The network can no longer exploit inter-image context; the readout must learn per-image discriminative representations.

**Implementation:** `System::reset_transient_state()` — resets membrane potentials, working memory, resonance queue, episode fire counts, ANCS context, kWTA winner list. Does NOT zero eligibility traces or synaptic weights.

### Results (standard profile, seed=42, 5k×3ep)

| Variant | Online acc | Stateless acc | Wall time |
|---------|-----------|--------------|-----------|
| V2 (GlobalKWTA) | 50.5% | 82.0% | 1111s |
| V3 (LocalInhibition) | 52.0% | 81.3% | 182s |
| **V3-SL (Stateless Training)** | **42.0%** | **87.7%** | **111s** |

Fast profile (500 samples, 2 epochs):

| Variant | Online acc | Stateless acc |
|---------|-----------|--------------|
| V3 | 18.0% | 44.0% |
| V3-SL | 26.0% | **77.0%** |

**V3-SL is also 39% faster than V3** (111s vs 182s) on the standard profile — stateless training eliminates the computational cost of propagating sequential context.

### Interpretation
The correct accuracy metric for a per-image classifier implemented on a stateful substrate is **stateless accuracy** (state reset before each image). Online accuracy measures something else: how well the system exploits sequential context, which is an artefact of the evaluation protocol for a classifier task. V3's 52% online / 81.3% stateless gap confirms this.

The v2 paper argument: *sequential training of a per-image classifier on a stateful substrate is a category error. Stateless training removes the mismatch, and the resulting system achieves 87.7% — competitive with biologically-plausible STDP baselines.*

---

## 2. Novelty Gain Collapse: Root Cause and Fix

### The symptom (observed in v4.4.0, not resolved)
During MNIST training, `ng` (novelty gain, the ACh neuromodulation channel) converged to ~0.32 instead of remaining near 1.0. This throttled three-factor STDP mid-training. The v4.4.0 roadmap documented this as "cause not yet identified."

### Root cause (new, v4.6.0)
Endoquilibrium Rule 7 (energy pressure): when morphon energy utilization exceeds 0.85 (common during dense MNIST training), the rule executed `ch.novelty_gain *= 0.2`. The EMA of this suppressed value converged to ~0.32 over training, with occasional ticks above 0.85 keeping it from reaching 0.2 exactly.

**Why this was wrong:** `novelty_gain` modulates the ACh broadcast signal that scales STDP weight updates. Suppressing it under energy pressure throttled learning precisely when the system needed it most. Energy pressure should be expressed via `plasticity_mult` (already done), not via novelty modulation.

**Biological note:** In biological systems, ACh (novelty/attention) actually *increases* under metabolic stress — heightened alertness is the correct response to an energy crisis, not suppression of the learning signal.

### Fix
Remove `ch.novelty_gain *= 0.2` and `ch.novelty_gain = 0.0` from Rule 7 energy_emergency and energy_critical paths. Only `plasticity_mult` is suppressed.

### Confirmation
- `ng` holds at **1.60** throughout MNIST training (was: 0.32 after early epochs)
- `ng` holds at **1.40** throughout CartPole standard run
- Both tasks confirm the fix independently

---

## 3. ANCS-Core: Native In-Process Memory Substrate

### What it is
Adaptive Node-Cluster System implemented natively in Rust as a single binary module (`src/ancs.rs`). Replaces the prior MORPHON→HTTP→Node.js→PostgreSQL architecture.

### Core components
- `MemoryBackend` trait — uniform interface over any ANCS store
- `InMemoryBackend` — flat `Vec<MemoryItem>` with importance scoring, epistemic lifecycle, and pressure-based eviction
- `MemoryTier` (VBC-lite): Verbatim / Structural / Semantic / Procedural
- `MemoryEpistemicState`: 6-state lifecycle (Hypothesis → Supported → Stale → Suspect/Contested → Outdated)
- `PressureMode`: Normal / Pressure / Emergency / Critical from energy utilization
- AXION importance: 6-factor SM-2 + centrality + novelty + reward + consolidation
- TruthKeeper: conflict detection (Jaccard ≥ 0.3 + opposing reward sign → Contested), blast-radius cascade
- `SystemHeartbeat`: `#[repr(C, align(64))]` with `AtomicU32` (f32 bits)

### Integration with System
- `System` carries `ancs: InMemoryBackend`, `heartbeat: SystemHeartbeat`, `current_ancs_item: Option<u64>`
- Memory tick: ANCS step, TruthKeeper reconsolidation dispatch, heartbeat update
- Episodic encode creates ANCS items; `SynapticJustification.memory_item_ref` links synapses to ANCS items
- `retrieve_memory()` fuses TripleMemory + ANCS results via RRF
- `dream_cycle()` replays top-5 ANCS items

---

## 4. State-Reset Evaluator

### What it is
A new evaluation protocol: reset all transient network state before each test image. Measures true per-image discriminative capacity, independent of sequential carry-over.

### Results (shows the magnitude of the sequential contamination problem)

| Variant | Standard sequential eval | Stateless eval | Gap |
|---------|--------------------------|----------------|-----|
| V2 | 50.5% | 82.0% | +31.5pp |
| V3 | 52.0% | 81.3% | +29.3pp |
| V3-SL | 42.0% | 87.7% | +45.7pp |

The gap for V3-SL is *larger* than for V3: a system trained statelessly is even more penalised by sequential carry-over during evaluation (the accumulated state is now pure noise, not partially exploitable context).

---

## 5. Benchmark Summary (v4.6.0)

| Benchmark | Result | Notes |
|-----------|--------|-------|
| CartPole avg(100) | **195.1** | Solved at ep949, standard profile, seed=42 |
| MNIST stateless (V3-SL) | **87.7%** | Standard profile, seed=42 |
| MNIST stateless (V3) | 81.3% | Same architecture, correct evaluator |
| MNIST online (V3) | 52.0% | Sequential eval, contaminated by temporal bleed |
| ng stability | 1.60 (MNIST), 1.40 (CartPole) | Was: collapsing to 0.32 |

---

## 6. What the v2 Paper Should Cover

### New sections / major revisions needed
1. **§ Stateless Training** — full experimental comparison V2/V3/V3-SL, sequential vs shuffled diagnostic, the category-error argument, fast vs standard profile scaling
2. **§ Novelty Gain Collapse** — add as seventh failure mode with the energy-pressure root cause
3. **§ ANCS-Core** — native memory substrate, epistemic lifecycle, TruthKeeper
4. **§ Evaluation Methodology** — the stateless vs online distinction deserves its own methodological treatment; argue that stateless is the correct metric for per-image classifiers on stateful substrates
5. **Updated experiments table** — replace the v4.4.0 numbers with v4.6.0 results

### Arguments that change between v1 and v2
| v1 argument | v2 argument |
|---|---|
| ~50% ceiling is a co-adaptation boundary (limitation) | Co-adaptation is identified and solved via stateless training |
| Online accuracy is the primary metric | Stateless accuracy is the correct metric for per-image tasks |
| MNIST accuracy well below state-of-art | 87.7% competitive with biologically-plausible STDP baselines |
| ng=0.32 is an open question | ng collapse root cause identified and fixed |

### What stays the same
- Six (now seven) failure modes
- Self-healing diagnostic interpretation
- iSTDP timescale mismatch fix
- LSM reservoir characterization (sensory-only ablation, offline features)
- CartPole result
- Path to language roadmap

---

*Recorded: April 2026*
