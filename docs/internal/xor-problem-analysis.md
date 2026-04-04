# XOR Function — Why We Need It, Why It's Broken, Full Problem Analysis

## What XOR Is Needed For

### 1. NLP Readiness Benchmark — Tier 3 "Composition"

The NLP readiness benchmark (`examples/nlp_readiness.rs`) has 4 tiers:

| Tier | Task | Input Dim | Purpose |
|------|------|-----------|---------|
| 0 | Bag-of-Chars | 27 | Character distribution discrimination |
| 1 | One-Hot Scale | 135 | Handle full text encoding dimensionality |
| 2 | Memory | 27×3 steps | Remember across sequential inputs |
| **3** | **Composition (XOR)** | **54** | **Combine token meanings** |

Tier 3 tests **compositional reasoning**: given two single-character "tokens", classify whether they're from the same group (both vowels or both consonants) or different groups (one vowel, one consonant). This is XOR:

```
Token A    Token B    Same Group?    XOR Output
vowel      vowel      YES (0)        0
consonant  consonant  YES (0)        0
vowel      consonant  NO  (1)        1
consonant  vowel      NO  (1)        1
```

**Why this matters for NLP**: Language is fundamentally compositional. "cat sat" means something different from "cat" + "sat" separately. The system must combine token representations, not just process them independently. Without XOR capability, the system can only do linear classification — it can recognize "vowel-heavy" vs "consonant-heavy" text (Tiers 0-1) and remember the first character (Tier 2), but it cannot combine two tokens to determine their relationship.

**Current status**: Tiers 0-2 pass. Tier 3 fails at ~40% (needs 60%). The NLP readiness level is stuck at 2/3.

### 2. DeMorphon Validation (Spec Section 11.2)

The DeMorphon spec (`docs/specs/future/demorphon-spec-v1.md`) defines XOR as a synthetic validation task:

> **Task:** Compute XOR(A, B) — output 1 if exactly one of two inputs is active.
> **Expected result:** Individual Morphon readout cannot learn XOR (it's a linear threshold unit). DeMorphon with internal inhibitory cross-connection can.

This is the proof that DeMorphons (composite organisms of specialized Morphons) can perform computations that individual Morphons cannot. Without this, the entire DeMorphon concept is unvalidated.

### 3. Roadmap Item 7.8

The complete roadmap (`docs/plans/morphon-complete-roadmap.md`) lists:

> | 7.8 Validation: XOR | Synthetic task: compute XOR(A,B) | 3 days |

And the DeMorphon completion criterion:

> **Gate:** At least one emergent computation (temporal, XOR, or working memory) that individual Morphons cannot perform.

XOR is one of three possible "gates" to prove DeMorphons work.

---

## Why XOR Is Not Working

### The Debug Evidence

Running the benchmark with detailed diagnostics shows:

```
00: out=0.02 ca=-10.0() cb=-10.0() exp=0 act=0 ✓
01: out=-0.04 ca=-10.0() cb=-10.0() exp=1 act=0 ✗
10: out=-0.00 ca=-10.0() cb=-10.0() exp=1 act=0 ✗
11: out=0.06 ca=-10.0() cb=-10.0() exp=0 act=0 ✓
```

**Both cores are clamped at -10.0 for ALL patterns.** They never fire. The output is always ~0.

### Root Cause Analysis

The XOR circuit creates 3 new morphons (Core A, Core B, Output) and wires them with pre-tuned weights:

```
Input A → Core A: +2.0    Input A → Core B: -2.0
Input B → Core A: -2.0    Input B → Core B: +2.0
Core A → Output: +1.5     Core B → Output: +1.5
```

The circuit morphons are added to the **main topology**. This means they receive signals from ALL firing morphons in the system (~46 morphons), not just the XOR inputs.

**The signal flow:**

1. `feed_input([3.0, 0.0])` → sensory morphons get input
2. Sensory morphons fire → spikes propagate through ALL their outgoing synapses
3. The XOR cores receive spikes from the sensory morphons (+2.0/-2.0 as designed)
4. **BUT** the XOR cores ALSO receive spikes from ~44 other firing morphons through the main topology's dense connectivity
5. The noise from 44 other morphons completely drowns out the +2.0/-2.0 XOR signals
6. Cores get clamped at -10.0 and never fire

### Why the Noise Is So Bad

The main topology has 138 synapses connecting 46 morphons. When morphons fire, they send spikes to ALL their outgoing connections. The XOR cores are in this topology, so they receive:

- The intended +2.0/-2.0 from the XOR inputs (2 synapses)
- **Plus** signals from ~44 other morphons through the main topology's synapses

The cumulative noise from 44 morphons far exceeds the +2.0/-2.0 signal.

---

## Why This Is Fundamentally Hard (Research)

### Moser & Lunglmayr (2024) — "On the Solvability of the XOR Problem by SNNs"

Key findings:

1. **Temporal encoding is required**: 0 and 1 must be encoded as different spike frequencies or timing patterns, NOT as static voltage levels
2. **Reset mechanism matters**: reset-to-mod gives 3x higher solvability than reset-by-subtraction
3. **2 hidden neurons minimum**: with graded spikes and proper encoding
4. **Success rate**: Even with optimal conditions, only **3-7%** of random weight initializations solve XOR

The paper's architecture uses:
- A single input neuron encoding 0/1 as different spike frequencies
- A reservoir of 2 fully-connected hidden LIF neurons with random weights
- A linear output classifier that learns decoder weights

**Our approach is fundamentally different**: We use static input values (0 or 3.0), not spike frequencies. The Morphon model integrates these as continuous potentials, not spike trains. This is incompatible with how SNNs solve XOR.

### Why Other SNN XOR Solutions Don't Apply

| Approach | What It Uses | Why We Can't Use It |
|----------|-------------|---------------------|
| Temporal encoding (Moser 2024) | 0=late spike, 1=early spike | Morphon uses static voltage, not spike timing |
| Rate encoding (Wade 2007) | 0=50Hz, 1=100Hz spike trains | Morphon doesn't encode frequency |
| Receptive fields (Reljan-Delaney 2017) | Frequency-selective filters | Morphon has no frequency selectivity |
| Izhikevich neurons (Enriquez-Gaytan 2018) | Advanced neuron model with 16 weights found by genetic algorithm | Morphon uses simpler LIF model |
| SpikeProp (Bohte 2002) | Backpropagation through spike times | Morphon uses three-factor learning, not backprop |

---

## The Full Problem Statement

**XOR is needed for:**
1. NLP Readiness Tier 3 (compositional reasoning) — currently stuck at Level 2/3
2. DeMorphon validation (proof of emergent computation)
3. Roadmap completion criterion

**XOR is broken because:**
1. The circuit morphons are in the main topology and receive noise from all ~46 firing morphons
2. The pre-wired +2.0/-2.0 signals are drowned out by cumulative noise
3. Cores clamp at -10.0 and never fire
4. Output stays at ~0 for all patterns

**Why fixing it is hard:**
1. SNNs solve XOR with temporal encoding (spike frequencies), not static values
2. Even with optimal conditions, only 3-7% of random weight initializations succeed
3. The Morphon model's static input encoding is fundamentally incompatible with how SNNs solve XOR
4. Isolating the XOR circuit from the main topology would require architectural changes

**What would actually work:**
1. **Temporal encoding**: Encode 0/1 as different spike frequencies or timing patterns
2. **Reset-to-mod**: Implement instantaneous charge-discharge instead of reset-to-zero
3. **Isolated circuit**: Don't add XOR cores to the main topology; compute output analytically
4. **DeMorphon body plan**: The full DeMorphon spec proposes internal role specialization with Input cells, Core cells, Memory cells, and Output cells — this would naturally isolate the circuit from the main topology

**What we tried and why it failed:**
1. Pre-wired weights — cores overwhelmed by topology noise
2. Different weight values — still clamped at -10.0
3. Different thresholds — still clamped
4. Adding interneuron — still clamped
5. More training — no improvement (circuit never fires, so no learning signal)
