# Compositionality in Morphons — Why We're Different From Transformers

## ⚠️ Dead Path: XOR Circuits

**Status:** Investigated and abandoned. See "Why It Failed" below.

We attempted to solve the XOR problem (needed for NLP Tier 3 compositionality) by hardwiring a dedicated circuit with pre-tuned weights into the main topology. This approach was tested extensively and **consistently failed at ~50% accuracy** (chance level).

### What Was Tried

1. **Pre-wired XOR circuit** — 3 morphons (Core A, Core B, Output) with fixed weights (+2.0/-2.0)
2. **Multiple weight configurations** — Tested various excitation/inhibition ratios
3. **Different thresholds** — Tried 0.3, 0.4, 0.5, 0.7 for cores and output
4. **With and without interneuron** — Both full and minimal circuit variants
5. **Pretrained vs learned** — Both static weights and training through three-factor learning

### Why It Failed

**Root cause:** The circuit morphons were added to the main topology, where they received signals from ALL ~46 firing morphons. The intended +2.0/-2.0 XOR signals were completely drowned out by cumulative noise.

Debug output confirmed both cores clamped at -10.0 for ALL patterns — they never fired:

```
00: out=0.02 ca=-10.0() cb=-10.0() exp=0 act=0 ✓
01: out=-0.04 ca=-10.0() cb=-10.0() exp=1 act=0 ✗
10: out=-0.00 ca=-10.0() cb=-10.0() exp=1 act=0 ✗
11: out=0.06 ca=-10.0() cb=-10.0() exp=0 act=0 ✓
```

### Why It Can't Work

1. **Topology noise is unavoidable** — Any morphon in the main topology receives signals from all connected morphons. Isolating the circuit would require removing it from the topology, which breaks signal propagation.

2. **Static inputs don't work with spiking dynamics** — Research (Moser & Lunglmayr 2024) shows SNNs solve XOR with temporal encoding (spike frequencies), not static voltage levels. Even with optimal conditions, only 3-7% of random weight initializations succeed.

3. **Doesn't scale** — The circuit works for 2 binary inputs. NLP Tier 3 has 54 inputs. The approach doesn't generalize.

### What Remains

The `xor_circuit.rs` module exists in the codebase but is **non-functional**. It should be either:
- Removed entirely, or
- Repurposed as part of the DeMorphon lifecycle (where internal wiring IS isolated from the main topology)

**Do not invest more time in this approach.**

---

## The Fundamental Difference: Morphons vs Transformers

Transformers solve compositionality through **attention over token embeddings**:
- Tokens are projected into a shared vector space
- Self-attention computes pairwise relationships
- Feed-forward layers apply nonlinear transformations
- The output is a weighted sum of attended representations

**Morphons are not transformers.** They don't have attention layers, token embeddings, or feed-forward networks. They solve the same problem through completely different mechanisms:

| Aspect | Transformers | Morphons |
|--------|-------------|----------|
| Representation | Dense token embeddings | Distributed spike patterns |
| Composition | Attention-weighted sums | Attractor dynamics in topology |
| Learning | Backpropagation through layers | Three-factor STDP + neuromodulation |
| Structure | Fixed architecture | Self-organizing through morphogenesis |
| Time | Positional encoding (static) | Intrinsic temporal dynamics |
| Binding | Attention scores | Spike synchronization + resonance |

---

## How Morphons Actually Solve Composition

### 1. Attractor Dynamics (Nam et al., 2023)

The most relevant paper: **"Discrete, compositional, and symbolic representations through attractor dynamics"** (Nam, Elmoznino, Malkin, McClelland, Bengio, Lajoie — Princeton/Mila/Stanford, 2023).

Key finding: **Discrete, compositional representations emerge from attractor dynamics in neural systems without any explicit symbolic structure.**

The mechanism:
1. Input creates an initial state in the representational space
2. The system's dynamics (learned transition function) evolve this state over time
3. The trajectory converges to an **attractor basin** — a stable state
4. Different inputs converge to different attractors
5. **Composition**: combined inputs converge to attractors that are NOT simple averages of the individual attractors

This is exactly what Morphons do:
- Sensory input creates initial potentials in the network
- Spike propagation + leaky integration evolve the state
- The system converges to a stable firing pattern (attractor)
- The readout classifies based on which attractor was reached

### 2. Spike Timing and Binding (Zheng et al., 2022)

**"Dance of SNN and ANN: Solving binding problem by combining spike timing and reconstructive attention"** (Tsinghua, NeurIPS 2022).

The binding problem: how do you represent "red square" vs "red circle" vs "blue square" without confusing the features?

Their solution: **synchronized spike timing**. Features that belong together fire in synchrony. Different objects fire at different times.

Morphons have this naturally:
- The resonance engine propagates spikes with delays
- Correlated inputs create synchronized firing patterns
- The topology's weighted connections determine which morphons fire together
- Three-factor learning strengthens connections between synchronized morphons

### 3. The Binding Problem in ANNs (Greff et al., 2020)

**"On the Binding Problem in Artificial Neural Networks"** (Google Brain/IDSIA, 2020).

Key mechanisms for binding in neural networks:
- **Synchronization**: neurons representing the same object fire together
- **Routing**: dynamic assignment of features to slots
- **Slot attention**: iterative refinement of object representations

Morphons implement synchronization through the resonance engine and routing through the topology's weighted connections. The hyperbolic geometry naturally creates "slots" — morphons near each other in the Poincaré ball form functional clusters.

---

## The Right Approach: Temporal Encoding + Attractor Dynamics

### 1. Sequential Input (Not Simultaneous)

Present characters/words **one at a time**, not as a bag. Each character triggers a spike pattern in sensory morphons. The timing of spikes encodes the character identity.

```
Input: "cat"
Step 1: 'c' → sensory morphon A fires
Step 2: 'a' → sensory morphon B fires
Step 3: 't' → sensory morphon C fires
```

The associative layer integrates these over time. The trajectory through state space is different for "cat" vs "act" vs "tac" — even though they contain the same characters.

### 2. Let Attractors Emerge

Don't hardwire anything. Present many examples and let the system learn an energy landscape where:
- Similar sequences converge to similar attractors
- Different sequences converge to different attractors
- The readout learns to classify based on which attractor was reached

### 3. Use the Multiple Timescales

| Timescale | Language Role |
|-----------|--------------|
| Fast (1) | Character-level spike propagation |
| Medium (10) | Word-level integration (eligibility traces) |
| Slow (100) | Phrase-level structure formation |
| Glacial (1000) | Grammar rule consolidation |

The dual-clock scheduler already supports this.

### 4. Hyperbolic Geometry for Hierarchy

The Poincaré ball naturally represents hierarchies:
- Origin = general/stem (common words: "the", "is")
- Mid-radius = phrases ("the cat", "is running")
- Boundary = specialized (rare words, specific meanings)

Migration during the slow timescale moves morphons to their appropriate hierarchical position.

### 5. Neuromodulation for Learning Signals

| Modulator | Language Role |
|-----------|--------------|
| Reward | Correct prediction → strengthen connections |
| Novelty | New word/structure → increase plasticity |
| Arousal | Important context → boost signal |
| Homeostasis | Prevent overfitting → stabilize representations |

---

## DeMorphons: The Biological Approach

The DeMorphon spec describes a different path: composite organisms with internal body plans that naturally solve composition. Each DeMorphon is like a mini-column in cortex — it receives inputs, processes them internally, and produces an output.

The key insight: **DeMorphons compete as units through their Output cells**, and their internal wiring is ISOLATED from the main topology. This solves the noise problem that killed the XOR circuit approach.

This is more biologically plausible and more efficient. But it requires the full DeMorphon lifecycle (~35 days per roadmap).

---

## Comparison: Approaches to Composition

| Approach | Status | Mechanism | Verdict |
|----------|--------|-----------|---------|
| **Hardwired XOR circuit** | ❌ Dead | Pre-tuned weights in main topology | Failed — drowned by topology noise |
| **Transformers** | ✅ Proven | Attention over embeddings | Works but biologically implausible |
| **Morphons (attractor)** | ⏳ Unproven | Temporal encoding + attractor dynamics | Should work, needs validation |
| **DeMorphons** | 📋 Spec'd | Internal body plans + isolated wiring | Best long-term, not yet built |

---

## Conclusion

**The XOR circuit approach is dead.** It was thoroughly tested and consistently fails. The hardwired circuit cannot overcome the noise from the main topology, and static input encoding is fundamentally incompatible with how SNNs solve XOR.

**The path forward for NLP is:**
1. **Temporal encoding** — sequential input, not simultaneous
2. **Attractor dynamics** — let the system learn the energy landscape
3. **Gradual scaling** — characters → words → phrases → sentences
4. **Eventually DeMorphons** — for efficient, isolated composition modules

---

## References

- Nam et al. (2023) — "Discrete, compositional, and symbolic representations through attractor dynamics" — arXiv:2310.01807
- Zheng et al. (2022) — "Dance of SNN and ANN: Solving binding problem by combining spike timing and reconstructive attention" — NeurIPS 2022
- Greff et al. (2020) — "On the Binding Problem in Artificial Neural Networks" — arXiv:2012.05208
- Moser & Lunglmayr (2024) — "On the Solvability of the XOR Problem by Spiking Neural Networks" — arXiv:2408.05845
- DeMorphon Spec — `docs/specs/future/demorphon-spec-v1.md`
