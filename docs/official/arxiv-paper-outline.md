# Morphogenic Intelligence: A Self-Organizing, Developmentally-Guided Neural Architecture

## arXiv Paper Outline

**Target venues:** NeurIPS Workshop on Neuromorphic Computing, ICONS (International Conference on Neuromorphic Systems)

**Narrative arc:** We built it, found biology-informed failure modes, fixed them with biology-informed solutions.

---

## 1. Title + Abstract

**Title:** Morphogenic Intelligence: Runtime Neural Development Beyond Static Architectures

**Abstract draft (target ~250 words):**

We introduce Morphogenic Intelligence (MI), a post-Transformer neural architecture in which compute units -- called Morphons -- grow, self-organize, differentiate, fuse, and undergo apoptosis at runtime. Unlike conventional deep learning, where network topology is fixed at design time and only weights are learned, MI treats architecture itself as the primary learned artifact. Each Morphon carries a developmental program inspired by biological morphogenesis: local chemical gradients guide connectivity, neuromodulatory signals gate plasticity, and a dual-clock system separates fast inference from slow structural remodeling. We implement the full MI engine in Rust and evaluate on CartPole control, MNIST classification, and streaming anomaly detection. Our experiments reveal characteristic failure modes -- modulatory explosion, motor silencing from burst dead zones, LTD vicious cycles, and mode collapse -- each of which maps onto known pathologies in developmental neurobiology. We show that biology-informed fixes (receptor saturation, tonic baseline restoration, metaplasticity thresholds, and morphogen diversity pressure) resolve these failures without introducing ad-hoc engineering heuristics. While MI does not yet match the classification accuracy of mature static architectures such as SADP (99.1% on MNIST), it demonstrates qualitatively distinct capabilities: zero-shot structural adaptation, graceful degradation under ablation, and emergent functional modularity. We argue that developmental dynamics represent a principled and underexplored axis of neural architecture design.

- **Key terms to define:** Morphon, morphogenesis, developmental program, resonance, neuromodulation, apoptosis
- **Claims:** (1) runtime topology change is feasible and useful, (2) biological failure modes predict computational failure modes, (3) biological fixes transfer

---

## 2. Introduction

The introduction establishes that all dominant AI architectures (Transformers, CNNs, RNNs, SSMs) share a common assumption: network topology is fixed before training begins, and learning is reduced to weight optimization within that frozen graph. We argue this is a fundamental bottleneck -- biological neural systems develop, prune, and reorganize continuously, and this structural plasticity is not merely incidental but computationally essential. The MI thesis states that a system capable of growing its own topology under local developmental rules will exhibit qualitatively different generalization and adaptation properties.

- The "static graph" assumption in modern deep learning
  - Transformers, CNNs, RNNs, SSMs all fix topology at design time
  - NAS and pruning are offline approximations, not runtime development
  - Biological brains never stop remodeling (synaptogenesis, pruning, neurogenesis in hippocampus)
- The cost of static architectures
  - Overparameterization as a substitute for structural adaptation
  - Catastrophic forgetting as a symptom of frozen topology
  - Inability to allocate capacity where it is needed at runtime
- The Morphogenic Intelligence thesis
  - Topology is the primary learned representation
  - Local rules, global emergence (no centralized controller)
  - Development as a computational paradigm, not just a metaphor
- Paper contributions (numbered list)
  1. A complete developmental neural architecture with six biological principles
  2. A Rust implementation demonstrating feasibility
  3. A taxonomy of biology-informed failure modes and their fixes
  4. Preliminary experimental results on three benchmarks

---

## 3. Related Work

This section positions MI within the landscape of biologically-inspired and self-modifying neural architectures. We distinguish MI from prior work along three axes: whether topology changes at runtime, whether the developmental rules are local, and whether the system integrates neuromodulation with structural plasticity.

- **Self-Modifying Growing Recurrent Neural Networks (SMGrNN)**
  - Topology growth via node/edge insertion
  - Difference from MI: no morphogen-guided differentiation, no apoptosis, no neuromodulatory gating
- **Self-Assembling Particle-Interaction Networks (SAPIN)**
  - Agents self-organize into functional topologies
  - Difference from MI: no synaptic plasticity, no memory hierarchy, particle abstraction vs. neuron abstraction
- **Local Neural Development Programs (LNDP) and Neural Developmental Programs (NDP)**
  - Learned local rules produce network topology
  - Difference from MI: NDP uses differentiable relaxation (soft adjacency), MI uses discrete morphogenetic events; LNDP operates in a grid world, MI in hyperbolic space
- **Cortical Labs (DishBrain)**
  - Biological neurons playing Pong -- proof that self-organizing wetware can do real-time control
  - Connection to MI: shared emphasis on intrinsic activity and reward-modulated plasticity
  - Difference from MI: biological substrate vs. in-silico developmental simulation
- **Three-factor learning rules**
  - Hebbian + neuromodulatory third factor (dopamine, acetylcholine analogues)
  - MI's receptor-gated modulation as a principled implementation
- **SADP (Self-Adaptive Dynamic Pruning)**
  - Achieves 99.1% on MNIST with dynamic sparsity
  - Difference: pruning within a fixed architecture vs. growing from nothing
  - MI's target benchmark for classification accuracy

---

## 4. Architecture

This section presents the six biological principles underlying MI and the core data structures and algorithms that implement them. Each subsection maps a biological concept to a precise computational mechanism, making the design choices auditable against the neuroscience literature.

### 4.1 The Six Biological Principles

A concise enumeration of the design axioms that constrain every architectural decision. Each principle is stated as a one-sentence invariant with a biological citation.

- **P1: Local computation only** -- no global loss, no backpropagation; all updates use locally available information
- **P2: Developmental lifecycle** -- Morphons are born, differentiate, mature, and can die (apoptosis)
- **P3: Chemical signaling** -- morphogen gradients guide connectivity and differentiation decisions
- **P4: Neuromodulatory gating** -- plasticity rules are gated by modulatory signals (dopamine/serotonin/acetylcholine analogues)
- **P5: Multi-scale memory** -- synaptic (fast), structural (medium), morphogenetic (slow) memory operate on different timescales
- **P6: Metabolic cost** -- every Morphon consumes energy; resource scarcity drives pruning and efficiency

### 4.2 The Morphon Struct

The Morphon is the fundamental compute unit. We describe its internal state (membrane potential, receptor densities, morphogen production profile, developmental age, energy budget) and its interface with neighbors. This subsection serves as a reference for all subsequent algorithmic descriptions.

- Internal state vector: membrane potential, adaptation variable, refractory timer
- Receptor profile: per-modulator sensitivity (dopamine_r, serotonin_r, acetylcholine_r)
- Morphogen emission: each Morphon produces and diffuses a set of chemical signals
- Developmental metadata: birth_tick, differentiation_state (stem / progenitor / excitatory / inhibitory / modulatory), lineage_id
- Energy budget: metabolic cost per tick, energy income from activity, apoptosis threshold
- **Figure: Morphon struct diagram** -- box diagram showing internal state, receptor interface, and morphogen emission cone

### 4.3 Resonance and Communication

Morphons communicate through resonance -- a biophysically-inspired signaling mechanism that replaces matrix multiplication. We describe the resonance kernel, how signal strength decays with distance in the embedding space, and how resonance events trigger downstream state updates.

- Resonance as spatially-decaying signal propagation (not dot-product attention)
- Resonance kernel: amplitude, frequency, phase; compatibility function between sender and receiver
- Connection formation: resonance above threshold triggers synaptogenesis
- Connection removal: sustained low resonance triggers synaptic pruning
- **Figure: Resonance field visualization** -- 2D projection of Morphon positions colored by resonance amplitude, showing emergent clusters

### 4.4 Morphogenesis Lifecycle

The developmental lifecycle is the mechanism by which the network grows, specializes, and prunes itself. We describe each lifecycle stage, the transition conditions, and the role of morphogen gradients in guiding differentiation.

- **Birth:** triggered by sustained high activity in a region (demand-driven neurogenesis)
- **Differentiation:** stem Morphons read local morphogen concentrations to select cell type
  - Gradient thresholds determine excitatory vs. inhibitory vs. modulatory fate
- **Maturation:** connection stabilization, receptor tuning, critical period closing
- **Fusion:** two Morphons with highly correlated activity merge into one (redundancy reduction)
- **Apoptosis:** energy-depleted or chronically inactive Morphons are removed
  - Graceful death: outgoing connections are redistributed before removal
- **Figure: Lineage tree** -- dendrogram showing Morphon birth, differentiation, fusion, and death events over developmental time

### 4.5 Neuromodulation

Neuromodulation provides the "third factor" that gates plasticity and biases developmental decisions. We describe the four modulatory channels, their computational roles, and how receptor densities determine a Morphon's sensitivity to each signal.

- **Dopamine analogue:** reward prediction error; gates reinforcement of recently active synapses
- **Serotonin analogue:** tonic mood / exploration-exploitation balance; modulates morphogenesis rate
- **Acetylcholine analogue:** attentional gating; sharpens receptive fields during salient input
- **Norepinephrine analogue:** arousal / global gain; scales firing thresholds network-wide
- Receptor-gated modulation: effect = modulator_level * receptor_density * plasticity_eligibility
- Receptor saturation as a safety mechanism against modulatory explosion (see Section 7)

### 4.6 Triple Memory System

MI implements three memory systems operating at different timescales, loosely corresponding to synaptic plasticity, systems consolidation, and epigenetic memory in biological brains.

- **Synaptic memory (fast, ~ms):** weight changes via trace-based STDP, eligibility traces
- **Structural memory (medium, ~minutes):** topology changes -- new connections, pruning, Morphon birth/death
- **Morphogenetic memory (slow, ~hours):** developmental program parameters, morphogen sensitivity profiles
  - Encodes "what kind of network to grow" rather than "what weights to set"
- Interaction between timescales: fast synaptic changes can trigger structural consolidation; structural patterns feed back into morphogenetic programs
- **Figure: Memory timescale diagram** -- three-tier schematic with example events at each timescale

### 4.7 Hyperbolic Embedding Space

Morphons are embedded in hyperbolic space rather than Euclidean space. We justify this choice by the natural tree-like hierarchy of developmental structures and the exponential volume growth of hyperbolic space, which provides room for topology expansion without coordinate crowding.

- Hyperbolic space (Poincare disk model) as the embedding manifold
- Why hyperbolic: biological neural networks have hierarchical, tree-like branching; hyperbolic space represents trees with low distortion
- Exponential volume growth accommodates dynamic topology expansion
- Distance in hyperbolic space governs resonance decay and connection probability
- Morphon position updates via Riemannian SGD on the Poincare disk

### 4.8 Dual-Clock System

A dual-clock architecture separates fast neural dynamics (inference) from slow developmental dynamics (morphogenesis). This prevents structural changes from destabilizing ongoing computation and mirrors the separation of fast electrophysiology from slow developmental timescales in biological brains.

- **Fast clock (inference tick):** membrane potential updates, spike propagation, neuromodulator release
- **Slow clock (developmental tick):** morphogenesis events, apoptosis checks, morphogen diffusion, structural consolidation
- Ratio between clocks is a hyperparameter (default: 100 inference ticks per developmental tick)
- Gating: developmental events are suppressed during high-salience inference periods (attentional gating)

---

## 5. Implementation

This section describes the Rust implementation, focusing on engineering decisions that make MI feasible as a real-time system. We provide enough detail for reproducibility without turning the paper into a systems manual.

### 5.1 Rust Engine

The choice of Rust is motivated by the need for memory safety without garbage collection pauses, zero-cost abstractions for the inner simulation loop, and fearless concurrency for parallel Morphon updates. We describe the core event loop, data layout, and parallelism strategy.

- Core simulation loop: tick-based discrete event simulation
- Data layout: struct-of-arrays for cache-friendly Morphon state updates
- Parallelism: rayon-based data parallelism over Morphon populations; no global locks on the fast path
- Memory management: arena allocation for Morphon structs, generational indices for stable references across birth/death events
- Performance targets: 10k Morphons at real-time on a single core

### 5.2 Developmental Programs

Each Morphon type carries a developmental program -- a compact set of rules that govern its lifecycle transitions. We describe the DSL for developmental programs and how they are parameterized.

- Developmental program as a finite state machine with continuous guards
- States: stem, progenitor, excitatory, inhibitory, modulatory, apoptotic
- Transition guards: morphogen concentration thresholds, energy levels, activity statistics, developmental age
- Program inheritance: daughter Morphons inherit (possibly mutated) programs from parent
- Programs are the slow-timescale learned artifact (morphogenetic memory)

### 5.3 I/O Pathways

How external inputs enter the Morphon network and how outputs are read out. We describe sensory Morphons, motor Morphons, and the encoding/decoding schemes.

- **Sensory Morphons:** pinned to input channels, convert external signals to spike trains (rate coding or temporal coding)
- **Motor Morphons:** pinned to output channels, decode via firing rate integration over a readout window
- **Reward channel:** scalar reward signal converted to dopamine-analogue release (global broadcast, receptor-gated local effect)
- No fixed hidden-layer structure; all intermediate topology is emergent

### 5.4 Receptor-Gated Modulation

Implementation details of the neuromodulatory system, including the receptor saturation curves that prevent modulatory explosion.

- Modulator diffusion: exponential decay from release site
- Receptor binding: Hill-function saturation curve (prevents runaway amplification)
- Receptor adaptation: sustained high modulator exposure downregulates receptor density
- Per-Morphon receptor profiles evolve on the slow clock

### 5.5 Contrastive Reward Signal

The reward mechanism that replaces backpropagated gradients. We describe how a scalar reward is transformed into a spatially-distributed modulatory signal that credits recently active pathways.

- Reward prediction error (RPE): delta = reward - baseline (exponential moving average)
- RPE broadcast as dopamine-analogue pulse
- Credit assignment via eligibility traces: only synapses with recent pre-post coincidence are eligible
- Contrastive component: negative RPE triggers LTD on eligible synapses, positive RPE triggers LTP
- Baseline adaptation rate as a critical hyperparameter

### 5.6 Trace-Based STDP

The local synaptic learning rule. We describe the spike-timing-dependent plasticity implementation with eligibility traces and neuromodulatory gating.

- Pre-synaptic trace: exponential decay, incremented on pre-synaptic spike
- Post-synaptic trace: exponential decay, incremented on post-synaptic spike
- Eligibility trace: product of pre and post traces, decays on its own timescale
- Weight update: delta_w = learning_rate * eligibility * modulator_gate * (LTP_term - LTD_term)
- LTP/LTD balance ratio as a tunable parameter (see Section 7 for failure modes when mistuned)

---

## 6. Experiments

We evaluate MI on three tasks chosen to probe different capabilities: reactive control (CartPole), static classification (MNIST), and streaming adaptation (anomaly detection). The goal is not to claim state-of-the-art accuracy but to demonstrate that a self-organizing developmental architecture can solve non-trivial problems and to characterize its distinctive behaviors.

### 6.1 CartPole (Continuous Control)

CartPole is chosen as a minimal test of real-time sensorimotor integration. The network must grow from a minimal seed and develop a control policy through reward-modulated plasticity alone.

- **Setup:** OpenAI Gym CartPole-v1; 4 sensory Morphons (cart position, velocity, pole angle, angular velocity), 2 motor Morphons (left/right force)
- **Seed network:** 4 sensory + 2 motor + 6 stem Morphons (no hidden structure)
- **Metrics:** episode reward over developmental time, number of active Morphons, topology statistics (degree distribution, clustering coefficient)
- **What to show:** (1) learning curve with topology growth overlaid, (2) comparison of final topology across 5 seeds (structural diversity despite functional convergence), (3) ablation: disable morphogenesis (frozen topology) vs. full MI
- **Figure: Topology growth visualization** -- snapshots of the Poincare disk embedding at developmental ticks 0, 100, 500, 1000 showing Morphon positions and connections

### 6.2 MNIST (Static Classification)

MNIST tests whether MI can grow a discriminative topology from scratch. This is the hardest benchmark for MI because classification of static images does not play to the strengths of a temporal, developmental system.

- **Setup:** 784 sensory Morphons (one per pixel, rate-coded), 10 motor Morphons (one per digit class), readout via argmax of firing rates over a 50-tick window
- **Seed network:** 784 sensory + 10 motor + 50 stem Morphons
- **Training:** images presented sequentially, reward = 1.0 for correct classification, 0.0 otherwise
- **Metrics:** test accuracy over developmental time, Morphon count, connection count, energy consumption
- **What to show:** (1) accuracy curve compared to a static network of equivalent parameter count, (2) emergent functional specialization (do Morphons develop digit-selective tuning?), (3) comparison to SADP's 99.1% (honest gap analysis)
- **Figure: Receptive field visualization** -- for selected hidden Morphons, plot input weights as 28x28 heatmaps; compare to Gabor-like filters in biological V1
- **Figure: Firing rate evolution** -- per-class mean firing rates of motor Morphons over training, showing sharpening of selectivity

### 6.3 Anomaly Detection (Streaming Adaptation)

Anomaly detection in a non-stationary stream tests MI's ability to adapt its topology to changing data distributions -- the scenario where developmental plasticity should provide the clearest advantage over static architectures.

- **Setup:** synthetic time series with injected regime changes (mean shift, variance change, periodic pattern onset)
- **Sensory encoding:** sliding window of 20 timesteps, rate-coded into 20 sensory Morphons
- **Anomaly signal:** a dedicated motor Morphon whose firing rate indicates anomaly confidence
- **Metrics:** F1 score, detection latency, false positive rate, topology change rate (births + deaths per developmental tick)
- **What to show:** (1) detection performance across regime changes, (2) correlation between topology change rate and actual distribution shifts (does the network "notice" distribution change structurally?), (3) comparison to a static autoencoder baseline
- **Figure: Topology change rate vs. anomaly timeline** -- dual-axis plot showing anomaly ground truth and Morphon birth/death rate, demonstrating structural response to distributional shift

---

## 7. Analysis: Biology-Informed Failure Modes and Fixes

This is the core contribution section. We document four failure modes discovered during development, show that each has a direct biological analogue, and demonstrate that the biological solution transfers to the computational setting. The structure is: (1) describe the failure, (2) identify the biological parallel, (3) describe the fix, (4) show the fix works.

### 7.1 Modulatory Explosion

Positive feedback loop in which high reward triggers dopamine release, which potentiates active synapses, which increases reward, which triggers more dopamine. The network locks into a single high-activation pattern.

- **Symptom:** all motor Morphons saturate at maximum firing rate; reward signal becomes meaningless
- **Biological parallel:** dopaminergic excitotoxicity; in biology, receptor desensitization and reuptake prevent this
- **Fix:** receptor saturation (Hill function with n=2), receptor downregulation on sustained exposure, modulator reuptake (exponential decay from release site)
- **Result:** stable modulatory dynamics; dopamine signal retains information content
- **Figure: Modulator concentration over time** -- before/after fix, showing runaway vs. bounded dynamics

### 7.2 Motor Silencing from Burst Dead Zone

When Morphons enter a high-frequency bursting regime, the STDP window becomes ineffective because inter-spike intervals are shorter than the LTP window. Synapses onto motor Morphons weaken and motor output goes silent.

- **Symptom:** motor Morphons stop firing entirely after an initial burst phase; network becomes unresponsive
- **Biological parallel:** depolarization block; in biology, tonic baseline activity and intrinsic excitability homeostasis prevent complete silencing
- **Fix:** tonic baseline current injection into motor Morphons (analogous to intrinsic excitability), adaptive STDP window that widens under low-activity conditions
- **Result:** motor Morphons maintain baseline activity; STDP remains effective across firing rate regimes
- **Figure: Motor Morphon firing rate over time** -- before/after fix, showing collapse-to-zero vs. maintained baseline

### 7.3 LTD Vicious Cycle

When the LTD component of STDP is too strong relative to LTP, weakened synapses carry less current, which reduces post-synaptic firing, which further increases the LTD/LTP ratio (because post-before-pre events dominate), which weakens synapses further. The network self-destructs.

- **Symptom:** global weight decay to near-zero; network falls silent; all Morphons die from energy starvation
- **Biological parallel:** runaway LTD / synaptic scaling failure; in biology, homeostatic synaptic scaling (Turrigiano) and BCM-like metaplasticity thresholds prevent this
- **Fix:** BCM-like sliding threshold -- the LTP/LTD crossover point shifts based on recent post-synaptic activity history; when activity is low, the threshold drops, making LTP easier to trigger
- **Result:** network maintains stable mean firing rate; weight distribution has healthy variance
- **Figure: Weight distribution histograms** -- before fix (collapsing to zero) vs. after fix (stable bimodal or log-normal distribution, matching biological observations)

### 7.4 Mode Collapse in Morphogenesis

All stem Morphons differentiate into the same cell type (typically excitatory) because the first-mover advantage creates a morphogen gradient that biases all subsequent differentiation decisions in the same direction. The network lacks inhibitory balance.

- **Symptom:** E/I ratio diverges from ~80/20; network exhibits pathological synchrony (analogous to epileptiform activity)
- **Biological parallel:** disrupted E/I balance in cortical development; in biology, Notch-Delta lateral inhibition ensures neighbor cells adopt different fates
- **Fix:** morphogen diversity pressure -- a lateral-inhibition-inspired mechanism where recently differentiated Morphons emit a "taken" signal that biases neighbors toward complementary fates
- **Result:** stable E/I ratio near 80/20; network exhibits desynchronized, computationally useful dynamics
- **Figure: E/I ratio over developmental time** -- before/after fix, showing runaway excitatory dominance vs. stable balance

---

## 8. Discussion

The discussion contextualizes MI's current limitations honestly, identifies the key open problems, and positions the work relative to the broader neuromorphic computing landscape.

### 8.1 The Credit Assignment Gap

MI's local learning rules (trace-based STDP + neuromodulatory gating) are less sample-efficient than backpropagation for supervised tasks. We quantify this gap on MNIST and discuss whether it is fundamental or an artifact of immature hyperparameter tuning. The contrastive reward signal provides only a scalar error signal, whereas backpropagation provides per-weight gradients -- bridging this gap without abandoning locality is the central open problem.

- Sample efficiency comparison: MI vs. backprop-trained network of equivalent size
- Information-theoretic argument: scalar reward carries O(1) bits per update vs. O(n) bits for full gradient
- Possible mitigation: hierarchical reward decomposition, local contrastive objectives at each Morphon

### 8.2 Scalability

Current implementation handles ~10k Morphons in real time. We discuss the scaling bottlenecks (morphogen diffusion is O(n^2) naively, resonance computation likewise) and potential solutions (spatial hashing, approximate nearest-neighbor in hyperbolic space, GPU offload for the fast clock).

- Profiling data: where does wall-clock time go?
- Spatial partitioning in hyperbolic space for sub-quadratic morphogen diffusion
- Comparison to biological scale: cortical columns contain ~100k neurons; MI at 10k is at "minicolumn" scale

### 8.3 Comparison to SADP and Static Architectures

Honest comparison to SADP's 99.1% MNIST accuracy. We argue that raw accuracy on static benchmarks is not the right metric for MI -- the point is not to beat Transformers at their own game but to demonstrate capabilities they lack (structural adaptation, graceful degradation, zero-shot transfer across tasks via topology reuse).

- SADP comparison: accuracy gap, parameter efficiency, adaptation speed
- Qualitative advantages of MI: ablation robustness (remove 20% of Morphons, measure graceful degradation vs. catastrophic failure in static nets)
- The right benchmarks for developmental architectures: continual learning, distribution shift, multi-task without replay

---

## 9. Conclusion + Future Work

We summarize the contributions and lay out a concrete research roadmap. The conclusion reinforces the narrative: developmental dynamics are not just a biological curiosity but a computationally principled design axis that reveals (and resolves) failure modes invisible to static architectures.

### Conclusion

- MI demonstrates that runtime neural development is feasible and produces functional networks from minimal seeds
- Biology-informed failure modes are predictive: knowing the neuroscience helped us diagnose and fix computational pathologies
- The architecture is not yet competitive on standard benchmarks, but exhibits qualitatively distinct capabilities (structural adaptation, emergent modularity, graceful degradation)

### Future Work

- **Hierarchical morphogenesis:** allow Morphon populations to self-organize into cortical-column-like modules, enabling compositional reasoning
- **Multi-task development:** train a single MI network on multiple tasks simultaneously; study whether functional specialization emerges (analogous to cortical area formation)
- **Hardware mapping:** MI's local-only computation maps naturally onto neuromorphic hardware (Loihi, SpiNNaker); explore deployment on physical neuromorphic chips
- **Evolved developmental programs:** use evolutionary strategies to optimize the developmental programs themselves (morphogenetic memory), creating an outer loop of "phylogenetic" optimization around the inner loop of "ontogenetic" development
- **Theoretical analysis:** characterize the expressivity of developmental architectures -- what function classes can MI represent that static networks cannot (or can represent more efficiently)?
- **Scaling to language:** investigate whether MI can grow topology suitable for sequential prediction tasks, potentially as a neuromorphic alternative to Transformer-based language models

---

## Appendix: Planned Figures Summary

| Figure | Section | Description |
|--------|---------|-------------|
| Morphon struct diagram | 4.2 | Box diagram of internal state, receptors, morphogen emission |
| Resonance field visualization | 4.3 | 2D Poincare disk projection colored by resonance amplitude |
| Lineage tree | 4.4 | Dendrogram of Morphon birth, differentiation, fusion, death |
| Memory timescale diagram | 4.6 | Three-tier schematic with example events per timescale |
| Topology growth (CartPole) | 6.1 | Poincare disk snapshots at developmental ticks 0, 100, 500, 1000 |
| Receptive fields (MNIST) | 6.2 | 28x28 heatmaps of input weights for selected hidden Morphons |
| Firing rate evolution (MNIST) | 6.2 | Per-class motor Morphon firing rates over training |
| Topology change rate vs. anomaly | 6.3 | Dual-axis plot: anomaly ground truth + birth/death rate |
| Modulator dynamics | 7.1 | Before/after modulatory explosion fix |
| Motor firing rate | 7.2 | Before/after burst dead zone fix |
| Weight distributions | 7.3 | Histograms before/after LTD vicious cycle fix |
| E/I ratio over time | 7.4 | Before/after morphogenesis mode collapse fix |
