# Paper References — Representational Drift & Heterogeneous Plasticity

Collected 2026-04-01 during CartPole convergence research.

---

## Core Papers

### Heterogeneous Plasticity & Stability

1. **Zenke, Agnes & Gerstner (2015)** — "Diverse synaptic plasticity mechanisms orchestrated to form and retrieve memories in spiking neural networks"
   - Nature Communications 6:6922
   - https://www.nature.com/articles/ncomms7922
   - **Key:** Triplet STDP + heterosynaptic depression + transmitter-induced potentiation. Three timescales: ms (short-term plasticity, rate bistability), seconds (STDP induction), minutes (consolidation). Network: 4096 excitatory + 1024 inhibitory neurons. At low rates, transmitter-induced potentiation counteracts homosynaptic LTD (prevents silence). At high rates, heterosynaptic depression prevents runaway potentiation.
   - **Relevance:** Direct blueprint for heterogeneous plasticity timescales in MORPHON.

2. **Maryada et al. (2025)** — "Stable recurrent dynamics in heterogeneous neuromorphic systems using excitatory and inhibitory plasticity"
   - Nature Communications
   - https://www.nature.com/articles/s41467-025-60697-2
   - **Key:** Cross-homeostatic plasticity — excitatory weights increase when inhibitory weights are too strong, and vice versa. Target firing rate acts as setpoint. Paradoxical stabilization through E/I balance.
   - **Relevance:** Already in our reference list (#28). Validates heterogeneous E/I plasticity for stability.

3. **Perez-Nieves et al. (2021)** — "Neural heterogeneity promotes robust learning"
   - Nature Communications 12:5791
   - https://www.nature.com/articles/s41467-021-26022-3
   - **Key:** Networks with heterogeneous membrane time constants significantly outperform homogeneous ones. Optimal tau distribution: Gamma-shaped, mean ~20ms, range 2-200ms. Excitatory tau ~30ms mean, inhibitory tau ~10ms mean (3:1 ratio). Performance gain: 5-15% on temporal benchmarks. Much less hyperparameter sensitivity.
   - **Relevance:** Concrete numbers for per-morphon `tau_membrane` heterogeneity.

4. **Pilzak, Pennington & Thivierge (2026)** — "Intrinsic stabilization of synaptic plasticity improves learning and robustness in artificial neural networks"
   - Nature Communications
   - https://www.nature.com/articles/s41467-026-70920-3
   - **Key:** iTDS — slow output-derived feedback signals that stabilize synaptic plasticity. Synapses receiving consistent top-down confirmation become more stable. Integration time constant ~100x the learning update rate.
   - **Relevance:** Direct model for readout-coupled consolidation (H2). The readout "votes" on which hidden features to stabilize.

### Representational Drift

5. **Driscoll et al. (2022)** — "Representational drift: Emerging theories for continual learning and experimental future directions"
   - Current Opinion in Neurobiology
   - https://www.sciencedirect.com/science/article/pii/S0959438822001039
   - **Key:** Drift reflects mixture of homeostatic turnover + learning-related plasticity. Not all dimensions drift equally — low-dimensional stable subspaces encode task-relevant variables while higher dimensions drift freely. Drift can improve robustness by exploring equivalent representations.
   - **Relevance:** Theoretical framework for why our hidden layer drifts and why it might not be entirely bad.

6. **arXiv:2512.22045 (2025)** — "Learning continually with representational drift"
   - https://arxiv.org/abs/2512.22045
   - **Key:** Drift improves memory maintenance during new learning by increasing robustness to weight perturbations.
   - **Relevance:** Suggests we should manage drift rather than eliminate it.

7. **PMC (2025)** — "Stability through plasticity: Finding robust memories through representational drift"
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC12625983/
   - **Key:** Sparse representations are more robust to weight perturbations. Drift itself can help find robust solutions.
   - **Relevance:** Supports k-WTA sparsity as a stability mechanism.

### Consolidation & Continual Learning

8. **Kirkpatrick et al. (2017)** — "Overcoming catastrophic forgetting in neural networks" (EWC)
   - PNAS 114(13):3521-3526
   - https://www.pnas.org/doi/10.1073/pnas.1611835114
   - **Key:** Elastic Weight Consolidation. Fisher information matrix diagonal measures per-weight importance. Penalty: `L += (lambda/2) * F_ii * (theta_i - theta*_i)^2`. Lambda typically 400-5000.
   - **Relevance:** Conceptual basis for importance-weighted consolidation, though Fisher computation isn't biologically plausible.

9. **AGMP (2025)** — "Astrocyte-gated multi-timescale plasticity for online continual learning in deep spiking neural networks"
   - Frontiers in Neuroscience
   - https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1768235/full
   - **Key:** Biologically plausible alternative to EWC. Slow astrocytic variable integrates local activity. When high (stable), plasticity suppressed. When it drops (novelty), plasticity reopens. Effectively a four-factor rule: `dw = eligibility * modulation * (1 - stability_gate)`. Astrocytic tau ~10x eligibility tau.
   - **Relevance:** Direct implementation model for per-synapse stability gating in MORPHON.

### Multi-Timescale RL

10. **Momennejad et al. (2025)** — "Multi-timescale reinforcement learning in the brain"
    - Nature
    - https://www.nature.com/articles/s41586-025-08929-9
    - **Key:** Dopamine neurons encode RPEs across a distribution of discount time constants (gamma 0.8 to 0.99). Short-timescale for immediate credit, long-timescale for slow structure. Distributional multi-timescale RL outperforms single-timescale.
    - **Relevance:** Supports multiple eligibility traces at different tau values rather than a single trace.

11. **Geerts et al. (2020)** — "A complementary learning systems approach to temporal difference learning"
    - Neural Networks 122:218-230
    - https://pmc.ncbi.nlm.nih.gov/articles/PMC6964152/
    - **Key:** Fast hippocampal (lr ~10x) + slow neocortical learning. Replay ratio: 4-8 replays per new experience.
    - **Relevance:** Validates our episodic replay architecture.

### Anchor Neurons

12. **Open Mind / MIT Press (2025)** — "Semantic Anchors Facilitate Task Encoding in Continual Learning"
    - https://direct.mit.edu/opmi/article/doi/10.1162/OPMI.a.28/133355/
    - **Key:** Fixed reference points (anchor embeddings) reduce catastrophic forgetting. Tasks encoded relative to anchors show less interference.
    - **Relevance:** Theoretical support for designating stable "anchor morphons."

### Heterogeneous Time Constants

13. **arXiv:2506.07341 (2025)** — "Slow and Fast Neurons Cooperate in Contextual Working Memory through Timescale Diversity"
    - https://arxiv.org/abs/2506.07341
    - **Key:** Slow neurons (tau 50-200ms) maintain state; fast neurons (tau 2-10ms) encode transients. Cooperation between the two is essential.
    - **Relevance:** Supports per-morphon tau_membrane with bimodal distribution.

### Reservoir / Readout

14. **RECAP (2026)** — "Local Hebbian Prototype Learning as a Self-Organizing Readout for Reservoir Dynamics"
    - https://arxiv.org/html/2603.06639
    - **Key:** Self-organizing readout for reservoir computing using local Hebbian rules rather than supervised training.
    - **Relevance:** Alternative readout architecture that might be more compatible with drifting representations.

15. **Nicola & Clopath (2018)** — "Learning recurrent dynamics in spiking networks"
    - eLife 7:e37124
    - https://elifesciences.org/articles/37124
    - **Key:** Readout-derived error shapes recurrent weights via feedback alignment in reservoir computing.
    - **Relevance:** Validates our DFA approach for reservoir-like networks.

### PV+ Interneurons

16. **Frontiers (2022)** — "Parvalbumin-Positive Interneurons Regulate Cortical Sensory Plasticity in Adulthood and Development Through Shared Mechanisms"
    - https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2022.886629/full
    - **Key:** PV+ interneurons sharpen STDP timing window and restrict plasticity via perisomatic feedback inhibition. Maturation of PV+ circuits coincides with critical period closure.
    - **Relevance:** Biological basis for "Anchor" morphons having narrower STDP windows and inhibitory stabilization.

### Synaptic Consolidation Cascades

17. **Benna & Fusi (2016)** — "Computational principles of synaptic memory consolidation"
    - Nature Neuroscience
    - https://www.nature.com/articles/nn.4401
    - http://www.gatsby.ucl.ac.uk/~pel/tnlectures/papers/benna_fusi.pdf
    - **Key:** Each synapse has m internal variables (u_1...u_m) in a linear chain with geometrically increasing timescales (~100x between levels). New memories written to u_1 (fast), progressively diffuse to u_m (slow). Memory capacity scales O(N) vs O(sqrt(N)) for naive models. m=3 levels sufficient.
    - **Relevance:** Extension of tag-and-capture to multi-level weight cascade. Would replace single `weight` with `w_fast/w_medium/w_slow`.

18. **Comms Biology (2021)** — "Memory consolidation and improvement by synaptic tagging and capture in recurrent neural networks"
    - https://www.nature.com/articles/s42003-021-01778-y
    - **Key:** STC in recurrent networks. Tag set by weak activity, capture requires strong neuromodulation. Only rewarded plasticity is consolidated.
    - **Relevance:** Validates our existing tag-and-capture approach in recurrent context.

### Cross-Homeostatic Plasticity

19. **Soldado-Magraner et al. (2022)** — "Paradoxical self-sustained dynamics emerge from orchestrated excitatory and inhibitory homeostatic plasticity rules"
    - PNAS 119(32)
    - https://www.pnas.org/doi/10.1073/pnas.2200621119
    - Code: https://github.com/SMDynamicsLab/Paradoxical2022
    - **Key:** Standard per-neuron homeostasis can't produce inhibition-stabilized networks (paradoxical effect). Fix: **cross-homeostatic plasticity** — weights onto excitatory neurons use *inhibitory* population's error, and vice versa. Produces self-sustained dynamics with emergent soft-WTA.
    - **Relevance:** Could replace MORPHON's per-morphon homeostasis for better E/I balance. Directly addresses mode collapse.

### Fast Non-Hebbian Compensatory Mechanisms

20. **Zenke (2017)** — "Hebbian plasticity requires compensatory processes on multiple timescales"
    - Phil. Trans. R. Soc. B 372(1715)
    - https://royalsocietypublishing.org/rstb/article/372/1715/20160259/
    - Code: https://github.com/fzenke/pub2015orchestrated
    - **Key:** Four plasticity mechanisms must co-exist: (1) triplet STDP, (2) transmitter-induced potentiation (prevents silence at low rates), (3) heterosynaptic depression (prevents runaway at high rates), (4) slow consolidation. Critically, (2) and (3) must operate on the SAME timescale as STDP — not slower.
    - **Relevance:** MORPHON has (1) and (4) but lacks the fast non-Hebbian compensatory mechanisms (2) and (3). These are the "missing immune system" preventing stable assembly formation.

### E-prop (Eligibility Propagation)

21. **Bellec et al. (2020)** — "A solution to the learning dilemma for recurrent networks of spiking neurons"
    - Nature Communications
    - https://www.nature.com/articles/s41467-020-17236-y
    - **Key:** e-prop derives eligibility traces for recurrent SNNs from BPTT, bridging biology and machine learning. Three-factor rule with eligibility traces that propagate through recurrent connections.
    - **Relevance:** Theoretical validation that three-factor + eligibility traces CAN solve RL tasks in recurrent SNNs when properly tuned.
