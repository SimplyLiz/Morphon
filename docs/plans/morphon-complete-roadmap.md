# MORPHON — Complete Development Roadmap
## From Validated Prototype to Morphogenetic Intelligence Platform
### TasteHub GmbH — April 2026

---

## Current State (What Exists Today)

### Validated & Working
- **morphon-core** Rust crate: 6,000+ lines, 154 tests, 24+ commits
- **CartPole-v1: SOLVED** (avg=195.2, episode 895, best=468)
- **Endoquilibrium V1**: 550 lines, 7 vitals, 6 rules, 5 developmental stages, dynamic stage transitions validated
- **Episode-gated tag-and-capture**: selective consolidation, weight entropy maintained
- **Supervised analog readout**: cerebellar Purkinje pattern (unsupervised MI features + supervised task readout)
- **MNIST**: 26% post-recovery (self-healing demonstrated: system better after damage+recovery than intact)
- **Five failure modes diagnosed and fixed**: FR deadlock, directionless credit, TD-concentrated learning, readout bias, entropy collapse
- **Encoding discovery**: sparse zero-bias encoding + centered sigmoid + learnable bias
- **Activity stabilization**: Jaccard=0.97 with reduced associative noise
- **CMA-ES meta-optimization**: validated teach_hidden dominance, TD critic for RL

### Specified & Ready to Build
- Endoquilibrium V2 spec (local inhibition + iSTDP + astrocytic gate + Phase A/B levers)
- Pulse Kernel Lite v2.0 spec (4 hot arrays, 6–8 hours)
- Pulse Kernel v1.0 spec (full SoA + CSR — deferred, kept as reference)
- LocalParams spec (per-morphon meta-plasticity — merged into Endo as Phase 2)
- DeMorphon spec (multicellular transition — complete theoretical + implementation spec)
- MorphonGenome spec (heritable blueprint — PRD ready)
- Axonal Properties plan (myelination, distance-dependent cost)
- ANCS-Core plan (7 phases, memory backend + reconsolidation + importance scoring)
- FFG Kleinprojekt application v2 (ready for eCall submission)

---

## Phase 0: Publication & Funding (NOW — April 2026)

**Duration: 1–2 weeks. Zero code changes.**

| Deliverable | Status | Action |
|---|---|---|
| arXiv preprint | Content complete, needs writing | Write the paper: architecture, failure-mode catalogue, Endoquilibrium analysis, CartPole SOLVED, MNIST self-healing, encoding discovery |
| FFG Kleinprojekt | Application v2 ready | Fill Martyna's name, align costs with KLF 3.2, attach Saldenliste + CVs, submit via eCall |
| MNIST diagnostics | Partially done | Sensory-only baseline, damage+recovery sweep (5 seeds), confusion matrix, per-class accuracy — for paper figures |

**Gate:** Paper on arXiv. FFG submitted. These create the timestamp and unlock funding.

**Why this is Phase 0:** Everything after this builds on the credibility the paper and funding provide. Without the paper, the work isn't citable. Without the FFG, the next 12 months of development isn't funded.

---

## Phase 1: Local Inhibitory Competition (May–June 2026)

**Duration: 3–4 weeks. The most important architectural change in the roadmap.**

**Goal:** Replace global k-WTA with biologically correct local inhibition using iSTDP (Vogels et al. 2011). This is the prerequisite for every subsequent phase.

| Step | What | Effort |
|---|---|---|
| 1.1 | Extract hardcoded winner_boost to HomeostasisParams | 1 hour |
| 1.2 | CompetitionMode enum (GlobalKWTA / LocalInhibition) for A/B testing | 2 hours |
| 1.3 | iSTDP rule in learning.rs for inhibitory synapses | 3 hours |
| 1.4 | create_local_inhibitory_interneurons() — intra-cluster wiring | 4 hours |
| 1.5 | Activity-dependent threshold adaptation (replaces winner-list boost) | 2 hours |
| 1.6 | LocalInhibition branch in step() — skip global sort | 3 hours |
| 1.7 | Validation metrics: population sparsity, lifetime sparsity, winner diversity entropy | 3 hours |
| 1.8 | A/B benchmark: CartPole, 10 seeds per mode | Compute |
| 1.9 | A/B benchmark: MNIST, 10 seeds per mode | Compute |
| 1.10 | If validated: delete GlobalKWTA code path | 1 hour |

**Gate:** CartPole not regressed (avg ≥ 195). MNIST winner diversity entropy significantly improved.

**Expected impact:** MNIST accuracy from 26% to 40–60% (diverse representations → readout has features to learn from). CartPole unchanged or slightly improved.

**Paper update:** Add competition comparison results to arXiv v2.

---

## Phase 2: Endoquilibrium V2 — Wider Regulatory Surface (July–August 2026)

**Duration: 4–5 weeks. Three sub-phases.**

### Phase 2A: High-Value Levers (2 weeks)

| Lever | What it regulates | Effort |
|---|---|---|
| winner_adaptation_mult | How fast winners rotate (stage-dependent) | 3 hours |
| capture_threshold_mult | Consolidation sensitivity (Rule 5 extension) | 2 hours |
| rollback_pe_threshold_mult | Checkpoint sensitivity (stage-dependent) | 2 hours |
| consolidation_gain (reward-based stage detection) | Already implemented — validate on extended runs | 1 hour |

**Gate:** CartPole solves with dynamic stage regulation. All Endo V1 tests pass.

### Phase 2A+: Astrocytic Gating (1 week)

| What | Implementation | Effort |
|---|---|---|
| Per-morphon `astrocytic_state` field | 1 new f64 on Morphon | 30 min |
| Slow EMA update rule | AGMP-inspired: activity integration over τ=500–1000 ticks | 1 hour |
| Sigmoid gate computation | g_i = sigmoid(astrocytic_state - threshold) | 30 min |
| Integration into weight update | Δw = eligibility × modulation × gate_i × plasticity_mult | 1 hour |
| CMA-ES search over τ_a, threshold_a, η coefficients | 3 new searchable params | Compute |

**Gate:** Continual learning improves (sequential digit classes without catastrophic forgetting). Consolidated morphons are measurably protected from spurious updates.

**Paper value:** High — directly comparable to AGMP (Dong & He, 2025). Morphon extends AGMP with dual-timescale EMAs and systemic Endo regulation.

### Phase 2B: Structural Plasticity Regulation (2 weeks)

| Lever | Risk | Depends on |
|---|---|---|
| division_threshold_mult | Medium — can't un-divide | Phase 2A (rollback_pe needs to work first) |
| pruning_threshold_mult | Medium | Phase 2A |
| frustration_sensitivity_mult | Low | Phase 2A |

**Gate:** Morphogenesis is demonstrably stage-appropriate. Proliferating networks grow faster. Mature networks resist structural change.

---

## Phase 3: MNIST Breakthrough + Paper v2 (August–September 2026)

**Duration: 2–3 weeks. Exploit Phase 1+2 for benchmark results.**

| Task | What | Expected result |
|---|---|---|
| MNIST with local inhibition + astrocytic gate | Full 10-class, 10K images, 5 epochs | >50% accuracy (target: 70%) |
| Sensory-only baseline | Logistic regression on raw pixels | ~85% (ceiling for linear readout) |
| Confusion matrix | Per-class accuracy + misclassification patterns | Paper figure |
| DVS-Gesture or SHD | Neuromorphic benchmark (temporal/event-based) | First temporal benchmark for MORPHON |
| Self-healing sweep | 5 seeds, 30% damage + recovery | Error bars for paper |
| Update arXiv paper | Add Phase 1–3 results | arXiv v2 with competition comparison + improved MNIST |
| Conference submission | ICONS 2027 or NeurIPS Workshop on Neuromorphic Computing | Deadline-dependent |

**Gate:** MNIST >50%. At least one additional benchmark beyond CartPole + MNIST.

---

## Phase 4: Python SDK + Developer Experience (September–November 2026)

**Duration: 6–8 weeks. The product layer.**

| Deliverable | What | Effort |
|---|---|---|
| PyO3 bindings | morphon.System, morphon.Reward, morphon.DevelopmentalProgram | 2 weeks |
| SDK API design | Pythonic interface, type hints, docstrings | 1 week |
| WASM compilation | Browser demos via wasm-pack | 1 week |
| Jupyter notebooks | 3 tutorials: CartPole, MNIST, custom task | 1 week |
| GitHub repository | CI/CD (GitHub Actions), README, CONTRIBUTING, LICENSE | 1 week |
| PyPI publication | morphon-core v0.1.0 on PyPI | 2 days |
| Documentation site | mkdocs or mdbook, hosted on GitHub Pages | 1 week |

**Gate:** `pip install morphon-core` works. 3 example notebooks run end-to-end. CI passes on every PR.

---

## Phase 5: Edge Deployment + Application PoC (November 2026 – January 2027)

**Duration: 8–10 weeks. Proving ground.**

| Deliverable | What | Effort |
|---|---|---|
| ARM cross-compilation | Raspberry Pi 4/5, NVIDIA Jetson Nano | 1 week |
| Dart/Flutter bindings | Mobile integration for TasteHub | 2 weeks |
| TasteHub PoC | On-device recipe personalization — learns from user preferences without cloud | 3 weeks |
| Latency benchmarks | Inference time on ARM vs. TFLite comparison | 1 week |
| Energy benchmarks | Power consumption comparison on Jetson | 1 week |

**Gate:** MORPHON runs on ARM with <50ms inference. TasteHub PoC shows measurable personalization over 100 interactions.

**This is the FFG Kleinprojekt M4 deliverable.**

---

## Phase 6: MorphonGenome + Foundation for DeMorphon (January–February 2027)

**Duration: 3 weeks. Pure refactor + new capability.**

| Week | What | Effort |
|---|---|---|
| Week 1 | MorphonGenome struct, snapshot/express/mutate/distance operations | 5 hours |
| Week 1 | Refactor divide_morphon() to use genome pipeline | 3 hours |
| Week 1 | Founder genome on System, genetic drift diagnostic | 2 hours |
| Week 2 | Validate: CartPole + MNIST identical results with genome-based division | 1 day |
| Week 2 | DeMorphonGenome struct, composite expression, body plan inheritance | 4 hours |
| Week 2 | EpigeneticOverride for role specialization | 3 hours |
| Week 3 | crossover() for two-parent reproduction | 2 hours |
| Week 3 | Genome serialization + lineage tree visualization | 3 hours |

**Gate:** All existing tests pass. Division uses genome pipeline. Genetic drift measurable.

---

## Phase 7: DeMorphon — The Multicellular Transition (February–April 2027)

**Duration: 6–8 weeks. The most novel feature in the roadmap.**

| Sub-phase | What | Effort |
|---|---|---|
| 7.1 Formation | Three conditions (adhesion, tradeoff, division-of-labor), body plan assignment | 1 week |
| 7.2 Internal wiring | Input/Core/Memory/Output roles, fixed internal body plan | 1 week |
| 7.3 Competition as unit | Output cells compete on behalf of DeMorphon in local inhibition | 3 days |
| 7.4 Fission | Dissolution back to individual Morphons with snapshot restore | 3 days |
| 7.5 Reproduction | DeMorphon division with body plan inheritance + mutation | 1 week |
| 7.6 Metabolic integration | Shared energy pool, collective earning/spending | 3 days |
| 7.7 Validation: Temporal pattern detection | Synthetic task: detect "A then B" sequence | 1 week |
| 7.8 Validation: XOR | Synthetic task: compute XOR(A,B) | 3 days |
| 7.9 Validation: Working memory | Delayed match-to-sample task | 1 week |
| 7.10 MNIST with DeMorphons | Do DeMorphons improve classification? | 1 week |

**Gate:** At least one emergent computation (temporal, XOR, or working memory) that individual Morphons cannot perform. MNIST with DeMorphons ≥ MNIST without.

**Paper:** "DeMorphon: The Multicellular Transition in Morphogenetic Intelligence" — target ALIFE 2027 or GECCO 2027.

---

## Phase 8: ANCS-Core — Memory Substrate (April–June 2027)

**Duration: 6–8 weeks. The knowledge layer.**

| Sub-phase | What | Priority |
|---|---|---|
| 8.0 | Backend Trait + MemoryItem struct (wrap existing TripleMemory) | Foundation |
| 8.1 | VBC-lite tier classification (Verbatim/Structural/Semantic/Procedural) | High |
| 8.2 | AXION 6-factor importance scoring + F7 pressure modes | High |
| 8.3 | TruthKeeper reconsolidation loop (contested → reopen synapses) | High — most novel |
| 8.4 | Forward-importance synapse pruning | Medium |
| 8.5 | RRF fused retrieval across memory tiers | Medium |
| 8.6 | SOMNUS sleep/wake consolidation cycle | Lower — validate need first |

**Gate:** CartPole + MNIST not regressed. TruthKeeper reconsolidation measurably improves early-episode recovery.

---

## Phase 9: Pulse Kernel Lite — Performance Optimization (June–July 2027)

**Duration: 2 weeks. Only when profiling shows it's needed.**

| Step | What | Effort |
|---|---|---|
| 9.1 | Profile with cargo bench / perf — confirm fast path is >50% of wall-clock | 2 hours |
| 9.2 | HotArrays struct (voltage, threshold, fired, refractory) | 2 hours |
| 9.3 | fast_integrate(), fast_threshold_check(), fast_reset() | 4 hours |
| 9.4 | Sync protocol (hot arrays ↔ Morphon structs) | 3 hours |
| 9.5 | Rebuild hot arrays after structural changes | 2 hours |
| 9.6 | Validate: identical results, measure speedup | 2 hours |

**Gate:** ≥2x speedup on fast path. All benchmarks produce identical results.

**Note:** This might not be needed. If MORPHON at 2K morphons runs fast enough without PKL, skip it entirely. The spec is ready if needed.

---

## Phase 10: Axonal Properties — Signal Transmission (July–August 2027)

**Duration: 2 weeks. Refinement layer.**

| Feature | What | Effort |
|---|---|---|
| Activity-dependent myelination | Consolidated synapses get faster delivery (lower effective delay) | 3 hours |
| Distance-dependent metabolic cost | Long-range connections cost more energy → encourages locality | 2 hours |
| Validation | Does myelination improve self-healing speed? Does distance cost improve cluster formation? | 1 week |

**Gate:** Measurable improvement on at least one benchmark or self-healing metric.

---

## Phase 11: Multi-Instance + Inter-System Communication (Q3–Q4 2027)

**Duration: 8–12 weeks. The platform vision.**

| Deliverable | What |
|---|---|
| Genome-based knowledge transfer | Robot A exports DeMorphon genome → Robot B expresses it in its own network |
| AXION-compatible genome serialization | Compact wire format for inter-system transmission |
| Translational Hubs | MORPHON-to-MORPHON communication protocol |
| Knowledge Hypergraph | DashMap-based entity-relation store with bi-temporal queries |
| CRDT-based multi-instance sync | Eventually-consistent shared knowledge across instances |
| TruthKeeper source watchers | Detect when external data sources change → cascade invalidation |

**Gate:** Two MORPHON instances share a DeMorphon genome, both solve a task, and knowledge transfer measurably accelerates learning on the receiving instance.

---

## Phase 12: Hardware Acceleration (2028+)

**Duration: 12+ months. Separate FFG Basisprogramm (full, up to €3M) in cooperation with TU Wien or FH Technikum Wien.**

| Step | What | Timeline |
|---|---|---|
| 12.1 | FPGA prototype for Pulse Kernel only (spike propagation accelerator) | 6 months |
| 12.2 | Structural plasticity on FPGA (pre-allocated PE slots, alive bitfield) | 6 months |
| 12.3 | Endoquilibrium regulation on FPGA (reduction tree for vitals, broadcast for gains) | 3 months |
| 12.4 | Validation: MORPHON on FPGA matches software results | 3 months |
| 12.5 | ASIC feasibility study (only if FPGA validates + market demand) | 6 months |

**Unique selling point:** First neuromorphic accelerator with native support for structural plasticity (cell division, migration, apoptosis as hardware operations). No existing chip (Loihi, SpiNNaker, BrainScaleS) supports runtime topology changes.

---

## Version Summary

| Version | Phase | Key Feature | Target Date |
|---|---|---|---|
| **v2.1** (current) | — | CartPole SOLVED, Endoquilibrium V1, MNIST 26% | April 2026 |
| **v2.2** | Phase 0 | arXiv paper + FFG submitted | April 2026 |
| **v3.0** | Phase 1 | Local inhibitory competition (iSTDP) | June 2026 |
| **v3.1** | Phase 2A | Endo V2 Phase A levers | July 2026 |
| **v3.2** | Phase 2A+ | Astrocytic gating (per-morphon plasticity gate) | August 2026 |
| **v3.3** | Phase 2B | Structural plasticity regulation | August 2026 |
| **v4.0** | Phase 3 | MNIST >50%, paper v2, conference submission | September 2026 |
| **v5.0** | Phase 4 | Python SDK on PyPI | November 2026 |
| **v5.1** | Phase 5 | Edge deployment, TasteHub PoC | January 2027 |
| **v6.0** | Phase 6 | MorphonGenome | February 2027 |
| **v7.0** | Phase 7 | DeMorphon (multicellular transition) | April 2027 |
| **v8.0** | Phase 8 | ANCS-Core (memory substrate) | June 2027 |
| **v9.0** | Phase 9 | Pulse Kernel Lite (if needed) | July 2027 |
| **v10.0** | Phase 10 | Axonal properties | August 2027 |
| **v11.0** | Phase 11 | Multi-instance communication | Q4 2027 |
| **v12.0** | Phase 12 | Hardware acceleration (FPGA) | 2028+ |

---

## Papers Planned

| Paper | Venue | Content | Target |
|---|---|---|---|
| **MORPHON: Morphogenetic Intelligence for Adaptive Computing** | arXiv → ICONS 2027 / NeurIPS WS | Architecture, CartPole SOLVED, failure-mode catalogue, Endoquilibrium, MNIST self-healing | April 2026 (v1), September 2026 (v2 with local inhibition results) |
| **DeMorphon: The Multicellular Transition in Morphogenetic Intelligence** | ALIFE 2027 / GECCO 2027 | Unicellular→multicellular transition, emergent computation, evolutionary dynamics | April 2027 |
| **Endoquilibrium: Predictive Neuroendocrine Regulation for Self-Organizing Neural Systems** | Frontiers in Neuroscience / Neural Computation | Full Endo V1+V2 analysis, comparison with AGMP, four-level regulatory hierarchy | Late 2027 |
| **MORPHON-Chip: Neuromorphic Hardware with Native Structural Plasticity** | DATE / ISSCC (hardware venues) | FPGA results, comparison with Loihi/SpiNNaker | 2028+ |

---

## Funding Timeline

| Application | Amount | Status | Funds |
|---|---|---|---|
| **FFG Kleinprojekt** | €88.5K grant | Ready to submit | Phases 1–5 (12 months) |
| **FFG Basisprogramm** (full) | Up to €3M (grant + loan) | Plan after Kleinprojekt | Phases 6–11 + team expansion |
| **Horizon Europe** (EIC Pathfinder) | Up to €3M | After arXiv + conference publication | DeMorphon + multi-instance + hardware |
| **aws Seed Financing** | Up to €800K | After SDK + first customers | Commercial scaling |

---

## The Big Picture

```
2026 Q2: Paper + FFG → Credibility
2026 Q3: Local inhibition → MNIST breakthrough → Paper v2
2026 Q4: Python SDK → Developer adoption → Community
2027 Q1: Edge deployment → TasteHub PoC → Product validation
2027 Q2: DeMorphon → Novel contribution → Second paper
2027 Q3: ANCS-Core → Knowledge persistence → Enterprise readiness
2027 Q4: Multi-instance → Robot-to-robot learning → Platform vision
2028+:   Hardware acceleration → MORPHON-Chip → Industry disruption
```

The progression is deliberate: credibility (paper) → capability (benchmarks) → accessibility (SDK) → application (edge PoC) → novelty (DeMorphon) → infrastructure (ANCS) → scale (multi-instance) → hardware. Each phase builds on the previous. No phase is skippable without weakening the next.

---

*From 300 morphons solving CartPole to a hardware-accelerated platform for morphogenetic intelligence.*
*One phase at a time. Each validated before the next begins.*

*TasteHub GmbH, Wien, April 2026*
