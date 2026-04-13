# Piagetian Milestones — Developmental Evaluation Framework for Morphon

**Version:** 1.0 — April 2026  
**Status:** Specification (partially implementable today)  
**Author:** Lisa Welsch

---

## Motivation

Morphon is currently evaluated on task performance: CartPole avg score, MNIST accuracy. These are useful but narrow — they measure whether the system can solve a specific problem, not whether it is developing genuine intelligence. A trained MLP can score 99% on MNIST without any of the properties we actually care about.

The BSB (Behavioral Stages of Belief) developmental track — general awareness → object permanence → prediction → self/other distinction → theory of mind — provides a richer benchmark ladder grounded in cognitive science. These milestones characterize what kind of understanding the system has acquired, not just what task it can execute.

This spec defines each milestone as a concrete, runnable test protocol with measurable success criteria. The first four milestones (M0–M3) are testable today with the current Morphon architecture. M4–M5 require modest new infrastructure. M6–M7 require multi-agent capability.

---

## Milestone Hierarchy

```
M0  Sensorimotor Response        — responds differently to different inputs
M1  Habituation / Dishabituation  — notices novelty; habituates to repetition
M2  Object Permanence            — maintains representation during occlusion
M3  Temporal Prediction          — anticipates what comes next in a sequence
M4  Causal Attribution           — distinguishes self-caused from external events
M5  Self/Other Distinction       — models own behavior vs. observed behavior
M6  Imitation                    — reproduces another agent's behavior pattern
M7  Theory of Mind               — models another agent's internal belief state
```

M0–M3 map roughly to Piaget's **Sensorimotor Stage** (0–2yr). M4–M5 map to early **Preoperational**. M6–M7 map to **Concrete Operational** and beyond.

---

## M0 — Sensorimotor Response

**Biological analog:** Orienting reflex. The infant turns toward a stimulus; different stimuli evoke different responses.

**What it tests:** Does the system produce discriminably different outputs for different inputs? This is the absolute baseline — if the system cannot do this, nothing above is meaningful.

**Protocol:**
1. Present 10 distinct input patterns (e.g., unit vectors, sine waves of different frequencies).
2. For each pattern, run N=10 steps and read the output vector.
3. Compute pairwise cosine distance between all output vectors.

**Success criterion:** Mean pairwise output distance > 0.2. All 10 patterns produce distinct outputs (no two outputs within 0.05 cosine distance of each other).

**Current status:** Satisfied implicitly by CartPole and MNIST. No dedicated test exists.

**Implementation note:** Trivial to add to `examples/milestones.rs`. No new Morphon capabilities needed.

---

## M1 — Habituation / Dishabituation

**Biological analog:** Habituation. Infants stop orienting to a repeated stimulus (it becomes predictable); a novel stimulus reactivates full orienting response. Tapped by the "looking time" paradigm.

**What it tests:** Does the system have a prediction-error signal that decreases with familiarity and spikes on novelty? In Morphon terms: does `prediction_error_mean` habituate, and does it dishabituate to a new pattern?

**Protocol:**
1. **Habituation phase:** Present pattern X repeatedly for 200 steps. Record `prediction_error_mean` (PE) via `endo_json()` every 10 steps.
2. **Test (same):** Present pattern X for 20 more steps. Record mean PE.
3. **Test (novel):** Present pattern Y (never seen) for 20 steps. Record mean PE.
4. Compute `dishabituation_ratio = PE_novel / PE_habituated`.

**Success criterion:** 
- `PE_habituated < 0.7 × PE_initial` (PE drops ≥30% over habituation)
- `dishabituation_ratio > 1.5` (novel pattern causes ≥50% higher PE than habituated baseline)

**Current status (v4.9.0, seed=42):** PASS — 96.6% PE drop (criterion ≥30%), 12.49× dishabituation ratio (criterion ≥1.5×). Both margins are large — the PE mechanism is working as intended. `prediction_error_mean` is accessed via `system.inspect().avg_prediction_error`.

**Implementation note:** Pure evaluation — runs `system.step()` in a loop and reads `endo_json()`. No new architecture needed. Could add `system.inject_novelty()` as a separate novelty injection to see if the system's own PE matches hand-labeled novelty.

---

## M2 — Object Permanence

**Biological analog:** Infants under ~8 months show no evidence of seeking a hidden object (A-not-B error). After ~8–12 months, they search for objects they saw hidden. The key: representation survives occlusion.

**What it tests:** Can the system maintain a representation of an input pattern in working memory (persistent activity) when the input is removed? The Morphon working memory (`memory.rs` persistent activity) and the slow-decay resonance system should support this.

**Protocol:**
1. **Encoding phase:** Present pattern X for `encoding_steps=30` steps (sufficient to engage working memory).
2. **Occlusion phase:** Feed zero input for up to 100 steps. At each step, classify the output (argmax).
3. **Retrieval test:** At the end of occlusion, inject a "probe" (half-strength pattern X) and classify.
4. Measure: `persist_steps` = length of the **consecutive** correct run starting from step 0.
   - Note: `last_correct_step` (latest step where classification was ever correct) is a distinct and misleading metric — a system that briefly classifies correctly at step 95 but failed at steps 1–94 has `persist_steps=0`, not 95. The spec uses consecutive from step 0.

**Success criterion (graded, majority vote: ≥50% of classes must achieve):**
- Bronze: `persist_steps` ≥ 10 (basic working memory)
- Silver: `persist_steps` ≥ 30 (robust short-term retention)
- Gold: `persist_steps` ≥ 100 AND probe recovery (consolidation-level persistence)

**Success distinguishes:** The system has truly formed a working-memory trace, not just output inertia from the last step. To verify: also test with pattern Y (different class) in occlusion to confirm the representation is stimulus-specific, not just "system hasn't fully reset."

**Current status (v4.9.0, seed=42, pre-training):** None. Best single class: 2 consecutive steps. Pre-training, the readout has no class-specific representations to maintain, so there is nothing to persist. `last_correct_step` can give spuriously high values (up to 100) from noise — see metric note above. Next measurement: post-MNIST epoch 1. The architecture (persistent activity in `memory.rs`, eligibility trace carry-over) should support at least Bronze once the readout has learned class-specific targets.

**Implementation note:** The test must be run on a freshly-reset system for each pattern to prevent interference. Use `system.reset_voltages()` between trials. The test is meaningful after the readout has been trained (post-MNIST epoch 1). A `--milestones` flag in `examples/mnist_v2.rs` could run this automatically after training.

---

## M3 — Temporal Prediction

**Biological analog:** Anticipatory eye movements. An infant who has seen a ball roll behind a screen will look to where it should reappear before it does so. Prediction is the basis of sensorimotor control.

**What it tests:** Can the system learn a temporal sequence A→B→C and activate B-associated outputs when only A is shown? In Morphon terms: do eligibility traces and tag-and-capture encode temporal relationships, not just co-activation?

**Protocol:**
1. **Training phase:** Present the sequence [A, B, C] repeatedly for 50 cycles. Each element shown for 5 steps. Deliver reward after C (reward signals sequence completion, not individual elements).
2. **Test — full sequence:** Present [A, B, C] with no reward. Measure output accuracy on each element.
3. **Test — prediction:** Present only A. After 5 steps (during the gap where B would appear), read the output. Measure: is the most active output class B?
4. **Test — completion:** Present [A, B]. At the gap where C would appear, read the output. Is it C?

**Success criterion:**
- Prediction accuracy on B given only A: > 50% (chance = 10% for 10 classes)
- Completion accuracy on C given [A, B]: > 70%

**Current status:** The temporal eligibility trace system should support this. The `temporal-sequence-processing-spec_v2.md` already specifies this capability. No dedicated evaluation example exists.

**Implementation note:** Closely related to `examples/temporal_sequences.rs` (not yet built per roadmap). This milestone gives a concrete evaluation target for that work. The sequence [A, B, C] should be cleanly separated inputs (orthogonal patterns), not overlapping, to avoid trivial co-activation being mistaken for temporal prediction.

---

## M4 — Causal Attribution

**Biological analog:** The infant discovers that some events in the world are caused by its own actions (moving a hand causes a mobile to shake) and some are not (the mobile also shakes when the wind blows). This is the beginning of agency.

**What it tests:** In a closed-loop environment, does the system develop a lower prediction error for state transitions that follow its own actions vs. external perturbations? The PE signal should be systematically lower for self-caused transitions the system has learned.

**Protocol:**
- Environment: CartPole variant where on 20% of steps an external perturbation is applied (random pole angle impulse, not linked to the system's action).
- After training to solve CartPole, run inference-only episodes.
- For each step, record: (a) the system's action, (b) whether a perturbation was applied, (c) the current `prediction_error_mean`.
- Compute `PE_self` (steps without perturbation) vs. `PE_external` (steps with perturbation).

**Success criterion:** `PE_external / PE_self > 1.3` — external perturbations produce ≥30% higher prediction error than self-caused transitions. The system has learned the causal structure of its own actions.

**Current status:** Requires extending `examples/cartpole.rs` with an external perturbation injection. The PE infrastructure already exists. CartPole must be solved first (already done).

---

## M5 — Self/Other Distinction

**Biological analog:** Mirror neuron research. The infant eventually distinguishes between its own sensations and observed sensations of others. Self-model emergence.

**What it tests:** If the system observes a sequence of inputs that correspond to a *different* agent's behavior, does it respond with higher uncertainty (PE, arousal) than when producing that same behavior itself? This is a crude proxy for self-model vs. other-model.

**Protocol:**
1. Train agent A on CartPole until solved.
2. Run agent A to collect a behavioral trajectory (state, action) pairs.
3. Feed the state sequence from agent A's trajectory as input to a *freshly initialized* agent B, without agent B taking any actions.
4. Compare: does agent B's PE during "observation of A" exceed its own PE during its own training trajectory?

**Success criterion:** Mean PE during observation > mean PE during self-training at matched competence level. The system doesn't just have a lower-PE representation for self-generated input — it actively builds it.

**Current status:** Requires multi-system infrastructure (agent A produces trajectories, agent B observes them). The Morphon Python bindings could drive this in a notebook. Not implementable in a single example today.

---

## M6 — Imitation

**Biological analog:** Neonatal imitation (tongue protrusion, mouth opening). Later: deferred imitation of novel behaviors. Prerequisite for social learning.

**What it tests:** Given observations of another agent's behavior, can the system reproduce it without direct reward for the reproduction?

**Protocol:**
1. Train agent A on a task (e.g., a 2D navigation maze with a unique path solution).
2. Record agent A's state→action trajectory.
3. Provide the state sequence to agent B as input (no action labels, just states).
4. Test: does agent B, when placed in the same task, reproduce the same path?

**Success criterion:** Path overlap between agent B and agent A's solution > 70%.

**Current status:** Requires: (1) an imitation-compatible task (maze), (2) multi-agent infrastructure. Future work — dependent on M5. Relates to Social Learning Vision (`project_social_learning.md` in memory).

---

## M7 — Theory of Mind

**Biological analog:** False-belief task (Sally-Anne). The child understands that another agent has a belief about the world that may differ from reality and from the child's own belief.

**What it tests:** Can the system maintain a model of another agent's internal state (what the other agent "believes" about the world), separate from its own state?

**Protocol (false-belief variant):**
1. Agent A is trained to expect object X at location L1.
2. Unbeknownst to A (the observer), X is moved to L2.
3. The test agent (B) has observed both the original placement and the move.
4. B must predict where A will look: L1 (A's outdated belief) not L2 (where X actually is).

**Success criterion:** B predicts A's behavior correctly (L1 search) with accuracy > 70%, despite B itself knowing X is at L2.

**Current status:** Requires: (1) multi-agent MI system, (2) agent-B having separate internal models for "what I know" and "what A knows", (3) a task with hidden-state structure. Significant new architecture. This is the long-term research goal.

---

## Implementation Roadmap

| Milestone | Testable Today? | Effort | Blocks |
|-----------|----------------|--------|--------|
| M0 — Sensorimotor | ✅ Yes | 1h — add to milestones.rs | Nothing |
| M1 — Habituation | ✅ Yes | 2h — loop + endo_json() | Nothing |
| M2 — Object Permanence | ✅ Yes | 4h — post-training probe | Trained readout |
| M3 — Prediction | ⚠️ Needs example | 1 day — temporal sequence example | temporal_sequences.rs |
| M4 — Causal Attribution | ⚠️ Needs extension | 4h — CartPole + perturbation | CartPole solved ✅ |
| M5 — Self/Other | ❌ Needs multi-system | 2 days | M4 + Python bindings |
| M6 — Imitation | ❌ Needs task + multi-agent | 1 week | M5 + new task |
| M7 — Theory of Mind | ❌ Architecture open question | Research | M6 + belief representation |

**Recommended immediate work:**
1. `examples/milestones.rs` — implements M0, M1, M2 as a single runnable benchmark
2. Profile: quick (M0+M1 only, ~30s), standard (M0–M2, ~5min), extended (M0–M3 if temporal sequences exist)
3. Results saved to `docs/benchmark_results/v{version}/milestones_latest.json`

---

## Relationship to Existing Architecture

| Milestone | Morphon mechanism tested |
|-----------|--------------------------|
| M0 | Sensory→Associative→Motor pathway, readout discriminability |
| M1 | `prediction_error_mean` in VitalSigns, novelty channel, PE EMA in AllostasisPredictor |
| M2 | Persistent activity in `memory.rs`, eligibility trace decay, resonance with delays |
| M3 | Temporal STDP via eligibility traces, tag-and-capture for delayed reward |
| M4 | PE signal under self-generated vs. external state transitions, Endo Stressed detection |
| M5 | Requires explicit self-model separate from other-model — not yet architected |
| M6 | Requires imitation-specific reward shaping — not yet architected |
| M7 | Requires second-order belief representation — open research question |

---

## Notes on M2 — the Most Immediately Interesting Test

Object permanence deserves special attention because it directly tests one of Morphon's claimed advantages: that the developmental, recurrent spiking architecture can maintain working memory in a way that feedforward networks cannot.

A standard MLP reset between images has *zero* object permanence — the representation evaporates the instant the input is removed. Morphon's persistent activity (slow-decay spiking, eligibility trace carry-over) should give it non-zero persistence.

The question is whether it's *principled* persistence (the specific representation stays alive) or just *noise* (random residual activity). The test distinguishes these: noise gives chance-level classification during occlusion; principled persistence gives above-chance classification specifically for the encoded pattern.

If Morphon achieves Silver (30-step persistence) on M2, that is a genuinely novel result worth reporting — no standard neural network architecture achieves this without explicit memory cells.
