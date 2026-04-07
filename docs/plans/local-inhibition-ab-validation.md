# Local Inhibitory Competition — A/B Validation Plan

**Status:** Blocked on benchmarks. GlobalKWTA is still the default.  
**Prerequisite for:** Deleting GlobalKWTA code path (spec Section 3, steps 8–10).

---

## What this is

`CompetitionMode::LocalInhibition` replaces the global k-WTA sort with biologically local
inhibitory interneurons. Instead of a central algorithm picking top-k associative morphons
each tick, inhibitory interneurons receive excitatory drive from cluster members and send
negative-weight spikes back, suppressing neighbors through normal spike propagation.
iSTDP (Vogels et al. 2011) self-tunes the inhibitory synapse weights to maintain a target
firing rate — meaning the "number of winners" is emergent rather than a fixed `kwta_fraction`
parameter.

The code for both modes is complete and coexists. The default is still `GlobalKWTA`.
The flag `--local-inhibition` switches `cartpole.rs` to `LocalInhibition` mode.

---

## What needs to happen before GlobalKWTA can be deleted

1. Run CartPole standard (1000 eps) with both modes, 10 seeds each.
2. If LocalInhibition matches or beats GlobalKWTA: delete GlobalKWTA, `kwta_fraction`,
   `kwta_winners`, and the global sort code in `system.rs:503–626`.
3. If there's a regression: tune iSTDP parameters, do not revert architecture.
4. Validate on MNIST standard (10 seeds) after CartPole passes.

---

## How to get the data

```bash
# GlobalKWTA baseline — 10 seeds
for i in $(seq 1 10); do
  cargo run --example cartpole --release -- --standard
done

# LocalInhibition — 10 seeds
for i in $(seq 1 10); do
  cargo run --example cartpole --release -- --standard --local-inhibition
done
```

Results land in `docs/benchmark_results/v3.0.0/`. Compare `avg_last_100` and `solved` across
seeds. Pass criterion: LocalInhibition mean `avg_last_100` within 10% of GlobalKWTA, no seed
catastrophically worse (< 50 steps avg).

For MNIST, use `mnist_v2.rs` (not `mnist.rs`):

```bash
cargo run --example mnist_v2 --release               # GlobalKWTA
cargo run --example mnist_v2 --release -- --local-inhibition   # LocalInhibition (needs flag added)
```

Note: `--local-inhibition` flag currently only exists on `cartpole.rs`. It needs to be added to
`mnist_v2.rs` before MNIST A/B is possible — same pattern as cartpole.

---

## What could go wrong

### 1. CartPole regression (High risk)
The quick-profile run already showed GlobalKWTA avg=151.9 vs LocalInhibition avg=111.2.
That's a real gap, not noise. Two causes:

- **iSTDP not tuned yet.** On a fresh system, inhibitory weights start at -0.3 and iSTDP needs
  time to find the right per-synapse value. GlobalKWTA gives correct sparsity from tick 1.
  LocalInhibition may produce wrong sparsity for hundreds of episodes before iSTDP settles.
  Fix: lower `istdp_rate` or pre-set `initial_inh_weight` closer to the eventual equilibrium.

- **CartPole uses `fusion: false`.** Clusters never form, so `create_local_inhibitory_interneurons()`
  never runs via the fusion path. The 6 `InhibitoryInterneuron` morphons visible in the output
  come from somewhere in the developmental bootstrap — probably `DevelopmentalConfig::cerebellar()`
  — but they are NOT wired as a full intra-cluster circuit. In LocalInhibition mode without
  clusters, competition depends entirely on whatever interneurons the bootstrap happens to create,
  which may not be enough or correctly positioned.
  Fix: verify where bootstrap interneurons come from and whether they're covering the associative
  layer properly. May need `fusion: true` (or at minimum a lightweight cluster-seeding step at
  init) to get proper intra-cluster coverage.

### 2. Emergent sparsity too high or too low (Medium risk)
GlobalKWTA fires exactly `ceil(n × 0.15)` associatives per tick. LocalInhibition sparsity is
determined by inhibitory synapse strength and iSTDP equilibrium. If iSTDP converges to a very
sparse representation (e.g. 1–2 winners instead of ~9), the network has much less capacity per
step. If too dense, the anti-Hebbian LTD signal that suppressed neurons used to get is absent —
the network may collapse to mode-sharing.
Observable: check `population_sparsity` in diagnostics. Target range: 0.3–0.7 (same as
GlobalKWTA baseline).

### 3. iSTDP + three-factor STDP interaction (Medium risk)
Our excitatory STDP is three-factor (eligibility × modulation × astrocytic gate), not vanilla
STDP. Vogels et al. (2011) proved iSTDP stability with vanilla STDP. The interaction with
three-factor is untested. In theory: iSTDP adjusts who fires (structural sparsity), three-factor
adjusts what is learned (weights). They operate on different synapses (inhibitory vs excitatory).
But if iSTDP forces a morphon to fire less than its target rate, the three-factor rule's
eligibility trace may starve — that morphon stops learning even if it should be adapting.
Observable: check `eligibility_density` — if it drops significantly vs GlobalKWTA baseline,
iSTDP is suppressing learning.

### 4. Bootstrap interneurons not covering the associative layer (High risk for CartPole)
As noted above, with `fusion: false` the main interneuron creation path never runs. This is
the most likely cause of the quick-profile regression and needs to be investigated before
drawing conclusions from 1000-episode runs. The quick fix is to check at system init whether
`LocalInhibition` mode is active and, if so, create a minimal set of interneurons seeded to
cover the associative layer — essentially a bootstrap analog of `create_local_inhibitory_interneurons`.

### 5. Winner rotation changes learning dynamics (Low risk)
In GlobalKWTA, `winner_boost` is applied only to k-WTA winners. In LocalInhibition, it applies
to any fired associative morphon scaled by `winner_adaptation_mult`. The effective boost per
morphon may be higher or lower depending on the firing rate. This changes how quickly feature
detectors stabilize. Probably fine given Endo is regulating `winner_adaptation_mult` by stage,
but worth checking if weight entropy behaves differently.

---

## Parameters to tune if LocalInhibition underperforms

| Parameter | Location | Current | Try |
|---|---|---|---|
| `istdp_rate` | `CompetitionMode::LocalInhibition` | 0.005 | 0.001–0.05 |
| `initial_inh_weight` | `CompetitionMode::LocalInhibition` | -0.3 | -0.1 to -0.6 |
| `interneuron_ratio` | `CompetitionMode::LocalInhibition` | 0.1 | 0.15–0.25 |
| `target_rate` | `CompetitionMode::LocalInhibition` | None (Endo-derived) | 0.10–0.15 explicit |

Do not touch `kwta_fraction` for comparison — that's the GlobalKWTA path.

---

## What to delete after successful validation

```
src/system.rs:503–626    — GlobalKWTA branch in step() fast path
src/system.rs:777–799    — GlobalKWTA branch in winner_boost application
src/system.rs:226        — kwta_winners field on System
src/system.rs:361        — kwta_winners: Vec::new() in System::new()
src/homeostasis.rs       — GlobalKWTA variant of CompetitionMode enum
                           (keep LocalInhibition, make it the only variant or the default)
HomeostasisParams        — remove kwta_fraction references
```

Keep `CompetitionMode` as an enum but remove the `GlobalKWTA` variant, or flatten to a struct
if there's no other mode planned.
