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

- **Bootstrap grouping is geometric, not functional.** The 6 interneurons in the CartPole output
  come from `developmental.rs:242` — a bootstrap phase that groups Associative morphons by spatial
  proximity in the Poincaré ball and creates one interneuron per group. Coverage exists, but the
  groups are based on initial position, not on which morphons actually co-fire. Before iSTDP
  has tuned the weights, competition may be suppressing the wrong morphons — interneuron A might
  inhibit morphons that never compete with each other, while morphons that do compete land in
  different groups.
  This is a transient problem that iSTDP should fix over episodes, but it means the first 100–200
  episodes in LocalInhibition mode are worse than GlobalKWTA by construction. The standard profile
  (1000 eps) should show convergence if it's going to converge.

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

### 4. Bootstrap grouping vs cluster topology (Low risk, previously misjudged)
With `fusion: false`, `create_local_inhibitory_interneurons()` in `morphogenesis.rs` never
runs — but `developmental.rs:242` has its own bootstrap version that runs at init regardless
of fusion. It groups Associative morphons by spatial proximity and creates one interneuron
per group, wired bidirectionally. This is where the 6 interneurons visible in CartPole output
come from. Coverage is real.

The difference vs cluster-formation interneurons: bootstrap groups by evenly chunking sorted
positions; cluster-formation groups by actual co-firing patterns. Bootstrap coverage may be
geometrically correct but not functionally aligned with which morphons compete. Whether this
matters in practice needs to be observed — check `winners_per_cluster` across runs to see if
competition is resolving per-group or collapsing globally.

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
src/system.rs:525–650    — GlobalKWTA branch in fast-path competition (the entire match arm,
                           including local_radius sub-path and anti-Hebbian LTD injection)
src/system.rs:822–848    — GlobalKWTA branch in winner_boost application (the match arm that
                           iterates kwta_winners; keep the LocalInhibition arm)
src/system.rs:225        — pub(crate) kwta_winners: Vec<MorphonId> field on System
src/system.rs:362        — kwta_winners: Vec::new() in System::new()
src/homeostasis.rs       — GlobalKWTA { fraction, local_radius, local_k } variant of
                           CompetitionMode enum and the default_local_k() helper
                           (kwta_fraction lives inside this variant, not in HomeostasisParams)
```

`CompetitionMode` can either be removed entirely (flattening `LocalInhibition` fields into
`HomeostasisParams`) or kept as a single-variant enum. The simplest path: delete the enum,
move `interneuron_ratio`, `istdp_rate`, `initial_inh_weight`, `inhibition_radius`,
`target_rate` directly onto `HomeostasisParams` with sensible defaults.
