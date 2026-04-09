# Changelog

All notable changes to Morphon-Core are documented here.

The project follows a loose semantic-versioning convention: major versions
mark architectural shifts (V1 → V2 → V3), minor versions add subsystems, and
patches fix bugs. Benchmark numbers are not cross-version comparable unless
the version tag is the same.

## [3.0.0] — 2026-04-08 — **Paper release**

The first version released alongside the arXiv/Zenodo preprint
[*Morphogenic Intelligence: Runtime Neural Development Beyond Static Architectures*](docs/paper/paper/Morphogenic_Intelligence.pdf).

### Added
- Full LaTeX source for the paper in `docs/paper/paper/`, with 6 figures
  generated from the benchmark JSONs via `figures/generate.py` (matplotlib)
  and the WASM visualizer (Playwright)
- TikZ architecture diagram (`figures/architecture.tex`)
- Metabolic pressure primitives:
  `reward_energy_coefficient`, `superlinear_firing_factor`
- Reward-correlated energy in `reward_contrastive()` so fired morphons
  earn energy proportional to reward arrival
- Benchmark guide (`docs/BENCHMARKS.md`) and reproducibility JSONs
  (`docs/benchmark_results/v3.0.0/`)
- Per-episode step logging in CartPole results JSON for learning-curve figures
- MNIST V2 receptive-field dumper (top-K associative morphon S→A weights
  as 28×28 grids) for paper figure generation
- Zenodo upload prep (`.zenodo.json`, `docs/paper/ZENODO_UPLOAD.md`)

### Fixed
- **Endoquilibrium premature Mature bug** on dense-reward classification tasks.
  Root cause: raw reward EMA inflated to ~0.65 within 500 samples regardless
  of accuracy. Fix: Mature stage transition now requires
  `total_updates >= 2000` (monotonically-increasing counter, not the
  500-cap `reward_history.len()`). Consolidating gate moved to 500 to
  preserve the natural Differentiating↔Consolidating oscillation.
- `reward_contrastive()` now broadcasts into `modulation.reward`
  via `inject_reward()` so metabolic selection pressure and three-factor
  learning both see the signal
- `teach_supervised()` cold-start deadlock: continuous sigmoid pre-activity
  instead of binary firing gate
- `teach_supervised_with_input()` weight decay reduced from 0.01 to 0.001
  (was fighting the learning signal)
- Anti-Hebbian LTD signal on k-WTA losers (`post_trace += 0.3`) so
  non-winners specialize away from shared initial weight patterns instead
  of mode-collapsing

### Changed
- Version bump 2.4.0 → 3.0.0 for the paper release
- Paper renamed from `main.tex` to `Morphogenic_Intelligence.tex` for
  clearer download/share names

### Known issues
- CartPole v3.0.0 quick profile reaches avg=166 (not the 195.2 SOLVED number
  from v0.5.0). The SOLVED result requires the standard profile (1000
  episodes); reproducing on v3.0.0 is documented as a follow-up.
- MNIST intact accuracy is 30% / post-recovery 48% — well below
  Diehl & Cook (~95%) and SOTA supervised SNNs. The substrate works
  (self-healing proves features exist); the supervised pathway still loses
  information at every spike conversion. Closing this gap is the central
  engineering challenge for v4.

---

## [2.4.0] — 2026-04-06

### Added
- Metabolic pressure primitives (pre-release of what ships in 3.0.0)
- MNIST training-loop fixes: fired tracking, threshold decay, binary voting
- WASM visualizer: Sketch Arena interactive pattern recognition demo

### Fixed
- Multiple MNIST learning-pipeline bugs identified through session-level
  diagnostics

---

## [2.3.0] — 2026-04-01

### Added
- V3 governance: epistemic states, justification tracking, constitutional
  constraints outside the learning loop
- Phase 2 activation for V3 governance features

---

## [2.2.0] — 2026-03-20

### Added
- Endoquilibrium V2: local inhibitory competition with iSTDP (Vogels
  et al. 2011), astrocytic gating, per-morphon intrinsic noise
- Activity-dependent myelination + distance-dependent metabolic cost
- Anti-hub mechanisms: synaptogenesis cap, anti-hub scaling, uniform init

### Results
- MNIST V2 baseline ~26–31% on the standard profile

---

## [2.1.0] — 2026-03-08 — **CartPole recovery**

### Fixed
- v2.0.0 regression identified and recovered: the tag-accumulation rate
  drop (instant → 0.05) was the primary cause of CartPole dropping from
  195.2 to 162. Fixed by making `tag_accumulation_rate` configurable
  (default 0.3) and restoring the single-gate consolidation path.
- Added `consolidation_gain` channel to Endoquilibrium as the biologically
  correct way to gate capture (replacing the `endo_tag_factor` experiment
  which was biologically incorrect).
- Dynamic developmental-stage detection: replaced always-Stressed PE-based
  logic with reward-relative trend detection.

### Results
- CartPole v2.1.0: SOLVED at episode 918, avg=195.1 (standard profile)

---

## [2.0.0] — 2026-03-01

### Added
- V2 primitives: bioelectric field, target morphology, self-healing,
  frustration-driven exploration, dreaming engine (offline consolidation)
- Adaptive receptors, collective compute
- V3 governance Phase 1 scaffolding

### Regressions (fixed in 2.1.0)
- CartPole dropped from 195.2 to ~162 on the standard profile due to
  the gradual tag-accumulation change. Root cause identified and
  documented in `docs/paper/sources/v2.1.0-benchmark-findings.md`.

---

## [0.5.0] — 2026-02-10 — **CartPole SOLVED**

### Added
- Cerebellar dual-speed architecture (fast readout + slow MI hidden layer)
- Population coding for CartPole (32 Gaussian channels instead of 8 binary)
- Centered sigmoid + learnable bias + no L2 decay in the analog readout
  (the breakthrough — each fix individually necessary, together sufficient)
- Tag-and-capture consolidation with episode-gated single threshold
- TD-LTP critic infrastructure

### Results
- **CartPole-v1 SOLVED**: avg=195.2, episode 895, best single episode 468
  steps, 1000-episode standard profile

This is the version cited in the paper as the SOLVED result. All later
versions introduced regressions and fixes (documented in v2.1.0 and v3.0.0
entries above) without returning to the 195.2 average in a short-profile
run.

---

## Earlier versions

Earlier versions (v0.1–v0.4) were the initial developmental lifecycle,
eligibility traces, first morphogenesis operations, and Endoquilibrium V1.
See the git log for the full history.

---

## Versioning policy

- **Major** (x.0.0): architectural shift (V1→V2: bioelectric field etc.;
  V2→V3: governance + paper release)
- **Minor** (x.y.0): new subsystems or benchmark results
- **Patch** (x.y.z): bug fixes that don't change benchmark interpretation

Benchmark results are tracked per-version in
`docs/benchmark_results/v{version}/`. Numbers are not cross-version
comparable unless explicitly tested on the target version.
