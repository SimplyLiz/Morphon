# Morphogenic Intelligence — arXiv Paper

LaTeX source for the arXiv preprint.

## Build

```bash
cd docs/paper/paper
make
make view   # opens main.pdf
make clean  # removes build artifacts
```

Requires `pdflatex` and `bibtex` (any standard TeX distribution: MacTeX, TeX Live, MiKTeX).

## Structure

```
main.tex          — top-level document
sections/
  00_abstract.tex
  01_introduction.tex
  02_related_work.tex
  03_architecture.tex
  04_implementation.tex
  05_experiments.tex
  06_failure_modes.tex   ← analytical core
  07_discussion.tex
  08_conclusion.tex
references.bib    — bibliography
figures/          — generated plots (TODO)
```

## Status

Draft. Reproducible numbers from the v3.0.0 codebase, but the paper still needs:

1. **Figures.** TODO list:
   - Topology growth: Poincaré embedding snapshots (CartPole, ticks 0/100/500/1000)
   - Receptive field heatmaps for selected MNIST hidden morphons
   - Plasticity oscillation timeline (pm trajectory across one MNIST epoch)
   - Self-healing curve (accuracy before / after damage / after recovery)
   - NLP readiness benchmark bar chart (4 tiers × 2 readouts)

2. **Polishing.**
   - Bibliography entries marked `(Placeholder reference)` need real citations
   - Author block + affiliation
   - Anonymization for double-blind submissions

3. **Verification.** Re-run all benchmarks on the v3.0.0 release tag and confirm
   the JSON results in `docs/benchmark_results/v3.0.0/` match the numbers in the
   paper.

## Source materials

The paper draws from the following internal documents:

- `docs/paper/sources/cartpole-findings.md`
- `docs/paper/sources/metabolic-pruning-findings.md` (the new findings doc)
- `docs/paper/sources/v2.1.0-benchmark-findings.md`
- `docs/paper/sources/endoquilibrium-findings.md`
- `docs/paper/sources/readout-architecture.md`
- `docs/paper/sources/negative-results.md`
- `docs/official/arxiv-paper-outline.md` (the original outline)
