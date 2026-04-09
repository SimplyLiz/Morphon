# Morphogenic Intelligence — arXiv Paper

LaTeX source for the arXiv preprint.

## Build

```bash
cd docs/paper/paper
make
make view   # opens Morphogenic_Intelligence.pdf
make clean  # removes build artifacts
```

Requires `pdflatex` and `bibtex` (any standard TeX distribution: MacTeX, TeX Live, MiKTeX).

## Structure

```
Morphogenic_Intelligence.tex   — top-level document
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

Draft. 14 pages. All headline numbers reproduce on v3.0.0.

### Done
- All 8 sections written (abstract, intro, related work, architecture, implementation, experiments, failure modes, discussion, conclusion)
- 28 bibliography entries (placeholder refs removed)
- 3 figures generated from JSON benchmark results:
  - `figures/self_healing.pdf` — MNIST 4-bar (random, intact, damaged, recovered)
  - `figures/plasticity_accuracy.pdf` — pm vs accuracy scatter
- v2.4.0 MNIST self-healing reproduced from JSON (31.0 → 52.5%)

### Still TODO
- **Author block + affiliation.** Done — Lisa Welsch & Martyna Kwiecień, no institutional affiliation.
- **CartPole figure.** Current `figures/cartpole_curve.pdf` is a fallback (no per-episode data in the JSONs). Either dump per-episode steps from a fresh CartPole run or remove the figure.
- **Architecture diagram.** No Morphon struct diagram or topology snapshot yet (would require either TikZ work or visualizer screenshots).
- **MNIST receptive field heatmaps.** Would need to expose hidden-morphon weight extraction in the example.
- **v3.0.0 MNIST verification run.** The 31% / 52.5% numbers came from v2.4.0; should verify they reproduce on v3.0.0 (~25 min quick profile).

## Generating figures

```bash
cd docs/paper/paper
python3 figures/generate.py    # reads JSON from docs/benchmark_results/v*/
make                            # rebuild PDF
```

Requires `python3` and `matplotlib`.

## Source materials

The paper draws from the following internal documents:

- `docs/paper/sources/cartpole-findings.md`
- `docs/paper/sources/metabolic-pruning-findings.md` (the new findings doc)
- `docs/paper/sources/v2.1.0-benchmark-findings.md`
- `docs/paper/sources/endoquilibrium-findings.md`
- `docs/paper/sources/readout-architecture.md`
- `docs/paper/sources/negative-results.md`
- `docs/official/arxiv-paper-outline.md` (the original outline)
