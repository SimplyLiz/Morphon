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
  - `figures/nlp_readiness.pdf` — 4 tiers × 2 readouts comparison
  - `figures/plasticity_accuracy.pdf` — pm vs accuracy scatter
- v3.0.0 NLP readiness verified: Tier 0/1/2/3 = 99/62/85/42
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

The paper draws from an internal corpus of experimental findings documents
(CartPole fix chain, Endoquilibrium regression and recovery, metabolic
pruning discovery, readout architecture fixes, negative results, etc.).
These are not shipped with Morphon-OSS; they exist in the authors' private
development repository. All empirical claims in the paper, however, are
reproducible from the code and benchmark JSONs in this repository:

```bash
cargo run --example cartpole --release
cargo run --example mnist_v2 --release
cargo run --example nlp_readiness --release
```

Results are saved to `docs/benchmark_results/v{version}/`.
