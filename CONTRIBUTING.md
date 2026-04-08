# Contributing to Morphon-Core

Thanks for your interest. This is a research preprint codebase, not
production software — please read this first so expectations are aligned.

## What this repository is

Morphon-Core is the reference implementation for the paper
[*Morphogenic Intelligence: Runtime Neural Development Beyond Static Architectures*](docs/paper/Morphogenic_Intelligence.pdf).
The goal is to make every experiment in the paper reproducible from a clean
checkout and to document the architectural decisions that produced the
published numbers.

It is actively developed by a small team and moves fast. APIs are not stable,
benchmarks are not cross-version comparable unless the version tag is the same,
and there may be known regressions between releases (see the
[paper Section 6 failure modes](docs/paper/Morphogenic_Intelligence.pdf)
for the documented regression/recovery history).

## What kinds of contributions are welcome

All of these are welcome, in rough order of enthusiasm:

1. **Bug reports with reproducers.** If a benchmark gives a different number than
   what's in `docs/benchmark_results/v{version}/`, open an issue with the run
   command, the git SHA, and the JSON output. This is the highest-value report.

2. **Additional benchmark runs on different hardware.** If you run the
   benchmarks on Linux / AMD / a Raspberry Pi / a neuromorphic board, we want
   to know the numbers. Open an issue or PR with your JSON results and we'll
   add them to `docs/benchmark_results/`.

3. **Re-implementations of the failure modes from the paper.** If you test
   premature Mature, the oscillation-is-the-feature finding, or the
   self-healing effect on a different spiking framework and confirm or
   falsify them, that's directly useful.

4. **Documentation fixes.** Typos, broken links, stale paths, unclear sections.

5. **Performance patches on the hot paths.** The medium-path learning loop
   and the anti-Hebbian LTD injection are the two biggest bottlenecks
   currently. Sparse eligibility updates and rayon parallelization on
   flat arrays are both on the roadmap — if you want to take a crack, open
   an issue first so we can agree on the interface.

## What kinds of contributions are less likely to land quickly

- **New benchmarks or tasks.** The planned next benchmarks (temporal sequence
  processing, NLP once prerequisites are met, etc.) are described in the paper
  Section 7 "Discussion" and Section "Path to Language". Adding unrelated
  benchmarks is unlikely to be merged unless they fit that trajectory.

- **Refactors that don't change numbers.** The code is not pretty. Cleaning
  it up is fine, but the authors have limited time for review of pure-cosmetic
  PRs. If you want to refactor, focus on something that also unblocks a
  roadmap item.

- **New core architectural features.** The six biological principles
  (Section 3 of the paper) are load-bearing. Adding a seventh principle
  requires a conversation first — please open an issue describing the
  biological motivation and the expected effect on the benchmarks before
  writing code.

- **Port to another language.** A Python or C++ port would be welcome as
  a separate repository. It's unlikely to land in this one because we want
  to keep the Rust engine as the reference implementation.

## How to report a bug

1. Include the version: `cargo run --example <name> --release` prints it.
2. Include the git SHA: `git rev-parse --short HEAD`.
3. Include the benchmark JSON output file from
   `docs/benchmark_results/v{version}/`.
4. Include your platform (OS, CPU, Rust version).
5. If possible, a minimal reproduction: the shortest command that surfaces
   the bug.

## How to open a PR

1. Open an issue first if the change is non-trivial, so we can agree on the
   approach before you spend time on it.
2. Work from a feature branch off `main`.
3. Run `cargo test` locally and include the result in the PR description.
4. Run the relevant benchmark (CartPole / mnist_v2 / nlp_readiness) and
   include the JSON output.
5. Reference the issue the PR addresses.
6. Expect a review turnaround of days to weeks — this is a small team.

## Style

- Rust code: follow `cargo fmt` and `cargo clippy -- -D warnings` (clean on
  main as of v3.0.0). Avoid new dependencies without discussion.
- Commit messages: short imperative subject line, longer body if needed.
  Conventional commits are welcome but not required.
- Documentation: markdown files are preferred for long-form docs; rustdoc
  comments for API.

## License

By contributing, you agree that your contribution will be licensed under the
Apache License 2.0, the same license as the rest of the project. See
[`LICENSE`](LICENSE) for the full text.

## Contact

- Open an issue on [GitHub](https://github.com/SimplyLiz/Morphon/issues)
- Email the authors: `lisa@tastehub.io`, `martyna@tastehub.io`

We may be slow to respond but we read everything.
