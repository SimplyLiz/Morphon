# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build --release                # Build optimized
cargo test                           # All tests (23: 4 unit + 18 integration + 1 doctest)
cargo test <name>                    # Single test
cargo test -- --nocapture            # Show stdout
cargo bench                          # Criterion benchmarks (benches/benchmarks.rs)

# Examples with run profiles (quick is default, fast dev cycle)
cargo run --example cartpole --release              # quick
cargo run --example cartpole --release -- --standard
cargo run --example cartpole --release -- --extended
# Same for: anomaly, mnist (mnist requires ./data/ with MNIST files)

# Python bindings
maturin develop --features python

# WASM build + serve
wasm-pack build --target web --features wasm --no-default-features
cd web && python3 -m http.server 8080
```

## Architecture

**Morphon-Core** is a Morphogenic Intelligence engine: an adaptive system that grows, self-organizes, and learns at runtime without backpropagation. Biological inspiration throughout.

### Core Loop (`System::step()`)
Four temporal scales via dual-clock scheduler, all configured in `SchedulerConfig`:

| Scale | Default Period | Operations |
|-------|---------------|------------|
| **Fast** | 1 | Spike propagation (resonance), morphon firing, input integration |
| **Medium** | 10 | Eligibility traces, three-factor weight updates, tag-and-capture |
| **Slow** | 100 | Synaptogenesis, pruning, migration in hyperbolic space |
| **Glacial** | 1000 | Division, differentiation, fusion, apoptosis (with checkpoint/rollback) |

Plus homeostasis (synaptic scaling, inter-cluster inhibition) and memory recording at their own periods.

### Key Modules
- **system.rs** — Top-level orchestrator. `System::new()`, `step()`, `process()`, `inject_reward()`, `inspect()`
- **morphon.rs** — `Morphon` (compute unit) and `Synapse` structs. Each morphon has energy, threshold, activation, cell type, position in Poincaré ball
- **topology.rs** — petgraph-backed directed graph of morphon connections
- **learning.rs** — Three-factor learning rule: `Δw = eligibility × modulation`. Tag-and-capture for delayed reward (no backprop)
- **resonance.rs** — Spike propagation with delays. O(k·N) not O(N²)
- **morphogenesis.rs** — Structural plasticity: division, differentiation, fusion, migration, apoptosis, synaptogenesis, pruning
- **neuromodulation.rs** — Four broadcast channels: Reward (dopamine), Novelty (ACh), Arousal (noradrenaline), Homeostasis (serotonin)
- **developmental.rs** — Bootstrap programs (Cortical/Hippocampal/Cerebellar). Creates guaranteed I/O pathways. `target_input_size`/`target_output_size` for exact I/O matching
- **homeostasis.rs** — Stability: synaptic scaling, inter-cluster inhibition, migration damping, checkpoint/rollback
- **memory.rs** — Triple memory: working (persistent activity), episodic (one-shot), procedural (topology snapshots)
- **diagnostics.rs** — Learning pipeline observability: weight stats, eligibility, tags, firing rates, spike delivery
- **snapshot.rs** — Full system state serialization to JSON

### Bindings
- **python.rs** (feature `python`) — PyO3 bindings, built via maturin
- **wasm.rs** (feature `wasm`) — wasm-bindgen, powers the Three.js web visualizer in `web/`

### Key Design Decisions
- **Hyperbolic geometry**: Morphons live in Poincaré ball. Origin = general/stem, boundary = specialized. Curvature is learnable per-point.
- **No backpropagation**: Credit assignment via eligibility traces + neuromodulatory broadcast + tag-and-capture consolidation.
- **Contrastive reward** (`system.reward_contrastive()`): Targeted reward/inhibition at specific output ports. Breaks mode collapse in classification tasks.
- **Parallelization**: rayon on fast path (morphon updates, spike generation). Feature-gated behind `parallel`.

### Benchmark Results
Examples save JSON results to `docs/benchmark_results/v{version}/`. Each run writes a timestamped file plus `{bench}_latest.json`. Profile is recorded in the JSON.

## Conventions
- Lib crate type is `cdylib` + `rlib` (supports C FFI, Python, WASM, and Rust consumers)
- Edition 2024
- `petgraph` for all graph operations — don't introduce a second graph library
- `serde` derives on all public structs that cross API boundaries
- Benchmark examples use `env!("CARGO_PKG_VERSION")` for version tracking
