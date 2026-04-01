# Build & Run

## Prerequisites

- Rust toolchain (1.85+ recommended, edition 2024)
- Python 3.9+ (for Python bindings only)
- No external system dependencies — pure Rust

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `petgraph` | 0.7 | Directed graph for topology (with serde-1 feature) |
| `rayon` | 1.10 | Parallel iteration for Morphon updates and spike propagation |
| `rand` | 0.9 | Random number generation |
| `crossbeam-channel` | 0.5 | Lock-free channels (reserved for future use) |
| `serde` + `serde_json` | 1 | Serialization for system snapshots |
| `pyo3` | 0.25 | Python bindings (optional, behind `python` feature) |
| `wasm-bindgen` | 0.2 | WASM bindings (optional, behind `wasm` feature) |
| `getrandom` | 0.3 | RNG for WASM (optional, behind `wasm` feature) |
| `criterion` | 0.5 (dev) | Benchmarking framework |
| `cmaes` | 0.2 (dev) | CMA-ES optimizer for hyperparameter search |
| `mnist` | 0.6 (dev) | MNIST dataset loader for classification example |

## Build

```bash
cargo build                  # debug build
cargo build --release        # optimized build
cargo build --features python # with Python bindings
```

## Test

```bash
cargo test                   # 113 tests (94 unit + 18 integration + 1 doctest)
cargo test -- --nocapture    # with output
```

## Bench

```bash
cargo bench
```

## Run Examples

All examples support run profiles: `--quick` (default), `--standard`, `--extended`.

```bash
cargo run --example cartpole --release              # quick (default, fast dev cycle)
cargo run --example cartpole --release -- --standard # longer run
cargo run --example cartpole --release -- --extended # full benchmark
cargo run --example anomaly --release
cargo run --example mnist --release                  # requires ./data/ with MNIST files
cargo run --example classify_tiny --release          # minimal sanity check
cargo run --example classify_3class --release        # 3-class classification
cargo run --example learn_compare --release          # compare learning configurations
cargo run --example cma_optimize --release           # CMA-ES hyperparameter search
```

## Python Bindings

Requires [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
maturin develop --features python
```

Then in Python:

```python
import morphon

system = morphon.System(seed_size=100, growth_program="cortical")
output = system.process([1.0, 0.5, 0.3])
system.inject_reward(0.8)
stats = system.inspect()
print(stats)  # SystemStats(morphons=..., synapses=..., clusters=..., steps=...)

# Save/load
json = system.save_json()
restored = morphon.System.load_json(json)
```

## Project Structure

```
Morphon/
├── Cargo.toml
├── pyproject.toml          # Maturin config for Python wheels
├── src/
│   ├── lib.rs              # crate root
│   ├── types.rs            # HyperbolicPoint, CellType, ActivationFn, RingBuffer
│   ├── morphon.rs          # Morphon + Synapse (with tag-and-capture fields)
│   ├── topology.rs         # petgraph wrapper
│   ├── neuromodulation.rs  # 4-channel broadcast
│   ├── learning.rs         # three-factor learning + tag-and-capture
│   ├── resonance.rs        # spike propagation (rayon-parallelized)
│   ├── morphogenesis.rs    # 7 structural change mechanisms
│   ├── memory.rs           # triple memory system
│   ├── homeostasis.rs      # stability mechanisms
│   ├── scheduler.rs        # dual-clock architecture
│   ├── developmental.rs    # bootstrapping with I/O pathway guarantees
│   ├── lineage.rs          # lineage tree export for visualization
│   ├── diagnostics.rs      # Learning pipeline observability
│   ├── snapshot.rs         # JSON serialization
│   ├── system.rs           # top-level orchestration
│   ├── python.rs           # PyO3 bindings (optional)
│   └── wasm.rs             # wasm-bindgen bindings (optional)
├── tests/
│   └── integration_test.rs
├── examples/
│   ├── cartpole.rs         # CartPole RL control task
│   ├── anomaly.rs          # Sensor anomaly detection
│   ├── mnist.rs            # MNIST digit classification (full 784px)
│   └── classify_tiny.rs    # Minimal classification sanity check
├── benches/
│   └── benchmarks.rs
└── docs/
    ├── MORPHON-product-concept.md
    ├── morphogenic-intelligence-concept.md
    └── internal/
        ├── architecture.md
        ├── api-reference.md
        ├── concept-mapping.md
        ├── testing.md
        └── build-and-run.md
```

## WASM Build

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/):

```bash
wasm-pack build --target web --features wasm --no-default-features
mv pkg web/pkg
```

Serve the demo:

```bash
cd web && python3 -m http.server 8080
# Open http://localhost:8080
```

The demo shows a live Poincare disk visualization of the MI topology with:
- Color-coded cell types (Sensory=blue, Associative=orange, Motor=red, Modulatory=green)
- Firing morphons flash white with a glow ring
- Edges colored by weight (positive=blue, negative=red)
- Controls: Run/Pause/Step, Reward/Novelty/Arousal injection buttons, speed slider
- Live stats: step count, morphon/synapse counts, firing rate, prediction error

The WASM binary is ~350KB (release optimized).

## Serialization

Save and restore full system state:

```rust
let mut system = System::new(SystemConfig::default());
// ... run steps ...

// Save
let json = system.save_json().unwrap();
std::fs::write("checkpoint.json", &json).unwrap();

// Load
let json = std::fs::read_to_string("checkpoint.json").unwrap();
let mut restored = System::load_json(&json).unwrap();
restored.step(); // continues from where it left off
```

## Lineage Export

Export the morphon family tree for visualization:

```rust
let tree = system.lineage_tree();
let json = tree.to_json();
std::fs::write("lineage.json", &json).unwrap();

println!("Roots: {:?}", tree.root_ids());
println!("Max depth: {}", tree.max_depth());
```
