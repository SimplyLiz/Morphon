# Testing

## Test Suite

Run with:

```bash
cargo test              # all tests
cargo test -- --nocapture  # see println! output
```

Current status: **23/23 passing** (4 lineage unit + 18 integration + 1 doctest), 0 warnings.

## Test Inventory

### Unit Tests (src/lineage.rs)

| Test | What It Validates |
|------|-------------------|
| `empty_set_produces_empty_tree` | Empty morphon set → empty lineage tree |
| `single_root` | One morphon with no parent → single root |
| `parent_child_relationship` | Lineage links and generation tracking |
| `json_roundtrip` | LineageTree serializes/deserializes via JSON |

### Integration Tests (tests/integration_test.rs)

| Test | What It Validates |
|------|-------------------|
| `test_system_creates_and_develops` | Developmental program produces Morphons, synapses, and multiple cell types |
| `test_system_step_runs_without_panic` | 100 simulation steps complete without errors |
| `test_process_input_output` | `process()` feeds input to sensory Morphons and reads motor output |
| `test_system_grows_under_stimulation` | Heavy stimulation (500 steps) changes the system |
| `test_lifecycle_config_disables_features` | All lifecycle flags = false → morphon count stable |
| `test_developmental_programs` | Cortical, hippocampal, and cerebellar presets each produce >50 Morphons |
| `test_neuromodulation_injection` | inject_reward/novelty/arousal set channel levels |
| `test_neuromodulation_decays` | Reward channel decays |
| `test_hyperbolic_distance` | Poincare ball distance is positive and exceeds Euclidean |
| `test_hyperbolic_exp_log_roundtrip` | `exp_map(log_map(q))` recovers q (error < 0.1) |
| `test_hyperbolic_points_stay_in_ball` | 100 random points all have norm < 1.0 |
| `test_specificity_increases_near_boundary` | Points further from origin have higher specificity |
| `test_tag_and_capture_delayed_reward` | Hebbian coincidence sets tag; fast eligibility decays; delayed reward captures tag |
| `test_dual_clock_scheduler` | Correct tick patterns at step 1/10/100/1000 |
| `test_system_runs_with_dual_clock` | 200 steps with custom scheduler periods |
| `test_ring_buffer` | Push, mean, overflow, length tracking |
| `test_activation_functions` | Sigmoid(0)=0.5, HardThreshold sign behavior |
| `test_serialization_roundtrip` | save_json → load_json preserves morphon/synapse/cluster/step counts |

### Doc Tests

| Test | What It Validates |
|------|-------------------|
| `lib.rs` quick start example | System creation, process(), inject_reward(), inspect() |

## Benchmarks

```bash
cargo bench
```

| Benchmark | What It Measures |
|-----------|-----------------|
| `system_step_100_morphons` | Time per `step()` with default config |
| `process_input` | Time per `process()` call |

## Examples

```bash
cargo run --example cartpole --release
```

| Example | Description |
|---------|-------------|
| `cartpole` | CartPole control task: 1000 episodes, epsilon-greedy exploration, graded reward shaping. Demonstrates all engine features end-to-end. |
| `mnist` | MNIST digit classification: 7x7 downsampled input, 10-class output, 5 epochs. Requires MNIST data in `./data/`. |
