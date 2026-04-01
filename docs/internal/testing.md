# Testing

## Test Suite

Run with:

```bash
cargo test              # all tests
cargo test -- --nocapture  # see println! output
```

Current status: **113/113 passing** (94 unit + 18 integration + 1 doctest).

## Test Inventory

### Unit Tests — by module (94 total)

**morphogenesis (25 tests)**

| Test | What It Validates |
|------|-------------------|
| `synaptogenesis_creates_connections_between_active_nearby_morphons` | Active, proximate morphons get new connections |
| `synaptogenesis_skips_inactive_morphons` | Inactive morphons don't grow connections |
| `synaptogenesis_respects_cell_type_hierarchy` | Connections follow Sensory→Assoc→Motor flow |
| `pruning_removes_weak_old_unused_synapses` | Weak, old, unused synapses get pruned |
| `pruning_keeps_strong_synapses` | Strong synapses survive pruning |
| `division_creates_child_morphon` | Mitosis produces child with inherited state |
| `division_skips_low_energy_morphons` | Division requires energy > 0.3 |
| `division_respects_max_morphons` | Growth cap prevents unbounded division |
| `differentiation_converts_mature_active_stem_cells` | Mature stems specialize based on activity |
| `differentiation_skips_young_morphons` | Young morphons don't differentiate |
| `differentiation_skips_already_differentiated` | Terminally differentiated morphons stay put |
| `dedifferentiation_under_extreme_stress` | High stress reverts specialization |
| `dedifferentiation_does_not_affect_stem` | Stems can't de-differentiate further |
| `fusion_groups_correlated_active_morphons` | Correlated groups merge into clusters |
| `defusion_breaks_clusters_with_diverging_errors` | High PE variance splits clusters |
| `defusion_cleans_up_inhibitory_morphons` | Defusion removes inhibitory guards |
| `migration_moves_high_desire_morphons` | High desire drives migration |
| `migration_skips_low_desire_morphons` | Low desire morphons stay put |
| `migration_skips_fused_low_autonomy` | Fused morphons don't migrate |
| `apoptosis_removes_old_inactive_low_energy_morphons` | Death criteria work |
| `apoptosis_keeps_young_morphons` | Young morphons survive |
| `apoptosis_keeps_fused_morphons` | Fused morphons survive |
| `apoptosis_keeps_well_connected_morphons` | Well-connected morphons survive |
| `step_slow_returns_report` | Slow path returns MorphogenesisReport |
| `step_glacial_respects_lifecycle_config` | Lifecycle flags disable features |

**learning (17 tests)**

| Test | What It Validates |
|------|-------------------|
| `eligibility_decays_without_activity` | Eligibility trace decays exponentially |
| `eligibility_clamped_to_unit_range` | Eligibility stays in [-1, 1] |
| `pre_trace_increments_on_pre_spike` | Pre-synaptic trace updates on spike |
| `post_trace_increments_on_post_activity` | Post-synaptic trace updates on activity |
| `traces_decay_exponentially` | Both traces decay with tau_trace |
| `tag_set_when_eligibility_exceeds_threshold` | Eligibility > threshold sets tag |
| `tag_decays_slower_than_eligibility` | Tag persists longer than eligibility |
| `tag_and_capture_consolidates` | Strong reward captures tagged synapses |
| `consolidated_synapse_not_captured_again` | Double-consolidation prevented |
| `weight_update_receptor_gated` | Only matching receptors allow updates |
| `weight_clamped_to_max` | Weight stays within bounds |
| `age_increments_on_weight_update` | Synapse age tracks updates |
| `usage_count_increments_with_active_eligibility` | Active synapses track usage |
| `should_prune_old_weak_unused` | Prune criteria work |
| `should_not_prune_young_synapse` | Young synapses survive |
| `should_not_prune_strong_synapse` | Strong synapses survive |
| `should_not_prune_consolidated` | Consolidated synapses survive |

**memory (15 tests)**

| Test | What It Validates |
|------|-------------------|
| `working_memory_store_and_retrieve` | Store and read back patterns |
| `working_memory_refreshes_similar_pattern` | >50% overlap refreshes existing pattern |
| `working_memory_stores_dissimilar_patterns_separately` | Distinct patterns stored separately |
| `working_memory_evicts_weakest_at_capacity` | LRU eviction when full |
| `working_memory_decay_removes_weak_items` | Decay removes old patterns |
| `working_memory_activation_capped_at_one` | Activation max is 1.0 |
| `episodic_encode_and_len` | Encoding stores episodes |
| `episodic_replay_consolidates` | Replay increases consolidation |
| `episodic_replay_prioritizes_high_reward_high_novelty` | Priority queue correct |
| `episodic_evicts_least_consolidated_at_capacity` | Eviction targets least consolidated |
| `procedural_record_and_history` | Topology snapshots recorded |
| `procedural_avg_connectivity_computed` | Avg connectivity calculation |
| `procedural_evicts_oldest_at_capacity` | FIFO eviction for snapshots |
| `procedural_zero_morphons_no_panic` | Zero morphons handled gracefully |
| `triple_memory_creates_all_subsystems` | Constructor initializes all three |

**homeostasis (14 tests)**

| Test | What It Validates |
|------|-------------------|
| `synaptic_scaling_scales_weights_toward_setpoint` | Weight scaling corrects firing rate |
| `synaptic_scaling_no_change_at_setpoint` | No scaling when at target rate |
| `synaptic_scaling_increases_weights_when_underactive` | Underactive morphons get boosted |
| `synaptic_scaling_preserves_relative_ratios` | Proportional scaling preserves ratios |
| `can_migrate_when_cooldown_zero` | Migration allowed without cooldown |
| `cannot_migrate_during_cooldown` | Migration blocked during cooldown |
| `apply_migration_cooldown_sets_timer` | Cooldown timer set correctly |
| `high_homeostasis_reduces_migration` | Homeostasis brakes migration |
| `high_error_increases_migration` | High PE allows migration |
| `migration_rate_modifier_clamped` | Rate modifier stays in bounds |
| `create_checkpoint_captures_state` | Checkpoint captures PE and weights |
| `should_rollback_detects_pe_increase` | PE spike triggers rollback |
| `should_rollback_false_for_empty_morphons` | Empty set doesn't trigger |
| `rollback_restores_synapse_weights` | Rollback restores weights correctly |

**topology (12 tests)**

| Test | What It Validates |
|------|-------------------|
| `add_and_count_morphons` | Add morphons, count correct |
| `add_synapse_creates_connection` | Synapse creation works |
| `add_synapse_returns_none_for_missing_node` | Missing node returns None |
| `remove_morphon_removes_connections` | Removing morphon cleans up edges |
| `remove_nonexistent_morphon_returns_false` | Missing morphon returns false |
| `remove_synapse_works` | Synapse removal works |
| `incoming_outgoing_queries` | Directional queries return correct results |
| `synapse_between_returns_correct_edge` | Edge lookup works |
| `synapse_mut_modifies_weight` | Mutable edge access works |
| `degree_counts_both_directions` | Degree = in + out |
| `all_edges_returns_all` | Edge enumeration complete |
| `duplicate_connections_copies_subset` | Mitosis copies ~50% of connections |

**resonance (7 tests)**

| Test | What It Validates |
|------|-------------------|
| `propagate_generates_spikes_from_firing_morphons` | Firing morphons produce spikes |
| `non_firing_morphons_generate_no_spikes` | Non-firing morphons are silent |
| `deliver_respects_delay` | Spikes arrive after delay |
| `delivered_spike_adds_to_input_accumulator` | Delivered spikes affect target |
| `multiple_spikes_accumulate` | Multiple spikes sum |
| `spike_to_nonexistent_target_does_not_panic` | Missing target handled gracefully |
| `clear_removes_all_pending` | Clear empties the queue |

**lineage (4 tests)**

| Test | What It Validates |
|------|-------------------|
| `empty_set_produces_empty_tree` | Empty morphon set → empty lineage tree |
| `single_root` | One morphon with no parent → single root |
| `parent_child_relationship` | Lineage links and generation tracking |
| `json_roundtrip` | LineageTree serializes/deserializes via JSON |

### Integration Tests (tests/integration_test.rs — 18 tests)

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

### Doc Tests (1 test)

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
| `resonance_propagate/{50,200,500}` | Spike propagation scaling across system sizes |
| `eligibility_update_1000_synapses` | STDP eligibility update throughput |
| `weight_update_1000_synapses` | Three-factor weight update throughput |
| `pruning_500_edges` | Pruning evaluation throughput |
| `synaptogenesis_100_morphons` | New connection growth throughput |
| `system_step_scaling/{50,200,500}` | Full step time scaling across system sizes |
| `synaptic_scaling_100_morphons` | Homeostatic scaling throughput |

## Examples

```bash
cargo run --example cartpole --release
```

| Example | Description |
|---------|-------------|
| `cartpole` | CartPole control task: epsilon-greedy exploration, graded reward shaping. Demonstrates all engine features end-to-end. Supports --quick/--standard/--extended profiles. |
| `anomaly` | Sensor anomaly detection: learns normal patterns, detects anomalous input. Uses contrastive reward. |
| `mnist` | MNIST digit classification: full 784px input (no downsampling), 10-class output, auto-scaled seed. Requires MNIST data in `./data/`. |
| `classify_tiny` | Minimal classification sanity check. Quick validation of contrastive reward and learning pipeline. |
| `classify_3class` | 3-class classification with contrastive reward. Tests multi-class discrimination. |
| `learn_compare` | Side-by-side comparison of different learning configurations on the same task. |
| `cma_optimize` | CMA-ES hyperparameter search over learning parameters. Finds optimal learning rates, thresholds, etc. |
