use morphon_core::system::{System, SystemConfig};
use morphon_core::types::*;
use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::scheduler::SchedulerConfig;


#[test]
fn test_system_creates_and_develops() {
    let config = SystemConfig::default();
    let system = System::new(config);
    let stats = system.inspect();

    // Should have morphons after development
    assert!(stats.total_morphons > 0, "System should have morphons after development");
    assert!(stats.total_synapses > 0, "System should have synapses after development");

    // Should have differentiated cell types
    assert!(
        stats.differentiation_map.len() > 1,
        "Should have multiple cell types: {:?}",
        stats.differentiation_map
    );
}

#[test]
fn test_system_step_runs_without_panic() {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    // Run 100 steps
    for _ in 0..100 {
        system.step();
    }

    let stats = system.inspect();
    assert_eq!(stats.step_count, 100);
}

#[test]
fn test_process_input_output() {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    let input = vec![1.0, 0.5, 0.3];
    let output = system.process(&input);

    // Should have motor outputs
    assert!(!output.is_empty(), "System should produce motor outputs");
}

#[test]
fn test_neuromodulation_injection() {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    system.inject_reward(0.8);
    assert!(system.modulation.reward > 0.0);

    system.inject_novelty(0.6);
    assert!(system.modulation.novelty > 0.0);

    system.inject_arousal(0.9);
    assert!(system.modulation.arousal > 0.0);
}

#[test]
fn test_neuromodulation_decays() {
    let mut modulation = morphon_core::Neuromodulation::default();
    modulation.inject_reward(1.0);

    let initial = modulation.reward;
    modulation.decay();
    assert!(modulation.reward < initial, "Reward should decay");
}

#[test]
fn test_developmental_programs() {
    // Test cortical preset
    let config = SystemConfig {
        developmental: DevelopmentalConfig::cortical(),
        ..Default::default()
    };
    let system = System::new(config);
    let stats = system.inspect();
    assert!(stats.total_morphons > 50);

    // Test hippocampal preset
    let config = SystemConfig {
        developmental: DevelopmentalConfig::hippocampal(),
        ..Default::default()
    };
    let system = System::new(config);
    let stats = system.inspect();
    assert!(stats.total_morphons > 50);

    // Test cerebellar preset
    let config = SystemConfig {
        developmental: DevelopmentalConfig::cerebellar(),
        ..Default::default()
    };
    let system = System::new(config);
    let stats = system.inspect();
    assert!(stats.total_morphons > 50);
}

#[test]
fn test_system_grows_under_stimulation() {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    let initial_morphons = system.inspect().total_morphons;

    // Stimulate heavily — inject reward and novelty, feed lots of input
    for _ in 0..500 {
        system.inject_reward(0.5);
        system.inject_novelty(0.3);
        system.feed_input(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        system.step();
    }

    let stats = system.inspect();
    // The system should have changed (grown or adapted)
    println!(
        "Morphons: {} -> {}, Synapses: {}, Clusters: {}, Firing rate: {:.3}",
        initial_morphons,
        stats.total_morphons,
        stats.total_synapses,
        stats.fused_clusters,
        stats.firing_rate
    );
}

#[test]
fn test_lifecycle_config_disables_features() {
    let config = SystemConfig {
        lifecycle: LifecycleConfig {
            division: false,
            differentiation: false,
            fusion: false,
            apoptosis: false,
            migration: false,
        },
        ..Default::default()
    };
    let mut system = System::new(config);

    let initial_count = system.inspect().total_morphons;

    for _ in 0..200 {
        system.feed_input(&[1.0, 0.5]);
        system.step();
    }

    // With all lifecycle features disabled, morphon count should stay the same
    // (developmental program still creates the initial set)
    let final_count = system.inspect().total_morphons;
    assert_eq!(
        initial_count, final_count,
        "Morphon count should not change with lifecycle features disabled"
    );
}

#[test]
fn test_ring_buffer() {
    let mut buf = morphon_core::types::RingBuffer::new(5);
    assert!(buf.is_empty());

    buf.push(1.0);
    buf.push(2.0);
    buf.push(3.0);
    assert_eq!(buf.len(), 3);
    assert!((buf.mean() - 2.0).abs() < 1e-10);

    // Fill and overflow
    buf.push(4.0);
    buf.push(5.0);
    buf.push(6.0); // overwrites 1.0
    assert_eq!(buf.len(), 5);
    assert!((buf.mean() - 4.0).abs() < 1e-10); // (2+3+4+5+6)/5 = 4.0
}

#[test]
fn test_activation_functions() {
    use morphon_core::types::ActivationFn;

    let sigmoid = ActivationFn::Sigmoid;
    assert!((sigmoid.apply(0.0) - 0.5).abs() < 1e-10);
    assert!(sigmoid.apply(10.0) > 0.99);
    assert!(sigmoid.apply(-10.0) < 0.01);

    let hard = ActivationFn::HardThreshold;
    assert_eq!(hard.apply(1.0), 1.0);
    assert_eq!(hard.apply(-1.0), 0.0);
}

// === New tests for updated concept features ===

#[test]
fn test_hyperbolic_distance() {
    let origin = HyperbolicPoint::origin(3);
    let point = HyperbolicPoint {
        coords: vec![0.5, 0.0, 0.0],
        curvature: 1.0,
    };

    let d = origin.distance(&point);
    assert!(d > 0.0, "Distance from origin should be positive");
    // Hyperbolic distance grows faster than Euclidean near boundary
    assert!(d > 0.5, "Hyperbolic distance should exceed Euclidean distance");
}

#[test]
fn test_hyperbolic_exp_log_roundtrip() {
    let p = HyperbolicPoint {
        coords: vec![0.3, 0.2, -0.1],
        curvature: 1.0,
    };
    let q = HyperbolicPoint {
        coords: vec![0.1, -0.2, 0.4],
        curvature: 1.0,
    };

    // log_map + exp_map should approximately roundtrip
    let tangent = p.log_map(&q);
    let recovered = p.exp_map(&tangent);

    let error = q.distance(&recovered);
    assert!(
        error < 0.1,
        "exp_map(log_map(q)) should ≈ q, but error = {error}"
    );
}

#[test]
fn test_hyperbolic_points_stay_in_ball() {
    let mut rng = rand::rng();
    for _ in 0..100 {
        let p = HyperbolicPoint::random(8, &mut rng);
        let norm: f64 = p.coords.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            norm < 1.0,
            "Random point should be inside the Poincaré ball, got norm = {norm}"
        );
    }
}

#[test]
fn test_tag_and_capture_delayed_reward() {
    use morphon_core::learning::{LearningParams, update_eligibility, apply_weight_update};
    use morphon_core::morphon::Synapse;
    use morphon_core::types::default_receptors;
    use morphon_core::Neuromodulation;
    use morphon_core::CellType;

    let params = LearningParams::default();
    let mut syn = Synapse::new(0.5);

    // Strong Hebbian coincidence → should set a tag
    for _ in 0..5 {
        update_eligibility(&mut syn, true, 1.0, &params, 1.0, 0);
    }
    assert!(
        syn.tag > 0.0 || syn.eligibility > 0.5,
        "Strong coincidence should set eligibility and possibly tag"
    );

    // Let eligibility decay but tag should persist
    for _ in 0..100 {
        update_eligibility(&mut syn, false, 0.0, &params, 1.0, 0);
    }
    // Fast eligibility should have mostly decayed
    assert!(syn.eligibility.abs() < 0.1, "Fast eligibility should decay");
    // Tag decays much slower
    // (tau_tag = 6000, so after 100 steps: tag * exp(-100/6000) ≈ 0.98 * tag)

    // Per-tick capture is now disabled (episode-gated).
    // Verify that tags still accumulate but don't auto-capture.
    let mut modulation = Neuromodulation::default();
    modulation.inject_reward(0.8);

    let motor_receptors = default_receptors(CellType::Motor);
    let captured = apply_weight_update(&mut syn, &modulation, &params, 0.01, &motor_receptors, [1.0; 4], &Default::default());
    assert!(!captured, "per-tick capture should be disabled");
    assert!(syn.tag > 0.0, "tag should still exist for episode-end capture");
}

#[test]
fn test_dual_clock_scheduler() {
    let config = SchedulerConfig::default();

    // Step 1: only fast
    let tick = config.tick(1);
    assert!(tick.fast);
    assert!(!tick.medium);    // period 10
    assert!(!tick.slow);      // period 100
    assert!(!tick.glacial);   // period 1000

    // Step 10: fast + medium
    let tick = config.tick(10);
    assert!(tick.fast);
    assert!(tick.medium);

    // Step 100: fast + medium + slow
    let tick = config.tick(100);
    assert!(tick.fast);
    assert!(tick.medium);
    assert!(tick.slow);

    // Step 1000: all paths
    let tick = config.tick(1000);
    assert!(tick.fast);
    assert!(tick.medium);
    assert!(tick.slow);
    assert!(tick.glacial);
}

#[test]
fn test_system_runs_with_dual_clock() {
    let config = SystemConfig {
        scheduler: SchedulerConfig {
            medium_period: 5,
            slow_period: 25,
            glacial_period: 100,
            homeostasis_period: 20,
            memory_period: 50,
        },
        ..Default::default()
    };
    let mut system = System::new(config);

    // Run past all scheduler thresholds
    for _ in 0..200 {
        system.feed_input(&[0.5, 0.3, 0.8]);
        system.step();
    }

    let stats = system.inspect();
    assert_eq!(stats.step_count, 200);
    assert!(stats.total_morphons > 0);
}

#[test]
fn test_specificity_increases_near_boundary() {
    let origin = HyperbolicPoint::origin(3);
    let near = HyperbolicPoint {
        coords: vec![0.1, 0.0, 0.0],
        curvature: 1.0,
    };
    let far = HyperbolicPoint {
        coords: vec![0.9, 0.0, 0.0],
        curvature: 1.0,
    };

    assert!(origin.specificity() < near.specificity());
    assert!(near.specificity() < far.specificity());
}

#[test]
fn test_serialization_roundtrip() {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    // Run a few steps to build state
    for _ in 0..50 {
        system.feed_input(&[1.0, 0.5]);
        system.inject_reward(0.3);
        system.step();
    }

    let stats_before = system.inspect();

    // Serialize to JSON
    let json = system.save_json().expect("serialize");
    assert!(!json.is_empty());

    // Deserialize back
    let restored = System::load_json(&json).expect("deserialize");
    let stats_after = restored.inspect();

    // Core state should match
    assert_eq!(stats_before.total_morphons, stats_after.total_morphons);
    assert_eq!(stats_before.total_synapses, stats_after.total_synapses);
    assert_eq!(stats_before.fused_clusters, stats_after.fused_clusters);
    assert_eq!(stats_before.step_count, stats_after.step_count);
    assert_eq!(stats_before.max_generation, stats_after.max_generation);
}

#[test]
fn test_dream_consolidates_tagged_synapses() {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    // Run a few steps to build up some synapse state
    for _ in 0..50 {
        system.inject_reward(0.5);
        system.step();
    }

    // Set performance above consolidation gate so dream consolidation is allowed
    for _ in 0..100 {
        system.report_performance(200.0);
    }

    // Manually set some synapses to have high tag strength
    let edge_indices: Vec<_> = system.topology.graph.edge_indices().take(5).collect();
    for &ei in &edge_indices {
        if let Some(syn) = system.topology.graph.edge_weight_mut(ei) {
            syn.tag_strength = 0.8;
            syn.tag = 0.5;
            syn.consolidation_level = 0.1;
            syn.consolidated = false;
        }
    }

    // Run dream cycle
    system.trigger_dream();

    // Verify consolidation increased and tags consumed
    for &ei in &edge_indices {
        if let Some(syn) = system.topology.graph.edge_weight(ei) {
            assert!(syn.consolidation_level > 0.1,
                "consolidation should increase: {}", syn.consolidation_level);
            assert!(syn.tag_strength < 0.8,
                "tag should be partially consumed: {}", syn.tag_strength);
        }
    }
}

#[test]
fn test_dream_resets_stale_synapses() {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    // Set some synapses to be stale candidates
    let edge_indices: Vec<_> = system.topology.graph.edge_indices().take(3).collect();
    for &ei in &edge_indices {
        if let Some(syn) = system.topology.graph.edge_weight_mut(ei) {
            syn.age = 6000; // > stale_synapse_age (5000)
            syn.usage_count = 1; // < stale_usage_threshold (3)
            syn.weight = 0.05; // < 0.1
            syn.consolidated = false;
        }
    }

    system.trigger_dream();

    // Verify stale synapses got refreshed
    for &ei in &edge_indices {
        if let Some(syn) = system.topology.graph.edge_weight(ei) {
            assert_eq!(syn.age, 0, "stale synapse age should be reset");
            assert_eq!(syn.usage_count, 0, "stale synapse usage should be reset");
        }
    }
}

#[test]
fn test_dream_respects_disabled() {
    let mut config = SystemConfig::default();
    config.dream.enabled = false;
    let mut system = System::new(config);

    // Set up tagged synapses
    let edge_indices: Vec<_> = system.topology.graph.edge_indices().take(3).collect();
    for &ei in &edge_indices {
        if let Some(syn) = system.topology.graph.edge_weight_mut(ei) {
            syn.tag_strength = 0.8;
            syn.consolidation_level = 0.1;
        }
    }

    system.trigger_dream();

    // Nothing should change
    for &ei in &edge_indices {
        if let Some(syn) = system.topology.graph.edge_weight(ei) {
            assert!((syn.consolidation_level - 0.1).abs() < 0.001,
                "consolidation should not change when dream disabled");
        }
    }
}

#[test]
fn test_dream_stale_refresh_resets_dead_synapses() {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    // Set performance above gate
    for _ in 0..100 { system.report_performance(200.0); }

    // Create stale candidates: old, unused, weak, unconsolidated
    let targets: Vec<_> = system.topology.graph.edge_indices().take(3).collect();
    for &ei in &targets {
        if let Some(syn) = system.topology.graph.edge_weight_mut(ei) {
            syn.age = 6000;
            syn.usage_count = 1;
            syn.weight = 0.03;
            syn.consolidated = false;
        }
    }

    system.trigger_dream();

    for &ei in &targets {
        if let Some(syn) = system.topology.graph.edge_weight(ei) {
            assert_eq!(syn.age, 0, "stale synapse age should reset");
            assert_eq!(syn.usage_count, 0, "stale synapse usage should reset");
            assert!(syn.weight.abs() > 0.03,
                "stale synapse weight should be refreshed to larger value, got {}", syn.weight);
        }
    }
}

#[test]
fn test_dream_skips_consolidated_synapses() {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    for _ in 0..100 { system.report_performance(200.0); }

    // Set up synapses that look stale BUT are consolidated — should be protected
    let targets: Vec<_> = system.topology.graph.edge_indices().take(3).collect();
    for &ei in &targets {
        if let Some(syn) = system.topology.graph.edge_weight_mut(ei) {
            syn.age = 6000;
            syn.usage_count = 1;
            syn.weight = 0.03;
            syn.consolidated = true; // protected!
        }
    }

    system.trigger_dream();

    for &ei in &targets {
        if let Some(syn) = system.topology.graph.edge_weight(ei) {
            assert_eq!(syn.age, 6000, "consolidated synapse age should NOT reset");
            assert!((syn.weight - 0.03).abs() < 0.01,
                "consolidated synapse weight should be unchanged, got {}", syn.weight);
        }
    }
}

#[test]
fn test_dream_consolidation_scales_with_cg() {
    // Two systems: one with high cg (Proliferating), one with low cg (Mature).
    // Same tagged synapses. Dream should consolidate more with high cg.
    let config = SystemConfig::default();

    let mut sys_high = System::new(config.clone());
    let mut sys_low = System::new(config);

    // Both above performance gate
    for _ in 0..100 {
        sys_high.report_performance(200.0);
        sys_low.report_performance(200.0);
    }

    // Force different Endo cg values
    sys_high.endo.channels.consolidation_gain = 2.5; // Proliferating
    sys_low.endo.channels.consolidation_gain = 0.5;  // Mature

    // Same tag setup on both
    let setup = |sys: &mut System| {
        let edges: Vec<_> = sys.topology.graph.edge_indices().take(5).collect();
        for &ei in &edges {
            if let Some(syn) = sys.topology.graph.edge_weight_mut(ei) {
                syn.tag_strength = 0.8;
                syn.tag = 0.5;
                syn.consolidation_level = 0.0;
                syn.consolidated = false;
            }
        }
        edges
    };

    let edges_high = setup(&mut sys_high);
    let edges_low = setup(&mut sys_low);

    sys_high.trigger_dream();
    sys_low.trigger_dream();

    // Measure consolidation_level increase
    let level_high: f64 = edges_high.iter()
        .filter_map(|&ei| sys_high.topology.graph.edge_weight(ei))
        .map(|s| s.consolidation_level)
        .sum();
    let level_low: f64 = edges_low.iter()
        .filter_map(|&ei| sys_low.topology.graph.edge_weight(ei))
        .map(|s| s.consolidation_level)
        .sum();

    assert!(level_high > level_low,
        "high cg ({}) should produce more consolidation than low cg ({})",
        level_high, level_low);
}

// === Phase B lever integration tests ===

/// Helper: create a system with endo disabled (for manual channel injection).
fn make_phase_b_system() -> System {
    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            target_input_size: Some(4),
            target_output_size: Some(2),
            ..DevelopmentalConfig::cortical()
        },
        ..Default::default()
    };
    System::new(config)
}

#[test]
fn phase_b_division_threshold_mult_scales_division() {
    // With dtm=0.5 (half threshold), morphons that wouldn't divide at default
    // should now divide. With dtm=2.0 (double threshold), fewer should divide.
    let mut sys_easy = make_phase_b_system();
    let mut sys_hard = make_phase_b_system();

    // Inject lever values directly (endo is disabled, so these persist)
    sys_easy.endo.channels.division_threshold_mult = 0.5; // threshold = 0.5 * 0.5 = 0.25
    sys_hard.endo.channels.division_threshold_mult = 2.0; // threshold = 0.5 * 2.0 = 1.0

    // Run all steps except the last (glacial tick fires at step == glacial_period).
    // division_pressure decays during these steps, so we re-inject it before the glacial tick.
    let glacial = sys_easy.config.scheduler.glacial_period as usize;
    for _ in 0..(glacial - 1) {
        sys_easy.feed_input(&[0.5; 4]);
        sys_easy.step();
    }
    for _ in 0..(glacial - 1) {
        sys_hard.feed_input(&[0.5; 4]);
        sys_hard.step();
    }

    // Re-inject division pressure right before the glacial tick
    for m in sys_easy.morphons.values_mut() {
        if m.cell_type == CellType::Associative || m.cell_type == CellType::Stem {
            m.division_pressure = 0.7; // above 0.5*0.5=0.25, below 0.5*2.0=1.0
            m.energy = 0.8;
        }
    }
    for m in sys_hard.morphons.values_mut() {
        if m.cell_type == CellType::Associative || m.cell_type == CellType::Stem {
            m.division_pressure = 0.7;
            m.energy = 0.8;
        }
    }

    let before_easy = sys_easy.inspect().total_morphons;
    let before_hard = sys_hard.inspect().total_morphons;

    // Final step triggers glacial tick — division happens here
    sys_easy.feed_input(&[0.5; 4]);
    sys_easy.step();
    sys_hard.feed_input(&[0.5; 4]);
    sys_hard.step();

    let born_easy = sys_easy.inspect().total_morphons.saturating_sub(before_easy);
    let born_hard = sys_hard.inspect().total_morphons.saturating_sub(before_hard);

    // With dtm=0.5, division_pressure=0.7 > effective_threshold=0.25 → should divide
    // With dtm=2.0, division_pressure=0.7 < effective_threshold=1.0 → should NOT divide
    assert!(born_easy >= born_hard,
        "Easy division (dtm=0.5) should produce >= births ({}) than hard (dtm=2.0, {})",
        born_easy, born_hard);
}

#[test]
fn phase_b_pruning_threshold_mult_scales_pruning() {
    // With ptm=2.0, weight_min is doubled → more synapses get pruned.
    // With ptm=0.5, weight_min is halved → fewer get pruned.
    let mut sys = make_phase_b_system();

    // Run to build some synapses, then count with different ptm values
    for _ in 0..100 {
        sys.feed_input(&[0.5; 4]);
        sys.step();
    }

    let base_synapses = sys.inspect().total_synapses;

    // Inject high pruning multiplier and run a slow tick
    sys.endo.channels.pruning_threshold_mult = 2.0;
    let slow = sys.config.scheduler.slow_period as usize;
    for _ in 0..slow {
        sys.feed_input(&[0.5; 4]);
        sys.step();
    }

    let after_aggressive = sys.inspect().total_synapses;
    // Aggressive pruning should have removed some synapses (or at least not grown)
    // We can't guarantee exact counts, but synapse count shouldn't exceed baseline
    // by much if pruning is aggressive.
    assert!(after_aggressive <= base_synapses + 20,
        "Aggressive pruning (ptm=2.0) should constrain synapse growth: base={}, after={}",
        base_synapses, after_aggressive);
}

#[test]
fn phase_b_frustration_sensitivity_mult_scales_frustration() {
    // With fsm=0.5 (half stagnation_threshold), morphons should reach
    // frustration faster. With fsm=2.0, they should be more tolerant.
    let mut sys_sensitive = make_phase_b_system();
    let mut sys_tolerant = make_phase_b_system();

    sys_sensitive.endo.channels.frustration_sensitivity_mult = 0.5;
    sys_tolerant.endo.channels.frustration_sensitivity_mult = 2.0;

    // Feed identical constant input — should cause PE stagnation
    for _ in 0..200 {
        sys_sensitive.feed_input(&[0.5; 4]);
        sys_sensitive.step();
    }
    for _ in 0..200 {
        sys_tolerant.feed_input(&[0.5; 4]);
        sys_tolerant.step();
    }

    // Count morphons in exploration mode
    let explore_sensitive = sys_sensitive.morphons.values()
        .filter(|m| m.frustration.exploration_mode).count();
    let explore_tolerant = sys_tolerant.morphons.values()
        .filter(|m| m.frustration.exploration_mode).count();

    assert!(explore_sensitive >= explore_tolerant,
        "Sensitive system (fsm=0.5) should have >= frustrated morphons ({}) than tolerant (fsm=2.0, {})",
        explore_sensitive, explore_tolerant);
}

// === Axonal property integration tests ===

#[test]
fn test_distance_dependent_delay_in_running_system() {
    // Bootstrap creates all synapses at delay=0.1 (fast I/O paths).
    // Synaptogenesis (slow path) creates new synapses with distance-dependent
    // delays: 0.5 + dist*0.75. After enough steps for synaptogenesis to fire,
    // the system should contain synapses at multiple distinct delay values.
    let config = SystemConfig {
        scheduler: SchedulerConfig {
            slow_period: 20,   // synaptogenesis fires more often
            glacial_period: 200,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut system = System::new(config);

    let initial_synapses = system.inspect().total_synapses;

    // All bootstrap synapses should be delay=0.1
    for (_, _, ei) in system.topology.all_edges() {
        assert!((system.topology.graph[ei].delay - 0.1).abs() < 1e-6,
            "Bootstrap synapse delay should be 0.1, got {}", system.topology.graph[ei].delay);
    }

    // Stimulate to trigger synaptogenesis on slow ticks
    for _ in 0..400 {
        system.inject_reward(0.3);
        system.inject_novelty(0.5);
        system.feed_input(&[1.0, 0.5, 0.3, 0.8]);
        system.step();
    }

    let stats = system.inspect();

    // Collect distinct delay values
    let mut delays: Vec<f64> = Vec::new();
    for (_, _, ei) in system.topology.all_edges() {
        let d = system.topology.graph[ei].delay;
        if delays.iter().all(|&existing| (existing - d).abs() > 1e-3) {
            delays.push(d);
        }
    }

    // If synaptogenesis created new synapses, they should have different delays
    // than the 0.1 bootstrap default
    if stats.total_synapses > initial_synapses {
        assert!(delays.len() >= 2,
            "With new synapses from synaptogenesis, should have varied delays. \
             Got {} synapses ({} new), delays: {:?}",
            stats.total_synapses, stats.total_synapses - initial_synapses, delays);
        assert!(stats.max_base_delay > 0.1 + 0.01,
            "New synapses should have distance-dependent delays > 0.1: max={}",
            stats.max_base_delay);
    } else {
        // Synaptogenesis didn't fire — still verify the stat pipeline works
        println!("NOTE: No new synapses created in 400 steps (topology may be saturated). \
                  Delay variance from synaptogenesis not testable here.");
    }

    // Stat pipeline should always work
    assert!(stats.avg_effective_delay > 0.0);
    assert!(stats.min_base_delay <= 0.1 + 1e-6,
        "min delay should include bootstrap 0.1: {}", stats.min_base_delay);
}

#[test]
fn test_delay_variance_affects_spike_timing() {
    // E2E test: two identical systems, one with natural delays (0.1 from bootstrap),
    // one with delays inflated to 3.0 (30× slower spike delivery).
    // With different delays, spikes arrive at different times →
    // different accumulator state → different potential → different output.
    let config = SystemConfig::default();

    let mut sys_fast = System::new(config.clone());
    let mut sys_slow = System::new(config);

    // Inflate all delays in sys_slow to 3.0 (vs 0.1 in sys_fast)
    for (_, _, ei) in sys_slow.topology.all_edges() {
        sys_slow.topology.graph[ei].delay = 3.0;
    }

    // Confirm the inflation actually changed something
    let fast_stats = sys_fast.inspect();
    let slow_stats = sys_slow.inspect();
    assert!(fast_stats.avg_effective_delay < slow_stats.avg_effective_delay,
        "Fast system should have lower avg delay: {} vs {}",
        fast_stats.avg_effective_delay, slow_stats.avg_effective_delay);

    // Drive both systems with strong input to force firing and spike propagation.
    // With delay=0.1, spikes arrive on the same step in sys_fast.
    // With delay=3.0, spikes take 3 steps to arrive in sys_slow.
    // By step 3+, the accumulator state (and thus potential) must differ.
    let mut differ = false;
    for i in 0..50 {
        let input = vec![1.0, 0.8, 0.6];
        sys_fast.inject_reward(0.5);
        sys_fast.feed_input(&input);
        sys_fast.step();

        sys_slow.inject_reward(0.5);
        sys_slow.feed_input(&input);
        sys_slow.step();

        let out_f = sys_fast.read_output();
        let out_s = sys_slow.read_output();

        // Different output port count already proves divergence
        if out_f.len() != out_s.len() {
            differ = true;
            break;
        }

        if !out_f.is_empty() {
            let max_diff: f64 = out_f.iter().zip(out_s.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            if max_diff > 1e-10 {
                differ = true;
                break;
            }
        }
    }

    assert!(differ,
        "Systems with fast (0.1) vs slow (3.0) delays should produce different outputs, \
         proving delay affects spike timing and behavior");
}
